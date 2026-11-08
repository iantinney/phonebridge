"""The phonebridge interface (docs/projects/phonebridge.md) over a swappable backend.

Coordinates everywhere are screenshot pixels, the same space `ocr()` returns, so an agent can
tap what it reads. Every call is timed and appended to <state>/calls.jsonl (mode 600).

Every gesture is drawn by the `humanize` layer (humanize.py) before a backend plays it: tap points
spread inside the target, press durations, curved swipe paths with a human velocity profile, typing
cadence with the odd corrected slip, and a think time before each input. Backends draw nothing themselves.

Input (tap, swipe, type, paste, home, launch, open_url) only goes to a configured device, inside its
device lock: the authhub lease `device:<alias>`. Either the caller passes its own lease token (verified;
its lifetime is the caller's), or phonebridge holds the lease for that caller from its first input until
`lease_idle` seconds after its last one, refusing other callers. A caller is an MCP session where the
transport has one; stateless MCP-over-HTTP clients must supply their own controller ID.

A task is a live device lease in use: from its first input until phonebridge's own hold ends or the caller's
lease is no longer live. While any task runs the Mac holds a power assertion (host/power.py). Guards on every
input (DEV-03): the task's app (the one it last launched) must be frontmost, checked <= 1 s before the input,
else the only allowed input is launch(app); after one flick onto an unfamiliar screen there are no more flicks until the
screen is recognised again or the app is relaunched; the operator guard judges the agent's last look (no
screenshot round-trip before a gesture); and a parked phone (Mac going to sleep, USB lost, too hot) takes no
input at all until host/watch.py has re-checked it.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import inspect
import io
import json
import math
import os
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Callable, Protocol
from urllib.parse import urlsplit

from . import ocr as vision
from .config import CONFIG_PATH, DATA_DIR, STATE_DIR, Config, read_config, udid_key
from .host.power import TaskPower
from .humanize import Humanizer, Key, PathPoint
from .lease import LeaseError, Leases, holder_id

RESERVATIONS = STATE_DIR / "controllers"

CONTROLLER = ContextVar("phonebridge_controller", default=None)
INPUT_CHECK = ContextVar("phonebridge_input_check", default=None)

URL_APPS = {"http": "com.apple.mobilesafari", "https": "com.apple.mobilesafari", "shortcuts": "com.apple.shortcuts"}
TYPE_MAX, PASTE_MAX, URL_MAX, SWIPE_MAX_S = 300, 64_000, 2048, 5.0
WARM_EVERY_S = 15.0  # a running task restarts the backend's idle timers (45 s) this often
STALE_INPUT_S = 1.0  # a tap/swipe held longer than this before it starts is refused (StaleInput): re-look first
POSITIONAL = ("tap", "double_tap", "swipe")  # inputs decided from a look's coordinates
STRIP_MAX_S = 900.0  # a strip stops by itself after 15 min
EVIDENCE_ROOT = Path(os.environ.get("PHONEBRIDGE_EVIDENCE", DATA_DIR / "evidence")).expanduser()
LEASE_REFRESH_S = 8.0  # a caller's lease is re-checked this often in the background while its task runs
LEASE_TRUST_S = 16.0  # an input trusts a check this recent (two refreshes); older, it verifies by ssh itself
AWAKE_S = 20.0  # iOS auto-locks after ≥ 30 s idle, so a phone seen unlocked (or touched) this recently still is
FG_LEAD, FG_MAX_AGE = 0.5, 1.0  # s: the frontmost-app query starts this long before an input, and is never older
LAUNCHES = ("launch", "open_url")  # the inputs allowed when the task's app is not frontmost
CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class Backend(Protocol):
    async def devices(self) -> list[dict]: ...  # [{udid, name, ios, product}]
    async def screen(self, udid: str) -> bytes: ...  # PNG
    async def tap(self, udid: str, x: float, y: float, press: float) -> None: ...  # finger down `press` s
    async def swipe(self, udid: str, path: list[PathPoint]) -> None: ...  # (x, y, t): down at [0], up after [-1]
    async def type(self, udid: str, keys: list[Key]) -> None: ...  # (char, hold s, pause s); "\b" Backspace, "\n" Return
    async def paste(self, udid: str, text: str) -> None: ...
    async def home(self, udid: str) -> None: ...
    async def launch(self, udid: str, bundle_id: str, url: str | None = None) -> None: ...
    async def lock_state(self, udid: str) -> dict: ...
    async def close(self) -> None: ...
    # optional: foreground(udid) -> bundle id of the frontmost app (com.apple.springboard on Home);
    # attached() -> UDIDs on USB now (host/watch.py)


class PhoneLocked(RuntimeError):
    pass


class OperatorActive(RuntimeError):
    pass


class NotFrontmost(RuntimeError):
    """The task's app is not frontmost: the only allowed input is launch(app)."""

    def __init__(self, device: str, app: str, foreground: str | None):
        self.app, self.foreground = app, foreground
        super().__init__(f"{app} is not frontmost on {device} ({foreground or 'nothing'} is); "
                         f"zero input sent; only launch({app}) is allowed")


class UnfamiliarScreen(RuntimeError):
    """One flick landed on an Unfamiliar screen (glossary): no more flicks until it is recognised or the app relaunched."""


class Parked(RuntimeError):
    """The phone is parked (the Mac is going to sleep, USB is gone, it is too hot): no input until re-checked."""

    def __init__(self, device: str, reasons: set[str]):
        self.reasons = reasons
        super().__init__(f"{device} is parked ({', '.join(sorted(reasons))}); zero input sent; "
                         f"input resumes after phonebridge re-checks the phone")


class StaleInput(RuntimeError):
    """A tap or swipe whose pre-input checks took longer than STALE_INPUT_S (beyond the think pause) or needed a
    backend recovery: the look it was decided on may no longer be the screen. Zero input sent; look again and
    re-decide (controller-verify S3: an input lands on a fresh look)."""


class TaskEnded(RuntimeError):
    """The task was ended under its owner (stale after a long sleep, the phone lost past T): back out."""

    def __init__(self, device: str, reason: str):
        self.reason = reason
        super().__init__(f"the task on {device} ended ({reason}); back out at the journal step and start a new task")


@dataclass
class Task:
    owner: str  # "lease:<token hash>" (the caller's own lease) or "caller:<caller>" (phonebridge holds it)
    identity: str  # device:<alias>
    alias: str
    token: str | None = None  # the caller's lease token, in memory only, to watch it end
    app: str | None = None  # bundle id of the app the task last launched: it must stay frontmost
    flicked: bool = False  # a flick since the screen was last recognised
    stopped: str | None = None  # why flicking stopped
    watcher: asyncio.Task | None = None
    warm: asyncio.Task | None = None  # keeps the touch session and the capture from idling out while the task runs


def _owner(caller: str, lease: str | None) -> str:
    return f"lease:{hashlib.sha256(lease.encode()).hexdigest()[:16]}" if lease else f"caller:{caller}"


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("coordinates must be finite numbers")


def swipe_shape(points: list, tail_s: float = 0.03) -> dict:
    """Where a swipe started and ended (screenshot px) and how fast the finger moved over its last `tail_s`
    (px/ms, the lift velocity a pan recognizer and UIScrollView act on), for the call log: a drag that did
    nothing can then be diagnosed from the log alone (surfaces, 2026-10-19)."""
    (x0, y0, _), (xe, ye, te) = points[0], points[-1]
    tail = [p for p in points if te - p[2] <= tail_s] or points[-2:]
    a = tail[0] if len(tail) > 1 else points[max(0, len(points) - 2)]
    dt = (te - a[2]) * 1000
    v = ((xe - a[0]) ** 2 + (ye - a[1]) ** 2) ** 0.5 / dt if dt > 0 else None
    return {"start": [round(x0), round(y0)], "end": [round(xe), round(ye)],
            "lift_v": None if v is None else round(v, 2), "lift_dir": [round(xe - a[0]), round(ye - a[1])]}


def _ms_since(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)


class Bridge:
    planned_glides = True  # swipe(..., release=) keeps a planned glide's length/duration/release (director closed loop)

    def __init__(self, backend: Backend, config: Config | None = None, log_path=None, config_path=CONFIG_PATH,
                 lease_idle: float = 60.0, power: TaskPower | None = None,
                 recognise: Callable[[str, bytes], str | None] | None = None, task_poll: float = 20.0,
                 controller_idle: float = 300.0):
        self.backend = backend
        self._config, self._path, self._config_text = config or Config(), None if config else config_path, None
        self._loaded_config = config is not None
        self.leases = Leases(self._config.lease_cmd)
        self.human = Humanizer(self._config.humanize)
        self._humans: dict[str, Humanizer] = {}
        self._inputs: dict[str, asyncio.Lock] = {}  # serialize a phone's checks, pacing and gesture; never other phones
        self._size: dict[str, tuple[int, int]] = {}  # udid -> screenshot pixels, for scroll()
        self.controller_idle = controller_idle  # explicit controllers: allow the 180 s media wait, bound abandoned visits
        self.lease_idle = lease_idle  # a device lock is released this long after its owner's last input
        self._owner: dict[str, str] = {}  # lease identity -> caller holding the device
        self._busy: dict[str, int] = {}  # lease identity -> input calls in flight
        self._idle: dict[str, asyncio.TimerHandle] = {}
        self._releasing: dict[str, asyncio.Future] = {}
        self._awake: dict[str, float] = {}  # udid -> monotonic time it was last seen unlocked or touched
        self._strips: dict = {}  # udid -> the running Strip (record_start)
        self._seen_app: dict[str, tuple[str, float]] = {}  # udid -> (bundle a look recognised, monotonic frame time)
        # operator-activity backoff: someone else changing the screen between our inputs means the operator
        self.op_gap, self.op_settle, self.op_backoff = 5.0, 3.0, 300.0  # seconds
        # the screen-change heuristic itself; the served daemon turns it off unless `[host] operator_watch = true`:
        # Ian unplugs the phone when he uses it (grill G3-G17), and USB loss parks (host/watch.py)
        self.op_watch = True
        self._baseline: dict[str, tuple[bytes, float]] = {}  # udid -> (screen after our last input settled, epoch)
        self._settle: dict[str, asyncio.Task] = {}
        self.log_path = log_path or STATE_DIR / "calls.jsonl"
        self.power = power or TaskPower(marker=None)  # the server passes one with the busy marker
        self.recognise = recognise  # (app, png) -> screen id | None; the screens lane plugs in here
        self.task_poll = task_poll  # s between checks that a caller's lease is still live
        self.tasks: dict[str, Task] = {}  # udid -> the running task
        self._ended: dict[str, tuple[str, str]] = {}  # udid -> (owner, reason) of a task ended under its owner
        self._parked: dict[str, set[str]] = {}  # udid (or "*": every phone) -> why it takes no input
        self._verdict: dict[str, asyncio.Task] = {}  # udid -> operator verdict on the last look, in the background
        self._looked: dict[str, tuple[float, str]] = {}  # udid -> (epoch of the last judged look, verdict)
        # when any phonebridge process here (MCP server, CLI, a test run) last sent input / saw the operator
        self._marks = self.log_path.parent / "phones"
        self._reservations: dict[str, dict] = {}
        self._lost_controllers: set[tuple[str, str]] = set()
        self.log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.log_path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.log_path, 0o600)

    @property
    def config(self) -> Config:
        """Reload pacing/policy; changes to device routing, leases or driver settings require a restart."""
        if self._path is not None:
            text = read_config(self._path)  # same-size edits can share a filesystem timestamp; compare actual bytes
            if text != self._config_text:
                new = Config.from_text(text, self._path) if text is not None else Config()
                fixed = ("devices", "lease_cmd", "lease_enforce", "input", "eyes", "tunnel", "pointer_surface")
                if self._loaded_config and any(getattr(new, k) != getattr(self._config, k) for k in fixed):
                    raise ValueError("device routing, lease or driver settings changed; restart the bridge before continuing")
                if new.lease_cmd != self.leases.cmd:  # don't carry leases across authhub endpoints
                    self.leases.release_all()
                    self._owner.clear()
                    self.leases.cmd = new.lease_cmd
                if new.humanize != self._config.humanize:
                    self.human = Humanizer(new.humanize)
                    self._humans.clear()
                self._config = new
                self._config_text, self._loaded_config = text, True
        return self._config

    def _human(self, udid: str) -> Humanizer:
        """Independent gesture/cadence state per physical phone, reproducible with a configured seed."""
        config = self.config.humanize
        if udid not in self._humans:
            seed = None if config.seed is None else int.from_bytes(
                hashlib.sha256(f"{config.seed}:{udid}".encode()).digest()[:8], "big")
            self._humans[udid] = Humanizer(config, seed=seed)
        return self._humans[udid]

    @asynccontextmanager
    async def _timed(self, op: str, device: str | None, **extra):
        t0 = time.perf_counter()
        rec = {"ts": round(time.time(), 3), "op": op, "device": device, **extra}
        try:
            yield rec
            rec["ok"] = True
        except BaseException as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
            raise
        finally:
            rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    async def reserve(self, device: str, owner: str) -> None:
        """Reserve a physical UDID, shared by all daemon/CLI processes using this state root."""
        dev = self.config.resolve(device)
        udid = dev.udid
        if held := self._reservations.get(udid):
            if held["owner"] != owner or held["closing"]:
                raise LeaseError(f"busy: controller {held['owner']} reserved {device}")
            return
        RESERVATIONS.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = RESERVATIONS / (hashlib.sha256(udid_key(udid).encode()).hexdigest() + ".lock")
        handle = path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise LeaseError(f"busy: another process reserved {device}") from None
        idle = asyncio.Event()
        idle.set()
        self._reservations[udid] = dict(owner=owner, handle=handle, active=0, idle=idle,
                                        closing=False, timer=None, device=dev)

    async def release_reservation(self, device: str, owner: str) -> None:
        udid = self.config.resolve(device).udid
        await self._release_reservation(udid, owner)

    async def _release_reservation(self, udid: str, owner: str) -> None:
        """Release the identity acquired earlier, even if the config changed while an input was in flight."""
        held = self._reservations.get(udid)
        if held is None:
            return
        if held["owner"] != owner:
            raise LeaseError("only the controller holder can release its reservation")
        held["closing"] = True
        await held["idle"].wait()  # park/release must never unlock an in-flight input
        if self._reservations.get(udid) is held:
            self._reservations.pop(udid)
            if held["timer"]:
                held["timer"].cancel()
            held["handle"].close()
            self._end_task(udid)

    @asynccontextmanager
    async def controller(self, device: str, owner: str, check=None):
        udid = self.config.resolve(device).udid
        await self.reserve(device, owner)
        token, guard = CONTROLLER.set(owner), INPUT_CHECK.set(check)
        try:
            yield
        finally:
            INPUT_CHECK.reset(guard)
            CONTROLLER.reset(token)
            await self._release_reservation(udid, owner)

    @asynccontextmanager
    async def _hold(self, device: str, caller: str, lease: str | None = None):
        owner = CONTROLLER.get() or caller
        dev = self.config.resolve(device)
        if (dev.udid, owner) in self._lost_controllers:
            raise LeaseError("controller reservation expired; begin a new controller session and re-observe")
        await self.reserve(device, owner)
        held = self._reservations[dev.udid]
        if held["timer"]:
            held["timer"].cancel()
        held["active"] += 1
        held["idle"].clear()
        try:
            if check := INPUT_CHECK.get():
                check()
            udid = dev.udid
            async with self._inputs.setdefault(udid, asyncio.Lock()):
                async with self._leased(device, owner, lease) as udid:
                    yield udid
        finally:
            held["active"] -= 1
            if not held["active"]:
                held["idle"].set()
            if not held["active"] and not CONTROLLER.get() and not held["closing"]:
                if held["timer"]:
                    held["timer"].cancel()
                # Renewal cannot keep an abandoned explicit controller forever. TaskAPI
                # contexts own their lifetime; raw inputs get a bounded idle interval.
                expiry = self.leases._expires.get((lease, dev.lease), 0)
                if (lease and expiry <= time.monotonic()) or (not lease and not self.leases.holds(dev.lease)):
                    await self._release_reservation(dev.udid, owner)
                else:
                    held["idle_until"] = time.monotonic() + self.controller_idle
                    delay = min(max(0, expiry - time.monotonic()), self.controller_idle) if lease else self.lease_idle
                    held["timer"] = asyncio.get_running_loop().call_later(delay,
                        lambda: asyncio.create_task(self._expire_controller(device, owner, held, lease)))

    async def _expire_controller(self, device, owner, held, lease):
        dev = held["device"]
        if self._reservations.get(dev.udid) is held:
            if lease:
                left = min(self.leases._expires.get((lease, dev.lease), 0),
                           held["idle_until"]) - time.monotonic()
                if left > 0:  # a heartbeat or task event reverified this same lease meanwhile
                    held["timer"] = asyncio.get_running_loop().call_later(left,
                        lambda: asyncio.create_task(self._expire_controller(device, owner, held, lease)))
                    return
                self._lost_controllers.add((dev.udid, owner))
            await self._release_reservation(dev.udid, owner)

    @asynccontextmanager
    async def _leased(self, device: str, caller: str, lease: str | None = None):
        """Yield the UDID while `caller` holds the device lock; see the module docstring. With `lease` (the
        caller's own authhub token for device:<alias>) it is only verified: the caller owns its lifetime."""
        d = self.config.resolve(device)
        if not d.lease:
            raise PermissionError(f"{device} is not in the phonebridge config; only configured phones take input")
        if lease:
            # verified by ssh once; the task's lease watcher re-checks it every LEASE_REFRESH_S off the input path and
            # vouches for it, so an input never waits on ssh (R1-IG attempt 2: inputs held ~0.3-1 s each for it)
            if not self.leases.vouched(lease, d.lease, LEASE_TRUST_S):
                await asyncio.to_thread(self.leases.verify, lease, d.lease, LEASE_TRUST_S)
            yield d.udid
            return
        if not self.leases.cmd and self.config.lease_enforce:
            raise LeaseError("no [lease] cmd configured (phonebridge setup writes phonebridge-local-lease); set it, or `enforce = false` to run without a device lock")
        ident = d.lease
        owner = self._owner.setdefault(ident, caller)  # claimed synchronously: no await between check and set
        if owner != caller:
            raise LeaseError(f"{ident} is in use by another phonebridge session (free {self.lease_idle:.0f} s after its last input)")
        self._busy[ident] = self._busy.get(ident, 0) + 1
        if timer := self._idle.pop(ident, None):
            timer.cancel()
        try:
            if pending := self._releasing.pop(ident, None):
                await pending  # an idle release in flight finishes before we re-acquire
            await asyncio.to_thread(self.leases.ensure, ident, holder_id(self.leases.holder, caller))
            yield d.udid
        finally:
            self._busy[ident] -= 1
            if not self._busy[ident]:
                if self.leases.holds(ident):  # idle timer starts when the last input *ends*
                    self._idle[ident] = asyncio.get_running_loop().call_later(self.lease_idle, self._release_idle, ident)
                else:  # refused or failed before holding it: drop the claim
                    self._owner.pop(ident, None)

    def _release_idle(self, ident: str) -> None:
        self._idle.pop(ident, None)
        if self._busy.get(ident):
            return  # an input started meanwhile; it reschedules when it ends
        self._owner.pop(ident, None)
        for udid, t in list(self.tasks.items()):
            if t.identity == ident and not t.token:
                self._end_task(udid)  # phonebridge's own hold is over: so is the task
        self._releasing[ident] = asyncio.get_running_loop().run_in_executor(None, self.leases.release, ident)

    # --- eyes -------------------------------------------------------------------------------
    async def devices(self) -> list[dict]:
        async with self._timed("devices", None):
            found = await self.backend.devices()
        out = []
        for d in found:
            alias = self.config.alias_of(d["udid"])
            supervision = self.config.devices[alias].supervision if alias else "read-only"  # unconfigured: no input
            out.append({"id": alias or d["udid"], **d, "supervision": supervision})
        return out

    async def screen(self, device: str) -> bytes:
        async with self._timed("screen", device) as rec:
            udid = self.config.resolve(device).udid
            png = await self.backend.screen(udid)
            rec["bytes"] = len(png)
        self._look(udid, png)
        self._saw(udid, png)
        self._size[udid] = vision.png_size(png)
        if (t := self.tasks.get(udid)) and t.app and (t.flicked or t.stopped) and self.recognise:
            if await asyncio.to_thread(self.recognise, t.app, png):
                t.flicked, t.stopped = False, None
            elif t.flicked:
                t.stopped = "the screen after a flick is unfamiliar"
        return png

    def _saw(self, udid: str, png: bytes) -> None:
        """Remember a screenshot as the baseline if our own last input has settled (≥ op_settle s ago)."""
        if self._since(udid, "input") >= self.op_settle:
            self._baseline[udid] = (png, time.time())

    def _mark(self, udid: str, kind: str) -> None:
        """Note "now" for `kind` (input, operator) on this phone, for every phonebridge process on this Mac:
        one process's input is not another's operator, and an operator seen by one stops them all."""
        self._marks.mkdir(mode=0o700, exist_ok=True)
        (self._marks / f"{udid}.{kind}").touch(mode=0o600)

    def _since(self, udid: str, kind: str) -> float:
        with contextlib.suppress(FileNotFoundError):
            return time.time() - (self._marks / f"{udid}.{kind}").stat().st_mtime
        return math.inf

    # --- the fast look: the stream's frames judged where they are (screens/fast.py) ----------------------------
    def eyes_from(self, store=None, **kw) -> None:
        """Build the fast look over the backend's stream: the known-screens store (the repo's records by default)
        and Vision. The daemon calls this once; tests pass a store and a scripted OCR."""
        from .screens.fast import Eyes
        from .screens.store import RECORD_DIRS, Store

        kw.setdefault("templates_dir", RECORD_DIRS[0] / "templates")  # the records' template anchors live beside them
        kw.setdefault("still", self.backend.screen)  # no live frame: a DVT still keeps the caller seeing (source: dvt)
        self._eyes = Eyes(store or Store(records_dirs=RECORD_DIRS), self._frames, **kw)

    async def _frames(self, udid: str):
        if not hasattr(self.backend, "frames"):
            raise RuntimeError(f"{type(self.backend).__name__} has no screen stream: use screen() + ocr()")
        return await self.backend.frames(udid)

    JUDGED_KEEP = 8  # judged frames kept per phone for frame(seq=…): the last looks' pixels, not every frame

    def _keep_judged(self, udid: str, seq: int, pixels) -> None:
        kept = self.__dict__.setdefault("_judged", {}).setdefault(udid, {})
        kept[seq] = pixels
        while len(kept) > self.JUDGED_KEEP:
            kept.pop(next(iter(kept)))

    @staticmethod
    def _encode(pixels, fmt: str = "jpeg", quality: int = 80) -> bytes:
        """A look's pixels (a live image, or PNG bytes for a still) as JPEG or PNG."""
        from PIL import Image as PILImage
        im = pixels.source if getattr(pixels, "source", None) is not None else PILImage.open(io.BytesIO(pixels.png))
        im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, "JPEG", quality=quality) if fmt == "jpeg" else im.save(out, "PNG", compress_level=1)
        return out.getvalue()

    async def look(self, device: str, platform: str | None = None, expect: list[str] | None = None,
                   settle: bool | str = True, with_frame: bool = False) -> dict:
        """Settle on the live stream (settle: True/"content" | "chrome" | False, see screens.fast.settle_mode), recognise the screen, time every layer. No image leaves the Mac unless
        `with_frame`: then the very frame it judged rides along as base64 JPEG (`frame_jpeg`). Judged stream frames
        stay retrievable by `frame(seq=…)` for the last JUDGED_KEEP looks."""
        async with self._timed("look", device, platform=platform) as rec:  # the whole tool call: `ms` is its wall time
            udid = self.config.resolve(device).udid
            # only the NEWEST look may vouch for the frontmost app: cleared now, set again below only when this look
            # recognises an app screen (unknown, an alert, a popup, a stop, a still or a failed look leave it clear)
            self._seen_app.pop(udid, None)
            if getattr(self, "_eyes", None) is None:
                self.eyes_from()
            look = await self._eyes.look(udid, platform=platform, expect=expect, settle=settle)
            out = look.summary()
            if look.pixels is not None:
                self._size[udid] = (look.pixels.width, look.pixels.height)
                if self.op_watch:  # the operator guard judges the look as PNG; off on the served daemon, so no encode
                    self._look(udid, look.pixels.png)  # (29 ms median for a 1170x2532 frame, measured on the M5 Pro)
                if look.frame is not None:
                    self._keep_judged(udid, look.frame.seq, look.pixels)
                    r = self._eyes.store.records.get(look.screen) if look.screen else None
                    if r is not None and r.bundle and r.bundle != "*" and (r.kind or "screen") == "screen":
                        self._seen_app[udid] = (r.bundle, look.frame.t)  # the app on screen, at the frame's arrival
            out = {**out, "frame_wall": look.frame.wall if look.frame else look.pixels.ts, "items": look.items}
            if with_frame and look.pixels is not None:
                import base64
                out["frame_jpeg"] = base64.b64encode(await asyncio.to_thread(self._encode, look.pixels)).decode()
            rec.update({k: v for k, v in out.items() if k not in ("ms", "items", "frame_jpeg")},
                       **{f"{k}_ms": v for k, v in look.ms.items()})
        return out

    async def frame(self, device: str, format: str = "jpeg", quality: int = 80, seq: int | None = None) -> tuple[bytes, dict]:
        """The latest stream frame (opens the stream) as JPEG or PNG, with its seq, wall time and age; a DVT still
        (`source: dvt`, no seq) when this backend has no stream or it yields nothing in 3 s, so callers that prefer
        `frame` keep working (a stills-only session is a reliability event for the controller to journal).
        `seq`: the frame a recent look judged (`source: look`); LookupError once it is no longer kept."""
        from PIL import Image as PILImage

        if seq is not None:
            udid = self.config.resolve(device).udid
            kept = self.__dict__.get("_judged", {}).get(udid, {})
            if seq not in kept:
                raise LookupError(f"frame #{seq} is no longer kept (kept: {sorted(kept)}); look(with_frame=True) returns it inline")
            pixels = kept[seq]
            data = await asyncio.to_thread(self._encode, pixels, format, quality)
            return data, {"source": "look", "seq": seq, "wall": pixels.ts, "width": pixels.width, "height": pixels.height}

        async with self._timed("frame", device, format=format) as rec:
            udid = self.config.resolve(device).udid
            f = None
            if hasattr(self.backend, "frames"):
                with contextlib.suppress(asyncio.TimeoutError):
                    f = await (await self.backend.frames(udid)).wait(timeout=3.0)
            if f is not None:
                image, meta = f.image, {"source": "stream", "seq": f.seq, "wall": f.wall, "age_ms": round(f.age_ms(), 1),
                                        "width": f.width, "height": f.height}
            else:
                t = time.time()
                png = await self.backend.screen(udid)
                image = PILImage.open(io.BytesIO(png)).convert("RGB")
                meta = {"source": "dvt", "seq": None, "wall": t, "age_ms": round((time.time() - t) * 1000, 1),
                        "width": image.width, "height": image.height}

            def encode():
                out = io.BytesIO()
                (image.save(out, "JPEG", quality=quality) if format == "jpeg" else image.save(out, "PNG", compress_level=1))
                return out.getvalue()
            data = await asyncio.to_thread(encode)
            rec.update(meta, bytes=len(data))
        self._size[udid] = (image.width, image.height)
        return data, meta

    # --- understanding: the latest frame to the local model service (understand.py) -------------------------
    async def understand(self, device: str, prompt: str | None = None, max_tokens: int | None = None) -> dict:
        """One sentence (who, what, hook, topic) + interest 0–3 for what is on screen now, from the warm local model
        on this Mac; the frame's seq/wall/age come along so the caller knows what was judged."""
        if getattr(self, "understanding", None) is None:
            from .understand import Client

            self.understanding = Client()
        async with self._timed("understand", device) as rec:
            jpeg, meta = await self.frame(device, format="jpeg", quality=85)
            out = await self.understanding.understand(jpeg, prompt=prompt, max_tokens=max_tokens)
            rec.update(model=out.get("model"), model_ms=out.get("ms"), interest=out.get("interest"), parsed=out.get("parsed"),
                       frame_age_ms=meta.get("age_ms"))
        return {**out, "frame": meta}

    async def ocr(self, device: str, fast: bool = False, with_image: bool = False) -> list[dict] | dict:
        """A screenshot of its own + Apple Vision OCR. A separate `screen` call is a different screenshot, so its
        pixels need not match these lines (seen during a coast: the same line 0.27 h here, 0.35 h there).
        `with_image`: {items, image_jpeg (base64, the very image read), width, height}."""
        png = await self.screen(device)
        async with self._timed("ocr", device, fast=fast) as rec:
            items = await asyncio.to_thread(vision.recognize, png, fast)  # keep the loop free for gestures
            rec["n"] = len(items)
        if not with_image:
            return items
        import base64
        from PIL import Image as PILImage
        im = PILImage.open(io.BytesIO(png))
        out = io.BytesIO()
        await asyncio.to_thread(lambda: im.convert("RGB").save(out, "JPEG", quality=80))
        return {"items": items, "image_jpeg": base64.b64encode(out.getvalue()).decode(), "width": im.width, "height": im.height}

    async def find_text(self, device: str, substring: str) -> dict | None:
        return vision.find(await self.ocr(device), substring)

    async def lock_state(self, device: str) -> dict:
        async with self._timed("lock_state", device):
            return await self.backend.lock_state(self.config.resolve(device).udid)

    # --- presence: is the phone really there (presence.py) ---------------------------------------------------------
    def _daemon_info(self) -> dict:
        if not hasattr(self, "_daemon"):
            import subprocess
            from pathlib import Path
            sha = None
            try:
                sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True, timeout=3).stdout.strip() or None
            except Exception:
                pass
            from importlib.metadata import PackageNotFoundError, version
            try:
                ver = version("phonebridge")  # an install without git (a wheel, a copied tree) still says what it is
            except PackageNotFoundError:
                ver = None
            self._daemon = {"sha": sha, "version": ver, "pid": os.getpid(), "started": round(time.time(), 3)}
        return self._daemon

    async def _presence_extra(self, udid: str, probe: dict) -> dict:
        """What only the daemon knows, inside the cached result: the frontmost app (one DVT query, ≤ 1 s, only when
        the phone answered) and when the phone was last seen awake by this daemon (iOS 27 reports no lock state)."""
        fg = None
        query = getattr(self.backend, "foreground", None)
        if probe.get("lockdown") and query:
            try:
                fg = await asyncio.wait_for(query(udid), 1.0)
            except Exception:
                fg = None
        seen = self._awake.get(udid)
        return {"foreground": fg, "locked": None,
                "awake_seen_s_ago": None if seen is None else round(time.monotonic() - seen, 1)}

    async def presence(self, device: str, fresh: bool = False) -> dict:
        """{attached, transport usb|wifi|none, tunnel, lockdown, name, ios, foreground, locked (None: iOS 27 doesn't
        say), awake_seen_s_ago, errors, ms, ts, cached, age_s, device, daemon {sha, pid, started}}; ≤ ~2 s, cached 10 s."""
        if not hasattr(self, "_presence"):
            from .presence import Presence
            self._presence = Presence(extra=self._presence_extra)
        async with self._timed("presence", device, fresh=fresh) as rec:
            d = self.config.resolve(device)
            p = await self._presence.get(d.udid, fresh=fresh)
            rec.update(attached=p["attached"], transport=p["transport"], tunnel=p["tunnel"], lockdown=p["lockdown"],
                       cached=p["cached"])
        return {**p, "device": d.alias, "daemon": self._daemon_info()}

    async def foreground(self, device: str) -> str | None:
        """Bundle id of the frontmost app (com.apple.springboard on Home); None if this backend can't tell."""
        async with self._timed("foreground", device) as rec:
            query = getattr(self.backend, "foreground", None)
            try:
                rec["foreground"] = await query(self.config.resolve(device).udid) if query else None
            except NotImplementedError:
                rec["foreground"] = None
            return rec["foreground"]

    # --- operator activity ------------------------------------------------------------------
    async def _operator_check(self, device: str, udid: str, own_lease: str | None, rec: dict) -> bytes | None:
        """Refuse input while the operator appears to be using the phone: the screen changed since our own
        last input settled, without us, and it isn't just animating (video) or locked (auto-lock). Then drop
        our lease and back off for op_backoff seconds. The evidence is the agent's last look when there is one
        since our last input (judged in the background as it came in: no screenshot round-trip before the
        gesture, DEV-03), else a screenshot taken now (a session's first input, blind input). Returns that
        screenshot, if any. The verdict and the change score go to the call log."""
        if not self.op_watch:
            return None
        if (verdict := self._verdict.get(udid)) and not verdict.done():
            with contextlib.suppress(asyncio.TimeoutError):  # a look just came in: its verdict is nearly ready
                await asyncio.wait_for(asyncio.shield(verdict), 3.0)
        if (left := self.op_backoff - self._since(udid, "operator")) > 0:
            rec["operator_check"] = "backing-off"
            raise OperatorActive(f"operator appears active on {device}; phonebridge backs off for {left:.0f} more s")
        if (since_input := self._since(udid, "input")) <= self.op_gap:
            rec["operator_check"] = "just-acted"
            return None
        input_at = time.time() - since_input  # before the screenshot: it takes 0.5-2 s on the phone
        if (look := self._looked.get(udid)) and look[0] > input_at:
            rec["operator_check"] = f"look-{look[1]}"
            return None
        png = await self.backend.screen(udid)
        verdict, png = await self._judge(udid, png, self._baseline.get(udid), input_at, rec, own_lease)
        if verdict == "operator":
            raise OperatorActive(f"operator appears active on {device} (the screen changed without phonebridge); "
                                 f"backing off {self.op_backoff / 60:.0f} min")
        return png

    async def _judge(self, udid: str, png: bytes, base: tuple[bytes, float] | None, input_at: float, rec: dict,
                     own_lease: str | None) -> tuple[str, bytes]:
        """Was `png` changed by someone else since `base` (the settled screen after our last input)? A change
        is confirmed on a second frame 0.4 s later: still = the operator (or iOS locking it), moving = video."""
        rec["operator_check"] = "stale-baseline" if base else "no-baseline"  # stale: no settled shot since our input
        if base and base[1] - input_at >= self.op_settle:
            rec["change"] = round(await asyncio.to_thread(vision.change, base[0], png), 3)
            rec["operator_check"] = "same"
        if rec.get("change", 0) > vision.CHANGED:
            await asyncio.sleep(0.4)
            again = await self.backend.screen(udid)
            rec["operator_check"] = "moving"
            if await asyncio.to_thread(vision.change, png, again) <= vision.CHANGED:  # a still screen we didn't make...
                self._baseline[udid] = (again, time.time())  # judge the next attempt against this screen
                if (await self._lock_reason(udid, again))[0]:  # ...unless iOS locked it: the lock guard's case
                    rec["operator_check"] = "locked"
                    self._awake.pop(udid, None)
                    return "locked", again
                rec["operator_check"] = "operator"
                self._mark(udid, "operator")
                for name, shot in (("before", base[0] if base else png), ("after", again)):  # Ian, or a false alarm?
                    (path := self._marks / f"{udid}.operator-{name}.png").touch(mode=0o600)
                    path.write_bytes(shot)
                if own_lease and self.leases.holds(own_lease) and not self._busy.get(own_lease, 0) > 1:
                    self._owner.pop(own_lease, None)
                    if timer := self._idle.pop(own_lease, None):
                        timer.cancel()
                    await asyncio.to_thread(self.leases.release, own_lease)
                    for u, t in list(self.tasks.items()):
                        if t.identity == own_lease and not t.token:
                            self._end_task(u)
                return "operator", again
            png = again  # still moving (video): inconclusive, carry on
        self._baseline[udid] = (png, time.time())
        return rec["operator_check"], png

    def _look(self, udid: str, png: bytes) -> None:
        """An agent's look: judge it against the settled baseline in the background, off the gesture path."""
        if not self.op_watch:
            return
        since_input = self._since(udid, "input")
        base, input_at = self._baseline.get(udid), time.time() - since_input
        if (since_input <= self.op_gap or not base or base[1] - input_at < self.op_settle
                or ((v := self._verdict.get(udid)) and not v.done())):
            return  # just acted, nothing settled to judge against, or still judging the previous look
        alias = self.config.alias_of(udid)
        own = self.config.devices[alias].lease if alias else None
        at = time.time()
        self._looked[udid] = (at, "pending")

        async def judge():
            rec = {"ts": round(at, 3), "op": "look", "device": alias or udid}
            try:
                verdict, _ = await self._judge(udid, png, base, input_at, rec, own)
                self._looked[udid] = (at, verdict)
            except Exception as e:  # inconclusive: the next input falls back to its own screenshot
                self._looked.pop(udid, None)
                rec["error"] = f"{type(e).__name__}: {e}"
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        self._verdict[udid] = asyncio.get_running_loop().create_task(judge())

    def _acted(self, udid: str) -> None:
        """After our input: note it, and take the settled screen as the next baseline in the background."""
        self._awake[udid] = time.monotonic()
        self._mark(udid, "input")
        if task := self._settle.pop(udid, None):
            task.cancel()

        async def settle():
            await asyncio.sleep(self.op_settle)
            with contextlib.suppress(Exception):
                self._saw(udid, await self.backend.screen(udid))
        self._settle[udid] = asyncio.get_running_loop().create_task(settle())

    async def _think(self, op: str, udid: str, rec: dict, task: Task | None = None, device: str = "", pause: bool = True) -> None:
        """Pause like a person deciding on the next action: the drawn think time, less what has already
        passed since our last input on this phone (an agent's own latency counts). With a task app to keep
        frontmost, its query runs in the last FG_LEAD s of the pause, so it is <= FG_MAX_AGE s old at the input."""
        want = self._human(udid).think(op) if pause else 0.0  # pause=False: the caller paces its own inputs (the director)
        if not pause:
            rec["think"] = "caller"
        wait = want - self._since(udid, "input")
        rec["think_ms"] = round(max(0.0, wait) * 1000)
        deadline = time.monotonic() + max(0.0, wait)
        if not (task and task.app and op not in LAUNCHES):
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            return
        await asyncio.sleep(max(0.0, deadline - FG_LEAD - time.monotonic()))
        t = time.perf_counter()
        fg, at = await self._frontmost(device, udid, task)
        rec["fg_ms"] = _ms_since(t)  # the query itself, not the pause around it
        await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        if time.monotonic() - at > FG_MAX_AGE:  # the query itself was slow
            t = time.perf_counter()
            fg, at = await self._frontmost(device, udid, task)
            rec["fg_ms"] = round(rec["fg_ms"] + _ms_since(t), 1)
        rec["foreground"], rec["fg_age_ms"] = fg, round((time.monotonic() - at) * 1000)
        if fg is None:
            return  # this backend can't tell (the rule needs pmd3's foreground())
        if fg != task.app:
            if task.flicked:
                task.stopped = f"the last flick left {task.app} for {fg}"
            raise NotFrontmost(device, task.app, fg)
        if op == "swipe" and task.stopped:
            raise UnfamiliarScreen(f"no more flicks on {device}: {task.stopped} (an unfamiliar screen after one flick); "
                                     f"zero input sent; look until the screen is recognised, or launch({task.app})")
        if not self.recognise:
            task.flicked = False  # with no screen recogniser, the frontmost app is the recognition

    async def _frontmost(self, device: str, udid: str, task: Task) -> tuple[str | None, float]:
        # a look that recognised one of the task app's screens on a frame under FG_MAX_AGE old is the same evidence
        # as the process list, at no cost (R1 attempt 4: the DVT query took ~0.5 s before every swipe)
        if (seen := self._seen_app.get(udid)) and seen[0] == task.app and time.monotonic() - seen[1] <= FG_MAX_AGE:
            return seen[0], seen[1]
        query = getattr(self.backend, "foreground", None)
        if query is None:
            return None, time.monotonic()
        try:
            return await query(udid), time.monotonic()
        except NotImplementedError:  # this phone can't say (an iOS without the flag): logged as foreground null
            return None, time.monotonic()
        except Exception as e:  # couldn't confirm the app is frontmost: no input, but nothing said another app is
            raise StaleInput(f"stale: re-look. On {device} the frontmost-app check couldn't answer "
                             f"({type(e).__name__}: {e}); zero input sent; look again and re-decide") from e

    @asynccontextmanager
    async def _act(self, op: str, device: str, caller: str, lease: str | None, swipe=None, lock_guard=False, think=True, **log):
        """One input call: log it, hold the device lock (the task), refuse while parked, check for the operator
        (and the lock screen), wait the think time with the frontmost-app check at its end, act."""
        target_udid = self.config.resolve(device).udid
        async with self._timed(op, device, backend=type(self.backend).__name__, **log) as rec:
            mark, t = self._recovery_mark(), time.perf_counter()
            t_act = t
            try:
                async with self._hold(device, caller, lease) as udid:  # every input names the hands that made it (INP-01)
                    rec["hold_ms"] = _ms_since(t)  # the device lock / lease check
                    task = self._task(udid, device, caller, lease)
                    self._unparked(device, udid)
                    t = time.perf_counter()
                    png = await self._operator_check(device, udid, None if lease else self.config.resolve(device).lease, rec)
                    rec["operator_ms"] = _ms_since(t)
                    if lock_guard:
                        t = time.perf_counter()
                        await self._unlocked(udid, swipe, png)
                        rec["lock_ms"] = _ms_since(t)
                    await self._think(op, udid, rec, task, device, pause=think)
                    self._unparked(device, udid)  # the Mac may have announced sleep during the pause
                    if op in POSITIONAL:
                        self._fresh_enough(device, rec, t_act, mark)
                    if check := INPUT_CHECK.get():
                        check()
                    if lease:
                        self.leases.require_live(lease, self.config.resolve(device).lease)
                    else:
                        self.leases.require_held(self.config.resolve(device).lease)
                    try:
                        yield udid, rec
                        if op == "swipe":
                            task.flicked = True
                    finally:
                        self._acted(udid)
            finally:
                if got := self._recoveries_since(mark, target_udid):
                    rec["recoveries"] = got  # a backend call failed inside this input and was reset/retried

    def _fresh_enough(self, device: str, rec: dict, t_act: float, mark: int) -> None:
        """Refuse a tap/swipe held too long before it starts (StaleInput): the look it was decided on is stale."""
        # the checks, not the think pause, and not the frontmost-app query: that query re-verifies the app at the input
        # (FG_MAX_AGE), and on iOS 27.2 it alone takes 0.4-2.3 s (one 5 s; 0.26 s on 27.0.1), live 2026-11-08
        fg_over = min(max(0.0, rec.get("fg_ms", 0) - FG_LEAD * 1000), 2000)  # ≤2 s of query overrun excused (27.2: 0.4-2.3 s)
        rec["pre_ms"] = pre = round(_ms_since(t_act) - rec.get("think_ms", 0) - fg_over, 1)
        why = []
        if pre > STALE_INPUT_S * 1000:
            steps = {k: rec[k] for k in ("hold_ms", "operator_ms", "lock_ms", "fg_ms") if rec.get(k)}
            why.append(f"the pre-input checks took {pre / 1000:.1f} s ({', '.join(f'{k} {v:.0f}' for k, v in steps.items())})")
        if got := self._recoveries_since(mark, self.config.resolve(device).udid):
            why.append("a backend call recovered (" + "; ".join(f"{r['call']}: {r['error']}" for r in got) + ")")
        if why:
            rec["stale"] = True
            raise StaleInput(f"stale: re-look. On {device} {' and '.join(why)}; zero input sent; look again and re-decide")

    def _recovery_mark(self) -> int:
        r = getattr(self.backend, "recoveries", None)
        return r[-1]["n"] if r else 0

    def _recoveries_since(self, mark: int, udid: str) -> list[dict]:
        # Older backends omit UDID: retain those conservatively, but a peer's recovery cannot stale this phone.
        return [dict(x) for x in getattr(self.backend, "recoveries", ())
                if x["n"] > mark and x.get("udid", udid) == udid]

    # --- tasks and parking (host/watch.py drives these) -------------------------------------
    def _task(self, udid: str, device: str, caller: str, lease: str | None) -> Task:
        """The running task on this phone for this owner: joined, or begun (power assertion, busy marker)."""
        owner, d = _owner(caller, lease), self.config.resolve(device)
        if (ended := self._ended.get(udid)) and ended[0] == owner:
            if not lease:
                self._ended.pop(udid)  # phonebridge's own hold: told once, then a new task may start
            raise TaskEnded(device, ended[1])
        t = self.tasks.get(udid)
        if t and t.owner != owner:
            self._end_task(udid)  # its lease is gone: authhub let a new holder in
            t = None
        if t is None:
            t = self.tasks[udid] = Task(owner, d.lease, d.alias)
            self.power.begin(d.alias, owner=owner, identity=d.lease)
            if hasattr(self.backend, "keep_warm"):
                t.warm = asyncio.get_running_loop().create_task(self._keep_warm(udid, t))
        if lease and t.token is None:
            t.token = lease
            t.watcher = asyncio.get_running_loop().create_task(self._watch_lease(udid, t))
        return t

    async def _keep_warm(self, udid: str, t: Task) -> None:
        """While the task runs (a session holds the phone), keep the touch session and the capture from idling out
        between inputs, so a long dwell doesn't make the next input pay a cold start; parked, nothing is kept."""
        while await asyncio.sleep(WARM_EVERY_S, True):
            if self.tasks.get(udid) is not t:
                return
            if not (self._parked.get(udid) or self._parked.get("*")):
                with contextlib.suppress(Exception):
                    self.backend.keep_warm(udid)

    async def _watch_lease(self, udid: str, t: Task) -> None:
        while await asyncio.sleep(min(self.task_poll, LEASE_REFRESH_S), True):
            alive = await asyncio.to_thread(self.leases.alive, t.token, t.identity)
            if alive is False:
                break  # None (authhub unreachable): don't know, keep the task (the vouch ages out after LEASE_TRUST_S)
            if alive:
                self.leases.vouch(t.token, t.identity)
        self.leases.forget(t.token, t.identity)
        if self.tasks.get(udid) is t:
            self._end_task(udid)

    def _end_task(self, udid: str, reason: str | None = None) -> None:
        """End the task on this phone; with a reason (stale, device_lost) its owner is told to back out."""
        t = self.tasks.pop(udid, None)
        if t is None:
            return
        if t.watcher and t.watcher is not asyncio.current_task():
            t.watcher.cancel()
        if t.warm and t.warm is not asyncio.current_task():
            t.warm.cancel()
        self.power.end(t.alias)
        if reason:
            self._ended[udid] = (t.owner, reason)
            if not t.token and self.leases.holds(t.identity) and not self._busy.get(t.identity):
                if timer := self._idle.pop(t.identity, None):
                    timer.cancel()
                self._release_idle(t.identity)

    def end_task(self, device: str, reason: str) -> None:
        self._end_task(self.config.resolve(device).udid, reason)

    def restore_tasks(self, live: Callable[[dict], bool]) -> list[str]:
        """After a restart: take back the tasks the busy marker lists whose lease `live(task)` still is."""
        for alias in self.power.restore(live):
            d, info = self.config.resolve(alias), self.power.tasks[alias]
            self.tasks[d.udid] = Task(info["owner"], d.lease, alias, app=info.get("app"))
        return [t.alias for t in self.tasks.values()]

    def park(self, reason: str, device: str | None = None) -> None:
        """No input to this phone (None: every phone) until unpark(reason); the screen capture stops with it."""
        self._parked.setdefault(self.config.resolve(device).udid if device else "*", set()).add(reason)
        if stop := getattr(self.backend, "stop_eyes", None):
            if device and "udid" in inspect.signature(stop).parameters:
                stop(f"park: {reason}", udid=self.config.resolve(device).udid)
            else:
                stop(f"park: {reason}")

    def unpark(self, reason: str, device: str | None = None) -> None:
        key = self.config.resolve(device).udid if device else "*"
        self._parked.get(key, set()).discard(reason)

    def parked(self, device: str) -> set[str]:
        return self._parked.get(self.config.resolve(device).udid, set()) | self._parked.get("*", set())

    def _unparked(self, device: str, udid: str) -> None:
        if reasons := self._parked.get(udid, set()) | self._parked.get("*", set()):
            raise Parked(device, reasons)

    async def recheck(self, device: str, found: list[dict] | None = None) -> dict:
        """Before resuming: is the phone on USB, trusted, unlocked, and what is frontmost? The lock guard and
        the foreground rule enforce the last two on the next input; this records them. `found`: a devices()
        listing already taken."""
        d = self.config.resolve(device)
        if found is None:
            if probe := getattr(self.backend, "device_info", None):
                info = await probe(d.udid)
                found = [info] if info else []
            else:
                found = await self.backend.devices()
        found = {udid_key(x["udid"]): x for x in found}
        info = found.get(udid_key(d.udid))
        out = {"usb": info is not None}
        out["trusted"] = out["usb"] and bool(info.get("paired", "error" not in info))
        if out["trusted"]:
            with contextlib.suppress(Exception):
                out["locked"] = (await self.backend.lock_state(d.udid)).get("locked")
            if query := getattr(self.backend, "foreground", None):
                with contextlib.suppress(Exception):
                    out["foreground"] = await query(d.udid)
        self._awake.pop(d.udid, None)  # after a sleep or a replug, the lock screen is checked again
        self._baseline.pop(d.udid, None)  # ...and the operator guard starts from a new settled screen
        return out

    # --- lock screen ------------------------------------------------------------------------
    async def _lock_reason(self, udid: str, png: bytes | None = None) -> tuple[str | None, tuple[int, int]]:
        png = png or await self.backend.screen(udid)
        items = await asyncio.to_thread(vision.recognize, png)  # accurate: fast mode misses the lock clock
        return vision.lock_screen(png, items), vision.png_size(png)

    async def _unlocked(self, udid: str, swipe: tuple[float, float, float, float] | None = None,
                        png: bytes | None = None) -> None:
        """Refuse a tap/swipe on the lock screen (a sideways swipe there opens the Camera), except the
        unlock gesture itself: a swipe up from the bottom edge."""
        if time.monotonic() - self._awake.get(udid, -AWAKE_S) < AWAKE_S:
            return
        reason, (w, h) = await self._lock_reason(udid, png)
        if reason:
            if swipe and swipe[1] >= 0.92 * h and swipe[1] - swipe[3] >= 0.25 * h and abs(swipe[2] - swipe[0]) <= 0.2 * w:
                return  # the unlock gesture
            raise PhoneLocked(f"phone is locked ({reason}); call unlock(), or swipe up from the bottom edge")
        self._awake[udid] = time.monotonic()

    async def unlock(self, device: str, caller: str = "local", lease: str | None = None) -> dict:
        """Wake the phone and swipe up from the bottom edge (only works with no passcode)."""
        async with self._timed("unlock", device) as rec, self._hold(device, caller, lease) as udid:
            self._task(udid, device, caller, lease)
            self._unparked(device, udid)
            png = await self._operator_check(device, udid, None if lease else self.config.resolve(device).lease, rec)
            reason, (w, h) = await self._lock_reason(udid, png)
            rec["was_locked"] = bool(reason)
            if reason:
                if check := INPUT_CHECK.get():
                    check()
                if lease:
                    self.leases.require_live(lease, self.config.resolve(device).lease)
                else:
                    self.leases.require_held(self.config.resolve(device).lease)
                if "display is off" in reason:
                    await self.backend.home(udid)
                    await asyncio.sleep(0.8)
                if check := INPUT_CHECK.get():
                    check()
                if lease:
                    self.leases.require_live(lease, self.config.resolve(device).lease)
                else:
                    self.leases.require_held(self.config.resolve(device).lease)
                await self.backend.swipe(udid, self._human(udid).swipe(w / 2, h * 0.985, w / 2, h * 0.45, 0.25))
                await asyncio.sleep(1.0)
                self._acted(udid)
                still, _ = await self._lock_reason(udid)
                if still:
                    raise PhoneLocked(f"still locked after swiping up ({still}); a passcode may be set")
            self._awake[udid] = time.monotonic()
            return {"was_locked": bool(reason), "unlocked": True}

    # --- hands ------------------------------------------------------------------------------
    async def tap(self, device: str, x: float, y: float, w: float = 0, h: float = 0, caller: str = "local",
                  lease: str | None = None, path: str = "report", think: bool = True, contact_mm: float = 9.0) -> None:
        """Tap the target centred on (x, y); with its size `w` x `h` (e.g. an OCR box) the finger lands
        somewhere inside it, else inside an assumed 44 pt target. `path` = "indigo" or "vdigitizer" is an experiment
        (pmd3 only), to compare what apps see: the same tap as a CoreDevice IndigoDigitizerEvent, or through a virtual
        digitizer of ours whose reports carry a `contact_mm` contact size."""
        _finite(x, y, w, h, contact_mm)
        if w < 0 or h < 0:
            raise ValueError("target dimensions must be non-negative")
        paths = {"report": "tap", "indigo": "tap_indigo", "vdigitizer": "tap_vdigitizer"}
        if path not in paths:
            raise ValueError(f"tap path {path!r}: {' | '.join(paths)}")
        if not hasattr(self.backend, paths[path]):
            raise RuntimeError(f"{type(self.backend).__name__} has no {path} tap path")
        extra = {"contact_mm": contact_mm} if path == "vdigitizer" else {}
        async with self._act("tap", device, caller, lease, lock_guard=True, think=think, x=x, y=y, path=path,
                             **extra) as (udid, rec):
            hx, hy, press = self._human(udid).tap(x, y, w, h)
            rec["at"], rec["press_ms"] = [round(hx), round(hy)], round(press * 1000)
            if path == "indigo":
                await self.backend.tap_indigo(udid, hx, hy, press)
            elif path == "vdigitizer":
                await self.backend.tap_vdigitizer(udid, hx, hy, press, contact_mm)
            else:
                await self.backend.tap(udid, hx, hy, press)

    async def ax_probe(self, device: str, ops: list[str], max_nodes: int = 120, caller: str = "local",
                       lease: str | None = None) -> dict:
        """EXPERIMENT (accessibility-tree spike): what the accessibility audit daemon says about the foreground app
        (phonebridge/axprobe.py lists the ops). Held like an input: "focus_first" moves the inspector focus, and the
        daemon scrolls the focused element into view."""
        if not hasattr(self.backend, "ax_probe"):
            raise RuntimeError(f"{type(self.backend).__name__} has no ax_probe")
        async with self._act("ax_probe", device, caller, lease, think=False, ops=list(ops)) as (udid, rec):
            out = await self.backend.ax_probe(udid, list(ops), max_nodes)
            rec["op_ms"] = out.get("op_ms")
            return out

    async def double_tap(self, device: str, x: float, y: float, w: float = 0, h: float = 0, caller: str = "local",
                         lease: str | None = None, think: bool = True) -> dict:
        """Two quick taps as ONE input (a like on IG/TikTok): the humanizer's pair, ~100–300 ms apart, with no think delay
        between them (two `tap` calls would get the ordinary think and land as two single taps)."""
        _finite(x, y, w, h)
        if w < 0 or h < 0:
            raise ValueError("target dimensions must be non-negative")
        async with self._act("double_tap", device, caller, lease, lock_guard=True, think=think, x=x, y=y) as (udid, rec):
            (x1, y1, p1), (x2, y2, p2), gap = self._human(udid).double_tap(x, y, w, h)
            rec["at"], rec["gap_ms"] = [[round(x1), round(y1)], [round(x2), round(y2)]], round(gap * 1000)
            await self.backend.tap(udid, x1, y1, p1)
            await asyncio.sleep(max(0.0, gap - p1))
            await self.backend.tap(udid, x2, y2, p2)
            return {"at": rec["at"], "gap_ms": rec["gap_ms"]}

    async def swipe(self, device: str, x1: float, y1: float, x2: float, y2: float, duration: float = 0.3,
                    caller: str = "local", lease: str | None = None, release: float | None = None,
                    path: str = "report", think: bool = True) -> None:
        """Swipe from about (x1, y1) to about (x2, y2) in about `duration` seconds, along a human path. With `release`
        (lift speed / mean speed, 0–1) the caller has planned the glide: length, duration and release are kept exactly
        and only the texture is drawn (the director's closed-loop scroll measures the travel afterwards). think=False:
        the caller paces its own inputs (no drawn pause; every guard still runs). `path` =
        "indigo" is an experiment (CoreDevice IndigoDigitizerEvents, pmd3 only)."""
        _finite(x1, y1, x2, y2, duration)
        if not 0 < duration <= SWIPE_MAX_S:
            raise ValueError(f"swipe duration must be in (0, {SWIPE_MAX_S}] seconds")
        if release is not None and not 0 <= release <= 1:
            raise ValueError("release must be in [0, 1]")
        if path not in ("report", "indigo"):
            raise ValueError(f"swipe path {path!r}: report | indigo")
        if path == "indigo" and not hasattr(self.backend, "swipe_indigo"):
            raise RuntimeError(f"{type(self.backend).__name__} has no indigo swipe path")
        async with self._act("swipe", device, caller, lease, swipe=(x1, y1, x2, y2), lock_guard=True, think=think,
                             dx=round(x2 - x1), dy=round(y2 - y1), release=release, path=path) as (udid, rec):
            human = self._human(udid)
            points = human.swipe(x1, y1, x2, y2, duration) if release is None else human.glide_path(x1, y1, x2, y2, duration, release)
            rec["points"], rec["path_ms"] = len(points), round(points[-1][2] * 1000)
            rec.update(swipe_shape(points))
            await (self.backend.swipe_indigo(udid, points) if path == "indigo" else self.backend.swipe(udid, points))

    async def scroll(self, device: str, direction: str = "up", caller: str = "local", lease: str | None = None) -> None:
        """One scroll flick of natural, varying length and speed. `direction` is where the content goes:
        "up" (the usual feed scroll), "down", "left", "right"."""
        if direction not in ("up", "down", "left", "right"):
            raise ValueError("direction must be up, down, left or right")
        udid = self.config.resolve(device).udid
        if udid not in self._size:
            await self.screen(device)
        w, h = self._size[udid]
        await self.swipe(device, *self._human(self.config.resolve(device).udid).flick(w, h, direction), caller=caller, lease=lease)

    async def type(self, device: str, text: str, submit: bool = False, caller: str = "local",
                   lease: str | None = None) -> None:
        """Type `text` (printable, ≤ TYPE_MAX chars) at a human cadence; `submit` presses Return after it."""
        if len(text) > TYPE_MAX:
            raise ValueError(f"type() takes at most {TYPE_MAX} characters; use paste() for longer text")
        if CONTROL.search(text):
            raise ValueError("type() text can't contain control characters (a newline would press Return); "
                             "pass submit=True to press Return at the end")
        async with self._act("type", device, caller, lease, chars=len(text), submit=submit) as (udid, rec):
            keys = self._human(udid).keystrokes(text + ("\n" if submit else ""))
            rec["keys"], rec["plan_ms"] = len(keys), round(sum(h + g for _, h, g in keys) * 1000)
            await self.backend.type(udid, keys)

    async def paste(self, device: str, text: str, caller: str = "local", lease: str | None = None) -> None:
        if len(text) > PASTE_MAX:
            raise ValueError(f"paste() takes at most {PASTE_MAX} characters")
        async with self._act("paste", device, caller, lease, chars=len(text)) as (udid, _):
            await self.backend.paste(udid, text)

    async def home(self, device: str, caller: str = "local", lease: str | None = None) -> None:
        async with self._act("home", device, caller, lease) as (udid, _):
            await self.backend.home(udid)

    async def launch(self, device: str, bundle_id: str, caller: str = "local", lease: str | None = None) -> None:
        """Bring `bundle_id` to the front. It becomes the task's app: input needs it frontmost from now on."""
        async with self._act("launch", device, caller, lease, bundle_id=bundle_id) as (udid, _):
            await self.backend.launch(udid, bundle_id)
            self._bind(udid, bundle_id)

    def _bind(self, udid: str, bundle_id: str) -> None:
        if t := self.tasks.get(udid):
            t.app, t.flicked, t.stopped = bundle_id, False, None
            self.power.begin(t.alias, app=bundle_id)

    async def open_url(self, device: str, url: str, bundle_id: str | None = None, caller: str = "local",
                       lease: str | None = None) -> None:
        """Open `url` in its app (picked from the scheme, or `bundle_id`) if `[policy] urls` allows it,
        e.g. shortcuts://run-shortcut?name=ReadyFeed."""
        if len(url) > URL_MAX or CONTROL.search(url) or " " in url:
            raise ValueError(f"URL must be ≤ {URL_MAX} characters without spaces or control characters")
        parts = urlsplit(url)
        if not any(fnmatchcase(url, pattern) for pattern in self.config.urls):
            raise PermissionError(f"{parts.scheme}://{parts.netloc} is not allowed by [policy] urls in the phonebridge config")
        bundle_id = bundle_id or URL_APPS.get(parts.scheme.lower())
        if not bundle_id:
            raise ValueError(f"no app known for {parts.scheme}:// URLs; pass bundle_id")
        # log scheme + host only: query strings can carry tokens
        async with self._act("open_url", device, caller, lease, url=f"{parts.scheme}://{parts.netloc}",
                             bundle_id=bundle_id) as (udid, _):
            await self.backend.launch(udid, bundle_id, url)
            self._bind(udid, bundle_id)

    # --- strips: evidence recordings from the daemon's own live frames (one capture owner) -------------------
    async def record_start(self, device: str, path: str, fps: float = 30.0, lease: str | None = None,
                           caller: str = "local") -> dict:
        """Record the eyes' live frames to a video-only mp4 at `path` (under the evidence root) at `fps`, until
        record_stop, the lease's end or STRIP_MAX_S. A second capture of the phone is never opened (two fight: each
        start re-enumerates the phone on USB). Capture resets and parks show as gaps, not ends."""
        from .strip import Strip
        d = self.config.resolve(device)
        if lease:
            if not self.leases.vouched(lease, d.lease, LEASE_TRUST_S):
                await asyncio.to_thread(self.leases.verify, lease, d.lease, LEASE_TRUST_S)
        elif self.config.lease_enforce:
            raise LeaseError("record_start needs the session's lease: the capture runs only during sessions")
        out = Path(path).expanduser().resolve()
        if out.suffix != ".mp4" or not out.is_relative_to(EVIDENCE_ROOT.resolve()):
            raise ValueError(f"record_start writes an .mp4 under {EVIDENCE_ROOT} only, not {path}")
        if d.udid in self._strips:
            raise RuntimeError(f"{device} is already recording to {self._strips[d.udid].path}; record_stop first")

        async def frames_of():
            if self.parked(device):  # never restart a capture the bridge stopped (park: the operator has the phone)
                return None
            try:
                return await self._frames(d.udid)  # also keeps the capture alive (its idle timer restarts)
            except Exception:
                return None

        async def alive():  # fail closed: a recording whose lease can't be verified stops (tw-phone-review, 11-08)
            return lease is None or await asyncio.to_thread(self.leases.alive, lease, d.lease) is True
        strip = Strip(out, fps, frames_of, max_s=STRIP_MAX_S, alive=alive if lease else None)
        async with self._timed("record_start", device, path=str(out), fps=fps):
            await strip.start()
            self._strips[d.udid] = strip
        return {"path": str(out), "fps": fps, "started": strip.started}

    async def record_stop(self, device: str) -> dict:
        """Stop the strip: {path, t0 (wall time of frame 0), fps, frames, dropped, duration_s, gaps: [[from, to] s],
        ended: stopped | max_s | lease ended, size}."""
        d = self.config.resolve(device)
        strip = self._strips.pop(d.udid, None)
        if strip is None:
            raise RuntimeError(f"{device} is not recording")
        async with self._timed("record_stop", device) as rec:
            out = await strip.stop()
            rec.update({k: out[k] for k in ("frames", "dropped", "duration_s", "ended")}, gaps=len(out["gaps"]))
        return out

    async def close(self) -> None:
        for strip in list(self._strips.values()):  # a strip is finished (its mp4 closed) before the capture goes
            with contextlib.suppress(Exception):
                await strip.stop()
        self._strips.clear()
        for timer in self._idle.values():
            timer.cancel()
        for task in (*self._settle.values(), *self._verdict.values()):
            task.cancel()
        for udid in list(self.tasks):
            self._end_task(udid)
        for pending in self._releasing.values():
            await pending
        await asyncio.to_thread(self.leases.release_all)
        await self.backend.close()
        for udid, held in list(self._reservations.items()):
            await self._release_reservation(udid, held["owner"])
