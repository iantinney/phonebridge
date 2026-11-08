"""The device operator's task API (ARC-01; PLAN §3.3, §3.5 A7/A8, §7): the VM submits, the Mac runs and journals.

    run_submit(device, task) -> {task_id, status}     task: {task_id, identity, kind, payload, leases{device,account},
                                                             lease_s, stop_texts, vm_now?}
    run_events(task_id, after_seq, limit?, extend_s?) -> {events, status, final, reason, result}   (pulled, never pushed)
    run_status(task_id) · run_park(task_id) (idempotent) · run_resume(task_id, leases, lease_s) · run_ack(task_id, key)
    run_submit / run_events / run_status replies also carry {proto, build, task_proto} (proto.py; authhub's driver
    warns on a proto mismatch; apps/autopilot warns when `task_proto` < 2: an old daemon, no queue)

Rules this module enforces:
- No task without live leases: the device's (`device:<alias>`) and the account's, both checked with authhub at
  submit and resume. The Mac keeps a lease *copy*: `lease_s` seconds counted on its monotonic clock from the VM's
  figure (VM time is the authority, ARC-04); each pull may extend it. When the copy runs out the task parks at the
  next step boundary, so with the tunnel down it keeps going only until the lease would have expired.
- Steps are journaled `started` / `ok` (journal.py). A restarted daemon parks whatever was in flight and resumes it
  only when the VM calls `run_resume` with live leases; completed steps are never re-run.
- One Post tap per task: the tap step first asks the VM to acknowledge ledger `publishing` (an event with
  `need_ack`, answered by `run_ack`); without a fresh acknowledgement it parks. POST_TAPPED is fsync'd before the
  physical tap, and a task whose journal says POST_TAPPED never taps again: it goes to verify.
- A task parked at STAGED / ARMED for 30 minutes backs out only while its original leases remain valid.
  Otherwise cleanup is deferred until a resume supplies fresh leases; no input is sent without ownership.
  Only successful discard ends `backed_out`, allowing the VM to return its ledger row to `claimed`.
- Stop texts come from the VM (config/pacing.toml, one source); a match ends the task `stopped` and goes Home.
- Task protocol 2 (`task.v`, lane one-loop; v1 is unchanged): one runner per phone. A v2 task is admitted (leases,
  kind) at submit, then waits `queued` in its phone's FIFO until no other task runs there; a parked one leaves the
  queue and rejoins it at the back on resume. While a v2 task holds a phone, a v1 submit for it is refused (two
  controllers on one phone, review F02). `post` (run/post.py) is a v2 kind only.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import bridge as bridge_mod
from .. import ocr as vision
from ..config import STATE_DIR
from ..lease import LeaseError, LeaseUnavailable
from ..proto import TASKS, hello
from .journal import Journal

RUNS = STATE_DIR / "runs"
# The bridge's refusals that send zero input and mean "not now" (host lane's guards included where present):
# the task parks, resumable. Looked up by name so this runs against either bridge.
PARK_ERRORS = tuple(c for c in (getattr(bridge_mod, n, None) for n in (
    "OperatorActive", "PhoneLocked", "NotFrontmost", "Parked", "TaskEnded", "UnrecognisedScreen"))
    if isinstance(c, type)) + (LeaseError,)
STALE_S = 30 * 60  # STAGED / ARMED this long without progress: back out (PLAN §3.5 A7)
TASK_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class Refused(Exception):
    """The task can't be accepted (no live lease, unknown kind, malformed)."""


class VerificationUnavailable(Refused):
    """No fresh lease answer; an existing verified deadline remains authoritative."""


class Park(Exception):
    """Stop at this step boundary; resumable."""


class Stop(Exception):
    """The platform pushed back (a stop text on screen): final, the VM hands off to the operator."""


@dataclass
class Ctx:
    api: TaskAPI
    journal: Journal
    task: dict
    device: str
    lease: str | None  # the device lease token (the bridge verifies it on every input)
    data: dict = field(default_factory=dict)

    @property
    def caller(self):
        return f"task:{self.task['task_id']}"

    @property
    def bridge(self):
        return self.api.bridge

    def note(self, **fields) -> dict:
        return self.api._append(self.journal, **fields)

    async def see(self) -> tuple[bytes, list[dict]]:
        """One screenshot and the OCR of that very frame (review F10: pixels and text never from two frames)."""
        png = await self.bridge.screen(self.device)
        return png, await asyncio.to_thread(vision.recognize, png, False)

    def save(self, name: str, png: bytes, items: list[dict] | None) -> str:
        """Keep a frame (and its OCR) in the run directory; returns its ref."""
        frames = self.journal.path / "frames"
        frames.mkdir(mode=0o700, exist_ok=True)
        ref = f"frames/{len(list(frames.glob('*.png'))) + 1:03d}-{name}.png"
        (self.journal.path / ref).write_bytes(png)
        if items is not None:
            (self.journal.path / ref).with_suffix(".ocr.json").write_text(json.dumps(items))
        self.data["last_frame"] = ref
        return ref

    async def settle(self, seconds: float | None = None) -> None:
        await asyncio.sleep(self.api.settle if seconds is None else seconds)

    async def look(self, name: str, allow: tuple = (), lock: bool = False) -> tuple[list[dict], int, int, str]:
        """Screenshot + OCR, saved as a frame in the run directory; a stop text (other than `allow`) raises Stop;
        with `lock`, the lock screen parks the task."""
        png, items = await self.see()
        ref = self.save(name, png, items)
        hit = stop_reason(self.task.get("stop_texts", ()), items, allow)
        w, h = vision.png_size(png)
        if hit:
            raise Stop(f"stop text on screen: {hit!r} ({ref})")
        if lock and vision.lock_screen(png, items):
            raise Park(f"the phone is locked ({ref}); resumes when it is unlocked")
        return items, w, h, ref

    async def tap_frac(self, fx: float, fy: float, w: int, h: int) -> None:
        await self.bridge.tap(self.device, fx * w, fy * h, lease=self.lease, caller=self.caller)


def stop_reason(texts, items: list[dict], allow: tuple = ()) -> str | None:
    """The OCR line showing a stop text, matched like authhub.pacing_config.stop_reason: a 1–2 word phrase only
    on a short line (a button, a toast), a 3+ word phrase anywhere in a line."""
    skip = {a.lower() for a in allow}
    for i in items:
        low = i["text"].strip().lower()
        for t in texts:
            tl = t.lower()
            if tl in skip or tl not in low:
                continue
            if len(tl.split()) >= 3 or len(low.split()) - len(tl.split()) <= 3:
                return i["text"].strip()
    return None


class Flow:
    """A task kind: named steps, each idempotent unless listed in `irreversible`."""
    kind = ""
    steps: tuple[str, ...] = ()
    irreversible: frozenset = frozenset()
    mock = False  # never touches the phone; every event says so
    v = 1  # the lowest task protocol that may submit this kind

    def __init__(self, task: dict):
        self.task = task

    async def run_step(self, name: str, ctx: Ctx) -> dict | None:
        return await getattr(self, f"step_{name}")(ctx)

    async def cleanup(self, ctx: Ctx) -> None:
        """Best effort after a stop or failure (the read flows go Home)."""

    async def back_out(self, ctx: Ctx) -> None:
        """Discard what a stale task staged (the draft)."""

    def result(self, ctx: Ctx) -> dict:
        return {}


class PostFlow(Flow):
    """The post protocol: PICKED → STAGED → ARMED → (VM ack) POST_TAPPED → CONFIRMED | POST_UNCONFIRMED. The
    platform screens are the flows lane's; subclasses fill the hooks."""
    steps = ("pick", "stage", "arm", "tap", "verify")
    irreversible = frozenset({"tap"})

    @property
    def key(self) -> str:
        return self.task["payload"]["content_key"]

    async def step_pick(self, ctx):
        return {"state": "PICKED", "content_key": self.key}

    async def step_stage(self, ctx):
        await self.do_stage(ctx)
        return {"state": "STAGED"}

    async def step_arm(self, ctx):
        return {"state": "ARMED"}

    async def step_tap(self, ctx):
        await ctx.api._wait_ack(ctx, self.key, timeout=getattr(self, "ack_timeout", None))
        ctx.note(state="POST_TAPPED", content_key=self.key)  # fsync'd before the tap: never a second one
        await self.do_tap(ctx)
        return {}

    async def step_verify(self, ctx):
        return {"state": "CONFIRMED" if await self.do_verify(ctx) else "POST_UNCONFIRMED"}

    async def back_out(self, ctx):
        await self.do_back_out(ctx)

    async def do_stage(self, ctx): ...
    async def do_tap(self, ctx): ...
    async def do_back_out(self, ctx): ...

    async def do_verify(self, ctx) -> bool:
        return False


class RehearsePost(PostFlow):
    """The post protocol with no phone at all (label: mock): journal, acknowledgement and recovery on the real
    daemon, for the TST-03 contract suite. Nothing is staged, tapped or posted."""
    kind = "rehearse_post"
    mock = True

    async def do_verify(self, ctx) -> bool:
        return True


APPS = {"ig": ("com.burbn.instagram", "ig"), "tiktok": ("com.zhiliaoapp.musically", "tiktok")}
NUM = re.compile(r"^(\d[\d,.]*)\s*([KkMm]?)$")
HEALTHY = ("meets all our guidelines", "don't have removed content", "haven't posted anything", "no issues",
           "good standing", "nothing to show", "not affected", "no violations", "everything looks good")
NOTICE = ("removed", "violat", "restricted", "at risk", "can't", "limited", "disabled", "suspended")


def account_status(text: str) -> str:
    """healthy | notice | unknown from an Account Status screen's text. The healthy wording names what is absent
    ("You don't have removed content … feature restrictions …", IG 2026-10-18), so it is checked first."""
    low = text.lower().replace("’", "'")
    if any(m in low for m in HEALTHY):
        return "healthy"
    return "notice" if any(m in low for m in NOTICE) else "unknown"


def _count(text: str) -> int | None:
    m = NUM.match(text.strip())
    if not m:
        return None
    n = float(m.group(1).replace(",", ""))
    return int(n * {"k": 1e3, "m": 1e6}.get(m.group(2).lower(), 1))


def _has(items, *anchors) -> bool:
    return any(vision.find(items, a) for a in anchors)


DENY_LINES = ("share", "next", "post", "discard", "done")  # a whole OCR line: a composer's buttons
DENY_PHRASES = ("save draft", "new post", "new reel", "tag people", "add location", "add music", "edit cover",
                "advanced settings", "write a caption", "add a caption", "share to")


def composer(items) -> str | None:
    """A posting screen's button or field on screen (the reason no blind tap may happen), else None."""
    for i in items:
        low = i["text"].strip().lower()
        if low in DENY_LINES or any(p in low for p in DENY_PHRASES):
            return i["text"].strip()
    return None


def tab_screen(items, h: int) -> bool:
    """Only icons in the tab-bar band: an app's main tab screen (OCR reads an icon as 1–2 characters at most)."""
    return not any(i["y"] > 0.9 * h and len(i["text"].strip()) > 2 for i in items)


class AppFlow(Flow):
    """A flow inside one account's app: open it (unlock, launch, frontmost check), go Home at the end."""

    def __init__(self, task):
        super().__init__(task)
        plat = task["identity"].split(":")[0]
        if plat not in APPS:
            raise Refused(f"{self.kind} is not wired for {plat}")
        self.bundle, self.plat = APPS[plat]

    def screen_id(self, items) -> str:
        if _has(items, "followers") and _has(items, "following"):
            return f"{self.plat}_profile"
        if _has(items, "Settings and activity"):
            return f"{self.plat}_settings"
        if self.plat == "ig" and _has(items, "Your story"):
            return "ig_feed"
        return f"{self.plat}_unknown"

    async def in_app(self, ctx) -> str | None:
        """Why this is not our app in front (None: it is). Needs the bridge's frontmost-app query (lane host);
        without it nothing is tapped blind."""
        fg = getattr(ctx.bridge, "foreground", None)
        if fg is None:
            return "no frontmost-app check on this bridge"
        front = await fg(ctx.device)
        return None if front == self.bundle else f"{front or 'nothing'} is in front, not {self.bundle}"

    async def step_open(self, ctx):
        png = await ctx.bridge.screen(ctx.device)
        if vision.lock_screen(png, await asyncio.to_thread(vision.recognize, png, False)):
            await ctx.bridge.unlock(ctx.device, lease=ctx.lease, caller=ctx.caller)  # no passcode on the posting phone; else PhoneLocked
            await ctx.settle(1.0 if ctx.api.settle else 0)
        await ctx.look("before_open", lock=True)
        for attempt in range(2):  # launch can leave another app in front (live 19:52Z: Shadowrocket)
            await ctx.bridge.launch(ctx.device, self.bundle, lease=ctx.lease, caller=ctx.caller)
            await ctx.settle(3.0 if ctx.api.settle else 0)
            if not (why := await self.in_app(ctx)) or why.startswith("no frontmost"):
                break
        else:
            await ctx.look("open_not_in_front")
            raise Park(f"{why} after two launches: zero taps; resumes later")
        items, _, _, ref = await ctx.look("open")
        return {"screen_id": self.screen_id(items), "frame_ref": ref}

    async def step_home(self, ctx):
        if (why := await self.in_app(ctx)) and not why.startswith("no frontmost"):
            return {"skipped": f"{why}: Home not pressed (only our app may get input)"}
        await ctx.bridge.home(ctx.device, lease=ctx.lease, caller=ctx.caller)
        await ctx.settle(1.0 if ctx.api.settle else 0)
        png = await ctx.bridge.screen(ctx.device)
        frames = ctx.journal.path / "frames"
        frames.mkdir(mode=0o700, exist_ok=True)
        ref = f"frames/{len(list(frames.glob('*.png'))) + 1:03d}-home.png"
        (ctx.journal.path / ref).write_bytes(png)
        return {"screen_id": "home", "frame_ref": ref}

    async def cleanup(self, ctx):
        await ctx.bridge.home(ctx.device, lease=ctx.lease, caller=ctx.caller)


class SnapshotStatus(AppFlow):
    """Read-only: open the app, read the profile (handle, counts) and the account status screen, go Home.
    Instagram: profile tab → ≡ → Settings and activity → Account Status. TikTok: profile only for now."""
    kind = "snapshot_status"
    steps = ("open", "profile", "status", "home")
    PROFILE_TAB = (0.897, 0.930)  # Instagram's rightmost tab, the avatar (calibrated on the iPhone 12, IG 451)
    MENU = (0.928, 0.085)  # ≡ on the own profile, top right ("+" = create is top LEFT: never near it)

    async def step_profile(self, ctx):
        items, w, h, ref = await ctx.look("profile_before")
        for _ in range(2):
            if self.screen_id(items).endswith("_profile"):
                break
            if (why := composer(items)) or not tab_screen(items, h):  # A5: no blind tap on a screen we don't know
                return {"screen_id": f"{self.plat}_unknown", "frame_ref": ref, "found": False,
                        "skipped": f"not a main tab screen ({why or 'text in the tab bar'}): no tap"}
            if why := await self.in_app(ctx):  # e.g. resumed after a restart with the phone on Home
                await ctx.bridge.launch(ctx.device, self.bundle, lease=ctx.lease, caller=ctx.caller)
                await ctx.settle()
                if why := await self.in_app(ctx):
                    return {"screen_id": f"{self.plat}_unknown", "frame_ref": ref, "found": False,
                            "skipped": f"{why}: no tap"}
                items, w, h, ref = await ctx.look("profile_relaunched")
                continue
            if self.plat == "tiktok" and (hit := vision.find([i for i in items if i["y"] > 0.85 * h], "Profile")):
                await ctx.bridge.tap(ctx.device, hit["x"], hit["y"], hit["w"], hit["h"], lease=ctx.lease, caller=ctx.caller)
            else:
                await ctx.tap_frac(*self.PROFILE_TAB, w, h)
            await ctx.settle()
            items, w, h, ref = await ctx.look("profile")
        sid = self.screen_id(items)
        if not sid.endswith("_profile"):
            return {"screen_id": sid, "frame_ref": ref, "found": False}
        ctx.data.update(self.read_profile(items, w, h))
        return {"screen_id": sid, "frame_ref": ref, "found": True, **{k: ctx.data.get(k) for k in
                                                                      ("handle", "posts", "followers", "following")}}

    def read_profile(self, items, w, h) -> dict:
        out = {}
        for label in ("posts", "followers", "following"):
            lab = vision.find(items, label)
            if not lab:
                continue
            above = [i for i in items if _count(i["text"]) is not None and abs(i["x"] - lab["x"]) < 0.08 * w
                     and 0 < lab["y"] - i["y"] < 0.06 * h]
            if above:
                out[label] = _count(min(above, key=lambda i: lab["y"] - i["y"])["text"])
        top = [(m.group(1), i) for i in items if i["y"] < 0.12 * h and not CLOCKISH.fullmatch(i["text"].strip())
               and (m := HANDLE.fullmatch(i["text"].strip()))]
        if top:
            out["handle"] = max(top, key=lambda t: t[1]["w"])[0]  # "samplecreator.co v": the name, not the ⌄
        return out

    async def step_status(self, ctx):
        if self.plat != "ig":
            return {"skipped": "account status reader not wired for this app yet"}
        items, w, h, ref = await ctx.look("status_before")
        if not self.screen_id(items).endswith("_profile"):
            return {"skipped": "not on the profile", "frame_ref": ref}
        allow = ("Account Status", "Account status")  # the destination itself, not a notice
        await ctx.tap_frac(*self.MENU, w, h)
        await ctx.settle()
        items, w, h, ref = await ctx.look("menu", allow)
        field = vision.find(items, "Search")
        if _has(items, "Settings and activity") and field and field["y"] < 0.25 * h:
            await ctx.bridge.tap(ctx.device, field["x"], field["y"], field["w"], field["h"], lease=ctx.lease, caller=ctx.caller)
            await ctx.settle()
            await ctx.bridge.type(ctx.device, "account status", lease=ctx.lease, caller=ctx.caller)  # Settings' own search box
            await ctx.settle()
        for n in range(6):
            items, w, h, ref = await ctx.look(f"menu{n}", allow)
            below = field["y"] + 0.02 * h if field else 0
            hit = next((i for i in items if i["text"].strip().lower() == "account status" and i["y"] > below), None)
            if hit:
                await ctx.bridge.tap(ctx.device, hit["x"], hit["y"], hit["w"], hit["h"], lease=ctx.lease, caller=ctx.caller)
                await ctx.settle()
                items, w, h, ref = await ctx.look("account_status", allow)
                text = " ".join(i["text"] for i in items if i["y"] > 0.1 * h)
                low = text.lower()
                status = account_status(text)
                ctx.data.update(status=status, status_text=text[:600])
                return {"screen_id": "ig_account_status", "frame_ref": ref, "status": status}
            if n < 5:
                try:
                    await ctx.bridge.scroll(ctx.device, "up", lease=ctx.lease, caller=ctx.caller)
                except Exception as e:  # noqa: BLE001 - e.g. the host guard's UnrecognisedScreen: stop looking
                    if type(e).__name__ != "UnrecognisedScreen":
                        raise
                    return {"skipped": f"menu scroll refused: {e}", "frame_ref": ref}
                await ctx.settle()
        return {"skipped": "Account Status not found in the menu", "frame_ref": ref}

    def result(self, ctx):
        return {k: ctx.data[k] for k in ("handle", "posts", "followers", "following", "status", "status_text")
                if k in ctx.data}


async def director_visit(phone, device: str, *, app: str, identity: str, minutes: float, evidence: Path,
                         lease: str | None, knobs: dict | None = None, full_length: bool = False, **director) -> dict:
    """One read-only visit by the session director v1 (lane director, `phonebridge.director`): it alone decides what
    happens on the feed, post by post from what is on screen. Its evidence (events.jsonl, frames, session.json) goes
    to `evidence`, inside the run directory."""
    from datetime import date

    from ..director.cues import CueEngine, vision_classifier
    from ..director.persona import prepare
    from ..director.session import Director

    persona, params = prepare("public")  # BEH-16 persistence, the shared hand, persona + operator params validated before anything runs
    knobs = knobs or {}  # Ian's operator.toml, sent by the VM with the visit
    if "exploration_rate" in knobs:
        persona.exploration = {**persona.exploration, "chance": knobs["exploration_rate"]}
    try:
        classify = vision_classifier(persona.interest)
    except Exception:  # noqa: BLE001 - not on a Mac: cues stay OCR + motion
        classify = None
    d = Director(phone, device, app=app, persona=persona, evidence=evidence, read_only=True, identity=identity,
                 cues=CueEngine(classify), lease=lease, via="taskapi", require_look=True,  # live: S3 on the daemon's look
                 comments=knobs.get("comments", "preview"), params=params, day=date.today(), **director)  # a do task's detour(task)
    return await d.run(minutes=minutes, full_length=full_length)


def director_on_feed(app: str, items: list[dict]) -> bool:
    from ..director.cues import on_feed

    return on_feed(app, items)


class Watched:
    """The bridge as the director sees it inside a task, with the director's daemon-client shapes (fast-eyes' `look`,
    stream frames for `screen` with DVT stills counted as `stream_fallbacks`): every look first checks the lease copy
    (a VM park or an expired copy stops the visit at the next post), every recognition's text is checked for the
    account's stop texts, and every input checks the lease copy again and carries the task's device lease."""

    def __init__(self, ctx: Ctx):
        self.ctx, self.tid = ctx, ctx.journal.task["task_id"]
        self.stream_fallbacks, self.last_fallback = 0, None

    def _stops(self, items: list[dict]) -> list[dict]:
        if hit := stop_reason(self.ctx.task.get("stop_texts", ()), items):
            raise Stop(f"stop text on screen mid-visit: {hit!r}")
        return items

    async def look(self, device: str, platform: str | None = None) -> dict | None:
        """None without fast-eyes on this bridge (like the daemon client): the director then won't run live (S3)."""
        self.ctx.api._check(self.tid)
        if not hasattr(self.ctx.bridge, "look"):
            return None
        meta = await self.ctx.bridge.look(device, platform)
        self._stops(meta.get("items") or [])
        return meta

    async def screen(self, device: str) -> bytes:
        self.ctx.api._check(self.tid)
        if hasattr(self.ctx.bridge, "frame"):
            try:
                return (await self.ctx.bridge.frame(device, format="png"))[0]
            except Exception as e:  # noqa: BLE001 - the stream is down: a DVT still answers, counted (CONTROLLER_BAR L1)
                self.stream_fallbacks += 1
                self.last_fallback = str(e)[:160]
        return await self.ctx.bridge.screen(device)

    async def frame_at(self, device: str, seq) -> bytes | None:
        """The very frame a recent look judged (the bridge keeps them by seq): text and pixels of one frame (director's
        paired_frame). None when it is no longer kept or this bridge has no stream."""
        self.ctx.api._check(self.tid)
        if seq is None or not hasattr(self.ctx.bridge, "frame"):
            return None
        try:
            return (await self.ctx.bridge.frame(device, format="png", seq=seq))[0]
        except (LookupError, TypeError):
            return None

    async def ocr(self, device: str, fast: bool = False) -> list[dict]:
        self.ctx.api._check(self.tid)
        return self._stops(await self.ctx.bridge.ocr(device, fast))

    async def swipe(self, device: str, x1, y1, x2, y2, duration: float = 0.3, caller: str = "", lease=None) -> None:
        self.ctx.api._check(self.tid)
        await self.ctx.bridge.swipe(device, x1, y1, x2, y2, duration, lease=self.ctx.lease)

    async def tap(self, device: str, x, y, w=0, h=0, caller: str = "", lease=None) -> None:
        self.ctx.api._check(self.tid)
        await self.ctx.bridge.tap(device, x, y, w, h, lease=self.ctx.lease)

    async def type(self, device: str, text: str, submit: bool = False, caller: str = "", lease=None) -> None:
        self.ctx.api._check(self.tid)
        await self.ctx.bridge.type(device, text, submit=submit, lease=self.ctx.lease)


MAX_VISIT_MINUTES = 60


class Browse(AppFlow):
    """One visit (M5 autopilot): open the app, get to its feed, let the session director browse it for the task's
    minutes (read-only: no writes are wired), go Home. This flow only gets to the feed, on recognised screens only:
    from the own profile the Home tab, from a known deeper screen (where the daily read leaves the app: Account Status,
    Settings) an edge-back; a posting screen or anything unrecognised gets no gesture and no visit (the screens lane's
    slow model is the way through unknown screens). The marker check alone never makes a profile or settings the feed
    (lane director's markers are substrings: "follow" is in "followers")."""
    kind = "browse"
    steps = ("open", "feed", "home")
    HOME_TAB = (0.1, 0.93)  # Instagram's leftmost tab (PROFILE_TAB is the rightmost)
    DEEPER = ("Account Status", "Settings and activity", "Settings and privacy")  # screens a back gesture leaves
    NAV = 4  # navigation moves before giving up on the feed
    visit = staticmethod(director_visit)
    on_feed = staticmethod(director_on_feed)

    def __init__(self, task):
        super().__init__(task)
        m = task["payload"].get("minutes")
        if isinstance(m, bool) or not isinstance(m, int | float) or not 0 < m <= MAX_VISIT_MINUTES:
            raise Refused(f"browse.minutes must be a number in (0, {MAX_VISIT_MINUTES}]")
        self.minutes = float(m)
        self.app = {"ig": "instagram", "tiktok": "tiktok"}[self.plat]
        self.knobs = task["payload"].get("settings") or {}
        if self.knobs.get("capture") is False:
            raise Refused("screen capture is off (operator.toml capture.screen = false): no visit, the director never "
                          "acts without seeing")

    async def step_feed(self, ctx):
        ref = await self.to_feed(ctx)
        res = await self.visit(Watched(ctx), ctx.device, app=self.app, identity=self.task["identity"],
                               minutes=self.minutes, evidence=ctx.journal.path / "director", lease=ctx.lease,
                               knobs=self.knobs)
        ctx.data["visit"] = {k: res[k] for k in ("status", "reason", "label", "seconds", "writes", "counts", "cue")
                             if k in res}
        if res.get("status") != "ok":
            if str(res.get("reason") or "").startswith(("stop_screen", "phone: Stop")):  # the platform pushed back: a stop hand-off + hold
                raise Stop(f"the platform pushed back during the visit: {res.get('reason')}")
            raise RuntimeError(f"the director ended the visit: {res.get('reason') or res.get('status')}")
        return {"screen_id": f"{self.plat}_feed", "frame_ref": ref, "posts": (res.get("counts") or {}).get("posts")}

    def where(self, items) -> str:
        """composer | profile | deeper | feed | unknown, from the OCR of one look (the composer check first)."""
        if composer(items):
            return "composer"
        sid = self.screen_id(items)
        if sid.endswith("_profile"):
            return "profile"
        if sid.endswith("_settings") or _has(items, *self.DEEPER):
            return "deeper"
        return "feed" if self.on_feed(self.app, items) else "unknown"

    async def to_feed(self, ctx) -> str:
        items, w, h, ref = await ctx.look("feed_before", lock=True)
        for n in range(self.NAV + 1):
            here = self.where(items)
            if here == "feed":
                return ref
            if here == "composer":
                raise RuntimeError(f"a posting screen is open ({composer(items)!r}, {ref}): zero gestures, no visit")
            if here == "unknown" or n == self.NAV:
                raise RuntimeError(f"not on the {self.app} feed: an unrecognised screen ({ref}); zero gestures, no visit")
            if (why := await self.in_app(ctx)) and not why.startswith("no frontmost"):
                raise Park(f"{why}: zero gestures; resumes later")
            if here == "profile":
                await self.home_tab(ctx, items, w, h)
            else:  # a known deeper screen: back out one level
                await ctx.bridge.swipe(ctx.device, 4, 0.45 * h, 0.7 * w, 0.46 * h, 0.25, lease=ctx.lease, caller=ctx.caller)
            await ctx.settle()
            items, w, h, ref = await ctx.look(f"feed_nav{n + 1}", lock=True)
        raise AssertionError("unreachable")

    async def home_tab(self, ctx, items, w, h) -> None:
        if self.plat == "tiktok" and (hit := vision.find([i for i in items if i["y"] > 0.85 * h], "Home")):
            await ctx.bridge.tap(ctx.device, hit["x"], hit["y"], hit["w"], hit["h"], lease=ctx.lease, caller=ctx.caller)
        else:
            await ctx.tap_frac(*self.HOME_TAB, w, h)

    def result(self, ctx):
        return ctx.data.get("visit", {})


CLOCKISH = re.compile(r"\d{1,2}[:.]\d{2}")
HANDLE = re.compile(r"@?([A-Za-z0-9._]{2,30})(?:\s+[v⌄˅▾>])?")


class TaskAPI:
    def __init__(self, bridge, runs_dir: Path = RUNS, flows: dict | None = None, verify=None, clock=time.monotonic,
                 ack_timeout: float = 60.0, stale_s: float = STALE_S, poll: float = 0.2, settle: float = 2.0,
                 park_wait: float = 30.0, sweep_s: float = 60.0):
        self.bridge, self.runs_dir = bridge, Path(runs_dir)
        self.flows = FLOWS if flows is None else flows
        self.verify = verify or (lambda token, identity: bridge.leases.verify(token, identity, cache_s=0))
        self.clock, self.ack_timeout, self.stale_s, self.poll, self.settle = clock, ack_timeout, stale_s, poll, settle
        self.park_wait, self.sweep_s = park_wait, sweep_s
        self.live: dict[str, dict] = {}  # task_id -> {leases, device, deadline, park}: memory only, never on disk
        self.runners: dict[str, asyncio.Task] = {}
        self.acks: dict[str, dict] = {}  # task_id -> {key, event}
        self.journals: dict[str, Journal] = {}  # one Journal per task in this process: one seq counter
        self.locks: dict[str, asyncio.Lock] = {}  # submit / park / resume / back-out of one task, one at a time
        self.queues: dict[str, list[str]] = {}  # udid -> v2 tasks waiting for that phone, in order (memory only)
        self.holders: dict[str, str] = {}  # udid -> the v2 task running on it
        self._recovered = False
        self._sweeper: asyncio.Task | None = None

    # --- the API ------------------------------------------------------------------------------------
    async def submit(self, device: str, task: dict) -> dict:
        self._recover()
        tid = str(task.get("task_id") or "")
        if not TASK_ID.fullmatch(tid):
            raise Refused("task_id: 1-64 of [A-Za-z0-9._-]")
        async with self._lock(tid):
            return {**await self._submit(device, task, tid), **hello()}

    async def _submit(self, device: str, task: dict, tid: str) -> dict:
        try:
            j = self._open(tid)
            return {"task_id": tid, "status": j.status}  # idempotent: a resubmission is the same task
        except KeyError:
            pass
        kind = task.get("kind")
        if kind not in self.flows:
            raise Refused(f"{kind!r} is not wired on this device operator (wired: {', '.join(sorted(self.flows))})")
        v = task.get("v", 1)
        if v not in range(1, TASKS + 1):
            raise Refused(f"task protocol {v!r}: this device operator runs 1..{TASKS} (deploy the Mac)")
        if v < self.flows[kind].v:
            raise Refused(f"{kind!r} needs task protocol {self.flows[kind].v} (the per-phone queue), not {v}")
        for k in ("identity", "payload"):
            if k not in task:
                raise Refused(f"task needs {k!r}")
        flow_cls = self.flows[kind]
        if not flow_cls.mock and not task.get("stop_texts"):
            raise Refused("a task that touches the phone needs the account's stop_texts (from the VM's pacing config)")
        flow_cls(task)  # raises Refused for an unsupported identity
        leases = await self._check_leases(device, task)
        if v == 1 and (held := self.holders.get(self.bridge.config.resolve(device).udid)):
            raise Refused(f"busy: task {held} holds {device} (one runner per phone)")
        j = self.journals[tid] = Journal.create(self.runs_dir, {**task, "device": device}, clock=self.clock)
        if (skew := task.get("vm_now")) and abs(float(skew) - time.time()) > 30:
            self._append(j, skew_s=round(float(skew) - time.time(), 1))  # ARC-04: logged, VM time rules
        self.live[tid] = {"leases": leases, "device": device, "deadline": leases["deadline"],
                          "park": False}
        self._admit(tid)
        return {"task_id": tid, "status": j.status}

    async def events(self, task_id: str, after_seq: int = 0, limit: int | None = None,
                     extend_s: float | None = None) -> dict:
        self._recover()
        j = self._open(task_id)
        if extend_s is not None and task_id in self.live:
            live = self.live[task_id]
            try:
                checked = await self._check_leases(live["device"],
                    {**j.task, "leases": live["leases"], "lease_s": extend_s})
            except VerificationUnavailable:
                pass  # never extend on an outage; the already verified deadline still bounds input
            except Refused:
                live["park"] = True
                raise
            else:
                live["deadline"] = checked["deadline"]
        await self._settle(j)
        return self._summary(j, j.events(after_seq, limit))

    async def status(self, task_id: str) -> dict:
        self._recover()
        j = self._open(task_id)
        await self._settle(j)
        return self._summary(j, None)

    async def park(self, task_id: str) -> dict:
        """Idempotent. A running task parks at its next step boundary (an ack wait at once); the summary may still
        say `running` if a step outlasts `park_wait`: the runner writes `parked` itself, never this call."""
        self._recover()
        j = self._open(task_id)
        async with self._lock(task_id):
            if j.final or j.status == "parked":
                return self._summary(j, None)
            if task_id in self.live:
                self.live[task_id]["park"] = True
            self._dequeue(task_id)
            r = self.runners.get(task_id)
            if r and not r.done():
                if ack := self.acks.get(task_id):
                    ack["event"].set()  # wake an ack wait so it parks now
                await asyncio.wait({r}, timeout=self.park_wait)
            elif not j.final and j.status != "parked":
                self._append(j, state="parked", reason="parked by the VM")
        return self._summary(j, None)

    async def resume(self, task_id: str, leases: dict, lease_s: float) -> dict:
        self._recover()
        j = self._open(task_id)
        await self._settle(j)
        async with self._lock(task_id):  # two resumes never start two runners (two runners = two taps)
            if j.final or not self.runner_done(task_id) or self._queued(task_id):
                return self._summary(j, None)
            checked = await self._check_leases(j.task["device"], {**j.task, "leases": leases, "lease_s": lease_s})
            self.live[task_id] = {"leases": checked, "device": j.task["device"],
                                  "deadline": checked["deadline"], "park": False}
            if j.post_state in ("STAGED", "ARMED") and not self.flows[j.task["kind"]].mock:
                # Another controller may have changed media while we relinquished the phone.
                # Discard under fresh leases; a new task must stage the approved artifact again.
                await self._back_out(j)
                return self._summary(j, None)
            self._append(j, state="resumed", after=(j.done_steps or ["-"])[-1])
            self._admit(task_id)
        return self._summary(j, None)

    async def ack(self, task_id: str, content_key: str) -> dict:
        a = self.acks.get(task_id)
        if not a:  # a stale request (an earlier run's need_ack): harmless, nothing is tapped on it
            return {"task_id": task_id, "acked": None, "reason": "not waiting for an acknowledgement now"}
        if a["key"] != content_key:
            raise Refused(f"task {task_id} waits for an acknowledgement of {a['key']!r}, not {content_key!r}")
        a["acked"] = True
        a["event"].set()
        return {"task_id": task_id, "acked": content_key}

    # --- test and server helpers ----------------------------------------------------------------------
    def runner_done(self, task_id: str) -> bool:
        r = self.runners.get(task_id)
        return r is None or r.done()

    async def wait_runner(self, task_id: str) -> None:
        if r := self.runners.get(task_id):
            await r

    # --- internals ------------------------------------------------------------------------------------
    def _open(self, task_id: str) -> Journal:
        if task_id not in self.journals:
            self.journals[task_id] = Journal.open(self.runs_dir, task_id, clock=self.clock)
        return self.journals[task_id]

    def _lock(self, task_id: str) -> asyncio.Lock:
        return self.locks.setdefault(task_id, asyncio.Lock())

    def _append(self, j: Journal, **fields) -> dict:
        if self.flows.get(j.task["kind"], Flow).mock:
            fields["mock"] = True
        return j.append(**fields)

    async def _check_leases(self, device: str, task: dict) -> dict:
        try:
            requested = float(task.get("lease_s", 0))
        except (ValueError, TypeError):
            raise Refused("lease_s must be finite and positive") from None
        if not math.isfinite(requested) or requested <= 0:
            raise Refused("lease_s must be finite and positive")
        deadline = self.clock() + requested
        leases = task.get("leases") or {}
        dev = self.bridge.config.resolve(device)
        need = {"device": dev.lease, "account": task["identity"]}
        for role, identity in need.items():
            token = leases.get(role)
            if not token:
                raise Refused(f"no {role} lease ({identity}): the device operator takes no task without leases")
            try:
                started = self.clock()
                remaining = float(await asyncio.to_thread(self.verify, token, identity))
                if not math.isfinite(remaining) or remaining <= 0:
                    raise LeaseError("invalid remaining lifetime")
                deadline = min(deadline, started + remaining)
            except (LeaseUnavailable, subprocess.TimeoutExpired, OSError) as e:
                raise VerificationUnavailable(f"lease verification unavailable for {identity}: {e}") from None
            except Exception as e:  # noqa: BLE001 - invalid lease or invalid broker answer
                raise Refused(f"no live lease on {identity}: {e}") from None
        return {"device": leases["device"], "account": leases["account"], "deadline": deadline}

    def _start(self, tid: str) -> None:
        self.runners[tid] = asyncio.get_running_loop().create_task(self._run(tid))

    # --- task protocol 2: one runner per phone, the rest wait in its queue ---------------------------------------
    def _udid(self, tid: str) -> str:
        return self.bridge.config.resolve(self._open(tid).task["device"]).udid

    def _admit(self, tid: str) -> None:
        """v1: start now (as always). v2: join the phone's queue; start when it is this task's turn."""
        if self._open(tid).task.get("v", 1) < 2:
            self._start(tid)
            return
        udid = self._udid(tid)
        self.queues.setdefault(udid, []).append(tid)
        self._pump(udid)
        if tid in self.queues.get(udid, ()):
            self._append(self._open(tid), state="queued", ahead=self.queues[udid].index(tid) + bool(self._busy(udid)))

    def _queued(self, tid: str) -> bool:
        return any(tid in q for q in self.queues.values())

    def _dequeue(self, tid: str) -> None:
        for q in self.queues.values():
            if tid in q:
                q.remove(tid)

    def _busy(self, udid: str) -> str | None:
        """The task running on this phone (v2 holder, or a v1 runner), else None."""
        if (h := self.holders.get(udid)) and not self.runner_done(h):
            return h
        return next((t for t, r in self.runners.items() if not r.done() and t in self.live
                     and self.bridge.config.resolve(self.live[t]["device"]).udid == udid), None)

    def _pump(self, udid: str) -> None:
        q = self.queues.get(udid)
        if not q or self._busy(udid):
            return
        tid = self.holders[udid] = q.pop(0)
        self._start(tid)

    def _check(self, tid: str) -> None:
        live = self.live.get(tid)
        if not live:
            raise Park("no live lease copy (the daemon restarted): waits for the VM to resume it")
        if live["park"]:
            raise Park("parked by the VM")
        if self.clock() >= live["deadline"]:
            raise Park("the lease copy expired: the VM stopped renewing (tunnel down?)")

    async def _run(self, tid: str) -> None:
        j = self._open(tid)
        try:
            if j.final:
                return
            async with self.bridge.controller(j.task["device"], f"task:{tid}", lambda: self._check(tid)):
                await self._run_reserved(tid)
        except LeaseError as e:
            self._append(j, state="parked", reason=str(e))
        finally:
            if self.queues or self.holders:  # task protocol 2: the phone is free for the next in its queue
                udid = self._udid(tid)
                if self.holders.get(udid) == tid:
                    del self.holders[udid]
                asyncio.get_running_loop().call_soon(self._pump, udid)

    async def _run_reserved(self, tid: str) -> None:
        j = self._open(tid)
        flow = ctx = None
        if j.final:  # backed out (or ended) while it waited in its phone's queue
            return
        try:
            flow = self.flows[j.task["kind"]](j.task)
            ctx = Ctx(self, j, j.task, j.task["device"], self.live.get(tid, {}).get("leases", {}).get("device"))
            self._append(j, state="running")
            for name in flow.steps:
                if name in j.done_steps:
                    continue
                if name in flow.irreversible and j.post_state in ("POST_TAPPED", "CONFIRMED", "POST_UNCONFIRMED"):
                    self._append(j, step=name, outcome="skipped", reason="already tapped: never a second time")
                    continue
                self._check(tid)
                self._append(j, step=name, outcome="started")
                out = await flow.run_step(name, ctx) or {}
                self._append(j, step=name, outcome="ok", **out)
                if j.final:
                    return
            self._append(j, state="done", result=flow.result(ctx))
        except Park as e:
            self._append(j, state="parked", reason=str(e))
        except PARK_ERRORS as e:
            self._append(j, state="parked", reason=f"{type(e).__name__}: {e}")
        except Stop as e:  # clean up first: a final state means the task has let go of the phone
            await self._cleanup(flow, ctx, j)
            self._append(j, state="stopped", reason=str(e), frame_ref=ctx.data.get("last_frame") if ctx else None)
        except Exception as e:  # noqa: BLE001 - recorded; the VM decides (reads requeue, outward fail)
            await self._cleanup(flow, ctx, j)
            self._append(j, state="failed", reason=f"{type(e).__name__}: {e}"[:500])
        finally:
            self.acks.pop(tid, None)

    async def _cleanup(self, flow, ctx, j) -> None:
        if flow is None or ctx is None:
            return
        try:
            await flow.cleanup(ctx)
        except Exception as e:  # noqa: BLE001 - best effort
            self._append(j, step="cleanup", outcome="failed", reason=f"{type(e).__name__}: {e}"[:300])

    async def _wait_ack(self, ctx: Ctx, key: str, need: dict | None = None, timeout: float | None = None,
                        on_timeout=None, between=None) -> None:
        """Wait for the VM's `run_ack(key)`. `need`: the event asking for it (default: the Post tap's ledger
        `publishing`); `on_timeout`: the exception to raise when it doesn't come (default: park, not tapping);
        `between`: awaited between waits (a look at the screen while the VM checks)."""
        tid = ctx.journal.task["task_id"]
        a = self.acks[tid] = {"key": key, "event": asyncio.Event(), "acked": False}
        ctx.note(**(need or {"need_ack": key,
                             "reason": "ledger publishing must be acknowledged by the VM before the Post tap"}))
        timeout = self.ack_timeout if timeout is None else timeout
        end = time.monotonic() + timeout
        while not a["acked"]:
            self._check(tid)
            left = end - time.monotonic()
            if left <= 0:
                raise on_timeout or Park(f"no VM acknowledgement of ledger publishing within {timeout:g} s: not tapping")
            try:
                await asyncio.wait_for(a["event"].wait(), timeout=min(left, self.poll * 10))
            except TimeoutError:
                if between:
                    await between()
            a["event"].clear()
        self.acks.pop(tid, None)
        self._check(tid)  # an ack and a park (or an expired lease copy) in the same wake: the park wins

    def _age(self, e: dict) -> float:
        now = self.clock()
        return now - e["mono"] if now >= e["mono"] else time.time() - e["ts"]  # another boot: wall clock

    async def _settle(self, j: Journal) -> None:
        """Back out stale drafts with live ownership; defer invalid leases until fresh resume."""
        tid = j.task["task_id"]
        if j.final or j.post_state not in ("STAGED", "ARMED") or not self.runner_done(tid):
            return
        if j.task.get("v", 1) >= 2 and (self._queued(tid) or self._busy(self._udid(tid))):
            return  # its back-out touches the phone: not while another task drives it
        cleanup = j.last(lambda e: e.get("step") == "back_out")
        if cleanup and cleanup.get("outcome") == "deferred":
            return
        entered = j.last(lambda e: e.get("state") == j.post_state)
        if self._age(entered) < self.stale_s:
            return
        async with self._lock(tid):
            if not j.final and self.runner_done(tid) and not self._queued(tid):  # a resume may have re-queued it
                await self._back_out(j)

    async def _back_out(self, j: Journal) -> None:
        flow = self.flows[j.task["kind"]](j.task)
        tid = j.task["task_id"]
        live = self.live.get(tid, {})
        try:
            checked = (await self._check_leases(j.task["device"],
                       {**j.task, "leases": live.get("leases", {}), "lease_s": 30})
                       if not flow.mock else None)
            ctx = Ctx(self, j, j.task, j.task["device"], checked["device"] if checked else None)
            def check():
                if checked and self.clock() >= checked["deadline"]:
                    raise Park("lease expired before draft cleanup")
            async with self.bridge.controller(ctx.device, ctx.caller, check):
                self._append(j, step="back_out", outcome="started", reason="discard relinquished draft context")
                await flow.back_out(ctx)
                self._append(j, step="back_out", outcome="ok")
        except Refused as e:
            self._append(j, step="back_out", outcome="deferred",
                         reason=f"fresh leases required to discard draft: {e}"[:300])
            return
        except Exception as e:
            self._append(j, step="back_out", outcome="failed", reason=f"{type(e).__name__}: {e}"[:300])
            return  # failed cleanup is not evidence that the draft is gone
        self._append(j, state="backed_out", content_key=j.task["payload"].get("content_key"))

    def _recover(self) -> None:
        """Once per process: whatever a previous daemon left in flight is parked until the VM resumes it, and a
        sweeper backs out STAGED / ARMED tasks nobody asks about any more."""
        if self._recovered:
            return
        self._recovered = True
        for j in Journal.all(self.runs_dir, clock=self.clock):
            self.journals.setdefault(j.task["task_id"], j)
            if not j.final and j.status != "parked" and j.task["task_id"] not in self.runners:
                self._append(j, state="parked", reason="daemon restarted: waits for the VM to resume it")
        self._sweeper = asyncio.get_running_loop().create_task(self._sweep())

    async def _sweep(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_s)
            for j in [j for j in self.journals.values() if not j.final]:
                try:
                    await self._settle(j)
                except Exception:  # noqa: BLE001 - one bad journal must not stop the sweeper
                    continue

    def _summary(self, j: Journal, events: list | None) -> dict:
        last = j.last(lambda e: e.get("state") in ("parked", "stopped", "failed", "backed_out"))
        done = j.last(lambda e: "result" in e)
        out = {"task_id": j.task["task_id"], "status": j.status, "final": j.final, "post_state": j.post_state,
               "last_seq": j.events()[-1]["seq"] if j.events() else 0,
               "reason": last["reason"] if last and last.get("reason") and last["state"] == j.status else None,
               "result": done["result"] if done else None, "steps_done": j.done_steps, **hello()}
        if events is not None:
            out["events"] = events
        return out


def register(mcp, tool, bridge) -> TaskAPI:
    """Add the run_* tools to phonebridge's MCP server (mcp_server.build)."""
    api = TaskAPI(bridge)

    @tool
    async def run_submit(device: str, task: dict) -> dict:
        """Submit a task (kinds: snapshot_status, rehearse_post, browse; with task.v = 2 also post) under live device +
        account leases."""
        return await api.submit(device, task)

    @tool
    async def run_events(task_id: str, after_seq: int = 0, limit: int | None = None,
                         extend_s: float | None = None) -> dict:
        """Events after `after_seq` (contiguous seq); `extend_s` renews the Mac's lease copy (VM seconds left)."""
        return await api.events(task_id, after_seq, limit, extend_s)

    @tool
    async def run_status(task_id: str) -> dict:
        """The task's status from its journal."""
        return await api.status(task_id)

    @tool
    async def run_park(task_id: str) -> dict:
        """Park the task at its next safe step (idempotent)."""
        return await api.park(task_id)

    @tool
    async def run_resume(task_id: str, leases: dict, lease_s: float) -> dict:
        """Resume a parked task from its journal under live leases."""
        return await api.resume(task_id, leases, lease_s)

    @tool
    async def run_ack(task_id: str, content_key: str) -> dict:
        """The VM's acknowledgement that ledger `publishing` is written for content_key (one Post tap)."""
        return await api.ack(task_id, content_key)

    return api



# `post` (run/post.py, built on PostFlow and Ctx) is added by the package's __init__, which imports both
FLOWS: dict[str, type[Flow]] = {"snapshot_status": SnapshotStatus, "rehearse_post": RehearsePost, "browse": Browse}
