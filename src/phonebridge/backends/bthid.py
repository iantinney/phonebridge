"""Bluetooth HID input: the Mac's own Bluetooth as the iPhone's keyboard and mouse.

Hands go through `bthidd` (bthid/bthidd.swift, a LaunchAgent on the Mac) as raw HID reports; eyes
(devices, screen, lock state) come from another backend. iOS turns the pointer into touches only
with AssistiveTouch on, which this backend switches on over USB lockdown, so Developer Mode can stay off.

Pointer model (ghosthand.dev and egarl004/mirrordeck findings, iOS 17+): the absolute report puts the
pointer exactly on a target, but iOS ignores its buttons and the AssistiveTouch finger follows only
relative motion. So a tap = absolute move, then press/release on the relative report; a swipe =
absolute move to the start, press, relative steps, release, absolute report at the end.
"""

from __future__ import annotations

import asyncio
import logging
import itertools
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.services.installation_proxy import InstallationProxyService

from ..config import STATE_DIR
from ..humanize import Key, PathPoint
from ..ocr import png_size
from . import KEYS

ABS_MAX = 32767
SHIFT, GUI = 0x02, 0x08  # keyboard modifier bits
SPACE, ENTER = 0x2C, 0x28
HOME_USAGE = 0x0040  # consumer Menu; see docs/research/bluetooth-hid.md for what iOS accepts as Home
# Relative step length (report units) for drags, and its travel in points: d * (A + B * d).
# Measured by ghosthand on an SE2 / iOS 18.6 with Tracking Sensitivity below default; it scales
# with that setting, so swipe lengths are approximate.
DRAG_STEP, DRAG_A, DRAG_B = 12, 0.226, 0.0458
# Measured on the iPhone 12 (2026-10-18, `phonebridge measure`, Safari touch events): a press reads ~20 ms
# longer than the plan (two reports and bluetoothd spacing between press and release), each drag report
# costs ~2 ms beyond its sleep, and drags travelled 3-19% further than the model. Compensated here.
PRESS_OVERHEAD_MS, REPORT_COST_MS, DRAG_GAIN = 20.0, 2.0, 1.0
POINT_SCALE = {"iPhone13,2": 3, "iPhone17,3": 3}  # px per pt; @3x unless listed otherwise


def _kb(mods: int = 0, usage: int = 0) -> str:
    return bytes([1, mods, 0, usage, 0, 0, 0, 0, 0]).hex()


def _abs(x: int, y: int, buttons: int = 0) -> str:
    return bytes([2, buttons, x & 0xFF, x >> 8, y & 0xFF, y >> 8]).hex()


def _rel(dx: int = 0, dy: int = 0, buttons: int = 0) -> str:
    return bytes([3, buttons, dx & 0xFF, dy & 0xFF, 0]).hex()


def _consumer(usage: int) -> str:
    return bytes([4, usage & 0xFF, usage >> 8]).hex()


def even_keys(text: str) -> list[Key]:
    """A flat cadence, used only when no humanizer is given (tests)."""
    return [(ch, 0.05, 0.06) for ch in text]


def drag_path(path: list[PathPoint], px_per_pt: float) -> list[tuple[int, int, float]]:
    """Relative reports (dx, dy, ms after) that follow `path` (pixels, seconds) with its own timing: each
    segment's travel is split into reports at most DRAG_STEP long, its time spread evenly over them, and
    rounding error carried so the sum stays on the path."""
    per = DRAG_STEP * (DRAG_A + DRAG_B * DRAG_STEP)  # points per full-length report
    out, ex, ey = [], 0.0, 0.0
    for (x0, y0, t0), (x1, y1, t1) in zip(path, path[1:]):
        dx, dy, ms = (x1 - x0) / px_per_pt / DRAG_GAIN, (y1 - y0) / px_per_pt / DRAG_GAIN, (t1 - t0) * 1000
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            out.append((0, 0, max(0.0, ms - REPORT_COST_MS)))
            continue
        n = max(1, math.ceil(dist / per))
        # report length d for this slice, solving d * (A + B * d) = dist / n
        d = (-DRAG_A + (DRAG_A ** 2 + 4 * DRAG_B * dist / n) ** 0.5) / (2 * DRAG_B)
        for _ in range(n):
            ex += dx / dist * d
            ey += dy / dist * d
            sx, sy = round(ex), round(ey)
            ex -= sx
            ey -= sy
            out.append((sx, sy, max(0.0, ms / n - REPORT_COST_MS)))
    return out


class Daemon:
    """JSON-lines client for bthidd's Unix socket."""

    def __init__(self, path: Path):
        self.path = path

    async def call(self, op: str, **kw) -> dict:
        r, w = await asyncio.open_unix_connection(str(self.path))
        try:
            w.write((json.dumps({"op": op, **kw}) + "\n").encode())
            await w.drain()
            resp = json.loads(await r.readline())
        finally:
            w.close()
        if not resp.get("ok"):
            raise RuntimeError(f"bthidd {op}: {resp.get('error')}")
        return resp


class _Phone:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.address: str | None = None
        self.product: str | None = None
        self.size: tuple[int, int] | None = None  # screenshot pixels
        self.pos: tuple[int, int] | None = None  # last absolute pointer report


log = logging.getLogger(__name__)


class BtHidBackend:
    def __init__(self, eyes, sock: Path = STATE_DIR / "bthid.sock", connect_timeout: float = 15.0,
                 screen_timeout: float = 15.0, awake: float = 60.0, keystrokes: Callable[[str], list[Key]] = even_keys):
        self.eyes = eyes
        self._assistive_on: set[str] = set()  # udids whose AssistiveTouch we switched on
        self.keystrokes = keystrokes  # cadence for text this backend types on its own (Spotlight launch)
        self.daemon = Daemon(sock)
        self.connect_timeout = connect_timeout
        self.screen_timeout = screen_timeout  # a stalled capture must not hold the phone
        self.awake = awake
        self._phones: dict[str, _Phone] = {}

    # --- eyes: delegated ----------------------------------------------------------------------
    def __getattr__(self, name):  # eyes-only extras, e.g. pair() for USB trust
        return getattr(self.eyes, name)

    async def devices(self) -> list[dict]:
        return await self.eyes.devices()

    async def screen(self, udid: str) -> bytes:
        png = await asyncio.wait_for(self.eyes.screen(udid), self.screen_timeout)
        self._phones.setdefault(udid, _Phone()).size = png_size(png)
        return png

    async def lock_state(self, udid: str) -> dict:
        return await self.eyes.lock_state(udid)

    # --- link ---------------------------------------------------------------------------------
    async def _ready(self, udid: str) -> _Phone:
        """Bluetooth address and AssistiveTouch over USB lockdown, then an open HID link."""
        p = self._phones.setdefault(udid, _Phone())
        if p.address is None:
            p.address, p.product = await self._usb_setup(udid)
        if p.size is None:
            await self.screen(udid)
        if not await self._linked(p.address):
            st = await self.daemon.call("status")
            if not any(ph["address"] == p.address and ph["paired"] for ph in st["phones"]):
                await self.daemon.call("discoverable", seconds=300)
                raise RuntimeError("iPhone not paired with the Mac over Bluetooth: on the iPhone open Settings > "
                                   "Bluetooth, tap this Mac under Other Devices, then Pair and confirm the code on both")
            start = time.monotonic()
            for i in itertools.count():
                if i % 6 == 0:  # bthidd dials control then interrupt (~1-2 s); a no-op while a dial is in flight
                    await self.daemon.call("connect", address=p.address)
                await asyncio.sleep(0.5)
                if await self._linked(p.address):
                    break
                if time.monotonic() - start > self.connect_timeout:
                    raise RuntimeError(f"Bluetooth HID link to {p.address} did not open (Bluetooth off on the iPhone?)")
            p.pos = None  # a fresh link has no known pointer position
        return p

    async def _usb_setup(self, udid: str) -> tuple[str, str]:
        async with await create_using_usbmux(udid, autopair=False) as ld:
            if not await ld.get_assistive_touch():
                await ld.set_assistive_touch(True)  # without it iOS shows no pointer and ignores clicks
                self._assistive_on.add(udid)  # ours to switch off again: it is a visible pointer and an app-readable flag
            address = (await ld.get_value(key="BluetoothAddress")).lower().replace(":", "-")
            return address, await ld.get_value(key="ProductType")

    async def assistive_touch_off(self) -> list[str]:
        """Switch AssistiveTouch off on every phone this backend switched it on (close, idle, or on request)."""
        done = []
        for udid in list(self._assistive_on):
            try:
                async with await create_using_usbmux(udid, autopair=False) as ld:
                    await ld.set_assistive_touch(False)
                done.append(udid)
            except Exception as e:
                log.warning("assistive touch stays on for %s: %s", udid, e)
            self._assistive_on.discard(udid)
        return done

    async def _linked(self, address: str) -> bool:
        st = await self.daemon.call("status")
        return any(ph["address"] == address and ph["interrupt"] for ph in st["phones"])

    async def _send(self, p: _Phone, reports: list[list]) -> dict:
        # awake: bthidd keeps the phone from auto-locking this long after the input (an invisible Left
        # Control tap every 2 s of idle), so an agent's think time doesn't lose the session.
        return await self.daemon.call("send", address=p.address, reports=reports, awake=self.awake)

    async def _run(self, udid: str, build) -> None:
        p = self._phones.setdefault(udid, _Phone())
        async with p.lock:
            p = await self._ready(udid)
            await self._send(p, build(p))

    # --- pointer ------------------------------------------------------------------------------
    def _to_abs(self, p: _Phone, x: float, y: float) -> tuple[int, int]:
        w, h = p.size
        clamp = lambda v: max(0, min(ABS_MAX, round(v)))
        return clamp(x * ABS_MAX / (w - 1)), clamp(y * ABS_MAX / (h - 1))

    def _glide(self, p: _Phone, x: int, y: int) -> list[list]:
        """Eased absolute reports from the last position to (x, y). The AssistiveTouch pointer's travel is
        not a touch (apps never see it), so it is plain; the touch itself is what the humanizer shaped."""
        if p.pos is None:
            p.pos = (x, y)
            return [[_abs(x, y), 110]]
        (x0, y0), n = p.pos, 4
        out = [[_abs(round(x0 + (x - x0) * (1 - (1 - i / n) ** 2)), round(y0 + (y - y0) * (1 - (1 - i / n) ** 2))), 20]
               for i in range(1, n)]
        out.append([_abs(x, y), 110])  # land exactly, let the pointer settle before the click
        p.pos = (x, y)
        return out

    async def tap(self, udid: str, x: float, y: float, press: float) -> None:
        def build(p):
            ax, ay = self._to_abs(p, x, y)
            # Press and release go in both reports: iOS reads buttons from the relative one.
            return self._glide(p, ax, ay) + [
                [_abs(ax, ay, 1), 0], [_rel(buttons=1), round(max(10.0, press * 1000 - PRESS_OVERHEAD_MS), 1)],
                [_abs(ax, ay, 0), 0], [_rel(), 0],
            ]
        await self._run(udid, build)

    async def swipe(self, udid: str, path: list[PathPoint]) -> None:
        def build(p):
            ax, ay = self._to_abs(p, path[0][0], path[0][1])
            out = self._glide(p, ax, ay) + [[_abs(ax, ay, 1), 0], [_rel(buttons=1), 15]]
            out += [[_rel(dx, dy, 1), round(ms, 1)] for dx, dy, ms in drag_path(path, POINT_SCALE.get(p.product, 3))]
            ex, ey = self._to_abs(p, path[-1][0], path[-1][1])
            p.pos = (ex, ey)
            # Hold still briefly before lifting, or Safari loses the momentum; then resync the pointer.
            return out + [[_rel(buttons=1), 50], [_rel(), 20], [_abs(ex, ey), 0]]
        await self._run(udid, build)

    # --- keyboard -----------------------------------------------------------------------------
    @staticmethod
    def _keys(keys: list[Key]) -> list[list]:
        if bad := sorted({c for c, _, _ in keys if c not in KEYS}):
            raise ValueError(f"type() is ASCII-only over Bluetooth HID; not {''.join(bad)!r}")
        out = []
        for ch, hold, gap in keys:
            usage, shift = KEYS[ch]
            if gap > 0:
                if out:
                    out[-1][1] += round(gap * 1000, 1)  # the pause before a key follows the previous release
                else:
                    out.append([_kb(), round(gap * 1000, 1)])
            if shift:  # Shift leads the key, as on pmd3 (one report with both: iOS may read the letter first)
                out.append([_kb(SHIFT), 30])
            out += [[_kb(SHIFT if shift else 0, usage), round(hold * 1000, 1)], [_kb(), 0]]
        return out

    async def type(self, udid: str, keys: list[Key]) -> None:
        reports = self._keys(keys)
        await self._run(udid, lambda p: reports)

    async def paste(self, udid: str, text: str) -> None:
        # The pasteboard is a CoreDevice service (Developer Mode); over Bluetooth only typing exists.
        await self.type(udid, text)

    async def home(self, udid: str) -> None:
        await self._run(udid, lambda p: [[_consumer(HOME_USAGE), 60], [_consumer(0), 0]])

    async def launch(self, udid: str, bundle_id: str, url: str | None = None) -> None:
        """Spotlight, as a person would: Cmd+Space, the app's name, Return. A URL goes to the eyes backend."""
        if url:
            return await self.eyes.launch(udid, bundle_id, url)
        async with await create_using_usbmux(udid, autopair=False) as ld:
            async with InstallationProxyService(lockdown=ld) as ip:
                apps = await ip.get_apps(bundle_identifiers=[bundle_id])
        if bundle_id not in apps:
            raise RuntimeError(f"{bundle_id} is not installed")
        name = apps[bundle_id].get("CFBundleDisplayName") or apps[bundle_id].get("CFBundleName")
        await self._run(udid, lambda p: [[_kb(GUI, SPACE), 80], [_kb(), 450]]
                        + self._keys(self.keystrokes(name)) + [[_kb(), 300], [_kb(0, ENTER), 60], [_kb(), 0]])

    async def close(self) -> None:
        await self.assistive_touch_off()  # the grey pointer and UIAccessibility.isAssistiveTouchRunning go with us
        await self.eyes.close()  # the Bluetooth link stays up: bthidd owns it, not this process

