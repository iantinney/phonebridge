"""pymobiledevice3 backend: the live screen stream + DVT screenshots, CoreDevice HID touch/keyboard over USB
(touch: iOS 27+).

Per phone it keeps one RSD (from a running `pymobiledevice3 remote tunneld` if there is one, else
an in-process userspace tunnel — no root, but one phone per process), and one `ScreenStream`
session (`stream.py`) while input or fast looks are flowing: the display stream that opens
backboardd's HID gate, decoded into `frames()`. The session shows the phone's screen-capture
indicator and blocks in-app camera/mic, so it closes after `idle` seconds without input or looks.
`screen()` returns a fresh stream frame when the session is up, else a DVT still (0.3–0.9 s).

The eyes (`eyes=`): "avf" = the phone's screen through AVFoundation (`avf.py`, the QuickTime path, ~58 fps,
no tunnel, the Mac's GUI session; default on macOS), "stream" = the CoreDevice display stream decoded in-process
(`stream.py`; also what the HID gate runs on); "auto" picks avf on macOS. A capture that fails to open falls
back to the stream's frames, and `screen()` to a DVT still. The capture runs only while something looks
(PLAN E26: the phone's UIScreen.isCaptured is true meanwhile, as with QuickTime) and stops after `idle` s.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import io
import logging
import plistlib
import struct
import sys
import time

from PIL import Image
from pymobiledevice3 import usbmux
from pymobiledevice3.exceptions import AlreadyMountedError, CoreDeviceError, TunneldConnectionError
from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.remote.core_device.app_service import AppServiceService
from pymobiledevice3.remote.core_device.device_info import DeviceInfoService
from pymobiledevice3.remote.core_device.hid_service import (
    HID_BUTTON_STATE_DOWN,
    HID_BUTTON_STATE_UP,
    KEY_LEFT_GUI,
    KEY_LEFT_SHIFT,
    KEY_V,
    KEYBOARD_ATTACH_SETTLE_SECONDS,
    TOUCHSCREEN_STATE_CONTACT,
    TOUCHSCREEN_STATE_RELEASE,
    IndigoHIDService,
)
from pymobiledevice3.remote.core_device.pasteboard_service import PasteboardService
from pymobiledevice3.remote.xpc_message import XpcInt64Type, XpcUInt64Type
from pymobiledevice3.remote.userspace_tunnel import UserspaceRsdTunnel
from pymobiledevice3.services.diagnostics import DiagnosticsService
from pymobiledevice3.services.installation_proxy import InstallationProxyService
from pymobiledevice3.services.dvt.instruments.device_info import DeviceInfo
from pymobiledevice3.services.dvt.instruments.dvt_provider import DvtProvider
from pymobiledevice3.services.dvt.instruments.screenshot import Screenshot
from pymobiledevice3.services.mobile_image_mounter import auto_mount
from pymobiledevice3.tunneld.api import get_tunneld_device_by_udid

from ..humanize import Key, PathPoint
from .. import avf, presence
from ..ocr import png_size
from ..stream import Frames, ScreenStream, display_rect, margin_dark
from . import KEYS

log = logging.getLogger(__name__)

HID_MAX = 65535  # touchscreen coordinates are normalized 0..65535, origin top-left
POINTER_SURFACE = 1281  # the host's "CoreDevice touchscreenGesture" surface: Generic Desktop / Mouse usage
TOUCHSCREEN_SERVICE = 257  # the phone's built-in "CoreDevice touchscreen" surface (Digitizer page, usage Touch Screen)
TOUCHSCREEN_REPORT_ID = 0x09


MACH_TICK_NS = 125 / 3  # mach_absolute_time on Apple silicon: 24 MHz, timebase 125/3 ns per tick (Mac and iPhone alike)


def mach_ticks() -> int:
    """The host's mach_absolute_time (what Xcode's Device Hub writes into remoteTimestamp); off macOS the monotonic
    clock converted to the same 24 MHz tick scale, so inter-report deltas mean the same thing."""
    if sys.platform == "darwin":
        import ctypes
        libc = ctypes.CDLL(None)
        libc.mach_absolute_time.restype = ctypes.c_uint64
        return int(libc.mach_absolute_time())
    return int(time.monotonic_ns() / MACH_TICK_NS)


def touch_report(state: int, x: int, y: int, timestamp: int | None = None) -> bytes:
    """Apple's 58-byte DigitizerReport for one contact, laid out as the phone reads it (UniversalHID.framework,
    confirmed against dtuhidd by iPhoneMirror and against live five-finger input by idevice):

        0      report id 0x09
        1-2    contact count 1, contact count maximum 5
        3-27   five 5-byte contacts: state (bits 0-4 slot, 5 resting, 6 touch, 7 range), X u16 LE, Y u16 LE
        28-39  scroll collection (zero for touch)
        40-44  contact identity per slot (Xcode sends 0x02 for its contact; kept)
        45-52  remoteTimestamp, Int64 LE (read only when the report is >= 53 bytes)
        53-57  gesture extension flags (zero)

    pymobiledevice3 11.34.1 writes a 6-byte timestamp at byte 44: the phone then reads our timestamps / 256 and a
    stray byte in identity slot 4 (lane fast-eyes, 2026-10-19). Xcode's Device Hub writes the Mac's mach_absolute_time
    (24 MHz ticks) there, so that is the unit (`mach_ticks`); pymobiledevice3 sent nanoseconds. State, X and Y stay
    exactly as pymobiledevice3 sends them (0xC2 contact / 0x02 release, proven on this phone)."""
    ts = mach_ticks() if timestamp is None else timestamp
    return (bytes([TOUCHSCREEN_REPORT_ID, 0x01, 0x05, state & 0xFF]) + struct.pack("<HH", x & 0xFFFF, y & 0xFFFF)
            + bytes(20)  # contacts 2-5 unused
            + bytes(12)  # scroll collection
            + bytes([0x02, 0, 0, 0, 0])  # identities per slot
            + struct.pack("<q", ts & 0x7FFFFFFFFFFFFFFF)  # remoteTimestamp at byte 45
            + bytes(5))
# EXPERIMENT (touch-radius evidence, 2026-10-19): a host-registered virtual digitizer whose descriptor declares the
# contact's Width/Height (Digitizer 0x48/0x49, in 0.1 mm), which IOHIDEventDriver turns into MinorRadius/MajorRadius.
# Its report keeps bytes 0-7 of the 58-byte DigitizerReport and puts Confidence, Width and Height in bytes 8-12, which
# that struct reads as an unused contact 2. So one tap answers which parser iOS runs: Apple's fixed struct (radius stays
# 4220 pt) or our descriptor (radius follows Height). iPhone 12 panel: 64.6 x 139.8 mm.
VDIG_SERVICE_ID = 0x100002002  # session-specific, like pymobiledevice3's virtual keyboard (0x100002001)
VDIG_DESCRIPTOR = bytes([
    0x05, 0x0D, 0x09, 0x04, 0xA1, 0x01, 0x85, TOUCHSCREEN_REPORT_ID,  # Digitizer / Touch Screen, report 0x09
    0x15, 0x00, 0x25, 0x05, 0x75, 0x08, 0x95, 0x01,
    0x09, 0x54, 0x81, 0x02, 0x09, 0x55, 0x81, 0x02,  # contact count, contact count maximum (bytes 1, 2)
    0x09, 0x22, 0xA1, 0x02,  # finger
    0x09, 0x51, 0x25, 0x1F, 0x75, 0x05, 0x81, 0x02,  # contact identifier (byte 3, bits 0-4)
    0x75, 0x01, 0x81, 0x03,  # bit 5 constant (the struct's "resting")
    0x09, 0x42, 0x09, 0x32, 0x25, 0x01, 0x95, 0x02, 0x81, 0x02,  # tip switch, in range (bits 6, 7)
    0x05, 0x01, 0x65, 0x11, 0x55, 0x0E, 0x35, 0x00, 0x27, 0xFF, 0xFF, 0x00, 0x00, 0x75, 0x10, 0x95, 0x01,  # cm e-2
    0x09, 0x30, 0x46, 0x86, 0x02, 0x81, 0x02,  # X (bytes 4-5), 64.6 mm
    0x09, 0x31, 0x46, 0x76, 0x05, 0x81, 0x02,  # Y (bytes 6-7), 139.8 mm
    0x05, 0x0D, 0x65, 0x00, 0x55, 0x00, 0x45, 0x00, 0x25, 0x01, 0x75, 0x01,
    0x09, 0x47, 0x81, 0x02, 0x75, 0x07, 0x81, 0x03,  # confidence (byte 8, bit 0), pad
    0x65, 0x11, 0x55, 0x0E, 0x26, 0x2C, 0x01, 0x46, 0x2C, 0x01, 0x75, 0x10, 0x95, 0x02,
    0x09, 0x48, 0x09, 0x49, 0x81, 0x02,  # width, height (bytes 9-12): 0-30 mm in 0.1 mm
    0xC0,
    0x75, 0x08, 0x95, 45, 0x81, 0x03,  # bytes 13-57 constant: the rest of Apple's struct
    0xC0])


def sized_touch_report(state: int, x: int, y: int, contact_mm: float) -> bytes:
    """touch_report plus Confidence 1 and a round contact of `contact_mm` in the bytes VDIG_DESCRIPTOR declares."""
    r = bytearray(touch_report(state, x, y))
    d = max(0, min(300, round(contact_mm * 10)))
    r[8:13] = bytes([0x01]) + struct.pack("<HH", d, d)
    return bytes(r)


async def create_vdigitizer(hid, service_id: int = VDIG_SERVICE_ID) -> int:
    """createService for VDIG_DESCRIPTOR: pymobiledevice3's create_keyboard_service payload (sniffed from macOS
    Universal Control) with the Digitizer / Touch Screen usage pair instead of the keyboard's."""
    page, usage = XpcInt64Type(0x0D), XpcInt64Type(0x04)
    vendor, product, sid = XpcInt64Type(0x05AC), XpcInt64Type(0x0251), XpcUInt64Type(service_id)
    name = "phonebridge probe digitizer"
    storage = {
        "Manufacturer": {"string": "phonebridge"}, "Product": {"string": name},
        "ProductID": {"int": product}, "VendorID": {"int": vendor},
        "PrimaryUsage": {"int": usage}, "PrimaryUsagePage": {"int": page},
        "DeviceUsagePairs": {"array": [{"dictionary": {"DeviceUsage": {"int": usage}, "DeviceUsagePage": {"int": page}}}]},
        "Transport": {"string": "USB"}, "ReportDescriptor": {"data": VDIG_DESCRIPTOR},
        "UniversalControlVirtualService": {"bool": True}, "_ServiceID": {"uint": sid},
    }
    response = await hid.service.send_receive_request({
        "featureIdentifier": "com.apple.coredevice.feature.remote.universalhidservice", "messageType": "Request",
        "payload": {"createService": {"_0": {
            "DeviceUsagePairs": [{"DeviceUsage": usage, "DeviceUsagePage": page}],
            "PrimaryUsage": XpcUInt64Type(0x04), "PrimaryUsagePage": XpcUInt64Type(0x0D), "Product": name,
            "ProductID": product, "VendorID": vendor, "_CoreDevice_codablePropertyStorage": storage, "_ServiceID": sid}}}})
    log.info("virtual digitizer createService: %s", response)
    await asyncio.sleep(KEYBOARD_ATTACH_SETTLE_SECONDS)
    return int(response.get("serviceID", service_id))


SHIFT_LEAD = 0.03  # s Shift is down before a shifted key (measured 2026-10-18: same-report shift was ignored)
PRESS_LAG = 0.008  # s: a tap read ~8 ms shorter than its plan on the iPhone 12 (measure, 2026-10-18); added back
HOME = (0x0C, 0x40)
TUNNEL_CHECK_S = 0.5  # a call re-checks tunneld's listing at most this often (a local HTTP GET, ~ms)
FG_TIMEOUT_S = 4.0  # the frontmost-app query: ~0.26 s on iOS 27.0.1, 0.4-2.3 s (one 5 s) on iOS 27.2 (live 11-08); a dead DVT channel hangs it longer
SPRINGBOARD = "com.apple.springboard"  # "frontmost" on the home and lock screens
# Always "foregroundRunning" beside the app (measured, iPhone 12 / iOS 27.0.1, 2026-10-18 18:5xZ): SpringBoard,
# the keyboard UI, and on the home screen its widget renderer. Never the frontmost app.
UI_SERVICES = (SPRINGBOARD, "com.apple.InputUI", "com.apple.chrono.WidgetRenderer", "com.apple.PosterBoard")


def frontmost(procs: list[dict]) -> str:
    """The frontmost app from DVT `runningProcesses`: the foreground application that isn't one of iOS's UI
    services, preferring an installed (App Store) app over a built-in one; SpringBoard when there is none."""
    if not any("foregroundRunning" in x for x in procs):
        raise NotImplementedError("this iOS's process list has no foregroundRunning flag")
    apps = [x for x in procs if x.get("foregroundRunning") and x.get("isApplication") and x.get("bundleIdentifier")
            and not str(x["bundleIdentifier"]).startswith(UI_SERVICES)]
    apps.sort(key=lambda x: "/containers/Bundle/Application/" not in str(x.get("realAppName", "")))
    return apps[0]["bundleIdentifier"] if apps else SPRINGBOARD


async def _touch(hid, state: int, x: int, y: int) -> None:
    """One touch report to the built-in touchscreen surface, laid out as the phone reads it (see touch_report)."""
    await hid.send_report(TOUCHSCREEN_SERVICE, touch_report(state, x, y))


@contextlib.asynccontextmanager
async def _finger(hid):
    """Touch down via at(x, y); the finger always lifts at the last point, even on error or cancellation
    (a finger left down reads as a long press)."""
    last: list[tuple[int, int]] = []

    async def at(x: int, y: int) -> None:
        last[:] = [(x, y)]
        await _touch(hid, TOUCHSCREEN_STATE_CONTACT, x, y)

    try:
        yield at
    finally:
        if last:
            await asyncio.shield(_touch(hid, TOUCHSCREEN_STATE_RELEASE, *last[0]))


class _Phone:
    def __init__(self, udid: str):
        self.udid = udid
        self.lock = asyncio.Lock()  # one gesture at a time per phone
        self.stack: contextlib.AsyncExitStack | None = None  # rsd + long-lived services
        self.session: ScreenStream | None = None  # the display stream: HID gate + frames (closed when idle)
        self.rsd = self.dvt = self.dvt_shot = self.dvt_info = self.buttons = self.kb = self.vdig = None
        self.dvt_stack: contextlib.AsyncExitStack | None = None  # the DVT channel and its services, recycled on their own
        self.rsd_addr: str | None = None  # the tunneld tunnel address the RSD was opened on
        self.tunnel_checked = 0.0  # monotonic time tunneld was last asked whether that tunnel still stands
        self.size: tuple[int, int] | None = None  # the latest frame's pixels (a stream frame may be coded larger)
        self.display: tuple[int, int] | None = None  # the device's display pixels, from its own DVT screenshot
        self.idle_timer: asyncio.TimerHandle | None = None


def _padded(f, display: tuple[int, int]) -> bool:
    """A live frame larger than the display whose pixels at the origin ARE the display's: the coded padding beyond it
    (right/bottom) checked dark. Its pixels then are the device's pixels, so a point on it maps by the display."""
    rect = display_rect(display, (f.width, f.height))
    return rect is not None and margin_dark(lambda off, n: f.bgra[off:off + n], f.width * 4, f.width, f.height, rect)


# The developer services phonebridge uses come from the personalized disk image. iOS 27.2 lists DVT without it (live
# 2026-11-08: dtservicehub listed, no com.apple.coredevice.*, `mounter list` empty, launch "No such service"), so DVT
# alone no longer tells whether the image is mounted.
DDI_SERVICES = ("com.apple.coredevice.appservice", "com.apple.coredevice.deviceinfo")


def needs_ddi(peer_info: dict) -> bool:
    have = peer_info.get("Services", {})
    return any(s not in have for s in (DvtProvider.RSD_SERVICE_NAME, *DDI_SERVICES))


class Pmd3Backend:
    def __init__(self, idle: float = 45.0, fresh: float = 0.5, eyes: str = "auto", tunnel: str = "auto",
                 pointer_surface: str = "keep"):
        if tunnel not in ("auto", "tunneld", "userspace"):
            raise ValueError(f"tunnel {tunnel!r}: auto | tunneld | userspace")
        if pointer_surface not in ("keep", "remove"):
            raise ValueError(f"pointer_surface {pointer_surface!r}: keep | remove")
        # CoreDevice registers a Mouse-usage surface (1281, "touchscreenGesture") for the host; GameController counts
        # it as a GCMouse, so every page sees (any-hover: hover) and (any-pointer: fine) and apps see a mouse attached.
        # "remove" asks dtuhidd to drop it once our session is open (UniversalHID removeService). Experiment.
        self.pointer_surface = pointer_surface
        self.recoveries: collections.deque[dict] = collections.deque(maxlen=50)  # failed calls and the reset each caused
        self._recovery_seq = 0
        self.tunnel = tunnel  # how the RSD is reached: tunneld (root LaunchDaemon, kernel utun) or in-process userspace
        self.idle = idle
        self.fresh = fresh  # s: a live frame older than this means a stalled source; screen() takes a DVT still instead
        self.eyes = ("avf" if sys.platform == "darwin" else "stream") if eyes == "auto" else eyes  # avf | stream | dvt
        self._phones: dict[str, _Phone] = {}
        self._avf: avf.AvfCapture | None = None  # one capture per Mac (the phone on USB), while looks keep coming
        self._avf_timer: asyncio.TimerHandle | None = None
        self._avf_lock = asyncio.Lock()

    # --- sessions ---------------------------------------------------------------------------
    def _phone(self, udid: str) -> _Phone:
        return self._phones.setdefault(udid, _Phone(udid))

    async def _open_rsd(self, p: _Phone):
        """The phone's RSD: through tunneld (a kernel utun) or an in-process userspace tunnel (no root). On iOS 27.2
        (2026-11-08) taps and the display stream both arrived over the userspace tunnel ("transport UserspaceUdp"),
        with the system tunneld also running; on 2026-10-18 the stream's RTP never arrived over it, and a Mac with no
        tunneld at all is unchecked. `tunnel` = auto tries tunneld first; tunneld refuses to fall back, so a test of it
        is a test of it."""
        p.stack = contextlib.AsyncExitStack()
        rsd = None
        if self.tunnel in ("auto", "tunneld"):
            with contextlib.suppress(TunneldConnectionError):
                rsd = await get_tunneld_device_by_udid(p.udid)
            if rsd is not None:
                p.stack.push_async_callback(rsd.close)
                p.rsd_addr, p.tunnel_checked = await self._tunnel_addr(p.udid), time.monotonic()
                log.warning("rsd for %s via tunneld (%s)", p.udid, p.rsd_addr)
                return rsd
            if self.tunnel == "tunneld":
                raise RuntimeError(f"tunnel=tunneld but tunneld (127.0.0.1:49151) does not list {p.udid}: on USB? tunneld running?")
        log.warning("rsd for %s via the in-process userspace tunnel", p.udid)
        return await p.stack.enter_async_context(UserspaceRsdTunnel(serial=p.udid, autopair=False))  # no Trust prompt

    async def _tunnel_addr(self, udid: str) -> str | None:
        """tunneld's current tunnel address for this phone; None when tunneld can't say."""
        try:
            entries = (await presence.tunneld_listing(presence.TUNNELD_URL, 0.5)).get(udid) or []
        except Exception:
            return None
        return entries[0].get("tunnel-address") if entries else None

    async def _check_tunnel(self, p: _Phone, force: bool = False) -> None:
        """The phone re-enumerates its USB link when a screen capture starts or stops, and tunneld replaces the
        tunnel (new address): everything we hold over the old one is dead (R1-IG attempts 2 and 3: the first
        input's process-list query timed out on it). Seen here, before a call, it costs a reconnect, not a
        timeout + retry + full reset. The capture itself (USB, not the tunnel) is kept."""
        if p.rsd is None or p.rsd_addr is None or (not force and time.monotonic() - p.tunnel_checked < TUNNEL_CHECK_S):
            return
        p.tunnel_checked = time.monotonic()
        now = await self._tunnel_addr(p.udid)
        if now and now != p.rsd_addr:
            log.warning("tunnel for %s was recreated (%s -> %s): reconnecting, capture kept", p.udid, p.rsd_addr, now)
            await self._reset(p, keep_eyes=True)

    async def _rewarm_after_capture(self, before: dict[str, str | None]) -> None:
        """After the capture starts, wait for tunneld's new tunnel (~1 s), then reconnect and warm the DVT channel
        with one process-list query, so the session's first input finds both warm."""
        for udid, old in before.items():
            p = self._phone(udid)
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                now = await self._tunnel_addr(udid)
                if now and now != old:
                    break
                await asyncio.sleep(0.25)
            else:
                continue  # no new tunnel: nothing went stale
            async with p.lock:
                with contextlib.suppress(Exception):
                    await self._check_tunnel(p, force=True)
                    dvt = await self._dvt(p)
                    if p.dvt_info is None:
                        p.dvt_info = await p.dvt_stack.enter_async_context(DeviceInfo(dvt))
                    await asyncio.wait_for(p.dvt_info.proclist(), 5)  # the warm-up may take its time; inputs don't
                    await self._session(p)  # the touch session too: the first swipe paid ~0.7 s opening it (attempt 4)
                    log.warning("rewarmed %s after the capture started (tunnel %s)", udid, p.rsd_addr)

    async def _rsd(self, p: _Phone):
        if p.rsd is None:
            rsd = await self._open_rsd(p)
            if needs_ddi(rsd.peer_info):
                # Developer services come from the personalized disk image: mount it (downloads once),
                # then reconnect so the RSD handshake lists them.
                with contextlib.suppress(AlreadyMountedError):
                    await auto_mount(rsd)
                await p.stack.aclose()
                rsd = await self._open_rsd(p)
            p.rsd = rsd
        return p.rsd

    async def _service(self, p: _Phone, attr: str, cls):
        if getattr(p, attr) is None:
            rsd = await self._rsd(p)
            setattr(p, attr, await p.stack.enter_async_context(cls(rsd)))
        return getattr(p, attr)

    async def _session(self, p: _Phone) -> ScreenStream:
        """The phone's display stream, opened on the first input or fast look and kept while they keep coming."""
        if p.session is None:
            rsd = await self._rsd(p)
            if p.display is None:  # the device's display size before the stream starts (one screenshot, 0.3-0.9 s, once):
                try:  # live 11-08 a frame pushed before it was known went out uncropped
                    await self._dvt_png(p)
                except Exception as e:  # noqa: BLE001 - frames stay uncropped and labelled; input refuses them
                    log.warning("the device's display size is unknown (%s): stream frames uncropped, input refused", e)
            session = ScreenStream(rsd, display=p.display)
            await session.__aenter__()
            p.session = session
            if self.pointer_surface == "remove":
                await self._remove_pointer_surface(session.hid)
        if p.idle_timer:
            p.idle_timer.cancel()
        p.idle_timer = asyncio.get_running_loop().call_later(
            self.idle, lambda: asyncio.ensure_future(self._close_session(p)))
        return p.session

    def keep_warm(self, udid: str) -> None:
        """Restart the idle timers of whatever is open for this phone (the touch session, the capture) without opening
        anything: the bridge calls it while a task runs, so the first input after a long dwell finds both warm (R1-IG
        attempt 7: a 70 s hold let both idle out; the look restarted the capture, the tunnel was recreated, the swipe
        waited 1.09 s). A stopped capture (park, USB loss, the task's end) stays stopped."""
        p = self._phones.get(udid)
        loop = asyncio.get_running_loop()
        if p is not None and p.session is not None and p.idle_timer is not None:
            p.idle_timer.cancel()
            p.idle_timer = loop.call_later(self.idle, lambda: asyncio.ensure_future(self._close_session(p)))
        if self._avf is not None and self._avf_timer is not None:
            self._avf_timer.cancel()
            self._avf_timer = loop.call_later(self.idle, self._stop_avf)

    async def _hid(self, p: _Phone):
        return (await self._session(p)).hid

    async def _remove_pointer_surface(self, hid) -> bool:
        """UniversalHID removeService for the Mouse-usage surface; logs whether it is still listed afterwards."""
        try:
            await hid.service.send_request({"featureIdentifier": "com.apple.coredevice.feature.remote.universalhidservice",
                                            "messageType": "Request",
                                            "payload": {"removeService": {"_0": XpcUInt64Type(POINTER_SURFACE)}}})
            listed = await hid.list_connected_services()
            ids = [int(s.get("_ServiceID", -1)) for s in (listed.get("connectedServices") or [])]
            gone = POINTER_SURFACE not in ids
            log.warning("pointer surface %d removal: %s (surfaces now %s)", POINTER_SURFACE, "gone" if gone else "STILL LISTED", ids)
            return gone
        except Exception as e:
            log.warning("pointer surface %d removal failed: %s", POINTER_SURFACE, e)
            return False

    # --- the eyes: AVFoundation capture first, the stream's frames second ---------------------------------
    async def _avf_frames(self) -> Frames:
        async with self._avf_lock:
            if self._avf is None:
                before = {u: await self._tunnel_addr(u) for u in self._phones} if self.tunnel in ("auto", "tunneld") else {}
                cap = avf.AvfCapture(Frames())
                await cap.start()
                self._avf = cap
                if before:
                    self._rewarm_task = asyncio.ensure_future(self._rewarm_after_capture(before))
            if self._avf_timer:
                self._avf_timer.cancel()
            self._avf_timer = asyncio.get_running_loop().call_later(self.idle, self._stop_avf)
            return self._avf.frames

    def _stop_avf(self) -> None:
        self.stop_eyes("idle")

    def stop_eyes(self, reason: str = "requested") -> None:
        """End the screen capture now (idle, park, USB loss, shutdown): the phone's isCaptured and the Mac's
        capture indicator go off with it. Idempotent, never raises, always logged."""
        if self._avf_timer:
            self._avf_timer.cancel()
            self._avf_timer = None
        if self._avf is not None:
            cap, self._avf = self._avf, None
            try:
                cap.stop()
            except Exception as e:
                log.warning("avf stop failed (%s): %s", reason, e)
            log.warning("eyes: capture stopped (%s)", reason)

    def _live(self, p: _Phone):
        """The newest live frame from whichever source runs: the capture, else the stream session."""
        if self._avf is not None and (f := self._avf.frames.latest()):
            return f
        if p.session is not None:
            return p.session.frames.latest()
        return None

    async def _keyboard(self, p: _Phone):
        hid = await self._hid(p)
        if p.kb is None:
            p.kb = await hid.create_keyboard_service()
        return hid, p.kb

    async def _close_session(self, p: _Phone) -> None:
        async with p.lock:
            await self._drop_session(p)

    async def _drop_session(self, p: _Phone) -> None:
        if p.session:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(p.session.__aexit__(None, None, None), 10)
        p.session = p.kb = p.vdig = None

    async def _reset(self, p: _Phone, keep_eyes: bool = False) -> None:
        if p.idle_timer:
            p.idle_timer.cancel()
        # no capture outlives its phone: stop it when the phone left USB. A failed call with the phone still
        # attached keeps it: restarting the capture would re-enumerate USB and recreate the tunnel once more
        if not keep_eyes and not await self._still_attached(p.udid):
            self.stop_eyes("reset: the phone left USB")
        await self._reset_dvt(p)
        await self._drop_session(p)
        if p.stack:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(p.stack.aclose(), 5)
        p.stack = p.rsd = p.dvt = p.dvt_shot = p.dvt_info = p.buttons = None
        p.rsd_addr = None

    async def _run(self, udid: str, fn, retry: bool, narrow=None):
        """Run fn(phone) under the phone's lock; on failure drop the session (the phone may have been
        unplugged or locked) and, for side-effect-free calls, retry once on a fresh one. `narrow` resets only the
        part that failed before the retry (the DVT channel); a second failure resets everything. Every recovery is
        kept in `recoveries` for the call log."""
        p = self._phone(udid)
        async with p.lock:
            await self._check_tunnel(p)
            t = time.monotonic()
            try:
                return await fn(p)
            except Exception as e:
                scope = "dvt" if (retry and narrow) else "all"
                self._recovered(fn, e, t, scope)
                await (narrow(p) if scope == "dvt" else self._reset(p))
                if not retry:
                    raise
            t = time.monotonic()
            try:
                return await fn(p)
            except Exception as e:
                self._recovered(fn, e, t, "all")
                await self._reset(p)  # the next call starts clean too, not on what just failed
                raise

    def _recovered(self, fn, e: BaseException, t: float, scope: str) -> None:
        self._recovery_seq += 1
        self.recoveries.append({"n": self._recovery_seq, "call": getattr(fn, "__name__", "?"), "error": f"{type(e).__name__}: {e}"[:120],
                                "after_ms": round((time.monotonic() - t) * 1000), "reset": scope})
        log.warning("backend %s failed after %.0f ms (%s): reset %s", getattr(fn, "__name__", "?"),
                    (time.monotonic() - t) * 1000, type(e).__name__, scope)

    async def close(self) -> None:
        self.stop_eyes("shutdown")
        for p in self._phones.values():
            await self._reset(p)

    # --- eyes -------------------------------------------------------------------------------
    async def devices(self) -> list[dict]:
        out = []
        for d in await usbmux.list_devices():
            if not d.is_usb:
                continue
            try:
                async with await create_using_usbmux(d.serial, autopair=False, connection_type=d.connection_type) as ld:
                    i = ld.short_info
                    dev = {"udid": d.serial, "name": i.get("DeviceName"), "ios": i.get("ProductVersion"),
                           "product": i.get("ProductType"), "model": ld.display_name, "paired": ld.paired}
                    if ld.paired:
                        dev["developer_mode"] = await ld.get_developer_mode_status()
                    else:
                        dev["hint"] = "unlock the phone and tap Trust (phonebridge pair <device>)"
                    out.append(dev)
            except Exception as e:  # e.g. locked after boot: still listed, with the reason
                out.append({"udid": d.serial, "error": f"{type(e).__name__}: {e}"})
        return out

    async def pair(self, udid: str, timeout: float = 120) -> None:
        """Ask the phone to trust this Mac (shows "Trust This Computer?"; the operator taps Trust)."""
        async with await create_using_usbmux(udid, autopair=False) as ld:
            if not ld.paired:
                await ld.pair(timeout=timeout)

    async def _capture(self, p: _Phone) -> bytes:
        """A fresh frame of the running stream as PNG (~25–60 ms to encode, no round trip); otherwise a PNG
        from the DVT screenshot instrument: 0.3–0.9 s on iOS 27 (p50 364 ms, p95 847 ms over 1,487 calls).
        CoreDevice screen capture measured 0.8 s but stalled 15–40 s, wedged on "a screenshot capture is
        already in progress" (code 25003) and dropped its connection, so it isn't used."""
        if (f := self._live(p)) and f.age_ms() <= self.fresh * 1000:
            p.size = (f.width, f.height)
            return await asyncio.to_thread(f.png)
        return await self._dvt_png(p)

    async def _dvt_png(self, p: _Phone) -> bytes:
        """The DVT screenshot: the device's own pixels, so its size is the display's (refreshed on every still, e.g. a
        rotation), handed to the stream's decoder as its crop."""
        if p.dvt_shot is None:
            dvt = await self._dvt(p)  # opens p.stack: evaluate before p.stack.enter_async_context
            p.dvt_shot = await p.dvt_stack.enter_async_context(Screenshot(dvt))
        png = await asyncio.wait_for(p.dvt_shot.get_screenshot(), 10)
        p.size = p.display = png_size(png)
        if (rcv := getattr(p.session, "receiver", None)) is not None and rcv.display != p.display:
            rcv.set_display(p.display)
        return png

    async def _dvt(self, p: _Phone):
        if p.dvt is None:
            rsd = await self._rsd(p)  # opens p.stack first
            p.dvt_stack = contextlib.AsyncExitStack()
            p.dvt = await p.dvt_stack.enter_async_context(DvtProvider(rsd))
        return p.dvt

    async def _reset_dvt(self, p: _Phone) -> None:
        """Recycle only the DVT channel (process list, stills): a hung or dropped DVT call says nothing about the touch
        session, the capture or the tunnel, and tearing those down cost R1-IG attempt 2 ~2.5 s before a swipe and
        a cold capture after it."""
        stack, p.dvt_stack = p.dvt_stack, None
        p.dvt = p.dvt_shot = p.dvt_info = None
        if stack:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stack.aclose(), 3)

    async def screen(self, udid: str) -> bytes:
        return await self._run(udid, self._capture, retry=True, narrow=self._reset_dvt)

    async def frames(self, udid: str) -> Frames:
        """The live frames: the AVFoundation capture (opened on demand; the phone shows it as a screen capture,
        like input's session does), else the stream session's frames."""
        if self.eyes == "dvt":  # PHONEBRIDGE_EYES=dvt: no live source at all; look/frame fall back to stills
            raise RuntimeError("eyes=dvt: live frames are switched off on this host")
        if self.eyes == "avf":
            try:
                return await self._avf_frames()
            except Exception as e:
                log.warning("avf capture unavailable (%s: %s): falling back to the stream's frames", type(e).__name__, e)

        async def frames(p):
            return (await self._session(p)).frames
        return await self._run(udid, frames, retry=True)

    async def foreground(self, udid: str) -> str:
        """Bundle id of the frontmost app (com.apple.springboard on the home and lock screens), from DVT's
        process list: see frontmost()."""
        async def fg(p):
            if p.dvt_info is None:
                dvt = await self._dvt(p)  # opens p.stack first
                p.dvt_info = await p.dvt_stack.enter_async_context(DeviceInfo(dvt))
            return frontmost(await asyncio.wait_for(p.dvt_info.proclist(), FG_TIMEOUT_S))
        return await self._run(udid, fg, retry=True, narrow=self._reset_dvt)

    async def _still_attached(self, udid: str) -> bool:
        try:
            return udid in await asyncio.wait_for(self.attached(), 2)
        except Exception:
            return False  # can't tell: treat as gone (the capture stops, as before)

    async def attached(self) -> list[str]:
        """UDIDs usbmux lists on USB right now (cheap: no lockdown session)."""
        return [d.serial for d in await usbmux.list_devices() if d.is_usb]

    async def battery(self, udid: str) -> dict:
        """{temperature °C, level %, charging, external} from the diagnostics relay's IOPMPowerSource."""
        async def battery(p):
            async with DiagnosticsService(await self._rsd(p)) as diag:
                b = await asyncio.wait_for(diag.get_battery(), 10) or {}
            temp = b.get("Temperature")
            return {"temperature": None if temp is None else temp / 100 if temp > 150 else float(temp),  # centi-°C
                    "level": b.get("CurrentCapacity"), "charging": b.get("IsCharging"),
                    "external": b.get("ExternalConnected")}
        return await self._run(udid, battery, retry=True)

    async def storage(self, udid: str) -> dict:
        """{free_gb, total_gb} of the phone's data volume (lockdown's com.apple.disk_usage), for preflight."""
        async with await create_using_usbmux(udid, autopair=False) as ld:
            du = await ld.get_value(domain="com.apple.disk_usage") or {}
        return {"free_gb": du.get("AmountDataAvailable", 0) / 2**30, "total_gb": du.get("TotalDataCapacity", 0) / 2**30}

    async def app_version(self, udid: str, bundle_id: str) -> str | None:
        """The installed version of `bundle_id` (CFBundleShortVersionString), None if it isn't installed."""
        async with await create_using_usbmux(udid, autopair=False) as ld:
            async with InstallationProxyService(lockdown=ld) as proxy:
                apps = await proxy.get_apps(bundle_identifiers=[bundle_id])
        return (apps.get(bundle_id) or {}).get("CFBundleShortVersionString")

    # --- L3 resets (host/recover.py) --------------------------------------------------------
    async def reset(self, udid: str) -> None:
        """Drop the RSD, DVT channels and touch session; the next call opens fresh ones."""
        p = self._phone(udid)
        async with p.lock:
            await self._reset(p)

    async def remount(self, udid: str) -> None:
        """Reset, then reopen the RSD: the developer disk image is mounted again if its services are gone."""
        p = self._phone(udid)
        async with p.lock:
            await self._reset(p)
            await self._rsd(p)

    async def lock_state(self, udid: str) -> dict:
        async def ls(p):
            try:
                async with DeviceInfoService(await self._rsd(p)) as info:
                    return await info.get_lockstate()
            except CoreDeviceError:  # "getlockstate is not implemented" before iOS 27
                async with await create_using_usbmux(udid, autopair=False) as ld:
                    protected = await ld.get_value(key="PasswordProtected")
                png = await self._capture(p)
                dark = Image.open(io.BytesIO(png)).convert("L").getextrema()[1] < 8  # display asleep => locked
                return {"locked": True if dark else None, "display": "off" if dark else "on", "passcode_set": protected,
                        "note": "this iOS doesn't report lock state; with the display on, read the screen"}
        return await self._run(udid, ls, retry=True)

    # --- hands ------------------------------------------------------------------------------
    async def _norm(self, p: _Phone, x: float, y: float) -> tuple[int, int]:
        """Screen pixels to HID units against the DEVICE's display (its own screenshot's size), never a frame's coded
        size: live 11-08 stream frames were coded 1184×2576 around a 1170×2532 display, and taps placed on them landed
        up to ~43 px short. A live frame of any other geometry, or an unknown display, refuses: zero input sent."""
        if p.display is None:
            try:
                await self._dvt_png(p)
            except Exception as e:
                raise RuntimeError(f"the device's display size is unknown ({e}): zero input sent") from e
        w, h = p.display
        if (f := self._live(p)) is not None and (f.width, f.height) != (w, h) and not _padded(f, (w, h)):
            raise RuntimeError(f"the screen frame is {f.width}x{f.height} but the device's display is {w}x{h} "
                               "(unsupported geometry): zero input sent")
        clamp = lambda v: max(0, min(HID_MAX, round(v)))
        return clamp(x * HID_MAX / w), clamp(y * HID_MAX / h)

    async def tap(self, udid: str, x: float, y: float, press: float) -> None:
        async def tap(p):
            nx, ny = await self._norm(p, x, y)
            hid = await self._hid(p)
            async with _finger(hid) as at:
                await at(nx, ny)
                await asyncio.sleep(press + PRESS_LAG)
        await self._run(udid, tap, retry=False)

    async def _indigo(self, p: _Phone, kind: int, x: float, y: float) -> None:
        """One IndigoDigitizerEvent (kind 0 start, 1 position, 2 end) at a screenshot-pixel point on the main screen
        (layout per jkcoxson/idevice hid.rs)."""
        indigo = await self._service(p, "buttons", IndigoHIDService)
        await indigo.service.send_request({
            "messageType": "IndigoDigitizerEvent",
            "payload": {"pointOne": {"x": float(x), "y": float(y)}, "eventType": XpcUInt64Type(kind),
                        "edge": XpcUInt64Type(0), "target": XpcUInt64Type(0)},
            "featureIdentifier": "com.apple.coredevice.feature.remote.hid.digitizer"})

    async def tap_indigo(self, udid: str, x: float, y: float, press: float) -> None:
        """EXPERIMENT (touch-size probe, 2026-10-19): the same tap through CoreDevice's other digitizer path, the
        `IndigoDigitizerEvent` on com.apple.coredevice.hid.indigo. The 58-byte report path gives every touch a
        4220 pt radius; this tests whether iOS builds Indigo touches differently. Held inside the touch session so
        the HID auth gate is open."""
        async def tap(p):
            await self._hid(p)
            await self._indigo(p, 0, x, y)
            try:
                await asyncio.sleep(press + PRESS_LAG)
            finally:
                await asyncio.shield(self._indigo(p, 2, x, y))
        await self._run(udid, tap, retry=False)

    async def tap_vdigitizer(self, udid: str, x: float, y: float, press: float, contact_mm: float) -> None:
        """EXPERIMENT (touch-radius evidence): the same tap through our own virtual digitizer (VDIG_DESCRIPTOR), with a
        `contact_mm` contact, inside the touch session (the HID auth gate)."""
        async def tap(p):
            nx, ny = await self._norm(p, x, y)
            hid = await self._hid(p)
            if p.vdig is None:
                p.vdig = await create_vdigitizer(hid)
            await hid.send_report(p.vdig, sized_touch_report(TOUCHSCREEN_STATE_CONTACT, nx, ny, contact_mm))
            try:
                await asyncio.sleep(press + PRESS_LAG)
            finally:
                await asyncio.shield(hid.send_report(p.vdig, sized_touch_report(TOUCHSCREEN_STATE_RELEASE, nx, ny, contact_mm)))
        await self._run(udid, tap, retry=False)

    async def ax_probe(self, udid: str, ops: list[str], max_nodes: int = 120) -> dict:
        """EXPERIMENT (accessibility-tree spike): the accessibility audit daemon's view of the foreground app
        (phonebridge/axprobe.py has the ops), over this phone's RSD. App monitoring is switched off again after."""
        from pymobiledevice3.services.accessibilityaudit import AccessibilityAudit
        from ..axprobe import AxProbe

        async def probe(p):
            async with AccessibilityAudit(await self._rsd(p)) as audit:
                try:
                    return await AxProbe(audit).run(ops, max_nodes)
                finally:
                    with contextlib.suppress(Exception):
                        await audit.set_app_monitoring_enabled(False)
        return await self._run(udid, probe, retry=False)

    async def swipe_indigo(self, udid: str, path: list[PathPoint]) -> None:
        """EXPERIMENT: the swipe path as Indigo digitizer events (start, positions on the path's own clock, end)."""
        async def swipe(p):
            await self._hid(p)
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            last = path[0]
            try:
                for i, (x, y, t) in enumerate(path):
                    if (wait := t0 + t - loop.time()) > 0:
                        await asyncio.sleep(wait)
                    await self._indigo(p, 0 if i == 0 else 1, x, y)
                    last = (x, y, t)
            finally:
                await asyncio.shield(self._indigo(p, 2, last[0], last[1]))
        await self._run(udid, swipe, retry=False)

    async def swipe(self, udid: str, path: list[PathPoint]) -> None:
        """Play the path on its own clock: each point goes out at its time since touch-down."""
        async def swipe(p):
            pts = [(*await self._norm(p, x, y), t) for x, y, t in path]
            hid = await self._hid(p)
            loop = asyncio.get_running_loop()
            async with _finger(hid) as at:
                t0 = loop.time()
                for nx, ny, t in pts:
                    if (wait := t0 + t - loop.time()) > 0:
                        await asyncio.sleep(wait)
                    await at(nx, ny)
        await self._run(udid, swipe, retry=False)

    async def type(self, udid: str, keys: list[Key]) -> None:
        if bad := sorted({c for c, _, _ in keys if c not in KEYS}):
            raise ValueError(f"type() is ASCII-only; use paste() for {''.join(bad)!r}")

        async def type_(p):
            hid, kb = await self._keyboard(p)
            for ch, hold, gap in keys:
                usage, shift = KEYS[ch]
                await asyncio.sleep(gap)
                if shift:  # Shift goes down first on its own: in one report iOS reads the letter before it
                    await hid.send_keyboard(kb, (KEY_LEFT_SHIFT,))
                    await asyncio.sleep(SHIFT_LEAD)
                await hid.send_keyboard(kb, (KEY_LEFT_SHIFT, usage) if shift else (usage,))
                await asyncio.sleep(hold)
                await hid.send_keyboard(kb, ())
        await self._run(udid, type_, retry=False)

    async def paste(self, udid: str, text: str) -> None:
        async def paste(p):
            async with PasteboardService(await self._rsd(p)) as pb:
                await pb.set_text(text)
            hid, kb = await self._keyboard(p)
            await hid.send_keyboard(kb, (KEY_LEFT_GUI,))
            await asyncio.sleep(0.04)
            await hid.send_keyboard(kb, (KEY_LEFT_GUI, KEY_V))
            await asyncio.sleep(0.06)
            await hid.send_keyboard(kb, ())
        await self._run(udid, paste, retry=False)

    async def home(self, udid: str) -> None:
        """One Home press. The button connection is cached and the phone may have closed it (2026-10-19 07:09Z:
        ConnectionTerminatedError after 90 s idle), so a failure retries once on a fresh connection: the whole
        press if the down never reached the phone, only the release if it did (a second press after a delivered
        down could read as a long press: Siri)."""
        down_sent = False

        async def home(p):
            nonlocal down_sent
            b = await self._service(p, "buttons", IndigoHIDService)
            if not down_sent:
                await b.send_button(*HOME, HID_BUTTON_STATE_DOWN)
                down_sent = True
                await asyncio.sleep(0.05)
            await b.send_button(*HOME, HID_BUTTON_STATE_UP)
            await asyncio.sleep(0.1)
        await self._run(udid, home, retry=True)

    async def launch(self, udid: str, bundle_id: str, url: str | None = None) -> None:
        """Bring an app to the front; with `url`, hand it that URL (devicectl's --payload-url)."""
        async def launch(p):
            options = {"arguments": [], "environmentVariables": {}, "standardIOUsesPseudoterminals": True,
                       "startStopped": False, "terminateExisting": False, "user": {"shortName": "mobile"},
                       "platformSpecificOptions": plistlib.dumps({})}
            if url:
                options["payloadURL"] = {"relative": url}
            async with AppServiceService(await self._rsd(p)) as apps:
                await apps.invoke("com.apple.coredevice.feature.launchapplication", {
                    "applicationSpecifier": {"bundleIdentifier": {"_0": bundle_id}},
                    "options": options, "standardIOIdentifiers": {}})
        await self._run(udid, launch, retry=False)
