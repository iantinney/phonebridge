"""The phone's live screen as the eyes (PLAN A2 / E1): the CoreDevice display stream decoded on the Mac.

backboardd only routes our HID reports while a media stream is running (pymobiledevice3's `touch_session`
opens one and throws its frames away). `ScreenStream` opens that same stream once per phone and keeps the
frames: it is the HID auth gate *and* `frames.latest()`, so the eyes and the hands never fight over the
one stream the phone allows. RTP/HEVC comes in over the RSD tunnel (userspace or tunneld), pymobiledevice3's
depacketizer reassembles the access units, and VideoToolbox (macOS, ctypes, no extra dependency) or PyAV
(elsewhere) turns them into BGRA. Measured by pymobiledevice3 on Apple silicon: ~1 ms decode per frame at
60 fps; the phone's encoder sends ~6 Mbit/s.

Keeping the device's encoder honest: a Receiver Report every second (the phone stalls the encoder after
~25 s of RTCP silence), and a Picture Loss Indication, rate-limited, whenever an RTP gap or a decode error
means the next deltas would reference a frame we never had; those deltas are held until the fresh IDR, so a
corrupted picture is never published (VideoToolbox would silently render a mosaic). The frames carry
`seq`, a monotonic `t` and a wall-clock `wall`, so a look can say how old its pixels are.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import io
import logging
import struct
import sys
import threading
import time
from typing import Callable, NamedTuple

from PIL import Image
from pymobiledevice3.remote.core_device.screen_stream import depacketize_hevc

log = logging.getLogger(__name__)

NAL_VPS, NAL_SPS, NAL_PPS = 32, 33, 34
KEY_NALS = (19, 20, 21)  # IDR_W_RADL, IDR_N_LP, CRA
RR_INTERVAL = 1.0  # s; the device's RTCPTimeoutEnabled stalls the encoder after ~25 s of silence
PLI_MIN_GAP = 0.7  # s; a PLI storm corrupts and finally wedges the phone's encoder (pymobiledevice3's finding)
RCVBUF = (8 << 20, 4 << 20, 1 << 20)
SETTLE_S = 0.5  # backboardd re-matches the HID surfaces after the stream starts (pymobiledevice3: TOUCH_AUTH_SETTLE_SECONDS)


class StreamFrame(NamedTuple):
    seq: int
    t: float  # time.monotonic() when decoded
    wall: float  # time.time() when decoded
    width: int
    height: int
    bgra: bytes
    cropped: tuple[int, int, int, int] | None = None  # the capture's clean aperture applied (x, y, w, h); None: as delivered

    @property
    def rgba(self) -> Image.Image:
        return Image.frombuffer("RGBA", (self.width, self.height), self.bgra, "raw", "BGRA", 0, 1)

    @property
    def image(self) -> Image.Image:
        return self.rgba.convert("RGB")

    @property
    def gray(self) -> Image.Image:
        return Image.frombuffer("RGBA", (self.width, self.height), self.bgra, "raw", "BGRA", 0, 1).convert("L")

    def png(self, compress_level: int = 1) -> bytes:
        out = io.BytesIO()
        self.image.save(out, "PNG", compress_level=compress_level)
        return out.getvalue()

    def age_ms(self) -> float:
        return (time.monotonic() - self.t) * 1000


RECENT = 8  # frames kept for settle's content window: ~130 ms at 60 fps, ~12 MB each at 1170x2532


class Frames:
    """The latest decoded frame, pushed from the decoder's thread, awaited from the event loop."""

    def __init__(self):
        self._latest: StreamFrame | None = None
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None
        self.stats = {"frames": 0, "aus": 0, "dropped_aus": 0, "decode_errors": 0, "pli": 0, "rr": 0, "fps": 0.0}
        self._fps_window: list[float] = []
        self.alive: Callable[[], bool] | None = None  # the source's own health, when it can tell (the AVF capture)
        self._recent: collections.deque[StreamFrame] = collections.deque(maxlen=RECENT)

    def latest(self) -> StreamFrame | None:
        return self._latest

    def recent(self) -> list[StreamFrame]:
        """The last few frames, oldest first (settle compares a frame with one ~100 ms older without waiting)."""
        with self._lock:
            return list(self._recent)

    def push(self, bgra: bytes, width: int, height: int, cropped: tuple[int, int, int, int] | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            seq = (self._latest.seq if self._latest else 0) + 1
            self._latest = StreamFrame(seq, now, time.time(), width, height, bgra, cropped)
            self._recent.append(self._latest)
            self.stats["frames"] = seq
            self._fps_window = [t for t in self._fps_window if now - t < 1.0] + [now]
            self.stats["fps"] = float(len(self._fps_window))
            loop, event = self._loop, self._event
        if loop is not None and event is not None:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # loop closed: nobody is waiting
                pass

    async def wait(self, after: int = -1, timeout: float = 1.0) -> StreamFrame:
        """The first frame with seq > `after` (the latest one at once if it qualifies); TimeoutError otherwise."""
        if self._event is None:
            self._loop, self._event = asyncio.get_running_loop(), asyncio.Event()
        deadline = time.monotonic() + timeout
        while True:
            if (f := self._latest) is not None and f.seq > after:
                return f
            self._event.clear()
            left = deadline - time.monotonic()
            if left <= 0:
                raise asyncio.TimeoutError(f"no frame newer than #{after} within {timeout} s (stream counters: {self.stats})")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._event.wait(), left)


def decoder_class(choice: str = "auto"):
    """VideoToolbox on macOS (ctypes), PyAV elsewhere: both expose feed()/close()/width/height and call
    on_frame(bgra) from a worker thread."""
    choice = "vt" if choice == "auto" and sys.platform == "darwin" else ("av" if choice == "auto" else choice)
    if choice == "vt":
        return _vt_decoder()
    from pymobiledevice3.remote.core_device.hevc_av import HevcToBgraTranscoder
    return HevcToBgraTranscoder


Rect = tuple[int, int, int, int]


def clean_rect(rect, width: int, height: int) -> Rect | None:
    """The clean aperture (x, y, w, h) in whole pixels, origin top-left, as CMVideoFormatDescriptionGetCleanAperture(fmt,
    originIsAtTopLeft=True) gives it for the stream's own parameter sets (HEVC's conformance window); None when it is
    missing or doesn't fit a width×height buffer (the frame then goes on uncropped, labelled so). Live 11-08 (iOS 27.2,
    the display stream over the userspace tunnel) the decoded buffers were coded 1184×2576, 1170×2532 rounded up to
    16-px blocks, with the screen at the top left and black beyond it: every fraction-of-frame region moved and a
    template fell under its threshold (Explore unrecognised). Never the clean-aperture attachment's offsets: those place
    the aperture's CENTRE, not its origin."""
    try:
        x, y, w, h = (round(float(v)) for v in rect)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        return None
    return x, y, w, h


def crop_rows(read: Callable[[int, int], bytes], bytes_per_row: int, width: int, height: int,
              rect: Rect | None) -> tuple[bytes, int, int]:
    """One copy of a 32BGRA buffer's pixels inside `rect` (the whole buffer when None), row by row at the buffer's
    stride (bytesPerRow may exceed width×4); `read(offset, n)` reads the locked buffer. Returns (bgra, w, h)."""
    x, y, w, h = rect or (0, 0, width, height)
    if (x, y, w, h) == (0, 0, width, height) and bytes_per_row == width * 4:
        return read(0, height * bytes_per_row), width, height
    return b"".join(read((y + r) * bytes_per_row + x * 4, w * 4) for r in range(h)), w, h


MARGIN_DARK = 40  # a coded frame's padding beyond the display is black (live 11-08: max 25 on the right, 5 below)


def display_rect(display: tuple[int, int] | None, coded: tuple[int, int]) -> Rect | None:
    """The device's display (width, height), reported by the device itself (a DVT screenshot's size), as the crop of a
    coded frame at its origin: block coding pads at the right and bottom only. None when unknown or it doesn't fit."""
    if not display or display[0] <= 0 or display[1] <= 0 or display[0] > coded[0] or display[1] > coded[1]:
        return None
    return 0, 0, display[0], display[1]


def margin_dark(read: Callable[[int, int], bytes], bytes_per_row: int, width: int, height: int, rect: Rect) -> bool:
    """A conservative check that the pixels outside a display crop at the origin are padding: 16 samples down the
    right margin and 16 along the bottom margin, every channel dark. It validates the known bounds; it never finds them."""
    x, y, w, h = rect
    pts = [(x + w + (width - x - w) // 2, (height - 1) * i // 15) for i in range(16)] if x + w < width else []
    pts += [((width - 1) * i // 15, y + h + (height - y - h) // 2) for i in range(16)] if y + h < height else []
    return all(max(read(py * bytes_per_row + px * 4, 3)) < MARGIN_DARK for px, py in pts)


def _vt_decoder():
    """pymobiledevice3's VideoToolbox transcoder, made prompt: it decodes asynchronously and only drains outputs
    after the next feed, so every frame came out one frame late (measured on the M5 Pro with a paced synthetic
    stream: 70 ms behind its last packet at 30 fps, 30 ms at 60 fps, i.e. two frame intervals). Waiting for the
    session's asynchronous frames right after each feed hands the frame over as soon as it is decoded."""
    import ctypes

    from pymobiledevice3.remote.core_device import vt_jpeg

    vt = vt_jpeg.vt
    if not hasattr(vt.VTDecompressionSessionWaitForAsynchronousFrames, "_prompt"):
        vt.VTDecompressionSessionWaitForAsynchronousFrames.restype = ctypes.c_int32
        vt.VTDecompressionSessionWaitForAsynchronousFrames.argtypes = [ctypes.c_void_p]
        vt.VTDecompressionSessionWaitForAsynchronousFrames._prompt = True

    class _CGRect(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("w", ctypes.c_double), ("h", ctypes.c_double)]

    aperture = vt_jpeg.cm.CMVideoFormatDescriptionGetCleanAperture
    aperture.restype, aperture.argtypes = _CGRect, [ctypes.c_void_p, ctypes.c_bool]

    class PromptVT(vt_jpeg.HevcToBgraTranscoder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.coded = (self.width, self.height)  # the bitstream's coded size (16-px blocks)
            r = aperture(self._dec._fmt, True)  # once per parameter sets: the clean aperture, origin top-left
            rect = clean_rect((r.x, r.y, r.w, r.h), self.width, self.height)
            # a real conformance window wins; live 11-08 iOS 27.2 sent none (aperture = coded 1184×2576), so the
            # Receiver sets the device's display instead (crop_from = "display", margins checked per buffer)
            self.crop, self.crop_from = (rect, "bitstream") if rect and rect[2:] != self.coded else (None, None)
            if self.crop:
                self.width, self.height = self.crop[2], self.crop[3]

        def _emit(self, image_buf: int) -> None:
            """One copy out of the pixel buffer. The parent also rewrote every alpha byte (an 11.8 MB strided
            assignment, ~20 ms per frame); nothing here reads alpha (frames become RGB or L)."""
            cv = vt_jpeg.cv
            if cv.CVPixelBufferLockBaseAddress(image_buf, vt_jpeg.kCVPixelBufferLock_ReadOnly) != 0:
                return
            try:
                w, h = int(cv.CVPixelBufferGetWidth(image_buf)), int(cv.CVPixelBufferGetHeight(image_buf))
                bpr, base = int(cv.CVPixelBufferGetBytesPerRow(image_buf)), cv.CVPixelBufferGetBaseAddress(image_buf)
                if not base or w <= 0 or h <= 0:
                    return
                read = lambda off, n: ctypes.string_at(base + off, n)  # noqa: E731
                rect, source = self.crop, self.crop_from  # in this order (Receiver._apply_display writes the reverse)
                if rect and (rect[0] + rect[2] > w or rect[1] + rect[3] > h):
                    rect = None  # this buffer is smaller than the crop: never read past it (uncropped, labelled)
                if rect and source != "bitstream" and not margin_dark(read, bpr, w, h, rect):
                    rect = None  # the margin isn't padding: uncropped and labelled (input refuses it)
                data, cw, ch = crop_rows(read, bpr, w, h, rect)
            finally:
                cv.CVPixelBufferUnlockBaseAddress(image_buf, vt_jpeg.kCVPixelBufferLock_ReadOnly)
            with contextlib.suppress(Exception):
                self._on_frame(data, cw, ch, rect)

        def _run(self) -> None:
            while not self._stop.is_set():
                try:
                    item = self._inq.get(timeout=0.05)
                except vt_jpeg.queue.Empty:
                    continue
                if item is None:
                    break
                try:
                    self._dec.feed_annexb(item)
                    vt.VTDecompressionSessionWaitForAsynchronousFrames(self._dec._session)
                except Exception:
                    continue
                for status, image_buf in self._dec.drain():
                    if status != 0 or not image_buf:
                        if self._on_decode_error is not None:
                            with contextlib.suppress(Exception):
                                self._on_decode_error()
                        continue
                    try:
                        self._emit(image_buf)
                    finally:
                        vt_jpeg.cf.CFRelease(image_buf)

    return PromptVT


class Receiver:
    """RTP in, frames out: depacketize, gate on parameter sets and keyframes, decode, keep the encoder fed
    with RTCP. `transport` is the UDP receiver from `open_media_receiver` (recv/sendto/close)."""

    def __init__(self, transport, frames: Frames, decoder: str | Callable = "auto", local_ssrc: int = 0,
                 remote_ssrc: int = 0, rtcp_dest: tuple[str, int] | None = None, rr_interval: float = RR_INTERVAL,
                 pli_min_gap: float = PLI_MIN_GAP):
        self.transport, self.frames = transport, frames
        self._decoder_cls = decoder if callable(decoder) else decoder_class(decoder)
        self.local_ssrc, self.remote_ssrc, self.rtcp_dest = local_ssrc, remote_ssrc, rtcp_dest
        self.rr_interval, self.pli_min_gap = rr_interval, pli_min_gap
        self._decoder = None
        self.display: tuple[int, int] | None = None  # the device's display size (set_display), not the coded size
        self._highest_seq = 0
        self._last_pli = 0.0
        self._needs_key = True  # until the first IDR, and again after every loss
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- RTCP -------------------------------------------------------------------------------------
    def _sdes(self) -> bytes:
        cname = b"phonebridge"
        chunk = struct.pack("!I", self.local_ssrc & 0xFFFFFFFF) + bytes([1, len(cname)]) + cname + b"\0"
        chunk += b"\0" * (-len(chunk) % 4)
        return struct.pack("!BBH", 0x81, 0xCA, len(chunk) // 4) + chunk

    def receiver_report(self) -> bytes:
        rr = struct.pack("!BBHII BBBB IIII", 0x81, 0xC9, 7, self.local_ssrc & 0xFFFFFFFF, self.remote_ssrc & 0xFFFFFFFF,
                         0, 0, 0, 0, self._highest_seq & 0xFFFFFFFF, 0, 0, 0)
        return rr + self._sdes()

    def pli(self) -> bytes:
        return struct.pack("!BBHII", 0x81, 0xCE, 2, self.local_ssrc & 0xFFFFFFFF, self.remote_ssrc & 0xFFFFFFFF)

    async def _send(self, packet: bytes) -> bool:
        if self.rtcp_dest is None or not self.local_ssrc:
            return False
        try:
            await self.transport.sendto(packet, *self.rtcp_dest)
            return True
        except OSError as e:
            log.debug("rtcp send failed: %s", e)
            return False

    async def _rr_loop(self) -> None:
        while True:
            await asyncio.sleep(self.rr_interval)
            if await self._send(self.receiver_report()):
                self.frames.stats["rr"] += 1

    def _request_key(self, reason: str) -> None:
        self._needs_key = True
        now = time.monotonic()
        if now - self._last_pli < self.pli_min_gap or self._loop is None:
            return
        self._last_pli = now
        self.frames.stats["pli"] += 1
        log.debug("keyframe requested (%s)", reason)
        self._loop.create_task(self._send(self.pli()))

    # --- decoding ---------------------------------------------------------------------------------
    def _on_decode_error(self) -> None:
        self.frames.stats["decode_errors"] += 1
        if self._loop:
            self._loop.call_soon_threadsafe(self._request_key, "decode-error")

    def _start_decoder(self, vps: bytes, sps: bytes, pps: bytes) -> None:
        self._stop_decoder()
        self._decoder = self._decoder_cls(vps, sps, pps, on_frame=self._on_frame, on_decode_error=self._on_decode_error)
        self._apply_display()

    def set_display(self, display: tuple[int, int] | None) -> None:
        """The device's display size (from its own screenshot): coded frames are cropped to it at the origin, their
        right/bottom margins checked per buffer. A new size (rotation) is applied to the running decoder too."""
        self.display = display
        self._apply_display()

    def _apply_display(self) -> None:
        d = self._decoder
        if d is None:
            return
        if hasattr(d, "crop") and d.crop_from != "bitstream":
            rect = display_rect(self.display, d.coded)
            # the decoder thread reads crop, then crop_from: a crop is never visible without its margin check
            if rect:
                d.crop_from, d.crop = "display", rect
            else:
                d.crop, d.crop_from = None, None
            d.width, d.height = (rect[2], rect[3]) if rect else d.coded
        log.warning("screen stream decoder: %s %dx%d (coded %s, crop %s from %s; device display %s)", self._decoder_cls.__name__,
                    d.width, d.height, getattr(d, "coded", None), getattr(d, "crop", None), getattr(d, "crop_from", None),
                    self.display)

    def _on_frame(self, bgra: bytes, width: int | None = None, height: int | None = None, cropped: Rect | None = None) -> None:
        self.frames.push(bgra, width or self._decoder.width, height or self._decoder.height, cropped)

    def _stop_decoder(self) -> None:
        if self._decoder is not None:
            with contextlib.suppress(Exception):
                self._decoder.close()
            self._decoder = None

    async def run(self) -> None:
        """Until the transport closes (OSError) or the task is cancelled."""
        self._loop = asyncio.get_running_loop()
        rr = asyncio.ensure_future(self._rr_loop())
        try:
            await self._recv_loop()
        finally:
            rr.cancel()
            with contextlib.suppress(BaseException):
                await rr
            self._stop_decoder()

    async def _recv_loop(self) -> None:
        fu = bytearray()
        au: list[bytes] = []
        nals: list[bytes] = []
        vps = sps = pps = None
        au_key = au_corrupt = False
        last_seq: int | None = None
        packets = 0
        while True:
            try:
                data = await self.transport.recv(65535)
            except (OSError, asyncio.CancelledError) as e:
                log.warning("screen stream receive loop ended after %d packets: %r", packets, e)
                return
            except Exception as e:
                log.warning("screen stream receive loop crashed after %d packets: %r", packets, e)
                return
            packets += 1
            if packets == 1:
                log.warning("screen stream: first packet, %d bytes, pt %d", len(data), data[1] & 0x7F)
            if len(data) < 12 or 64 <= (data[1] & 0x7F) <= 95:  # short, or RTCP from the device
                continue
            marker = bool(data[1] & 0x80)
            hdr = 12 + (data[0] & 0x0F) * 4
            if data[0] & 0x10:
                hdr += 4 + int.from_bytes(data[hdr + 2:hdr + 4], "big") * 4
            seq = int.from_bytes(data[2:4], "big")
            if last_seq is not None and seq != (last_seq + 1) & 0xFFFF:
                fu.clear()
                au_corrupt = True
            if last_seq is None or (seq - last_seq) & 0xFFFF < 0x8000:
                last_seq = seq
                cycles = self._highest_seq >> 16
                if seq < (self._highest_seq & 0xFFFF) and (self._highest_seq & 0xFFFF) - seq > 0x8000:
                    cycles += 1
                self._highest_seq = (cycles << 16) | seq
            nals.clear()
            depacketize_hevc(data[hdr:], fu, nals)
            for nal in nals:
                if not nal:
                    continue
                nt = (nal[0] >> 1) & 0x3F
                if nt == NAL_VPS:
                    vps = bytes(nal)
                elif nt == NAL_SPS:
                    sps = bytes(nal)
                elif nt == NAL_PPS:
                    pps = bytes(nal)
                elif nt in KEY_NALS:
                    au_key = True
                au.append(nal)
            if not marker:
                continue
            self.frames.stats["aus"] += 1
            if au_corrupt:
                self.frames.stats["dropped_aus"] += 1
                self._request_key("rtp-gap")
            elif au:
                if au_key and vps and sps and pps and (self._needs_key or self._decoder is None):
                    try:
                        self._start_decoder(vps, sps, pps)  # fresh reference state at every recovery IDR
                        self._needs_key = False
                    except Exception as e:
                        log.warning("decoder start failed: %s", e)
                if self._decoder is not None and not self._needs_key:
                    self._decoder.feed(b"".join(b"\x00\x00\x00\x01" + n for n in au))
            au, au_key, au_corrupt = [], False, False


class ScreenStream:
    """`async with ScreenStream(rsd) as s:` the display stream is running, `s.frames` fills at the phone's
    frame rate, and `s.hid` is a UniversalHIDServiceService whose reports reach UIKit (the stream is its
    auth gate). Replaces pymobiledevice3's `touch_session` for the pmd3 backend."""

    def __init__(self, rsd, display_id: int = 1, decoder: str = "auto", display: tuple[int, int] | None = None):
        self.rsd, self.display_id, self.decoder = rsd, display_id, decoder
        self.display_size = display  # the device's display size, known before the first frame (Receiver.set_display)
        self.frames = Frames()
        self.hid = None
        self.receiver: Receiver | None = None
        self._display = self._transport = self._task = None
        self._stack = contextlib.AsyncExitStack()

    async def __aenter__(self) -> "ScreenStream":
        from pymobiledevice3.remote.core_device.display_service import DisplayService
        from pymobiledevice3.remote.core_device.hid_service import UniversalHIDServiceService
        from pymobiledevice3.remote.core_device.screen_stream import open_media_receiver

        sender_ip = self.rsd.service.address[0]
        try:
            self._display = await self._stack.enter_async_context(DisplayService(self.rsd))
            self._transport, receiver_ip = open_media_receiver(self._display, RCVBUF)
            try:
                answer = await asyncio.wait_for(self._display.start_video_stream(
                    receiver_ip=receiver_ip, receiver_port=self._transport.port, sender_ip=sender_ip,
                    display_id=self.display_id), 10.0)  # the offer touch_session proved on this phone
            except asyncio.TimeoutError as e:
                raise RuntimeError("timed out starting the phone's screen stream (its mediastream/dtuhidd daemon "
                                   "is likely wedged: reboot the phone)") from e
            cfg = answer.get("connection", {}).get("streamConfig", {}) or {}
            source_port = int(cfg.get("SourcePort", 0) or 0)
            log.warning("screen stream started: receiver %s:%s sender %s source_port %s ssrc local %s remote %s transport %s",
                        receiver_ip, self._transport.port, sender_ip, source_port, cfg.get("RemoteSSRC"), cfg.get("LocalSSRC"),
                        type(self._transport).__name__)
            self.receiver = Receiver(self._transport, self.frames, self.decoder,
                                     local_ssrc=int(cfg.get("RemoteSSRC", 0) or 0),  # ours
                                     remote_ssrc=int(cfg.get("LocalSSRC", 0) or 0),  # the phone's
                                     rtcp_dest=(sender_ip, source_port) if source_port else None)
            self.receiver.set_display(self.display_size)  # before run(): the very first frame is already cropped
            self._task = asyncio.ensure_future(self.receiver.run())
            await asyncio.sleep(SETTLE_S)
            self.hid = await self._stack.enter_async_context(UniversalHIDServiceService(self.rsd))
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *exc) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
            self._task = None
        if self._display is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._display.stop_media_stream(stop_all=True), 5)
        if self._transport is not None:
            with contextlib.suppress(Exception):
                self._transport.close()
            self._transport = None
        with contextlib.suppress(Exception):
            await self._stack.aclose()
        self.hid = self._display = None
