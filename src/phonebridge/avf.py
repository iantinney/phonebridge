"""The iPhone's screen through AVFoundation as `frames.latest()` (the QuickTime path; supervisor 2026-10-18 23:4xZ).

With CoreMediaIO's AllowScreenCaptureDevices set in-process, the phone on USB appears to AVFoundation as an
external muxed device named "iPhone" (lane director: 1170×2532 at ~58 fps, hardware-decoded, first frame in
~1–3 s, from a process in the Mac's GUI session with Camera + Microphone access; the daemon is such a process).
An `AVCaptureVideoDataOutput` hands every frame to a delegate on its own dispatch queue as a 32BGRA pixel buffer;
one copy later it is the newest `StreamFrame`. No tunnel, no sudo, no RTP: the CoreDevice stream (`stream.py`)
stays as the second track and DVT stills as the fallback.

Device discovery needs the main thread's run loop (a worker thread never sees the phone), so `open_capture`
pumps it in short slices between `await`s and runs on the daemon's main thread. While a capture runs the phone's
`UIScreen.isCaptured` is true, as with QuickTime (PLAN E26): the backend keeps it up only while something looks.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import time
from typing import Awaitable, Callable

from .stream import Frames

log = logging.getLogger(__name__)

Push = Callable[[bytes, int, int], None]
Opener = Callable[..., Awaitable[object]]  # open_capture(push, name=..., wait_s=...) -> handle with .stop()


def enable_screen_capture_devices() -> int:
    """CMIOObjectSetPropertyData(kCMIOHardwarePropertyAllowScreenCaptureDevices = 1): iOS devices then appear to
    AVFoundation in this process, as QuickTime sees them (lane director's finding). macOS only."""
    cmio = ctypes.CDLL("/System/Library/Frameworks/CoreMediaIO.framework/CoreMediaIO")

    class Addr(ctypes.Structure):
        _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

    four = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    addr, allow = Addr(four("yes "), four("glob"), 1), ctypes.c_uint32(1)
    return cmio.CMIOObjectSetPropertyData(ctypes.c_uint32(1), ctypes.byref(addr), 0, None, 4, ctypes.byref(allow))


class AudioConnection(RuntimeError):
    """The capture session would carry audio: the iPhone is a muxed audio+video device and macOS shows the
    microphone indicator for any session whose input has an enabled audio port (Ian, 2026-10-19 01:2xZ:
    "why is python recording audio on my mac"). Video only, always."""


def disable_audio_ports(inp, audio_type: str = "soun") -> list:
    """Disable every audio port of a capture input (AVCaptureInput.Port.enabled = NO); returns the ports disabled."""
    off = []
    for port in inp.ports():
        if str(port.mediaType()) == str(audio_type):
            port.setEnabled_(False)
            off.append(port)
    return off


def video_only(session, audio_type: str = "soun") -> list[str]:
    """The media types of every connection's input ports and, when an output exposes them, its own connections;
    raises AudioConnection if any carries audio (an enabled audio port on either side)."""
    types = []
    conns = list(session.connections())
    for out in getattr(session, "outputs", lambda: [])() or []:
        if hasattr(out, "connections"):
            conns += list(out.connections())
    for c in conns:
        for port in c.inputPorts():
            t = str(port.mediaType())
            types.append(t)
            if t == str(audio_type) and (not hasattr(port, "isEnabled") or port.isEnabled()):
                raise AudioConnection(f"capture session has an audio connection ({t}); refusing to run")
        if hasattr(c, "audioChannels") and c.audioChannels():  # an audio connection by its own account
            raise AudioConnection("capture session has a connection with audio channels; refusing to run")
    return types


class _Device:
    """A running AVCaptureSession on the phone's screen device; `stop()` ends it and releases the device."""

    def __init__(self, session, output, delegate, queue, inp=None):
        self.session, self.output, self.delegate, self.queue, self.inp = session, output, delegate, queue, inp
        self.frames_seen = self.dropped = 0
        self.started = time.monotonic()
        self.stopped = False

    def stop(self) -> None:
        """stopRunning, then detach the output and the input, so the device (camera/mic path) is released and
        not merely paused; logged with the session's state after, for the next log reader."""
        if self.stopped:
            return
        self.stopped = True
        running = None
        try:
            self.session.stopRunning()
            if self.output is not None:
                self.session.removeOutput_(self.output)
            if self.inp is not None:
                self.session.removeInput_(self.inp)
            running = bool(self.session.isRunning()) if hasattr(self.session, "isRunning") else None
        except Exception as e:
            log.warning("avf stop raised: %s", e)
        finally:
            log.warning("avf capture stopped after %.0f s, %d frames (session running=%s)",
                        time.monotonic() - self.started, self.frames_seen, running)
            self.output = self.inp = self.delegate = None


async def pump_run_loop(running: Callable[[], bool], every: float = 0.02, slice_s: float = 0.005) -> None:
    """Keep the main thread's run loop turning while a capture runs: AVFoundation delivers nothing to a process whose
    main run loop never spins (director's recorder pumps it; the daemon's main thread is uvicorn's asyncio loop).
    A no-op off macOS."""
    try:
        from Foundation import NSDate, NSRunLoop
    except ImportError:
        return
    while running():
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(slice_s))
        await asyncio.sleep(every)


async def _pump(seconds: float) -> None:
    """Run the main thread's run loop for a slice, then yield to asyncio (discovery and session notifications
    arrive through the run loop; the daemon's loop must keep serving meanwhile)."""
    from Foundation import NSDate, NSRunLoop

    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(seconds))
    await asyncio.sleep(0)


async def _find_device(name: str, wait_s: float):
    import AVFoundation as AV

    types = [t for t in (getattr(AV, "AVCaptureDeviceTypeExternal", None),
                         getattr(AV, "AVCaptureDeviceTypeExternalUnknown", None)) if t]
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        await _pump(0.1)
        s = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            types, None, AV.AVCaptureDevicePositionUnspecified)
        for d in s.devices():
            if d.hasMediaType_(AV.AVMediaTypeMuxed) and (name.lower() in str(d.localizedName()).lower()
                                                         or str(d.modelID()) == "iOS Device"):
                return d
    raise RuntimeError(f"no iPhone screen device ({name!r}) within {wait_s:.0f} s: phone on USB, unlocked and trusted, "
                       "and the daemon's Python has Camera + Microphone access?")


_DELEGATE_CLASS = None


def _delegate_class():
    """The sample-buffer delegate class, defined once per process: pyobjc registers an Objective-C class per Python
    class statement, and a second capture in the same daemon would fail with "overriding existing Objective-C
    class" (seen live 2026-10-19 00:28Z, the second session after the idle stop). Instances carry `handler`."""
    global _DELEGATE_CLASS
    if _DELEGATE_CLASS is None:
        import objc
        import AVFoundation  # noqa: F401  (loads the framework: the protocol exists only after that)
        from Foundation import NSObject

        class PhonebridgeFrameDelegate(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
            def captureOutput_didOutputSampleBuffer_fromConnection_(self, out, sample_buffer, connection):
                handler = getattr(self, "handler", None)
                if handler is not None:
                    handler(sample_buffer)

            def captureOutput_didDropSampleBuffer_fromConnection_(self, out, sample_buffer, connection):
                dropped = getattr(self, "on_drop", None)
                if dropped is not None:
                    dropped()

        _DELEGATE_CLASS = PhonebridgeFrameDelegate
    return _DELEGATE_CLASS


async def open_capture(push: Push, name: str = "iPhone", wait_s: float = 10.0) -> _Device:
    """macOS: start capturing the phone's screen; every frame reaches `push(bgra, width, height)` from the
    capture's dispatch queue. Call from the main thread (the daemon's event loop)."""
    import objc
    import AVFoundation as AV
    import CoreMedia
    import libdispatch
    import Quartz

    enable_screen_capture_devices()
    dev = await _find_device(name, wait_s)
    session = AV.AVCaptureSession.alloc().init()
    inp, err = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(dev, None)
    if inp is None:
        raise RuntimeError(f"device input: {err}")
    off = disable_audio_ports(inp, AV.AVMediaTypeAudio)  # the muxed device offers audio too: never take it
    session.addInput_(inp)
    output = AV.AVCaptureVideoDataOutput.alloc().init()
    output.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
    output.setAlwaysDiscardsLateVideoFrames_(True)  # the newest frame is the only one anyone wants
    cv = ctypes.CDLL("/System/Library/Frameworks/CoreVideo.framework/CoreVideo")
    for fn, res in (("CVPixelBufferLockBaseAddress", ctypes.c_int32), ("CVPixelBufferUnlockBaseAddress", ctypes.c_int32),
                    ("CVPixelBufferGetBaseAddress", ctypes.c_void_p), ("CVPixelBufferGetBytesPerRow", ctypes.c_size_t),
                    ("CVPixelBufferGetWidth", ctypes.c_size_t), ("CVPixelBufferGetHeight", ctypes.c_size_t)):
        getattr(cv, fn).restype = res
        getattr(cv, fn).argtypes = [ctypes.c_void_p, ctypes.c_uint64] if "lock" in fn.lower() else [ctypes.c_void_p]
    handle = _Device(session, output, None, None, inp)

    def on_sample(sample_buffer) -> None:
        try:
            pb = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
            if pb is None:
                return
            ref = objc.pyobjc_id(pb)
            if cv.CVPixelBufferLockBaseAddress(ref, 1) != 0:  # kCVPixelBufferLock_ReadOnly
                return
            try:
                w, h = cv.CVPixelBufferGetWidth(ref), cv.CVPixelBufferGetHeight(ref)
                bpr, base = cv.CVPixelBufferGetBytesPerRow(ref), cv.CVPixelBufferGetBaseAddress(ref)
                if not base or w <= 0 or h <= 0:
                    return
                data = ctypes.string_at(base, h * bpr) if bpr == w * 4 else \
                    b"".join(ctypes.string_at(base + y * bpr, w * 4) for y in range(h))
            finally:
                cv.CVPixelBufferUnlockBaseAddress(ref, 1)
            handle.frames_seen += 1
            if handle.frames_seen == 1:
                log.warning("avf first frame %dx%d after %.1f s", w, h, time.monotonic() - handle.started)
            push(data, int(w), int(h))
        except Exception as e:  # never let a frame kill the capture
            log.warning("avf frame dropped: %s", e)

    def on_drop() -> None:
        handle.dropped += 1

    queue = libdispatch.dispatch_queue_create(b"phonebridge.avf", None)
    delegate = _delegate_class().alloc().init()
    delegate.handler, delegate.on_drop = on_sample, on_drop
    output.setSampleBufferDelegate_queue_(delegate, queue)
    if not session.canAddOutput_(output):
        raise RuntimeError("the screen device refuses a video data output")
    session.addOutput_(output)
    handle.delegate, handle.queue = delegate, queue
    try:
        types = video_only(session, AV.AVMediaTypeAudio)
    except AudioConnection:
        handle.stop()
        raise
    log.warning("avf session: connections %s, audio ports disabled %d", types, len(off))
    session.startRunning()  # on the main thread, like director's recorder (blocks ~0.5 s once per session)
    await _pump(0.2)  # let the session's first notifications through before the first frame is awaited
    try:
        types_running = video_only(session, AV.AVMediaTypeAudio)  # once more with the session running: nothing was re-enabled
    except AudioConnection:
        handle.stop()
        raise
    log.warning("avf capture started on %s (running=%s, connections %s, device audio ports off: %s)", dev.localizedName(),
                bool(session.isRunning()), types_running, [str(p.mediaType()) for p in off])
    return handle


class AvfCapture:
    """`await AvfCapture(frames).start()`: the phone's screen flows into `frames` until `stop()`."""

    def __init__(self, frames: Frames, name: str = "iPhone", opener: Opener | None = None, wait_s: float = 10.0):
        self.frames, self.name, self.wait_s = frames, name, wait_s
        self._opener = opener or open_capture
        self.device = None
        self.started_at: float | None = None

    async def start(self):
        self.device = await self._opener(self.frames.push, name=self.name, wait_s=self.wait_s)
        self.started_at = time.monotonic()
        self.frames.alive = self.alive
        self._pump = asyncio.ensure_future(pump_run_loop(lambda: self.running))
        return self.device

    @property
    def running(self) -> bool:
        return self.device is not None

    def alive(self) -> bool:
        """The capture can vouch for its latest frame: the session runs and the phone's screen device is still
        connected (it goes away when the phone leaves USB). A static screen sends no frames; this tells the two apart."""
        d = self.device
        if d is None or getattr(d, "stopped", False):
            return False
        try:
            running = bool(d.session.isRunning())
            connected = d.inp is None or bool(d.inp.device().isConnected())
            return running and connected
        except Exception:
            return False

    def stop(self) -> None:
        if self.device is not None:
            try:
                self.device.stop()
            finally:
                self.device = None
        if getattr(self, "_pump", None) is not None:
            self._pump.cancel()
            self._pump = None
