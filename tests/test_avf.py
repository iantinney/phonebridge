"""The QuickTime path as the eyes (supervisor 2026-10-18 23:4xZ): AVFoundation's iPhone screen device feeds
frames.latest(); the CoreDevice stream and DVT stills stay behind it. Offline: the device is a fake opener."""
import asyncio
import time

import pytest

from phonebridge import avf, stream
from phonebridge.backends import pmd3
from test_stream import FakeCtx, FakeSession


class FakeDevice:
    """What `avf.open_capture` returns: a running capture pushing BGRA frames until stop()."""
    opened = 0
    fail = False

    def __init__(self, push):
        FakeDevice.opened += 1
        self.push, self.stopped = push, False

    def feed(self, n=1, w=2, h=2):
        for _ in range(n):
            self.push(b"\0\0\xff\xff" * (w * h), w, h)

    def stop(self):
        self.stopped = True


async def fake_open(push, udid, wait_s=10.0):
    if FakeDevice.fail:
        raise RuntimeError("no iPhone screen device")
    return FakeDevice(push)


@pytest.fixture(autouse=True)
def reset():
    FakeDevice.opened, FakeDevice.fail = 0, False


def test_capture_pushes_the_device_frames_into_frames_latest():
    async def go():
        frames = stream.Frames()
        cap = avf.AvfCapture(frames, "udid", opener=fake_open)
        dev = await cap.start()
        dev.feed(2)
        f = frames.latest()
        assert f.seq == 2 and (f.width, f.height) == (2, 2) and f.image.getpixel((0, 0)) == (255, 0, 0)
        assert cap.running and cap._pump is not None and not cap._pump.done()  # the run-loop pump lives with the capture
        cap.stop()
        assert dev.stopped and not cap.running and cap._pump is None
    asyncio.run(go())


@pytest.fixture
def backend(monkeypatch):
    async def fake_rsd(self, p):
        p.stack = p.stack or __import__("contextlib").AsyncExitStack()
        return object()

    FakeSession.opened = 0
    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", fake_rsd)
    monkeypatch.setattr(pmd3, "ScreenStream", FakeSession)
    monkeypatch.setattr(pmd3, "DvtProvider", FakeCtx)
    monkeypatch.setattr(pmd3, "Screenshot", FakeCtx)
    monkeypatch.setattr(avf, "open_capture", fake_open)

    async def alone(self):
        return ["udid"]
    monkeypatch.setattr(pmd3.Pmd3Backend, "attached", alone)
    return pmd3.Pmd3Backend(idle=60, eyes="avf")


def test_backend_frames_come_from_avf_and_screen_uses_them(backend):
    async def go():
        frames = await backend.frames("udid")
        assert FakeDevice.opened == 1 and FakeSession.opened == 0  # looks don't open the HID stream
        backend._avf.device.feed(1, 3, 2)
        assert frames.latest().seq == 1
        assert pmd3.png_size(await backend.screen("udid")) == (3, 2)  # a fresh capture frame, no DVT round trip
        assert await backend.frames("udid") is frames and FakeDevice.opened == 1  # one capture, reused
        await backend.close()
        assert backend._avf is None
    asyncio.run(go())


def test_backend_falls_back_to_the_stream_when_the_capture_device_is_missing(backend):
    FakeDevice.fail = True

    async def go():
        frames = await backend.frames("udid")
        assert FakeSession.opened == 1 and isinstance(frames, stream.Frames)
        p = backend._phone("udid")
        assert frames is p.session.frames
        await backend.close()
    asyncio.run(go())


def test_capture_stops_when_nobody_looks(monkeypatch, backend):
    backend.idle = 0.05

    async def go():
        await backend.frames("udid")
        dev = backend._avf.device
        await asyncio.sleep(0.2)
        assert dev.stopped and backend._avf is None  # the phone's isCaptured flag is up only while we look
    asyncio.run(go())


def test_eyes_auto_means_avf_on_macos_and_the_stream_elsewhere(monkeypatch):
    monkeypatch.setattr(pmd3.sys, "platform", "darwin")
    assert pmd3.Pmd3Backend(eyes="auto").eyes == "avf"
    monkeypatch.setattr(pmd3.sys, "platform", "linux")
    assert pmd3.Pmd3Backend(eyes="auto").eyes == "stream"


# --- video only, and a capture that never outlives its session (Ian, 2026-10-19 01:2xZ) ----------------------------

class Port:
    def __init__(self, media):
        self._media, self._enabled = media, True

    def mediaType(self):
        return self._media

    def isEnabled(self):
        return self._enabled

    def setEnabled_(self, v):
        self._enabled = bool(v)


class Input:
    def __init__(self, *media):
        self._ports = [Port(m) for m in media]

    def ports(self):
        return self._ports


class Connection:
    def __init__(self, port):
        self._port = port

    def inputPorts(self):
        return [self._port]


class Output:
    """An output that reports its own connections (AVCaptureOutput.connections), as the strict check reads them."""
    def __init__(self, *media):
        self._conns = [Connection(Port(m)) for m in media]

    def connections(self):
        return self._conns


class Session:
    def __init__(self):
        self.inputs, self._outputs, self.running, self.calls = [], [], False, []

    def outputs(self):
        return self._outputs

    def addInput_(self, i):
        self.inputs.append(i)

    def addOutput_(self, o):
        self._outputs.append(o)

    def removeInput_(self, i):
        self.inputs.remove(i)
        self.calls.append("removeInput")

    def removeOutput_(self, o):
        self._outputs.remove(o)
        self.calls.append("removeOutput")

    def connections(self):
        return [Connection(p) for i in self.inputs for p in i.ports() if p.isEnabled()]

    def startRunning(self):
        self.running = True

    def stopRunning(self):
        self.running = False
        self.calls.append("stopRunning")

    def isRunning(self):
        return self.running


def test_the_muxed_devices_audio_port_is_disabled_and_an_audio_connection_is_refused():
    inp = Input("vide", "soun")
    assert [p.mediaType() for p in avf.disable_audio_ports(inp, "soun")] == ["soun"]
    s = Session()
    s.addInput_(inp)
    assert avf.video_only(s, "soun") == ["vide"]  # the audio port is off: no audio connection
    s2 = Session()
    s2.addInput_(Input("vide", "soun"))  # nobody disabled the audio port
    with pytest.raises(avf.AudioConnection):
        avf.video_only(s2, "soun")
    s3 = Session()
    s3.addInput_(Input("vide"))
    s3.addOutput_(Output("soun"))  # an output with an audio connection of its own (a movie file output would)
    with pytest.raises(avf.AudioConnection):
        avf.video_only(s3, "soun")


def test_stop_releases_the_device_not_just_pauses_it():
    s, inp, out = Session(), Input("vide"), object()
    s.addInput_(inp)
    s.addOutput_(out)
    s.startRunning()
    d = avf._Device(s, out, None, None, inp)
    d.stop()
    assert s.calls == ["stopRunning", "removeOutput", "removeInput"] and not s.running and s.inputs == [] and s.outputs() == []
    d.stop()  # idempotent
    assert s.calls.count("stopRunning") == 1


def test_park_and_reset_stop_the_capture(backend, tmp_path):
    from phonebridge.bridge import Bridge
    from phonebridge.config import Device
    from test_bridge import config

    async def go():
        await backend.frames("udid")
        dev = backend._avf.device
        cfg = config([])
        cfg.devices["iphone12"] = Device("iphone12", "udid", "device:iphone12")
        b = Bridge(backend, cfg, log_path=tmp_path / "calls.jsonl")
        b.park("usb", "iphone12")
        assert dev.stopped and backend._avf is None
        await backend.frames("udid")  # a new capture after the park, as a new session would
        dev2 = backend._avf.device

        async def gone():
            return []
        backend.attached = gone  # the phone left USB
        await backend._reset(backend._phone("udid"))
        assert dev2.stopped and backend._avf is None
    asyncio.run(go())


def test_eyes_dvt_is_the_kill_switch(monkeypatch):
    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", lambda self, p: None)
    be = pmd3.Pmd3Backend(eyes="dvt")

    async def go():
        with pytest.raises(RuntimeError, match="dvt"):
            await be.frames("udid")
        assert FakeDevice.opened == 0
    asyncio.run(go())


def test_the_capture_vouches_only_while_its_session_runs_and_the_device_is_connected():
    from phonebridge import avf, stream

    class Dev:
        connected = True

        def isConnected(self):
            return self.connected

    class Inp:
        def __init__(self):
            self.d = Dev()

        def device(self):
            return self.d

    class Sess:
        running = True

        def isRunning(self):
            return self.running

    class Handle:
        def __init__(self):
            self.session, self.inp, self.stopped = Sess(), Inp(), False

        def stop(self):
            self.stopped = True
    h = Handle()

    async def opener(push, udid, wait_s):
        return h
    cap = avf.AvfCapture(stream.Frames(), "udid", opener=opener)

    async def go():
        await cap.start()
        assert cap.frames.alive() is True
        h.inp.d.connected = False
        assert cap.frames.alive() is False
        h.inp.d.connected = True
        h.session.running = False
        assert cap.frames.alive() is False
        cap.stop()
        assert cap.frames.alive() is False
    asyncio.run(go())
