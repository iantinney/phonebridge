"""The phone's live screen stream as the eyes (PLAN E1 / A2): frames.latest() with a timestamp, decoded from the
same CoreDevice display stream that opens backboardd's HID gate. Offline: a synthetic HEVC/RTP stream (stream_helpers)
through the receiver and the PyAV decoder; the Mac uses VideoToolbox for the same bytes."""
import asyncio
import contextlib
import time

import pytest
from PIL import Image

from phonebridge import stream
from phonebridge.backends import pmd3
from stream_helpers import hevc_aus, rtp_packets, solid
from test_bridge import tiny_png

W, H = 128, 256
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]


class FakeTransport:
    """The UDP receiver the device streams to: scripted packets in, RTCP out."""

    def __init__(self, packets, idle=0.5):
        self.q = asyncio.Queue()
        for p in packets:
            self.q.put_nowait(p)
        self.sent, self.idle = [], idle  # idle: the stream "ends" this long after its last packet
        self.port, self.local_ip = 5000, "fd00::1"

    async def recv(self, bufsize=65535):
        try:
            return await asyncio.wait_for(self.q.get(), self.idle)
        except asyncio.TimeoutError:
            raise OSError("stream ended")

    async def sendto(self, data, ip, port):
        self.sent.append((data, ip, port))

    def close(self):
        pass


def run(coro):
    return asyncio.run(coro)


# --- Frames: the latest decoded frame, with a timestamp -------------------------------------------

def test_frames_latest_is_none_until_the_first_frame_and_wait_times_out():
    async def go():
        f = stream.Frames()
        assert f.latest() is None
        with pytest.raises(asyncio.TimeoutError):
            await f.wait(timeout=0.05)
    run(go())


def test_frames_latest_carries_seq_and_timestamps_and_wait_returns_only_newer_frames():
    async def go():
        f = stream.Frames()
        t0 = time.monotonic()
        f.push(b"\0\0\0\xff" * 6, 3, 2)
        a = f.latest()
        assert (a.seq, a.width, a.height) == (1, 3, 2) and t0 <= a.t <= time.monotonic() and a.wall > 1.7e9
        assert a.image.size == (3, 2) and a.image.mode == "RGB"
        waiter = asyncio.ensure_future(f.wait(after=a.seq, timeout=1.0))
        await asyncio.sleep(0.01)
        f.push(b"\xff\xff\xff\xff" * 6, 3, 2)
        b = await waiter
        assert b.seq == 2 and b.image.getpixel((0, 0)) == (255, 255, 255)
        assert (await f.wait(after=1, timeout=0.1)).seq == 2  # already there: no waiting
        assert f.stats["frames"] == 2
    run(go())


def test_frame_png_is_a_real_png_of_the_frame():
    f = stream.Frames()
    f.push(bytes([0, 0, 255, 255]) * 4, 2, 2)  # BGRA red
    png = f.latest().png()
    im = Image.open(__import__("io").BytesIO(png))
    assert im.size == (2, 2) and im.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


# --- Receiver: RTP/HEVC -> decoder -> Frames, with the device's feedback kept alive --------------

@pytest.fixture(scope="module")
def aus():
    return hevc_aus([solid(W, H, c) for c in COLORS])


def receiver(packets, frames=None, idle=0.5, **kw):
    frames = frames or stream.Frames()
    tr = FakeTransport(packets, idle)
    rx = stream.Receiver(tr, frames, decoder="av", local_ssrc=0xAAAA, remote_ssrc=0x1234, rtcp_dest=("fd00::2", 6000),
                         **kw)
    return tr, frames, rx


def test_receiver_decodes_every_frame_of_a_clean_stream(aus):
    async def go():
        tr, frames, rx = receiver(rtp_packets(aus))
        await rx.run()
        last = await frames.wait(after=len(COLORS) - 1, timeout=2.0)
        assert last.seq == len(COLORS) and (last.width, last.height) == (W, H)
        r, g, b = last.image.getpixel((W // 2, H // 2))
        assert (r, g, b) == pytest.approx(COLORS[-1], abs=12)  # the last frame's colour survived HEVC
        assert frames.stats["frames"] == len(COLORS) and frames.stats["dropped_aus"] == 0
    run(go())


def test_a_packet_gap_drops_that_frame_asks_for_a_keyframe_and_resumes_at_the_next_idr(aus):
    async def go():
        packets = rtp_packets(aus)
        # drop one packet from the second frame (a P-frame after the first IDR; keyint=3 -> frame 4 is an IDR)
        second = [i for i, p in enumerate(packets) if int.from_bytes(p[4:8], "big") == 800]
        del packets[second[0]]
        tr, frames, rx = receiver(packets)
        await rx.run()
        await asyncio.sleep(0.2)
        assert frames.stats["dropped_aus"] >= 1
        plis = [d for d, ip, port in tr.sent if d[1] == 0xCE and (d[0] & 0x1F) == 1]
        assert plis and (ip_port := [(ip, port) for d, ip, port in tr.sent][0]) == ("fd00::2", 6000)
        last = frames.latest()
        assert last is not None and last.image.getpixel((W // 2, H // 2)) == pytest.approx(COLORS[-1], abs=12)
        # frames 2 and 3 reference the dropped one: held until the IDR at frame 4, never shown corrupted
        assert frames.stats["frames"] == len(COLORS) - 2
    run(go())


def test_receiver_sends_a_receiver_report_on_its_interval(aus):
    async def go():
        tr, frames, rx = receiver(rtp_packets(aus), rr_interval=0.05)
        task = asyncio.ensure_future(rx.run())
        await asyncio.sleep(0.3)
        await task
        rrs = [d for d, *_ in tr.sent if d[1] == 0xC9]
        assert len(rrs) >= 3 and all(int.from_bytes(d[4:8], "big") == 0xAAAA for d in rrs)
    run(go())


# --- pmd3: one session is both the HID gate and the eyes; DVT stills stay the fallback --------------

class FakeHid:
    def __init__(self):
        self.reports = []

    async def send_report(self, service_id, report):  # the backend's own 58-byte DigitizerReport
        assert service_id == 257 and len(report) == 58
        assert int.from_bytes(report[45:53], "little") > 0  # remoteTimestamp where the phone reads it (Int64 at 45)
        self.reports.append((report[3], int.from_bytes(report[4:6], "little"), int.from_bytes(report[6:8], "little")))

    async def send_touchscreen(self, state, x, y):
        self.reports.append((state, x, y))


class FakeSession:
    """Stands in for stream.ScreenStream: counts how often a session is opened."""
    opened = 0
    last_display = None

    def __init__(self, rsd, **kw):
        self.hid, self.frames = FakeHid(), stream.Frames()
        FakeSession.last_display = kw.get("display")

    async def __aenter__(self):
        FakeSession.opened += 1
        return self

    async def __aexit__(self, *exc):
        pass


class FakeCtx:
    def __init__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def get_screenshot(self):
        return tiny_png(3, 5)


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
    return pmd3.Pmd3Backend(idle=60)


def test_tap_and_frames_share_one_stream_session(backend):
    async def go():
        frames = await backend.frames("udid")
        frames.push(b"\0" * 4 * 15, 3, 5)  # the device's display size (the fake DVT still is 3×5): taps map by it
        await backend.tap("udid", 1, 1, 0.01)
        assert FakeSession.opened == 1
        p = backend._phone("udid")
        assert p.session.hid.reports and p.session.frames is frames
        await backend.close()
    run(go())


def test_screen_prefers_a_fresh_stream_frame_and_falls_back_to_dvt(backend):
    async def go():
        assert pmd3.png_size(await backend.screen("udid")) == (3, 5)  # no session yet: a DVT still
        assert FakeSession.opened == 0  # a plain screenshot never lights the capture indicator
        frames = await backend.frames("udid")
        frames.push(b"\0" * 4 * 8, 2, 4)
        assert pmd3.png_size(await backend.screen("udid")) == (2, 4)  # fresh frame: no DVT round trip
        frames._latest = frames._latest._replace(t=time.monotonic() - 5)  # stale: the stream stalled
        assert pmd3.png_size(await backend.screen("udid")) == (3, 5)
        await backend.close()
    run(go())


def test_full_size_frames_decode_well_inside_the_frame_budget():
    """The receiver keeps up with full-size 1170×2532 frames end to end (depacketize, software decode, BGRA, publish).
    The number printed is this machine's software path (a loaded 4-core VM measured ~200 ms/frame); the Mac decodes
    with VideoToolbox (pymobiledevice3 measured ~1 ms/frame), and CONTROLLER_BAR L1 (frame available p95 ≤ 150 ms)
    is measured live there, never here."""
    from stream_helpers import solid
    frames_in = [solid(1170, 2532, (40 * i % 255, 90, 200 - 30 * i)) for i in range(8)]
    aus_full = hevc_aus(frames_in, keyint=8)

    async def go():
        tr, frames, rx = receiver(rtp_packets(aus_full, mtu=1400), idle=5.0)  # a software decoder on a loaded box lags
        t = time.perf_counter()
        task = asyncio.ensure_future(rx.run())
        last = await frames.wait(after=len(frames_in) - 1, timeout=10.0)
        per_frame_ms = (time.perf_counter() - t) * 1000 / len(frames_in)
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        # libx265 pads to its coding-block size and PyAV publishes the coded size (Apple's SPS carries the display
        # size, and VideoToolbox on the Mac crops): within one block of the input is right here
        assert last.seq == len(frames_in) and abs(last.width - 1170) <= 16 and abs(last.height - 2532) <= 16
        assert per_frame_ms < 1000, per_frame_ms  # keeps up at the phone's rate once decode is hardware
        return per_frame_ms
    print("full-size decode ms/frame:", round(run(go()), 1))


# ---- the stream decoder's clean aperture (live 11-08: HEVC coded 1184×2576 carrying the 1170×2532 screen at the top left)

def _rect(x, y, w, h):
    return (x, y, w, h)


def _buffer(w, h, bpr):
    """A 32BGRA buffer whose every pixel encodes its own (x, y) in B,G (low bytes) and R,A (high bytes)."""
    rows = []
    for y in range(h):
        row = b"".join(bytes((x & 255, y & 255, x >> 8, y >> 8)) for x in range(w))
        rows.append(row + b"\x00" * (bpr - w * 4))
    buf = b"".join(rows)
    return lambda off, n: buf[off:off + n]


def _px(data, w, x, y):
    b = data[(y * w + x) * 4:(y * w + x) * 4 + 4]
    return b[0] | b[2] << 8, b[1] | b[3] << 8


def test_a_padded_buffer_is_cropped_to_its_clean_aperture_at_the_buffers_stride():
    w, h, bpr = 1184, 2576, 1184 * 4 + 64  # macroblock-padded coded size, and a stride wider than the row
    rect = stream.clean_rect(_rect(0, 0, 1170, 2532), w, h)
    data, cw, ch = stream.crop_rows(_buffer(w, h, bpr), bpr, w, h, rect)
    assert (cw, ch) == (1170, 2532) and len(data) == 1170 * 2532 * 4
    assert _px(data, cw, 0, 0) == (0, 0) and _px(data, cw, 1169, 2531) == (1169, 2531)  # the screen's corners, unmoved


def test_an_aperture_off_the_origin_keeps_its_own_origin():
    w, h, bpr = 64, 48, 64 * 4 + 16
    data, cw, ch = stream.crop_rows(_buffer(w, h, bpr), bpr, w, h, stream.clean_rect(_rect(7, 5, 50, 40), w, h))
    assert (cw, ch) == (50, 40) and _px(data, cw, 0, 0) == (7, 5) and _px(data, cw, 49, 39) == (56, 44)


def test_no_or_unusable_aperture_passes_the_buffer_uncropped_without_its_stride():
    w, h, bpr = 40, 30, 40 * 4 + 32
    for bad in (None, _rect(0, 0, 41, 30), _rect(-1, 0, 10, 10), _rect(0, 0, 0, 30), "not a rect", (float("nan"), 0, 1, 1)):
        rect = stream.clean_rect(bad, w, h)
        assert rect is None
        data, cw, ch = stream.crop_rows(_buffer(w, h, bpr), bpr, w, h, rect)
        assert (cw, ch) == (40, 30) and len(data) == 40 * 30 * 4 and _px(data, cw, 39, 29) == (39, 29)


def test_a_tight_full_buffer_is_one_read():
    reads = []
    buf = _buffer(8, 4, 32)
    data, cw, ch = stream.crop_rows(lambda off, n: reads.append((off, n)) or buf(off, n), 32, 8, 4, (0, 0, 8, 4))
    assert reads == [(0, 128)] and (cw, ch) == (8, 4)


def test_frames_carry_the_aperture_label():
    from phonebridge.stream import Frames
    f = Frames()
    f.push(b"\x00" * 16, 2, 2, cropped=(0, 0, 2, 2))
    assert f.latest().cropped == (0, 0, 2, 2)
    f.push(b"\x00" * 16, 2, 2)
    assert f.latest().cropped is None


def test_the_live_padded_explore_frame_is_recognised_once_cropped():
    """frm_4bdfd13b34a7 (live 11-08, recut 39d0c12, the display stream over the userspace tunnel): coded 1184×2576, Explore unrecognised (bookmark 0.909 <
    0.92). Cropped to its 1170×2532 aperture it is Explore again; the threshold is untouched. A private corpus frame."""
    import json
    from PIL import Image
    from phonebridge.screens.cascade import Cascade, Frame
    from phonebridge.screens.store import CORPUS, RECORD_DIRS, Store
    src = CORPUS / "sequences" / "avf_padding" / "1" / "01-explore_padded.jpg"
    if not src.exists():
        pytest.skip("private corpus frame not here")
    im = Image.open(src).convert("RGBA")
    bgra = im.tobytes("raw", "BGRA")
    items = json.loads(src.with_suffix(".ocr.json").read_text())
    cas = Cascade(Store(records_dirs=RECORD_DIRS, corpus=CORPUS))

    def recognise(data, w, h):
        png = __import__("io").BytesIO()
        Image.frombuffer("RGBA", (w, h), data, "raw", "BGRA", 0, 1).convert("RGB").save(png, "PNG")
        return cas.recognise(Frame.from_png(png.getvalue(), items), platform="instagram").screen_id

    w, h = im.size
    assert (w, h) == (1184, 2576) and recognise(bgra, w, h) is None  # as delivered: unrecognised
    data, cw, ch = stream.crop_rows(lambda off, n: bgra[off:off + n], w * 4, w, h, stream.clean_rect(_rect(0, 0, 1170, 2532), w, h))
    assert (cw, ch) == (1170, 2532) and recognise(data, cw, ch) == "instagram.screen.instagram_explore_search_grid"


# ---- the device's display size as the stream's crop (root GO 11-08: no conformance window in iOS 27.2's stream) ------

def test_the_display_crops_a_coded_frame_at_its_origin_only_when_it_fits():
    assert stream.display_rect((1170, 2532), (1184, 2576)) == (0, 0, 1170, 2532)
    assert stream.display_rect((1170, 2532), (1170, 2532)) == (0, 0, 1170, 2532)
    for display in (None, (1200, 2532), (1170, 2600), (2532, 1170), (0, 2532)):  # unknown, too big, rotated, empty
        assert stream.display_rect(display, (1184, 2576)) is None


def test_the_margin_beyond_the_display_must_be_dark_padding():
    w, h, bpr = 1184, 2576, 1184 * 4 + 64
    black = bytes(bpr * h)
    rect = (0, 0, 1170, 2532)
    assert stream.margin_dark(lambda off, n: black[off:off + n], bpr, w, h, rect)
    lit = bytearray(black)
    lit[2532 * bpr:] = b"\xc8" * (len(lit) - 2532 * bpr)  # the bottom margin is content, not padding (a sparse check)
    assert not stream.margin_dark(lambda off, n: bytes(lit[off:off + n]), bpr, w, h, rect)


def test_the_receiver_hands_the_display_to_the_running_decoder_and_back_on_rotation():
    class Dec:  # a decoder with a coded size and no conformance window (live iOS 27.2)
        def __init__(self, *a, **kw):
            self.coded, self.width, self.height, self.crop, self.crop_from = (1184, 2576), 1184, 2576, None, None
    r = stream.Receiver(None, stream.Frames(), decoder=Dec)
    r._start_decoder(b"", b"", b"")
    assert r._decoder.crop is None and (r._decoder.width, r._decoder.height) == (1184, 2576)  # display unknown
    r.set_display((1170, 2532))
    assert r._decoder.crop == (0, 0, 1170, 2532) and r._decoder.crop_from == "display" and r._decoder.width == 1170
    r.set_display((2532, 1170))  # rotated: doesn't fit the coded portrait frame, so uncropped (and input refuses it)
    assert r._decoder.crop is None and (r._decoder.width, r._decoder.height) == (1184, 2576)
    r._on_frame(b"\0" * 16, 2, 2, (0, 0, 2, 2))
    assert (r.frames.latest().width, r.frames.latest().cropped) == (2, (0, 0, 2, 2))


def test_a_stream_knows_the_display_before_its_first_frame(backend):
    """Live 11-08 09:13Z: the first frame of a new stream was pushed before the display reached the decoder (uncropped).
    The backend learns the display from a DVT still BEFORE starting the stream and hands it over at creation."""
    async def go():
        await backend.frames("udid")
        p = backend._phone("udid")
        assert p.display == (3, 5) and FakeSession.last_display == (3, 5)  # the fake DVT still is 3×5
        await backend.close()
    run(go())


def test_the_first_decoded_frame_is_cropped_when_the_display_comes_at_creation():
    """Root 11-08: the display given before run() reaches the decoder as it starts, so its FIRST frame is cropped."""
    class Dec:
        def __init__(self, *a, on_frame=None, **kw):
            self.coded, self.width, self.height, self.crop, self.crop_from, self.on_frame = (8, 6), 8, 6, None, None, on_frame

        def feed(self):  # what PromptVT._emit does with a padded buffer, by the crop it holds at that moment
            w, h = (self.crop[2], self.crop[3]) if self.crop else self.coded
            self.on_frame(bytes(w * h * 4), w, h, self.crop)
    frames = stream.Frames()
    r = stream.Receiver(None, frames, decoder=Dec)
    r.set_display((6, 4))  # ScreenStream(display=) → before run(), before any parameter sets
    r._start_decoder(b"", b"", b"")
    r._decoder.feed()
    first = frames.latest()
    assert first.seq == 1 and (first.width, first.height, first.cropped) == (6, 4, (0, 0, 6, 4))


def test_a_failed_display_lookup_starts_the_stream_uncropped_and_input_refuses_once(backend, monkeypatch):
    """Startup failure: no DVT still → the stream still opens (uncropped, labelled); a tap tries the screenshot once more,
    then refuses with zero input; nothing recurses into the session."""
    calls = []

    async def no_still(self, p):
        calls.append(1)
        raise RuntimeError("DVT down")
    monkeypatch.setattr(pmd3.Pmd3Backend, "_dvt_png", no_still)

    async def go():
        await backend.frames("udid")
        assert FakeSession.last_display is None and len(calls) == 1
        hid = backend._phone("udid").session.hid
        with pytest.raises(RuntimeError, match="display size is unknown"):
            await backend.tap("udid", 1, 1, 0.01)
        assert hid.reports == [] and len(calls) == 2  # zero input; one more screenshot try, no loop (a failed tap resets)
        await backend.close()
    run(go())
