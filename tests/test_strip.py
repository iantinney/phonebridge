"""Strips: evidence recordings from the daemon's own live frames (the director asked: its second AVF capture lost the
phone after 16 s of a 42.5 s session, R1-IG attempt 5). Real ffmpeg on small frames; decoded back with PyAV."""
import asyncio
import json
import shutil
import threading
import time

import pytest

from phonebridge import stream
from phonebridge.bridge import Bridge
from phonebridge.lease import LeaseError
from phonebridge.strip import Strip

from test_bridge import FakeBackend, config

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")
W, H = 120, 260


def _frame(v):
    return bytes([v, v, v, 255]) * (W * H)


def _pusher(frames, plan):
    """plan: [(seconds, push?)]: push ~30 fps frames during a True span, nothing during a False one."""
    def run():
        v = 0
        for secs, push in plan:
            end = time.monotonic() + secs
            while time.monotonic() < end:
                if push:
                    v = (v + 7) % 256
                    frames.push(_frame(v), W, H)
                time.sleep(1 / 30)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _decoded_frames(path):
    av = pytest.importorskip("av")
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(c.streams.video[0]))


def test_a_strip_records_live_frames_at_a_fixed_rate_and_marks_a_gap(tmp_path):
    frames = stream.Frames()

    async def source():
        return frames

    async def go():
        t = _pusher(frames, [(0.6, True), (0.8, False), (0.6, True)])
        s = Strip(tmp_path / "strip.mp4", 10, source)
        await s.start()
        await asyncio.sleep(2.0)
        out = await s.stop()
        t.join()
        return out
    out = asyncio.run(go())
    assert out["ended"] == "stopped" and out["source_size"] == [W, H] and out["t0"] is not None
    assert out["size"] == [W // 2, H // 2]  # half size into the pipe (live 11-08: full size outran it)
    assert 15 <= out["frames"] <= 22 and out["dropped"] == 0  # ~2 s at 10 fps, the gap included (last frame repeats)
    assert len(out["gaps"]) == 1 and 0.4 <= out["gaps"][0][1] - out["gaps"][0][0] <= 1.1, out["gaps"]
    assert _decoded_frames(out["path"]) == out["frames"]


def test_a_strip_stops_itself_at_its_cap_and_when_the_lease_ends(tmp_path):
    frames = stream.Frames()

    async def source():
        return frames
    alive = {"v": True}

    async def lease_alive():
        return alive["v"]

    async def go():
        t = _pusher(frames, [(1.5, True)])
        a = Strip(tmp_path / "a.mp4", 10, source, max_s=0.5)
        await a.start()
        await asyncio.sleep(0.9)
        out_a = await a.stop()
        b = Strip(tmp_path / "b.mp4", 10, source, alive=lease_alive)
        await b.start()
        await asyncio.sleep(0.3)
        alive["v"] = False
        await asyncio.sleep(0.8)
        out_b = await b.stop()
        t.join()
        return out_a, out_b
    out_a, out_b = asyncio.run(go())
    assert out_a["ended"] == "max_s" and out_a["frames"] <= 8
    assert out_b["ended"] == "lease ended"


class StreamingBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.stream = stream.Frames()

    async def frames(self, udid):
        return self.stream


def test_bridge_record_start_needs_the_lease_and_the_evidence_root(tmp_path, monkeypatch, authhub):
    from phonebridge import bridge as bridge_mod
    monkeypatch.setattr(bridge_mod, "EVIDENCE_ROOT", tmp_path / "evidence")
    b = Bridge(StreamingBackend(), config(authhub[0]), log_path=tmp_path / "calls.jsonl")

    async def go():
        with pytest.raises(LeaseError, match="needs the session's lease"):
            await b.record_start("iphone12", str(tmp_path / "evidence" / "x.mp4"))
        with pytest.raises(ValueError, match="under"):
            await b.record_start("iphone12", str(tmp_path / "elsewhere.mp4"), lease="callertok")
        with pytest.raises(ValueError, match="under"):
            await b.record_start("iphone12", str(tmp_path / "evidence" / "x.mov"), lease="callertok")
        t = _pusher(b.backend.stream, [(1.2, True)])
        started = await b.record_start("iphone12", str(tmp_path / "evidence" / "s1" / "strip.mp4"), fps=10, lease="callertok")
        with pytest.raises(RuntimeError, match="already recording"):
            await b.record_start("iphone12", str(tmp_path / "evidence" / "s2.mp4"), lease="callertok")
        await asyncio.sleep(0.6)
        b.park("operator")  # the operator took the phone: the strip goes on, as a gap; nothing restarts the capture
        await asyncio.sleep(0.8)
        out = await b.record_stop("iphone12")
        t.join()
        return started, out
    started, out = asyncio.run(go())
    assert started["path"].endswith("s1/strip.mp4") and out["frames"] >= 10 and out["gaps"], out
    rows = [json.loads(l)["op"] for l in b.log_path.read_text().splitlines()]
    assert "record_start" in rows and "record_stop" in rows


@pytest.fixture
def authhub(tmp_path):
    from test_bridge import fake_authhub
    return fake_authhub(tmp_path)


def test_ffmpeg_is_found_outside_path_too(tmp_path, monkeypatch):
    """launchd's PATH lacks /opt/homebrew/bin: the daemon must still find ffmpeg there (11a8159, live)."""
    from phonebridge import strip
    fake = tmp_path / "bin" / "ffmpeg"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.delenv("PHONEBRIDGE_FFMPEG", raising=False)
    monkeypatch.setattr(strip, "FFMPEG_DIRS", (str(fake.parent),))
    assert strip.ffmpeg() == str(fake)
    monkeypatch.setattr(strip, "FFMPEG_DIRS", ())
    assert strip.ffmpeg() is None
    monkeypatch.setenv("PHONEBRIDGE_FFMPEG", str(fake))
    assert strip.ffmpeg() == str(fake)


def test_a_slow_encoder_makes_slots_late_never_lost(tmp_path, monkeypatch):
    """R1 probe (13:05Z, dad8e99): 59 slots were dropped when the encoder fell behind, so the strip ran 2 s short of
    the wall clock and every later frame sat early. A slot that can't be queued is owed and written as a repeat once
    there is room: frames == the slots of the run, frame n stays at t0 + n/fps."""
    from phonebridge import strip as strip_mod
    frames = stream.Frames()

    async def source():
        return frames
    real_write = strip_mod.Strip._write

    def slow_write(self):  # the encoder stalls 0.5 s at its start (a cold VideoToolbox session, say)
        time.sleep(0.5)
        real_write(self)
    monkeypatch.setattr(strip_mod.Strip, "_write", slow_write)

    async def go():
        t = _pusher(frames, [(1.6, True)])
        s = strip_mod.Strip(tmp_path / "late.mp4", 30, source)
        s._q = strip_mod.queue.Queue(maxsize=2)  # a tiny queue, so the stall shows
        await s.start()
        await asyncio.sleep(1.5)
        out = await s.stop()
        t.join()
        return out
    out = asyncio.run(go())
    assert out["dropped"] == 0 and out["late"] > 0 and out["late_spans"], out
    assert abs(out["frames"] / out["fps"] - out["duration_s"]) <= 0.15, out  # the strip's clock is the wall clock
    assert _decoded_frames(out["path"]) == out["frames"]


def test_a_strip_that_falls_behind_says_its_timeline_is_unreliable(tmp_path):
    """Live 11-08: 817 of 915 slots late (full-size frames outran the pipe); the summary must not pass for a timeline."""
    s = Strip(tmp_path / "s.mp4", 10, lambda: None)
    s.frames, s.late = 100, 3
    assert s.summary()["timeline"] == "reliable"
    s.late = 40
    assert s.summary()["timeline"].startswith("unreliable: 40 of 100")


def test_frames_go_into_the_pipe_at_half_size():
    from phonebridge.strip import _half
    w, h = 6, 4
    bgra = bytes(range(w * h * 4))
    out, ow, oh = _half(bgra, w, h)
    assert (ow, oh) == (3, 2) and len(out) == 3 * 2 * 4 and out[:4] == bgra[:4]


LEVELS = [10 + 12 * k for k in range(20)]  # a frame's gray level says which frame it is (libx264 keeps it within ±1)


def _numbered(frames, fps=10):
    """LEVELS pushed at `fps`, each frame's arrival (wall clock) recorded: {level: time.time() when it arrived}."""
    arrived = {}

    def run():
        for v in LEVELS:
            frames.push(_frame(v), W, H)
            arrived[v] = frames.latest().wall
            time.sleep(1 / fps)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, arrived


def _early(out, arrived):
    """The strip's frames in order (as levels), and every slot showing a frame that arrived after the slot's time."""
    av = pytest.importorskip("av")
    with av.open(out["path"]) as c:
        shown = [min(LEVELS, key=lambda v: abs(v - f.to_ndarray(format="gray").mean())) for f in c.decode(c.streams.video[0])]
    early = [(n, v, round(arrived[v] - (out["t0"] + n / out["fps"]), 3)) for n, v in enumerate(shown)
             if arrived[v] > out["t0"] + n / out["fps"] + 0.02]
    side = json.loads(open(out["sidecar"]).read())["slots"]  # the strip's own record: the same chronology, per slot
    assert len(side) == out["frames"] == len(shown) and [s for s, _ in side] == sorted(s for s, _ in side)
    assert all(wall <= out["t0"] + n / out["fps"] + 0.02 for n, (_, wall) in enumerate(side)), side
    assert out["new_content_frames"] == len({s for s, _ in side}) and out["latency_s"] is None
    return shown, early


class LatestOnly:
    """A frame source that only has its latest frame (no recent window)."""

    def __init__(self, frames):
        self.frames = frames

    def latest(self):
        return self.frames.latest()


@pytest.mark.parametrize("window", [True, False], ids=["recent-window", "latest-only"])
def test_a_slow_lease_check_never_puts_later_pixels_at_an_earlier_time(tmp_path, window):
    """Root's read (11-08): a slot handled after a slow await (the lease check is an authhub round trip from the Mac)
    took the newest frame, so pixels that arrived later sat at an earlier time. Each slot shows the newest frame that
    had arrived by its own time: from the recent window, else the previous slot's frame repeats (late), never a later
    one. The strip's clock stays the wall clock."""
    frames = stream.Frames()

    async def source():
        return frames if window else LatestOnly(frames)

    async def slow_alive():
        await asyncio.sleep(0.35)
        return True

    async def go():
        t, arrived = _numbered(frames)
        s = Strip(tmp_path / "slow.mp4", 10, source, alive=slow_alive)
        await s.start()
        await asyncio.sleep(1.8)
        out = await s.stop()
        t.join()
        return out, arrived
    out, arrived = asyncio.run(go())
    shown, early = _early(out, arrived)
    assert not early, early
    assert shown == sorted(shown) and len(set(shown)) >= (8 if window else 3)  # chronological; the frames move on
    assert abs(out["frames"] / out["fps"] - out["duration_s"]) <= 0.15, out
    if not window:
        assert out["late"] > 0 and out["late_spans"], out  # repeats where the loop fell behind, said so


def test_slots_owed_to_a_stalled_encoder_repeat_their_own_frame_never_a_later_one(tmp_path, monkeypatch):
    """The owed slots (the encoder full) were written later as the newest frame; they repeat the first owed slot's."""
    from phonebridge import strip as strip_mod
    frames = stream.Frames()

    async def source():
        return frames
    real_write = strip_mod.Strip._write

    def slow_write(self):
        time.sleep(0.5)
        real_write(self)
    monkeypatch.setattr(strip_mod.Strip, "_write", slow_write)

    async def go():
        t, arrived = _numbered(frames)
        s = strip_mod.Strip(tmp_path / "owed.mp4", 10, source)
        s._q = strip_mod.queue.Queue(maxsize=2)
        await s.start()
        await asyncio.sleep(1.6)
        out = await s.stop()
        t.join()
        return out, arrived
    out, arrived = asyncio.run(go())
    shown, early = _early(out, arrived)
    assert out["late"] > 0 and not early, (early, out)
    assert shown == sorted(shown) and len(shown) == out["frames"]


def test_an_encoder_far_slower_than_the_strip_says_so_and_never_runs_ahead(tmp_path, monkeypatch):
    """Live 11-08 (tsk_91d67ff507cb, before this fix): the pipe took ~1.5 of 15 frames/s, write order set the slot, so
    the video ran up to 41 s ahead of the journal. Now: slots stay at their time (repeats), the timeline says
    unreliable, and the sidecar shows how little new content the strip carries; nothing runs ahead."""
    from phonebridge import strip as strip_mod
    frames = stream.Frames()

    async def source():
        return frames
    real_write = strip_mod.Strip._write

    def slow_pipe(self):  # every write takes 0.15 s: ~6.7 of the strip's 10 frames/s
        put = self._proc.stdin.write

        class Slow:
            def write(self, b):
                time.sleep(0.15)
                return put(b)

            def close(self):
                return None
        real = self._proc.stdin
        self._proc.stdin = Slow()
        try:
            real_write(self)
        finally:
            self._proc.stdin = real
            real.close()
    monkeypatch.setattr(strip_mod.Strip, "_write", slow_pipe)

    async def go():
        t, arrived = _numbered(frames)
        s = strip_mod.Strip(tmp_path / "slowpipe.mp4", 10, source)
        s._q = strip_mod.queue.Queue(maxsize=2)
        await s.start()
        await asyncio.sleep(1.6)
        out = await s.stop()
        t.join()
        return out, arrived
    out, arrived = asyncio.run(go())
    shown, early = _early(out, arrived)
    assert not early and out["timeline"].startswith("unreliable"), out
    assert out["new_content_frames"] < out["frames"] and abs(out["frames"] / out["fps"] - out["duration_s"]) <= 0.15, out
