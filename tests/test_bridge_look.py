"""The daemon's `look` and `frame` tools: the fast look runs where the frames are (the Mac), and only the
verdict, its timings and the OCR lines cross the tunnel."""
import asyncio
import json
import io

import pytest
from PIL import Image

from phonebridge import mcp_server, stream
from phonebridge.bridge import Bridge
from phonebridge.screens.cascade import dhash
from phonebridge.screens.schema import Record
from phonebridge.screens.store import Store
from screens_fixtures.synth import frame as synth_frame
from test_bridge import FakeBackend, config
from test_screens_fast import TOP, ScriptedOCR, bgra


class StreamingBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.stream = stream.Frames()
        self.opened = 0

    async def frames(self, udid):
        self.opened += 1
        return self.stream


@pytest.fixture
def home():
    return synth_frame(top="pink", tab="black", body="white")


@pytest.fixture
def world(tmp_path, home):
    be = StreamingBackend()
    rec = Record.from_dict({"id": "instagram.home", "platform": "instagram", "bundle": "com.burbn.instagram",
                            "status": "confirmed", "match": {"chrome_dhash": {"region": list(TOP),
                                                                             "value": f"{dhash(home, TOP):016x}", "max_bits": 4}}})
    store = Store(records=[rec], corpus=tmp_path / "corpus")
    b = Bridge(be, config([]), log_path=tmp_path / "calls.jsonl")
    b.eyes_from(store, ocr=ScriptedOCR())
    return b, be


def test_look_runs_in_the_daemon_and_returns_the_verdict_with_timings(world, home):
    b, be = world
    raw = bgra(home)

    async def go():
        for _ in range(12):  # a capture streams the still screen (~60 fps): settle reads its recent frames
            be.stream.push(*raw)
            await asyncio.sleep(1 / 60)
        got = await b.look("iphone12", platform="instagram")
        assert got["screen"] == "instagram.home" and got["layer"] == "R3" and got["settled"]
        assert set(got["ms"]) == {"settle", "ocr", "pixels", "total"} and got["frame_seq"] == 12
        assert got["frame_age_ms"] < 1000 and got["ocr"] == ["fast"] and got["items"] == []
        assert got["stream"]["frames"] == 12 and "fps" in got["stream"] and got["content"]["by"] == "frames agree"
        assert be.shots == 0  # no DVT still, no PNG built for the tunnel
        rows = [l for l in (b.log_path.read_text().splitlines()) if '"op": "look"' in l]
        assert rows and '"screen": "instagram.home"' in rows[-1]
    asyncio.run(go())


def test_frame_is_the_latest_stream_frame_as_jpeg_or_png(world, home):
    b, be = world
    be.stream.push(*bgra(home))

    async def go():
        jpeg, meta = await b.frame("iphone12")
        assert jpeg[:2] == b"\xff\xd8" and Image.open(io.BytesIO(jpeg)).size == (1170, 2532)
        assert meta["source"] == "stream" and meta["seq"] == 1 and meta["wall"] > 1.7e9 and meta["age_ms"] >= 0
        png, _ = await b.frame("iphone12", format="png")
        assert png[:4] == b"\x89PNG"
    asyncio.run(go())


def test_look_without_live_frames_falls_back_to_a_still_and_says_so(tmp_path, home):
    """The eyes down (no stream, no capture): one DVT still per look keeps the caller seeing; `source: dvt` journals it."""
    be = FakeBackend()
    be.current = home
    rec = Record.from_dict({"id": "instagram.home", "platform": "instagram", "bundle": "com.burbn.instagram",
                            "status": "confirmed", "match": {"chrome_dhash": {"region": list(TOP),
                                                                             "value": f"{dhash(home, TOP):016x}", "max_bits": 4}}})
    b = Bridge(be, config([]), log_path=tmp_path / "calls.jsonl")
    b.eyes_from(Store(records=[rec], corpus=tmp_path / "corpus"), ocr=ScriptedOCR())
    got = asyncio.run(b.look("iphone12", platform="instagram"))
    assert got["source"] == "dvt" and got["screen"] == "instagram.home" and got["frame_seq"] is None and be.shots == 1
    rows = [l for l in b.log_path.read_text().splitlines() if '"op": "look"' in l]
    assert '"source": "dvt"' in rows[-1]


def test_frame_without_a_stream_falls_back_to_a_dvt_still(tmp_path):
    be = FakeBackend()
    b = Bridge(be, config([]), log_path=tmp_path / "calls.jsonl")
    jpeg, meta = asyncio.run(b.frame("iphone12"))
    assert jpeg[:2] == b"\xff\xd8" and meta["source"] == "dvt" and meta["seq"] is None and be.shots == 1


def test_mcp_exposes_look_and_frame(world):
    b, _ = world
    server = mcp_server.build(b)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"look", "frame"} <= names


def test_bench_runs_against_the_daemon_and_writes_per_look_timings(world, home, tmp_path):
    """The E1 live proof's runner, offline: an in-process MCP server stands in for the Mac daemon; no input."""
    from phonebridge.screens.live import bench
    b, be = world
    server = mcp_server.build(b)
    out = tmp_path / "ev"

    async def go():
        raw = bgra(home)

        async def phone_screen():  # the phone keeps sending frames, as it does live (a stale frame is never judged)
            while True:
                be.stream.push(*raw)
                await asyncio.sleep(0.05)
        feed = asyncio.ensure_future(phone_screen())
        try:
            return await bench("iphone12", "instagram", server, None, out, minutes=0.03)  # defaults: zero input
        finally:
            feed.cancel()
    summary = asyncio.run(go())
    assert summary["looks"] >= 1 and summary["recognised"] == summary["looks"] and summary["in_daemon_ms"]["p95"] is not None
    assert summary["frame_age_ms"]["p99"] is not None and summary["stream"]["frames"] >= 1 and summary["input"] == {"flicks": 0, "launch": False}
    rows = [l for l in (out / "events.jsonl").read_text().splitlines() if l]
    assert len(rows) == summary["looks"] and '"rt_ms"' in rows[0] and '"screen": "instagram.home"' in rows[0]
    assert '"frame_age_ms"' in rows[0] and '"event": "look"' in rows[0]
    assert (out / "RESULT.md").read_text().count("label:") == 1 and list(out.glob("001-*.jpg"))
    assert be.calls == []  # looks only: nothing touched the phone


def test_summarise_reads_a_controller_sessions_look_rows(tmp_path):
    from phonebridge.screens.live import looks_from, summarise
    ev = tmp_path / "events.jsonl"
    rows = [{"event": "look", "screen": "instagram.home", "layer": "R3", "settled": True, "frame_age_ms": 40 + i,
             "ms": {"settle": 20, "ocr": 25, "pixels": 3, "total": 50 + i}, "ocr": ["fast"], "stream": {"fps": 58, "frames": 100 + i,
             "dropped_aus": 0, "pli": 0}} for i in range(20)]
    rows.append({"event": "flick"})
    rows.append({"ev": "look", "screen": None, "layer": "none", "frame_age_ms": 70})
    ev.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
    s = summarise(looks_from(ev))
    assert s["looks"] == 21 and s["recognised"] == 20 and s["unknown"] == 1
    assert s["in_daemon_ms"]["p95"] == pytest.approx(68, abs=0.2) and s["frame_age_ms"]["p99"] >= 59
    assert s["layers"]["R3"] == 20 and s["stream"]["fps"]["p50"] == 58 and s["round_trip_ms"]["p50"] is None


# --- the pixels of the exact frame a look judged (controller-verify: seam checks) ----------------------------------

def _decode_jpeg(b64):
    import base64
    data = base64.b64decode(b64)
    assert data[:2] == b"\xff\xd8"
    return Image.open(io.BytesIO(data))


def test_look_with_frame_returns_the_judged_frame_and_frame_seq_fetches_it_later(world, home):
    b, be = world

    async def go():
        be.stream.push(*bgra(home))
        got = await b.look("iphone12", platform="instagram", with_frame=True)
        assert _decode_jpeg(got["frame_jpeg"]).size == (1170, 2532) and got["frame_seq"] == 1
        later = synth_frame(top="lavender", tab="black", body="white")
        be.stream.push(*bgra(later))  # the screen moved on: `frame()` would show this one
        data, meta = await b.frame("iphone12", seq=got["frame_seq"])
        assert meta["seq"] == 1 and meta["source"] == "look"
        assert Image.open(io.BytesIO(data)).convert("RGB").getpixel((585, 120)) == pytest.approx(
            Image.open(io.BytesIO(home)).convert("RGB").getpixel((585, 120)), abs=20)  # the judged frame, not the latest
        plain = await b.look("iphone12", platform="instagram")
        assert "frame_jpeg" not in plain  # opt-in: no image over the tunnel by default
    asyncio.run(go())


def test_frame_seq_that_was_evicted_says_so(world, home):
    b, be = world

    async def go():
        seqs = []
        for i in range(b.JUDGED_KEEP + 2):
            be.stream.push(*bgra(home))
            seqs.append((await b.look("iphone12", platform="instagram", settle=False))["frame_seq"])
        with pytest.raises(LookupError, match="no longer kept"):
            await b.frame("iphone12", seq=seqs[0])
        data, meta = await b.frame("iphone12", seq=seqs[-1])
        assert meta["seq"] == seqs[-1]
    asyncio.run(go())


def test_ocr_with_image_returns_the_lines_and_the_very_image_they_came_from(tmp_path, monkeypatch, home):
    from phonebridge import ocr as vision
    be = FakeBackend()
    be.current = home
    seen = []

    def recognize(png, fast=False, min_conf=0.3):
        seen.append(png)
        return [{"text": "Instagram", "x": 300, "y": 90, "w": 60, "h": 20, "conf": 1.0}]
    monkeypatch.setattr(vision, "recognize", recognize)
    b = Bridge(be, config([]), log_path=tmp_path / "calls.jsonl")
    out = asyncio.run(b.ocr("iphone12", with_image=True))
    assert out["items"][0]["text"] == "Instagram" and seen == [home]
    assert _decode_jpeg(out["image_jpeg"]).size == Image.open(io.BytesIO(home)).size
    assert isinstance(asyncio.run(b.ocr("iphone12")), list)  # default unchanged


def test_mcp_tools_take_the_new_options(world):
    b, _ = world
    tools = {t.name: t for t in asyncio.run(mcp_server.build(b).list_tools())}
    props = lambda n: set((getattr(tools[n], "input_schema", None) or getattr(tools[n], "inputSchema", None) or {}).get("properties", {}))
    assert "with_frame" in props("look") and "seq" in props("frame") and "with_image" in props("ocr")


def test_a_look_encodes_no_png_unless_the_operator_guard_needs_it(world, home, monkeypatch):
    """The served daemon runs with the operator guard off; a look must not pay the PNG encode it would feed."""
    b, be = world
    b.op_watch = False
    encoded = []
    from phonebridge.screens import cascade
    real = cascade.Frame._get_png
    monkeypatch.setattr(cascade.Frame, "png", property(lambda self: encoded.append(1) or real(self), cascade.Frame._set_png))

    async def go():
        be.stream.push(*bgra(home))
        await b.look("iphone12", platform="instagram")
    asyncio.run(go())
    assert encoded == []
    rows = [json.loads(l) for l in b.log_path.read_text().splitlines() if '"op": "look"' in l]
    assert rows[-1]["ms"] >= rows[-1]["total_ms"]  # the logged wall time covers the whole call


def test_a_fresh_recognised_look_is_the_frontmost_check(world, home):
    """R1 attempt 4: the DVT frontmost query took ~0.5 s before every swipe. A look that recognised one of the task
    app's screens on a frame under FG_MAX_AGE old is the same evidence; older, the query runs."""
    b, be = world
    b.config.lease_enforce = False  # this world has no authhub: inputs run without a device lock
    fg_calls = []

    async def foreground(udid):
        fg_calls.append(1)
        return "com.burbn.instagram"
    be.foreground = foreground
    raw = bgra(home)

    async def go():
        await b.launch("iphone12", "com.burbn.instagram")
        for _ in range(12):
            be.stream.push(*raw)
            await asyncio.sleep(1 / 60)
        await b.look("iphone12", platform="instagram")
        await b.swipe("iphone12", 500, 1500, 500, 800, think=False)
        assert fg_calls == []  # the look was the evidence
        await asyncio.sleep(1.2)  # the look's frame is now older than FG_MAX_AGE
        await b.swipe("iphone12", 500, 1500, 500, 800, think=False)
        assert fg_calls == [1]
    asyncio.run(go())


def test_only_the_newest_look_vouches_for_the_frontmost_app(world, home):
    """controller-verify (S1): an IG look then an unknown look must not leave the IG vouch alive. The input runs the
    DVT query, and a different foreground app refuses it."""
    from phonebridge.bridge import NotFrontmost
    b, be = world
    b.config.lease_enforce = False
    fg = {"app": "com.burbn.instagram", "calls": 0}

    async def foreground(udid):
        fg["calls"] += 1
        return fg["app"]
    be.foreground = foreground
    other = synth_frame(top="pink", tab="black", body="white", icon=("black", 300, 120))  # another top bar: unknown

    async def push(png, n=12):
        raw = bgra(png)
        for _ in range(n):
            be.stream.push(*raw)
            await asyncio.sleep(1 / 60)

    async def go():
        await b.launch("iphone12", "com.burbn.instagram")
        await push(home)
        assert (await b.look("iphone12", platform="instagram"))["screen"] == "instagram.home"
        await push(other)
        assert (await b.look("iphone12", platform="instagram"))["screen"] is None
        await b.swipe("iphone12", 500, 1500, 500, 800, think=False)
        assert fg["calls"] == 1  # the unknown look cleared the vouch: the DVT query ran
        await push(home)
        await b.look("iphone12", platform="instagram")
        await push(other)
        await b.look("iphone12", platform="instagram")
        fg["app"] = "com.apple.mobilephone"
        with pytest.raises(NotFrontmost):
            await b.swipe("iphone12", 500, 1500, 500, 800, think=False)
    asyncio.run(go())


def test_a_fresh_recognised_look_is_the_frontmost_check_for_a_double_tap_too(world, home):
    b, be = world
    b.config.lease_enforce = False
    fg_calls = []

    async def foreground(udid):
        fg_calls.append(1)
        return "com.burbn.instagram"
    be.foreground = foreground
    raw = bgra(home)

    async def go():
        await b.launch("iphone12", "com.burbn.instagram")
        for _ in range(12):
            be.stream.push(*raw)
            await asyncio.sleep(1 / 60)
        await b.look("iphone12", platform="instagram")
        await b.double_tap("iphone12", 500, 1200, think=False)
    asyncio.run(go())
    assert fg_calls == []
