"""The fast look (PLAN §5.2 R0 stream frame → R1 settle → R2/R3 pixels → fast OCR for stops and popups; accurate
OCR only on a miss): recognised screens in well under 300 ms, measured inside the daemon."""
import asyncio
import io
import random
import time
from pathlib import Path

import pytest
from PIL import Image

from phonebridge import stream
from phonebridge.screens import fast
from phonebridge.screens.cascade import CHANGE_EPS, Frame, _gray, changed, dhash
from phonebridge.screens.schema import Record
from phonebridge.screens.store import Store
from screens_fixtures.synth import frame as synth_frame
from phonebridge.config import DATA_DIR

TOP = (0.0, 0.04, 1.0, 0.12)


def bgra(png: bytes) -> tuple[bytes, int, int]:
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    b, g, r, a = im.split()[2], im.split()[1], im.split()[0], im.split()[3]
    return Image.merge("RGBA", (b, g, r, a)).tobytes(), im.width, im.height


def item(text, x, y, w=60, h=20):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": 1.0}


def record(id, png, **extra):
    return Record.from_dict({"id": id, "platform": "instagram", "bundle": "com.burbn.instagram", "status": "confirmed",
                             "match": {"chrome_dhash": {"region": list(TOP), "value": f"{dhash(png, TOP):016x}",
                                                        "max_bits": 4}}, **extra})


POPUP = Record.from_dict({"id": "ios.allow_paste", "platform": "ios", "bundle": "*", "kind": "popup",
                          "status": "confirmed", "match": {"ocr_any": ["Allow Paste", "Don't Allow"], "at_least": 2},
                          "answer": "allow", "targets": {"allow": {"ocr": "Allow Paste"}}})
HOME_OCR = Record.from_dict({"id": "instagram.home", "platform": "instagram", "bundle": "com.burbn.instagram",
                             "status": "confirmed", "match": {"ocr_any": ["Instagram", "Your story"], "at_least": 2}})


class ScriptedOCR:
    """Vision stand-in: what each level returns, and which levels were asked for."""

    def __init__(self, fast_items=(), accurate_items=()):
        self.fast_items, self.accurate_items, self.levels = list(fast_items), list(accurate_items), []

    def __call__(self, png_or_image, fast=False, min_conf=0.3, correct=None):
        self.levels.append("fast" if fast else "accurate")
        self.corrections = getattr(self, "corrections", []) + [correct]
        return self.fast_items if fast else self.accurate_items


def producer(frames: stream.Frames, pngs: list[bytes], gap: float = 0.02):
    """Push the scripted frames into `frames` like the decoder would, one every `gap` seconds (the phone sends
    30–60 fps; the conversion below is test plumbing, done before the clock matters)."""
    raw = [bgra(png) for png in pngs]

    async def go():
        for b in raw:
            await asyncio.sleep(gap)
            frames.push(*b)
    return asyncio.ensure_future(go())


@pytest.fixture
def home_png():
    return synth_frame(top="pink", tab="black", body="white")


@pytest.fixture
def eyes(tmp_path, home_png):
    def make(records, ocr):
        store = Store(records=records, corpus=tmp_path / "corpus")
        frames = stream.Frames()

        async def frames_of(udid):
            return frames
        return fast.Eyes(store, frames_of, ocr=ocr), frames
    return make


def test_fast_look_settles_on_stream_frames_and_recognises_by_pixels(eyes, home_png):
    moving = [synth_frame(top=c, tab="black", body="noise", seed=i) for i, c in enumerate(("black", "white", "gray"))]  # luminance, not hue
    e, frames = eyes([record("instagram.home", home_png)], ocr := ScriptedOCR())

    async def go():
        producer(frames, moving + [home_png] * 10)  # a capture keeps sending the still screen (~60 fps on AVF)
        look = await e.look("udid", platform="instagram")
        assert look.screen == "instagram.home" and look.layer == "R3"
        assert look.frames_seen >= 2 and look.settled and look.content["by"] == "frames agree"
        assert ocr.levels == ["fast"]  # stops and popups are OCR-only: the fast pass runs on every look
        assert set(look.ms) >= {"settle", "pixels", "ocr", "total"}
        assert look.ms["pixels"] + look.ms["ocr"] < 100  # the loop's own work; the < 300 ms claim is measured live
        assert look.frame.seq > len(moving) and look.frame_age_ms < 1000  # a still frame, ~100 ms of them after the motion
        assert 0 <= look.age_at_select_ms <= look.frame_age_ms and "age_at_select_ms" in look.summary()  # the rest is judging time
    asyncio.run(go())


def test_a_popup_over_a_known_screen_is_seen_by_the_fast_ocr(eyes, home_png):
    e, frames = eyes([record("instagram.home", home_png), POPUP],
                     ScriptedOCR(fast_items=[item("Allow Paste", 585, 1500), item("Don't Allow", 585, 1600)]))

    async def go():
        producer(frames, [home_png, home_png])
        look = await e.look("udid", platform="instagram")
        assert look.screen == "ios.allow_paste"
    asyncio.run(go())


def test_a_pixel_miss_falls_back_to_accurate_ocr_and_r4(eyes, home_png):
    e, frames = eyes([HOME_OCR], ocr := ScriptedOCR(accurate_items=[item("Instagram", 300, 90), item("Your story", 70, 300)]))

    async def go():
        producer(frames, [home_png, home_png])
        look = await e.look("udid", platform="instagram")
        assert (look.screen, look.layer) == ("instagram.home", "R4") and ocr.levels == ["fast", "accurate"]
    asyncio.run(go())


def test_an_unsettled_screen_still_returns_after_the_timeout_with_the_last_frame(eyes, home_png):
    moving = [synth_frame(top=("black", "white", "gray", "pink")[i % 4], tab="black", body="noise", seed=i) for i in range(40)]
    e, frames = eyes([record("instagram.home", home_png)], ScriptedOCR())
    e.settle_timeout, e.still_after = 0.3, 0.05  # frames keep coming every ~10 ms: never "still"

    async def go():
        producer(frames, moving, gap=0.01)
        look = await e.look("udid", platform="instagram")
        assert not look.settled and look.frame is not None and look.ms["settle"] >= 300
    asyncio.run(go())


def test_expected_screens_resolve_at_r2_without_scanning_the_store(eyes, home_png):
    other = synth_frame(top="lavender", tab="black", body="white")
    e, frames = eyes([record("instagram.home", home_png), record("instagram.other", other)], ScriptedOCR())

    async def go():
        producer(frames, [home_png, home_png])
        look = await e.look("udid", platform="instagram", expect=["instagram.home"])
        assert (look.screen, look.layer) == ("instagram.home", "R2")
    asyncio.run(go())


# --- R1 on frames: the change measure is vectorised and equals the reference ---------------------

def reference_changed(prev: Image.Image, cur: Image.Image, chrome, threshold=40) -> float:
    a, b = _gray(prev), _gray(cur)
    w, h = a.size
    total = diff = 0
    for r in chrome:
        box = (int(r[0] * w), int(r[1] * h), int(r[2] * w), int(r[3] * h))
        pa, pb = a.crop(box).tobytes(), b.crop(box).tobytes()
        total += len(pa)
        diff += sum(1 for x, y in zip(pa, pb) if abs(x - y) > threshold)
    return diff / total


def test_changed_matches_the_reference_and_runs_in_milliseconds():
    rnd = random.Random(1)
    a = Image.effect_noise((1170, 2532), 60).convert("L")
    b = a.point(lambda v: min(255, v + rnd.randrange(0, 90)))
    chrome = [(0.0, 0.04, 1.0, 0.12), (0.0, 0.88, 1.0, 1.0)]
    t = time.perf_counter()
    got = changed(a, b, chrome=chrome)
    ms = (time.perf_counter() - t) * 1000
    assert got == pytest.approx(reference_changed(a, b, chrome), abs=0.01)
    assert ms < 150, ms  # the pure-Python loop it replaces took ~200 ms even on an idle machine
    assert changed(a, a, chrome=chrome) == 0.0 < CHANGE_EPS


def test_frame_from_image_has_gray_pixels_and_a_png():
    im = Image.new("RGB", (40, 80), "red")
    f = Frame.from_image(im)
    assert (f.width, f.height) == (40, 80) and f.image.mode == "L"
    assert f.png.startswith(b"\x89PNG") and Image.open(io.BytesIO(f.png)).size == (40, 80)
    assert dhash(f.image) == dhash(f.png)


def test_look_carries_the_records_header_and_avoid_zones_when_the_store_has_them(eyes, home_png):
    """Round 2 (director): the live director confirms the handle inside `header` and checks flick touch-downs against
    `avoid` without loading the store on the VM. The fields come from lane/screens' schema; absent, nothing is sent."""
    rec = record("instagram.home", home_png)
    rec.header, rec.avoid = (0.0, 0.04, 1.0, 0.17), [(0.0, 0.88, 1.0, 1.0), (0.85, 0.3, 1.0, 0.8)]
    e, frames = eyes([rec], ScriptedOCR())

    async def go():
        producer(frames, [home_png, home_png])
        look = await e.look("udid", platform="instagram")
        s = look.summary()
        assert s["header"] == [0.0, 0.04, 1.0, 0.17] and s["avoid"] == [[0.0, 0.88, 1.0, 1.0], [0.85, 0.3, 1.0, 0.8]]
        assert s["kind"] == "screen"
    asyncio.run(go())


def test_no_live_frame_falls_back_to_a_still_and_says_so(eyes, home_png):
    """The eyes down (no frame in time): one DVT still keeps the caller seeing, journaled as source dvt."""
    e, frames = eyes([record("instagram.home", home_png)], ScriptedOCR())
    e.frame_timeout, e.still = 0.1, None

    async def go():
        with pytest.raises(asyncio.TimeoutError):
            await e.look("udid", platform="instagram")
        stills = []

        async def still(udid):
            stills.append(udid)
            return home_png
        e.still = still
        look = await e.look("udid", platform="instagram")
        assert stills == ["udid"] and look.source == "dvt" and look.screen == "instagram.home" and look.frame is None
        assert look.summary()["source"] == "dvt" and look.frame_age_ms >= 0 and look.settled
    asyncio.run(go())


def test_a_stale_latest_frame_is_never_judged(eyes, home_png):
    """Trial 1's last look judged a 1.9 s old frame while the phone was leaving USB. A latest frame older than `stale`
    waits for the next one; when none comes, the still fallback (or the timeout) speaks, not the old picture."""
    e, frames = eyes([record("instagram.home", home_png)], ScriptedOCR())
    e.frame_timeout, e.stale = 0.2, 0.05

    async def go():
        frames.push(*bgra(home_png))
        await asyncio.sleep(0.15)  # the only frame is now stale
        with pytest.raises(asyncio.TimeoutError):
            await e.look("udid", platform="instagram")
        producer(frames, [home_png, home_png], gap=0.03)  # fresh frames arrive: judged as usual
        look = await e.look("udid", platform="instagram")
        assert look.source == "stream" and look.screen == "instagram.home" and look.frame.seq >= 2
    asyncio.run(go())


def test_the_fast_pass_runs_without_language_correction_and_the_accurate_pass_with_the_default(eyes, home_png):
    e, frames = eyes([HOME_OCR], ocr := ScriptedOCR(accurate_items=[item("Instagram", 300, 90), item("Your story", 70, 300)]))

    async def go():
        producer(frames, [home_png, home_png])
        await e.look("udid", platform="instagram")
        assert ocr.levels == ["fast", "accurate"] and ocr.corrections == [False, None]
    asyncio.run(go())


def test_an_old_frame_from_a_live_source_is_the_screen_now(eyes, home_png):
    """A static screen may send no new frames. When the source vouches for itself (the capture: device connected,
    session running), its old frame is judged at once instead of waiting for a frame that never comes."""
    e, frames = eyes([record("instagram.home", home_png)], ScriptedOCR())
    e.frame_timeout, e.stale = 0.5, 0.05
    frames.alive = lambda: True

    async def go():
        for _ in range(fast.WARM_FRAMES):  # a warm capture, then it goes quiet (the screen holds still)
            frames.push(*bgra(home_png))
        await asyncio.sleep(0.15)
        t = time.monotonic()
        look = await e.look("udid", platform="instagram")
        assert look.source == "stream" and look.screen == "instagram.home" and look.frame.seq == fast.WARM_FRAMES
        assert time.monotonic() - t < 1.0  # no frame wait (a 3 s timeout), the still window is already over
        frames.alive = lambda: False  # the phone left USB: the old frame is not the screen now
        with pytest.raises(asyncio.TimeoutError):
            await e.look("udid", platform="instagram")
    asyncio.run(go())


def test_the_still_window_runs_from_the_frames_arrival(eyes, home_png):
    """A latest frame already `still_after` old has shown the screen holding still: settle returns it at once."""
    e, frames = eyes([record("instagram.home", home_png)], ScriptedOCR())
    e.still_after = 0.15

    async def go():
        for _ in range(fast.WARM_FRAMES):  # a warm source that then sends nothing
            frames.push(*bgra(home_png))
        await asyncio.sleep(0.2)
        look = await e.look("udid", platform="instagram")
        assert look.settled and look.content["by"] in ("still window", "frames agree") and look.frame.seq == fast.WARM_FRAMES
        assert look.ms["settle"] < 140  # never the 150 ms window again, let alone the 2 s timeout
    asyncio.run(go())


def test_chrome_bands_judge_like_the_whole_frame_diff(home_png):
    """R1 on chrome bands cut from the BGRA rows agrees with the old whole-frame ¼-scale diff."""
    from phonebridge.screens.fast import SETTLE_CHROME, bands_changed, chrome_bands
    sf = lambda png, seq: stream.StreamFrame(seq, time.monotonic(), time.time(), *_wh_bgra(png))
    same = synth_frame(top="pink", tab="black", body="white")
    body = synth_frame(top="pink", tab="black", body="gray")  # only the content changed
    tab = synth_frame(top="pink", tab="white", body="white")  # the chrome changed
    a = sf(same, 1)
    for png, moved in ((same, False), (body, False), (tab, True)):
        b = sf(png, 2)
        new = bands_changed(chrome_bands(a, SETTLE_CHROME), chrome_bands(b, SETTLE_CHROME))
        old = changed(_gray(a.gray), _gray(b.gray), chrome=SETTLE_CHROME, scale=1)
        assert (new >= CHANGE_EPS) == moved == (old >= CHANGE_EPS) and abs(new - old) < 0.02


def _wh_bgra(png):
    raw, w, h = bgra(png)
    return w, h, raw


# --- content-aware settle (R1-IG attempt 2, 08:19Z: settled while the feed coasted; controller-verify's seam) --------

W_, H_ = 1170, 2532


def _page(seed=7, height=7000):
    """A tall feed-like page: blocks and text-ish bars of many greys, full width (posts span the screen)."""
    import random
    from PIL import ImageDraw
    rnd = random.Random(seed)
    im = Image.new("L", (W_, height), 235)
    d = ImageDraw.Draw(im)
    y = 0
    while y < height:
        h = rnd.randint(40, 260)
        d.rectangle((0, y, W_, y + h), fill=rnd.randint(20, 230))
        for _ in range(rnd.randint(2, 8)):
            x0 = rnd.randint(0, W_ - 200); yy = rnd.randint(y, y + h)
            d.rectangle((x0, yy, x0 + rnd.randint(40, 400), yy + rnd.randint(6, 24)), fill=rnd.randint(0, 255))
        y += h
    return im


def _screen(page, offset, video=None):
    """The viewport at `offset` px down the page, with fixed chrome bands; `video` = a seed for a noise block."""
    import random
    view = page.crop((0, offset, W_, offset + H_)).copy()
    view.paste(180, (0, 0, W_, int(0.12 * H_)))
    view.paste(60, (0, int(0.88 * H_), W_, H_))
    if video is not None:
        rnd = random.Random(video)
        noise = Image.frombytes("L", (W_ // 8, 120), bytes(rnd.randrange(256) for _ in range(W_ // 8 * 120))).resize((W_, int(0.3 * H_)))
        view.paste(noise, (0, int(0.3 * H_)))
    return view.convert("RGBA").tobytes("raw", "BGRA"), W_, H_


def _sf(seq, raw):
    return stream.StreamFrame(seq, time.monotonic(), time.time(), raw[1], raw[2], raw[0])


def test_content_shift_sees_a_scroll_but_not_a_video_or_a_still_screen():
    page = _page()
    a = fast.content_strip(_sf(1, _screen(page, 1000)))
    assert fast.content_shift(a, fast.content_strip(_sf(2, _screen(page, 1000))))[0] == 0  # the same screen
    shift = fast.content_shift(a, fast.content_strip(_sf(3, _screen(page, 1040))))[0]
    assert 32 <= shift <= 48  # scrolled 40 px
    assert abs(fast.content_shift(a, fast.content_strip(_sf(4, _screen(page, 1600))))[0] - 600) <= 8  # a fast flick
    v1 = fast.content_strip(_sf(5, _screen(page, 1000, video=1)))
    assert fast.content_shift(v1, fast.content_strip(_sf(6, _screen(page, 1000, video=2))))[0] == 0  # a video plays


def _eyes_on(frames, **kw):
    e = fast.Eyes(Store(records=[record("instagram.home", synth_frame(top="pink", tab="black", body="white"))], corpus=Path("/nonexistent")),
                  lambda udid: _ready(frames), ocr=ScriptedOCR(), **kw)
    return e


async def _ready(frames):
    return frames


def test_a_look_during_a_coast_never_settles_on_moving_content_and_returns_fast():
    """A glide's coast: the chrome holds still while the content decelerates. The look must not settle on a frame
    the content then moves on from (chrome-only settle did, at once), and must not sit out the coast's tail either:
    after `content_wait` it returns settled false, "content moving" (R1 attempt 4)."""
    page = _page()
    offsets, v, off = [], 40.0, 500.0
    while v >= 1.0:
        offsets.append(round(off)); off += v; v *= 0.9
    offsets += [round(off)] * 30
    raws = [_screen(page, o) for o in offsets]
    frames = stream.Frames()
    e = _eyes_on(frames)

    async def go():
        async def produce():
            for r in raws:
                frames.push(*r)
                await asyncio.sleep(1 / 60)
        prod = asyncio.ensure_future(produce())
        while frames.stats["frames"] < 10:  # mid-coast, with history
            await asyncio.sleep(0.002)
        t = time.monotonic()
        r = await e._settle(frames)
        took = time.monotonic() - t
        prod.cancel()
        return r, took
    (sf, settled, seen, info), took = asyncio.run(go())
    assert not settled or offsets[-1] - offsets[sf.seq - 1] <= 12, (
        f"settled at offset {offsets[sf.seq - 1]} while the content went on to {offsets[-1]}")
    assert settled or info["by"] == "content moving"  # returned by the content wait, not by sitting out the coast
    assert took < e.settle_timeout  # wall clock is loose on purpose: the producer shares the loop (VM load)


ATTEMPT4 = DATA_DIR / "evidence/M3-director/r1-instagram-20261019T101241Z/strip.mp4"


@pytest.mark.skipif(not ATTEMPT4.exists(), reason="R1-IG attempt 4's strip is not on this host")
def test_attempt_4s_look_after_a_flick_returns_fast_as_content_moving():
    """R1-IG attempt 4 (10:13:01Z): the look after a hard flick sat out the coast's 2 s tail (read as a playing video
    at first; the recording shows the whole page shifting, 848 → 8 px per 100 ms). Replayed from its strip: the look
    returns within content_wait, settled false, with a real page shift (a coast, not a video). The frames are decoded
    first and pushed on the clip's own timestamps, so a loaded VM's decoder can't starve the settle (controller-verify)."""
    av = pytest.importorskip("av")
    clip = []
    with av.open(str(ATTEMPT4)) as c:
        s = c.streams.video[0]
        c.seek(int(8.4 / s.time_base), stream=s)
        for k, f in enumerate(c.decode(s)):
            t = float(f.pts * s.time_base)
            if t > 9.7:
                break
            if t >= 8.8 and k % 2 == 0:  # ~30 fps: enough for a 100 ms content window, a third of the memory
                im = f.to_image().convert("RGBA")
                clip.append((t, im.tobytes("raw", "BGRA"), im.width, im.height))
    frames = stream.Frames()
    e = _eyes_on(frames)

    async def go():
        async def produce():
            t0, c0 = time.monotonic(), clip[0][0]
            for t, raw, w, h in clip:
                await asyncio.sleep(max(0.0, (t - c0) - (time.monotonic() - t0)))
                frames.push(raw, w, h)
        prod = asyncio.ensure_future(produce())
        while frames.stats["frames"] < 6:  # ~200 ms of history
            await asyncio.sleep(0.005)
        t = time.monotonic()
        r = await e._settle(frames)
        took = time.monotonic() - t
        prod.cancel()
        return r, took
    (sf, settled, seen, info), took = asyncio.run(go())
    assert not settled and info["by"] == "content moving" and info["shift_px"] > 0, info  # not the timeout
    assert took < e.settle_timeout


def test_a_still_screen_settles_from_recent_frames_at_once():
    page = _page()
    raw = _screen(page, 900)
    frames = stream.Frames()
    e = _eyes_on(frames)

    async def go():
        for _ in range(12):
            frames.push(*raw)
            await asyncio.sleep(1 / 60)
        return await e._settle(frames)
    sf, settled, seen, info = asyncio.run(go())
    assert settled and seen == 1 and info["by"] == "frames agree" and info["shift_px"] == 0  # no frame waited for


def test_a_playing_video_does_not_hold_settle():
    page = _page()
    raws = [_screen(page, 900, video=i) for i in range(30)]
    frames = stream.Frames()
    e = _eyes_on(frames)

    async def go():
        async def produce():
            for r in raws:
                frames.push(*r)
                await asyncio.sleep(1 / 60)
        prod = asyncio.ensure_future(produce())
        while frames.stats["frames"] < 10:
            await asyncio.sleep(0.002)
        sf, settled, seen, info = await e._settle(frames)
        prod.cancel()
        return settled, seen, info
    settled, seen, info = asyncio.run(go())
    assert settled and info["by"] == "frames agree" and seen <= 3  # the video's change is no shift: not motion


def test_settle_modes():
    assert fast.settle_mode(True) == fast.settle_mode("content") == "content"
    assert fast.settle_mode("chrome") == "chrome" and fast.settle_mode(False) is None
    with pytest.raises(ValueError):
        fast.settle_mode("maybe")


def test_a_repeated_frame_after_a_stall_does_not_settle_a_moving_page():
    """The phone repeats a frame now and then mid-coast (R1 strip 11.134 s); if delivery stalls between the two, the
    newest frame `content_window` older is the duplicate. Settle needs two consecutive frames still, so one pair of
    duplicates can't settle a page that keeps moving."""
    page = _page()
    frames = stream.Frames()
    e = _eyes_on(frames, settle_timeout=0.9)
    offs = [600 + 20 * i for i in range(12)]
    offs.insert(8, offs[7])  # a repeated frame
    gaps = [1 / 60] * len(offs)
    gaps[8] = 0.13  # delivery stalls before the repeat arrives

    async def go():
        async def produce():
            for o, g in zip(offs, gaps):
                await asyncio.sleep(g)
                frames.push(*_screen(page, o))
        prod = asyncio.ensure_future(produce())
        while frames.stats["frames"] < 6:
            await asyncio.sleep(0.002)
        r = await e._settle(frames)
        await prod
        return r
    sf, settled, seen, info = asyncio.run(go())
    assert not settled, f"settled on frame {sf.seq} (offset {offs[sf.seq - 1]}) by {info.get('by')}"
