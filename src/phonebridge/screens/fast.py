"""The fast look (PLAN §5.2 with E1 passed): R0 a frame of the live stream, R1 settle on consecutive frames,
then the cascade with the fast OCR level, and the accurate level only when that misses.

Why OCR still runs on a recognised screen: stops, popups and alerts are OCR-anchored (a sheet over a known
screen leaves the chrome, so its dHash still matches the screen underneath); Vision's fast level reads a
popup's words in ~100 ms where the accurate level takes ~220 ms (p50) and more. Everything here runs inside
the daemon on the Mac: no PNG crosses the tunnel for a look, only the verdict and its timings.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .. import ocr as vision
from PIL import Image, ImageChops, ImageStat

from ..stream import Frames, StreamFrame
from .cascade import CHANGE_EPS, Cascade, Frame, Match

SETTLE_CHROME = [(0.0, 0.04, 1.0, 0.12), (0.0, 0.88, 1.0, 1.0)]  # below the status bar (its clock ticks)
SETTLE_TIMEOUT_S = 2.0  # a feed that never stops moving still gets a look (journaled as not settled)
STILL_AFTER_S = 0.15  # no newer frame for this long = the screen holds still (a static screen gives the encoder nothing)
FRAME_TIMEOUT_S = 3.0  # no frame at all in this long: the stream is down (the caller falls back to DVT)
CONTENT_ROWS = (0.12, 0.88)  # between the chrome bands: what a feed scrolls
CONTENT_STRIPS = ((0.0, 0.2), (0.8, 1.0))  # its left and right edges: a scroll moves them, a playing video mostly doesn't
CONTENT_WINDOW_S = 0.1  # content is still when a frame matches one this much older (a 4 px/frame creep is 24 px here)
CONTENT_NOISE = 1.5  # mean grey difference (0-255, ¼ scale) below which the content strips are the same
CONTENT_WAIT_S = 0.3  # content still moving after this long: the look returns, settled false, "content moving" (a coast's
                      # tail runs ~2 s on Instagram; the caller, which made the glide, decides whether to wait: R1 attempt 4)
WARM_FRAMES = 30  # a capture that has sent fewer is still starting (sparse frames): its silence proves nothing
STALE_S = 1.0  # a "latest" frame older than this is not the screen now (the source stalled or the phone left): never judge it


@dataclass
class Look:
    screen: str | None
    layer: str
    score: float
    settled: bool
    frames_seen: int
    frame: StreamFrame | None
    frame_age_ms: float
    ocr_levels: list[str]
    ms: dict[str, float]
    items: list[dict] = field(default_factory=list, repr=False)
    match: Match | None = field(default=None, repr=False)
    pixels: Frame | None = field(default=None, repr=False)
    stream: dict = field(default_factory=dict)  # the stream's counters at the verdict: fps, frames, dropped_aus, pli, ...
    kind: str | None = None  # screen | popup | alert | stop: a controller stops on anything but a screen
    zones: dict = field(default_factory=dict)  # the record's header / avoid regions ([x0, y0, x1, y1] fractions), when it has them
    source: str = "stream"  # stream (a live frame) | dvt (no live frame came: a still, journaled as the fallback it is)
    age_at_select_ms: float = 0.0  # the frame's age when settle chose it (before OCR/cascade); the rest of frame_age_ms is judging time
    content: dict = field(default_factory=dict)  # settle's content check: mode, shift_px, diff, window_ms, by

    def summary(self) -> dict:
        return {"screen": self.screen, "kind": self.kind, **self.zones, "layer": self.layer, "score": round(self.score, 3), "settled": self.settled,
                "frames_seen": self.frames_seen, "frame_seq": self.frame.seq if self.frame else None,
                "frame_age_ms": round(self.frame_age_ms, 1), "age_at_select_ms": round(self.age_at_select_ms, 1),
                "source": self.source, "ocr": self.ocr_levels, "ms": self.ms,
                "detail": self.match.detail if self.match else None, "stream": self.stream, "content": self.content}


def chrome_bands(f: StreamFrame, chrome) -> list[Image.Image]:
    """R1's view of a frame: each chrome region, gray at ¼ scale, cut from the BGRA rows before any conversion (a
    fifth of the pixels; converting the whole frame first cost ~10 ms per frame on the M5 Pro, 20 ms per settle)."""
    row, out = f.width * 4, []
    for x0, y0, x1, y1 in chrome:
        top, bot = int(y0 * f.height), int(y1 * f.height)
        band = Image.frombuffer("RGBA", (f.width, bot - top), memoryview(f.bgra)[top * row:bot * row], "raw", "BGRA", 0, 1)
        if (x0, x1) != (0.0, 1.0):
            band = band.crop((int(x0 * f.width), 0, int(x1 * f.width), bot - top))
        out.append(band.convert("L").reduce(4))
    return out


def bands_changed(a: list[Image.Image], b: list[Image.Image], threshold: int = 40) -> float:
    """`changed()` over chrome bands: the fraction of their pixels that differ by more than `threshold`."""
    diff = total = 0
    for x, y in zip(a, b):
        hist = ImageChops.difference(x, y).point(lambda v: 255 if v > threshold else 0).histogram()
        total += sum(hist)
        diff += hist[255]
    return diff / total if total else 0.0


def content_strip(f: StreamFrame) -> Image.Image:
    """The content area's left and right edge strips, side by side, gray at ¼ scale (cut from the BGRA rows)."""
    row, top, bot = f.width * 4, int(CONTENT_ROWS[0] * f.height), int(CONTENT_ROWS[1] * f.height)
    band = Image.frombuffer("RGBA", (f.width, bot - top), memoryview(f.bgra)[top * row:bot * row], "raw", "BGRA", 0, 1)
    parts = [band.crop((int(x0 * f.width), 0, int(x1 * f.width), bot - top)).convert("L").reduce(4) for x0, x1 in CONTENT_STRIPS]
    out, x = Image.new("L", (sum(p.width for p in parts), parts[0].height)), 0
    for p in parts:
        out.paste(p, (x, 0))
        x += p.width
    return out


def _shift_err(a: Image.Image, b: Image.Image, d: int) -> float:
    w, h = a.size
    pa, pb = ((0, d, w, h), (0, 0, w, h - d)) if d >= 0 else ((0, 0, w, h + d), (0, -d, w, h))
    return ImageStat.Stat(ImageChops.difference(a.crop(pa), b.crop(pb))).mean[0]


def content_shift(a: Image.Image, b: Image.Image, noise: float = CONTENT_NOISE) -> tuple[int, float, float]:
    """How far the content moved vertically from strip `a` to strip `b`, in screenshot px (0 = it didn't), with the
    difference at no shift and at the best shift. Moved = the strips differ AND one vertical shift explains most of
    it (a scroll); a change no shift explains (a playing video, an animation) is not motion. Coarse search over half
    the strip at 1/16 scale, refined at ¼."""
    e0 = _shift_err(a, b, 0)
    if e0 <= noise:
        return 0, e0, e0
    ca, cb = a.reduce(4), b.reduce(4)
    half = ca.height // 2
    dc = min((d for d in range(-half, half + 1) if d), key=lambda d: _shift_err(ca, cb, d))
    d = min((d for d in range(4 * dc - 4, 4 * dc + 5) if d), key=lambda d: _shift_err(a, b, d))
    eb = _shift_err(a, b, d)
    return (d * 4 if eb < 0.5 * e0 else 0), e0, eb


def settle_mode(settle: bool | str) -> str | None:
    """look(settle=…): True or "content" (default: chrome and content still), "chrome" (chrome bands only, the
    pre-2026-10-19 settle), False or "none" (the latest frame)."""
    if settle in (True, "content"):
        return "content"
    if settle == "chrome":
        return "chrome"
    if settle in (False, None, "none"):
        return None
    raise ValueError(f"settle={settle!r}: true | 'content' | 'chrome' | false")


def _source_alive(frames: Frames) -> bool:
    try:
        return bool(frames.alive and frames.alive())
    except Exception:  # a source that can't answer can't vouch
        return False


class Eyes:
    """`await Eyes(store, frames_of).look(udid, platform=..., expect=[...])`: one recognised look, timed per layer."""

    def __init__(self, store, frames_of: Callable[[str], Awaitable[Frames]], ocr=vision.recognize, templates_dir=None,
                 eps: float = CHANGE_EPS, chrome=None, settle_timeout: float = SETTLE_TIMEOUT_S,
                 still_after: float = STILL_AFTER_S, frame_timeout: float = FRAME_TIMEOUT_S,
                 still: Callable[[str], Awaitable[bytes]] | None = None, stale: float = STALE_S):
        self.store, self.frames_of, self.ocr, self.still = store, frames_of, ocr, still  # still: a DVT PNG when no frame comes
        self.stale = stale
        self.cascade = Cascade(store, templates_dir)
        self.eps, self.chrome = eps, chrome or SETTLE_CHROME
        self.settle_timeout, self.still_after, self.frame_timeout = settle_timeout, still_after, frame_timeout
        self.content_window, self.content_wait = CONTENT_WINDOW_S, CONTENT_WAIT_S

    async def fresh(self, frames: Frames) -> StreamFrame:
        """The latest frame, unless it is older than `stale` and its source can't vouch for it: then the next one (a
        stalled source or a phone that left USB must not pass off an old picture as the screen now; trial 1's last
        verdict sat on a 1.9 s frame). A source that says it is alive (the capture: device connected, session
        running) vouches for an old frame: a static screen may send no new ones, and it is still the screen now."""
        f = await frames.wait(timeout=self.frame_timeout)
        if f.age_ms() > self.stale * 1000 and not _source_alive(frames):
            f = await frames.wait(after=f.seq, timeout=self.frame_timeout)
        return f

    async def settle(self, frames: Frames, mode: str = "content") -> tuple[StreamFrame, bool, int]:
        frame, settled, seen, _ = await self._settle(frames, mode)
        return frame, settled, seen

    async def _settle(self, frames: Frames, mode: str = "content") -> tuple[StreamFrame, bool, int, dict]:
        """The first frame whose chrome matches the frame before it (R1) and, in "content" mode, whose content strips
        show no vertical motion against a frame `content_window` older (a coasting feed keeps its chrome still: R1-IG
        attempt 2 settled 0.9 s before the coast ended). Recent frames count, so a still screen settles at once.
        Content motion is waited out for at most `content_wait`: then the look returns, settled false, "content
        moving" (R1 attempt 4: two looks after glides sat out a coast's 2 s tail). Chrome motion (a transition) is
        waited out until the timeout. Silence (no newer frame for `still_after`) counts as still only from a warm source that showed no motion in
        this settle: a cold capture's sparse first frames are not a still screen. The last frame, unsettled, at the
        timeout."""
        prev = await self.fresh(frames)
        started = time.monotonic()
        deadline = started + self.settle_timeout
        known = [f for f in frames.recent() if f.seq < prev.seq][-8:] + [prev]  # oldest first
        seen, moved, info = 1, False, {"mode": mode}
        bands_of, strips_of = {}, {}
        bands = lambda f: bands_of[f.seq] if f.seq in bands_of else bands_of.setdefault(f.seq, chrome_bands(f, self.chrome))
        strip = lambda f: strips_of[f.seq] if f.seq in strips_of else strips_of.setdefault(f.seq, content_strip(f))
        verdicts: dict[int, bool | None] = {}

        def content_still(i: int) -> bool | None:
            """known[i] shows no vertical content motion against the newest frame `content_window` older; None when
            no frame is that old yet."""
            f = known[i]
            if f.seq not in verdicts:
                base = next((g for g in reversed(known[:i]) if f.t - g.t >= self.content_window), None)
                if base is None:
                    return None
                shift, e0, eb = content_shift(strip(base), strip(f))
                if f is cur:
                    info.update(shift_px=shift, diff=round(e0, 2), window_ms=round((f.t - base.t) * 1000))
                verdicts[f.seq] = not shift
            return verdicts[f.seq]

        cur = prev
        while True:
            if len(known) >= 2:
                still = bands_changed(bands(known[-2]), bands(cur)) < self.eps
                moved |= not still
                if still and mode == "content":
                    # this frame and the one before it each match a frame `content_window` older: one check alone can
                    # pair a frame with its own duplicate (the phone repeats frames mid-coast; strip 11.134 s) when
                    # delivery stalls, and call a moving page still
                    a, b = content_still(len(known) - 1), content_still(len(known) - 2)
                    if a is False or b is False:
                        still, moved = False, True
                        if time.monotonic() - started >= self.content_wait:  # the chrome is still; only the page moves
                            return cur, False, seen, {**info, "by": "content moving"}
                    elif a is None or b is None:
                        still = False  # not enough history yet: wait for more frames (this is not motion)
                if still:
                    return cur, True, seen, {**info, "by": "frames agree"}
            now = time.monotonic()
            if now >= deadline:
                return cur, False, seen, {**info, "by": "timeout"}
            silence = not moved and cur.seq >= WARM_FRAMES
            # the still window runs from the frame's arrival, not from this call: a frame already `still_after` old
            # has shown the screen holding still (trial 1: two quiet-screen looks paid the full 150 ms)
            wait = (self.still_after - cur.age_ms() / 1000) if silence else (deadline - now)
            try:
                nxt = await frames.wait(after=cur.seq, timeout=max(0.001, min(wait, deadline - now)))
            except asyncio.TimeoutError:
                if silence and time.monotonic() < deadline:
                    return cur, True, seen, {**info, "by": "still window"}
                return cur, False, seen, {**info, "by": "timeout"}
            known = (known + [nxt])[-24:]
            cur, seen = nxt, seen + 1

    def _ocr(self, pixels, fast: bool) -> list[dict]:
        if fast:
            try:
                return self.ocr(pixels, True, correct=False)
            except TypeError:  # an OCR callable without the option (tests, older backends)
                pass
        return self.ocr(pixels, fast)

    async def acquire(self, udid: str, settle: bool | str = True) -> tuple[Frame, StreamFrame | None, str, bool, int, float, dict, dict]:
        """The pixels to judge: a live frame (settled, or the latest) as (frame, stream_frame, "stream", settled, seen,
        age_at_select_ms, stream_stats); when no live frame comes in time and a `still` source exists, a DVT still as
        "dvt". The age here is the frame's age when it was chosen: the still window (no newer frame for
        `still_after`) is settle time, and what accrues after this point is judging time."""
        try:
            frames = await self.frames_of(udid)
            mode = settle_mode(settle)
            sf, settled, seen, info = await self._settle(frames, mode) if mode else (await self.fresh(frames), True, 1, {"mode": None})
            return Frame.from_image(sf.rgba, ts=sf.wall), sf, "stream", settled, seen, sf.age_ms(), dict(frames.stats), info
        except (asyncio.TimeoutError, RuntimeError):
            if self.still is None:
                raise
            t = time.time()
            png = await self.still(udid)  # the eyes are down: one still keeps the caller seeing, and says so
            return Frame.from_png(png), None, "dvt", True, 1, (time.time() - t) * 1000, {}, {"mode": "dvt"}

    async def look(self, udid: str, platform: str | None = None, expect: list[str] | None = None,
                   bundle: str | None = None, settle: bool | str = True) -> Look:
        t0 = time.perf_counter()
        frame, sf, source, settled, seen, age_at_select, stats, content = await self.acquire(udid, settle)
        t1 = time.perf_counter()
        levels, m, t_ocr, t_pix = [], None, 0.0, 0.0
        for fast in (True, False):
            levels.append("fast" if fast else "accurate")
            t = time.perf_counter()
            # the fast pass runs without language correction: on 59 labelled corpus frames on the M5 Pro it recognised
            # 53 vs 52 with it, at 13 vs 21 ms median, 20 vs 38 ms p95 (fast-eyes 2026-10-19); the accurate pass keeps it
            frame.ocr = await asyncio.to_thread(self._ocr, frame.pixels, fast)
            t_ocr += time.perf_counter() - t
            t = time.perf_counter()
            m = self.cascade.recognise(frame, expect=expect, platform=platform, bundle=bundle)
            t_pix += time.perf_counter() - t
            if not m.unknown:
                break
        ms = {"settle": round((t1 - t0) * 1000, 1), "ocr": round(t_ocr * 1000, 1), "pixels": round(t_pix * 1000, 1),
              "total": round((time.perf_counter() - t0) * 1000, 1)}
        rec = self.store.records.get(m.screen_id) if m.screen_id else None
        zones = {k: [list(r) for r in z] if isinstance(z, (list, tuple)) and z and isinstance(z[0], (list, tuple)) else
                 (list(z) if z else None) for k in ("header", "avoid") if (z := getattr(rec, k, None)) is not None}
        age_ms = sf.age_ms() if sf is not None else age_at_select + (time.perf_counter() - t1) * 1000
        return Look(m.screen_id, m.layer, m.score, settled, seen, sf, age_ms, levels, ms, frame.ocr, m, frame, stats,
                    rec.kind if rec else None, zones, source, age_at_select, content)
