"""The renderers (BEHAVIOUR_MODEL §1): `glide` (closed loop: plan → one gesture → the next look measures the travel → a
per-session gain → alignment → one nudge), `navigate(target)` (a tap/drag on a target resolved from the look, never a gesture
in an avoid zone), the contract `walk` for engage/detour/write kinds (journaled `kind, step, expect`, ending `kind_done,
verified`), and `until(pred, timeout)` — the only wait besides dwell. Every gesture is look-gated by the director (S3) and
checked against the look's `avoid` zones (S2). `glide_path` keeps the plan's length, duration and release exactly; the travel is
what the phone actually did, so the HID timestamp fix (fast-eyes dc6676c) or a new iOS is learned, not assumed."""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

from . import scrolling as sc
from .kinds import KINDS, Kind, allowed_screens
from .params import Params

CONTENT_TOP = sc.CONTENT_TOP
COUNT = re.compile(r"^[\d.,]+[kKmM]?$")


@dataclass
class Glide:
    kind: str
    target_px: float  # the travel wanted (page px)
    drag_px: float  # what the finger will move
    duration_s: float
    release: float  # release speed / mean speed
    gain: float  # the per-session gain the planner assumed (measured / predicted travel)
    pred_px: float  # the travel the planner predicts for this drag (drag × coast × gain)
    how: str
    tail: str | None = None  # a drawn tail component (slow_drag), journaled


@dataclass
class GlideState:
    """Per-session memory of the gain (measured travel / planned travel) — learned, never fitted on old data."""
    gain: float = 1.0
    n: int = 0
    history: list = field(default_factory=list)

    def overshot(self, step: float = 1.15) -> None:
        """The target header left over the top and nothing remains to measure: the phone travels further than assumed.
        Raise the gain estimate by a bounded step; measured scrolls take over once they land short enough to read."""
        self.gain = min(5.0, self.gain * step)
        self.history.append(f"overshoot→{round(self.gain, 3)}")

    def update(self, planned: float, measured: float | None) -> None:
        """`planned` is the planner's prediction (drag × coast × the gain it used). The phone's own gain is then
        gain × measured / planned, and the running mean converges to it (dividing by the scaled prediction alone converges
        to √gain: naturalness audit, controller-verify test_closed_loop)."""
        if measured is None or planned <= 0 or measured <= 0:
            return  # the tracked header left the screen: no update (reviewer, round 3)
        g = self.gain * measured / planned
        self.n += 1
        self.gain = self.gain + (g - self.gain) / min(self.n, 5)  # a short running mean: adapts in a few glides
        self.history.append(round(g, 3))


def plan_glide(kind: str, items: list[dict], w: float, h: float, p: Params, rng: random.Random, state: GlideState, header=None, app: str | None = None) -> Glide:
    """The gesture for one glide kind from the look: `next` targets the next post's header (sc), `partial` a fraction of a
    post, `back` one post back, `page` a snap flick (TikTok/Reels)."""
    if kind == "page":
        d, _ = p.glide["flick_s"].draw(rng)
        return Glide("page", 0.6 * h, 0.55 * h, d, 0.3, 1.0, 0.6 * h, "snap page", None)
    if kind == "next":
        target, how = sc.next_post_distance(items, h, rng, app)
    elif kind.startswith("skim"):  # skim:N passes N posts: the next header, then N-1 typical post heights; the nudge corrects the landing
        n = int(kind.split(":")[1])
        first, _ = sc.next_post_distance(items, h, rng, app)
        target, how = first + (n - 1) * sc.POST_DEFAULT[0] * h * rng.lognormvariate(0, sc.POST_DEFAULT[1]), f"skim {n} posts"
    elif kind == "partial":
        target, how = rng.uniform(0.15, 0.3) * h, "partial"
    else:  # back
        target, how = -rng.uniform(0.5, 0.75) * h, "back one post"
    T, tcomp = p.glide["drag_s"].draw(rng)
    release, _ = p.glide["release_frac"].draw(rng)
    # a short contact is a flick that lifts at speed and lets the page coast; a long drag places the post and lifts almost
    # still (Touchalytics: stroke duration and lift speed go together)
    release = min(0.6, release * min(2.0, (0.2 / T) ** 0.7))
    # the phone's measured response (sc.travel: stops with the finger below ~250 px/s, flings ≈ 1.0 s × lift speed above ~450);
    # the lift is kept out of the ambiguous band, and the learned gain corrects what remains
    page, release = sc.plan_drag(target, T, release)  # in page px: the phone flings on the page's speed, the gain maps finger → page
    drag = page / state.gain
    return Glide(kind, target, drag, T, release, state.gain, sc.travel(drag * state.gain, T, release), how, "slow_drag" if tcomp.tail else None)


def header_y(items: list[dict], h: float, app: str | None = None) -> float | None:
    """y of the post header nearest the content top (for travel measurement and alignment)."""
    ys, top = sc.header_ys(items, h, app), sc.layout(items, h, app)[0]
    return min(ys, key=lambda y: abs(y - top * h)) if ys else None


GENERIC = re.compile(r"^(follow(ing)?|suggested for you|sponsored|see translation|original audio|view all|liked by|add a comment|"
                     r"\d+\s*(s|m|h|d|w|sec|min|hour|day|week|month|year)s?(\s+ago)?|(yesterday|just now)|\d{1,2}:\d{2})\b", re.I)  # on every post: not evidence of which post


def next_heads(items: list[dict], h: float, app: str | None = None) -> list[dict]:
    """The header lines below the current post's own (sc.layout's next-post line): the next post's first."""
    nxt = sc.layout(items, h, app)[1]
    return [i for i in sc.header_items(items, h, app) if i.get("y", 0) > nxt * h]


def target_header_y(before: list[dict], h: float, app: str | None = None) -> float | None:
    """The y of the next post's header line before the glide (the glide's target), or None."""
    heads = next_heads(before, h, app)
    return heads[0]["y"] if heads else None


def tracked_left(before: list[dict], after: list[dict], h: float, app: str | None = None) -> bool:
    """Did the next post's header (the scroll's target) leave the screen? On a forward scroll that means it went over the
    top: an overshoot, even when nothing is left to measure the travel by. (Matched against the after look's boxes and its
    header lines: X's header line is one box on one frame and split on the next.)"""
    import difflib

    heads = next_heads(before, h, app)
    if not heads:
        return False
    t = heads[0]["text"].lower()
    return not any(difflib.SequenceMatcher(None, t, (i.get("text") or "").lower()).ratio() >= 0.8 for i in after + sc.header_items(after, h, app))


def _unique_lines(items: list[dict]) -> dict:
    seen: dict = {}
    for i in items:
        t = (i.get("text") or "").strip()
        if len(t) >= 5 and not GENERIC.search(t):
            seen.setdefault(t, []).append(i)
    return {t: v[0] for t, v in seen.items() if len(v) == 1}


def text_shifts(before: list[dict], after: list[dict]) -> list[float]:
    """y_before − y_after for every non-generic line that appears exactly once in both frames."""
    a, b = _unique_lines(before), _unique_lines(after)
    return [a[t]["y"] - b[t]["y"] for t in a.keys() & b.keys()]


def text_shift(before: list[dict], after: list[dict]) -> float | None:
    """The page's movement between two looks (median of the shared lines' shifts); None when no line is shared (it moved
    a long way, or the screen changed)."""
    import statistics

    d = text_shifts(before, after)
    return float(statistics.median(d)) if d else None


def measure_travel(before: list[dict], after: list[dict], h: float, planned: float) -> float | None:
    """How far the page moved, from text that is on screen both before and after: every OCR line that appears exactly
    once in each frame (a handle, a caption, a count) gives y_before − y_after; the median of the shifts that agree with
    the scroll's direction and rough size is the travel. The tracked post's header may leave the screen on an overshoot;
    its caption or action row usually stays, so overshoots are measured too (they had been invisible to the gain). A
    different post where the next one was expected shares no text: None (naturalness audit)."""
    import statistics

    shifts = [d for d in text_shifts(before, after) if (d > 0) == (planned > 0) and 0.25 * abs(planned) <= abs(d) <= 2.5 * abs(planned)]
    if not shifts:
        return None
    m = statistics.median(shifts)
    agree = [d for d in shifts if abs(d - m) <= 0.03 * h]
    return float(statistics.median(agree)) if agree else None


def alignment(items: list[dict], h: float, app: str | None = None) -> tuple[str, float | None]:
    """aligned | misaligned | unknown (no header visible), with the offset in h of the nearest header."""
    ok, off = sc.alignment(items, h, app)
    if off is None:
        return "unknown", None
    return ("aligned" if ok else "misaligned"), off


POLL_S = 0.25  # between looks inside `until`


async def until(look_fn, pred, timeout_s: float, reaction_s: float, sleep, now=time.monotonic, abort=None) -> dict | None:
    """Wait for the look to satisfy `pred` within `timeout_s` on the director's clock (`now`), having first waited the drawn
    reaction time; at most timeout/POLL_S + 1 looks, each bounded by the time left (Codex review #14: the real clock spun
    an instant simulator, and a hanging look was never cut). The only wait besides dwell. Returns the look, or None (also
    at once when a look that fails `pred` satisfies `abort`: an unrecognised screen is never looked at twice in a row)."""
    import asyncio

    await sleep(reaction_s)
    t0 = now()
    for _ in range(int(timeout_s / POLL_S) + 1):
        left = timeout_s - (now() - t0)
        try:
            meta = await asyncio.wait_for(look_fn(), timeout=max(1.0, left + 1.0))
        except asyncio.TimeoutError:
            return None
        if pred(meta):
            return meta
        if abort is not None and abort(meta):
            return None
        if now() - t0 >= timeout_s:
            return None
        await sleep(POLL_S)
    return None


def _feedmap_resolver(app: str | None, target: str):
    if not app:
        return None
    try:
        from .feedmaps import FEEDMAPS
    except Exception:
        return None
    fm = FEEDMAPS.get(app)
    return (fm.targets or {}).get(target) if fm else None


def needs_pixels(app: str | None, target: str) -> bool:
    """Does surfaces' resolver for this target confirm it on the pixels (TikTok's rail icons, IG's comment bubble)?"""
    fn = _feedmap_resolver(app, target)
    return fn is not None and fn.__code__.co_argcount >= 4


def feedmap_target(app: str | None, target: str, look: dict, w: float, h: float):
    """surfaces' measured resolver for this app and target (feedmaps.FEEDMAPS[app].targets), tried before the generic one:
    (x, y, box_w, box_h) in CENTRE convention, or None. Resolvers that need the pixels take the look's frame."""
    fn = _feedmap_resolver(app, target)
    if fn is None:
        return None
    items = look.get("items") or []
    try:
        return fn(items, w, h, look.get("png")) if fn.__code__.co_argcount >= 4 else fn(items, w, h)
    except Exception:
        return None


MEASURED_ONLY = ("save_icon", "heart", "follow_button")  # write targets: where an app has a measured resolver, only it decides


def resolve_target(target: str, look: dict, w: float, h: float, rng: random.Random, app: str | None = None) -> tuple[str, float, float, float, float] | None:
    """A target name → (gesture, x, y, x2, y2) from the look's items/zones: 'more', 'comment_icon', 'media', 'media_double',
    'media_left' (carousel drag), 'header_handle', 'sheet_down', 'heart' (the like button: surfaces' measured resolver only,
    never a generic guess). None when the target isn't on screen."""
    from .kinds import caption_more

    fm = feedmap_target(app, target, look, w, h)
    if fm is not None:  # a tap on the measured target; (x2, y2) carry its box so the daemon's scatter stays inside it
        return ("tap", fm[0], fm[1], fm[2], fm[3])
    if target in MEASURED_ONLY and _feedmap_resolver(app, target) is not None:
        return None  # a write's target: the measured resolver or nothing, never a guessed point (integrate's review: saves)
    items = look.get("items") or []  # OCR boxes: x, y are CENTRES (ocr.py), w, h the box size
    if target == "more":  # the word "more" ends the truncated caption line: tap just inside the box's right edge
        i = next((i for i in items if caption_more(i)), None)
        return ("tap", i["x"] + i.get("w", 100) / 2 - min(30, i.get("w", 100) / 4), i["y"], 0, 0) if i else None
    if target == "comment_icon":  # the icon sits just left of the comment count in the action row
        c = next((i for i in items if COUNT.match(i.get("text", "").strip()) and 0.2 * w < i.get("x", 0) < 0.45 * w and 0.15 * h < i.get("y", 0) < 0.9 * h), None)
        return ("tap", c["x"] - c.get("w", 60) / 2 - 0.03 * w, c["y"], 0, 0) if c else None
    if target in ("media", "media_double", "media_left"):
        from .kinds import CAROUSEL_COUNTER

        counter = next((i for i in items if CAROUSEL_COUNTER.match(i.get("text") or "")), None) if target == "media_left" else None
        band = look.get("media") or media_band(items, w, h, counter["y"] if counter else None, app)
        if band is None:
            return None  # no clear media area on screen (between the post's header row and its action row): no tap
    if target == "media_left":  # a carousel card swipe across the middle of the media band
        y = h * rng.uniform(band[1] + 0.35 * (band[3] - band[1]), band[3] - 0.35 * (band[3] - band[1]))
        return ("drag", w * rng.uniform(0.68, 0.8), y, w * rng.uniform(0.2, 0.32), y)
    if target in ("media", "media_double"):
        for _ in range(8):  # a point on the media clear of any OCR text (an overlay's "Tap to unmute", a button, a prompt): S2
            x = w * rng.uniform(band[0] + 0.15, band[2] - 0.15)
            y = h * rng.uniform(band[1] + 0.05, band[3] - 0.05)
            if clear_of_items(x, y, items):
                return ("double_tap" if target == "media_double" else "tap", x, y, 0, 0)
        return None
    if target == "header_handle":
        hz = look.get("header")
        heads = sc.header_items(items, h, app)  # the boxes that are header lines; the look's header zone, when given, picks the post on screen
        cand = [i for i in heads if not hz or (hz[1] * h <= i.get("y", 0) <= hz[3] * h)] or heads
        if not cand:
            return None
        from .feedmaps import handle_box

        b = handle_box(min(cand, key=lambda i: i.get("y", 0)))  # the shared clearance rule: the handle only, never a merged Follow
        return ("tap", b[0], b[1], b[2], b[3]) if b else None
    if target == "sheet_down":  # only from the grabber the screen record names (surfaces verifies it per app); never a guessed point
        g = (look.get("targets") or {}).get("sheet_grabber")
        return ("drag", g[0] * w, g[1] * h, g[0] * w, min(0.92, g[1] + 0.45) * h) if g else None
    if target.endswith("_tab"):  # the tab bar (outside the feed's avoid zones by design): the store's point, else the app's declared one
        t = (look.get("targets") or {}).get(target) or _tab(app, target)
        return ("tab", t[0] * w, t[1] * h, 0.06 * w, 0.03 * h) if t else None
    if target == "story_ring":  # a ring in the tray (never "Your story", never a live)
        from .kinds import story_rings

        rings = story_rings(items, w, h)
        if not rings:
            return None
        x, y = rng.choice(rings[:3])
        return ("tap", x, y, 0.1 * w, 0.04 * h)
    if target == "grid_tile" and (look.get("targets") or {}).get("grid_tile"):  # the record's tile (surfaces4: 46/46 real frames)
        t = look["targets"]["grid_tile"]
        return ("tap", t[0] * w, t[1] * h, 0.2 * w, 0.08 * h)
    if target in ("next_zone", "grid_tile"):  # the story's next zone (surfaces4 proposal; the left goes back) | a tile; clear of text (stickers, links)
        for _ in range(8):
            if target == "next_zone":
                x, y = w * rng.uniform(0.88, 0.96), h * rng.uniform(0.38, 0.55)
            else:
                x, y = w * (rng.choice((1, 3, 5)) / 6 + rng.uniform(-0.08, 0.08)), h * rng.uniform(0.3, 0.75)
            if clear_of_items(x, y, items):
                return ("tap", x, y, 0, 0)
        return None
    if target in ("close", "grid_up", "reel_next"):  # close the story viewer (a slow swipe down 0.38 → 0.85 h) | scroll the grid | the next reel
        x = w * rng.uniform(0.35, 0.6)
        if target == "close":
            return ("close", x, h * rng.uniform(0.37, 0.39), x + w * rng.uniform(-0.02, 0.02), h * rng.uniform(0.84, 0.85))
        y1 = h * rng.uniform(0.66, 0.74)
        return ("flick" if target == "reel_next" else "drag", x, y1, x + w * rng.uniform(-0.02, 0.02), y1 - h * rng.uniform(0.32, 0.42))
    if target == "comment_handle":  # the first comment's handle, on its first characters (their profile); else the record's point
        from .kinds import comment_handles

        rows = comment_handles({**look, "width": w, "height": h})
        if rows:
            i = rows[0]
            t = (i.get("text") or "").strip()
            bw = min(200.0, i.get("w", 200) * len(t.split()[0]) / max(1, len(t)))
            return ("tap", i["x"] - i.get("w", 200) / 2 + bw / 2, i["y"], bw, i.get("h", 36))
        t = (look.get("targets") or {}).get("comment_handle")
        return ("tap", t[0] * w, t[1] * h, 0.12 * w, 0.02 * h) if t else None
    if target == "save_icon":  # the bookmark at the right end of the post's action row
        from .actions import save_at

        at = save_at(items, w, h)
        return ("tap", at[0], at[1], 0.05 * w, 0.025 * h) if at else None
    if target == "back_button":  # the top-left chevron: the store's target when the record names one, else its usual place
        t = (look.get("targets") or {}).get("back")
        return ("tap", (t[0] if t else 0.055) * w, (t[1] if t else 0.07) * h, 0, 0)
    return None


def _tab(app: str | None, target: str):
    """The app's declared tab-bar point for a tab target (feedmaps: home_tab, tabs), or None (a tab that is never a target)."""
    from .feedmaps import FEEDMAPS

    fm = FEEDMAPS.get(app or "")
    return None if fm is None else (fm.home_tab if target == "home_tab" else fm.tabs.get(target))


FULL_MEDIA = [0.0, 0.2, 1.0, 0.65]  # a full-screen video (TikTok, Reels) or a post taller than the screen: the middle is media


def media_band(items: list[dict], w: float, h: float, anchor_y: float | None = None, app: str | None = None) -> list[float] | None:
    """The post's media as fractions [x0, y0, x1, y1], from the post's own rows when the look names no media zone: below its
    header line (or the content top when that header has scrolled away), above its action row (the like/comment counts).
    `anchor_y` picks the post containing that point (a carousel's k/n counter); else the first post on screen. None when no
    media area of at least 0.15 h is clear: a fixed band could put a tap on an avatar or the next post."""
    if app == "tiktok":
        return FULL_MEDIA  # the video fills the screen; its counts sit on the right rail (an avoid zone), never below it
    heads = [i["y"] for i in sc.header_items(items, h, app)]
    counts = sorted(i["y"] for i in items if COUNT.match((i.get("text") or "").strip()) and i.get("x", 0) < 0.45 * w)
    if not heads and not counts:
        return FULL_MEDIA  # no header and no action row on screen: the post's media fills it
    if anchor_y is None:
        anchor_y = (heads[0] + 0.1 * h) if heads else (counts[0] - 0.1 * h if counts else None)
        if anchor_y is None:
            return None
    above = [y for y in heads if y < anchor_y]
    below = [y for y in counts if y > anchor_y]
    top = (max(above) + 0.035 * h) if above else 0.1 * h
    bottom = (min(below) - 0.03 * h) if below else 0.86 * h
    top, bottom = max(top, 0.1 * h), min(bottom, 0.86 * h)
    return [0.0, top / h, 1.0, bottom / h] if bottom - top >= 0.15 * h else None


def clear_of_items(x: float, y: float, items: list[dict], margin: float = 40) -> bool:
    """Is (x, y) outside every OCR item's box (plus a margin)? A tap meant for the media must not land on text, a button or
    an overlay's prompt."""
    for i in items:  # centre-convention boxes
        cx, cy, hw, hh = i.get("x", 0), i.get("y", 0), i.get("w", 0) / 2, i.get("h", 0) / 2
        if abs(x - cx) <= hw + margin and abs(y - cy) <= hh + margin:
            return False
    return True


def in_avoid(look: dict, x: float, y: float, w: float, h: float, margin: float = 0.0) -> bool:
    """Is (x, y), grown by `margin` px (the tap's scatter, a swipe's bow), inside one of the look's avoid zones?"""
    for z in look.get("avoid") or []:
        if len(z) == 4 and z[0] * w - margin <= x <= z[2] * w + margin and z[1] * h - margin <= y <= z[3] * h + margin:
            return True
    return False


# --- the contract walker ------------------------------------------------------------------------------------------------------
def _screen_matches(screen: str | None, items: list[dict], app: str, expect: tuple[str, ...], feed_ok, require_look: bool = False) -> bool:
    if screen is None:  # no look (a sim/replay without eyes): the OCR feed markers can confirm the feed and nothing else; live, never
        return not require_look and any(e.startswith(app) and e in FEED_IDS.get(app, ()) for e in expect) and feed_ok(items)
    return any(screen == e or screen.startswith(e) for e in expect)


FEED_IDS = {"instagram": ("instagram.home", "instagram.reels"), "tiktok": ("tiktok.feed", "tiktok.foryou"), "x": ("x.home",)}


def _unknown(d, meta: dict) -> bool:
    """A look from the eyes that recognised nothing: never looked at twice in a row (controller-verify), never a gesture (S3)."""
    return hasattr(d.phone, "look") and (meta or {}).get("screen") is None


async def _step(d, kind: Kind, n, step, look: dict, items: list[dict], words: int, source: tuple | None = None,
                by: float | None = None) -> tuple[dict, list[dict], bool]:
    """One step of a kind. Before any gesture: a FRESH look, gated (a stop/alert ends the session; the screen must be one
    the previous step led to, or the feed for the first step: Codex review #1; `source` overrides it for a way out of an
    abandoned kind); the target is resolved on that look and must clear the avoid zones with the tap's scatter. Then
    `until` the step's declared screens show; an unrecognised screen ends the wait at once. A dwell lasts its drawn time,
    never past `by`. Returns (look, items, reached)."""
    from .kinds import allowed_screens, screens_of
    from .session import GestureRefused, StaleInput

    w, h = d.size
    p = d.params
    expect = screens_of(step, d.app)
    if source is None:
        source = allowed_screens(kind, n - 1, d.app) if n > 0 else allowed_screens(None, None, d.app)
    act, _, target = step.action.partition(":")
    lands = lambda m: _screen_matches(m.get("screen"), m.get("items") or [], d.app, expect, d.feed_markers, d.require_look)  # noqa: E731
    if act == "dwell":
        await d.sleep(d.dwell_for(step.dwell, words, by))
        return look, items, True
    if act == "until":  # no input: the screens the step names (a toast fading); an unrecognised look ends it
        got = await until(d.look_dict, lands, step.until_s, 0.0, d.sleep, d.now, abort=lambda m: _unknown(d, m))
        if got is None:
            d.ev("kind_timeout", kind=kind.name, step=n, expect=list(expect))
            return look, items, False
        return got, got.get("items") or items, True
    if kind.group == "write" and act == "tap":  # a write: the budget, the tap gap, then the final paired frame with its guard, then the tap
        if not await d.take_budget(kind.name):
            d.ev("kind_skip", kind=kind.name, step=n, why="the pacer refused (no budget)")
            return look, items, False
        await d.tap_gap()  # waited before the final look, so the look is still fresh at the tap (S3)
        fresh = await d.write_look(source)
    else:
        fresh = await d.gate_look(source)
    if fresh is None:
        d.ev("kind_skip", kind=kind.name, step=n, why="the screen changed before the gesture")
        return look, items, False
    look, items = fresh, fresh.get("items") or items
    if act == "leave":  # back only if the steps before left the feed (a reel's viewer); on the feed: no input at all
        if _screen_matches(look.get("screen"), items, d.app, allowed_screens(None, None, d.app), d.feed_markers, d.require_look):
            return look, items, True
        act = "back"
    try:
        fm_back = None
        if act == "back":
            try:
                from .feedmaps import FEEDMAPS

                fm_back = (FEEDMAPS.get(d.app).back if FEEDMAPS.get(d.app) else None)
            except Exception:
                fm_back = None
        if act == "back" and fm_back is not None:  # the app ignores the edge swipe here (X): its own back control (surfaces)
            b = fm_back(look.get("items") or [], w, h)
            if b is None:
                d.ev("kind_timeout", kind=kind.name, step=n, why="no back control on screen")
                return look, items, False
            await d.tap(b[0], b[1], "back control", kind=kind.name, step=n, expect=expect, w=b[2], h=b[3])
        elif act == "back":
            x1, y1, x2, y2, T = d.human.edge_back(w, h)
            await d.swipe(x1, y1, x2, y2, T, "edge back", kind=kind.name, step=n, expect=expect, zone_exempt=True)  # the system edge gesture
            reaction, _ = p.glide["reaction_s"].draw(d.rng)
            got = await until(d.look_dict, lands, step.until_s, reaction, d.sleep, d.now, abort=lambda m: _unknown(d, m))
            if got is not None:
                return got, got.get("items") or items, True
            if _unknown(d, d.last_look):
                d.ev("kind_timeout", kind=kind.name, step=n, expect=list(expect), why="an unrecognised screen after the back")
                return look, items, False
            again = await d.gate_look(source)  # still where the back started: only then the back button (never on a stop or a sheet)
            if again is None or again.get("kind") not in (None, "screen") or (d.require_look and again.get("kind") != "screen"):
                d.ev("kind_timeout", kind=kind.name, step=n, expect=list(expect), why="back did not return and the screen is not an ordinary one")
                return look, items, False
            d.ev("back_fallback", kind=kind.name, step=n, why="the edge swipe did not return; tapping the back button")
            _, bx, by, _, _ = resolve_target("back_button", again, w, h, d.rng, d.app)
            await d.tap(bx, by, "back button", kind=kind.name, step=n, expect=expect)
        if act != "back":
            if look.get("png") is None and needs_pixels(d.app, target):  # the gate look carries no pixels: its own frame (by seq when served)
                look = {**look, "png": await d.pixels(None)}
            res = resolve_target(target, look, w, h, d.rng, d.app)
            if res is None:
                d.ev("kind_skip", kind=kind.name, step=n, why=f"no target {target} on screen")
                return look, items, False
            gesture, x, y, x2, y2 = res
            if gesture in ("tap", "tab"):  # a tab lives in the tab bar, outside the feed's avoid zones by design
                await d.tap(x, y, target, kind=kind.name, step=n, expect=expect, w=x2 or 0, h=y2 or 0, zone_exempt=gesture == "tab")  # a measured target's own box bounds the scatter
            elif gesture == "double_tap":
                await d.tap(x, y, target, kind=kind.name, step=n, expect=expect, double=True)
            else:
                T, _ = p.glide[{"flick": "flick_s", "close": "close_s"}.get(gesture, "nudge_s")].draw(d.rng)
                await d.swipe(x, y, x2, y2, T, target, kind=kind.name, step=n, expect=expect)
    except GestureRefused:
        d.ev("kind_skip", kind=kind.name, step=n, why="the gesture would touch an avoid zone")
        return look, items, False
    except StaleInput:  # zero input sent: the kind is abandoned on this post; the next gesture starts from a fresh look
        d.ev("kind_skip", kind=kind.name, step=n, why="the daemon refused the input as stale")
        return look, items, False
    if step.until_s:
        reaction, _ = p.glide["reaction_s"].draw(d.rng)
        got = await until(d.look_dict, lands, step.until_s, reaction, d.sleep, d.now, abort=lambda m: _unknown(d, m))
        if got is None:
            d.ev("kind_timeout", kind=kind.name, step=n, expect=list(expect), why="an unrecognised screen" if _unknown(d, d.last_look) else None)
            return look, items, False
        if got.get("kind") in ("stop", "alert", "popup", "warning", "challenge"):
            raise_stop(d, got)
        look, items = got, got.get("items") or items
    return look, items, True


def raise_stop(d, look: dict):
    from .session import Stopped

    raise Stopped("stop_screen", f"{look.get('screen') or look.get('kind')} after a gesture")


async def walk(d, kind: Kind, look: dict, items: list[dict], words: int = 0) -> tuple[float, bool]:
    """Render one kind's contract on the director `d`: every gesture journaled `kind, step, expect`, every transition waited
    for with `until`, the end journaled `kind_done, verified`. A step with `repeat` runs `params.depth[repeat]` times (bounded
    by `upto` on the look, e.g. the carousel cards left), its dwell after each (a glance by default). At TURN_BACK_S the
    dwells are cut and the repeats stop, so the remaining steps bring it home within 90 s. Returns (seconds spent, verified)."""
    from .kinds import TURN_BACK_S

    t0 = d.now()
    by = t0 + TURN_BACK_S
    reached = True
    for n, step in enumerate(kind.steps):
        reps = 1
        if step.repeat:
            depth, _ = d.params.depth[step.repeat].draw(d.rng)
            reps = round(depth) if step.upto is None else min(round(depth), step.upto(look))
        for rep in range(reps):
            if step.repeat and d.now() >= by:
                d.ev("turn_back", kind=kind.name, step=n, done=rep, of=reps, spent_s=round(d.now() - t0, 2))
                break
            look, items, reached = await _step(d, kind, n, step, look, items, words, by=by)
            if not reached:
                break
            if step.repeat:
                await d.sleep(d.dwell_for(step.dwell or "glance", 0, by))  # a glance at the card; a story, a reel watched
        if not reached:
            break
    if kind.verify == "feed_back":
        verified = reached and _screen_matches((look or {}).get("screen"), (look or {}).get("items") or items, d.app, FEED_IDS.get(d.app, ()), d.feed_markers, d.require_look)
    elif kind.verify == "none":
        verified = reached
    else:
        verified = None  # heart_red: the `write` row verifies (before/after under the confirmed handle)
    d.ev("kind_done", kind=kind.name, verified=verified if verified is None else bool(verified), spent_s=round(d.now() - t0, 2))
    return d.now() - t0, bool(verified)
