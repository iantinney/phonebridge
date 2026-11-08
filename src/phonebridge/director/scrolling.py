"""Post-by-post scrolling on a feed that doesn't snap (Instagram home), the way a person does it (Ian, trial 1: "scrolling way
too far, multiple at a time"): a drag sized so the NEXT post's header lands at the top of the content area, with the
scroll view's coast accounted for, then an alignment check on the next look and a small corrective nudge when needed.

Layout (iPhone 12, measured on trial-1 frames): status bar + top bar end ≈ 0.075 h; a post = header (avatar, handle, Follow)
≈ 0.045 h, media 1170 px wide (square 0.46 h to 4:5 0.58 h), action row ≈ 0.04 h, caption + date ≈ 0.06 h → 0.62–0.75 h per post.
UIScrollView coasts ≈ 0.5 s × release speed (0.998/ms decay, memo §1); the humanizer releases at ≈ 0.3 × mean speed, so a
drag of d over T seconds travels ≈ d · (1 + 0.15 / T): the drag is sized down by that factor."""

from __future__ import annotations

import random
import re

from .cues import HANDLE, HANDLE_STOP, COUNTER, _norm
from .feedmaps import x_action_rows, x_headers, x_top_bar

CONTENT_TOP = 0.075  # where a post's header sits when the post is "the one on screen"
HEADER_COMPANIONS = ("follow", "suggested for you", "•", "original audio", "verified")
POST_DEFAULT = (0.68, 0.06)  # median post height as a fraction of h, log-sigma, when no next header is visible
RELEASE_FRACTION = 0.3  # the humanizer's release speed over its mean (Touchalytics, humanize.py)
COAST_S = 0.5  # UIScrollView: 0.998/ms → ≈ 0.5 s × release speed (the old smooth model; see `travel` for the measured one)
# The phone's scroll response, measured on 111 text-tracked glides of R1-IG attempts 7, 8 and 10: below ~250 px/s of lift speed
# the page stops with the finger (travel ≈ 0.92 × drag: touch slop); above ~450 px/s it flings, coasting ≈ 1.0 s × the lift
# speed; in between it may or may not fling. The old smooth model (coast = 0.5 s × lift speed) missed both regimes: 88 % of
# attempt 10's landings were misaligned and the learned gain swung 1.2–2.2.
STICK = 0.92
FLING_V = (250.0, 450.0)
COAST_K = 1.0
DRAG_T = (0.32, 0.6)  # a drag, not a flick: a person moving one post takes a third to two thirds of a second
TOL = 0.06  # alignment: the header within this of CONTENT_TOP is "on the post"
# X (the 10-19 surfaces captures): a post's header line rests 60-64 px under its post's top hairline (11 real headers); the
# hairline under the home tab row sits at 405 px, the one under the bare status bar at 141 px once a forward scroll has hidden
# the top bar. (content top, next-post line) per bar state, the line 0.145 h under the top as on Instagram (0.075 → 0.22).
X_TOPS = {True: (0.184, 0.329), False: (0.080, 0.225)}
X_BAND = (0.06, 0.93)  # the glide reads X's headers over the whole content area (the bars hide while scrolled)


def _w(h: float) -> float:
    return h * 1170 / 2532  # the iPhone's aspect: X's detectors use the width only for their column bounds


def layout(items: list[dict], h: float, app: str | None = None) -> tuple[float, float]:
    """(content top, next-post line) as fractions of h: where a post's header rests when the post is "the one on screen",
    and the line below which a header is the next post's rather than the current one's. On X both move with its top bar."""
    return X_TOPS[x_top_bar(items, h)] if app == "x" else (CONTENT_TOP, 0.22)


def header_ys(items: list[dict], h: float, app: str | None = None) -> list[float]:
    """y of every post header line visible: a handle-shaped token with a header companion on the same or the next line; on
    X its own header line (feedmaps.x_headers: the status bar's back link, the tab row and Who-to-follow rows are none)."""
    if app == "x":
        return [float(i["y"]) for i in x_headers(items, _w(h), h, X_BAND)]
    lines = sorted((i for i in items if i.get("text")), key=lambda i: i.get("y", 0))
    out = []
    for k, i in enumerate(lines):
        t = _norm(i["text"]).replace("•", " • ")
        toks = t.split()
        if not any(HANDLE.match(tok.rstrip(",:.")) and HANDLE.match(tok.rstrip(",:.")).group(1) not in HANDLE_STOP and not COUNTER.match(tok) and sum(c.isalpha() for c in tok) >= 3 for tok in toks):
            continue
        nearby = " ".join([t] + [_norm(lines[j]["text"]) for j in (k + 1, k + 2) if j < len(lines) and 0 <= lines[j].get("y", 0) - i.get("y", 0) < 0.05 * h])
        if any(c in nearby for c in HEADER_COMPANIONS) and 0.03 * h < i.get("y", 0) < 0.93 * h:
            out.append(float(i.get("y", 0)))
    # a caption line right above a header row borrows that row's companion ("Follow"); posts are far taller than 0.05 h, so of
    # two candidates that close, the lower one is the header
    return [y for k, y in enumerate(out) if not any(0 < y2 - y < 0.05 * h for y2 in out[k + 1:])]


def _handle_token(tok: str) -> bool:
    m = HANDLE.match(tok.rstrip(",:."))
    return bool(m) and m.group(1) not in HANDLE_STOP and not COUNTER.match(tok) and sum(c.isalpha() for c in tok) >= 3


def header_items(items: list[dict], h: float, app: str | None = None) -> list[dict]:
    """The OCR boxes that ARE post header lines: a handle-shaped token (the same test as header_ys, so a separate "Follow"
    box on the same row is not one), top to bottom. On X the line's first box carrying the whole line's text: X splits one
    header line differently from frame to frame ("Sam Rivera • @SamRivera • 18h" | "Sam Rivera", "@SamRivera • 18h")."""
    if app == "x":
        return [{**i, "text": " ".join(j.get("text", "").strip() for j in i["line"])} for i in x_headers(items, _w(h), h, X_BAND)]
    ys = set(header_ys(items, h))
    return sorted((i for i in items if i.get("text") and float(i.get("y", 0)) in ys
                   and any(_handle_token(t) for t in _norm(i["text"]).replace("•", " • ").split()) and "follow" not in _norm(i["text"]).split()),
                  key=lambda i: i.get("y", 0))


TIME_AGO = re.compile(r"^(\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago|yesterday|just now)$", re.I)
NEXT_AFTER_TIME = 0.05  # the next post's header line sits about this far (fraction of h) below the post's "N days ago" line


def next_post_distance(items: list[dict], h: float, rng: random.Random, app: str | None = None) -> tuple[float, str]:
    """(distance in px to bring the next post's header to the content top, how it was found): the next header when it is on
    screen; else the current post's last line, its "N days ago" line (X: its action row; the next header follows just below
    it, 0.047-0.051 h on X's 10-19 frames); else a typical post height."""
    top, nxt = layout(items, h, app)
    ys = [y for y in header_ys(items, h, app) if y > nxt * h]  # below the current post's own header
    if ys:
        d = ys[0] - top * h
        return max(0.3 * h, min(0.95 * h, d)), "next header" if d >= 0.3 * h else "next header, min step"  # a min step passes it on purpose
    last = x_action_rows(items, _w(h), h) if app == "x" else [i.get("y", 0) for i in items if TIME_AGO.match(_norm(i.get("text", "")).strip())]
    stamps = sorted(y for y in last if nxt * h < y < 0.88 * h)
    if stamps:
        return max(0.3 * h, min(0.95 * h, stamps[0] + NEXT_AFTER_TIME * h - top * h)), "after the post's action row" if app == "x" else "after the post's time line"
    med, sig = POST_DEFAULT
    return max(0.4 * h, min(0.9 * h, med * h * rng.lognormvariate(0, sig))), "default post height"


def travel(drag: float, T: float, release: float, rng: random.Random | None = None) -> float:
    """The page's travel for a drag of `drag` px over T s lifting at `release` × its mean speed (the measured model above).
    Inside the ambiguous band the fling happens with a probability rising across it (rng given), else counts as half."""
    v = abs(release * drag / max(T, 1e-3))
    if v <= FLING_V[0]:
        fling = 0.0
    elif v >= FLING_V[1]:
        fling = 1.0
    else:
        x = (v - FLING_V[0]) / (FLING_V[1] - FLING_V[0])
        fling = float(rng.random() < x) if rng is not None else x
    return STICK * drag + fling * COAST_K * v * (1 if drag >= 0 else -1)


def plan_drag(target: float, T: float, release: float) -> tuple[float, float]:
    """(drag px, release) that travels `target` px under `travel`, keeping the lift speed out of the band where the phone may
    or may not fling: a short contact lifts fast enough to fling for sure, a long one lifts slowly enough not to."""
    sign, target = (1 if target >= 0 else -1), abs(target)
    fl = target / (STICK + COAST_K * release / T)  # a flick: the page coasts
    if release * fl / T >= FLING_V[1]:
        return sign * fl, release
    st = target / STICK  # a placed drag: it stops with the finger
    if release * st / T <= FLING_V[0]:
        return sign * st, release
    hi = FLING_V[1] * 1.05
    if T < 0.25 and target > hi * COAST_K:  # a flick: lift faster (rel·fl/T ≥ hi ⇔ rel ≥ hi·T·STICK / (target − hi·K))
        need = hi * T * STICK / (target - hi * COAST_K)
        if need <= 0.6:
            rel = max(release, need)
            return sign * target / (STICK + COAST_K * rel / T), rel
    return sign * st, min(release, 0.95 * FLING_V[0] * T / st)  # a placed drag: lift slower


def alignment(items: list[dict], h: float, app: str | None = None) -> tuple[bool, float | None]:
    """After a scroll: is a post header sitting at the content top? (aligned, offset in h of the nearest header)."""
    ys = header_ys(items, h, app)
    if not ys:
        return True, None  # no header visible: a tall post fills the screen, nothing to correct
    top = layout(items, h, app)[0]
    off = min((y / h - top for y in ys), key=abs)
    return abs(off) <= TOL, off


def post_items(items: list[dict], h: float, app: str | None = None) -> list[dict]:
    """The judged post's own lines on a half-framed rest: from its header (the one nearest the content top, as `alignment`
    judges) down to the next header (X: through its action row, its last line; an ad or a Who-to-follow module below has no
    header); without headers, everything. Corrections are occasional (people read half-framed posts), so the previous post's
    caption tail and the next post's lines would otherwise be read as this post's."""
    ys = sorted(header_ys(items, h, app))
    if not ys:
        return items
    top = layout(items, h, app)[0]
    y0 = min(ys, key=lambda y: abs(y / h - top))
    y1 = next((y for y in ys if y > y0 + 0.01 * h), None)
    if app == "x":
        y1 = min([y + 0.02 * h for y in x_action_rows(items, _w(h), h) if y > y0] + ([y1] if y1 is not None else []), default=None)
    return [i for i in items if i.get("y", 0) >= y0 - 0.01 * h and (y1 is None or i.get("y", 0) < y1 - 0.01 * h)]


def nudge_for(offset_h: float, h: float) -> float:
    """Signed drag (px, positive = scroll on) that brings the header back to the content top; small and human: a partial scroll."""
    return offset_h * h
