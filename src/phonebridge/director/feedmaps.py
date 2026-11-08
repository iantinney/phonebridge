"""Per-app feed maps as DATA (lane surfaces, 10-19): what working one app's feed needs, in one place per app, measured on
real frames (evidence/surfaces-capture/*, screens corpus). The director owns the renderer that reads it (primitives, session);
nothing here gestures. OCR items are box CENTRES (ocr.py:60-61: x = origin + w/2).

feed: the screens a glide may land on (S3) · home_tab: navigate(home) · glide: next (post by post) | page (snap) ·
read_only: no write kind at all (X, orchestrator 06:1xZ) · kinds: the kinds the app offers (director.kinds.KINDS names, or this
module's X_KINDS) · targets: app-specific resolvers, (items, w, h[, png]) → (x, y, w, h) or None, tried before the generic
ones · back: the back control when the iOS edge swipe does not pop (X ignores it on a profile) · like_state: read a post's
like state from its pixels (write verification)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .kinds import Kind, Step

COUNT = re.compile(r"^[\d.,]+[kKmMbB]?$")
TEMPLATES = (Path(__file__).parent.parent / "screens" / "records" / "templates").resolve()


def _left(i: dict) -> float:
    return i["x"] - i.get("w", 0) / 2


# --- X (10-19 06:26Z/06:34Z frames: corpus/sequences/surfaces_x/{1,2}) --------------------------------------------------------
X_TIME = re.compile(r"(?:@\S*|[•·]|\.\.\.).{0,40}?\b\d{1,2}[smhd]\s*$|[•·]\s*(?:[A-Z][a-z]{2} \d{1,2}(?:, \d{4})?|\d{1,2}/\d{1,2}/\d{2})\s*$")


def lines(items: list[dict], h: float, tol: float = 0.008) -> list[list[dict]]:
    """OCR items grouped into screen lines (same y within `tol` h), each left to right: X splits one author line into
    "Sam Rivera", "@SamRivera", "19h" on some frames and keeps it whole on others."""
    out: list[list[dict]] = []
    for i in sorted(items, key=lambda i: i["y"]):
        if out and abs(i["y"] - out[-1][0]["y"]) < tol * h:
            out[-1].append(i)
        else:
            out.append([i])
    return [sorted(l, key=lambda i: i["x"]) for l in out]


def x_headers(items: list[dict], w: float, h: float, band: tuple[float, float] = (0.12, 0.88)) -> list[dict]:
    """X post header lines (post boundaries): a screen line whose text ends in the handle/bullet + age token
    ("@samplenews•16h", "_sample_• 15h", "@sam... 23h", or split, even without the handle: "Sam Rivera | 1 | 19h"), or a date
    on an older post ("@sample_22 • 9/14/26"). Who-to-follow rows and ads carry neither, so they are never a post. Returns the
    line's first item (left of the avatar margin excluded). `band`: the screen heights searched (fractions of h; the default
    keeps taps off the bars, the director's glide reads the whole content area)."""
    out = []
    for line in lines(items, h):
        line = [i for i in line if i["x"] < 0.88 * w]  # the post's ✕ dismiss / "Ad …" sit in the right column
        text = " ".join(i.get("text", "").strip() for i in line)
        if line and band[0] * h < line[0]["y"] < band[1] * h and _left(line[0]) < 0.6 * w and \
                (X_TIME.search(text) or (len(line) >= 2 and re.fullmatch(r"\d{1,2}[smhd]", line[-1].get("text", "").strip()))):
            out.append({**line[0], "line": line})
    return out


def x_top_bar(items: list[dict], h: float) -> bool:
    """X's top bar (logo + home tabs) is on screen: X hides it on a forward scroll and shows it again on a downward flick."""
    return any(i["y"] < 0.2 * h and re.fullmatch(r"(?:for you|following)(?: \S{1,2})?", i.get("text", "").strip().lower()) for i in items)


def x_post_body(items, w, h):
    """A plain text line of a post whose header is on screen (a tap opens the post): ≥ 3 words, no mention, hashtag, link,
    count or translation note, inside that post (before the next header)."""
    heads = x_headers(items, w, h)
    for k, hd in enumerate(heads):
        stop = heads[k + 1]["y"] if k + 1 < len(heads) else hd["y"] + 0.14 * h
        for i in sorted(items, key=lambda i: i["y"]):
            t = i.get("text", "")
            if hd["y"] + 0.012 * h < i["y"] < min(stop, hd["y"] + 0.14 * h, 0.88 * h) and _left(i) > 0.12 * w and len(t.split()) >= 3 \
                    and not re.search(r"[@#]|https?:|\.com\b|show more|translated from", t, re.I) and not COUNT.match(t.strip()):
                bw = min(i.get("w", 200), 300)
                return (_left(i) + bw / 2, i["y"], bw, i.get("h", 30))
    return None


def x_post_author(items, w, h):
    """The display name: the header line's first item when the line is split ("Sam Rivera"), else the left part of the
    whole line. A tap there opens the author's profile."""
    hd = next(iter(x_headers(items, w, h)), None)
    if hd is None:
        return None
    first = hd["line"][0]
    if len(hd["line"]) > 1 and not first.get("text", "").lstrip().startswith("@"):
        bw = min(first.get("w", 200), 240)
        return (_left(first) + bw / 2, first["y"], bw, first.get("h", 30))
    bw = max(40, min(first.get("w", 200) * 0.25, 160))
    return (_left(first) + bw / 2, first["y"], bw, first.get("h", 30))


def x_pushed(items: list[dict], h: float) -> bool:
    """A profile or post pushed over X's home: "12.9K posts" / "Following … Followers" / the "Post" title."""
    return any((re.fullmatch(r"[\d.,]+[kKmM]? posts", i.get("text", "").strip()) and i["y"] < 0.16 * h)
               or re.search(r"\bfollowing\b.*\bfollowers\b", i.get("text", ""), re.I)
               or (i.get("text", "").strip() == "Post" and i["y"] < 0.12 * h) for i in items)


def x_screen(items: list[dict], h: float) -> str:
    """player | post | profile | home, from the screen's own words (the records x.media.video / x.post / x.profile /
    x.home say the same with the cascade's anchors)."""
    bottom = [i.get("text", "").strip() for i in items if i["y"] > 0.85 * h]
    if "1x" in bottom and any(re.fullmatch(r"-?\d{1,2}:\d{2}", t) for t in bottom):
        return "player"
    if any(i.get("text", "").strip() == "Post your reply" and i["y"] > 0.8 * h for i in items):
        return "post"
    return "profile" if x_pushed(items, h) else "home"


def x_back_button(items: list[dict], w: float, h: float):
    """X's ← for this screen (bright-pixel centres on the 10-19 frames): the player (0.087 w, 0.289 h), a post (0.051 w,
    0.082 h; only while its "Post" title shows: X hides the bar on a scrolled thread, a small downward nudge brings it
    back), a profile (0.090 w, 0.089 h; pinned in both header states). None: nothing to go back from, or the bar is hidden."""
    s = x_screen(items, h)
    if s == "player":
        return (0.087 * w, 0.289 * h, 0.07 * w, 0.03 * h)
    if s == "post" and any(i.get("text", "").strip() == "Post" and i["y"] < 0.12 * h for i in items):
        return (0.051 * w, 0.082 * h, 0.06 * w, 0.028 * h)
    if s == "profile":
        return (0.09 * w, 0.089 * h, 0.07 * w, 0.03 * h)
    return None


def x_selected_tab(png: bytes, items: list[dict], w: float, h: float) -> str | None:
    """for_you | following | None (tabs hidden): the selected home tab carries a white underline below its label (≈ 1,280
    white px in the label's width × 0.03 h band on the 10-19 frames; the other tab 0), so a tab label alone never decides."""
    import io

    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("L")
    sx, sy = im.width / w, im.height / h
    for name, label in (("for_you", "for you"), ("following", "following")):
        i = next((i for i in items if i["y"] < 0.2 * h and re.fullmatch(re.escape(label) + r"(?: \S{1,2})?", i.get("text", "").strip().lower())), None)
        if i is None:
            continue
        band = im.crop((int(_left(i) * sx), int((i["y"] + i.get("h", 40) / 2 + 8) * sy), int((i["x"] + i.get("w", 0) / 2) * sx), int((i["y"] + 0.03 * h) * sy)))
        if sum(1 for v in band.getdata() if v > 200) > 300:
            return name
    return None


X_VIEWS = re.compile(r"^(?:ı?l?ı?l|ill|Ill|ıll|ılıl)\s*[\d.,]+[KkMm]?$")
X_PINK = (249, 24, 128)  # X's like colour (#F91880): the heart and its count turn pink once liked


def x_action_rows(items: list[dict], w: float, h: float) -> list[float]:
    """The y of every action row on screen, found by its views count ("ılıl 546K", 0.72–0.76 w) or the like count
    (0.54–0.61 w) next to a reply/repost count on the same line."""
    ys = [i["y"] for i in items if X_VIEWS.match(i.get("text", "").strip()) and 0.7 * w < i["x"] < 0.8 * w]
    for i in items:
        t = i.get("text", "").strip().lstrip("•").strip()
        if COUNT.match(t) and 0.53 * w < i["x"] < 0.62 * w and any(abs(j["y"] - i["y"]) < 0.01 * h and j["x"] < 0.4 * w for j in items if j is not i):
            ys.append(i["y"])
    out: list[float] = []
    for y in sorted(ys):
        if not out or y - out[-1] > 0.02 * h:
            out.append(y)
    return out


def x_like_state(png: bytes, y: float, w: float, h: float) -> bool:
    """Liked? Pink pixels around the heart of the action row at `y` (the heart sits at 0.50–0.52 w on timeline rows, the
    opened post's stats row and replies alike; the 10-19 frames, all unliked, read 0 % pink)."""
    import io

    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    sx, sy = im.width / w, im.height / h
    box = im.crop((int(0.47 * w * sx), int((y - 0.012 * h) * sy), int(0.56 * w * sx), int((y + 0.012 * h) * sy)))
    px = list(box.getdata())
    pink = sum(1 for r, g, b in px if r > 200 and g < 90 and b > 90 and abs(r - X_PINK[0]) < 50 and abs(b - X_PINK[2]) < 60)
    return pink / max(1, len(px)) > 0.02


X_HEART_X = 0.526  # the heart's centre on every action row (bright-pixel box 0.505-0.547 w on the 10-19 rows: home, posts, replies)


def x_heart(items, w, h):
    """The like heart of the post the director is on: the topmost action row (x_action_rows) in the content band 0.18-0.82 h
    (clear of the top bar, the reply bar and the tab bar) whose post's header is on screen above it and no other row lies
    between (the director's author guard reads that header); on an opened post (the "Post your reply" bar) the
    topmost row is the post's own stats row, whose header carries no age token. Centre (0.526 w, row y), a 50 px box."""
    heads = [hd["y"] for hd in x_headers(items, w, h, (0.12, 0.88))]
    post = x_screen(items, h) == "post"
    rows = x_action_rows(items, w, h)
    for k, y in enumerate(rows):
        if not 0.18 * h < y < 0.82 * h:
            continue
        above = [hy for hy in heads if hy < y]
        if post or (above and (k == 0 or rows[k - 1] < max(above))):  # no earlier row after the header: it is this post's own row
            return (X_HEART_X * w, y, 50, 50)
    return None


def x_has_post(look: dict, app: str) -> bool:
    w, h = look.get("width") or 1170, look.get("height") or 2532
    return x_post_body(look.get("items") or [], w, h) is not None


def x_has_author(look: dict, app: str) -> bool:
    w, h = look.get("width") or 1170, look.get("height") or 2532
    return x_post_author(look.get("items") or [], w, h) is not None


# Read-only on X: open a post and read it, peek at an author's profile; each leaves by X's ← (`back`), never the edge swipe.
X_KINDS: dict[str, Kind] = {
    "thread": Kind("thread", "detour", x_has_post, (Step("tap:post_body", ("x.post",), 3.0), Step("dwell:read", ("x.post",), 0.0, "read"),
                                                    Step("back", ("*feed",), 3.0)), "feed_back"),
    "profile": Kind("profile", "detour", x_has_author, (Step("tap:post_author", ("x.profile",), 3.0),
                                                        Step("dwell:read", ("x.profile",), 0.0, "read"), Step("back", ("*feed",), 3.0)),
                    "feed_back", ("intentional",)),
}


# --- TikTok (the 10-18 feed frame: corpus/tiktok/tiktok.feed/2026-10-18-v-01) ------------------------------------------------
def rail_counts(items: list[dict], w: float, h: float) -> list[dict]:
    """The right rail's counts, top to bottom: the longest run of ≥ 3 evenly spaced ones (0.06–0.10 h apart). The avatar's
    badge and the music disc read as stray digits ("5" at 0.385 h and 0.789 h on the 10-18 frame) and fall outside the run."""
    cs = sorted((i for i in items if COUNT.match(i.get("text", "").strip()) and i["x"] > 0.85 * w and 0.3 * h < i["y"] < 0.85 * h),
                key=lambda i: i["y"])
    best: list[dict] = []
    run: list[dict] = []
    for c in cs:
        run = run + [c] if run and 0.06 * h <= c["y"] - run[-1]["y"] <= 0.1 * h else [c]
        best = run if len(run) > len(best) else best
    return best if len(best) >= 3 else []


def tiktok_comment_icon(items, w, h, png=None):
    """The comment bubble, 0.028 h above the second rail count (measured: bubble centre (1078, 1391) over the '150' at
    1467), CONFIRMED on the pixels by its three dots (templates/tiktok/comment_icon.png, half scale: bubble 0.97, heart
    0.85, bookmark 0.84, share 0.77 on the 10-18 frame): if OCR dropped the like count the second count is the SAVES one and
    the bookmark is a write. The search stays within ±0.03 h of the predicted slot. No pixels or no bubble → None."""
    cs = rail_counts(items, w, h)
    if not cs or png is None:
        return None
    from ..screens.cascade import find_template
    x, y = cs[1]["x"], cs[1]["y"] - 0.028 * h
    hit = find_template(png, TEMPLATES / "tiktok" / "comment_icon.png",
                        ((x - 0.06 * w) / w, (y - 0.03 * h) / h, min(1.0, (x + 0.06 * w) / w), (y + 0.03 * h) / h), 0.92, scale=2)
    return (hit[0], hit[1], 60, 50) if hit else None


def tiktok_heart(items, w, h, png=None):
    """The like heart on the right rail, 0.029 h above the first rail count (the like count): bright-pixel box 1035–1122 ×
    1155–1233 px, centre (1078, 1194), over '7,139' at 1268 on the 10-18 feed frame, and the same box over '189.1K' / '4,072'
    at 1270 on the 10-19 surfaces_tiktok/1 frames. Only when the comment bubble one rail step below is confirmed on the
    pixels (tiktok_comment_icon): with the like count missing from the OCR the first count is the comment count and the slot
    above it is the bubble. On the bubble's x (the rail's icon column); the box keeps the scatter inside the 87 × 78 px heart."""
    cs = rail_counts(items, w, h)
    bubble = tiktok_comment_icon(items, w, h, png)
    if not cs or bubble is None:
        return None
    return (bubble[0], cs[0]["y"] - 0.029 * h, 60, 54)


TT_AD = re.compile(r"^(ad|sponsored|learn more\b.*|shop now\b.*)$", re.I)


def tiktok_author(items, w, h):
    """The creator line: the first line of the caption block, whose lines all start at TikTok's text margin (left edge
    0.02–0.045 w; video overlay text and tag lines don't). TikTok shows the display name there now ("Jo Parker", 10-19;
    "samplecook" on 10-18). A tap opens the profile; never the rail avatar, whose red '+' FOLLOW badge sits 0.03 h below
    it. Ads ("Ad", "Learn more ›") → None: no advertiser profile."""
    low = [i for i in items if 0.6 * h < i["y"] < 0.87 * h]
    if any(TT_AD.match(i.get("text", "").strip()) for i in low):
        return None
    block = sorted((i for i in low if 0.015 * w <= _left(i) <= 0.045 * w and not re.match(r"^(q )?search\b", i.get("text", "").strip(), re.I)),
                   key=lambda i: i["y"])
    if not block:
        return None
    i = block[0]
    return handle_box(i, 240.0)  # the shared clearance rule: never over a 'Follow' OCR merged into the line


# --- Instagram (the 10-18 home frames: corpus/instagram/instagram.home) -------------------------------------------------------
def ig_action_row(items, w, h) -> list[dict]:
    """One post's action-row counts, left to right on one line, when the first is the like count (starting 0.09–0.14 w,
    right after the heart). OCR can drop a count (v-08 lost '142'), so the order is never trusted past that first one."""
    cs = [i for i in items if COUNT.match(i.get("text", "").strip()) and i["x"] < 0.8 * w and 0.15 * h < i["y"] < 0.9 * h]
    for c in sorted(cs, key=lambda i: i["y"]):
        row = sorted((i for i in cs if abs(i["y"] - c["y"]) < 0.01 * h), key=lambda i: i["x"])
        if row and 0.09 * w <= _left(row[0]) <= 0.14 * w:
            return row
    return []


def ig_comment_icon(items, w, h, png=None):
    """The comment bubble right after the like count (centre 0.053–0.061 w past the count's right edge), confirmed on the
    pixels by the bubble template (templates/instagram/comment_icon.png): with a count missing from the OCR the second
    count is the REPOST one, so position alone is never enough. No pixels or no bubble → None."""
    row = ig_action_row(items, w, h)
    if not row or png is None:
        return None
    from ..screens.cascade import find_template
    x = row[0]["x"] + row[0].get("w", 0) / 2 + 0.057 * w
    y = row[0]["y"]
    hit = find_template(png, TEMPLATES / "instagram" / "comment_icon.png",
                        ((x - 0.06 * w) / w, (y - 0.025 * h) / h, (x + 0.06 * w) / w, (y + 0.025 * h) / h), 0.85)
    return (hit[0], hit[1], 50, 50) if hit else None


IG_HEART_X = 0.0615  # the heart's centre on every action row: bright-pixel box 38–105 px at 1170 w


def ig_heart(items, w, h):
    """The like heart, the action row's first icon at the left margin: centre 0.0615 w, 0.002 h above the like count's centre
    (bright-pixel box 38–105 × count−34…count+25 px on the 10-18 home frames v-01 and v-08 and the surfaces_instagram/1
    frames, the same on all four). Only with the like count where IG puts it (ig_action_row: left edge 0.09–0.14 w, right
    after the heart): a hidden like count or one the OCR dropped leaves the comment count first, further right, and no row.
    The 50 px box keeps the tap's scatter inside the 67 × 59 px heart."""
    row = ig_action_row(items, w, h)
    return (IG_HEART_X * w, row[0]["y"] - 0.002 * h, 50, 50) if row else None


def ig_story_ring(items, w, h):
    """The first friend's ring in the story tray: the avatar centre 0.07 h above the first handle label right of 'Your story'
    (never that tile: its tap opens the camera/composer). The tray exists only unscrolled; no 'Your story' label → None."""
    mine = next((i for i in items if i.get("text", "").strip() == "Your story" and i["y"] < 0.35 * h), None)
    if mine is None:
        return None
    row = [i for i in items if abs(i["y"] - mine["y"]) < 0.01 * h and i["x"] > mine["x"] + 0.12 * w and re.fullmatch(r"[A-Za-z0-9_.]{3,30}", i.get("text", "").strip())]
    if not row:
        return None
    i = min(row, key=lambda i: i["x"])
    return (i["x"], i["y"] - 0.0695 * h, 0.2 * w, 0.1 * h)


SAVE_COLUMN = (0.91, 0.15, 0.97, 0.87)  # the bookmark's column at the action row's right end (centre 0.938 w), below the top band, above the tab bar


def ig_save_icon(items, w, h, png=None):
    """A post's bookmark on its action row, found on the PIXELS (templates/instagram/save_icon.png, the outline glyph): the
    row's counts are unreliable at the fast OCR level (the icons read as '11'), and a save is a write, so no pixels or no
    glyph → None, never a guessed point. The filled (saved) glyph scores under the threshold; see ig_saved_state."""
    if png is None:
        return None
    from ..screens.cascade import find_template
    hit = find_template(png, TEMPLATES / "instagram" / "save_icon.png", SAVE_COLUMN, 0.9)
    return (hit[0], hit[1], 50, 60) if hit else None


def ig_saved_state(png, y: float, w: float, h: float) -> bool:
    """True when the bookmark at row `y` (px) is FILLED: the glyph box's inner third is solid white where the outline's is dark
    (outline 0.24 bright share on the 10-18 frame; a filled glyph reads ~0.9). Same-frame verification of a save."""
    from io import BytesIO
    import numpy as np
    from PIL import Image
    im = (png if isinstance(png, Image.Image) else Image.open(BytesIO(png))).convert("L")
    cx = int(0.938 * im.width)
    a = np.asarray(im.crop((cx - 12, int(y) - 10, cx + 12, int(y) + 14)))
    return bool((a > 180).mean() > 0.6)


FOLLOW_WORD = re.compile(r"follow(ing| back)?\b", re.I)


def handle_box(i: dict, cap: float = 200.0):
    """A tap box over a header line's HANDLE only, (x, y, box_w, box_h) in centre convention. OCR merges 'handle • Follow'
    ('•Follow', '· Following', 'Follow back') into the handle's box (integrate's review): the box covers the handle's own
    characters (its share of the line's width) and box + the daemon's 0.4 scatter ends 12 px before any Follow word; a handle
    too short for that clearance gives None, never a tap that could land on Follow."""
    t, lw = i.get("text", "").strip(), i.get("w", 120)
    if not t:
        return None
    left = i["x"] - lw / 2
    glued = re.search(r"(?<=[a-z0-9_.])Follow", t)  # 'abcFollow': OCR dropped the separator; a capital F starts the button word
    handle = t[:glued.start()] if glued else re.split(r"[\s•·]", t, maxsplit=1)[0]
    if not handle or FOLLOW_WORD.match(handle):
        return None
    m = FOLLOW_WORD.search(t, len(handle))
    limit = left + lw * (m.start() / len(t)) - 12 if m else math.inf
    bw = min(lw * len(handle) / max(1, len(t)), cap, (limit - left) / 1.4)
    if bw < 36:
        return None
    return (left + bw / 2, i["y"], bw, i.get("h", 30))


def ig_author(items, w, h):
    """The author handle in the first post card's header line (instagram.home's header zone, unscrolled): never Follow (the
    shared clearance rule, handle_box)."""
    for i in sorted(items, key=lambda i: i["y"]):
        t = i.get("text", "").strip()
        if 0.26 * h <= i["y"] <= 0.33 * h and 0.08 * w <= _left(i) <= 0.8 * w and re.fullmatch(r"[a-z0-9_.]{3,30}", t.split(" ")[0]):
            return handle_box(i, 200.0)
    return None


# --- the Follow button on another account's profile (surfaces5: the director's follow write kind) ------------------------------
# region = the button's row as frame fractions [x0, y0, x1, y1]; `solid` = what "Follow" (not following) looks like by pixels
FOLLOW_BUTTONS = {
    "tiktok": {"region": (0.06, 0.262, 0.40, 0.298), "solid": "red"},   # measured: 10-19 07:38Z profile frame, button 0.04-0.43 w, 0.258-0.304 h
    "x": {"region": (0.5, 0.04, 1.0, 0.62), "solid": "bright"},        # expanded header: white pill 0.74 w 0.564 h; pinned (scrolled) header: dark pill 0.84 w 0.09 h
    "instagram": {"region": (0.0, 0.26, 0.5, 0.36), "solid": "blue"},  # NOT measured: no real IG other-profile frame yet (OCR leads)
}
FOLLOW_TEXT = {"follow": "not_following", "follow back": "not_following", "following": "following", "friends": "following",
               "requested": "requested"}


def _solid_share(png, region, kind: str) -> float:
    from io import BytesIO
    from PIL import Image
    im = (png if isinstance(png, Image.Image) else Image.open(BytesIO(png))).convert("RGB")
    W, H = im.size
    b = im.crop((int(region[0] * W), int(region[1] * H), int(region[2] * W), int(region[3] * H))).resize((40, 12)).tobytes()
    px = list(zip(b[0::3], b[1::3], b[2::3]))
    hit = {"red": lambda r, g, b: r > 200 and g < 90 and 60 < b < 140,
           "bright": lambda r, g, b: r > 225 and g > 225 and b > 225,
           "blue": lambda r, g, b: r < 80 and 100 < g < 190 and b > 200}[kind]
    return sum(1 for p in px if hit(*p)) / len(px)


def follow_button(app: str, items: list[dict], w: float, h: float):
    """The Follow / Following / Requested button on another account's profile: (x, y, box_w, box_h) in CENTRE convention, from the
    OCR text of the button inside the app's button row (exact words only: 'Who to follow', 'Following' counts and bios never
    match), else None. X's first match is the expanded header's, the topmost otherwise."""
    fb = FOLLOW_BUTTONS.get(app)
    if fb is None:
        return None
    x0, y0, x1, y1 = fb["region"]
    hit = [i for i in items if i.get("text", "").strip().lower() in FOLLOW_TEXT and x0 * w <= i["x"] <= x1 * w and y0 * h <= i["y"] <= y1 * h]
    if not hit:
        return None
    i = min(hit, key=lambda i: i["y"])
    return (i["x"], i["y"], max(i.get("w", 120), 0.2 * w), 0.04 * h)


def x_follow_button(items, w, h):
    """X's follow_button: only on a profile (x_screen), never the home feed's Who-to-follow cards."""
    return follow_button("x", items, w, h) if x_screen(items, h) == "profile" else None


def follow_state(app: str, items: list[dict], png=None, w: float = 1170, h: float = 2532) -> str | None:
    """'not_following' | 'following' | 'requested' | None (unreadable) for the profile on screen, for same-frame verification of a
    follow: the button's own words (Follow/Follow back, Following/Friends, Requested), and — when pixels are given — the solid
    'Follow' fill (TikTok red, X white, IG blue) must agree: words 'Following' over a solid Follow fill is a conflict → None."""
    b = follow_button(app, items, w, h)
    if b is None:
        return None
    word = next(i["text"].strip().lower() for i in items if abs(i["x"] - b[0]) < 1 and abs(i["y"] - b[1]) < 1)
    state = FOLLOW_TEXT[word]
    if png is not None and not (app == "x" and b[1] < 0.2 * h):  # X's pinned (scrolled) header pill is translucent dark with white words: words only
        solid = _solid_share(png, (b[0] / w - 0.1, b[1] / h - 0.015, b[0] / w + 0.1, b[1] / h + 0.015), FOLLOW_BUTTONS[app]["solid"]) > 0.12
        if solid != (state == "not_following"):
            return None
    return state


@dataclass(frozen=True)
class FeedMap:
    app: str
    bundle: str
    feed: tuple[str, ...]
    home_tab: tuple[float, float]
    glide: str
    read_only: bool
    kinds: tuple[str, ...]
    targets: dict[str, Callable] = field(default_factory=dict)
    back: Callable | None = None
    like_state: Callable | None = None
    tabs: dict[str, tuple[float, float]] = field(default_factory=dict)  # tab-bar points the detours tap (fractions); DM and profile never
    saved_state: Callable | None = None  # (png, y_px, w, h) → the post's bookmark is filled (save verification); None: actions.saved_state


FEEDMAPS: dict[str, FeedMap] = {
    "instagram": FeedMap("instagram", "com.burbn.instagram", ("instagram.home", "instagram.reels"), (0.10, 0.929), "next", False,
                         ("expand", "carousel", "tap_media", "comments_peek", "profile", "like", "follow", "comment_profile", "save", "stories", "explore", "reels_tab"),
                         {"comment_icon": ig_comment_icon, "header_handle": ig_author, "heart": ig_heart, "save_icon": ig_save_icon, "story_ring": ig_story_ring,
                          "follow_button": lambda items, w, h: follow_button("instagram", items, w, h)},
                         tabs={"reels_tab": (0.30, 0.935), "search_tab": (0.70, 0.935)},  # iPhone 12 (surfaces4): home, reels, DM, search, profile at 0.1/0.3/0.5/0.7/0.9 w
                         saved_state=ig_saved_state),
    "tiktok": FeedMap("tiktok", "com.zhiliaoapp.musically", ("tiktok.feed", "tiktok.foryou"), (0.10, 0.955), "page", False,
                      ("tap_media", "comments_peek", "profile", "like", "follow"),
                      {"comment_icon": tiktok_comment_icon, "header_handle": tiktok_author, "heart": tiktok_heart,
                       "follow_button": lambda items, w, h: follow_button("tiktok", items, w, h)}),
    "x": FeedMap("x", "com.atebits.Tweetie2", ("x.home",), (0.10, 0.93), "next", True, tuple(X_KINDS),
                 {"post_body": x_post_body, "post_author": x_post_author, "heart": x_heart,
                  "follow_button": x_follow_button}, back=x_back_button, like_state=x_like_state),
}


def kinds_for(app: str) -> dict[str, Kind]:
    """The kinds this app offers: X's own read-only detours, else the director's KINDS the map names."""
    from .kinds import KINDS
    fm = FEEDMAPS[app]
    return {n: X_KINDS[n] if app == "x" else KINDS[n] for n in fm.kinds}
