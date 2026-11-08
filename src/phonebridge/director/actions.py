"""Writes the director may do when Ian allows them (01:2xZ: likes at a low rate, clearly on-niche, within caps, each verified
on the next look). Each action: do it, then look again and verify (BEH-08); return the verdict; the director journals it and
stops on an unverified write."""

from __future__ import annotations

import io


def red_share(png: bytes, x: float, y: float, rx: float = 40, ry: float = 36) -> float:
    """Share of red pixels in a box around (x, y): Instagram's liked heart is solid red (engage.py's check)."""
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = im.size
    box = (max(0, int(x - rx)), max(0, int(y - ry)), min(w, int(x + rx)), min(h, int(y + ry)))
    px = list(im.crop(box).getdata())
    if not px:
        return 0.0
    return sum(1 for r, g, b in px if r > 170 and g < 90 and b < 110) / len(px)


def heart_at(items: list[dict], w: float, h: float) -> tuple[float, float] | None:
    """The heart icon: just left of the like count, the first number in the action row under the media."""
    import re

    counts = [i for i in items if re.fullmatch(r"[\d.,]+[kKmM]?", i.get("text", "").strip()) and i.get("x", 0) < 0.3 * w and 0.15 * h < i.get("y", 0) < 0.9 * h]
    if not counts:
        return None
    c = min(counts, key=lambda i: i.get("x", 0))
    return c["x"] - c.get("w", 60) / 2 - 0.035 * w, c["y"]  # OCR boxes are centres: the heart is just left of the like count, on its row


def save_at(items: list[dict], w: float, h: float) -> tuple[float, float] | None:
    """The bookmark at the right end of the action row of the topmost post on screen (the aligned one): the row is the
    like count's line (the first count left of 0.3 w); the icon's centre at 0.938 w (surfaces4). The generic fallback the sim
    uses: live, feedmaps' pixel resolver (ig_save_icon) comes first and the save target is unproven (kinds.UNPROVEN_LIVE)."""
    import re

    counts = [i for i in items if re.fullmatch(r"[\d.,]+[kKmM]?", i.get("text", "").strip()) and i.get("x", 0) < 0.3 * w and 0.15 * h < i.get("y", 0) < 0.9 * h]
    if not counts:
        return None
    return 0.938 * w, min(counts, key=lambda i: i.get("y", 0))["y"]


def saved_state(png: bytes, y: float, w: float, h: float) -> bool:
    """Is the bookmark on the row at `y` (px) filled? Its inner box (centre 0.938 w) is solid bright where an outline's is dark
    (dark appearance; feedmaps.ig_saved_state's reading, the generic one until it is wired)."""
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("L")
    cx = int(0.938 * im.width)
    raw = im.crop((cx - 12, int(y) - 10, cx + 12, int(y) + 14)).tobytes()
    return bool(raw) and sum(1 for v in raw if v > 180) / len(raw) > 0.6
