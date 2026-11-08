"""Synthetic phone frames for the screens tests: a status/top bar, a tab bar and a content area."""
import io
import random

from PIL import Image, ImageDraw

W, H = 1170, 2532
PALETTE = ("pink", "lightblue", "lightgreen", "wheat", "lavender", "khaki", "beige", "mistyrose")  # light chrome


def frame(top="red", tab="blue", body="white", icon=None, seed=0, tab_icons=5) -> bytes:
    im = Image.new("RGB", (W, H), body if body != "noise" else "white")
    d = ImageDraw.Draw(im)
    if body == "noise":
        rnd = random.Random(seed)
        for _ in range(400):
            x, y = rnd.randrange(W), rnd.randrange(int(H * 0.14), int(H * 0.86))
            d.rectangle([x, y, x + rnd.randrange(20, 200), y + rnd.randrange(20, 200)],
                        fill=(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    d.rectangle([0, 0, W, int(H * 0.12)], fill=top)
    d.rectangle([0, int(H * 0.88), W, H], fill=tab)
    for i in range(tab_icons):  # tab-bar glyphs: a flat bar has no gradients for a dHash to see
        cx, cy = int(W * (i + 0.5) / tab_icons), int(H * 0.92)
        d.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill="black" if tab != "black" else "white")
    if icon:
        color, x, y = icon
        d.rectangle([x, y, x + 60, y + 60], fill=color)
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()
