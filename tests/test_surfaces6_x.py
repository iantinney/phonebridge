"""X writes (DECISIONS 646df80): the director's targets on the real 10-19 X frames: `heart` (the like button of the post in
view, on the pixels of the heart glyph) and `follow_button` (profiles only), with the liked / following state reads."""
import io
import json

import pytest
from PIL import Image, ImageDraw

from phonebridge.director import feedmaps as fm
from phonebridge.director import primitives
from phonebridge.screens.store import CORPUS

W, H = 1170, 2532
XSEQ = CORPUS / "sequences" / "surfaces_x"
COASTING = {"1/04-scroll3"}  # the OCR and the png are two instants of a coasting feed (the row is 45 px lower in the png)
real = pytest.mark.skipif(not XSEQ.exists(), reason=f"no X captures at {XSEQ}")


def frames():
    for p in sorted(XSEQ.glob("*/*.ocr.json")):
        yield p.parent.name + "/" + p.name[:-9], json.loads(p.read_text()), p.with_suffix("").with_suffix(".png").read_bytes()


@real
def test_heart_lands_on_the_heart_glyph_of_a_header_post_on_real_frames():
    n = 0
    for name, items, png in frames():
        hit = primitives.feedmap_target("x", "heart", {"items": items}, W, H)
        if hit is None:
            continue
        n += 1
        x, y, bw, bh = hit
        im = Image.open(io.BytesIO(png)).convert("L")
        box = im.crop((int(x - bw / 2), int(y - bh / 2), int(x + bw / 2), int(y + bh / 2)))
        light = sum(1 for v in box.getdata() if v > 90)
        if name not in COASTING:
            assert light > 0.08 * bw * bh, (name, light)   # the glyph's grey outline inside the box (the box is mostly black)
        assert 0.18 * H < y < 0.82 * H and not fm.x_like_state(png, y, W, H), name
    assert n >= 6


@real
def test_heart_belongs_to_a_post_with_its_header_on_screen():
    """No header above the row (a scrolled-off post's tail) → no target; the ones with a header sit below it."""
    for name, items, _ in frames():
        hit = fm.x_heart(items, W, H)
        if hit is None:
            continue
        heads = [hd["y"] for hd in fm.x_headers(items, W, H)]
        assert fm.x_screen(items, H) == "post" or any(hy < hit[1] for hy in heads), name


def test_heart_none_without_an_action_row():
    assert fm.x_heart([{"text": "Home", "x": 100, "y": 2400, "w": 80, "h": 30}], W, H) is None


def test_like_state_reads_the_heart_target_pink():
    hit = (fm.X_HEART_X * W, 1300, 50, 50)
    im = Image.new("RGB", (W, H), (0, 0, 0))
    ImageDraw.Draw(im).ellipse([hit[0] - 22, hit[1] - 22, hit[0] + 22, hit[1] + 22], fill=fm.X_PINK)
    b = io.BytesIO()
    im.save(b, "PNG")
    assert fm.x_like_state(b.getvalue(), hit[1], W, H)


@real
def test_follow_button_only_on_a_profile_and_it_reads_the_state():
    prof = 0
    for name, items, png in frames():
        hit = primitives.feedmap_target("x", "follow_button", {"items": items}, W, H)
        if fm.x_screen(items, H) != "profile":
            assert hit is None, name  # Who-to-follow cards on the feed are never the button
            continue
        if hit is not None:
            prof += 1
            assert fm.follow_state("x", items, png, W, H) == "not_following", name
    assert prof >= 2
