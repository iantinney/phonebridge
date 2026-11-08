"""surfaces3: pixel `inks` anchors (TikTok feed read without OCR), x.home.tabs, the IG Reels viewer's back target."""
from PIL import Image, ImageDraw

from phonebridge.screens.cascade import Cascade, Frame, ink_share, score
from phonebridge.screens.schema import MatchSpec, SchemaError
from phonebridge.screens.store import RECORD_DIRS, Store
import pytest

STORE = Store(records_dirs=RECORD_DIRS)


def tiktok_bar(home_lit=True, profile_lit=False) -> Image.Image:
    """A synthetic TikTok bottom bar over a dark video: white Home icon, the cyan/red '+', optionally a lit Profile."""
    im = Image.new("RGB", (1170, 2532), (30, 30, 30))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 2280, 1170, 2532), fill=(0, 0, 0))
    if home_lit:
        d.rectangle((60, 2345, 140, 2405), fill=(255, 255, 255))
    if profile_lit:
        d.rectangle((1000, 2345, 1080, 2405), fill=(255, 255, 255))
    d.rectangle((420, 2330, 560, 2410), fill=(30, 220, 230))
    d.rectangle((560, 2330, 740, 2410), fill=(250, 40, 90))
    return im


def test_ink_share_reads_colour_and_whiteness():
    f = Frame.from_image(tiktok_bar())
    assert ink_share(f, "bright", (0.04, 0.915, 0.14, 0.95)) > 0.15
    assert ink_share(f, "bright", (0.84, 0.915, 0.94, 0.95)) == 0
    assert ink_share(f, "cyan", (0.33, 0.905, 0.67, 0.96)) > 0.004
    assert ink_share(f, "red", (0.33, 0.905, 0.67, 0.96)) > 0.004


def test_inks_schema_rejects_unknown_kind():
    with pytest.raises(SchemaError):
        MatchSpec.from_dict({"inks": [{"name": "x", "kind": "purple", "region": [0, 0, 1, 1]}]})


@pytest.mark.parametrize("home,profile,words,expected", [
    (True, False, ["For You"], "tiktok.feed"),
    (True, False, [], "tiktok.feed.merged_nav"),  # the nav words garbled: pixels alone
    (False, True, [], None),  # the profile tab lit: not the feed
])
def test_tiktok_feed_from_pixels(home, profile, words, expected):
    ocr = [{"text": w, "x": 600, "y": 300, "w": 100, "h": 40, "conf": 0.9} for w in words]
    f = Frame.from_image(tiktok_bar(home, profile), ocr)
    assert Cascade(STORE).recognise(f, platform="tiktok").screen_id == expected


def test_x_home_tabs_and_compose_plus_never_both():
    tabs = [{"text": "For you", "x": 200, "y": 350, "conf": 0.9}, {"text": "Following", "x": 470, "y": 350, "conf": 0.9}]
    c = Cascade(STORE)
    assert c.recognise(Frame(ocr=tabs), platform="x").screen_id == "x.home.tabs"
    assert score(STORE.records["x.home.tabs"], Frame(ocr=tabs + [{"text": "+", "x": 1050, "y": 2300, "conf": 0.9}])).ok is False


def test_reel_viewer_names_its_back_target():
    r = STORE.records["instagram.screen.instagram_reel_fullscreen_viewer"]
    x0, y0, x1, y1 = r.targets["back"].region
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx((0.0775, 0.085))
