"""SCR-02/03/04, PER-03, DEV-05: the cascade on synthetic frames, the generic iOS alert class, the classifier."""
import pytest

from phonebridge.screens.cascade import Cascade, Frame, changed, dhash, find_template, hamming, system_alert, classify
from phonebridge.screens.schema import Record
from phonebridge.screens.store import Store
from screens_fixtures.synth import frame as synth_frame, W, H


def item(text, x, y, w=60, h=20, conf=1.0):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": conf}


HOME = Record.from_dict({"id": "instagram.home", "platform": "instagram", "bundle": "com.burbn.instagram",
                         "status": "confirmed",
                         "match": {"ocr_any": ["Instagram", "Your story"], "at_least": 2, "ocr_none": ["New post"]},
                         "targets": {"create": {"ocr": "+", "region": [0, 0, 0.3, 0.1]}}})
PICKER = Record.from_dict({"id": "instagram.new_post.picker", "platform": "instagram", "bundle": "com.burbn.instagram",
                           "status": "confirmed",
                           "match": {"ocr_any": ["New post", "Recents", "Next"], "at_least": 2,
                                     "ocr_in": {"Next": [0.6, 0, 1, 0.15]}},
                           "targets": {"next": {"ocr": "Next", "region": [0.6, 0, 1, 0.15]}}})
FINAL = Record.from_dict({"id": "instagram.reel.final", "platform": "instagram", "bundle": "com.burbn.instagram",
                          "status": "confirmed",
                          "match": {"ocr_any": ["New reel", "Write a caption", "Share"], "at_least": 2},
                          "targets": {"share": {"ocr": "Share", "region": [0, 0.8, 1, 1], "irreversible": True}}})


@pytest.fixture
def store(tmp_path):
    return Store(records=[HOME, PICKER, FINAL], corpus=tmp_path / "corpus")


def test_r4_ocr_anchors_match_with_margin(store):
    f = Frame(ocr=[item("Instagram", 300, 90), item("Your story", 70, 300), item("+", 40, 90)], width=W, height=H)
    m = Cascade(store).recognise(f, platform="instagram")
    assert m.screen_id == "instagram.home" and m.layer == "R4" and m.margin > 0


def test_unknown_without_margin_or_anchors(store):
    f = Frame(ocr=[item("Settings", 300, 90)], width=W, height=H)
    assert Cascade(store).recognise(f, platform="instagram").unknown
    # an anchor word in the wrong region does not count (ocr_in)
    f = Frame(ocr=[item("New post", 300, 90), item("Next", 100, 2000)], width=W, height=H)
    assert Cascade(store).recognise(f, platform="instagram").unknown


def test_r2_expectation_resolves_before_whole_store(store):
    f = Frame(ocr=[item("New post", 585, 212), item("Next", 1058, 210), item("Recents", 167, 1560)], width=W, height=H)
    m = Cascade(store).recognise(f, expect=["instagram.new_post.picker"], platform="instagram")
    assert m.screen_id == "instagram.new_post.picker" and m.layer == "R2"


def test_r3_chrome_dhash_nearest_neighbour(tmp_path):
    png = synth_frame(top="pink", tab="lightblue", body="noise", seed=1)
    h = dhash(png, region=(0, 0.88, 1, 1))
    rec = Record.from_dict({"id": "app.x", "platform": "app", "bundle": "b", "status": "confirmed",
                            "match": {"chrome_dhash": {"region": [0, 0.88, 1, 1], "value": f"{h:016x}", "max_bits": 4}}})
    store = Store(records=[rec], corpus=tmp_path)
    other = synth_frame(top="pink", tab="lightblue", body="noise", seed=2)  # same chrome, other content
    m = Cascade(store).recognise(Frame(png=other, ocr=[], width=W, height=H), platform="app")
    assert m.screen_id == "app.x" and m.layer == "R3"
    far = synth_frame(top="lightgreen", tab="white", body="noise", seed=3, tab_icons=2)
    assert Cascade(store).recognise(Frame(png=far, ocr=[], width=W, height=H), platform="app").unknown
    assert hamming(h, dhash(other, region=(0, 0.88, 1, 1))) <= 4


def test_r1_change_is_judged_on_chrome_not_content():
    a = synth_frame(top="pink", tab="lightblue", body="noise", seed=1)
    b = synth_frame(top="pink", tab="lightblue", body="noise", seed=2)  # a new video filling the screen is not a change
    c = synth_frame(top="pink", tab="white", body="noise", seed=1)
    chrome = [(0, 0, 1, 0.12), (0, 0.88, 1, 1)]
    assert changed(a, b, chrome=chrome) < 0.02
    assert changed(a, c, chrome=chrome) > 0.2


def test_template_finder_locates_icon_within_region(tmp_path):
    from PIL import Image
    png = synth_frame(top="pink", tab="lightblue", body="white", icon=("black", 80, 1400), seed=0)
    icon = Image.new("RGB", (60, 60), "black")
    hit = find_template(png, icon, region=(0, 0.5, 0.5, 0.65), min_score=0.8)
    assert hit is not None and abs(hit[0] - 110) <= 6 and abs(hit[1] - 1430) <= 6
    assert find_template(png, icon, region=(0.5, 0, 1, 0.3), min_score=0.8) is None


def test_system_alert_class_parks_with_text():
    f = Frame(ocr=[item("Photos", 585, 120), item('Allow "Instagram" full access to', 585, 1500, 500),
                   item("your photo library?", 585, 1540, 400), item("Allow Full Access", 585, 1900, 300),
                   item("Cancel", 585, 1990, 120)], width=W, height=H)
    text = system_alert(f)
    assert text and "full access" in text and "Cancel" not in text
    assert system_alert(Frame(ocr=[item("Instagram", 300, 90), item("Cancel", 1000, 90)], width=W, height=H)) is None


def test_classifier_labels_lock_home_app_and_unknown(store):
    assert classify(Frame(ocr=[item("Swipe up to open", 585, 2400)], width=W, height=H), store) == "locked"
    f = Frame(ocr=[item("Instagram", 300, 90), item("Your story", 70, 300)], width=W, height=H)
    assert classify(f, store, frontmost="com.burbn.instagram") == "app:com.burbn.instagram"
    assert classify(f, store, frontmost="com.apple.Preferences") == "unknown"  # no bundle agreement, no label
    assert classify(Frame(ocr=[item("Settings", 300, 2000), item("Instagram", 500, 2000)], width=W, height=H), store,
                    frontmost="com.apple.springboard") == "home"
