"""surfaces4: records the director's detour kinds read (stories, Explore grid, comments, home tab/story/save targets)."""
import shutil

import pytest

from phonebridge.config import DATA_DIR
from phonebridge.screens.store import RECORD_DIRS, Store


def test_repo_record_wins_over_a_learned_candidate_of_the_same_id(tmp_path):
    cand = tmp_path / "candidates" / "instagram"
    cand.mkdir(parents=True)
    src = RECORD_DIRS[0] / "instagram" / "instagram.screen.instagram_explore_search_grid.yaml"
    stale = src.read_text().replace("grid_tile", "stale_tile")
    (cand / src.name).write_text(stale)
    r = Store(records_dirs=RECORD_DIRS, corpus=tmp_path).records["instagram.screen.instagram_explore_search_grid"]
    assert "grid_tile" in r.targets and r.source == "surfaces"


import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from phonebridge.director.feedmaps import ig_save_icon, ig_saved_state, ig_story_ring
from phonebridge.screens.cascade import Cascade, Frame
from phonebridge.screens.store import CORPUS

HOME = CORPUS / "instagram" / "instagram.home"
real = pytest.mark.skipif(not (HOME / "2026-10-18-v-01.png").exists(), reason="no IG home frames")
W, H = 1170, 2532


def frame(n):
    return json.loads((HOME / f"2026-10-18-v-{n}.ocr.json").read_text()), (HOME / f"2026-10-18-v-{n}.png").read_bytes()


@real
def test_save_icon_is_the_bookmark_on_the_action_row_by_pixels_only():
    for n, y in (("01", 0.68), ("08", 0.682)):  # the comment bubble's row on the same frames (test_surfaces_capture_ig): 0.682 h
        items, png = frame(n)
        x, yy, _, _ = ig_save_icon(items, W, H, png)
        assert abs(x / W - 0.937) < 0.01 and abs(yy / H - y) < 0.01
        assert ig_save_icon(items, W, H) is None  # a save is a write: no pixels, no target
    for n in ("02", "03", "05"):  # no action row on screen
        assert ig_save_icon(*frame(n)[:1], W, H, frame(n)[1]) is None


@real
def test_saved_state_reads_filled_vs_outline():
    items, png = frame("01")
    x, y, _, _ = ig_save_icon(items, W, H, png)
    assert ig_saved_state(png, y, W, H) is False
    im = Image.open(BytesIO(png)).convert("RGB")
    ImageDraw.Draw(im).rectangle((x - 20, y - 24, x + 20, y + 24), fill=(255, 255, 255))  # a filled glyph (synthetic: no real save frame yet)
    assert ig_saved_state(im, y, W, H) is True


@real
def test_story_ring_is_never_your_story():
    for n in ("01", "02", "03"):
        items, _ = frame(n)
        x, y, _, _ = ig_story_ring(items, W, H)
        assert 0.3 < x / W < 0.5 and 0.15 < y / H < 0.2
    assert ig_story_ring([{"text": "Your story", "x": 156, "y": 620, "w": 200, "h": 40}], W, H) is None
    items, _ = frame("08")
    assert ig_story_ring(items, W, H) is None  # a scrolled home: no tray


def test_record_targets_land_on_the_measured_centres():
    s = Store(records_dirs=RECORD_DIRS, corpus=Path("/nonexistent"))
    def centre(rid, t):
        x0, y0, x1, y1 = s.records[rid].targets[t].region
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    assert centre("instagram.comments", "sheet_grabber") == pytest.approx((0.5, 0.3225))
    assert centre("instagram.comments", "comment_handle") == pytest.approx((0.205, 0.45))
    assert centre("instagram.screen.instagram_explore_search_grid", "grid_tile") == pytest.approx((0.5, 0.28))
    assert centre("instagram.home", "story_ring")[0] > 0.3  # past the Your-story tile (0.10 w)
    for t, x in (("home_tab", 0.1), ("reels_tab", 0.3), ("search_tab", 0.7)):
        assert centre("instagram.home", t) == pytest.approx((x, 0.935), abs=0.005)
    assert centre("instagram.reels", "reels_tab")[0] == pytest.approx(0.3)


def test_explore_grid_reads_real_frames_and_not_the_keyboard_or_results():
    p = DATA_DIR / "evidence/tracer-20261018/12-calibrate-tabs"
    if not (p / "02-explore.png").exists():
        pytest.skip("no explore frames")
    c = Cascade(Store(records_dirs=RECORD_DIRS, corpus=Path("/nonexistent")))
    def read(name):
        return c.recognise(Frame.from_files(p / f"{name}.png"), platform="instagram").screen_id
    assert read("02-explore") == "instagram.screen.instagram_explore_search_grid"
    assert read("01-now") != "instagram.screen.instagram_explore_search_grid"  # the keyboard over the grid: nothing to tap


# --- surfaces5: follow_button + follow_state (the director's follow write kind) ------------------------------------------------
from phonebridge.director.feedmaps import follow_button, follow_state

CAP = DATA_DIR / "evidence/surfaces-capture"
TT_PROFILE = CAP / "tiktok_profile-20261019T073826Z" / "02-profile"
X_PROFILE = CAP / "x-20261019T062643Z" / "05-post"
X_PINNED = CAP / "x-20261019T062643Z" / "06-thread"
real_follow = pytest.mark.skipif(not (TT_PROFILE.with_suffix(".png").exists() and X_PROFILE.with_suffix(".png").exists()), reason="no profile captures")


def load(p):
    return json.loads(p.with_suffix(".ocr.json").read_text()), p.with_suffix(".png").read_bytes()


def reword(items, old, new):
    return [{**i, "text": new} if i["text"].strip() == old else i for i in items]


def paint(png, centre, size, fill):
    im = Image.open(BytesIO(png)).convert("RGB")
    x, y = centre
    ImageDraw.Draw(im).rounded_rectangle((x - size[0], y - size[1], x + size[0], y + size[1]), radius=40, fill=fill)
    return im


@real_follow
def test_tiktok_follow_button_and_state_on_the_real_profile():
    items, png = load(TT_PROFILE)
    x, y, _, _ = follow_button("tiktok", items, W, H)
    assert (x / W, y / H) == pytest.approx((0.235, 0.281), abs=0.01)  # the red Follow pill; the 'Following' COUNT label (0.227 h) is not it
    assert follow_state("tiktok", items, png, W, H) == "not_following" == follow_state("tiktok", items, None, W, H)
    # the other side is synthetic (no real followed frame): grey pill + the word Following
    grey = paint(png, (x, y), (200, 50), (225, 225, 225))
    assert follow_state("tiktok", reword(items, "Follow", "Following"), grey, W, H) == "following"
    assert follow_state("tiktok", reword(items, "Follow", "Requested"), grey, W, H) == "requested"
    assert follow_state("tiktok", reword(items, "Follow", "Following"), png, W, H) is None  # the words say Following over a red Follow fill: no verdict
    assert follow_state("tiktok", [i for i in items if i["text"].strip() != "Follow"], png, W, H) is None  # no button words: unreadable, never a guess


@real_follow
def test_follow_button_is_not_on_a_feed_or_a_comment_sheet():
    for name in ("tiktok-20261019T065927Z/01-feed", "tiktok-20261019T065927Z/03-comments_scroll"):
        items, png = load(CAP / name)
        assert follow_button("tiktok", items, W, H) is None and follow_state("tiktok", items, png, W, H) is None
    items, png = load(CAP / "instagram-20261019T064821Z" / "03-comments")  # a 'Follow' sits in the dimmed post header above the sheet
    assert follow_button("instagram", items, W, H) is None  # (0.80 w, 0.205 h: above the button row band)
    items, png = load(CAP / "instagram-20261019T064821Z" / "01-home")
    assert follow_state("instagram", items, png, W, H) is None  # the feed post header's Follow (0.80, 0.29) is not a profile button


@real_follow
def test_x_follow_expanded_and_pinned():
    items, png = load(X_PROFILE)
    x, y, _, _ = follow_button("x", items, W, H)
    assert (x / W, y / H) == pytest.approx((0.743, 0.564), abs=0.01)
    assert follow_state("x", items, png, W, H) == "not_following"
    outlined = paint(png, (x, y), (220, 50), (0, 0, 0))  # Following = a dark outlined pill (synthetic)
    assert follow_state("x", reword(items, "Follow", "Following"), outlined, W, H) == "following"
    p_items, p_png = load(X_PINNED)  # the scrolled header's dark pill: words only
    assert follow_button("x", p_items, W, H)[1] / H == pytest.approx(0.09, abs=0.01)
    assert follow_state("x", p_items, p_png, W, H) == "not_following"


def test_instagram_follow_by_words_and_blue_fill():
    row = [{"text": "Follow", "x": 290, "y": 780, "w": 200, "h": 40, "conf": 0.9}, {"text": "Message", "x": 780, "y": 780, "w": 220, "h": 40, "conf": 0.9},
           {"text": "following", "x": 880, "y": 540, "w": 150, "h": 40, "conf": 0.9}]  # the count label (0.21 h) never matches
    x, y, _, _ = follow_button("instagram", row, W, H)
    assert (round(x), round(y)) == (290, 780)
    blue = Image.new("RGB", (W, H), (0, 0, 0))
    ImageDraw.Draw(blue).rounded_rectangle((60, 740, 520, 820), radius=30, fill=(0, 149, 246))
    assert follow_state("instagram", row, blue, W, H) == "not_following"
    grey = Image.new("RGB", (W, H), (0, 0, 0))
    ImageDraw.Draw(grey).rounded_rectangle((60, 740, 520, 820), radius=30, fill=(54, 54, 54))
    assert follow_state("instagram", reword(row, "Follow", "Following"), grey, W, H) == "following"
    assert follow_state("instagram", reword(row, "Follow", "Following"), blue, W, H) is None


def test_profile_records_name_follow_button_and_keep_it_out_of_avoid():
    s = Store(records_dirs=RECORD_DIRS, corpus=Path("/nonexistent"))
    for rid in ("instagram.profile.other", "tiktok.profile.other", "x.profile"):
        r = s.records[rid]
        assert "follow_button" in r.targets, rid
    t = s.records["tiktok.profile.other"]
    cx, cy = [(a + b) / 2 for a, b in zip(t.targets["follow_button"].region[:2], t.targets["follow_button"].region[2:])]
    assert all(not (z[0] <= cx <= z[2] and z[1] <= cy <= z[3]) for z in t.avoid)  # the target is not inside an avoid zone
    i = s.records["instagram.profile.other"]
    assert all(not (z[0] <= 0.25 <= z[2] and z[1] <= 0.3 <= z[3]) for z in i.avoid)


@real_follow
def test_tiktok_profile_other_reads_the_real_frame_and_the_feed_does_not():
    c = Cascade(Store(records_dirs=RECORD_DIRS, corpus=Path("/nonexistent")))
    assert c.recognise(Frame.from_files(TT_PROFILE.with_suffix(".png")), platform="tiktok").screen_id == "tiktok.profile.other"
    assert c.recognise(Frame.from_files((CAP / "tiktok_profile-20261019T073826Z" / "01-feed").with_suffix(".png")), platform="tiktok").screen_id != "tiktok.profile.other"
