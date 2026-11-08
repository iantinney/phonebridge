"""Lane surfaces' records (apps/feedagent/screens): every frame of tonight's real capture visits is recognised as its screen
(corpus/sequences/surfaces_*, outside git), and every record without a real frame yet is recognised on its mock fixture
(the app's UI strings laid out as ocr.py writes them: box centres) and on no other record's fixture. Mock = not a phone frame."""
import io
import json

import pytest

from phonebridge.screens.cascade import Cascade, Frame
from phonebridge.screens.store import CORPUS, RECORD_DIRS, Store
from screens_fixtures.synth import frame as synth
from test_surfaces3_records import tiktok_bar

W, H = 1170, 2532
SEQ = CORPUS / "sequences"
REAL = {  # sequence dir → {frame-name suffix: expected screen id}
    "surfaces_x/1": {"home": "x.home", "scroll1": "x.home", "scroll2": "x.home", "scroll3": "x.home", "post": "x.profile",
                     "thread": "x.profile", "back1": "x.profile", "profile_seek": "x.profile", "following_scroll": "x.profile",
                     "drawer_closed": "x.profile"},
    "surfaces_x/2": {"home": "x.home", "scroll1": "x.home", "scroll2": "x.home", "scroll3": "x.home", "post": "x.post",
                     "thread": "x.post", "back1_button": "x.media.video"},
    "surfaces_tiktok/1": {"feed": "tiktok.feed.merged_nav", "comments": "tiktok.comments", "comments_scroll": "tiktok.comments",
                          "comments_closed": "tiktok.feed.merged_nav", "profile_seek": "tiktok.feed.merged_nav"},
    "surfaces_tiktok/2": {"feed": "tiktok.feed.draft_banner", "profile": "tiktok.profile.other", "back": "tiktok.feed.draft_banner"},
    "surfaces_instagram/1": {"home": "instagram.home", "comments": "instagram.comments", "comments_scroll": "instagram.comments",
                             "comments_closed": "instagram.comments", "profile_seek": "instagram.comments"},
}


@pytest.fixture(scope="module")
def cascade(tmp_path_factory):
    return Cascade(Store(records_dirs=RECORD_DIRS, corpus=tmp_path_factory.mktemp("c")))


def real_cases():
    for seq, want in REAL.items():
        for suffix, sid in want.items():
            hits = sorted((SEQ / seq).glob(f"*-{suffix}.png")) if (SEQ / seq).exists() else []
            yield pytest.param(hits[0] if hits else None, sid, id=f"{seq}:{suffix}")


@pytest.mark.parametrize("png,want", list(real_cases()))
def test_real_capture_frames_recognised(cascade, png, want):  # noqa: D103
    if png is None:
        pytest.skip("capture not in this corpus")
    platform = want.split(".")[0]
    m = cascade.recognise(Frame.from_files(png), platform=platform)
    assert m.screen_id == want, (png.name, m.screen_id, m.detail)


def test_scrolled_ig_home_with_the_story_tray_in_the_top_band(cascade):
    """06:48Z frame 02: the story tray's photos lift the band's mean to 94, but dark pixels still outnumber light ones (share
    0.34), so instagram.home (which declares its chrome) stays dark and the home-tab template carries the match."""
    png = SEQ / "surfaces_instagram" / "1" / "02-comments_seek.png"
    if not png.exists():
        pytest.skip("no IG capture")
    m = cascade.recognise(Frame.from_files(png), platform="instagram")
    assert m.screen_id == "instagram.home" and "tpl:home_tab_active" in m.detail, m.detail


def it(text, x, y, w=None, h=44):
    """An OCR line centred at (x, y) fractions, as ocr.py writes it."""
    return {"text": text, "x": round(x * W), "y": round(y * H), "w": w or 24 * len(text), "h": h, "conf": 0.95}


KEYBOARD = [it(c, 0.05 + 0.1 * k, 0.70) for k, c in enumerate("qwertyuiop")] + \
           [it(c, 0.1 + 0.1 * k, 0.76) for k, c in enumerate("asdfghjkl")] + [it("123", 0.073, 0.889, 70)]
TT_SHEET = [it("340 comments", 0.52, 0.395), it("X", 0.932, 0.396, 40), it("great build", 0.3, 0.463), it("Reply", 0.27, 0.495, 90),
            it("View 55 replies", 0.36, 0.62)]
IG_SHEET = [it("Comments", 0.498, 0.348), it("itsnurasiva 16h", 0.244, 0.466), it("News", 0.18, 0.49), it("Reply", 0.17, 0.516, 90)]
MOCK = {
    "tiktok.feed.merged_nav": [it("LIVE ewHaven Following Friends ForYou", 0.5, 0.084), it("Home", 0.1, 0.947), it("Shop", 0.3, 0.948),
                               it("Inbox", 0.7, 0.948), it("Prof ile", 0.9, 0.948)],  # fast-eyes: what Vision's fast level reads
    "tiktok.stop.composer": TT_SHEET + [it("Add comment...", 0.32, 0.6)] + KEYBOARD,
    "instagram.stop.composer": IG_SHEET + [it("What do you think of this?", 0.42, 0.6)] + KEYBOARD,
    "tiktok.profile.other": [it("someone", 0.5, 0.17), it("@someone", 0.5, 0.2), it("12", 0.3, 0.25, 40), it("Following", 0.3, 0.27),
                             it("3,401", 0.5, 0.25), it("Followers", 0.5, 0.27), it("98K", 0.7, 0.25), it("Likes", 0.7, 0.27),
                             it("Follow", 0.42, 0.33), it("Message", 0.6, 0.33)],
    "tiktok.search.results": [it("calisthenics", 0.45, 0.07), it("Top", 0.08, 0.12), it("Videos", 0.24, 0.12), it("Users", 0.4, 0.12),
                              it("Sounds", 0.55, 0.12), it("LIVE", 0.7, 0.12), it("Hashtags", 0.88, 0.12)],
    "tiktok.menu.long_press": [it("Not interested", 0.2, 0.75), it("Report", 0.45, 0.75), it("Speed", 0.7, 0.75), it("Clear display", 0.2, 0.83),
                               it("Captions", 0.45, 0.83)],
    "tiktok.save_sheet": [it("Added to Favorites", 0.35, 0.8), it("Manage", 0.85, 0.8), it("Add to collection", 0.4, 0.86)],
    "tiktok.stop.share": [it("Send to", 0.15, 0.62), it("Repost", 0.15, 0.78), it("Copy link", 0.35, 0.78), it("Duet", 0.55, 0.78),
                          it("Stitch", 0.75, 0.78), it("Save video", 0.15, 0.88)],
    "instagram.profile.other": [it("someone", 0.5, 0.07), it("41", 0.4, 0.2, 40), it("posts", 0.4, 0.22), it("3,101", 0.6, 0.2),
                                it("followers", 0.6, 0.22), it("520", 0.8, 0.2), it("following", 0.8, 0.22), it("Follow", 0.3, 0.3),
                                it("Message", 0.65, 0.3)],
    "instagram.search.results": [it("calisthenics", 0.45, 0.07), it("For you", 0.1, 0.12), it("Accounts", 0.3, 0.12), it("Reels", 0.5, 0.12),
                                 it("Audio", 0.65, 0.12), it("Tags", 0.8, 0.12), it("Places", 0.95, 0.12)],
    "instagram.menu.post": [it("Interested", 0.3, 0.6), it("Not interested", 0.7, 0.6), it("Why you're seeing this post", 0.4, 0.68),
                            it("About this account", 0.35, 0.74), it("Hide", 0.2, 0.8), it("Report", 0.2, 0.86)],
    "instagram.save_sheet": [it("Save to", 0.5, 0.6), it("New collection", 0.3, 0.68), it("Saved", 0.3, 0.75)],
    "instagram.stop.share": [it("Search", 0.3, 0.45), it("Write a message...", 0.4, 0.8), it("Add to story", 0.15, 0.9), it("Copy link", 0.4, 0.9),
                             it("Share to...", 0.65, 0.9)],
    "x.stop.compose": [it("Cancel", 0.08, 0.07), it("Drafts", 0.7, 0.07), it("Post", 0.9, 0.07), it("What's happening?", 0.45, 0.15),
                       it("Everyone can reply", 0.3, 0.55)] + KEYBOARD,
    "x.stop.messages": [it("Messages", 0.5, 0.07), it("Search Direct Messages", 0.5, 0.13), it("Message requests", 0.3, 0.2)],
    "x.stop.settings": [it("Settings", 0.5, 0.07), it("Your account", 0.3, 0.2), it("Security and account access", 0.4, 0.26),
                        it("Privacy and safety", 0.35, 0.32), it("Monetization", 0.3, 0.38)],
}
NEW = sorted(MOCK)


def _png_for(want: str) -> bytes:
    """tiktok.feed* are read from pixels (the bottom bar's lit Home icon and the '+' button), so their mock frame carries them."""
    if not want.startswith("tiktok.feed"):
        return synth(top="black", tab="black", body="black")
    out = io.BytesIO()
    tiktok_bar().save(out, "PNG")
    return out.getvalue()


@pytest.mark.parametrize("want", NEW)
def test_mock_fixture_recognised_as_its_record(cascade, want):
    f = Frame.from_png(_png_for(want), MOCK[want])
    m = cascade.recognise(f, platform=want.split(".")[0])
    assert m.screen_id == want, (m.screen_id, m.detail)


@pytest.mark.parametrize("want", NEW)
def test_mock_fixture_matches_no_other_new_record(cascade, want):
    from phonebridge.screens.cascade import score
    f = Frame.from_png(_png_for(want), MOCK[want])
    mine = cascade.store.records[want]
    others = [r for r in cascade.store.records.values()  # the cascade narrows by platform: overlaps only matter within one;
              if r.platform == mine.platform and r.id != want and r.source == "surfaces"  # an overlay (stop) also shows the
              and not (mine.kind != "screen" and r.kind == "screen")]  # screen beneath it, and stops are checked first: by design
    assert not [r.id for r in others if score(r, f).ok]
