"""Independent review (azure/tw-phone-review) of missions' tab gate on looks whose keyboard the fast OCR missed (live
11-08: a taught Search-tab tap landed in the keyboard's margin). SIMULATED looks; the daemon's recognised screen faked (since 11-08 08:4xZ the look's own `screen`, tw-phone)."""

import pytest

from teachable_worker import missions

W, H = 1170, 2532


def _it(text, x, y):
    return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": 56}


def _look(*items):
    return {"items": list(items), "width": W, "height": H, "frame_id": "f", "captured_at": "t"}


TYPING_KEYS_UNREAD = _look(_it("natgeo", 420, 230), _it("Cancel", 1060, 230), _it("natgeo", 300, 420),
                           _it("natgeo • 268M followers", 420, 470))  # the keyboard's rows: nothing read


@pytest.mark.parametrize("scr", ["instagram.search.typing", None, "instagram.unknown"])
def test_no_tab_tap_on_a_search_screen_whose_keyboard_the_ocr_missed(scr, monkeypatch):
    look = {**TYPING_KEYS_UNREAD, "screen": scr}
    assert not missions.tabbed(look, scr)
    for name in missions.TABS:
        assert missions._tab(name)(look) is None  # resolve: no tap, whatever the tab


def test_a_home_feed_is_still_tabbed_and_a_comment_sheet_over_it_is_not(monkeypatch):
    home = _look(_it("Instagram", 300, 160), _it("Your story", 160, 620))
    sheet = _look(_it("Your story", 160, 300), _it("Comments", 585, 1100), _it("Add a comment…", 420, 1350), _it("Post", 1080, 1350))
    assert missions._tab("search")(home) is not None  # the positive case still taps
    assert missions._tab("search")(sheet) is None  # keys unread, the feed behind: the composer's Post refuses
