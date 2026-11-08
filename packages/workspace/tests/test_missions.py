"""The read-only missions' own decisions (no phone, no daemon): how a row, a profile, a tab screen and the keyboard are
read from OCR, and that an input refused with zero input sent is re-decided on a fresh look, never re-sent blind
(tw-phone-review MS1). Live behaviour is in the sprint's phone.md; nothing here is evidence a phone did anything."""

import pytest

from teachable_worker import missions as m

H, W = 2532, 1170


def line(text, y, x=500, w=300, h=40):
    return {"text": text, "x": x, "y": y, "w": w, "h": h}


def test_a_result_row_is_led_by_the_handle_never_merely_containing_it():
    assert m.lead("natgeo • 268M followers") == "natgeo"
    assert m.lead("natgeo ✓") == "natgeo" and m.lead("@natgeo") == "natgeo"
    assert m.lead("< natgeo") == "natgeo"  # the profile title, the back chevron in the same OCR line (live 11-08)
    assert m.lead("national geographic") == ""  # the name line is not a handle
    assert m.lead("natgeotv • 7.4M followers") != "natgeo"
    assert m.lead("NASA Internships • 743K followers") == "" and m.lead("NASA") == ""  # display names (tw-teach)
    assert m.lead("nasa v") == "nasa" and m.lead("nasa 0") == "nasa"  # a badge read as a letter


def test_the_profile_header_handle_and_whole_token_counts():
    items = [line("natgeo ✓", 0.06 * H, w=200), line("268M", 0.18 * H, x=400), line("posts", 0.205 * H, x=200),
             line("1,234", 0.18 * H, x=200), line("followers", 0.205 * H, x=400), line("following", 0.205 * H, x=650),
             line("120", 0.18 * H, x=650), line("@natgeo.fan bio line", 0.3 * H)]
    r = m.read_profile(items, W, H, "natgeo")
    assert r["handle"] == "natgeo"
    assert r["counts"] == {"posts": {"value": 1234, "shown": "1,234"}, "followers": {"value": 268000000, "shown": "268M"},
                           "following": {"value": 120, "shown": "120"}}
    assert m.read_profile([line("other.account", 0.06 * H), line("@natgeo", 0.3 * H)], W, H, "natgeo")["handle"] == "other.account"


def test_tab_screens_and_the_keyboard():
    feed = [line("Your story", 0.2 * H), line("natgeo", 0.4 * H)]
    assert m.tab_screen(feed, H)
    assert not m.tab_screen(feed + [line("Add a comment for natgeo", 0.95 * H)], H)  # text in the tab bar: a thread/sheet
    assert not m.tab_screen(feed + [line("Share", 0.5 * H)], H)  # a composer's button
    assert m.keyboard_up([line(c, 0.7 * H) for c in "qwertyui"], H) and m.keyboard_up([line("ABC", 0.9 * H)], H)
    assert not m.keyboard_up(feed, H)


def test_a_platform_stop_text_is_found_like_taskapis():
    stops = ["Try again later", "Log in", "We suspect automated behavior"]
    assert m.stop_text([line("Try Again Later", 0.5 * H)], stops) == "Try Again Later"
    assert m.stop_text([line("Log in", 0.5 * H)], stops) == "Log in"
    assert m.stop_text([line("how to log in to your account and make money fast today", 0.5 * H)], stops) is None


class Ctx:
    def __init__(self, looks):
        self.looks, self.steps, self.sent = list(looks), [], []
        self.hands = None

    def observe(self, note=None):
        o = self.looks.pop(0)
        return {**o, "frame_id": f"f{len(self.looks)}", "captured_at": "t", "width": W, "height": H}

    def step(self, kind, text, frame_id=None, pace_ms=None, **data):
        self.steps.append((kind, text))

    def check(self):
        pass


def test_a_zero_input_refusal_is_re_decided_on_the_fresh_look_never_resent_to_the_old_point(monkeypatch):
    """MS1: the row moved between the refused attempt and the re-look; the tap goes to where it is NOW."""
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    first = {"items": [line("natgeo • 268M followers", 0.15 * H)]}
    moved = {"items": [line("Follow", 0.15 * H), line("natgeo • 268M followers", 0.40 * H)]}  # a sheet pushed it down
    ctx = Ctx([moved])
    tries = []

    def hands(c, kind, what, tool, **args):
        tries.append(args["y"])
        if len(tries) == 1:
            raise m.ZeroInput("StaleInput: stale: re-look … zero input sent")
    ctx.hands = hands

    def resolve(o):
        r = m._line(o["items"], lambda t: m.lead(t) == "natgeo", (0.08, 0.9), H)
        return r and ("tap", "Open natgeo", "tap", {"x": r["x"], "y": r["y"]})
    assert m.act(ctx, {**first, "height": H, "width": W}, [], resolve, "row") is not None
    assert tries == [0.15 * H, 0.40 * H]  # re-resolved: never the old point (which is now Follow)
    assert ("note",) == tuple(k for k, _ in ctx.steps[:1])


def test_a_target_gone_on_the_fresh_look_sends_nothing(monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    ctx = Ctx([{"items": [line("Follow", 0.3 * H)]}])
    sent = []

    def hands(c, kind, what, tool, **args):
        sent.append(args)
        raise m.ZeroInput("Parked: iphone12 is parked (usb); zero input sent")
    ctx.hands = hands
    resolve = lambda o: (r := m._line(o["items"], lambda t: m.lead(t) == "natgeo", (0.08, 0.9), H)) and (  # noqa: E731
        "tap", "Open natgeo", "tap", {"x": r["x"], "y": r["y"]})
    assert m.act(ctx, {"items": [line("natgeo • 1M followers", 0.3 * H)], "height": H, "width": W}, [], resolve, "row") is None
    assert len(sent) == 1  # one refused try; on the fresh look the row is gone: nothing more


def test_tab_taps_need_a_main_tab_screen():
    on_feed = {"items": [line("Your story", 0.2 * H)], "width": W, "height": H}
    in_thread = {"items": [line("Message...", 0.95 * H)], "width": W, "height": H}
    assert m._tab("search")(on_feed)[3]["x"] == pytest.approx(0.70 * W)
    assert m._tab("search")(in_thread) is None


def test_where_names_instagram_454s_screens_from_their_text():
    """From the live 11-08 frames' text: tab-bar screens vs detail pages vs the story viewer."""
    o = lambda *ls: {"items": [line(t, y * H) for t, y in ls], "height": H, "width": W}  # noqa: E731
    assert m.where(o(("Your story", 0.2))) == "home"
    assert m.where(o(("samplecreator.co v", 0.08), ("Edit profile", 0.37), ("Share profile", 0.37))) == "own_profile"
    assert m.where(o(("< natgeo", 0.08), ("Follow", 0.38), ("Message", 0.38))) == "other_profile"
    assert m.where(o(("nasa", 0.08), ("For you", 0.14), ("Profiles", 0.14), ("Follow", 0.2))) == "keyword"
    assert m.where(o(("nasa v", 0.09), ("Follow", 0.1), ("Send message...", 0.93))) == "story"
    assert m.where(o(("Posts", 0.07), ("natgeo", 0.09), ("Follow", 0.08))) == "post"
    assert m.where(o(("samplecreator.co", 0.08), ("Messages", 0.4), ("Requests", 0.4), ("Tap to chat", 0.6))) == "inbox"
    assert m.where(o(("Search with Meta AI", 0.085), ("Recent", 0.15))) == "search"
    assert m.where(o(("Search with Meta AI", 0.085), *[(c, 0.7) for c in "qwertyui"])) == "typing"
    assert m.where(o(("natgeo", 0.085), ("national geographic", 0.115), ("natgeo • 268M followers", 0.13))) == "search"
    assert m.where(o(("Something new", 0.5))) == "unknown"


def test_handles_come_as_the_ui_sends_them_or_as_a_string():
    """The workspace UI splits a list field into an array (root, live UI 11-08); the API sends a string."""
    assert m.parse_handles(["natgeo", "@nasa"]) == ["natgeo", "nasa"]
    assert m.parse_handles("natgeo, @nasa") == ["natgeo", "nasa"]
    assert m.parse_handles(["a", "b", "c", "d", "e", "f"]) == ["a", "b", "c", "d", "e"] and m.parse_handles(None) == []


def test_a_recognised_typing_screen_is_never_a_tab_screen_even_when_ocr_misses_the_keyboard():
    """Live 11-08 (taught run tsk_94270f11ea25): the fast OCR returned no keys; a Search-tab tap hit the keyboard margin."""
    o = {"items": [line("Search with Meta Al", 0.085 * H), line("Recent", 0.15 * H)], "height": H, "width": W}
    assert m.tab_screen(o["items"], H) is True  # what the OCR alone said
    for scr in ("instagram.search.typing", "instagram.screen.instagram_search_typing"):
        assert m.tab_screen(o["items"], H, scr) is False and m.where(o, scr) == "typing" and m.keyboard_up(o["items"], H, scr)
    assert m.where(o, "instagram.screen.instagram_explore_search_grid") == "search"


def test_the_search_page_is_a_tab_screen_only_when_the_daemon_recognises_it():
    """tw-phone-review: a look whose keyboard keys the OCR missed reads like the search page; recognition decides."""
    o = {"items": [line("Search with Meta Al", 0.085 * H), line("Recent", 0.15 * H)], "height": H, "width": W}
    assert not m.tabbed(o, None) and not m.tabbed(o, "instagram.screen.instagram_search_typing")
    assert m.tabbed(o, "instagram.screen.instagram_explore_search_grid")
    feed = {"items": [line("Your story", 0.2 * H)], "height": H, "width": W}
    assert m.tabbed(feed, None)  # the home feed's own text is enough


GRID = "instagram.screen.instagram_explore_search_grid"
EXPLORE_OCR = [  # frm_233c85cd014c (live 11-08 07:33:24Z, IG 454's Explore grid: no chips, no chevron), stored OCR, top lines
    {"text": "G Search with Meta Al", "x": 359, "y": 213, "w": 526, "h": 52}, line("MURC/ELAGO", 0.138 * H, 737, 70),
    line("African Lion", 0.144 * H, 70, 70), line("18 years ago, Russian local TV caught the", 0.159 * H, 959, 316),
    line("most legendary interview of all time", 0.168 * H, 936, 281), line("2008", 0.177 * H, 433, 47),
    line("Not enough people, Madame!", 0.293 * H, 971, 234), line("6.7M", 0.316 * H, 105, 94),
    line("PHOTOS YOU WON'T", 0.488 * H, 573, 363), line("Peak male ego", 0.874 * H, 187, 234)]


class _Hands:
    device = "iphone12"

    def __init__(self):
        self.taps = []

    def __call__(self, c, kind, what, tool, **args):
        self.taps.append((what, round(args["x"] / W, 3), round(args["y"] / H, 3)))


def _nav(monkeypatch, looks):
    monkeypatch.setattr(m, "_wait", lambda ctx: None)
    monkeypatch.setattr(m, "_look", lambda ctx, note, stops: {**looks.pop(0), "frame_id": "f"})
    ctx = Ctx([])
    ctx.hands = _Hands()
    return ctx


def test_the_explore_grid_the_phone_did_not_recognise_stops_with_nothing_sent(monkeypatch):
    """Live 11-08 (tsk_d2da279d4755): recognition returned null on the grid; BACK there is the search bar, not a chevron."""
    grid = {"items": EXPLORE_OCR, "height": H, "width": W}
    ctx = _nav(monkeypatch, [])
    assert m.where(grid) == "search" and not m.tabbed(grid, None) and m._back()(grid) is None
    with pytest.raises(m.Stop, match="search page the phone didn't recognise"):
        m.to_tabs(ctx, [], grid)
    assert ctx.hands.taps == []


def test_back_from_the_keyword_page_to_the_recognised_grid(monkeypatch):
    """The live run's path: keyword page → its chevron → the Explore grid, now recognised (search-tab template)."""
    keyword = {"items": [line("C national geographic", 0.086 * H, 468, 503), line("For you", 0.144 * H, 140, 129),
                         line("Profiles", 0.142 * H, 363, 152), line("Audio", 0.142 * H, 585, 105)], "height": H, "width": W}
    grid = {"items": EXPLORE_OCR, "height": H, "width": W, "screen": GRID}  # the daemon names THIS frame
    ctx = _nav(monkeypatch, [grid])
    assert m.to_tabs(ctx, [], keyword)["items"] is EXPLORE_OCR
    assert ctx.hands.taps == [("Back (keyword)", m.BACK[0], m.BACK[1])]
    assert m.tabbed(grid, GRID) and m._tab("search")(grid) is not None
    assert m._tab("search")({**grid, "screen": None}) is None  # the same pixels unnamed: no blind tab tap


def test_a_falsy_resolver_sends_nothing():
    """act's contract: a guard written `cond and (...)` returns False, never unpacked (live 11-08 TypeError)."""
    ctx = Ctx([])
    ctx.hands = _Hands()
    assert m.act(ctx, {"items": [], "height": H, "width": W}, [], lambda o: False, "x") is None and ctx.hands.taps == []


def test_a_back_the_fresh_look_refuses_stops_at_once(monkeypatch):
    """Root 11-08: no string of no-op looks when the chevron's resolver declines; one stop with the reason."""
    keyword = {"items": [line("nasa", 0.08 * H), line("For you", 0.14 * H), line("Profiles", 0.14 * H)], "height": H, "width": W}
    ctx = _nav(monkeypatch, [])
    monkeypatch.setattr(m, "_back", lambda: lambda o: None)
    with pytest.raises(m.Stop, match="back chevron of the keyword screen"):
        m.to_tabs(ctx, [], keyword)
    assert ctx.hands.taps == []


def test_an_old_query_is_positive_and_the_placeholder_never_is():
    """Live 11-08 (tsk_dccf35570abb, frm_5ed6d345a92a): the empty field read "Ci bearch with Meta Al"; tw-teach's catch."""
    bar = lambda text, w=526: [{"text": "<", "x": 70, "y": 213, "w": 30, "h": 50}, {"text": text, "x": 458, "y": 212, "w": w, "h": 52}]  # noqa: E731
    for placeholder in ("Ci bearch with Meta Al", "G Search with Meta Al", "Q Search with Meta Al", "SearchwithMetaAl"):
        assert m.old_query(bar(placeholder), H) is None
    assert m.old_query(bar("Ci nasa", 200), H) == "nasa" and m.old_query(bar("@natgeo", 180), H) == "natgeo"
    assert m.old_query(bar("Ci national geographic", 400), H) is None  # not one handle-like token: never cleared blind
    assert m.old_query(bar("natgeo", 180) + [{"text": "Cancel", "x": 1076, "y": 213, "w": 140, "h": 50}], H) == "natgeo"
    assert m.old_query([{"text": "Recent", "x": 130, "y": 250, "w": 300, "h": 50}], H) is None  # chrome is never a query



def test_a_count_the_fast_ocr_dropped_is_read_once_at_the_accurate_level(monkeypatch):
    """Live 11-08 (cut a96955d): @apple's '9 following' was missing from the look's fast OCR; Vision accurate read it."""
    header = [line("apple", 0.083 * H, 270), line("1,413", 0.175 * H, 433), line("38.1M", 0.174 * H, 667),
              line("posts", 0.197 * H, 421), line("followers", 0.195 * H, 679), line("following", 0.196 * H, 959)]
    fast = {"items": header, "width": W, "height": H, "frame_id": "fast"}
    asked = []

    class Ctx2:
        def observe(self, note=None, accurate=False):
            asked.append(accurate)
            return {"items": header + [line("9", 0.175 * H, 889)], "width": W, "height": H, "frame_id": "accurate"}
    read, o = m.read_counts(Ctx2(), fast, "apple", "apple's profile", [])
    assert read["counts"]["following"] == {"value": 9, "shown": "9"} and o["frame_id"] == "accurate" and asked == [True]
    asked.clear()
    full = {**fast, "items": header + [line("9", 0.175 * H, 889)]}
    assert m.read_counts(Ctx2(), full, "apple", "x", [])[1]["frame_id"] == "fast" and asked == []  # all three: no re-read



def test_a_workspace_without_accurate_looks_keeps_the_fast_read():
    """Until the workspace's observe has `accurate`, a missing count stays missing (no TypeError, no second look)."""
    header = [line("apple", 0.083 * H, 270), line("1,413", 0.175 * H, 433), line("posts", 0.197 * H, 421)]

    class Old:
        def observe(self, note=None):
            raise AssertionError("no second look")
    read, o = m.read_counts(Old(), {"items": header, "width": W, "height": H, "frame_id": "fast"}, "apple", "x", [])
    assert o["frame_id"] == "fast" and "following" not in read["counts"]
