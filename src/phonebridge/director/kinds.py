"""Engage / detour kinds as DATA (BEHAVIOUR_MODEL §1b): dataclasses, no mini-language. A kind names its affordance (a predicate
over the look: OCR items, zones, screen), its steps (a target to tap/drag, the screens the step may land on, an `until` timeout)
and its verify. The renderer (`primitives.walk`) journals every gesture inside a kind as `kind, step, expect` and ends it with
`kind_done, verified`. A kind whose affordance is absent has probability 0 on that post; a kind without a renderer is not here."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

Look = dict  # the daemon's look: {screen, kind, items, header, avoid, settled, ...}; items' x, y are box CENTRES (ocr.py)
FEED = {"instagram": ("instagram.home", "instagram.reels"), "tiktok": ("tiktok.feed", "tiktok.foryou"), "x": ("x.home",)}
COUNT = re.compile(r"^[\d.,]+[kKmM]?$")


@dataclass(frozen=True)
class Step:
    action: str  # tap:<target> | drag:<target> | dwell:<what> | back | leave (a back only when off the feed; nothing on it) | until (no input: wait for `expect`)
    expect: tuple[str, ...]  # screen ids (or 'feed') the step may land on; '*feed' = the feed of the app
    until_s: float = 3.0  # how long to wait for `expect` after the gesture
    dwell: str | None = None  # read | glance | a params.dwell surface (story, grid, reels): a dwell step's dwell, or the one after each repeat
    repeat: str | None = None  # a gesture step done `params.depth[repeat]` times, a dwell after each (carousel cards, stories, reels)
    upto: Callable[[dict], int] | None = None  # bounds the repeats by what the look shows (carousel: the cards left)


@dataclass(frozen=True)
class Kind:
    name: str
    group: str  # engage | detour | write
    affordance: Callable[[Look, str], bool]
    steps: tuple[Step, ...]
    verify: str  # feed_back | heart_red | saved | none
    modes: tuple[str, ...] = ("passive", "intentional")
    start: str = "post"  # post: on the post, inside its dwell | home: between posts, from the home feed (app-level detours)
    variant_of: str | None = None  # walked in place of that kind at params.variant[name] (comment_profile ← comments)


def _items(look: Look) -> list[dict]:
    return look.get("items") or []


def caption_more(i: dict) -> bool:
    """A truncated caption ("… more" / "... more"), never a button that happens to end in "more" ("Learn more", "See more")."""
    t = (i.get("text") or "").rstrip()
    return t.endswith("more") and ("..." in t[-12:] or "…" in t[-12:])


def has_more(look: Look, app: str) -> bool:
    return any(caption_more(i) for i in _items(look))


def has_carousel(look: Look, app: str) -> bool:
    return cards_left(look) > 0


CAROUSEL_COUNTER = re.compile(r"^\s*(\d{1,2})\s*/\s*(\d{1,2})\s*(?:→|->|>)?\s*$")  # "2/5" and IG's current "01 / 10 →"


def cards_left(look: Look) -> int:
    """Cards after the one showing, from the carousel counter ("2/5", "01 / 10 →" → 9); 0 without a counter."""
    for i in _items(look):
        m = CAROUSEL_COUNTER.match(i.get("text", ""))
        if m:
            return max(0, int(m.group(2)) - int(m.group(1)))
    return 0


def has_comment_count(look: Look, app: str) -> bool:
    w = look.get("width") or 1170
    return any(COUNT.match(i.get("text", "").strip()) and 0.2 * w < i.get("x", 0) < 0.45 * w for i in _items(look))


def has_media(look: Look, app: str) -> bool:
    return True  # every post has media to tap


def has_header_handle(look: Look, app: str) -> bool:
    return bool(look.get("header")) or bool(look.get("who"))


REEL_VIEWER = "instagram.screen.instagram_reel_fullscreen_viewer"
RETURN_BACK = {"instagram": (REEL_VIEWER,)}  # recognised screens one back from the feed: an app reopening there is walked back once
HOME = {"instagram": "instagram.home", "tiktok": "tiktok.feed", "x": "x.home"}  # where app-level detours start and end
STORIES = "instagram.stories"  # the story viewer (no record yet: the kind stays off live until one lands)
REELS_TAB = "instagram.reels"  # the Reels tab (no record yet; the feed of a reels session, an excursion from home)
EXPLORE = "instagram.screen.instagram_explore_search_grid"
EXPLORE_VIEWER = "instagram.explore.viewer"  # the post viewer an Explore tile opens (header "Explore", ←; no record yet: surfaces4 plan)
SAVE_SHEET = "instagram.save_sheet"  # the "Saved · Save to collection" toast after the bookmark
PENDING_RECORDS = frozenset({STORIES, REELS_TAB, EXPLORE_VIEWER})  # declared before their records land: kind_ready keeps their kinds off live
UNPROVEN_LIVE = frozenset({"sheet_down", "save_icon", "follow_button"})  # targets not yet proven on a live phone (surfaces4): a kind using one stays
#   off live (follow_button: the Following/Requested side is synthetic until the first real post-follow frame per app)
PROFILES = ("instagram.profile", "tiktok.profile")  # another account's profile (the records extend these: instagram.profile.other, …)
TURN_BACK_S = 50.0  # a kind's dwells and repeats stop here and it heads back: every excursion is home within 90 s (controller-verify)
# a kind abandoned on a screen without the tab bar leaves it by the screen's own way out; any other screen takes navigate(home)
EXITS: dict[str, Step] = {STORIES: Step("drag:close", (HOME["instagram"],), 3.0),
                          REEL_VIEWER: Step("back", (EXPLORE, "*feed"), 3.0), EXPLORE_VIEWER: Step("back", (EXPLORE,), 3.0),
                          "instagram.comments": Step("drag:sheet_down", ("*feed",), 3.0)}


def story_rings(items: list[dict], w: float, h: float) -> list[tuple[float, float]]:
    """Rings in the stories tray, 0.07 h above their labels (surfaces4: ig_story_ring, the measured resolver, is tried first): on
    the row of "Your story" (never tapped: it opens his own camera), each label to its right wholly on screen without a LIVE
    badge in its column (joining a live shows him to its host)."""
    own = next((i for i in items if (i.get("text") or "").strip().lower() == "your story" and i.get("y", 0) < 0.4 * h), None)
    if own is None:
        return []
    live = [i for i in items if (i.get("text") or "").strip().upper() == "LIVE" and own["y"] - 0.08 * h < i.get("y", 0) < own["y"]]
    return [(i["x"], i["y"] - 0.0695 * h) for i in sorted(items, key=lambda i: i.get("x", 0))
            if i is not own and abs(i.get("y", 0) - own["y"]) < 0.01 * h and i["x"] > own["x"] + 0.1 * w
            and i["x"] + i.get("w", 0) / 2 < 0.98 * w and not any(abs(lv["x"] - i["x"]) < 0.06 * w for lv in live)]


def has_story_tray(look: Look, app: str) -> bool:
    return bool(story_rings(_items(look), look.get("width") or 1170, look.get("height") or 2532))


def on_home(look: Look, app: str) -> bool:
    return look.get("screen") == HOME.get(app)


COMMENT_HEAD = re.compile(r"^[a-z0-9_.]{3,30}(\s+\d{1,2}[smhdw])?$")  # a comment row's handle, alone or with its age ("handle 2h")


def comment_handles(look: Look) -> list[dict]:
    """The first comment's handle (surfaces4: an OCR handle in 0.40-0.50 h, starting right of the avatar, x ≥ 0.134 w)."""
    w, h = look.get("width") or 1170, look.get("height") or 2532
    return [i for i in _items(look) if COMMENT_HEAD.match((i.get("text") or "").strip()) and 0.4 * h <= i.get("y", 0) <= 0.5 * h
            and i.get("x", 0) - i.get("w", 0) / 2 >= 0.134 * w]


KINDS: dict[str, Kind] = {
    "expand": Kind("expand", "engage", has_more, (Step("tap:more", ("*feed",), 2.0), Step("dwell:read", ("*feed",), 0.0, "read")), "feed_back"),
    "carousel": Kind("carousel", "engage", has_carousel, (Step("drag:media_left", ("*feed",), 2.0, repeat="carousel", upto=lambda look: cards_left(look) or 1),),
                     "feed_back"),
    # on IG a single tap on a reel opens its full-screen viewer (R1 attempt 8 stopped there); a photo, or TikTok's pause, stays
    "tap_media": Kind("tap_media", "engage", has_media, (Step("tap:media", ("*feed", REEL_VIEWER), 1.5), Step("dwell:glance", ("*feed", REEL_VIEWER), 0.0, "glance"),
                                                         Step("leave", ("*feed",), 3.0)), "feed_back"),
    "comments_peek": Kind("comments_peek", "engage", has_comment_count,
                     (Step("tap:comment_icon", ("instagram.comments", "instagram.comments_sheet", "tiktok.comments"), 3.0),
                      Step("dwell:read", ("instagram.comments", "instagram.comments_sheet", "tiktok.comments"), 0.0, "read"),
                      Step("drag:sheet_down", ("*feed",), 3.0)), "feed_back", ("intentional",)),
    "profile": Kind("profile", "detour", has_header_handle,
                    (Step("tap:header_handle", ("instagram.profile", "instagram.profile_other", "tiktok.profile"), 3.0),
                     Step("dwell:read", ("instagram.profile", "instagram.profile_other", "tiktok.profile"), 0.0, "read"),
                     Step("back", ("*feed",), 3.0)), "feed_back", ("intentional",)),
    "like": Kind("like", "write", has_media, (Step("tap:media_double", ("*feed",), 1.5),), "heart_red", ("intentional",)),
    # a follow (DECISIONS 'Task agent go': follows are autonomous writes within the caps): on the post author's profile, the one tap
    # on Follow; Director.follow opens the profile (FOLLOW_OPEN), verifies the state flip on paired frames, and comes back (FOLLOW_BACK)
    "follow": Kind("follow", "write", has_header_handle, (Step("dwell:glance", PROFILES, 0.0, "glance"), Step("tap:follow_button", PROFILES, 3.0)),
                   "follow_state", ("intentional",)),  # the glance makes the profile the tap's starting screen (a kind's first gesture starts on the feed)
    # inside the comments: a commenter's profile, back to the sheet, close it (a variant of comments: intentional only, as it is)
    "comment_profile": Kind("comment_profile", "detour", has_comment_count,
                            (Step("tap:comment_icon", ("instagram.comments", "instagram.comments_sheet", "tiktok.comments"), 3.0),
                             Step("dwell:read", ("instagram.comments", "instagram.comments_sheet", "tiktok.comments"), 0.0, "read"),
                             Step("tap:comment_handle", ("instagram.profile.other", "tiktok.profile.other"), 3.0),
                             Step("dwell:read", ("instagram.profile.other", "tiktok.profile.other"), 0.0, "read"),
                             Step("back", ("instagram.comments", "instagram.comments_sheet", "tiktok.comments"), 3.0),
                             Step("drag:sheet_down", ("*feed",), 3.0)), "feed_back", ("intentional",), variant_of="comments_peek"),
    # a find saved: the bookmark on the action row, then the toast fades (the session verifies the filled bookmark on paired frames)
    "save": Kind("save", "write", has_media, (Step("tap:save_icon", ("*feed", SAVE_SHEET), 2.0), Step("until", ("*feed",), 4.0)),
                 "saved", ("intentional",)),
    # app-level detours (Ian 10-19: "moves around the app, not just the home page"): between posts, home → … → home
    "stories": Kind("stories", "detour", has_story_tray,
                    (Step("tap:story_ring", (STORIES,), 3.0), Step("dwell:watch", (STORIES,), 0.0, "story"),
                     Step("tap:next_zone", (STORIES,), 2.0, "story", repeat="stories"),
                     Step("drag:close", (HOME["instagram"],), 3.0)), "feed_back", start="home"),
    "explore": Kind("explore", "detour", on_home,
                    (Step("tap:search_tab", (EXPLORE,), 3.0), Step("dwell:glance", (EXPLORE,), 0.0, "grid"),
                     Step("drag:grid_up", (EXPLORE,), 2.0, "grid", repeat="explore_scroll"),
                     Step("tap:grid_tile", (EXPLORE_VIEWER,), 3.0), Step("dwell:watch", (EXPLORE_VIEWER,), 0.0, "reels"),
                     Step("back", (EXPLORE,), 3.0), Step("tap:home_tab", (HOME["instagram"],), 3.0)), "feed_back", start="home"),
    "reels_tab": Kind("reels_tab", "detour", on_home,
                      (Step("tap:reels_tab", (REELS_TAB,), 3.0), Step("dwell:watch", (REELS_TAB,), 0.0, "reels"),
                       Step("drag:reel_next", (REELS_TAB,), 2.0, "reels", repeat="reels"),
                       Step("tap:home_tab", (HOME["instagram"],), 3.0)), "feed_back", start="home"),
}
# the like's two gestures (DIRECTOR_GOAL_DRAFT: a double-tap on the media or the heart button, at a drawn ratio, params.like_heart):
# one kind, verified the same way on paired frames (session.like); only the step's target differs. The heart toggles: it is
# tapped only after the paired frame showed it unliked
LIKE_GESTURES = {"double_tap": KINDS["like"],
                 "heart": Kind("like", "write", has_media, (Step("tap:heart", ("*feed",), 1.5),), "heart_red", ("intentional",))}


def allowed_screens(kind: Kind | None, step_index: int | None, app: str) -> tuple[str, ...]:
    """The screens a gesture may land on right now: the feed, plus the active kind's declared screens for its step (S3)."""
    if kind is None or step_index is None:
        return FEED.get(app, ())
    return screens_of(kind.steps[min(step_index, len(kind.steps) - 1)], app)


def screens_of(step: Step, app: str) -> tuple[str, ...]:
    """A step's declared screens, '*feed' spelled out for the app."""
    return tuple(s for e in step.expect for s in (FEED.get(app, ()) if e == "*feed" else (e,)))


def needs(kind: Kind, app: str) -> list[tuple[str, ...]]:
    """The screen records a kind needs on this app: per step, the screens of this app it names (one of each group: the
    alternatives it may land on). A step that only stays on the feed ('*feed') needs none; a step naming no screen of this
    app can never land here (an empty group, never satisfied)."""
    out: list[tuple[str, ...]] = []
    for st in kind.steps:
        mine = tuple(e for e in st.expect if e.startswith(app + "."))
        if (mine or "*feed" not in st.expect) and mine not in out:
            out.append(mine)
    return out


def kind_ready(kind: Kind, store_ids, app: str) -> bool:
    """May this kind run where looks are required (live)? Every screen it needs has a record in the screen store (a screen
    without one is never recognised, so its step would time out off the feed), matched as the walker matches a look (equal,
    or the record's id extends the declared one), and none of its targets is still unproven live (UNPROVEN_LIVE)."""
    return not any(st.action.partition(":")[2] in UNPROVEN_LIVE for st in kind.steps) and \
        all(any(i == e or i.startswith(e) for i in store_ids for e in group) for group in needs(kind, app))


FOLLOW_OPEN = Kind("follow_open", "detour", has_header_handle, (Step("tap:header_handle", PROFILES, 3.0), Step("dwell:read", PROFILES, 0.0, "read")),
                   "none", ("intentional",))
FOLLOW_BACK = Kind("follow_back", "detour", has_header_handle, (Step("dwell:glance", PROFILES, 0.0, "glance"), Step("back", ("*feed",), 3.0)),
                   "feed_back", ("intentional",))
