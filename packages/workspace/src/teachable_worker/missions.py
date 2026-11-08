"""Read-only phone missions (L2, tw-phone) for the desktop workspace (INTERFACES §Workspace): `account_check` and
`feed_scout` on Instagram, driven step by step from here on the deployed phonebridge daemon's own primitives (look,
tap, type, scroll, launch, home: its humanizer shapes every input). No new driver, nothing published, liked, followed or
messaged. Taps come only from an allow-list: the Home and Search tabs, the search bar, a result row whose text is
exactly the handle asked for, and the feed's own flick. Every input is a `ctx.step` with its pace, every claim carries
the frame it was read from (`frame_id`, `observed_at`), a platform stop text ends the mission with the phone left alone.

`session(ctx)` is the work session workspace.session_hook uses: a private deployment's own (`_deployment.py`, which
the public release doesn't have), else the standalone one (standalone.py: the local device lease).

What it is not: no learned pace (pauses are drawn from fixed reading-time bounds below, the daemon's humanizer shapes
taps/typing/flicks), no share-link readback (evidence is the frame), recent posts are the profile grid's frame (thumbnails
carry no text)."""

from __future__ import annotations

import inspect
import json
import os
import random
import re
import time
import uuid
from pathlib import Path

from . import standalone, store, workspace
from .store import data_dir

try:  # a private deployment's account map and work session; the public release has none
    import teachable_worker._deployment as _site
except ModuleNotFoundError as e:
    if e.name != "teachable_worker._deployment":
        raise
    _site = None

IG = "com.burbn.instagram"
ACCOUNTS = json.loads(os.environ.get("TW_PHONE_ACCOUNTS") or "null") or (  # device -> the account it is signed in to
    {} if standalone.enabled() or _site is None else dict(_site.ACCOUNTS))
TTL = 300
TABS = {"home": (0.10, 0.931), "search": (0.70, 0.931)}  # IG 451-454 tab bar (feedagent instagram.home record, measured)
BACKS = 6  # edge swipes back to a recognised tab screen (profile -> keyword page -> results -> search -> grid)
TRIES = 6  # zero-input refusals tolerated per input (StaleInput: the Mac's 1.5 s frontmost check, slow on iOS 27.2)
MAX_S = 15 * 60  # a mission's hard bound (the Focus window is 20 min)
ACCOUNT_HANDLES = json.loads(os.environ.get("TW_ACCOUNT_HANDLES") or "null") or (  # account -> its handle, checked
    {} if standalone.enabled() or _site is None else dict(_site.ACCOUNT_HANDLES))    # on the phone before a mission
STOP_TEXTS = ["Try again later", "Action Blocked", "temporarily blocked from performing this action", "We limit how often",
              "We suspect automated behavior", "Confirm it's you", "Confirm Your Identity", "Help us confirm",
              "Suspicious login", "Was this you", "This wasn't me", "Features you can't use", "can't be recommended",
              "Enter the confirmation code", "Enter the code", "Your account has been restricted",
              "Your account was disabled", "We've disabled your account", "Something went wrong", "Log in", "Account Status"]
READ_S = (1.2, 0.45, 0.6, 4.0)  # a reading pause on arrival: log-normal median, sigma, bounds (s); not learned
NEVER = re.compile(r"^(follow|follow back|following|message|like|share|send|reply|comment|save|subscribe|"
                   r"not interested|add to favorites|seguir|suivre|requested|mute|block|report|restrict)\b", re.I)
HANDLE = re.compile(r"@?([A-Za-z0-9._]{1,30})(?:\s+[v⌄˅▾>])?")
COUNT = re.compile(r"^(\d[\d,.]*)\s*([KkMm]?)$")


DEMO_ROOT = Path(os.environ.get("TW_DEMO_ROOT") or getattr(_site, "DEMO_ROOT", None) or
                 data_dir().parent / "demos").expanduser()  # recorded demonstrations (private phone evidence)


class Demo:
    """A recorded demonstration as tw-teach ingests it (format a): journal.jsonl with 'in: <screen>' looks (+screen_id,
    frame_ref) and 'tap "<label>" (x,y)' / 'type <slot> (N chars)' inputs with ts; frames/NNN-in-<screen>.jpg + .ocr.json;
    the strip video (strip.mp4 + strip.json {t0, fps, ...}) from the daemon's recorder; narration.txt. Private evidence
    (it shows the phone): mode 700, never published."""

    def __init__(self, task_id: str, narration: str):
        self.dir = DEMO_ROOT / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-demo-{task_id}"
        store.private_dir(self.dir)  # private phone evidence: the folders 0700, every file 0600 from its creation
        store.private_dir(self.dir / "frames")
        store.write_private(self.dir / "narration.txt", narration.strip() + "\n", exclusive=True)
        store.write_private(self.dir / "journal.jsonl", "", exclusive=True)  # appended to below, its mode kept
        self.n, self.warning = 0, None

    def note(self, **rec) -> None:
        with open(self.dir / "journal.jsonl", "a") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), **rec}) + "\n")

    def look(self, o: dict, screen: str) -> None:
        self.n += 1
        ref = f"frames/{self.n:03d}-in-{screen}.jpg"
        store.write_private(self.dir / ref, o.get("jpeg") or b"", exclusive=True)
        store.write_private((self.dir / ref).with_suffix(".ocr.json"), json.dumps(o.get("items") or []), exclusive=True)
        self.note(msg=f"in: {screen}", screen_id=screen, frame_ref=ref, frame_id=o.get("frame_id"),
                  size=[o.get("width"), o.get("height")])


class Stop(RuntimeError):
    """A platform stop text: the mission ends, the phone left as it is."""


class ZeroInput(RuntimeError):
    """The daemon refused an input before sending anything (StaleInput, Parked): look again and re-decide."""


class Hands:
    """The daemon's input tools on this session's device lease (one controller id). Never retried here: a refusal
    with zero input sent is ZeroInput, and the mission step re-looks and re-resolves its target (`act`)."""

    def __init__(self, device: str, lease: str):
        self.device, self.lease, self.controller = device, lease, uuid.uuid4().hex
        self.b = workspace.Bridge()
        self.last = time.monotonic()
        self.started = time.monotonic()

    def __call__(self, ctx, kind: str, what: str, tool: str, **args) -> None:
        """Positional taps are never resent here (the step re-looks and re-resolves: act). Inputs that don't depend on
        where things are on screen (home, launch, a feed flick, the edge swipe back) are resent after a zero-input
        refusal, up to TRIES times."""
        for attempt in range(TRIES if tool in ("home", "launch", "scroll", "swipe") else 1):
            try:
                return self._send(ctx, kind, what, tool, **args)
            except ZeroInput as e:
                if attempt == TRIES - 1 or tool not in ("home", "launch", "scroll", "swipe"):
                    raise
                ctx.step("note", f"The phone refused that input before sending it ({str(e)[:100]}); sending it again",
                         refused=True)
                time.sleep(2.0)

    def _send(self, ctx, kind: str, what: str, tool: str, **args) -> None:
        ctx.check()  # a stop from the computer lands before the next input
        if time.monotonic() - self.started > MAX_S:
            raise RuntimeError(f"the mission's {MAX_S // 60:.0f}-minute bound is spent")
        elapsed = round((time.monotonic() - self.last) * 1000)
        self._init()
        if tool in ("tap", "scroll"):  # a recognised Instagram look just before: the daemon's frontmost evidence, so it
            try:                         # skips the DVT process list (0.4-5 s on iOS 27.2, refused past 1.5 s)
                workspace.bridge.tool("look", device=self.device, platform="instagram", settle="chrome")
            except Exception:  # noqa: BLE001 - the daemon then asks the process list as before
                pass
        out = self.b._rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": tool, "arguments": {
            "device": self.device, "lease": self.lease, "controller": self.controller, **args}}})
        res = (out or {}).get("result") or {}
        if (out or {}).get("error") or res.get("isError"):
            why = (out or {}).get("error", {}).get("message") or " ".join(c.get("text", "") for c in res.get("content", []))
            if "zero input sent" in why and any(k in why for k in ("StaleInput", "Parked")):
                raise ZeroInput(why[:300])
            raise RuntimeError(f"phonebridge {tool}: {why[:300]}")
        self.last = time.monotonic()
        if demo := getattr(ctx, "demo", None):
            if tool == "tap":
                demo.note(msg=f'tap "{what}" ({args.get("x", 0):.0f},{args.get("y", 0):.0f})', what=what, elapsed_ms=elapsed)
            elif tool == "type":
                demo.note(msg=f"type handle ({len(args.get('text', ''))} chars)", slot="handle", elapsed_ms=elapsed)
            else:
                demo.note(msg=f"{tool} {what}", args={k: v for k, v in args.items() if k != "text"}, elapsed_ms=elapsed)
        # pace_ms: VM-clock time since the previous input (our looks and reading pauses included); the daemon's
        # humanizer draws (think/press) are in its own call log
        w, h = getattr(self, "size", (None, None))
        touch = ({"at": [round(args["x"] / w, 4), round(args["y"] / h, 4)], "box": [args.get("w"), args.get("h")],
                  "target": args.get("target")} if tool == "tap" and w else {})  # the touch point, teachable (tw-teach)
        ctx.step(kind, what, pace_ms=elapsed, tool=tool, pace_source="elapsed since the previous input (VM clock)", **touch)

    def _init(self) -> None:
        if not self.b._session:
            self.b._rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "tw-missions", "version": "1"}}})
            self.b._rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def tool(self, name: str, **args):
        """A daemon tool on this session's lease and controller that isn't an input (the strip recorder); its value."""
        self._init()
        out = self.b._rpc({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": name, "arguments": {
            "device": self.device, **({"lease": self.lease, "controller": self.controller} if name == "record_start" else {}),
            **args}}})
        res = (out or {}).get("result") or {}
        if (out or {}).get("error") or res.get("isError"):
            return {"error": ((out or {}).get("error") or {}).get("message") or
                    " ".join(c.get("text", "") for c in res.get("content", []))[:200]}
        return workspace.Bridge._value(res)

    def release(self) -> None:
        try:
            self.b._rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
                "name": "release_controller", "arguments": {"device": self.device, "controller": self.controller}}})
        except Exception:  # noqa: BLE001 - the reservation goes with the lease anyway
            pass


# ---- reading the screen ------------------------------------------------------------------------------------------

def _stops() -> list[str]:
    """Instagram's stop texts: the deployment's tuned list when it has one, else STOP_TEXTS."""
    return (_site.stop_texts() if hasattr(_site, "stop_texts") else None) or list(STOP_TEXTS)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("’", "'")).strip().casefold()


def stop_text(items: list[dict], texts: list[str]) -> str | None:
    """As taskapi.stop_reason: a 1-2 word phrase on a short line only, 3+ words anywhere in a line."""
    for i in items:
        low = _norm(i.get("text", ""))
        for t in texts:
            tl = _norm(t)
            if tl in low and (len(tl.split()) >= 3 or len(low.split()) - len(tl.split()) <= 3):
                return i["text"].strip()
    return None


def _line(items, pred, band=None, h=2532):
    lo, hi = band or (0.0, 1.0)
    return next((i for i in items if lo * h <= i["y"] <= hi * h and pred(_norm(i["text"]))), None)


def _count(text: str) -> int | None:
    m = COUNT.match(text.strip())
    if not m:
        return None
    n = float(m.group(1).replace(",", ""))
    return int(n * {"k": 1e3, "m": 1e6}.get(m.group(2).lower(), 1))


def lead(text: str) -> str:
    """The handle a line leads with: the WHOLE first segment, as written (IG handles are lowercase), less a back chevron
    and a trailing badge glyph. 'natgeo • 268M followers', 'nasa v', '< natgeo' -> the handle; 'NASA Internships •
    743K followers' or 'NASA' -> '' (a display name, not @nasa: tw-teach's replay caught it on IG's Profiles list)."""
    first = re.split(r"\s[•·]\s|\s-\s", (text or "").strip())[0].strip().lstrip("<‹←@ ").strip()
    words = first.split()
    while len(words) > 1 and len(words[-1]) <= 2:  # a verified badge read as 'v', '0', '©', '✓'
        words.pop()
    return words[0] if len(words) == 1 and re.fullmatch(r"[a-z0-9._]{1,30}", words[0]) else ""


def read_profile(items: list[dict], w: int, h: int, expect: str | None = None) -> dict:
    """The header handle (top 12%) and the counts (the number just above posts / followers / following)."""
    out: dict = {"counts": {}}
    for label in ("posts", "followers", "following"):
        lab = _line(items, lambda t, label=label: t == label, h=h)
        if lab is None:
            continue
        above = [i for i in items if _count(i["text"]) is not None and abs(i["x"] - lab["x"]) < 0.08 * w
                 and 0 < lab["y"] - i["y"] < 0.06 * h]
        if above:
            num = min(above, key=lambda i: lab["y"] - i["y"])
            out["counts"][label] = {"value": _count(num["text"]), "shown": num["text"].strip()}
    head = [(lead(i["text"]), i) for i in items if 0.03 * h < i["y"] < 0.12 * h and lead(i["text"])
            and not re.fullmatch(r"\d{1,2}[:.]\d{2}", i["text"].strip())]
    if head:  # the header's title is the handle (a verified badge after it is not)
        out["handle"] = next((t for t, _ in head if expect and t == expect.casefold()), max(head, key=lambda t: t[1]["w"])[0])
    return out


TYPING = ("search.typing", "search_typing")  # recognised screens with the keyboard up


def read_counts(ctx, o: dict, expect: str, note: str, stops: list[str]) -> tuple[dict, dict]:
    """read_profile on the look; when a count is missing, ONE accurate look (the workspace's observe(accurate=True):
    the bridge's own screenshot read at Apple Vision's accurate level, text and image from that same frame). Live 11-08
    (cut a96955d, tsk_ab7297bd7e30): the fast level dropped @apple's lone '9 following'; accurate read it on the same
    stored frame. The new look counts only with the same header handle and more counts; never a guess."""
    read = read_profile(o["items"], o["width"], o["height"], expect)
    if len(read["counts"]) == 3 or "accurate" not in inspect.signature(ctx.observe).parameters:
        return read, o  # complete, or a workspace without accurate looks yet: the fast read stands (missing named)
    again = _look(ctx, f"{note}: read again at Vision's accurate level (a count was missing)", stops, accurate=True)
    better = read_profile(again["items"], again["width"], again["height"], expect)
    if better.get("handle") == read.get("handle") and len(better["counts"]) > len(read["counts"]):
        return better, again
    return read, o


def keyboard_up(items: list[dict], h: int, screen: str | None = None) -> bool:
    """The iOS keyboard is on screen: the daemon recognises a typing screen (decisive: the fast OCR level drops the keys;
    live 11-08 a taught Search-tab tap landed in the keyboard's margin), or its labelled keys, or a row of
    single-character keys, in the bottom 45%."""
    if screen and str(screen).endswith(TYPING):
        return True
    low = [_norm(i["text"]) for i in items if i["y"] > 0.55 * h]
    return any(t in ("abc", "space", "#+=", "123", "return") for t in low) or sum(len(t) == 1 for t in low) >= 6


def _wait(ctx, why: str = "reading") -> None:
    median, sigma, lo, hi = READ_S
    time.sleep(min(max(random.lognormvariate(__import__("math").log(median), sigma), lo), hi))


def _size(jpeg: bytes) -> tuple[int, int]:
    """A JPEG's width and height from its SOF marker (the look doesn't always say)."""
    i = 2
    while i + 9 < len(jpeg):
        if jpeg[i] != 0xFF:
            break
        marker, length = jpeg[i + 1], int.from_bytes(jpeg[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(jpeg[i + 7:i + 9], "big"), int.from_bytes(jpeg[i + 5:i + 7], "big")
        i += 2 + length
    return 1170, 2532  # the iPhone 12's screenshot size


def _look(ctx, note: str, stops: list[str], accurate: bool = False) -> dict:
    o = ctx.observe(note, accurate=True) if accurate else ctx.observe(note)
    if not (o.get("width") and o.get("height")):
        o["width"], o["height"] = _size(o.get("jpeg") or b"")
    if hands := getattr(ctx, "hands", None):
        hands.size = (o["width"], o["height"])
    if demo := getattr(ctx, "demo", None):
        demo.look(o, where(o))
    if hit := stop_text(o["items"], stops):
        ctx.step("note", f"Platform stop on screen: '{hit}'. Nothing more is sent.", o["frame_id"])
        raise Stop(hit)
    return o


# Screens the daemon's own recognition names (feedagent's screen store) that carry Instagram's tab bar. Text alone can't
# show the bar (icons have no text): live 11-08, another account's profile had none (its grid fills the bottom) and a
# "Home tab" tap there opened a post. A tab is tapped only on one of these, recognised on a fresh look.
TAB_SCREENS = {"instagram.home", "instagram.profile", "instagram.reels",  # never a typing screen: the keyboard covers
               "instagram.screen.instagram_explore_search_grid"}            # the bar (live 11-08: a Profile-tab tap there
                                                                            # only closed the keyboard); curated ids only


CHROME = {"recent", "see all", "for you", "profiles", "accounts", "audio", "tags", "places", "posts", "followers",
          "following", "follow", "message", "edit profile", "share profile", "your story", "messages", "requests",
          "cancel", "search"}  # Instagram's own words: never a query, a handle or a row


def old_query(items: list[dict], h: int, w: int = 1170) -> str | None:
    """The previous query the search field still holds, positively: the field's own line (the widest in the top bar) as
    one handle-like token, a search glyph read before it dropped; never a reading of the placeholder ('Q Search with Meta
    Al', 'Ci bearch with Meta Al', 'SearchwithMetaAl': live 11-08 the 'bearch' misread was taken for a query, the empty
    field was tapped and iOS opened its AutoFill menu over it). 'Cancel' beside the field, or any chrome, is never it.
    (tw-teach's teaching._old_query at 4a1fc67f, moved here as the one copy.)"""
    def label(t):  # no icon glyph read as a letter at either end, no trailing ellipsis (teaching._clean)
        toks = re.sub(r"(\.\.\.|…)$", "", (t or "").strip()).split()
        while len(toks) > 1 and len(toks[0]) == 1:
            toks = toks[1:]
        while len(toks) > 1 and len(toks[-1]) == 1:
            toks = toks[:-1]
        return _norm(" ".join(toks))
    line = max((i for i in items if 0.04 * h < i["y"] < 0.12 * h and i["x"] < 0.8 * w and label(i["text"]) not in CHROME),
               key=lambda i: i.get("w", 0), default=None)  # (the field's own region: Cancel sits at ~0.92 w)
    t = re.sub(r"^(?:[^\w@]+|[A-Za-z]{1,2}\s+)", "", (line or {}).get("text", "").strip()).strip()
    if re.fullmatch(r"@?[A-Za-z0-9._]{2,30}", t) and not any(x in t.casefold() for x in ("earch", "meta")):
        return t.lstrip("@")
    return None


def tab_screen(items: list[dict], h: int, screen: str | None = None) -> bool:
    """No text in the tab-bar band, no keyboard, no composer button on screen (taskapi's rule): necessary, never
    sufficient. Pass the look's own screen (o["screen"]): without it a keyboard the OCR missed is invisible."""
    if any(i["y"] > 0.9 * h and len(i["text"].strip()) > 2 for i in items) or keyboard_up(items, h, screen):
        return False
    return not any(_norm(i["text"]) in ("share", "next", "post", "discard", "done", "send") for i in items)


def act(ctx, o: dict, stops, resolve, note: str) -> dict | None:
    """One input resolved on the look `o`; after a zero-input refusal, look again and resolve AGAIN (never the old
    point). Returns the look it acted on, or None when the target isn't there (nothing sent)."""
    for attempt in range(TRIES):
        spec = resolve(o)
        if not spec:  # None or a falsy guard (`cond and (...)`): the target isn't there, nothing sent
            return None
        kind, what, tool, args = spec
        try:
            ctx.hands(ctx, kind, what, tool, **args)
            return o
        except ZeroInput as e:
            ctx.step("note", f"The phone refused that input before sending it ({str(e).split(': ', 1)[-1][:120]}); "
                     "looking again", refused=True)
            time.sleep(3 if "Parked" in str(e) else 2.0)  # let the Mac's frontmost-app query settle (iOS 27.2: 0.4-5 s)
            o = _look(ctx, f"Re-look ({note})", stops)
    raise RuntimeError(f"{note}: the phone refused the input {TRIES} times (nothing was sent)")


def _tab(name: str):
    def resolve(o):  # on the look's own screen: the daemon's name for THIS frame, never a newer look's (tw-verifier 11-08)
        if not tabbed(o, o.get("screen")):
            return None  # never a blind tap on a screen that doesn't carry the tab bar
        fx, fy = TABS[name]
        return "tap", f"Tap the {name.title()} tab", "tap", {"x": fx * o["width"], "y": fy * o["height"],
                                                            "w": 0.08 * o["width"], "h": 0.04 * o["height"]}
    return resolve


def _ocr(o: dict, n: int = 40) -> list:
    """The OCR of the evidence frame (text and position, rounded): what a claim was read from, for whoever checks it."""
    return [[i["text"], round(i["x"]), round(i["y"])] for i in o.get("items") or []][:n]


def _foreground(dev: str) -> str | None:
    try:
        return workspace.Bridge._value(workspace.bridge.tool("foreground", device=dev))
    except Exception:  # noqa: BLE001
        return None


def where(o: dict, screen: str | None = None) -> str:
    """Which kind of Instagram (454) screen this is, from its own text (live 11-08 frames): home, own_profile, inbox,
    search (each with the tab bar), typing (the keyboard covers the bar), other_profile, keyword, post, story, or unknown."""
    items, h = o["items"], o["height"]
    lines = [_norm(i["text"]) for i in items]
    top = [_norm(i["text"]) for i in items if i["y"] < 0.12 * h]
    has = lambda *ws: any(any(w in t for w in ws) for t in lines)  # noqa: E731
    if has("send message", "watch full reel", "reply to"):
        return "story"
    if keyboard_up(items, h, screen):
        return "typing"
    if any(t in ("for you", "profiles", "tags", "audio") for t in lines) or has("ask a follow"):
        return "keyword"
    if any(t == "posts" for t in top):
        return "post"
    if "message" in lines and any(t in ("follow", "following", "follow back", "requested") for t in lines):
        return "other_profile"
    if any(t in ("edit profile", "share profile") for t in lines):
        return "own_profile"
    if _line(items, lambda t: t == "your story", (0, 0.3), h):
        return "home"
    if "messages" in lines and has("requests", "tap to chat", "chats will appear"):
        return "inbox"
    if any("search" in t for t in top + [_norm(i["text"]) for i in items if i["y"] < 0.16 * h]) or "recent" in lines:
        return "search"
    if any(" followers" in t for t in lines) and _line(items, lambda t: bool(t), (0.05, 0.11), h):
        return "search"  # results under a typed query (the field shows the query, not "Search")
    return "unknown"


TABBED = ("home", "own_profile", "inbox", "search")  # the tab bar is on screen


def tabbed(o: dict, scr: str | None) -> bool:
    """A positive main-tab classification of this look: home / own profile / inbox from their own text, and the
    search page only when the daemon recognises it (OCR alone can't tell it from the typing screen with its keys unread)."""
    kind = where(o, scr)
    return kind in TABBED and (kind != "search" or scr in TAB_SCREENS) and tab_screen(o["items"], o["height"], scr)
BACK = (0.065, 0.082)  # the top-left chevron on detail pages (profile, keyword page, post, search typing): live 11-08


BACKABLE = ("typing", "keyword", "other_profile", "post")  # the screens with the top-left chevron (live 11-08 frames)


def _back():
    """The top-left chevron, decided on the look it's given (a re-look included), only on a screen that has one. Never on
    an unrecognised search page: IG 454's Explore grid has no chevron, and BACK there is the search bar (live 11-08)."""
    def resolve(o):
        scr = o.get("screen")
        kind = where(o, scr)
        if tabbed(o, scr) or kind not in BACKABLE:
            return None
        return "tap", f"Back ({kind})", "tap", {"x": BACK[0] * o["width"], "y": BACK[1] * o["height"],
                                                "w": 0.06 * o["width"], "h": 0.03 * o["height"]}
    return resolve


def to_tabs(ctx, stops, o: dict) -> dict:
    """Back to a screen with the tab bar: the chevron on detail pages, a swipe down on a story, at most BACKS steps;
    an unknown screen stops (no blind input). Never a horizontal swipe: IG 454 switches tabs with it (live 11-08 it
    walked from a profile to the DM inbox)."""
    for k in range(BACKS + 1):
        scr = o.get("screen")
        kind = where(o, scr)
        if tabbed(o, scr):
            return o
        if kind not in BACKABLE + ("story",) or k == BACKS:
            what = "a search page the phone didn't recognise" if kind == "search" else kind
            raise Stop(f"Instagram is on a screen I don't leave without a blind input ({what}): bring it to its home "
                       "feed and run again")
        w, h = o["width"], o["height"]
        if kind == "story":
            ctx.hands(ctx, "swipe", "Close the story (swipe down)", "swipe", x1=0.5 * w, y1=0.3 * h, x2=0.5 * w, y2=0.8 * h,
                      duration=0.3)
        else:
            if act(ctx, o, stops, _back(), "back") is None:
                raise Stop(f"the back chevron of the {kind} screen wasn't there on a fresh look: nothing sent; bring "
                           "Instagram to its home feed and run again")
        _wait(ctx)
        o = _look(ctx, "After going back", stops)
    return o


def _open_ig(ctx, stops) -> dict:
    """Instagram in front, on a recognised tab screen (edge-swiping back to one if needed)."""
    if _foreground(ctx.hands.device) != IG:
        ctx.hands(ctx, "open_app", "Open Instagram", "launch", bundle_id=IG)
        _wait(ctx)
    return to_tabs(ctx, stops, _look(ctx, "Instagram", stops))


def whoami(ctx, stops, o: dict) -> dict:
    """The signed-in account, once per session: its OWN profile (Edit/Share profile) through the profile tab, the
    header handle (top 12%) equal to the account the lease names; else the mission stops."""
    expect = ACCOUNT_HANDLES.get(ACCOUNTS.get(ctx.hands.device, ""), "")
    if not act(ctx, o, stops, lambda o: (_tab("home")(o) and ("tap", "Tap the Profile tab", "tap", {
            "x": 0.90 * o["width"], "y": 0.931 * o["height"], "w": 0.08 * o["width"], "h": 0.04 * o["height"]})), "profile tab"):
        raise Stop("not on a main tab screen: the signed-in account can't be checked")
    _wait(ctx)
    o = _look(ctx, "Own profile", stops)
    own = any(_norm(i["text"]) in ("edit profile", "share profile") for i in o["items"])
    seen = read_profile(o["items"], o["width"], o["height"], expect).get("handle")
    ctx.step("note", f"Signed in as @{seen}" if own else "Not the account's own profile", o["frame_id"], account_seen=seen,
             account_expected=expect)
    if not own or (expect and seen != expect.casefold()):
        raise Stop(f"the phone's Instagram shows @{seen} (own profile: {own}), not @{expect}: nothing more is done")
    return o


def parse_handles(raw) -> list[str]:
    """The handles as the workspace sends them: a string ("natgeo, nasa") or the UI's list (["natgeo", "nasa"]); at most
    5, each @-less."""
    parts = raw if isinstance(raw, list) else re.split(r"[,\s]+", str(raw or ""))
    return [h for h in (str(x).strip().lstrip("@") for x in parts) if h][:5]


class AccountCheck:
    """Look up named Instagram profiles through the app's own Search and read what each profile shows."""

    read_only = True
    route = "phonebridge primitives (look/tap/type) on Instagram's Search, under the device lease"
    input = {"handles": "Instagram handles to look up, comma separated (at most 5), e.g. natgeo, nasa",
             "record": "yes to record this run as a demonstration (journal, frames, phone-screen video) for teaching"}

    def available(self):
        if standalone.enabled():
            return standalone.available()
        if _site is None:
            return False, "Start the workspace with phonebridge-workspace (TW_PHONE_ADAPTER=local) to use the local device lease."
        return _site.available()

    def run(self, ctx):
        handles = parse_handles(ctx.input.get("handles"))
        if not handles or not all(re.fullmatch(r"[A-Za-z0-9._]{1,30}", h) for h in handles):
            raise ValueError("handles: Instagram handles, comma separated")
        stops, profiles = _stops(), []
        if str(ctx.input.get("record") or "").lower() in ("1", "yes", "true", "on"):
            ctx.demo = Demo(ctx.task["id"], f"Check {', '.join('@' + h for h in handles)}'s profile: open Instagram, the "
                            "Search tab, type the handle, open the account (the Profiles chip if a keyword page opens), "
                            "read the handle, posts, followers and following. Read-only: never Follow or Message.")
        try:
            with ctx.session():
                o = whoami(ctx, stops, _open_ig(ctx, stops))
                for h in handles:
                    rec, o = self.one(ctx, h, stops, o)
                    profiles.append(rec)
                ctx.hands(ctx, "tap", "Back to the Home Screen", "home")  # Instagram stays where it is
        except Stop as e:
            return {"status": "stopped", "stop": str(e), "profiles": profiles}
        read = sum(1 for x in profiles if x.get("found"))
        if demo := getattr(ctx, "demo", None):
            demo.note(msg="result", profiles=[{k: v for k, v in x.items() if k != "ocr"} for x in profiles])
        return {"status": "done" if read == len(profiles) else "partial", "summary": f"{read} of {len(profiles)} profiles read",
                **({"demo_session": str(ctx.demo.dir)} if getattr(ctx, "demo", None) else {}),
                **({"warning": ctx.demo.warning} if getattr(getattr(ctx, "demo", None), "warning", None) else {}),
                "profiles": profiles,
                "limits": "counts as the profile shows them; recent posts = the grid's visible text in the profile frame; "
                          "the search leaves the query in the account's recent searches"}

    def one(self, ctx, handle: str, stops, o: dict) -> tuple[dict, dict]:
        """Search -> the row led by the handle -> its profile, verified by the header -> read. Returns (record, look)."""
        hc = handle.casefold()
        miss = lambda o, why: ({"handle": handle, "found": False, "why": why, "frame_id": o["frame_id"],  # noqa: E731
                                "observed_at": o["captured_at"], "ocr": _ocr(o)}, o)
        o = to_tabs(ctx, stops, o)  # the previous handle left a profile open
        if not act(ctx, o, stops, _tab("search"), "search tab"):
            return miss(o, "not on a main tab screen: no Search tab tap")
        _wait(ctx)
        o = _look(ctx, "Search", stops)
        # a result row: the handle as a short title ("nasa ✓") or a subtitle ("natgeo • 268M followers"), never a
        # paragraph that starts with it (IG's AI summary "NASA (@ nasa), the U.S. space agency…": live 11-08 a tap there
        # opened a story viewer)
        row = lambda o: next((i for i in o["items"] if 0.12 * o["height"] <= i["y"] <= 0.9 * o["height"]  # noqa: E731
                              and lead(i["text"]) == hc and (len(i["text"]) <= len(hc) + 4 or "followers" in i["text"])), None)
        if row(o) is None:  # the query isn't there yet: focus the field, clear an old query, type
            field = _line(o["items"], lambda t: True, (0.04, 0.11), o["height"])
            if not keyboard_up(o["items"], o["height"]):
                if field is None or NEVER.match(field["text"].strip()):
                    return miss(o, "no search field on the Search tab")
                o = act(ctx, o, stops, lambda o: ("tap", "Tap the search field", "tap", {
                    "x": 0.45 * o["width"], "y": field["y"], "w": 0.5 * o["width"], "h": field.get("h", 60)}), "field")
                _wait(ctx)
                o = _look(ctx, "Search field", stops)
            focused = keyboard_up(o["items"], o["height"], o.get("screen"))  # the fast OCR may drop single-letter keys
            if focused and old_query(o["items"], o["height"], o["width"]):  # a previous query (never the placeholder): its (×)
                act(ctx, o, stops, lambda o: old_query(o["items"], o["height"], o["width"]) and ("tap", "Clear the previous search",
                    "tap", {"x": 0.795 * o["width"], "y": 0.084 * o["height"], "w": 0.05 * o["width"], "h": 0.03 * o["height"]}),
                    "clear")
                o = _look(ctx, "Search field cleared", stops)
            if not focused:
                return miss(o, "the search field didn't take focus")
            ctx.hands(ctx, "type", f"Type {handle}", "type", text=handle)
            time.sleep(2.0)  # results load
            o = _look(ctx, f"Results for {handle}", stops)
        shown_row = row(o)  # "natgeo • 268M followers": what the search showed about the account
        search_row = shown_row and {"text": shown_row["text"].strip(), "frame_id": o["frame_id"], "observed_at": o["captured_at"]}
        profile = False
        for attempt in range(2):  # the row; if it opens IG's keyword page, its Profiles chip, then the account's row
            if attempt and keyword_page:
                if not act(ctx, o, stops, lambda o: (c := _line(o["items"], lambda t: t == "profiles", (0.08, 0.2),
                                                                o["height"])) and ("tap", "Tap the Profiles chip", "tap", {
                        "x": c["x"], "y": c["y"], "w": c.get("w", 150), "h": c.get("h", 60)}), "profiles chip"):
                    break
                _wait(ctx)
                o = _look(ctx, "Profiles", stops)
            # the right half of the row's text: the humanizer spreads a tap over its box, and the box's left edge sits
            # next to the avatar, whose story ring opens the story (live 11-08: @nasa twice)
            if not act(ctx, o, stops, lambda o: where(o) in ("search", "typing", "keyword") and (r := row(o))
                    and not NEVER.match(r["text"].strip()) and (
                    "tap", f"Open {handle}", "tap", {"x": r["x"] + 0.2 * r.get("w", 200), "y": r["y"],
                                                     "w": 0.4 * r.get("w", 200), "h": r.get("h", 50)}), "result row"):
                return miss(o, f"no result row led by '{handle}'")
            _wait(ctx)
            o = _look(ctx, f"{handle}'s profile", stops)
            read = read_profile(o["items"], o["width"], o["height"], handle)
            lines = {_norm(i["text"]) for i in o["items"]}
            if any("send message" in t or "watch full reel" in t for t in lines):  # a story viewer: close it, no more
                ctx.hands(ctx, "swipe", "Close the story (swipe down)", "swipe", x1=0.5 * o["width"], y1=0.3 * o["height"],
                          x2=0.5 * o["width"], y2=0.8 * o["height"], duration=0.3)
                return miss(o, "the row opened a story, not the profile (closed at once)")
            # another account's profile: its handle in the header and a Message button beside Follow/Following. IG's
            # keyword page has an account card with Follow only, under For you/Profiles/Audio/Tags chips and "Ask a
            # follow up..." (live 11-08: it passed for a profile until this)
            keyword_page = bool(lines & {"for you", "profiles", "tags", "audio"}) or any("ask a follow" in t for t in lines)
            profile = read.get("handle") == hc and "message" in lines and not keyword_page
            if profile:
                break
        if not profile:
            return miss(o, "the row didn't open a profile")
        read, o = read_counts(ctx, o, handle, f"{handle}'s profile", stops)
        h = o["height"]
        tabs_y = max((i["y"] for i in o["items"] if _norm(i["text"]) in ("follow", "following", "message")), default=0.35 * h)
        grid = [i["text"].strip() for i in o["items"] if i["y"] > tabs_y + 0.12 * h and len(i["text"].strip()) > 3][:10]
        name = next((i["text"].strip() for i in o["items"] if 0.1 * h < i["y"] < 0.18 * h  # below the header's handle
                     and len(re.sub(r"[^A-Za-z]", "", i["text"])) >= 3), None)  # ("NASA" for @nasa is a name)
        summary = (", ".join(f"{v['shown']} {k}" for k, v in read["counts"].items()) or
                   (search_row["text"].split("•", 1)[-1].strip() + " (search row)" if search_row and "•" in search_row["text"]
                    else "no counts shown"))
        ctx.step("result", f"@{handle}: {summary}", o["frame_id"], handle_seen=read.get("handle"), ocr=_ocr(o))
        return ({"handle": handle, "found": True, "handle_seen": read.get("handle"), "name": name,
                 "counts": read["counts"] or None, "counts_note": None if read["counts"] else
                 "this Instagram version shows no counts on another account's profile header",
                 "search_row": search_row, "recent_posts_text": grid,
                 "frame_id": o["frame_id"], "observed_at": o["captured_at"], "ocr": _ocr(o)}, o)


UNDERSTAND = ("Request: {request}\nThis is one post on an Instagram feed. Is it relevant to the request? Judge only "
              "what is visible (image and text). Reply JSON only: {{\"relevant\": true|false, \"reason\": \"<one short "
              "sentence naming what you see>\", \"author\": \"<the post's handle if visible, else null>\"}}")


def understand(dev: str, request: str) -> dict | None:
    """The daemon's local vision model (Qwen on the Mac, read-only) on the current frame; None when it can't answer."""
    try:
        r = workspace.Bridge._value(workspace.bridge.tool("understand", device=dev,
                                                           prompt=UNDERSTAND.format(request=request), max_tokens=120))
    except Exception:  # noqa: BLE001 - not a read the workspace allows, or the model is down: keyword fallback
        return None
    raw = (r or {}).get("parsed") or {}
    if not isinstance(raw, dict) or "relevant" not in raw:
        try:
            raw = json.loads(re.search(r"\{.*\}", (r or {}).get("raw") or "", re.S).group(0))
        except (AttributeError, ValueError):
            return None
    return {"relevant": bool(raw.get("relevant")), "reason": str(raw.get("reason") or "")[:200],
            "author": raw.get("author"), "model": (r or {}).get("model"), "ms": (r or {}).get("ms"),
            "frame_seq": ((r or {}).get("frame") or {}).get("seq")}


class FeedScout:
    """Browse a bounded part of the home feed and keep the posts whose visible text matches the request."""

    read_only = True
    route = "phonebridge primitives (look/scroll) on Instagram's home feed, under the device lease"
    input = {"request": "what to look for, in a few words (e.g. AI tools, new phone launches)",
             "posts": "how many feed posts to look at (1-15, default 8)"}

    def available(self):
        return AccountCheck.available(self)

    def run(self, ctx):
        request = (ctx.input.get("request") or "").strip()
        if not request:
            raise ValueError("request: what to look for")
        n = max(1, min(15, int(ctx.input.get("posts") or 8)))
        words = {w for w in re.findall(r"[a-z0-9]{3,}", request.casefold())} - {"the", "and", "for", "new", "with"}
        stops, seen, finds, shown = _stops(), [], [], set()
        try:
            with ctx.session():
                o = whoami(ctx, stops, _open_ig(ctx, stops))
                o = act(ctx, to_tabs(ctx, stops, o), stops, _tab("home"), "home tab") and _look(ctx, "Home feed", stops)
                if not o:
                    raise Stop("not on a main tab screen: no way to the home feed without a blind tap")
                for k in range(n):
                    if k:
                        ctx.hands(ctx, "swipe", "Next post", "scroll", direction="up")
                        _wait(ctx)
                        o = _look(ctx, f"Post {k + 1}", stops)
                    for m in ctx.messages(after=len(seen) - 1 if seen else -1)[-1:]:  # a refinement from the computer
                        words |= set(re.findall(r"[a-z0-9]{3,}", m["text"].casefold()))
                    text = [i["text"].strip() for i in o["items"] if 0.08 * o["height"] < i["y"] < 0.9 * o["height"]]
                    key = " | ".join(text[:6])
                    if key in shown:
                        continue
                    shown.add(key)
                    hits = sorted({w for w in words if any(re.search(rf"\\b{re.escape(w)}\\b", _norm(t)) for t in text)})
                    entry = {"post": k + 1, "lines": text[:12], "frame_id": o["frame_id"], "observed_at": o["captured_at"]}
                    seen.append(entry)
                    u = understand(ctx.hands.device, request)  # a look at the image too, not just its text
                    if u is not None and u["relevant"]:
                        finds.append({**entry, "reason": u["reason"], "judged_by": f"local vision model ({u['model']})",
                                      "author": u.get("author"), "model_frame_seq": u.get("frame_seq")})
                        ctx.step("result", f"Post {k + 1}: {u['reason']}", o["frame_id"], judged_by="vision", ms=u.get("ms"))
                    elif u is None and hits:
                        finds.append({**entry, "reason": f"its visible text mentions {', '.join(hits)}",
                                      "judged_by": "keywords over OCR (the vision model didn't answer)"})
                        ctx.step("result", f"Post {k + 1}: text mentions {', '.join(hits)}", o["frame_id"], judged_by="keywords")
                    _wait(ctx)
                ctx.hands(ctx, "tap", "Back to the Home Screen", "home")
        except Stop as e:
            return {"status": "stopped", "stop": str(e), "finds": finds, "looked_at": len(seen)}
        return {"status": "done", "request": request, "looked_at": len(seen), "finds": finds,
                "limits": "relevance: the Mac's local vision model on the frame (each find says which judged it; "
                          "keywords over OCR only when the model didn't answer); no links (no share-link readback): each "
                          "find's evidence is its frame"}


TASKS = {"account_check": AccountCheck(), "feed_scout": FeedScout()}


session = getattr(_site, "session", None)  # workspace.session_hook: the deployment's work session, if it has one
