"""The session director (BEHAVIOUR_MODEL.md): one feed session rendered from one generative model.

Per post: `look` (the daemon's recognised screen + OCR items + settle flag; the only gate, S3) → the glance decides skip/stay
from markers, motion and taste (≤ 250 ms, no pixels on the path) → `Behaviour.post_plan()` has already drawn the post's dwell
quantile, its glide kind and the engagements to try → the dwell is the clock: understanding runs inside it in the background
(every stay ≥ `describe_min_s`, both modes), engagement kinds walk their data contracts (`kinds`, `primitives.walk`) when their
affordance is on screen, a like goes through the write gate + the pacer and is verified before/after under the confirmed
handle → `glide`: a planned drag (length, duration, release speed) to the next post's header; the next look measures the
travel, learns a per-session gain and journals aligned / misaligned / unknown, one nudge when misaligned.

Every number comes from `params.Params` (persona.json → operator.toml) or is a drawn reaction inside `until`; there are no
sleeps outside dwell and until. The phone is anything with `screen/ocr/swipe/tap` (+ `look` live). `now`/`sleep` are
injectable so sims and replays run instantly. Evidence (PLAN §9.6): `events.jsonl`, `frames/NNNN.png`, `session.json`.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from .. import ocr as vision
from ..humanize import Humanizer
from . import params as pm
from . import primitives as pr
from .actions import red_share
from .behaviour import Behaviour
from .content_vlm import CachedVLM, WriteGate
from .cues import BLIND, CueEngine, decide, on_feed
from .flick import FlickModel
from .kinds import EXITS, HOME, KINDS, LIKE_GESTURES, Kind, kind_ready, needs
from .persona import Persona

FEED_SCREENS = {"instagram": ("instagram.home", "instagram.reels", "instagram_home", "instagram_reels"),
                "tiktok": ("tiktok.feed", "tiktok.foryou", "tiktok_feed", "tiktok_home"),
                "x": ("x.home",)}  # the only screens a glide may land on (S3)
HOME_TAB = {"instagram": (0.10, 0.929), "tiktok": (0.10, 0.955), "x": (0.10, 0.93)}  # the Home tab: a store target for navigate(home)
TASK_KIND = Kind("task", "task", lambda look, app: True, (), "none")  # detour(task) as come_back sees it (run/do.py walks the task)
TASK_TAIL_S = (8.0, 40.0)  # after the task: a drawn glance at the feed before the visit ends
NO_HOME_TAP = ("stories", "comments", "composer", "login", "captcha", "alert", "sheet", "settings", "about", "status", "wall", "update", "verification", "challenge", "warning", "block")  # (the story viewer has no tab bar: the Home tab's place is its reply field)
WRITE_KINDS = ("like", "save", "follow", "comment")  # the kinds whose tap is a write: budget, final re-check, attempt row
COUPLED = ("expand", "comments_peek")  # kinds that are signs of interest: they follow the post's class (Director.coupled)
COUPLED_NEUTRAL = 0.3  # their weight on a NEUTRAL (or unclassified) post relative to IN
STOP_KINDS = ("stop", "alert", "popup", "warning", "challenge")  # a look record of these kinds ends the session: zero gestures, the frame, a notify
MAX_STREAM_FALLBACKS = 3  # looks served by DVT stills while the stream is down: a reliability event, not a mode (L1)
STUCK_LOOKS = 3  # the same screen this many posts in a row: the feed isn't moving
HEART_RED = 0.08  # a liked heart is solid red: share of red pixels around it
MIN_TAP_GAP_S = 0.75  # S2: single taps at least this far apart (controller-verify's DOUBLE_TAP_S is 0.6)
TAP_BOX = 40.0  # px: the target box every director tap passes (bounds the daemon's scatter)
STILL_GAP_S = 0.25  # between the two looks that must agree the page stopped after a glide
WAIT_JOURNAL_S = 0.5  # any wait beyond a call's own need longer than this is journaled with its cause
STALE_LIMIT = 3  # stale refusals in a row before the session stops (kind "stale")
NUDGE_RELEASE = 0.05  # a corrective nudge lifts almost still
SCATTER = 0.4  # humanize.tap_clip: the finger lands within this fraction of the box from its centre


@dataclass
class FeedPost:
    """One post of a simulated feed (tests and replays): its screenshot, OCR items, whether it moves."""
    kind: str
    png: bytes
    items: list
    moving: bool = False


class StaleInput(Exception):
    """The daemon refused an input as stale (its pre-input checks ran long or saw a backend recovery; fast-eyes 4ab18a2):
    ZERO input was sent. The caller takes a fresh look and re-decides; it never resends the old coordinates."""


class GestureRefused(Exception):
    """The director refused its own gesture (an avoid zone, an unclear path): the kind is abandoned, nothing was sent."""


class Stopped(Exception):
    def __init__(self, kind: str, reason: str):
        super().__init__(f"{kind}: {reason}")
        self.kind, self.reason = kind, reason


def about(items: list[dict], n: int = 120) -> str:
    return " / ".join(i["text"] for i in items if i.get("text"))[:n]


def caption_words(items: list[dict], h: float) -> int:
    """Words a person could read on this post: text items in the content band that aren't a handle, a count or chrome."""
    n = 0
    for i in items:
        t = (i.get("text") or "").strip()
        if len(t) < 12 or " " not in t or not 0.1 * h < i.get("y", 0) < 0.9 * h:
            continue
        n += len(t.split())
    return n


def screen_store_ids() -> set[str]:
    """The screens a look can recognise: the screen store's record ids (the repo's records and the learned candidates);
    empty when the store can't be read, so every kind that leaves the feed stays off."""
    try:
        from ..screens.store import RECORD_DIRS, Store

        return set(Store(records_dirs=RECORD_DIRS).records)
    except Exception:
        return set()


def observation_seq(observation) -> int | None:
    """The frame seq of a phonebridge_client Observation: its frame_seq (task-agent's addition) or the seq its frame_id starts
    with ("<seq>:<sha>"); None for a still with no seq. Write rows keep it (frame_seq == pixels_seq: controller-verify W1)."""
    seq = getattr(observation, "frame_seq", None)
    if seq is None:
        head = str(getattr(observation, "frame_id", "") or "").split(":", 1)[0]
        seq = int(head) if head.isdigit() else None
    return seq


def _profile_is(who: str | None, items: list[dict], h: float) -> bool:
    """Is this the profile of `who`? Its own handle line in the top band (y < 0.15 h), the whole token matched (an @who at the
    start of someone else's bio line further down never counts: integrate's review)."""
    import difflib

    if not who:
        return False
    want = who.lower().lstrip("@")
    for i in items:
        t = (i.get("text") or "").strip().lower().lstrip("@")
        if i.get("y", h) < 0.15 * h and t and difflib.SequenceMatcher(None, want, t.split()[0].rstrip(",:.")).ratio() >= 0.9:
            return True
    return False


def _author_on(who: str, heads: list[str]) -> bool:
    """The confirmed author is the handle of one of these header lines (its first token, fuzzily: OCR drops a character)."""
    import difflib

    return any(difflib.SequenceMatcher(None, who.lower(), t.split()[0].lower()).ratio() >= 0.8 for t in heads if t.split())


def git_sha() -> str | None:
    """The commit this code runs from, for RESULT.md (trial 1 recorded none): git at this file's checkout (launchd's bare
    PATH included), else None (an installed copy)."""
    for git in ("git", "/usr/bin/git", "/opt/homebrew/bin/git"):
        try:
            out = subprocess.run([git, "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return out
        except Exception:
            continue
    return None


class Director:
    def __init__(self, phone, device: str, *, app: str, persona: Persona, evidence: Path | None = None, read_only: bool = True,
                 vlm=None, pacer_ok=None, pacer_take=None, cues: CueEngine | None = None, lease: str | None = None,
                 now=None, sleep=None, rng: random.Random | None = None, humanizer: Humanizer | None = None,
                 first_write_after_s: float = 60.0, like_after_fraction: float = 0.5, identity: str | None = None, comment_write: str = "preview",
                 motion_gap_s: float = 0.15, frame_every: int = 1, via: str = "sim", explore=None, taste=None, require_look: bool | None = None,
                 mode: str = "passive", params: pm.Params | None = None, gallery: list[str] | None = None, intention=None,
                 classify=None, drafter=None, day: date | None = None, store_ids=None, task=None, task_at: int = 1, **legacy):
        self.phone, self.device, self.app, self.persona, self.read_only = phone, device, app, persona, read_only
        from .feedmaps import FEEDMAPS, kinds_for

        self.kinds = kinds_for(app) if app in FEEDMAPS else KINDS  # the kinds this app's feed map offers (X: its own read-only detours)
        self.store_ids = store_ids  # the screen store's record ids (live: kinds needing a screen without a record stay off); None = read the store
        self.detour_rates: dict[str, float] = {}  # app-level detour kind → this session's per-post rate (drawn at the start)
        self.evidence, self.lease, self.identity = Path(evidence) if evidence else None, lease, identity
        # a distinct stream per session (the persona's seed alone replayed the same draws every session: naturalness audit),
        # journaled so a session replays exactly
        self.rng_seed = None if rng is not None else (persona.seed * 1_000_003 + random.SystemRandom().randrange(2**31)) % 2**63
        self.rng = rng or random.Random(self.rng_seed)
        # app-level detours draw from their own stream, derived from the session's without drawing from it: whether and where
        # he wanders never shifts the post-by-post draws (a session renders the same posts with or without its detours)
        self.detour_rng = random.Random(repr(self.rng.getstate()[1][:16]))
        self.write_rng = random.Random(repr(self.rng.getstate()[1][16:32]))  # follow decisions on their own stream: the post-to-post draws stay as they were
        self.human = humanizer or Humanizer(seed=self.rng.randrange(2**31))
        self.human.side = persona.side
        self.now = now or time.monotonic
        self.sleep = sleep or asyncio.sleep
        self.cues = cues or CueEngine()
        self.params = params or pm.load(None)
        self.params_hash = pm.digest(self.params)  # provenance: the account's configured centre this session ran around (naturalness T8a)
        self.variation = None
        if day is not None:  # live: this day's and this session's parameters around that centre (pm.vary); the start row's
            # `variation` {date, session_key, day: {path: multiplier}, session: {path: multiplier}, hash: the session's effective params}
            session_key = self.rng_seed if self.rng_seed is not None else self.rng.randrange(2**63)
            self.params, draw = pm.vary(self.params, f"{persona.id}:{persona.seed}:{day.isoformat()}", str(session_key), getattr(persona, "traits", None))
            self.variation = {"date": day.isoformat(), "session_key": session_key, **draw, "hash": pm.digest(self.params)}
        self.intention = intention  # intent.Intention: his words for this session (mode, what counts as IN, whether IN posts are finds)
        self.finds: list[dict] = []
        self.drafter = drafter  # comments.Drafter: IN posts → comment drafts for Ian to vet when the intention asks (nothing posts)
        self.comment_posts: list[dict] = []
        self.classify = classify  # (items, h) -> (cls IN|NEUTRAL|AGAINST|AD|LIVE, scores, ms): lane classifier's fast read against his written lines
        self.mode = mode if mode in self.params.modes else "passive"
        self.beh = Behaviour(self.params, self.mode, app, self.rng)
        self.gallery = list(gallery) if gallery else None  # a forced-kind gallery session: render each kind once (controller-verify's naturalness judge)
        from .behaviour import snaps

        self.flicks = FlickModel(persona.flicks[app], rng=self.rng) if snaps(app) else None  # TikTok snap: the persona's measured flick rows
        self.glide_state = pr.GlideState()
        self._release_ok = bool(getattr(phone, "planned_glides", False))  # phones that declare it take a planned release (Bridge, DaemonPhone, SimPhone); others get the plain swipe
        self.pacer_ok = pacer_ok or (lambda a: False)  # the check (no spend): candidates, eligibility
        self.pacer_take = pacer_take  # the spend, once per write right before its final look and tap (live: authhub pace allow --take)
        self._write_ctx = None  # the write in progress: {who, caption, guard(items, png) -> bool}
        self.gate = WriteGate(first_write_after_s, like_after_fraction, comment_write)
        self.motion_gap_s, self.frame_every, self.via = motion_gap_s, frame_every, via
        self.require_look = (via == "daemon") if require_look is None else require_look  # S3: live, the recognised screen must come from the daemon's look
        self.taste = taste  # semantic.TasteModel / InterestModel
        self.explore = explore  # detour(exploration): `await explore(director)` visits an unmapped part of the app (Ian 19:3x)
        self.task, self.task_at, self.task_result = task, task_at, None  # detour(task): `await task(director)` once, after `task_at` posts (run/do.py)
        if vlm is not None and not hasattr(vlm, "describe"):
            fn = vlm
            vlm = CachedVLM(lambda png, prompt, effort: fn(png), self.evidence / "vlm-cache.jsonl" if self.evidence else None, persona.interest)
        self.vlm = vlm
        self.events: list[dict] = []
        self.last_look: dict = {}
        self.frame_wall = None
        self._fallbacks = self._dvt_looks = 0
        self._pending = None  # (items before, Glide) until the next look measures the travel
        self.aligned = "unknown"
        self.counts = {"posts": 0, "skipped": 0, "glides": 0, "nudges": 0, "engagements": 0, "writes": 0}
        # per kind, per post opportunity (naturalness T3): eligible → selected → attempted → completed → verified; the forced
        # gallery is kept apart so it never enters ordinary rates
        self.actions: dict[str, dict[str, int]] = {}
        self._stale: list[str] = []  # consecutive stale refusals (reset by any input that goes through)
        self._sent_kind: str | None = None
        self.t_start = None
        self._frames = 0
        if self.evidence:
            (self.evidence / "frames").mkdir(parents=True, exist_ok=True)

    # --- plumbing ---------------------------------------------------------------------------------
    def session_s(self) -> float:
        return 0.0 if self.t_start is None else self.now() - self.t_start

    def ev(self, name: str, **kw) -> dict:
        e = {"t": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "s": round(self.session_s(), 3), "ev": name, **kw}
        self.events.append(e)
        if self.evidence:
            with (self.evidence / "events.jsonl").open("a") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return e

    def feed_markers(self, items: list[dict]) -> bool:
        return on_feed(self.app, items)

    def read_seconds(self, words: int) -> float:
        """Reading time for a dwell step inside a kind: the words at the persona's pace plus a drawn reaction."""
        r, _ = self.params.glide["reaction_s"].draw(self.rng)
        return r + words / self.params.reading_wps

    def dwell_for(self, what: str | None, words: int = 0, by: float | None = None) -> float:
        """A dwell inside a kind: `read` (the words at his pace), `glance`, or a drawn watch from params.dwell[what] (a story,
        the Explore grid, a reel); never past `by` (the kind's turn-back time) nor the session's end."""
        s = self.params.dwell[what].draw(self.rng)[0] if what in self.params.dwell else self.read_seconds(words if what == "read" else 0)
        for limit in (by, getattr(self, "t_end", None)):
            if limit is not None:
                s = min(s, max(0.0, limit - self.now()))
        return s

    async def look(self) -> tuple[bytes | None, list[dict]]:
        """One look. With fast-eyes: `look` first (the judged frame's OCR items, screen, settle flag, age); the pixels come
        after the decision (`pixels`). Without it: a screenshot and an OCR pass."""
        meta = await self.look_journaled() if hasattr(self.phone, "look") else None
        self.frame_wall = self.now() - (meta.get("frame_age_ms", 0) / 1000 if meta else 0.0)
        items = meta["items"] if meta and "items" in meta else None
        png = None
        if items is None:
            if hasattr(self.phone, "observe"):
                observation = await self.phone.observe(self.device, use_look=False)
                png, items = observation.png, observation.items
            else:  # deterministic replay/simulation adapters
                png = await self.phone.screen(self.device)
                items = await self.phone.ocr(self.device)
        fb = getattr(self.phone, "stream_fallbacks", 0)
        if fb > self._fallbacks:
            self._fallbacks = fb
            self.ev("look_fallback", n=fb, why=getattr(self.phone, "last_fallback", None))
            if fb >= MAX_STREAM_FALLBACKS:
                raise Stopped("stream_down", f"{fb} looks served by DVT stills: the frame stream is down (L1)")
        if png is not None:
            self._keep_frame(png)
        return png, items

    async def look_dict(self) -> dict:
        """The look as one dict for `until`: {screen, items, ...} (screen None without eyes)."""
        png, items = await self.look()
        return {**self.last_look, "items": items, "screen": self.last_look.get("screen"), "png": png}

    def _keep_frame(self, png: bytes) -> str | None:
        """Save an evidence frame; returns its path relative to the evidence dir (the view row names the frame it judged)."""
        if self.evidence and self.counts["posts"] % self.frame_every == 0:
            path = f"frames/{self._frames:04d}.png"
            (self.evidence / path).write_bytes(png)
            self._frames += 1
            self.last_frame = path
            return path
        return None

    async def pixels(self, png: bytes | None) -> bytes | None:
        """The frame's pixels, fetched AFTER the decision (evidence, understanding, image-side taste, heart verification):
        the look's own frame by seq when the daemon serves it, else the latest frame. A daemon error here is a daemon error
        like any other: it propagates to the session's terminal handler (Codex review #9), never a silent continuation."""
        self.last_frame = None
        t_call = self.now()
        if png is None and hasattr(self.phone, "frame_at") and self.last_look.get("frame_seq") is not None:
            png = await self.phone.frame_at(self.device, self.last_look["frame_seq"])  # None = unsupported (not an error)
        if png is None:
            png = await self.phone.screen(self.device)
        self.waited("frame pixels", t_call)
        self._keep_frame(png)
        return png

    async def send(self, call: str, coro):
        """One daemon input. A stale refusal ("stale: re-look … zero input sent") becomes StaleInput after it is journaled;
        STALE_LIMIT in a row stop the session. Any input that goes through resets the streak."""
        try:
            out = await coro
        except Exception as e:
            if "stale: re-look" in str(e):
                self._stale.append(str(e)[:160])
                self.ev("stale", call=call, reason=str(e)[:200], streak=len(self._stale))
                if len(self._stale) >= STALE_LIMIT:
                    raise Stopped("stale", f"{len(self._stale)} stale refusals in a row: " + " | ".join(self._stale))
                raise StaleInput(call) from None
            raise
        self._stale = []
        return out

    def count_action(self, kind: str, stage: str) -> None:
        book = self.actions.setdefault("gallery" if self.gallery is not None else "ordinary", {}).setdefault(
            kind, {"eligible": 0, "selected": 0, "attempted": 0, "completed": 0, "verified": 0})
        book[stage] += 1

    async def _describe_timed(self, png: bytes, seen: float):
        """The understanding, with the moment it answered (L3 is frame → understanding; the vlm row is written when the
        director collects it, possibly later, inside the dwell)."""
        out = await self.vlm.describe(png)
        self._vlm_answered_s = round(self.session_s(), 3)
        self._vlm_l3_ms = round((self.now() - seen) * 1000, 1)
        return out

    async def warm(self) -> None:
        """Everything the first post would otherwise pay for, before it (controller-verify R0: the first view's L2 was 536 ms vs
        139 ms later): the screen size (the deployed look carries none, so the first post fetched pixels and then ran Vision on
        them inside the decision), the taste/intention embeddings and the background, the fast classifier, the cue engine."""
        t0 = self.now()
        if not hasattr(self, "size"):
            meta = await self.look_journaled() if hasattr(self.phone, "look") else None
            if meta and meta.get("width") and meta.get("height"):
                self.size = (meta["width"], meta["height"])
            else:
                png = await self.pixels(None)
                self.size = vision.png_size(png) if png else (1170, 2532)
        if self.taste is not None:
            self.taste.score("warm up the reference lines and the background")
        if self.classify is not None:
            try:
                self.classify([{"text": "warm up", "x": 500, "y": 1200, "w": 300, "h": 40}], self.size[1])
            except Exception as e:
                self.ev("classify_error", error=str(e)[:160], when="warm-up")
        await self.cues.judge([], False, png=None, h=self.size[1])
        self.ev("warm", s_spent=round(self.now() - t0, 3), size=list(self.size))

    def waited(self, cause: str, t0: float, expected: float = 0.0) -> None:
        """Journal any wait over WAIT_JOURNAL_S beyond what the call itself needs (a gesture's own duration): the dwell is the
        only intended wait, so every other one is named (controller-verify: no unjournaled wait; R1 attempt 2 sat 7.8 s inside
        a daemon swipe call)."""
        extra = self.now() - t0 - expected
        if extra > WAIT_JOURNAL_S:
            self.ev("wait", wait_s=round(extra, 2), cause=cause)

    async def look_journaled(self) -> dict | None:
        t = time.perf_counter()
        t_call = self.now()
        meta = await self.phone.look(self.device, self.app)
        self.waited("daemon look", t_call, expected=0.6)  # a look includes the eyes' settle wait (≤ ~0.5 s)
        rt_ms = round((time.perf_counter() - t) * 1000, 1)
        self.last_look = meta or {}
        if self.require_look and not meta:  # live: every gesture is gated on a recognised screen, not only the first (Codex review #5)
            raise Stopped("no_look", "the daemon's look returned nothing: no gesture without a recognised screen")
        if meta:
            self.ev("look", rt_ms=rt_ms, **{k: v for k, v in meta.items() if k not in ("items", "frame_jpeg", "image_jpeg")}, n_items=len(meta.get("items") or []))
            if meta.get("source") == "dvt":
                self._dvt_looks += 1
                self.ev("look_fallback", n=self._fallbacks + self._dvt_looks, why="look served by a DVT still (no live frame)")
                if self._fallbacks + self._dvt_looks >= MAX_STREAM_FALLBACKS:
                    raise Stopped("stream_down", f"{self._fallbacks + self._dvt_looks} looks served by DVT stills: the frame stream is down (L1)")
        return meta

    def screen_ok(self, items: list[dict]) -> tuple[bool, str]:
        """S3: may a feed gesture land here? Live: the look's `screen` must be a feed screen of the leased app; a `kind: stop`
        record or an iOS alert ends the session. Without eyes: the OCR feed markers."""
        meta = self.last_look
        if meta:
            screen, kind = meta.get("screen"), meta.get("kind")
            if kind in STOP_KINDS or (screen or "").startswith("ios.") or ".stop." in (screen or ""):
                raise Stopped("stop_screen", f"{screen or kind}: {about(items, 120)}")  # S3: a stop record (action block, login, captcha, 2FA, verification) never gets a gesture
            if screen is None:
                return False, "unknown screen"
            ok = any(screen == f or screen.startswith(f) for f in FEED_SCREENS.get(self.app, ())) or screen.startswith(f"{self.app}.feed")
            return ok, screen
        if self.require_look:
            raise Stopped("no_look", "no recognised screen: OCR markers alone never gate a live gesture")
        return on_feed(self.app, items), "ocr feed markers"

    async def gate_look(self, allowed: tuple[str, ...]) -> dict | None:
        """A fresh look right before a gesture: a stop/alert/iOS record ends the session (screen_ok raises); the gesture may
        go only when the current screen is one of `allowed` (the screens the kind's previous step led to, or the feed).
        Returns the look, or None when the screen is not allowed (the caller abandons the kind)."""
        look = await self.look_dict()
        ok, where = self.screen_ok(look.get("items") or [])
        screen = look.get("screen")
        if screen is None and not self.require_look:  # a sim without eyes: the OCR feed markers stand in for the feed
            return look if ok and any(a in FEED_SCREENS.get(self.app, ()) for a in allowed) else None
        if screen is not None and any(screen == a or screen.startswith(a) for a in allowed):
            return look
        self.ev("gate_refused", screen=screen, allowed=list(allowed))
        return None

    async def moving(self, png: bytes | None) -> bool:
        if "settled" in self.last_look:
            return self.last_look.get("settled") is False
        if png is None:
            return False
        await self.sleep(self.motion_gap_s)
        again = await self.phone.screen(self.device)
        return again != png and vision.change(png, again) > 0.0

    # --- gestures (every one journaled with its kind/step/expect; S2 avoid zones checked by the caller) ----------------------------
    async def tap(self, x: float, y: float, what: str, kind: str | None = None, step: int | None = None, expect=None, w: float = 0, h: float = 0,
                  double: bool = False, zone_exempt: bool = False) -> None:
        """One tap, or the like's double tap (`double`: the daemon's one-input double_tap). The target box w × h bounds the
        daemon's scatter (it lands within 0.4 of the box; w = h = 0 meant its 132-px default, up to ±53 px: Codex review #2),
        and the whole scatter box must clear the look's avoid zones (`zone_exempt` only for the Home tab, which lives in the
        tab bar by design, and the detours' tab-bar targets). S2: single taps never closer than MIN_TAP_GAP_S."""
        w, h = (w or TAP_BOX), (h or TAP_BOX)
        margin = SCATTER * max(w, h)
        if not zone_exempt and self.last_look and pr.in_avoid(self.last_look, x, y, *self.size, margin=margin):
            self.ev("tap_refused", what=what, x=round(x), y=round(y), why="the scatter box touches an avoid zone", kind=kind)
            raise GestureRefused(what)
        await self.tap_gap()
        if kind in WRITE_KINDS:  # integrate's review: a row BEFORE the input, so a stop or a daemon error after it still leaves it
            self.write_attempt(kind, what)
        t_call = self.now()
        if double:
            await self.send("double_tap", self.phone.double_tap(self.device, x, y, w, h, lease=self.lease))
        else:
            await self.send("tap", self.phone.tap(self.device, x, y, w, h, lease=self.lease))
        self.waited("double_tap" if double else "tap", t_call, expected=0.4 if double else 0.15)
        if kind and kind not in ("glide", "navigate"):
            self._sent_kind = kind
        self._last_tap_t = self.now()
        self.ev("tap", what=what, x=round(x), y=round(y), box=[round(w), round(h)], double=double or None, kind=kind, step=step, expect=list(expect) if expect else None)

    async def type(self, text: str, what: str, kind: str | None = None, step: int | None = None, submit: bool = False) -> None:
        """type(slot) (task-agent design §3.3): the daemon's `type`, humanised keystrokes (Humanizer.keystrokes: ~150 ms key
        gaps, the odd slip backspaced), journaled like every input. The caller reads the field back before any submit."""
        t_call = self.now()
        await self.send("type", self.phone.type(self.device, text, submit=submit, lease=self.lease))
        self.waited("type", t_call, expected=0.3 * len(text) + 1.0)
        if kind and kind not in ("glide", "navigate"):
            self._sent_kind = kind
        self.ev("type", what=what, chars=len(text), submit=submit or None, kind=kind, step=step)

    async def take_budget(self, action: str) -> bool:
        """One unit of the action's budget, spent now (integrate's review: a reservation per write, no default permit). Off
        the event loop (live it is an ssh round trip); journaled here, on the loop."""
        import asyncio

        fn = self.pacer_take or self.pacer_ok
        ok = bool(await asyncio.to_thread(fn, action))
        self.ev("pacer", action=action, ok=ok)
        return ok

    async def tap_gap(self) -> None:
        """S2: single taps never closer than MIN_TAP_GAP_S."""
        last = getattr(self, "_last_tap_t", None)
        if last is not None and self.now() - last < MIN_TAP_GAP_S:
            await self.sleep(MIN_TAP_GAP_S - (self.now() - last) + self.params.glide["reaction_s"].draw(self.rng)[0] * 0.2)

    def write_attempt(self, action: str, what: str) -> None:
        """The write about to be sent: who, the frame, the caption, the time (UTC). A write row follows when it is read back;
        without one (a stop screen, a daemon error), the digest lists it as uncertain for Ian to check and undo."""
        ctx = self._write_ctx or {}
        self._attempt = {"frame": getattr(self, "last_frame", None), "caption": ctx.get("caption"), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.ev("write_attempt", action=action, what=what, who=ctx.get("who"), **self._attempt)

    def post_guard(self, who: str | None, state_ok) -> callable:
        """The final re-check for a write on a post: its author is the header nearest the top, the post is aligned, and
        `state_ok(items, png)` (the heart not red, the bookmark not filled) holds on the frame just before the tap."""
        h = self.size[1]

        def guard(items, png) -> bool:
            heads = [i.get("text") or "" for i in pr.sc.header_items(items, h, self.app)]
            return bool(heads) and _author_on(who, heads[:1]) and pr.alignment(items, h, self.app)[0] == "aligned" and png is not None and state_ok(items, png)
        return guard

    async def write_look(self, source: tuple[str, ...]) -> dict | None:
        """The look a write's tap goes on (integrate's review: the target re-checked on the frame just before the tap): a fresh
        paired frame (text and pixels of one frame), gated like gate_look, and the write's own guard on it (the same author,
        aligned, the heart / bookmark / follow state still unset); the target is then resolved on this frame, pixels included."""
        pair = await self.paired_frame()
        if pair is None:
            self.ev("write_skipped", why="no paired frame just before the tap")
            return None
        items, png, seq = pair
        ok, where = self.screen_ok(items)  # a stop record ends the session here, before any input
        screen = self.last_look.get("screen")
        live = self.require_look or self.via != "sim"
        if screen is None:  # as gate_look: live, an unrecognised screen never takes a write (integrate's re-check: an overlay)
            if live or not ok or not any(a in FEED_SCREENS.get(self.app, ()) for a in source):
                self.ev("gate_refused", screen=None, allowed=list(source), why=f"unrecognised screen ({where})")
                return None
        elif not any(screen == a or screen.startswith(a) for a in source):  # a recognised screen: the allowed ones only (as gate_look)
            self.ev("gate_refused", screen=screen, allowed=list(source))
            return None
        guard = (self._write_ctx or {}).get("guard")
        if guard is not None and not guard(items, png):
            self.ev("write_skipped", why="the target changed before the tap (re-checked on the frame just before it)", frame_seq=seq)
            return None
        return {**self.last_look, "items": items, "png": png, "frame_seq": seq}

    def free_band(self, x: float, margin: float = 24.0) -> tuple[float, float]:
        """(top, bottom) px of the room a vertical gesture has in column x: below every avoid zone in the top half, above every
        one in the bottom half, each with `margin` (the humanizer's bow), and never outside 0.15–0.88 h."""
        w, h = self.size
        lo, hi = 0.15 * h, 0.88 * h
        for z in (self.last_look or {}).get("avoid") or []:
            if len(z) != 4 or not (z[0] * w - margin <= x <= z[2] * w + margin):
                continue
            if z[3] <= 0.5:
                lo = max(lo, z[3] * h + margin + 1)
            elif z[1] >= 0.5:
                hi = min(hi, z[1] * h - margin - 1)
        return lo, hi

    def path_clear(self, x1: float, y1: float, x2: float, y2: float, margin: float = 24.0) -> bool:
        """Does the straight path (plus the humanizer's bow/drift, bounded by `margin`) stay out of every avoid zone?"""
        if not self.last_look:
            return True
        w, h = self.size
        return not any(pr.in_avoid(self.last_look, x1 + (x2 - x1) * k / 12, y1 + (y2 - y1) * k / 12, w, h, margin=margin) for k in range(13))

    async def swipe(self, x1: float, y1: float, x2: float, y2: float, duration: float, what: str, kind: str | None = None, step: int | None = None,
                    expect=None, release: float | None = None, zone_exempt: bool = False) -> None:
        if not zone_exempt and not self.path_clear(x1, y1, x2, y2):
            self.ev("swipe_refused", what=what, from_xy=[round(x1), round(y1)], to_xy=[round(x2), round(y2)], why="the path crosses an avoid zone", kind=kind)
            raise GestureRefused(what)
        t_call = self.now()
        if release is None or not self._release_ok:
            await self.send("swipe", self.phone.swipe(self.device, x1, y1, x2, y2, duration, lease=self.lease))
        else:
            await self.send("swipe", self.phone.swipe(self.device, x1, y1, x2, y2, duration, lease=self.lease, release=release))
        self.waited("swipe", t_call, expected=duration)
        if kind and kind not in ("glide", "navigate"):
            self._sent_kind = kind
        self.ev("swipe", what=what, dy=round(y1 - y2), dx=round(x2 - x1), duration_s=round(duration, 3), release=release, kind=kind, step=step,
                expect=list(expect) if expect else None, unsupported_release=getattr(self.phone, "release_unsupported", 0) or None)

    def in_avoid(self, x: float, y: float) -> bool:
        w, h = self.size
        return pr.in_avoid(self.last_look, x, y, w, h)

    async def navigate_home(self) -> bool:
        """navigate(home): ONLY on an ordinary recognised screen of the leased app (the look's `kind == "screen"`: a profile,
        Explore, a viewer) → one tap on the Home tab target, one look. Never on a stop record, an iOS alert/popup, an unknown
        screen, another app, or a sheet/login/settings id (S3; policy agreed with controller-verify, `recoverable()` in their
        suite). The target is the look's `targets.home_tab` when the store names one, else the contract's declared tab-bar
        point for the app (engage's measured Home tab), journaled as such."""
        meta = self.last_look
        screen = (meta or {}).get("screen") or ""
        kind = (meta or {}).get("kind")
        if (not screen.startswith(self.app + ".") or kind != "screen" or ".stop." in screen or screen.startswith("ios.")
                or any(w in screen for w in NO_HOME_TAP)):
            return False
        w, h = self.size
        target = ((meta or {}).get("targets") or {}).get("home_tab")
        fx, fy = (target[0], target[1]) if target else HOME_TAB.get(self.app, (0.10, 0.929))
        try:
            await self.tap(w * fx, h * fy, "home tab", kind="navigate", step=0, expect=FEED_SCREENS.get(self.app, ()), w=0.06 * w, h=0.03 * h, zone_exempt=True)
        except StaleInput:
            return False
        self.ev("navigate_target", target="home_tab", source="look" if target else "declared")
        reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
        look = await pr.until(self.look_dict, lambda m: self.screen_ok(m.get("items") or [])[0], 3.0, reaction, self.sleep, self.now)
        self.ev("navigate", target="home", from_screen=screen, to_screen=self.last_look.get("screen"), ok=look is not None)
        return look is not None

    def variant(self, kind, look: dict):
        """The kind walked in place of `kind`: one of its variants (Kind.variant_of) at its params.variant rate, when this mode
        renders it and its affordance shows (comment_profile: of the comment opens, those that go on to a commenter's
        profile); else `kind`. Rolled only when the parent is walked, so a variant costs no draw on any other post."""
        for name, k in self.kinds.items():
            r = self.params.variant.get(name)
            if k.variant_of == kind.name and r is not None and self.mode in k.modes and k.affordance(look, self.app) and r.roll(self.rng):
                self.count_action(name, "eligible")
                self.count_action(name, "selected")
                return k
        return kind

    # --- app-level detours: between posts, home → … → home -----------------------------------------------------------------
    def ready_kinds(self) -> None:
        """Live (looks required), a kind runs only when every screen it needs has a record in the screen store and its
        targets are proven live (kinds.kind_ready); the others are dropped for the session, journaled with what they need."""
        if self.via == "sim" and not self.require_look:  # a real phone (daemon or bridge) is always gated (integrate's review: --via bridge)
            return
        ids = self.store_ids if self.store_ids is not None else screen_store_ids()
        off = {n: [list(g) for g in needs(k, self.app)] for n, k in self.kinds.items() if not kind_ready(k, ids, self.app)}
        if off:
            self.kinds = {n: k for n, k in self.kinds.items() if n not in off}
            self.ev("kinds_off", kinds=sorted(off), needs=off, why="a screen it needs has no record in the screen store, or a target it uses is unproven live (kinds.UNPROVEN_LIVE)")

    def draw_detour_rates(self) -> None:
        """This session's per-post rate for each app-level detour: params.app_detour × a multiplier drawn per kind, so one
        session wanders to the Reels tab and the next barely leaves the feed (Ian 10-19: "not every day is the same")."""
        pace = self.params.app_detour_pace
        self.detour_rates = {n: round(r.p * (pace.draw(self.detour_rng)[0] if pace else 1.0), 4) for n, r in self.params.app_detour.items()
                             if n in self.kinds and self.kinds[n].start == "home"}
        if self.detour_rates:
            self.ev("app_detours", rates=self.detour_rates)

    async def wander(self, items: list[dict], plan) -> None:
        """Between posts on the home feed (Ian 10-19: "moves around the app, not just the home page"): each app-level detour
        whose affordance shows rolls this session's rate, in a drawn order, one at most per gap (the gallery forces its kind).
        It walks home → … → home on the detours' own stream (`detour_rng`); one that does not come back cleanly is brought
        home by `come_back`."""
        if not self.last_look or self.last_look.get("screen") != HOME.get(self.app):
            return
        forced = [n for n in plan.engage if n in self.kinds and self.kinds[n].start == "home"] if self.gallery is not None else []
        names = forced or list(self.detour_rates)
        if not names:
            return
        self.detour_rng.shuffle(names)
        look = {**self.last_look, "items": items}
        for name in names:
            k = self.kinds.get(name)
            if k is None or k.start != "home" or self.mode not in k.modes or not k.affordance(look, self.app):
                continue
            self.count_action(name, "eligible")
            if not forced and not pm.Rate(self.detour_rates.get(name, 0.0)).roll(self.detour_rng):
                continue
            self.count_action(name, "selected")
            self.ev("detour", kind=name, app=self.app)
            self._sent_kind, n0 = None, len(self.events)
            main, self.rng = self.rng, self.detour_rng  # every draw inside the detour (targets, reactions, dwells) from its stream
            try:
                _, verified = await pr.walk(self, k, look, items)
                if not verified:
                    await self.come_back(k, (HOME[self.app],))
            finally:
                self.detour_rng, self.rng = self.rng, main
            if self._sent_kind == name:
                self.count_action(name, "attempted")
                if not any(e["ev"] in ("kind_timeout", "kind_skip") and e.get("kind") == name for e in self.events[n0:]):
                    self.count_action(name, "completed")
                if verified:
                    self.count_action(name, "verified")
            return

    async def task_detour(self) -> bool:
        """detour(task) (task-agent design §5.1: tasks are embedded, not teleported): once, after `task_at` posts of the visit, from
        the home feed, `await self.task(self)` (run/do.py's loop: every input through this director's gestures and guards); then
        back home by the screens' own ways out. Its result lands in the session result and the ONE digest."""
        if self.task is None or self.task_result is not None or self.counts["posts"] < self.task_at:
            return False
        if not self.last_look or not (self.last_look.get("screen") or "").startswith(HOME.get(self.app, "-")):  # x.home.tabs is X's home too
            return False
        self.ev("detour", kind="task", app=self.app)
        self.task_result = {"status": "started"}
        try:
            self.task_result = await self.task(self)
        except Exception as e:  # a Stopped, the runtime's Stop (a stop text) or Park (the lease): the task's own record goes up
            self.task_result = getattr(e, "task_result", None) or {"status": "operator", "reason": f"{type(e).__name__}: {e}"[:300],
                                                                  "stopped_by": getattr(e, "kind", type(e).__name__)}
            raise
        finally:
            self.ev("task_end", **{k: v for k, v in (self.task_result or {}).items() if k in ("status", "reason", "tier_trace", "path")})
        await self.come_back(TASK_KIND, (HOME[self.app],))  # back home by the screens' ways out (no write on the way), whatever the end
        return True

    async def come_back(self, kind, home: tuple[str, ...]) -> None:
        """A kind that did not come back cleanly (a step failed, an unexpected screen showed): back to `home` by each screen's
        own way out (kinds.EXITS: the screens without the tab bar) or navigate(home), at most 3 moves, every gesture journaled
        under the kind. A stop/alert/iOS record ends the session (screen_ok), as does an unrecognised screen, with no second
        look at it (controller-verify: never two unknown looks in a row), and a screen it cannot leave."""
        for _ in range(3):
            if pr._unknown(self, self.last_look):
                raise Stopped("unknown", f"an unrecognised screen during {kind.name}: no gesture on it and no second look (S3)")
            look = await self.look_dict()
            self.screen_ok(look.get("items") or [])  # a stop record ends the session here
            screen = look.get("screen")
            if screen in home:
                self.ev("came_back", kind=kind.name, screen=screen)
                return
            if pr._unknown(self, look):
                raise Stopped("unknown", f"an unrecognised screen during {kind.name}: no gesture on it and no second look (S3)")
            way_out = EXITS.get(screen)
            if way_out is not None:
                await pr._step(self, kind, "exit", way_out, look, look.get("items") or [], 0, source=(screen,))
            elif not await self.navigate_home():
                break
        raise Stopped("off_feed", f"{kind.name} did not come back to {home[0]} ({self.last_look.get('screen')})")

    async def save(self, look: dict, items: list[dict], who: str | None):
        """A find saved: the bookmark on its action row (kinds.save), verified as a like is: text and pixels of ONE frame before
        and after, under the same confirmed author, the bookmark outlined before and solid after. None: not attempted; False:
        sent and NOT verified (a silent write failure); "unpaired": sent but unreadable on one frame."""
        from .actions import saved_state
        from .feedmaps import FEEDMAPS

        w, h = self.size
        fm = FEEDMAPS.get(self.app)
        filled = (fm.saved_state if fm and fm.saved_state else None) or saved_state
        kind = self.kinds.get("save")
        why = ("no save kind here (live: its target is unproven)" if kind is None else "no look: a write needs the recognised screen" if not self.last_look
               else "an earlier save could not be verified this session" if getattr(self, "_saves_off", False)
               else "no confirmed author for this post" if not who else None)
        if why:
            self.ev("save_skipped", why=why)
            return None
        pair = await self.paired_frame()
        if pair is None:
            self.ev("save_skipped", why="no paired frame (text and pixels from one frame) before the save")
            return None
        items, png, before_seq = pair
        header_before = [i.get("text") for i in pr.sc.header_items(items, h, self.app)][:1]
        if not header_before or not _author_on(who, [header_before[0] or ""]):  # the saved post's own author (the header nearest the top)
            self.ev("save_skipped", why=f"the confirmed author {who!r} is not the header of the post being saved")
            return None
        if pr.alignment(items, h, self.app)[0] != "aligned":
            self.ev("save_skipped", why="the post is not aligned on the paired frame")
            return None
        at = pr.resolve_target("save_icon", {**look, "items": items, "png": png}, w, h, self.rng, self.app)  # the bookmark on THIS frame
        if at is None or filled(png, at[2], w, h):
            self.ev("save_skipped", why="no bookmark found" if at is None else "already saved")
            return None
        self._sent_kind = None
        unsaved = lambda its, pg: (lambda a: a is not None and not filled(pg, a[2], w, h))(  # noqa: E731
            pr.resolve_target("save_icon", {**self.last_look, "items": its, "png": pg}, w, h, self.rng, self.app))
        self._attempt, self._write_ctx = {}, {"who": who, "caption": about(pr.sc.post_items(items, h, self.app), 160), "guard": self.post_guard(who, unsaved)}
        try:
            await pr.walk(self, kind, {**look, "items": items}, items)
        finally:
            self._write_ctx = None
        self.screen_ok(self.last_look.get("items") or [])  # a stop record after the tap ends the session as a stop, not as a failed save
        if self._sent_kind != "save":
            self.ev("save_skipped", why="the tap was not sent (stale, refused, or no target)")
            return None
        reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
        await self.sleep(reaction)
        pair2 = await self.paired_frame()
        self.counts["writes"] += 1
        if pair2 is None:
            self._saves_off = True
            self.ev("write", **getattr(self, "_attempt", {}), action="save", verified=None, verify="unpaired", who=who, saved_before=False, before_frame_seq=before_seq,
                    before_pixels_seq=before_seq, frame_seq=None, pixels_seq=None)
            return "unpaired"
        items2, png2, after_seq = pair2
        at2 = pr.resolve_target("save_icon", {**self.last_look, "items": items2, "png": png2}, w, h, self.rng, self.app)
        if at2 is None and pr.text_shift(items, items2) is not None and abs(pr.text_shift(items, items2)) <= 0.004 * h:
            at2 = at  # the outline template no longer matches a FILLED bookmark: the page did not move, so read it where it was tapped
        after = None if at2 is None else filled(png2, at2[2], w, h)
        header_after = [i.get("text") for i in pr.sc.header_items(items2, h, self.app)][:1]
        ok = after is True and bool(header_after) and header_after == header_before
        self.ev("write", **getattr(self, "_attempt", {}), action="save", verified=bool(ok), verify="paired", who=who, saved_before=False, saved_after=after, same_post=header_after == header_before,
                frame_seq=after_seq, pixels_seq=after_seq, before_frame_seq=before_seq, before_pixels_seq=before_seq)
        return ok

    async def follow(self, look: dict, items: list[dict], who: str | None):
        """The author of an IN post followed (DECISIONS 'Task agent go'): open the profile from the post's header (FOLLOW_OPEN),
        read its state on a paired frame (only "not_following" goes on: an account already followed or requested is never
        tapped, the button toggles), the one tap on Follow, the state on a paired frame after: verified when it reads
        following or requested; then back to the feed (FOLLOW_BACK, else home). None: not attempted; False: sent and NOT
        verified; "unpaired": sent but unreadable on one frame."""
        from .feedmaps import follow_state
        from .kinds import FOLLOW_BACK, FOLLOW_OPEN

        w, h = self.size
        kind = self.kinds.get("follow")
        why = ("no follow kind here (live: its target is unproven)" if kind is None else "no look: a write needs the recognised screen" if not self.last_look
               else "an earlier follow could not be verified this session" if getattr(self, "_follows_off", False)
               else "no confirmed author for this post" if not who else None)
        if why:
            self.ev("follow_skipped", why=why)
            return None
        await pr.walk(self, FOLLOW_OPEN, {**look, "items": items}, items)
        result = None
        try:
            screen = self.last_look.get("screen")
            if (screen is None and (self.require_look or self.via != "sim")) or (screen is not None and not screen.startswith(("instagram.profile", "tiktok.profile"))):
                self.ev("follow_skipped", why=f"the profile did not open ({screen})")
                return None
            pair = await self.paired_frame()
            if pair is None:
                self.ev("follow_skipped", why="no paired frame on the profile")
                return None
            items1, png1, seq1 = pair
            if not _profile_is(who, items1, h):
                self.ev("follow_skipped", why=f"the profile is not {who!r}'s (its own handle line)")
                return None
            state = follow_state(self.app, items1, png1, w, h)
            if state != "not_following":
                self.ev("follow_skipped", why=f"follow state {state!r}: only an account not followed is followed")
                return None
            self._sent_kind = None
            not_followed = lambda its, pg: _profile_is(who, its, h) and follow_state(self.app, its, pg, w, h) == "not_following"  # noqa: E731
            self._attempt, self._write_ctx = {}, {"who": who, "caption": f"@{who}'s profile", "guard": not_followed}
            try:
                await pr.walk(self, kind, {**self.last_look, "items": items1, "png": png1}, items1)
            finally:
                self._write_ctx = None
            self.screen_ok(self.last_look.get("items") or [])
            if self._sent_kind != "follow":
                self.ev("follow_skipped", why="the tap was not sent (stale, refused, or no target)")
                return None
            self._follow_sent = True  # counted here: the back gesture after it overwrites _sent_kind
            self.counts["follows"] = self.counts.get("follows", 0) + 1
            reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
            await self.sleep(reaction)
            pair2 = await self.paired_frame()
            self.counts["writes"] += 1
            if pair2 is None:
                self._follows_off = True
                self.ev("write", **getattr(self, "_attempt", {}), action="follow", verified=None, verify="unpaired", who=who, state_before=state, before_frame_seq=seq1,
                        before_pixels_seq=seq1, frame_seq=None, pixels_seq=None)
                result = "unpaired"
                return result
            items2, png2, seq2 = pair2
            after = follow_state(self.app, items2, png2, w, h)
            same = _profile_is(who, items2, h)
            result = after in ("following", "requested") and same
            self.ev("write", **getattr(self, "_attempt", {}), action="follow", verified=bool(result), verify="paired", who=who, state_before=state, state_after=after, same_post=same,
                    frame_seq=seq2, pixels_seq=seq2, before_frame_seq=seq1, before_pixels_seq=seq1)
            return result
        finally:
            await pr.walk(self, FOLLOW_BACK, {**self.last_look, "items": self.last_look.get("items") or items}, self.last_look.get("items") or items)
            if self.last_look.get("screen") and not any(str(self.last_look["screen"]).startswith(f) for f in FEED_SCREENS.get(self.app, ())):
                await self.navigate_home()  # the back did not land on the feed: the home tab

    def unmetronome(self, dwell: float) -> float:
        """A person's pace never repeats three times running (controller-verify: 5 consecutive dwells ≥ 1 s within ±15 % in a
        284-post session; human clips never exceed 2): when the last two dwells (≥ 1 s) already match within ±15 % and this one
        would too, it moves out of the band by a drawn factor, shorter or longer. On the write stream: other draws unchanged."""
        r = []  # the last two view rows' dwells, as the journal (and the judge) has them
        for e in reversed(self.events):
            if e["ev"] == "view":
                r.insert(0, e["dwell"])
                if len(r) == 2:
                    break
        if dwell >= 1 and len(r) == 2 and min(r) >= 1 and abs(r[1] - r[0]) <= 0.15 * r[0] and abs(dwell - r[1]) <= 0.15 * r[1]:
            dwell = r[1] * (self.write_rng.uniform(0.55, 0.8) if self.write_rng.random() < 0.5 else self.write_rng.uniform(1.25, 1.7))  # off the band
            # around the previous dwell (relative to it: a factor on the candidate can land back inside the band)
        return dwell

    def coupled(self, cls: str | None) -> bool:
        """A sign of interest (expanding a caption, opening the comments) follows the post's class (Ian, afternoon 10-19): IN at
        full weight, NEUTRAL or no class at COUPLED_NEUTRAL, AGAINST / ad / live never; drawn on the write stream."""
        if cls in ("AGAINST", "AD", "LIVE"):
            return False
        return cls == "IN" or self.write_rng.random() < COUPLED_NEUTRAL

    def wants_nudge(self, off: float) -> bool:
        """Does he correct this rest? A header cut off above (whose post is this?) always; one resting below the top only past
        a drawn tolerance, and then at the nudge rate (controller-verify: ≤ ~30 % of landings nudged; attempt 10: 55 %)."""
        if off < 0:
            return True
        tol = self.params.glide["nudge_tol"].draw(self.rng)[0] if "nudge_tol" in self.params.glide else 0.0
        return off > tol and (self.params.nudge is None or self.params.nudge.roll(self.rng))

    async def await_still(self, png, items, limit_s: float = 4.0):
        """After a glide, until two consecutive looks agree the page stopped (the same text at the same height): R1 attempt 2
        showed the look reporting `settled` while the coast was still carrying the page, so the post was judged mid-motion."""
        if not hasattr(self.phone, "look"):
            return png, items
        t0 = self.now()
        while self.now() - t0 < limit_s:
            await self.sleep(STILL_GAP_S)
            png2, items2 = await self.look()
            shift = pr.text_shift(items, items2)
            if shift is not None and abs(shift) <= 0.004 * self.size[1]:  # the same lines at the same height: stopped
                return png2, items2
            self.ev("coasting", shift_px=None if shift is None else round(shift))
            png, items = png2, items2
        return png, items

    # --- glide: the closed loop -------------------------------------------------------------------------------------------------
    async def glide(self, kind: str, items: list[dict]) -> None:
        w, h = self.size
        if kind == "page":  # TikTok/Reels snap: the persona's measured flick rows
            row = self.flicks.draw()
            x1, y1, x2, y2, d = self.human.flick_from(w, h, row.row, "up")
            await self.swipe(x1, y1, x2, y2, d, "glide", kind="glide", step=0, expect=FEED_SCREENS.get(self.app, ()))
            self.counts["glides"] += 1
            self.ev("glide", kind="page", duration_s=round(float(d), 3), length_frac=round(float(row.length_frac), 3))
            return
        g = pr.plan_glide(kind, items, w, h, self.params, self.rng, self.glide_state, app=self.app)
        for _ in range(8):  # a start column and height whose whole path clears the avoid zones (Codex review #2)
            x = self.rng.uniform(0.3, 0.7) * w
            lo, hi = self.free_band(x)  # the vertical room the avoid zones leave in this column (R1 attempt 5: a start at 0.88 h sat on the tab bar's zone)
            drag = g.drag_px
            if drag >= 0:
                y1 = min(hi, max(lo + drag, self.rng.uniform(0.6, 0.85) * h))
                drag = min(drag, y1 - lo)
            else:
                y1 = max(lo, min(hi + drag, self.rng.uniform(0.3, 0.5) * h))
                drag = max(drag, y1 - hi)
            if self.path_clear(x, y1, x, y1 - drag):
                break
        await self.swipe(x, y1, x, y1 - drag, g.duration_s, "glide", kind="glide", step=0, expect=FEED_SCREENS.get(self.app, ()), release=g.release)
        self.counts["glides"] += 1
        self.ev("glide", kind=kind, planned_px=round(g.target_px), drag_px=round(drag), duration_s=round(g.duration_s, 3), release=round(g.release, 3),
                gain=round(g.gain, 3), how=g.how, tail=g.tail)
        self._pending = (items, g, drag)

    async def settle_glide(self, items: list[dict]) -> None:
        """The look after a glide: measure the travel against the plan (gain), judge alignment, nudge once if misaligned."""
        before, g, drag = self._pending
        self._pending = None
        w, h = self.size
        if g.kind in ("next", "partial"):  # (a skim's tracked header leaves the screen: no gain update)
            pred = g.pred_px * drag / g.drag_px if g.drag_px else g.pred_px  # the drag may have been clamped to the screen
            travel = pr.measure_travel(before, items, h, pred)
            how = "text"
            if travel is None and g.kind == "next":
                # live OCR varies frame to frame (badges, separators), so the tracked header's text may not match: a header
                # landing near the content top is taken as the target (travel from its position); only with no header near the
                # top did the target go over it (R1 attempt 5: a near-miss read as an overshoot raised the gain the wrong way)
                y0 = pr.target_header_y(before, h, self.app)
                top = pr.sc.layout(items, h, self.app)[0]
                near = [y for y in pr.sc.header_ys(items, h, self.app) if abs(y - top * h) < 0.25 * h]
                if y0 is not None and near:
                    travel, how = y0 - min(near, key=lambda y: abs(y - top * h)), "header near the top"
                elif pr.tracked_left(before, items, h, self.app):
                    how = "overshoot"
            if how == "overshoot" and "min step" in (g.how or ""):
                how = "past a min step"  # the plan moved further than the target on purpose: no evidence about the gain
            elif how == "overshoot":
                self.glide_state.overshot()  # the target went over the top: travelled further than planned, unmeasurable
            else:
                self.glide_state.update(pred, travel)  # the planner's prediction (with the gain it used): the update learns the absolute gain
        else:
            travel = None
        self.aligned, off = pr.alignment(items, h, self.app)
        self.ev("glide_result", kind=g.kind, aligned=self.aligned, offset_h=None if off is None else round(off, 3),
                travel_px=None if travel is None else round(travel), planned_px=round(g.target_px), travel_how=how if g.kind in ("next", "partial") else None,
                travel_per_plan=None if travel is None or not g.target_px else round(travel / g.target_px, 3),  # fast-eyes/controller-verify: compare with trial 1's 3.7 posts per gesture
                gain=round(self.glide_state.gain, 3), n_gain=self.glide_state.n)
        if self.aligned == "misaligned" and abs(off) <= 0.5 and (g.kind in ("next", "back") or g.kind.startswith("skim")):
            if not self.wants_nudge(off):  # he reads the half-framed post (the human refs rest framed only ~70 % of the time)
                self.ev("nudge_declined", offset_h=round(off, 3))
                return
            reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
            await self.sleep(reaction)
            # the hesitation aged the look: a fresh one right before the nudge (S3 "inputs land on a fresh look"; R1 attempt 8:
            # three nudges acted on looks 1.5–1.6 s old). Nudge only if it is still the feed and still misaligned, from its offset.
            look = await self.gate_look(FEED_SCREENS.get(self.app, ()))
            if look is None:
                return
            self.aligned, off = pr.alignment(look.get("items") or [], h, self.app)
            if self.aligned != "misaligned" or off is None or abs(off) > 0.5:
                self.ev("nudge_skipped", aligned=self.aligned, offset_h=None if off is None else round(off, 3))
                return
            T, _ = self.params.glide["nudge_s"].draw(self.rng)
            # the nudge coasts too (release 0.05 × its speed × 0.5 s) and the phone's gain applies: plan it like a glide, or the
            # coast carries the header under the top bar (the sim showed ~60 px overshoot → misaligned)
            nudge = pr.sc.plan_drag(pr.sc.nudge_for(off, h), T, NUDGE_RELEASE)[0] / self.glide_state.gain  # a slow lift: it stops with the finger
            x = self.rng.uniform(0.3, 0.7) * w
            lo, hi = self.free_band(x)  # a large correction stays inside the room the avoid zones leave (R1 attempt 7: a ~1000 px nudge ended in the top zone)
            if nudge >= 0:
                y1 = min(hi, max(lo + nudge, self.rng.uniform(0.45, 0.7) * h))
                nudge = min(nudge, y1 - lo)
            else:
                y1 = max(lo, min(hi + nudge, self.rng.uniform(0.3, 0.55) * h))
                nudge = max(nudge, y1 - hi)
            try:
                await self.swipe(x, y1, x, y1 - nudge, T, "nudge", kind="glide", step=1, expect=FEED_SCREENS.get(self.app, ()), release=NUDGE_RELEASE)
            except (GestureRefused, StaleInput):
                return  # the nudge path crosses an avoid zone, or the daemon refused it as stale: stay misaligned rather than retry blind
            self.counts["nudges"] += 1
            self.ev("nudge", offset_h=round(off, 3), px=round(nudge))

    async def paired_frame(self, tries: int = 3) -> tuple[list[dict], bytes, int | None] | None:
        """The OCR items and the pixels of ONE frame (controller-verify + surfaces: screen() and ocr()/look() are separate
        shots, so a coasting feed or a playing video puts the heart in different places). With eyes: the look's own frame by
        its seq when the daemon serves it; else a frame fetched between two looks that report the same frame_seq (a still
        screen). Without eyes (a sim): the screen and its OCR. None when no pair could be made."""
        if hasattr(self.phone, "observe"):  # one reply carries the OCR and the pixels of one frame: a true pair, with that frame's seq
            observation = await self.phone.observe(self.device, self.app)
            return observation.items, observation.png, observation_seq(observation)
        if not hasattr(self.phone, "look"):
            return await self.phone.ocr(self.device), await self.phone.screen(self.device), None
        for _ in range(tries):
            m1 = await self.look_journaled()
            if not m1:
                return None
            seq = m1.get("frame_seq")
            if hasattr(self.phone, "frame_at"):
                png = await self.phone.frame_at(self.device, seq)
                if png is not None:
                    return m1.get("items") or [], png, seq
            png = await self.phone.screen(self.device)
            m2 = await self.look_journaled()
            if m2 and seq is not None and m2.get("frame_seq") == seq:
                return m2.get("items") or [], png, seq
            reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
            await self.sleep(reaction)
        return None

    # --- a like: write + verify before/after under the confirmed handle ------------------------------------------------------------
    def heart(self, items: list[dict], png: bytes | None) -> tuple[float, float] | None:
        """The like heart on one frame: surfaces' measured resolver for this app (feedmaps), the point the heart gesture taps
        and the one both verifications read."""
        t = pr.feedmap_target(self.app, "heart", {"items": items, "png": png}, *self.size)
        return (t[0], t[1]) if t else None

    async def like(self, look: dict, items: list[dict], png: bytes | None, who: str | None) -> bool | None:
        """None: not attempted (no eyes, no heart, already red, no input sent). False: attempted and NOT verified (a silent
        write failure). The gesture is drawn (params.like_heart: the heart button, else a double tap on the media); both are
        verified the same way: the heart unliked on a paired frame before, red on a paired frame after, under the same header."""
        h = self.size[1]
        if not self.last_look:
            self.ev("like_skipped", why="no look: a write needs the recognised screen")
            return None
        if getattr(self, "_likes_off", False):
            self.ev("like_skipped", why="an earlier like could not be verified this session: no further likes")
            return None
        if not who:  # the approval belongs to ONE post: its author, confirmed by OCR in the header zone (Codex review #4)
            self.ev("like_skipped", why="no confirmed author for this post")
            return None
        gesture = "heart" if self.params.like_heart is not None and self.rng.random() < self.params.like_heart.p else "double_tap"  # a share, not a capped propensity
        if gesture == "double_tap" and not hasattr(self.phone, "double_tap"):
            self.ev("like_skipped", why="this phone has no one-input double tap")
            return None
        pair = await self.paired_frame()  # the heart's place and its pixels from ONE frame
        if pair is None:
            self.ev("like_skipped", why="no paired frame (text and pixels from one frame) before the like")
            return None
        items, png, before_seq = pair
        import difflib

        header_before = [i.get("text") for i in pr.sc.header_items(items, h, self.app)][:1]  # the post's header line: the like must land under the same one
        first = (header_before[0] or "").lower().split()[:1] if header_before else []
        if not first or difflib.SequenceMatcher(None, who.lower(), first[0]).ratio() < 0.8:  # the liked post's own author: a second
            self.ev("like_skipped", why=f"the confirmed author {who!r} is not the header of the post being liked")  # post's on screen is not it
            return None
        if pr.alignment(items, h, self.app)[0] != "aligned":
            self.ev("like_skipped", why="the post is not aligned on the paired frame")
            return None
        heart = self.heart(items, png)
        before = red_share(png, *heart) if heart is not None else None
        if heart is None or before is None or before > 0.04:  # the heart button toggles: never tapped on a red heart
            self.ev("like_skipped", why="no heart found" if before is None else "heart already red")
            return None
        before_events = len(self.events)
        heart_unset = lambda its, pg: (lambda hb: hb is not None and (r := red_share(pg, *hb)) is not None and r <= 0.04)(self.heart(its, pg))  # noqa: E731
        self._attempt, self._write_ctx = {}, {"who": who, "caption": about(pr.sc.post_items(items, h, self.app), 160), "guard": self.post_guard(who, heart_unset)}
        try:
            spent, _ = await pr.walk(self, LIKE_GESTURES[gesture], {**look, "items": items}, items)
        finally:
            self._write_ctx = None
        self.screen_ok(self.last_look.get("items") or [])  # a stop after the tap ends the session as a stop: no read-back under a sheet
        if self._sent_kind != "like":  # zero input: a stale refusal, no target, a refused tap; nothing to verify, no write
            stale = any(e["ev"] == "stale" for e in self.events[before_events:])
            why = next((e.get("why") for e in reversed(self.events[before_events:]) if e["ev"] in ("kind_skip", "kind_timeout")), None)
            self.ev("like_skipped", gesture=gesture, why=f"the {gesture} was refused as stale (zero input sent)" if stale else f"no {gesture} input sent ({why})")
            return None
        reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
        await self.sleep(reaction)
        pair2 = await self.paired_frame()
        self.counts["writes"] += 1
        if pair2 is None:  # the like was sent but can't be read on one frame: unverified; the session stops (Codex review #7)
            self._likes_off = True
            self.ev("write", **getattr(self, "_attempt", {}), action="like", gesture=gesture, verified=None, verify="unpaired", who=who, red_before=round(before, 3), before_frame_seq=before_seq,
                    before_pixels_seq=before_seq, frame_seq=None, pixels_seq=None)
            return "unpaired"
        items2, png2, after_seq = pair2
        heart2 = self.heart(items2, png2)
        header_after = [i.get("text") for i in pr.sc.header_items(items2, h, self.app)][:1]
        after = red_share(png2, *heart2) if heart2 is not None else None
        # the same post: its header again, or a page that did not move (every line shared by both frames at the same height: a
        # header resting on the top bar's edge drops out of the OCR on a 1 px drift, and the heart is still this post's)
        shift = pr.text_shift(items, items2)
        same = (bool(header_after) and header_after == header_before) or (shift is not None and abs(shift) <= 0.004 * h)
        ok = after is not None and after > HEART_RED and same
        self.ev("write", **getattr(self, "_attempt", {}), action="like", gesture=gesture, verified=bool(ok), verify="paired", who=who, red_before=round(before, 3),
                red_after=None if after is None else round(after, 3), same_post=same,
                frame_seq=after_seq, pixels_seq=after_seq, before_frame_seq=before_seq, before_pixels_seq=before_seq)  # text and pixels of one frame (controller-verify S2)
        return ok

    # --- the fast read against his written lines (lane classifier) -----------------------------------------------------------
    def fast_read(self, items: list[dict], cue, decision) -> tuple[str | None, dict | None, float | None]:
        """The classifier's class for this post, or (None, scores, ms) when it is absent or failed: a failure is missing
        evidence, never a neutral match (classifier-integration.md)."""
        if self.classify is None:
            return None, None, None
        try:
            cls, scores, ms = self.classify(pr.sc.post_items(items, self.size[1], self.app), self.size[1])  # this post's lines only
        except Exception as e:
            self.ev("classify_error", error=str(e)[:160])
            return None, None, None
        if (scores or {}).get("error"):
            return None, scores, ms
        return cls, scores or {}, round(ms, 2) if ms is not None else None

    def apply_class(self, cls: str, scores: dict, cue, decision):
        """IN → a stay at full interest; AGAINST → a quick skip; AD / LIVE → the hard skip; NEUTRAL → the glance's decision."""
        from .cues import Decision

        margin = scores.get("margin")
        if cls in ("AD", "LIVE"):
            cue.cls = cue.cls if cue.cls in ("ad", "live") else cls.lower()
            return Decision("skip", f"{cls} marker: he skips", 0, decision.matches)
        if cls == "AGAINST":
            return Decision("skip", f"against his written lines (margin {margin}): a quick skip", 0, decision.matches)
        if cls == "IN":
            return Decision("stay" if decision.action == "skip" else decision.action, f"in: matches his words (margin {margin})", 3, decision.matches)
        return decision

    # --- finds: IN posts noted for the session digest (the intention asks for it) --------------------------------------------
    def note_find(self, items: list[dict], decision, proposal, who: str | None) -> None:
        f = {"s": round(self.session_s(), 1), "who": who, "summary": (proposal.summary if proposal is not None else "")[:300],
             "post": about(items, 300), "reason": decision.reason[:200], "frame": f"frames/{self._frames - 1:04d}.png" if self.evidence else None}
        self.finds.append(f)
        self.ev("find", **f)

    # --- comment drafts: IN posts, drafted after the session and sent to Ian to vet (comments.py) ---------------------------
    def wants_comment(self, cls: str | None, decision) -> bool:
        """An IN post (the fast class when it ran, else full interest against his words) in a session whose intention asks
        for comments, up to the drafter's per-session number."""
        if self.drafter is None or self.intention is None or not self.intention.draft_comments or len(self.comment_posts) >= self.drafter.max_drafts:
            return False
        return cls == "IN" if cls is not None else decision.interest >= 3

    def note_comment_post(self, items: list[dict], frame: str | None, proposal, who: str | None, png: bytes) -> None:
        c = {"s": round(self.session_s(), 1), "who": who, "summary": (proposal.summary if proposal is not None else "")[:300],
             "text": about(pr.sc.post_items(items, self.size[1], self.app), 300), "frame": frame}
        self.comment_posts.append({**c, "png": png})
        self.ev("comment_candidate", **c)

    async def propose_comments(self) -> dict:
        """After the session and its recording (model time, not session time): draft a comment for each IN post noted, queue
        them for Ian and send him ONE vetting message. Nothing is posted; `comments.may_post` is the gate a poster must pass."""
        if self.drafter is None or not self.comment_posts:
            return {"comment_drafts": 0}
        items = await self.drafter.propose(self.comment_posts, self.intention.text, session=self.evidence.name if self.evidence else None)
        self.comment_drafts = items
        digest = self.write_digest()  # the ONE digest now carries the drafts too
        return {"comment_drafts": len(items), "comment_ids": [i["id"] for i in items], "comment_errors": list(self.drafter.errors),
                "digest": str(digest) if digest else None}

    UNDO = {"like": "tap the heart again on that post (unlike)", "save": "tap the bookmark again (unsave)",
            "follow": "open the profile, tap Following, then Unfollow", "comment": "long-press the comment, then Delete"}

    def write_digest(self) -> Path | None:
        """ONE digest per session (DECISIONS 'Task agent go'): exactly what was done, each with what he needs to undo it (the
        post's account and link when known, the action), the finds, and the session's comment drafts. digest.md beside the
        evidence; the CLI sends it once, after the comments are drafted."""
        writes = [e for e in self.events if e["ev"] == "write" and e.get("verified")]
        uncertain = [e for e in self.events if e["ev"] == "write" and not e.get("verified")]  # sent but not verified: he checks these
        pending: list[dict] = []  # attempts with no write row after them: a stop screen or a phone error right after the tap
        for e in self.events:
            if e["ev"] == "write_attempt":
                pending.append(e)
            elif e["ev"] == "write":
                for k in range(len(pending) - 1, -1, -1):
                    if pending[k]["action"] == e["action"]:
                        pending.pop(k)
                        break
        uncertain += [{**a, "verify": "no read-back (the session stopped or the phone failed right after the tap)"} for a in pending]
        drafts = getattr(self, "comment_drafts", [])
        task = getattr(self, "task_result", None)
        if not self.evidence or not (writes or uncertain or self.finds or drafts or task):
            return None
        who = self.identity or self.app
        lines = [f"# Session digest: {who}", "", f"{round(self.session_s() / 60, 1)} min, {self.counts.get('posts', 0)} posts" +
                 (f"; intention (the operator's words): {self.intention.text}" if self.intention else ""), ""]
        if writes:
            lines += [f"## Done ({len(writes)}): undo any of it", ""]
            for k, e in enumerate(writes, 1):
                where = e.get("url") or (f"@{e['who']}'s post" if e.get("who") else "a post")
                lines.append(f"{k}. {e['action']} on {where} ({e.get('utc') or 's %.0f' % e['s']}{', ' + e['gesture'] if e.get('gesture') else ''}): undo: "
                             f"{self.UNDO.get(e['action'], 'reverse it in the app')}" + (f" · “{e['caption'][:80]}”" if e.get("caption") else "")
                             + (f" · frame {e['frame']}" if e.get("frame") else ""))
            lines.append("")
        if uncertain:
            lines += [f"## Uncertain ({len(uncertain)}): sent but not verified; check each and undo it if it happened", ""]
            for k, e in enumerate(uncertain, 1):
                where = e.get("url") or (f"@{e['who']}'s post" if e.get("who") else "a post")
                lines.append(f"{k}. {e['action']} on {where} ({e.get('utc') or 's %.0f' % e['s']}, {e.get('verify') or 'unverified'}): if done, undo: "
                             f"{self.UNDO.get(e['action'], 'reverse it in the app')}" + (f" · “{e['caption'][:80]}”" if e.get("caption") else "")
                             + (f" · frame {e['frame']}" if e.get("frame") else ""))
            lines.append("")
        if writes or uncertain:
            lines += ["Find them in the app (newest first; match the UTC times): Instagram: Your activity → Interactions → Likes; "
                      "profile → ☰ → Saved; profile → Following. TikTok: profile → the Liked tab; profile → Following.", ""]
        if task:
            t = task
            lines += ["## Task", "", f"{t.get('goal', '')}: {t.get('status')}" + (f", answer: {t['answer']}" if t.get("answer") else "")
                      + (f" ({t['reason']})" if t.get("reason") else ""), ""]
        if self.finds:
            lines += [f"## Finds ({len(self.finds)})", ""]
            for k, f in enumerate(self.finds, 1):
                lines.append(f"{k}. {f['who'] or 'unknown account'}: {f['summary'] or f['post'][:160]}")
                lines.append(f"   why: {f['reason']}" + (f" · frame {f['frame']}" if f["frame"] else ""))
            lines.append("")
        if drafts:
            lines += [f"## Comment drafts ({len(drafts)})", ""]
            for k, d in enumerate(drafts, 1):
                state = "autonomous: would post within the caps, but no comment runner exists yet (inert)" if d.get("state") == "autonomous" \
                    else "waiting for your answer"
                lines.append(f"{k}. [{d['id']}] on {(d.get('post') or {}).get('who') or 'an unknown account'}: “{d['text']}” ({state})")
        path = self.evidence / "digest.md"
        path.write_text("\n".join(lines).rstrip() + "\n")
        return path

    # --- the loop ---------------------------------------------------------------------------------
    async def run(self, minutes: float, full_length: bool = False) -> dict:
        self.t_start = self.now()
        end = self.t_start + minutes * 60
        status, reason = "ok", None
        vp = self.beh.visit_plan()
        if full_length:  # a verdict session runs its declared length: no early-exit tail (TikTok R1 attempt 1 left after 4.5 s), and no
            vp["glance_and_leave"], vp["full_length"] = False, True  # phone-down idle (attempt 2 set it down 332 s of 630: the verdict tests scrolling)
            self.params = copy.copy(self.params)
            self.params.tails = {**self.params.tails, "long_idle": pm.Rate(0.0, tail=True)}
            self.beh.p = self.params
        task_scope = list(getattr(getattr(self.task, "t", None), "scope", None) or [])  # a do task's declared writes: the visit is not read-only
        self.ev("start", app=self.app, device=self.device, identity=self.identity, minutes=minutes, read_only=self.read_only and not task_scope,
                writes={"feed": "none" if self.read_only else "habits", "task": task_scope} if self.task is not None else None, via=self.via, mode=self.mode,
                persona=self.persona.id, calibration=self.params.calibration, params_hash=self.params_hash, variation=self.variation, seed=self.persona.seed, rng_seed=self.rng_seed, plan=vp, gallery=self.gallery,
                intention=None if self.intention is None else {"name": self.intention.name, "text": self.intention.text, "find": self.intention.find,
                                                               "draft_comments": self.intention.draft_comments})
        self.t_end = end
        self.ready_kinds()
        self.draw_detour_rates()
        prev_png, prev_about, same, off_feed = None, None, 0, 0
        max_posts = self.rng.randint(1, 3) if vp["glance_and_leave"] and not self.gallery else None
        self._vlm_task = None
        try:  # everything that can touch the daemon runs inside the terminal handler, the startup probe included (Codex review #8)
            if self.require_look and (not hasattr(self.phone, "look") or (await self.phone.look(self.device, self.app)) is None):
                raise Stopped("no_look", "the daemon offers no look(): OCR markers alone can't gate gestures (S3); not running")
            await self.warm()
            if self.params.detour["exploration"].roll(self.rng):
                self.ev("detour", kind="exploration", app=self.app, hook=self.explore is not None)
                if self.explore is not None:
                    await self.explore(self)
            while self.now() < end:
                t0 = self.now()
                png, items = await self.look()
                if not hasattr(self, "size"):
                    if self.last_look.get("width") and self.last_look.get("height"):
                        self.size = (self.last_look["width"], self.last_look["height"])
                    else:
                        png = await self.pixels(png)
                        self.size = vision.png_size(png) if png else (1170, 2532)
                ok, where = self.screen_ok(items)
                if ok and self._pending is not None:
                    png, items = await self.await_still(png, items)
                    ok, where = self.screen_ok(items)
                if ok and self._pending is not None:
                    await self.settle_glide(items)
                    if self.counts["nudges"] and self.events[-1]["ev"] == "nudge":
                        png, items = await self.look()
                        png, items = await self.await_still(png, items)  # the nudge coasts too: judge only once it has stopped (controller-verify)
                        self.aligned, off = pr.alignment(items, self.size[1], self.app)
                        self.ev("glide_result", kind="nudge", aligned=self.aligned, offset_h=None if off is None else round(off, 3))
                        ok, where = self.screen_ok(items)
                if not ok and off_feed == 0 and self.last_look and await self.navigate_home():
                    png, items = await self.look()
                    png, items = await self.await_still(png, items)  # the feed may animate back into place
                    ok, where = self.screen_ok(items)
                if not ok:  # S3: no feed gesture on a screen that isn't the feed; one more look, then park with the frame
                    off_feed += 1
                    self.ev("off_feed", screen=where, n=off_feed, post=about(items))
                    png = await self.pixels(png)
                    if self.evidence and png is not None:
                        (self.evidence / f"off-feed-{off_feed}.png").write_bytes(png)
                    if off_feed >= 2:
                        raise Stopped("off_feed", f"not on the {self.app} feed ({where}): {about(items, 120)}")
                    reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
                    await self.sleep(reaction)
                    continue
                off_feed = 0
                if prev_about is not None and about(items, 400) == prev_about and png is None:
                    png = await self.pixels(png)  # the same words as the last post: only the pixels can tell a playing video from a stuck feed
                same = same + 1 if prev_about is not None and about(items, 400) == prev_about and (png is None or prev_png is None or vision.change(prev_png, png) == 0.0) else 0
                prev_png, prev_about = png, about(items, 400)
                if same >= STUCK_LOOKS - 1:
                    raise Stopped("unknown", f"the feed didn't move after {STUCK_LOOKS} looks")
                moving = await self.moving(png)
                # live (looks required), the glance is pixel-free: the image classifier never sits on the decision path (L2)
                cue = await self.cues.judge(items, moving, png=png if (self.cues.classify and png is not None and not self.require_look) else None, h=self.size[1])
                self.counts["posts"] += 1
                seg_before = self.beh.segment
                plan = self.beh.post_plan()
                plan.engage = [k for k in plan.engage if k in self.kinds and self.mode in self.kinds[k].modes]  # the journal lists only kinds this mode renders
                if self.beh.segment is not seg_before:
                    sg = self.beh.segment
                    self.ev("segment", n_posts=sg.n_posts, pace=sg.pace, tail=sg.tail)
                if self.gallery:  # the gallery renders each kind once on the next posts (a like only on an aligned post)
                    plan.engage = [self.gallery.pop(0)] if (self.gallery[0] != "like" or self.aligned == "aligned") else []
                    plan.tail, plan.idle_s = None, 0.0  # the gallery shows kinds, not tails
                seen = self.frame_wall if self.frame_wall is not None else t0  # the dwell counts from the settled frame he sees (the coast wait is not the glance)
                match = None
                if self.taste is not None:
                    from .semantic import clean

                    match = self.taste.score(clean(items, self.size[1]), topics=cue.topics)
                decision = decide(cue, items, self.persona.interest, self.read_only, match=match)
                cls, cls_scores, cls_ms = self.fast_read(items, cue, decision)
                if cls is not None:
                    decision = self.apply_class(cls, cls_scores, cue, decision)
                u = None
                if self.mode == "passive" and cue.cls not in ("ad", "live"):  # no niche: the human dwell mixture splits skip/stay, the glance tilts it
                    stay, u = self.beh.passive_split(plan, decision.action, decision.interest, forced={"IN": True, "AGAINST": False}.get(cls))
                    cue.decision = "stay" if stay else "skip"
                else:  # intentional (the niche decides), and ads/lives are always skipped
                    cue.decision = "skip" if decision.action == "skip" else "stay"
                if self.counts["posts"] == 1 and self.gallery is None:  # the post found at launch was seen last time: an orienting glance,
                    cue.decision = "skip"  # then on (attempt 10 stayed 11 s on a finished reel left on screen; refs: first action 3.4–5.4 s)
                if self.gallery is not None and plan.engage:
                    cue.decision = "stay"
                words = caption_words(items, self.size[1])
                if cue.decision == "skip":
                    dwell = self.beh.dwell_s(plan, 0, skip=True, u=u)  # the glance before gliding past: the post's quantile inside the skip component
                    self.counts["skipped"] += 1
                else:
                    dwell = self.beh.dwell_s(plan, decision.interest, words, u=u, still=not moving)
                    if self.gallery is not None and plan.engage:  # the gallery shows the kind on a stay long enough for its understanding: a drawn stay on top, never a fixed value
                        dwell = max(dwell, self.params.describe_min_s + self.beh.dwell_s(plan, 1, words))
                if self.counts["posts"] == 1:
                    dwell = max(dwell, vp["orient_s"])  # the first post: orienting is part of looking at it (3.3: first action 1–4 s after the feed shows)
                dwell = self.unmetronome(dwell)
                post_end = min(seen + dwell, end)  # ONE absolute deadline for the post (and the session's end holds): every wait counts against it (Codex review #13)
                l2_ms = round((self.now() - seen) * 1000, 1)  # frame → decision, measured before any pixel fetch
                proposal, who, confirmed = None, None, False
                png = await self.pixels(png)  # after the decision; the view row then names the frame it judged (Codex review #16)
                post_ev = len(self.events)  # this post's rows start here (its signs of interest for a like)
                self.ev("view", cls=cue.cls, decision=cue.decision, action=decision.action, reason=decision.reason, source=cue.source,
                        cue_ms=round(cue.latency_ms, 1), l2_ms=l2_ms, screen=self.last_look.get("screen"),
                        frame_age_ms=self.last_look.get("frame_age_ms"), look_ms=self.last_look.get("ms"), frame_seq=self.last_look.get("frame_seq"),
                        frame=getattr(self, "last_frame", None), dwell=round(dwell, 2), remaining=round(max(0.0, post_end - self.now()), 2), q=round(plan.q, 4),
                        words=words, tail=plan.tail or plan.comp_tail, idle=plan.idle_s or None,
                        glide=plan.glide, engage=plan.engage, pace=self.beh.segment.pace, mode=self.mode, aligned=self.aligned,
                        fast_cls=cls, cls_margin=None if not cls_scores else cls_scores.get("margin"), cls_ms=cls_ms,
                        decision_source="blind" if cue.cls == BLIND else "content", topics=cue.topics, interest=decision.interest,
                        matches=decision.matches, moving=moving, post=about(items))
                judged = (items, getattr(self, "last_frame", None))  # the post's own lines and frame (a kind's walk moves `items` on)
                if cue.decision == "stay":
                    task = None
                    if self.vlm is not None and png is not None and dwell >= self.params.describe_min_s:  # every stay ≥ 2 s, both modes (Q1)
                        if self._vlm_task is not None and not self._vlm_task.done():
                            self.ev("vlm_busy", why="the previous post's understanding is still running: one at a time")  # bounded (Codex review #15)
                        else:
                            task = self._vlm_task = asyncio.ensure_future(self._describe_timed(png, seen))
                    frac, _ = self.params.engage_at.draw(self.rng)
                    await self.sleep(max(0.0, (post_end - self.now()) * frac))  # watch first; the engagement happens inside the dwell
                    look = {**self.last_look, "items": items, "png": png}
                    for name, k in self.kinds.items():  # opportunities on this post: the kinds this mode renders whose affordance is on screen
                        if k.group != "write" and k.start == "post" and self.mode in k.modes and self.params.engage.get(name) is not None and self.params.engage[name].p > 0 and k.affordance(look, self.app):
                            self.count_action(name, "eligible")
                            if name in plan.engage:
                                self.count_action(name, "selected")
                    for name in plan.engage:
                        kind = self.kinds.get(name)
                        if kind is None or self.mode not in kind.modes or kind.group == "write" or kind.start != "post" or not kind.affordance(look, self.app):
                            continue
                        if not self.last_look:  # every gesture on a post is look-gated: without eyes (a sim, a replay) nothing is tapped
                            self.ev("kind_skip", kind=name, why="no look: gestures on the post need the recognised screen")
                            continue
                        if name in COUPLED and self.gallery is None and not self.coupled(cls):  # signs of interest couple to the class (Ian 10-19)
                            self.ev("kind_skip", kind=name, why=f"class {cls or 'none'}: {name} is coupled to interest")
                            continue
                        if self.now() >= post_end:
                            break
                        kind = self.variant(kind, look)  # comment_profile in place of comments, at its rate
                        name = kind.name
                        self.counts["engagements"] += 1
                        self._sent_kind = None
                        base = look.get("screen")  # the feed screen the post is on: the kind comes back there
                        _, verified = await pr.walk(self, kind, look, items, words)
                        if self._sent_kind == name:
                            self.count_action(name, "attempted")
                            done = next((e for e in reversed(self.events) if e["ev"] == "kind_done" and e.get("kind") == name), None)
                            if done is not None and not any(e["ev"] in ("kind_timeout",) and e.get("kind") == name for e in self.events[self.events.index(done) - 3:]):
                                self.count_action(name, "completed")
                            if verified:
                                self.count_action(name, "verified")
                        look = {**self.last_look, "items": self.last_look.get("items") or items, "png": None}
                        items = look["items"]
                        if not verified:
                            if base and self.last_look.get("screen") != base:
                                await self.come_back(kind, (base,))  # left the post's screen and did not return: its way out, or home
                            break  # a kind that didn't come back cleanly ends this post's engagement (Codex review #1)
                    if task is not None:
                        try:  # the understanding is awaited only for what is left of the post (L3): late means no write, never a longer stay
                            t_call = self.now()
                            proposal = await asyncio.wait_for(asyncio.shield(task), timeout=max(0.05, post_end - self.now()))
                            self.waited("understanding (inside the dwell)", t_call)
                            if proposal.summary:
                                from dataclasses import replace as _replace

                                from .cues import confirm_handle

                                who, confirmed, fixed = confirm_handle(proposal.summary, items, self.size[1], self.app, self.last_look.get("header"), self.size[0])
                                proposal = _replace(proposal, summary=fixed)
                            if self.taste is not None and proposal.summary:
                                match = self.taste.score(f"{proposal.summary} tags: {', '.join(proposal.topics)}", png, topics=proposal.topics)
                            decision = decide(cue, items, self.persona.interest, self.read_only, described=proposal, match=match)
                            if cls is not None:
                                decision = self.apply_class(cls, cls_scores, cue, decision)
                            self.ev("vlm", answered_s=getattr(self, "_vlm_answered_s", None), l3_ms=getattr(self, "_vlm_l3_ms", None),
                                    cached=self.vlm.last_cached, kind=proposal.kind, is_ad=proposal.is_ad, topics=proposal.topics,
                                    interest=proposal.interest, quick_skip=proposal.quick_skip, summary=proposal.summary[:200], who=who, who_confirmed=confirmed,
                                    action=decision.action, reason=decision.reason, taste=(match.score if match else None))
                        except asyncio.TimeoutError:
                            self.ev("vlm_late", waited_s=round(max(0.05, post_end - self.now()), 2))
                        except Exception as e:
                            if type(e).__name__ == "DaemonError":
                                raise  # a daemon failure is terminal, not a missing description (Codex review #9)
                            self.ev("vlm_error", error=str(e)[:200])
                    if self.intention is not None and self.intention.find and decision.interest >= 3:
                        self.note_find(items, decision, proposal, who)
                    if png is not None and self.wants_comment(cls, decision):
                        self.note_comment_post(*judged, proposal, who, png)
                    margin = None if not cls_scores else cls_scores.get("margin")
                    eligible = self.beh.like_eligible(decision.interest, self.aligned == "aligned", cls) and not self.read_only
                    want_like = eligible and self.gallery is not None and "like" in plan.engage
                    if eligible and not want_like:  # the post's signs of interest so far (Ian, afternoon 10-19) scale the like's probability
                        rows = self.events[post_ev:]
                        signals = {"spent_s": round(self.now() - seen, 2),
                                   "expanded": any(e["ev"] == "kind_done" and e.get("kind") == "expand" and e.get("verified") for e in rows),
                                   "cards": sum(e["ev"] == "swipe" and e.get("kind") == "carousel" for e in rows)}
                        want_like = self.beh.like_wanted(decision.interest, True, cls, margin, signals)
                        self.ev("like_roll", cls=cls, margin=margin, **signals, **self.beh.last_like, wanted=want_like)
                    if eligible:
                        self.count_action("like", "eligible")
                    if want_like:
                        self.count_action("like", "selected")
                    if want_like and not self.read_only and proposal is not None and self.now() < post_end and \
                            "like" in self.gate.candidates(proposal, min(1.0, (self.now() - seen) / max(dwell, 1e-6)), self.session_s(), self.pacer_ok):
                        self._sent_kind = None
                        ok = await self.like(look, items, png, who if confirmed else None)
                        if self._sent_kind == "like":  # the one double-tap input: one attempt
                            self.count_action("like", "attempted")
                            if ok in (True, False):
                                self.count_action("like", "completed")
                            if ok is True:
                                self.count_action("like", "verified")
                        if ok is False:
                            raise Stopped("silent_write_failure", "like not verified on the next look")
                        if ok == "unpaired":
                            raise Stopped("unverified_write", "a like was sent but could not be read on one frame: stopping (Codex review #7)")
                    elif want_like:
                        self.ev("like_skipped", why="read-only" if self.read_only else ("no understanding" if proposal is None else "write gate / pacer / time"))
                    # a find on an IN post may be saved (controller-verify: saves only on IN posts, ≤ 0.10 per 10 posts)
                    k_save = self.kinds.get("save")
                    save_eligible = (not self.read_only and cls == "IN" and k_save is not None and self.mode in k_save.modes
                                     and ((self.gallery is not None and "save" in plan.engage)
                                          or (self.intention is not None and self.intention.find and decision.interest >= 3)))
                    if save_eligible:
                        self.count_action("save", "eligible")
                    want_save = save_eligible and ((self.gallery is not None and "save" in plan.engage) or (self.params.save is not None and self.params.save.roll(self.rng)))
                    if want_save:
                        self.count_action("save", "selected")
                    if want_save and self.now() < post_end and self.session_s() >= self.gate.first_write_after_s and self.pacer_ok("save"):
                        ok = await self.save(look, items, who if confirmed else None)
                        if self._sent_kind == "save":  # the one tap: one attempt
                            self.count_action("save", "attempted")
                            if ok in (True, False):
                                self.count_action("save", "completed")
                            if ok is True:
                                self.count_action("save", "verified")
                        if ok is False:
                            raise Stopped("silent_write_failure", "save not verified on the next look")
                        if ok == "unpaired":
                            raise Stopped("unverified_write", "a save was sent but could not be read on one frame: stopping")
                    elif want_save:
                        self.ev("save_skipped", why="write gate / pacer / time")
                    follow_eligible = (not self.read_only and self.beh.mode.writes and confirmed and cls == "IN" and "follow" in self.kinds
                                       and decision.interest >= self.params.like_interest)
                    if follow_eligible:
                        self.count_action("follow", "eligible")
                    follow_eligible = follow_eligible and self.counts.get("follows", 0) < self.params.follow_cap  # controller-verify W3: ≤ 2 a session
                    want_follow = follow_eligible and ((self.gallery is not None and "follow" in plan.engage) or (self.params.follow is not None and self.params.follow.roll(self.write_rng)))
                    if want_follow:
                        self.count_action("follow", "selected")
                    if want_follow and self.now() < post_end and self.session_s() >= self.gate.first_write_after_s and self.pacer_ok("follow"):
                        self._follow_sent = False
                        ok = await self.follow(look, items, who)
                        if self._follow_sent:  # the one tap: one attempt
                            self.count_action("follow", "attempted")
                            if ok in (True, False):
                                self.count_action("follow", "completed")
                            if ok is True:
                                self.count_action("follow", "verified")
                        if ok is False:
                            raise Stopped("silent_write_failure", "follow not verified on the profile")
                        if ok == "unpaired":
                            raise Stopped("unverified_write", "a follow was sent but could not be read on one frame: stopping")
                    elif want_follow:
                        self.ev("follow_skipped", why="write gate / pacer / time")
                    if plan.idle_s:
                        self.ev("idle", idle=plan.idle_s, tail="long_idle")  # an intentional idle inside the dwell (S4 must not read it as a stall)
                await self.sleep(max(0.0, post_end - self.now()))
                if self.now() >= end or (max_posts is not None and self.counts["posts"] >= max_posts):
                    if max_posts is not None:
                        self.ev("tail", kind="glance_and_leave", posts=self.counts["posts"])
                    break
                if await self.task_detour():  # the visit's task (run/do.py), then a short glance at the feed and leave
                    end = min(end, self.now() + self.rng.uniform(*TASK_TAIL_S))
                else:
                    await self.wander(items, plan)  # an app-level detour between this post and the next (home → … → home); the glide re-looks
                if hasattr(self.phone, "look"):  # S3 again right before the gesture, on a FRESH look: the dwell may have brought a sheet or an alert
                    for attempt in (1, 2):  # an unrecognised frame gets one more look (S3: one more look, then park); never a gesture on it
                        meta = await self.look_journaled()
                        items = (meta or {}).get("items") or items
                        ok, where = self.screen_ok(items)
                        if ok:
                            break
                        self.ev("off_feed", screen=where, n=attempt, when="before glide")
                        if attempt == 2:
                            raise Stopped("off_feed", f"screen changed during the dwell ({where})")
                        reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
                        await self.sleep(reaction)
                if self.gallery is not None and not self.gallery and not plan.engage:
                    break  # the gallery is rendered
                kind = plan.glide
                if kind == "next" and cue.decision == "skip" and plan.tail != "speed_run" and self.params.skim and self.params.skim.roll(self.rng):
                    n, _ = self.params.skim_posts.draw(self.rng)
                    kind = f"skim:{int(n)}"  # a quick skim past a run of posts he isn't stopping for
                top = pr.sc.layout(items, self.size[1], self.app)[0]
                if kind == "partial" and any(y > (top + 0.1) * self.size[1] for y in pr.sc.header_ys(items, self.size[1], self.app)):
                    kind = "next"  # the next post's header is already on screen: nothing of this post is hidden, a partial scroll would show nothing (checklist 2.4: content-driven)
                while True:
                    try:
                        await self.glide(kind, items)
                        break
                    except GestureRefused:
                        raise Stopped("no_safe_path", "no glide path clears the avoid zones on this screen")
                    except StaleInput:  # zero input sent: look again, re-gate, re-plan from what is on screen now
                        reaction, _ = self.params.glide["reaction_s"].draw(self.rng)
                        await self.sleep(reaction)
                        meta = await self.look_journaled() if hasattr(self.phone, "look") else None
                        items = (meta or {}).get("items") or items
                        ok, where = self.screen_ok(items)
                        if not ok:
                            raise Stopped("off_feed", f"the screen changed after a stale refusal ({where})")
        except Stopped as e:
            status, reason = "stopped", str(e)
            self.ev("stop", kind=e.kind, reason=e.reason)
        except Exception as e:  # S1: a daemon refusal (NotFrontmost, OperatorActive, a lost phone) is a clean stop, journaled
            status, reason = "stopped", f"phone: {type(e).__name__}: {str(e)[:200]}"
            self.ev("stop", kind="phone", reason=reason)
        finally:
            if self._vlm_task is not None and not self._vlm_task.done():
                self._vlm_task.cancel()  # no understanding outlives its session (Codex review #15)
        res = self.result(status, reason)
        digest = self.write_digest()
        res["finds"], res["digest"] = len(self.finds), (str(digest) if digest else None)
        res["actions"] = self.actions
        self.ev("end", status=status, reason=reason, counts=self.counts, plan=self.beh.plan_log, gain=self.glide_state.history, finds=len(self.finds), actions=self.actions)
        if self.evidence:
            (self.evidence / "session.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        return res

    def result(self, status: str, reason: str | None) -> dict:
        label = "done (read-only)" if self.read_only and not self.counts["writes"] else ("done" if self.counts["writes"] else "done (no writes)")
        if status != "ok":
            label = f"stopped: {reason}"
        return {"status": status, "reason": reason, "label": label, "task": self.task_result, "app": self.app, "device": self.device, "identity": self.identity, "via": self.via, "mode": self.mode,
                "read_only": self.read_only, "writes": self.counts["writes"], "counts": self.counts, "cue": self.cues.summary(),
                "persona": {"id": self.persona.id, "calibration": self.params.calibration, "seed": self.persona.seed, "params_hash": self.params_hash},
                "gain": self.glide_state.gain, "aligned_last": self.aligned,
                "seconds": round(self.session_s(), 1), "git": git_sha(), "modules": ["director.session", "director.behaviour", "director.params", "director.kinds",
                "director.primitives", "director.cues", "humanize.glide_path"] + (["director.content_vlm"] if self.vlm else [])}
