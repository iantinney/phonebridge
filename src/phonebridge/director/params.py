"""The one parameter source for in-session behaviour (BEHAVIOUR_MODEL §2): every rate and distribution is a `Mixture` or a
`Rate` in the persona, per mode; heavy components ARE the tails; nothing is a constant in the code that renders them.
Ownership: the scheduler (pacing.toml / operator.toml) owns when, how long and the caps; `operator.toml` may override any
value here; `humanize` owns texture only. Defaults come from the human-likeness checklist's sources and are flagged
`calibration: pending` until his data replaces them.

Overrides (persona.json `params`, then operator.toml `[director.params]`; `normalize` is the one validator): a rate as
"engage.like.p" = 0.02, the legacy "engage.like" = 0.02 or a nested table (all one typed Rate, its tail flag kept); a number
as reading_wps = 4; a mixture as a list of components [{weight, kind, a, b, lo, hi, tail, role}]. `digest` is the effective
set's canonical hash, journaled at session start. `vary` moves a documented set (dwell and segment pace, engage.*, skim,
nudge) around that centre per day and per session (Ian: "not every day is going to be the same")."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path

from ..config import CONFIG_DIR

OPERATOR_TOML = CONFIG_DIR / "operator.toml"


@dataclass(frozen=True)
class Component:
    """One component of a mixture: lognormal (median, log-sigma) or exponential (mean), clipped to [lo, hi], with a weight."""
    weight: float
    kind: str  # lognormal | exponential | constant
    a: float  # median (lognormal), mean (exponential), value (constant)
    b: float = 0.0  # log-sigma (lognormal)
    lo: float = 0.0
    hi: float = math.inf
    tail: bool = False  # a tail component: journaled as `tail: <name>` when drawn
    role: str = ""  # dwell: skip | read | watch | long — the decision picks the sub-mixture, the post's one quantile maps inside it

    def draw(self, rng: random.Random) -> float:
        if self.kind == "lognormal":
            x = self.a * math.exp(rng.gauss(0, self.b))
        elif self.kind == "exponential":
            x = rng.expovariate(1 / self.a) if self.a > 0 else 0.0
        else:
            x = self.a
        return min(self.hi, max(self.lo, x))


@dataclass
class Mixture:
    name: str
    components: list[Component]
    _sorted: dict = field(default_factory=dict, repr=False, compare=False)  # roles → the mixture's sorted sample (monotone quantiles)

    def draw(self, rng: random.Random, shift: float = 1.0) -> tuple[float, Component]:
        """A value and the component it came from; `shift` scales the non-tail components (a segment's pace, a mode)."""
        total = sum(c.weight for c in self.components)
        r = rng.random() * total
        acc = 0.0
        for c in self.components:
            acc += c.weight
            if r <= acc:
                v = c.draw(rng)
                return (v if c.tail or c.kind == "constant" else v * shift), c
        c = self.components[-1]
        return c.draw(rng), c

    def quantile_draw(self, rng: random.Random) -> float:
        """One uniform quantile in [0, 1): drawn once per post and re-mapped as inputs arrive (§1 dwell)."""
        return rng.random()

    QUANTILE_N = 4000

    def at_quantile(self, q: float, shift: float = 1.0, roles: tuple[str, ...] | None = None) -> tuple[float, Component]:
        """The value at quantile q of the mixture (or of its `roles` sub-mixture, renormalised): a MONOTONE inverse CDF, so
        a higher q is always a longer value (the per-component mapping had jumped down at component boundaries:
        naturalness audit). Built once per roles from a fixed-seed sorted sample; the component a value came from is kept
        so tails are journaled. `shift` scales the non-tail components (interest, pace)."""
        key = tuple(roles) if roles else None
        table = self._sorted.get(key)
        if table is None:
            comps = [c for c in self.components if roles is None or c.role in roles] or self.components
            sub = Mixture(self.name, comps)
            g = random.Random(f"{self.name}:{key}")
            table = sorted((sub.draw(g) for _ in range(self.QUANTILE_N)), key=lambda vc: vc[0])
            self._sorted[key] = table
        v, c = table[min(len(table) - 1, max(0, int(q * len(table))))]
        return (v if c.tail or c.kind == "constant" else v * shift), c


def _erfinv(y: float) -> float:
    # Winitzki's approximation, enough for quantile mapping
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


@dataclass(frozen=True)
class Rate:
    """A Poisson-ish propensity per opportunity (per post, per stay, per session); `roll` with the session's multiplier."""
    p: float
    tail: bool = False

    def roll(self, rng: random.Random, multiplier: float = 1.0, interest: float = 1.0) -> bool:
        return rng.random() < min(0.95, self.p * multiplier * interest)


@dataclass
class LikeSignals:
    """Signs of interest → the like's signal factor (Ian, DECISIONS afternoon 10-19: couple them to the probability of a like).
    On each post every sign draws its own multiplier, LogN(median, sigma) and never under 1, raised to the sign's strength:
    the caption expanded (0/1), carousel cards swiped (0..n), the doublings of the time already spent on the post past
    `read_s` (−2..3: under it, a like decided at a glance, the factor drops), the fast class's IN margin past like_margin
    (0..1). So a deeper look is liked more often, by a drawn amount, never a fixed step. Calibration pending: `read` 1.6 per
    doubling is KuaiRand's like rate, 0.65 % under 10 % watched to 3.4 % at 50–90 % (≈ ×5 over 3.3 doublings); `read_s` sits
    near the time a stay's like is decided (median 5 s on the sim's IN stays), so engage.like stays the typical stay's rate
    (sim, 8 sessions at the defaults: 4.5 likes per 100 posts; the human refs' IG 5.5)."""
    expanded: float = 1.8
    card: float = 1.25
    read: float = 1.6
    margin: float = 1.5
    sigma: float = 0.35
    read_s: float = 4.0


@dataclass
class ModeSet:
    niche_w: float
    taste_w: float
    engage_mult: float
    writes: bool


@dataclass
class Params:
    """In-session behaviour for one account (persona.json → operator.toml overrides)."""
    calibration: str = "pending"
    dwell: dict[str, Mixture] = field(default_factory=dict)  # by surface: instagram_home | reels | tiktok
    glide: dict[str, Mixture] = field(default_factory=dict)  # drag_s, flick_s, release_frac, nudge_s, reaction_s
    glide_kind: Mixture | None = None  # next | partial | back | top (weights; constants)
    skim: Rate | None = None  # on a skip, the scroll passes several posts at once (Ian 05:5xZ: rare skims allowed)
    nudge: Rate | None = None  # a misaligned rest past the drawn tolerance (glide.nudge_tol) is corrected; a header cut off above always is
    skim_posts: Mixture | None = None  # how many posts a skim passes
    segment_pace: Mixture | None = None  # per-segment multiplier on dwell
    post_pace: Mixture | None = None  # per-post jitter on the segment's pace
    segment_len: Mixture | None = None  # posts per segment
    engage: dict[str, Rate] = field(default_factory=dict)  # kind → propensity per stay
    depth: dict[str, Mixture] = field(default_factory=dict)  # kind → how many repeats of its repeated step (carousel cards)
    detour: dict[str, Rate] = field(default_factory=dict)  # kind → propensity per session
    variant: dict[str, Rate] = field(default_factory=dict)  # kind → share of its parent's walks (Kind.variant_of) that take it instead
    app_detour: dict[str, Rate] = field(default_factory=dict)  # app-level detour kind → propensity per post, between posts on the home feed
    app_detour_pace: Mixture | None = None  # per session and kind, a multiplier on its app_detour rate: sessions and days differ (Ian 10-19)
    save: Rate | None = None  # a find on an IN post is saved (bookmarked) at this rate, writes allowed
    follow: Rate | None = None  # the author of an IN post is followed at this rate, writes allowed (verified on the profile)
    follow_cap: int = 2  # follows per session at most (controller-verify W3: a 60-post session holds ≤ 2; the pacer's daily caps still apply)
    tails: dict[str, Rate] = field(default_factory=dict)  # speed_run (per segment), long_idle (per post), glance_and_leave (per visit)…
    modes: dict[str, ModeSet] = field(default_factory=dict)
    orient_s: Mixture | None = None
    engage_at: Mixture | None = None  # where inside the dwell the engagement happens (fraction)
    reading_wps: float = 4.2  # words per second when reading a caption (≈ 250 wpm)
    like_interest: int = 3  # IN: the like's full weight
    like_neutral: float = 0.45  # a NEUTRAL post's like weight relative to IN (Ian: rare neutral likes; KuaiRand 0.35–0.54); AGAINST, ads, lives: 0
    like_margin: float = 0.15  # an IN from the fast classifier counts for a like only above this margin until IN recall is validated (orchestrator 07:0xZ)
    # the weak likes (NEUTRAL, an IN under the margin) as a share of all likes: their weight is held so that, over the session's
    # opportunities, they stay this share in expectation whatever the feed's mix (orchestrator R2: NEUTRAL ≤ 20 % of likes;
    # 0.12 leaves room for the draw's spread). A fixed weight can't hold a share: at 0.45, a feed of 3 NEUTRAL stays per IN gives
    # NEUTRAL 57 % of the likes. like_neutral stays the per-view ceiling (KuaiRand) when IN dominates
    like_neutral_share: float = 0.12
    like_heart: Rate | None = None  # the share of likes made on the heart button; the rest double-tap the media (unknown: Ian's export measures it)
    like_signals: LikeSignals = field(default_factory=LikeSignals)  # signs of interest × the like's probability (behaviour.like_factor)
    describe_min_s: float = 2.0  # every stay at least this long is described (Q1 is scored per stay; both modes)
    # a day's and a session's move around this centre (`vary`): log-space sd, truncated at ±2 sd, on pace and on a rate's odds.
    # Unfitted engineering priors (Ian 2026-10-19: "not every day is going to be the same"); 0 turns a level off
    vary_day: float = 0.15
    vary_session: float = 0.06
    habit_sd: float = 0.15  # an account's own habits around the persona's values, drawn once (log sd; Ian: a habit per account)
    drift_sd: float = 0.02  # their slow daily drift: x' = 0.98·x + N(0, drift_sd) in log space, once per account and day (0 = none)

    def mode(self, name: str) -> ModeSet:
        return self.modes.get(name) or self.modes["passive"]


def defaults() -> Params:
    """Research defaults, `calibration: pending`. Dwell per surface from controller-verify's human references (8 public screen
    recordings, verify/human-refs, 2026-10-19) and the checklist's sources: IG home p10 0.4 / p50 0.9 / p90 5.3 s, 62 % under
    2 s, 7 % over 8 s, 1–2 % over 60 s; TikTok p50 3.7 s, 28 % over 8 s (the recordings; VANT26's TikTok-wide 12 % over 60 s needs
    long videos, so the long weight sits between the two at 7 %); Reels p50 9 s, 60 % over
    8 s, ~5 % over 60 s. The weights are the overall mix (skip weight ≈ the reference skip share); the session maps a skip into
    the skip component and a stay into the others."""
    def L(w, med, sig, lo=0.0, hi=math.inf, tail=False, role=""):
        return Component(w, "lognormal", med, sig, lo, hi, tail, role)

    def C(w, v, tail=False):
        return Component(w, "constant", v, tail=tail)

    dwell_ig = Mixture("dwell:instagram_home", [L(0.62, 0.8, 0.75, 0.2, 2.5, role="skip"), L(0.30, 2.8, 0.55, 1.5, 30, role="read"),
                                                L(0.065, 11, 0.5, 5, 60, role="watch"), L(0.015, 90, 0.6, 45, 600, tail=True, role="long")])
    dwell_reels = Mixture("dwell:reels", [L(0.25, 1.2, 0.65, 0.3, 3.5, role="skip"), L(0.45, 10, 0.5, 3, 40, role="watch"),
                                          L(0.25, 22, 0.5, 8, 60, role="watch"), L(0.05, 90, 0.6, 45, 900, tail=True, role="long")])
    # inside the app-level detours: one story (most are tapped through; an untouched photo story advances itself after ~5 s),
    # the first glance at the Explore grid and after each scroll of it; a reel on the Reels tab or from Explore uses dwell:reels
    dwell_story = Mixture("dwell:story", [L(0.7, 2.2, 0.5, 0.6, 6), L(0.3, 5.5, 0.3, 3, 15)])
    dwell_grid = Mixture("dwell:grid", [L(1.0, 2.0, 0.5, 0.6, 8)])
    dwell_tt = Mixture("dwell:tiktok", [L(0.40, 1.2, 0.65, 0.3, 3.5, role="skip"), L(0.36, 4.5, 0.55, 2, 40, role="read"),
                                        L(0.17, 16, 0.5, 8, 60, role="watch"), L(0.07, 95, 0.6, 45, 1200, tail=True, role="long")])
    return Params(
        dwell={"instagram_home": dwell_ig, "reels": dwell_reels, "tiktok": dwell_tt, "story": dwell_story, "grid": dwell_grid},
        # drag (finger contact) durations, controller-verify's bands from 8 human clips: p05 16–67 ms, median 34–233 ms,
        # p95 202–1067 ms; the slow tail is reading while scrolling
        glide={"drag_s": Mixture("drag_s", [L(0.98, 0.15, 0.75, 0.03, 1.2), L(0.02, 0.95, 0.15, 0.7, 1.3, tail=True)]),  # slow-drag tail trimmed:
               # controller-verify's dry-run judge, gesture p95 1.23 s vs the human ceiling 1.1 s (8 % of glides drew 1-2.5 s)
               "flick_s": Mixture("flick_s", [L(1.0, 0.17, 0.4, 0.07, 0.35)]),
               "release_frac": Mixture("release_frac", [L(1.0, 0.22, 0.4, 0.05, 0.6)]),  # release speed / mean speed for `next`
               "nudge_s": Mixture("nudge_s", [L(1.0, 0.3, 0.3, 0.2, 0.6)]),
               "close_s": Mixture("close_s", [L(1.0, 0.42, 0.12, 0.35, 0.5)]),  # the story viewer's slow close swipe (surfaces4: 0.35-0.5 s)
               # how far below the top a post's header may rest before he bothers (fraction of h): people mostly read a half-framed
               # post (controller-verify's refs rest framed ~70 %); attempt 10 nudged 63 of 114 glides
               "nudge_tol": Mixture("nudge_tol", [L(1.0, 0.16, 0.35, 0.08, 0.35)]),
               "reaction_s": Mixture("reaction_s", [L(1.0, 0.55, 0.5, 0.15, 3.0)])},
        glide_kind=Mixture("glide_kind", [C(0.90, 0), C(0.07, 1), C(0.03, 2)]),  # 0 next, 1 partial, 2 back
        # human refs: IG home posts per gesture p90 2.33; Ian chose rare skims, only while skipping: ~15 % of skips ≈ 8–10 % of scrolls
        skim=Rate(0.15), skim_posts=Mixture("skim_posts", [C(0.7, 2), C(0.3, 3)]), nudge=Rate(0.4),
        segment_pace=Mixture("segment_pace", [L(1.0, 1.0, 0.45, 0.4, 3.0)]),
        post_pace=Mixture("post_pace", [L(1.0, 1.0, 0.4, 0.4, 2.5)]),  # the pace drifts post to post inside a segment (no stretch of equal dwells)
        segment_len=Mixture("segment_len", [L(1.0, 8, 0.6, 2, 40)]),
        # per stay (× mode multiplier), only when the affordance is on screen; human refs: IG carousel swipes 2.2 / 10 posts,
        # likes 0.55 / 10 (ours stay lower: Ian 01:2xZ "low rate, clearly on-niche"); TikTok likes / comment opens / pauses ~0.33 / 10
        # comments: OFF until surfaces verifies how each app's sheet is dismissed (06:5xZ: IG ignored a fast drag from the title,
        # and the old start point was above the sheet, over the post header's Follow pill)
        engage={"expand": Rate(0.08), "carousel": Rate(0.8), "comments_peek": Rate(0.0), "tap_media": Rate(0.08), "profile": Rate(0.04), "like": Rate(0.15)},  # expand: controller-verify 1.43 per 10 posts vs the human tail ≤ 0.5 ([REF] 0 in 121); coupled to class
        variant={"comment_profile": Rate(0.3)},  # of the comment opens (off above until the sheet's dismiss is proven), these go on to a commenter
        depth={"carousel": Mixture("carousel_depth", [L(1.0, 1.8, 0.5, 1, 6)]),  # cards swiped per carousel stay (human refs: 2.2 swipes / 10 IG posts)
               "stories": Mixture("stories_depth", [L(1.0, 1.6, 0.7, 0, 4)]),  # taps to the next story after the first: 1–5 stories watched
               "explore_scroll": Mixture("explore_scroll", [L(1.0, 1.0, 0.8, 0, 3)]),  # scrolls of the Explore grid before a tile: 0–3
               "reels": Mixture("reels_depth", [L(1.0, 3.2, 0.4, 2, 6)])},  # snap flicks on the Reels tab: 2–6
        detour={"profile_dive": Rate(0.02, tail=True), "exploration": Rate(0.05)},
        # per post on the home feed (stories only while the tray shows, i.e. near the top: mostly at the start), × a per-session
        # multiplier per kind; calibration pending (no human reference for tab visits yet)
        app_detour={"stories": Rate(0.3), "explore": Rate(0.012), "reels_tab": Rate(0.012)},
        app_detour_pace=Mixture("app_detour_pace", [L(1.0, 1.0, 0.7, 0.1, 4.0)]),
        save=Rate(0.04),  # controller-verify's bar: saves ≤ the 97.5 % Poisson count of 0.10 per 10 posts
        follow=Rate(0.015),  # per eligible IN post (confident IN, confirmed author): controller-verify W3, IG ≤ 0.05 per 10 posts (TikTok ≤ 0.10)
        tails={"speed_run": Rate(0.02, tail=True), "long_idle": Rate(0.01, tail=True), "glance_and_leave": Rate(0.07, tail=True), "scroll_far_back": Rate(0.005, tail=True), "linger": Rate(0.01, tail=True)},
        # passive engagement = the human references (ordinary people scrolling); intentional engages more on the niche
        modes={"passive": ModeSet(0.5, 1.0, 1.0, False), "intentional": ModeSet(1.0, 0.5, 1.5, True)},
        orient_s=Mixture("orient_s", [L(1.0, 2.0, 0.5, 0.8, 6.0)]),
        engage_at=Mixture("engage_at", [L(1.0, 0.45, 0.4, 0.15, 0.8)]),
        like_heart=Rate(0.3),  # double-tap vs heart: no published ratio (DIRECTOR_GOAL_DRAFT); a double-tap-leaning prior, calibration pending
    )


class ParamsError(ValueError):
    """A config that can't take effect exactly (bad file, unknown field, wrong type, bad value, two spellings of one field);
    nothing was applied (all or nothing)."""


def read_json(path: Path) -> dict:
    """A JSON file, strictly: a key written twice in one object is an error (json would keep the last one silently); a parse
    error names the file and position."""
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"key {k!r} appears twice")
            out[k] = v
        return out
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    except ValueError as e:
        raise ParamsError(f"{path}: {e}") from None


def _pairs(d: dict, prefix: str = ""):
    """Nested tables ([director.params.engage.like] p = 0.02) give the same dotted keys as "engage.like.p" = 0.02; a field
    spelled twice stays twice (a conflict, never last-wins)."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from _pairs(v, key)
        else:
            yield key, v


def _child(box, part: str, key: str):
    """One step of a dotted path: a dict's key or a field of Params / ModeSet (never inside a Rate, Mixture or Component)."""
    if isinstance(box, dict):
        if part in box:
            return box[part]
        raise ParamsError(f"{key}: no {part!r} (known: {', '.join(sorted(box))})")
    if isinstance(box, (Params, ModeSet, LikeSignals)) and part in box.__dataclass_fields__ and not part.startswith("_"):
        return getattr(box, part)
    raise ParamsError(f"{key}: no {part!r}")


def _field(root: Params, key: str):
    """(canonical path, container, leaf, current value) of a dotted key. `x.p` names the rate x, so "engage.like.p" and the
    legacy parent scalar "engage.like" are one field: `engage.like`."""
    parts = key.split(".")
    rate = len(parts) > 1 and parts[-1] == "p"
    if rate:
        parts = parts[:-1]
    box = root
    for part in parts[:-1]:
        box = _child(box, part, key)
    cur = _child(box, parts[-1], key)
    if rate and not isinstance(cur, Rate):
        raise ParamsError(f"{key}: {'.'.join(parts)} is a {type(cur).__name__}, not a rate")
    return ".".join(parts), box, parts[-1], cur


# numeric fields with a meaning-bound range, by path (0.2: orchestrator R2's ceiling; a sign's median ≥ 1: a sign never lowers the like)
RANGES = {"like_neutral": (0.0, 1.0), "like_neutral_share": (0.0, 0.2), "like_margin": (0.0, 0.95),
          **{f"like_signals.{k}": (1.0, 5.0) for k in ("expanded", "card", "read", "margin")},
          "like_signals.sigma": (0.0, 1.5), "like_signals.read_s": (0.1, 60.0)}


def _number(key: str, val, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        raise ParamsError(f"{key} = {val!r}: a finite number is required")
    if (lo is not None and val < lo) or (hi is not None and val > hi):
        raise ParamsError(f"{key} = {val!r}: must be within [{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]")
    return float(val)


def _components(key: str, val) -> list[Component]:
    """A mixture override: a non-empty list of components {weight, kind, a[, b, lo, hi, tail, role]} with finite numbers,
    weights >= 0 and not all zero (a zero-weight component is allowed beside positive ones), lo <= hi (hi omitted: none)."""
    if not isinstance(val, list):
        raise ParamsError(f"{key} = {val!r}: a mixture needs a list of components ({{weight, kind, a, ...}})")
    if not val:
        raise ParamsError(f"{key}: an empty mixture")
    out = []
    for i, c in enumerate(val):
        at = f"{key}[{i}]"
        if not isinstance(c, dict):
            raise ParamsError(f"{at} = {c!r}: a component is a table {{weight, kind, a, ...}}")
        if bad := sorted(set(c) - set(Component.__dataclass_fields__)) + [f"missing {k}" for k in ("weight", "kind", "a") if k not in c]:
            raise ParamsError(f"{at}: {', '.join(bad)}")
        if c["kind"] not in ("lognormal", "exponential", "constant"):
            raise ParamsError(f"{at}.kind = {c['kind']!r}: lognormal, exponential or constant")
        if not isinstance(c.get("tail", False), bool) or not isinstance(c.get("role", ""), str):
            raise ParamsError(f"{at}: tail is true/false and role a string")
        lo = _number(f"{at}.lo", c.get("lo", 0.0))
        hi = _number(f"{at}.hi", c["hi"]) if "hi" in c else math.inf
        if lo > hi:
            raise ParamsError(f"{at}: lo {lo} > hi {hi}")
        out.append(Component(_number(f"{at}.weight", c["weight"], 0.0), c["kind"], _number(f"{at}.a", c["a"], 0.0),
                             _number(f"{at}.b", c.get("b", 0.0), 0.0), lo, hi, c.get("tail", False), c.get("role", "")))
    if not sum(c.weight for c in out) > 0:
        raise ParamsError(f"{key}: all component weights are zero")
    return out


def _value(key: str, cur, val, path: str | None = None):
    """The typed, validated value for a field whose current value is `cur`: a rate's p in [0, 1], a mixture's components,
    a finite non-negative number (whole for an int field)."""
    if isinstance(cur, Rate):
        return _number(key, val, 0.0, 1.0)
    if isinstance(cur, Mixture):
        return _components(key, val)
    if isinstance(cur, (int, float)) and not isinstance(cur, bool):
        v = _number(key, val, *RANGES.get(path or key, (0.0, None)))  # a field's own range where it has one (like_neutral_share ≤ 0.2)
        if isinstance(cur, int) and v != int(v):
            raise ParamsError(f"{key} = {val!r}: a whole number is required")
        return type(cur)(v)
    raise ParamsError(f"{key}: a {type(cur).__name__}; set a rate, a number or a mixture")


def normalize(overrides: dict | None, source: str = "overrides") -> dict:
    """One source's overrides as {canonical path: typed value}, checked against the defaults' schema. The documented forms
    ("engage.like.p", legacy "engage.like", nested tables) are one field; two spellings of one field in one source, an
    unknown path, a wrong type or a bad value raise ParamsError naming each, so nothing comes back half-valid."""
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ParamsError(f"{source}: a table of overrides is required, not {type(overrides).__name__}")
    schema, out, seen, errors = defaults(), {}, {}, []
    for key, val in _pairs(overrides):
        try:
            path, _, _, cur = _field(schema, key)
            if path in seen:
                raise ParamsError(f"{key}: {path} is already set by {seen[path]!r} in this source (one spelling per field)")
            seen[path] = key
            out[path] = _value(key, cur, val, path)
        except ParamsError as e:
            errors.append(str(e))
    if errors:
        raise ParamsError(f"{source}: " + "; ".join(errors))
    return out


def _apply(p: Params, values: dict) -> None:
    """Set normalize()'d values: a new Rate keeps the field's tail flag, a new Mixture its journaled name; fresh objects, so
    nothing shared with another load (or the defaults) is mutated."""
    for path, v in values.items():
        _, box, leaf, cur = _field(p, path)
        new = Rate(v, cur.tail) if isinstance(cur, Rate) else Mixture(cur.name, list(v)) if isinstance(cur, Mixture) else v
        if isinstance(box, dict):
            box[leaf] = new
        else:
            setattr(box, leaf, new)


def load(persona_dir: Path | None = None, operator_toml: Path = OPERATOR_TOML) -> Params:
    """Defaults, then persona.json's `params`, then operator.toml's `[director.params]`: operator over persona over defaults.
    Each source is normalised and validated whole before anything is applied, so every override takes effect exactly or
    the load raises ParamsError naming the field, before anything runs (naturalness T8a)."""
    layers = []
    f = Path(persona_dir) / "persona.json" if persona_dir else None
    if f is not None and f.exists():
        layers.append(normalize(read_json(f).get("params"), f"{f} params"))
    if Path(operator_toml).exists():
        try:
            toml = tomllib.loads(Path(operator_toml).read_text(encoding="utf-8"))
        except ValueError as e:
            raise ParamsError(f"{operator_toml}: {e}") from None
        layers.append(normalize(toml.get("director", {}).get("params"), f"{operator_toml} [director.params]"))
    p = defaults()
    for values in layers:
        _apply(p, values)
    return p


def digest(p: Params) -> str:
    """The canonical effective-parameter hash, for provenance (journaled at session start): values only (an int and the
    equal float are one value), so key order, file names, load time and the session seed don't enter it; any value change
    does."""
    def plain(x):
        if is_dataclass(x):
            return {f.name: plain(getattr(x, f.name)) for f in fields(x) if not f.name.startswith("_")}
        if isinstance(x, dict):
            return {str(k): plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        return float(x) if isinstance(x, int) and not isinstance(x, bool) else x  # 30 and 30.0 are one value
    return hashlib.sha256(json.dumps(plain(p), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def trait_paths(p: Params) -> list[str]:
    """The parameters an account's habits, its day and its session move (the documented set of `vary`)."""
    return ["dwell", "segment_pace", *(f"engage.{k}" for k in p.engage), "skim", "nudge"]


def vary(p: Params, day_key: str, session_key: str, traits: dict | None = None) -> tuple[Params, dict]:
    """This day's and this session's parameters around the account's centre `p` (its defaults → persona → operator values;
    `p` itself is never touched). Each varied parameter moves by a day multiplier, seeded by `day_key` (account + calendar
    date: the same day reproduces exactly and a restart redraws nothing), times a smaller session multiplier seeded by
    `session_key` (the session's rng seed); each is exp(N(0, sd)) truncated at ±2 sd, sd = vary_day / vary_session, drawn
    from its own stream per parameter. The documented set: `dwell` and `segment_pace` (pace: the non-tail components' median
    and bounds scale; tails stay), the `engage.*` rates, `skim` and `nudge` (the multiplier is on the odds, so p stays in
    [0, 1] and a rate that is off stays off). Never written taste, intentions or pacer caps (not params). Returns the
    session's params and the draw {day, session: {path: multiplier}} for the journal."""
    paths = trait_paths(p)

    def moves(key: str, sd: float) -> dict:
        out = {}
        for path in paths:
            z = random.Random(f"{key}:{path}").gauss(0, sd) if sd > 0 else 0.0
            out[path] = round(math.exp(max(-2 * sd, min(2 * sd, z))), 6)
        return out

    def pace(m: Mixture, f: float) -> Mixture:
        return Mixture(m.name, [c if c.tail or c.kind == "constant" else replace(c, a=c.a * f, lo=c.lo * f, hi=c.hi * f) for c in m.components])

    def odds(r: Rate, f: float) -> Rate:
        return r if f == 1 else Rate(r.p * f / (1 - r.p + r.p * f), r.tail)  # (1 - p + p) need not be exactly 1.0

    day, session = moves(day_key, p.vary_day), moves(session_key, p.vary_session)
    habit = {k: float((traits or {}).get(k, 1.0)) for k in paths}  # the account's persistent habits (persona.traits): the centre's offset
    f = {k: habit[k] * day[k] * session[k] for k in paths}
    q = copy.deepcopy(p)
    q.dwell = {s: pace(m, f["dwell"]) for s, m in q.dwell.items()}
    q.segment_pace = pace(q.segment_pace, f["segment_pace"])
    q.engage = {k: odds(r, f[f"engage.{k}"]) for k, r in q.engage.items()}
    q.skim, q.nudge = odds(q.skim, f["skim"]), odds(q.nudge, f["nudge"])
    return q, {"habit": habit, "day": day, "session": session}

