"""The generator (BEHAVIOUR_MODEL §1): Session → Segment → Post plan, every number drawn from `Params`. Pure and seeded: it
plans; `primitives` render. A plan is data the journal can carry (`plan` per session, `tail: kind` when a tail component or a
tail rate fired, `segment pace` per segment)."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from .params import Params

GLIDE_KINDS = ("next", "partial", "back")
SURFACE = {"instagram": "instagram_home", "tiktok": "tiktok", "reels": "reels"}


@dataclass
class Segment:
    n_posts: int
    pace: float  # multiplier on dwell (Gamma-like from the mixture)
    tail: str | None = None  # speed_run → every post quick-skipped


@dataclass
class PostPlan:
    q: float  # the one dwell quantile, drawn at arrival
    glide: str  # next | partial | back
    engage: list[str]  # kinds rolled before affordances are known (the renderer drops the ones without an affordance)
    tail: str | None = None  # long_idle, speed_run(from the segment), …
    idle_s: float = 0.0  # intentional idle inside the dwell (journaled `idle: s`)
    jitter: float = 1.0  # this post's drift on the segment's pace
    comp_tail: str | None = None  # the dwell drew a tail component (long_watch): journaled with the view


def snaps(app: str) -> bool:
    """Does this app's feed page (snap one video per flick: TikTok) rather than glide post by post (IG, X)? surfaces' feed maps."""
    from .feedmaps import FEEDMAPS

    return app in FEEDMAPS and FEEDMAPS[app].glide == "page"


class Behaviour:
    def __init__(self, params: Params, mode: str, app: str, rng: random.Random):
        self.p, self.mode, self.app, self.rng = params, params.mode(mode), app, rng
        self.mode_name = mode
        self.surface = SURFACE.get(app, "instagram_home")
        self.segment: Segment | None = None
        self.left = 0
        self.plan_log: list[dict] = []
        self.like_seen = {"full": 0.0, "weak": 0.0}  # like opportunities this session, each by its signal factor: confident IN vs the weak bucket
        self.last_like: dict = {}

    # --- session / visit -------------------------------------------------------------------------
    def visit_plan(self) -> dict:
        """What the visit looks like before the first post: orientation and the glance-and-leave tail."""
        orient, _ = self.p.orient_s.draw(self.rng)
        leave = self.p.tails["glance_and_leave"].roll(self.rng)
        plan = {"mode": self.mode_name, "surface": self.surface, "orient_s": round(orient, 2), "glance_and_leave": leave}
        self.plan_log.append({"visit": plan})
        return plan

    def next_segment(self) -> Segment:
        n, _ = self.p.segment_len.draw(self.rng)
        pace, _ = self.p.segment_pace.draw(self.rng)
        tail = "speed_run" if self.p.tails["speed_run"].roll(self.rng) else None
        if tail == "speed_run":
            n, pace = max(15, min(25, round(n * 2))), 0.25
        self.segment = Segment(max(1, round(n)), round(pace, 3), tail)
        self.left = self.segment.n_posts
        self.plan_log.append({"segment": asdict(self.segment)})
        return self.segment

    # --- per post ------------------------------------------------------------------------------------
    def post_plan(self) -> PostPlan:
        if self.segment is None or self.left <= 0:
            self.next_segment()
        self.left -= 1
        q = self.p.dwell[self.surface].quantile_draw(self.rng)
        kind_idx, _ = self.p.glide_kind.draw(self.rng)
        glide = "page" if snaps(self.app) else GLIDE_KINDS[int(kind_idx)]  # TikTok pages; IG and X glide post by post
        tail = self.segment.tail
        idle = 0.0
        if tail is None and self.p.tails["long_idle"].roll(self.rng):
            tail, idle = "long_idle", self.rng.lognormvariate(4.4, 0.5)  # ~80 s, the phone put down
        elif tail is None and self.p.tails["linger"].roll(self.rng):
            tail = "linger"  # stays on a post that doesn't interest him (9.T)
        engage = [] if tail == "speed_run" else [k for k, r in self.p.engage.items() if k not in ("like",) and r.roll(self.rng, self.mode.engage_mult)]
        jitter, _ = self.p.post_pace.draw(self.rng) if self.p.post_pace else (1.0, None)
        return PostPlan(q, glide, engage, tail, round(idle, 1), round(jitter, 3))

    def passive_split(self, plan: PostPlan, action: str, interest: int, forced: bool | None = None) -> tuple[bool, float]:
        """Passive mode has no niche (Ian 04:1xZ): the human dwell mixture itself decides. The post's one quantile, tilted by
        what the glance saw (a video or something of his tilts toward staying, a plain still toward skipping), lands in the
        skip part of the mixture or in the stay part; the returned u is the quantile re-scaled inside that part. `forced`
        (the fast class against his written words: IN stays, AGAINST is a quick skip; DIRECTOR_GOAL_DRAFT's table) takes
        the side, and the post's quantile spans that part whole."""
        if forced is not None:
            return forced, plan.q
        m = self.p.dwell[self.surface]
        w_skip = sum(c.weight for c in m.components if c.role == "skip") / sum(c.weight for c in m.components)
        tilt = 0.4 if interest >= 3 else 0.6 if interest == 2 else (0.85 if action != "skip" else 1.25)
        # the glance moves the threshold (q^tilt < w_skip ⇔ q < w_skip^(1/tilt)), not the post's place inside the part: a stay
        # keeps the human read/watch/long weights (R1 attempt 7: the tilted quantile gave 19 % of dwells > 8 s vs 7 % human)
        thr = w_skip ** (1 / tilt)
        if plan.q < thr:
            return False, plan.q / thr
        return True, (plan.q - thr) / (1 - thr)

    STILL_MAX_S = 45.0  # a still is read, not watched: no minutes-long hold (the phone-down idle is its own tail)
    SNAP_MAX_S = 60.0  # a snap-feed video (TikTok, Reels) watched one or two plays at most: no video length is read yet, so a cap
    #                  (controller-verify: a 143 s stay on one TikTok post; a short video would have looped several times)

    def dwell_s(self, plan: PostPlan, interest: int, words: int = 0, skip: bool = False, u: float | None = None, still: bool = False) -> float:
        """Seconds from the post's ONE quantile, re-mapped as inputs arrive (u: the same quantile re-scaled by
        `passive_split`). A skip maps it into the skip component (the glance); a stay into the read/watch/long components,
        shifted by interest and the pace, plus reading time for the caption's words. In passive mode (u given) the stay
        uses the mixture's own read/watch/long weights; in intentional mode a low-interest stay reads and moves on and only
        a post he cares about (or the off-niche linger tail, 9.T) reaches watch/long. A speed run glances at everything."""
        q = plan.q if u is None else u
        pace = (self.segment.pace if self.segment else 1.0) * plan.jitter
        m = self.p.dwell[self.surface]
        if plan.tail == "speed_run" or skip:  # the glance: no hard floor (a floor turns a run of quick skips into equal intervals)
            v, _ = m.at_quantile(q, pace ** 0.5 if plan.tail != "speed_run" else 0.6 * plan.jitter, roles=("skip",))
            return min(1.2, max(0.12, v)) if plan.tail == "speed_run" else max(0.12, v)  # a speed run: 10–30 posts at under a second each (2.T)
        if u is not None:
            shift, roles = pace, ("read", "watch", "long")
        else:
            shift = {0: 0.6, 1: 0.8, 2: 1.0, 3: 1.4}.get(interest, 1.0) * pace
            roles = ("read",) if interest <= 1 and plan.tail != "linger" else ("read", "watch", "long")
        if still:  # the long-watch tail is for videos (R1 attempt 7 held a still for 123 s); a still's longest stay is a careful read
            roles = tuple(r for r in roles if r != "long") or ("read",)
        v, comp = m.at_quantile(q, shift, roles=roles)
        plan.comp_tail = "long_watch" if comp.tail else None
        if words and not comp.tail and interest >= 2:  # the human dwells already include reading; a post of his (Ian's table: IN reads
            v = max(v, words / self.p.reading_wps)  # the text) is read through: a floor, not an addition (R1 attempt 7: 19 % > 8 s vs 7 %)
        if still:
            v = min(v, self.STILL_MAX_S)
        elif self.surface in ("tiktok", "reels"):
            v = min(v, self.SNAP_MAX_S)
        return v + plan.idle_s

    def like_eligible(self, interest: int, aligned: bool, cls: str | None = None) -> bool:
        """Whether a like is allowed at all on this post (mode, alignment, class): a gallery may force the opportunity,
        never this (Codex review #6). Only a post the classifier read as IN or NEUTRAL: AGAINST / ad / live never, and no
        class (no classifier, a failed read) means no like: missing evidence is never a neutral match."""
        return self.mode.writes and aligned and cls in ("IN", "NEUTRAL") and interest > 0

    def like_factor(self, cls: str | None, margin: float | None, signals: dict | None) -> float:
        """The signs of interest on this post as a multiplier on its like probability (params.LikeSignals): the caption
        expanded, carousel cards swiped, the time already spent (doublings past read_s), the IN margin past like_margin.
        Each sign draws its own multiplier for this post, so the effect rises with the sign by a drawn amount."""
        if not signals:
            return 1.0
        g, spent = self.p.like_signals, signals.get("spent_s") or 0.0
        strength = {"expanded": 1.0 if signals.get("expanded") else 0.0, "card": float(signals.get("cards") or 0),
                    "read": min(3.0, max(-2.0, math.log2(spent / g.read_s))) if spent > 0 else 0.0,
                    "margin": max(0.0, margin - self.p.like_margin) / (1 - self.p.like_margin) if cls == "IN" and margin is not None else 0.0}
        f = 1.0
        for k, s in strength.items():
            if s:
                f *= max(1.0, getattr(g, k) * math.exp(self.rng.gauss(0, g.sigma))) ** s
        return f

    def like_wanted(self, interest: int, aligned: bool, cls: str | None = None, margin: float | None = None, signals: dict | None = None) -> bool:
        """A like (Ian, overnight grill): intentional mode, aligned on the post, at the base rate × the class weight × the
        post's signs of interest (like_factor; Ian, afternoon 10-19); the pacer caps it. Each call is one opportunity. A
        confident IN (interest at like_interest, and the fast classifier's margin until IN recall is validated) weighs 1.0.
        Every other like is weak (NEUTRAL whatever the glance's interest, an IN under the margin): weight like_neutral, held
        down so weak likes stay like_neutral_share of all likes over this session's opportunities, each counted by its
        factor (expected weak likes Σ w·F_weak ≤ s/(1−s) × Σ F_full, however long NEUTRAL posts are read; none before a
        confident IN). `last_like` keeps the roll for the journal."""
        if not self.like_eligible(interest, aligned, cls):
            return False
        full = cls == "IN" and interest >= self.p.like_interest and (margin is None or margin >= self.p.like_margin)
        factor = self.like_factor(cls, margin, signals)
        self.like_seen["full" if full else "weak"] += factor
        s = self.p.like_neutral_share
        weight = 1.0 if full else min(self.p.like_neutral, s / (1 - s) * self.like_seen["full"] / self.like_seen["weak"])
        self.last_like = {"weight": round(weight, 4), "factor": round(factor, 3)}
        return self.p.engage["like"].roll(self.rng, self.mode.engage_mult, interest=weight * factor)
