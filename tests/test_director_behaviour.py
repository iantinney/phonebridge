"""The generator over many seeded sessions: dwell shape per interest, engagement at rates, tails as rates (Poisson-ish, never
regular), a speed run quick-skips, quantile re-mapping with words."""

import random
import statistics

import pytest

from phonebridge.director import params as pm
from phonebridge.director.behaviour import Behaviour


def sessions(n=400, mode="passive", app="instagram", posts=30):
    out = []
    for seed in range(n):
        b = Behaviour(pm.defaults(), mode, app, random.Random(seed))
        b.visit_plan()
        out.append((b, [b.post_plan() for _ in range(posts)]))
    return out


def test_dwell_reflects_interest_and_words():
    b = Behaviour(pm.defaults(), "passive", "instagram", random.Random(1))
    b.next_segment()
    b.segment.pace = 1.0
    plan = b.post_plan()
    plan.tail, plan.idle_s = None, 0.0
    d0, d1, d3 = b.dwell_s(plan, 0, skip=True), b.dwell_s(plan, 1), b.dwell_s(plan, 3)
    assert d0 < d1 < d3 and 0.3 <= d0 <= 2.0
    assert b.dwell_s(plan, 2, words=40) >= 40 / b.p.reading_wps  # a post of his is read through (~250 wpm): a floor
    assert b.dwell_s(plan, 1, words=40) == b.dwell_s(plan, 1)  # elsewhere the human dwells already include reading (R1 attempt 7)


def test_tails_fire_at_rates_not_on_a_schedule():
    runs = sessions(400)
    idle = sum(1 for _, plans in runs for p in plans if p.tail == "long_idle")
    speed = sum(1 for b, _ in runs for s in b.plan_log if "segment" in s and s["segment"]["tail"] == "speed_run")
    total_posts = sum(len(plans) for _, plans in runs)
    assert 0.004 < idle / total_posts < 0.02 and speed > 3
    gaps = []
    for _, plans in runs:
        idx = [i for i, p in enumerate(plans) if p.tail == "long_idle"]
        gaps += [b - a for a, b in zip(idx, idx[1:])]
    if len(gaps) > 10:
        assert statistics.pstdev(gaps) / statistics.mean(gaps) > 0.5  # Poisson-ish, not periodic


def test_speed_run_segments_quick_skip_everything():
    b = Behaviour(pm.defaults(), "passive", "instagram", random.Random(5))
    b.next_segment()
    b.segment.tail, b.segment.n_posts, b.left = "speed_run", 20, 20
    plans = [b.post_plan() for _ in range(20)]
    assert all(p.tail == "speed_run" and not p.engage for p in plans)
    assert all(b.dwell_s(p, 3) <= 1.2 for p in plans)


def test_engagement_rates_scale_with_mode_and_likes_need_the_threshold():
    passive = sessions(200, "passive")
    intent = sessions(200, "intentional")
    rate = lambda runs: sum(len(p.engage) for _, plans in runs for p in plans) / sum(len(plans) for _, plans in runs)  # noqa: E731
    assert rate(intent) > rate(passive) * 1.2  # intentional engages 1.5× the human base rate passive uses
    b = Behaviour(pm.defaults(), "intentional", "instagram", random.Random(2))
    rate = lambda *a: sum(b.like_wanted(*a) for _ in range(6000)) / 6000  # noqa: E731
    assert rate(1, True, "NEUTRAL") == 0  # no confident IN yet this session: no weak like
    r_in, r_neutral = rate(3, True, "IN"), rate(3, True, "NEUTRAL")  # NEUTRAL is weak whatever the glance's interest
    assert r_in > 0.1 and 0.05 * r_in < r_neutral < 0.25 * r_in  # rare NEUTRAL likes (Ian): held at like_neutral_share of likes
    assert rate(3, False, "IN") == 0 and rate(0, True, "IN") == 0 and rate(3, True, "AGAINST") == 0 and rate(3, True, "AD") == 0
    assert rate(3, True) == 0 and rate(3, True, None, 0.9) == 0  # no class (no classifier, a failed read): never a like
    assert rate(3, True, "IN", 0.05) < 0.7 * r_in  # an IN below the margin is weak until IN recall is validated
    bp = Behaviour(pm.defaults(), "passive", "instagram", random.Random(2))
    assert not any(bp.like_wanted(3, True, "IN") for _ in range(300))


def test_glide_kinds_are_mostly_next_on_instagram_and_page_on_tiktok():
    runs = sessions(100)
    kinds = [p.glide for _, plans in runs for p in plans]
    assert 0.85 < kinds.count("next") / len(kinds) < 0.95 and "partial" in kinds and "back" in kinds
    assert all(p.glide == "page" for _, plans in sessions(5, app="tiktok") for p in plans)


def test_a_still_never_draws_the_long_watch_tail():
    """R1 attempt 7: a still drew a 123 s hold; the long component is for videos, a still's stay is capped."""
    b = Behaviour(pm.defaults(), "passive", "instagram", random.Random(7))
    b.next_segment()
    worst = 0.0
    for q in [k / 200 for k in range(200)]:
        plan = b.post_plan()
        plan.q, plan.tail, plan.idle_s = q, None, 0.0
        worst = max(worst, b.dwell_s(plan, 3, 0, u=q, still=True))
    assert worst <= Behaviour.STILL_MAX_S
    plan.q = 0.999
    assert b.dwell_s(plan, 3, 0, u=0.999, still=False) > 45  # videos keep the tail


def test_passive_dwells_keep_the_human_long_share():
    """R1 attempt 7: 19 % of dwells > 8 s (human IG home ~7 %), with 123 s and 70 s holds in ten minutes. At attempt 7's glance
    mix the share over 8 s stays near the human band and minute-long holds are rare."""
    mix = [(0, "skip")] * 18 + [(1, "skip")] * 37 + [(2, "stay")] * 6 + [(3, "stay")]
    b = Behaviour(pm.defaults(), "passive", "instagram", random.Random(3))
    b.next_segment()
    rng, n, l8, l60 = random.Random(4), 6000, 0, 0
    for _ in range(n):
        interest, action = rng.choice(mix)
        plan = b.post_plan()
        stay, u = b.passive_split(plan, action, interest)
        d = b.dwell_s(plan, interest, rng.randint(5, 38), u=u, still=True) if stay else b.dwell_s(plan, 0, skip=True, u=u)
        l8 += d > 8
        l60 += d > 60
    assert 0.03 <= l8 / n <= 0.10 and l60 / n <= 0.006


def test_a_snap_feed_video_is_watched_a_play_or_two_not_minutes():
    """controller-verify: a 143 s stay on one TikTok post. Without a video length to read, a snap-feed watch is capped."""
    b = Behaviour(pm.defaults(), "passive", "tiktok", random.Random(5))
    b.next_segment()
    worst = 0.0
    for q in [k / 300 for k in range(300)]:
        plan = b.post_plan()
        plan.q, plan.tail, plan.idle_s = q, None, 0.0
        worst = max(worst, b.dwell_s(plan, 3, 0, u=q))
    assert worst <= Behaviour.SNAP_MAX_S


def test_no_metronome_run_of_three_matching_dwells():
    """controller-verify: 5 consecutive dwells ≥ 1 s within ±15 % in a 284-post session (human clips: never more than 2).
    Director.unmetronome moves a third matching dwell out of the band."""
    import sys

    sys.path.insert(0, "tests")
    session = pytest.importorskip("test_director_glide").session  # absent where the director's sim isn't shipped

    def longest(dw):
        best = cur = 1
        for a, b in zip(dw, dw[1:]):
            cur = cur + 1 if a >= 1 and b >= 1 and abs(b - a) <= 0.15 * a else 1
            best = max(best, cur)
        return best
    for seed in (1, 2, 3, 11):
        d, *_ = session(seed=seed, minutes=15, full_length=True)
        dw = [e["dwell"] for e in d.events if e["ev"] == "view"]
        assert longest(dw) <= 2, (seed, longest(dw))
