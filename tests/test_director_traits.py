"""Per-account habits (orchestrator 10-19, Ian: "a habit per account"): each identity gets its own persona, seeded once from
public's pools with its own draws, its own persistent traits around the persona's values, and their slow daily drift
(naturalness T8b: once per account and day, x' = 0.98·x + ε in log space). The habit layer sits in the start row's
`variation`, never in `params_hash` (one persona, one hash: controller-verify V0)."""

import json
import math
import random
from datetime import date

from phonebridge.director import params as pm
from phonebridge.director import persona as ps


def test_a_new_identity_is_seeded_once_from_public_with_its_own_draws(tmp_path):
    pub, _ = ps.prepare("public", tmp_path, seed=17)
    ig, params = ps.prepare("ig_public", tmp_path, seed=23, day=date(2026, 10, 19))
    assert ig.traits and set(ig.traits) == set(pm.trait_paths(params)) and not pub.traits  # public is the centre
    for app, by_cls in ig.dwell.items():
        for cls, v in by_cls.items():
            assert set(v) <= set(pub.dwell[app][cls]) and v != pub.dwell[app][cls]  # public's values, its own draw of them
    assert all(math.exp(-2 * params.habit_sd) - 1e-6 <= m <= math.exp(2 * params.habit_sd) + 1e-6 for m in ig.traits.values())
    again, _ = ps.prepare("ig_public", tmp_path, seed=99, day=date(2026, 10, 19))  # a later seed / restart redraws nothing
    assert again.seed == ig.seed and again.traits == ig.traits and again.dwell == ig.dwell
    tt, _ = ps.prepare("tiktok_public", tmp_path, seed=31)
    assert tt.traits != ig.traits  # each account its own habits
    assert ig.side == tt.side == pub.side  # one thumb on the phone (the shared motor record)


def test_the_traits_drift_slowly_once_a_day_by_the_t8b_equation(tmp_path):
    ps.prepare("public", tmp_path, seed=17)
    ig, params = ps.prepare("ig_public", tmp_path, seed=23)
    before = dict(ig.traits)
    d1 = date(2026, 10, 20)
    assert ps.drift(ig, d1, params.drift_sd) and not ps.drift(ig, d1, params.drift_sd)  # twice the same day: a no-op
    for path, v in before.items():
        eps = random.Random(f"{ig.id}:{ig.seed}:{d1.isoformat()}:drift:{path}").gauss(0, params.drift_sd)
        assert abs(math.log(ig.traits[path]) - (0.98 * math.log(v) + eps)) < 1e-12
    p = pm.defaults()
    p.drift_sd = 0.0
    frozen = dict(ig.traits)
    assert not ps.drift(ig, date(2026, 10, 21), p.drift_sd) and ig.traits == frozen  # drift off: nothing moves


def test_the_habit_layer_is_in_the_variation_not_the_params_hash(tmp_path):
    ps.prepare("public", tmp_path, seed=17)
    ig, params = ps.prepare("ig_public", tmp_path, seed=23)
    h0 = pm.digest(params)
    q, draw = pm.vary(params, "ig_public:23:2026-10-19", "1", ig.traits)
    assert draw["habit"] == {k: ig.traits.get(k, 1.0) for k in pm.trait_paths(params)} and pm.digest(params) == h0
    q2, _ = pm.vary(params, "ig_public:23:2026-10-19", "1", None)
    assert pm.digest(q) != pm.digest(q2)  # the habits move the effective values
    saved = json.loads((tmp_path / "ig_public" / "persona.json").read_text())
    assert saved["traits"] == ig.traits  # persisted


def test_t8b_injected_innovation_and_the_same_day_no_op(tmp_path):
    """naturalness T8b: inject +0.01 in log space into x' = mu + 0.98·(x − mu) + ε (mu = 0) and check it to 1e-12; a second
    drift for the same account and day is an exact no-op."""
    ps.prepare("public", tmp_path, seed=17)
    ig, params = ps.prepare("ig_public", tmp_path, seed=23)
    before = dict(ig.traits)
    d = date(2026, 10, 22)
    assert ps.drift(ig, d, params.drift_sd, innovation=0.01)
    for path, v in before.items():
        assert abs(math.log(ig.traits[path]) - (0.98 * math.log(v) + 0.01)) < 1e-12
    snapshot = dict(ig.traits)
    assert not ps.drift(ig, d, params.drift_sd, innovation=0.01) and ig.traits == snapshot  # exact no-op
    assert ig.traits_ref.startswith("v1:") and "evidence" in ig.traits_ref
