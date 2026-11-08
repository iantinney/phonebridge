"""The parameter source: mixtures with tails, quantile re-mapping, rates, operator.toml overrides."""

import random
import statistics

from phonebridge.director import params as pm


def test_dwell_mixture_has_the_surface_tails():
    p = pm.defaults()
    rng = random.Random(1)
    for surface, share in (("instagram_home", (0.005, 0.04)), ("reels", (0.02, 0.09)), ("tiktok", (0.04, 0.12))):
        xs = [p.dwell[surface].draw(rng)[0] for _ in range(6000)]
        long = sum(1 for x in xs if x > 60) / len(xs)
        assert share[0] <= long <= share[1], (surface, long)
        assert sum(1 for x in xs if x < 2) / len(xs) > 0.15  # the quick-skip spike
    ig = sorted(p.dwell["instagram_home"].draw(rng)[0] for _ in range(6000))
    assert 0.25 <= ig[600] <= 0.6 and 0.7 <= ig[3000] <= 1.6 and 3.5 <= ig[5400] <= 8  # the raw mix; the session-level shape (tilted split) is checked in test_director_glide
    assert 0.55 <= sum(1 for x in ig if x < 2) / 6000 <= 0.7 and 0.04 <= sum(1 for x in ig if x > 8) / 6000 <= 0.11


def test_quantile_remaps_the_same_draw():
    p = pm.defaults()
    m = p.dwell["instagram_home"]
    q = 0.5
    v1, c1 = m.at_quantile(q)
    v2, c2 = m.at_quantile(q, shift=1.5)
    assert c1 is c2 and abs(v2 - 1.5 * v1) < 1e-6
    assert m.at_quantile(0.999)[1].tail and not m.at_quantile(0.2)[1].tail
    assert m.at_quantile(0.3, roles=("skip",))[1].role == "skip" and m.at_quantile(0.3, roles=("read", "watch", "long"))[1].role == "read"
    assert m.at_quantile(0.01)[0] < m.at_quantile(0.6)[0] < m.at_quantile(0.999)[0]


def test_rates_and_tail_flags():
    p = pm.defaults()
    rng = random.Random(3)
    hits = sum(p.engage["like"].roll(rng) for _ in range(10000)) / 10000
    assert 0.12 < hits < 0.18 and p.tails["speed_run"].tail and not p.engage["like"].tail
    assert p.mode("intentional").writes and not p.mode("passive").writes


def test_bad_overrides_fail_loudly_and_never_half_apply(tmp_path):
    import pytest

    (tmp_path / "operator.toml").write_text('[director.params]\n"engage.like.p" = 0.04\n"engage.lyke" = 0.1\n')
    with pytest.raises(pm.ParamsError):
        pm.load(None, tmp_path / "operator.toml")
    (tmp_path / "operator.toml").write_text('[director.params.engage]\nlike = 0.03\n')
    p = pm.load(None, tmp_path / "operator.toml")
    assert p.engage["like"].p == 0.03 and isinstance(p.engage["like"], pm.Rate)


def test_operator_toml_overrides(tmp_path):
    (tmp_path / "operator.toml").write_text('[director.params]\n"engage.like" = 0.0\n"tails.speed_run" = 0.1\nreading_wps = 3.5\n')
    p = pm.load(None, tmp_path / "operator.toml")
    assert p.engage["like"].p == 0.0 and p.tails["speed_run"].p == 0.1 and p.tails["speed_run"].tail and p.reading_wps == 3.5
