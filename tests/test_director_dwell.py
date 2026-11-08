"""Dwell and flick models: smoothed bootstrap from a persona's pools (BEH-03), never a metronome (BEH-12),
"latency is the glance" (PER-04), persona defaults flagged pending (BEH-16)."""

import math
import random
import statistics

import pytest

from phonebridge.director import dwell as dw
from phonebridge.director import flick as fl
from phonebridge.director.persona import Persona, default_persona

from fixtures import metronome



def gaps_of(ts):
    """Seconds between consecutive scrolls of a session (the old engage run: the ~6.8 s metronome)."""
    return [b - a for a, b in zip(ts, ts[1:])]


def autocorr(xs, lag=1):
    m = statistics.mean(xs)
    num = sum((xs[i] - m) * (xs[i + lag] - m) for i in range(len(xs) - lag))
    den = sum((x - m) ** 2 for x in xs)
    return num / den if den else 0.0


def test_default_persona_is_flagged_pending_and_has_pools_per_app():
    p = default_persona("public", seed=3)
    assert p.calibration == "pending"
    for app in ("instagram", "tiktok"):
        assert len(p.dwell[app]["video"]) >= 100 and len(p.flicks[app]) >= 100
        assert all(len(row) == 3 for row in p.flicks[app])  # (length frac, duration s, onset s)


def test_persona_round_trips_through_its_directory(tmp_path):
    p = default_persona("public", seed=3)
    p.save(tmp_path)
    q = Persona.load("public", tmp_path)
    assert q == p
    assert (tmp_path / "public" / "persona.json").exists()


def test_bootstrap_dwell_quantiles_match_profile():
    """Draws follow the pool's distribution (quantiles within tolerance) without copying its values."""
    rng = random.Random(1)
    pool = [math.exp(rng.gauss(math.log(6.0), 0.6)) for _ in range(300)]
    model = dw.DwellModel({"video": pool}, rng=random.Random(2))
    draws = [model.draw("video") for _ in range(3000)]
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        a, b = statistics.quantiles(pool, n=100)[int(q * 100) - 1], statistics.quantiles(draws, n=100)[int(q * 100) - 1]
        assert abs(math.log(a) - math.log(b)) < 0.15, (q, a, b)
    assert not set(draws) & set(pool)  # smoothed: never a verbatim pool value


def test_dwell_is_drawn_when_the_class_arrives_minus_elapsed():
    model = dw.DwellModel({"video": [8.0] * 50, "photo": [3.0] * 50}, rng=random.Random(0))
    full = model.draw("video")
    remaining = model.remaining("video", elapsed=0.4, draw=full)
    assert remaining == pytest.approx(max(0.0, full - 0.4))
    assert model.remaining("video", elapsed=99.0, draw=full) == 0.0


def test_content_blind_state_pools_every_class():
    model = dw.DwellModel({"video": [8.0] * 50, "photo": [3.0] * 50, "ad": [1.0] * 50}, rng=random.Random(0))
    blind = [model.draw(dw.BLIND) for _ in range(600)]
    assert 1.0 < statistics.median(blind) < 8.0  # a mixture of every class, not one of them


def test_no_metronome_gap_autocorrelation():
    """The old run's scroll gaps were a beat; a session of our draws has no lag-1 structure and real spread."""
    old_gaps = gaps_of(metronome.SCROLL_S)
    assert statistics.pstdev(old_gaps) / statistics.mean(old_gaps) < 0.6  # the fixture really is a metronome
    p = default_persona("public", seed=5)
    rng = random.Random(5)
    model, flicks = dw.DwellModel(p.dwell["instagram"], rng=rng), fl.FlickModel(p.flicks["instagram"], rng=rng)
    classes = ["video", "photo", "photo", "ad", "video", "suggested"]
    gaps = [model.draw(rng.choice(classes)) + flicks.draw().onset_s for _ in range(400)]
    assert statistics.pstdev(gaps) / statistics.mean(gaps) > 0.6
    assert abs(autocorr(gaps)) < 0.15
    assert len({round(g, 2) for g in gaps}) > 300  # varies (10 ms bins: birthday collisions, not repeats)


def test_flick_joint_sample_is_never_verbatim_and_stays_in_envelope():
    rng = random.Random(9)
    pool = [(rng.uniform(0.2, 0.7), rng.uniform(0.09, 0.3), rng.uniform(0.2, 1.5)) for _ in range(200)]
    model = fl.FlickModel(pool, rng=random.Random(1))
    for _ in range(500):
        f = model.draw()
        assert min(d for d in (fl.distance(f.row, r) for r in pool)) > 0.0
        assert 0.05 <= f.length_frac <= 0.95 and 0.05 <= f.duration_s <= 1.0 and 0.0 <= f.onset_s <= 6.0
    # joint: the length/duration correlation of the pool survives in the draws
    xs = [r[0] for r in pool]; ys = [r[1] for r in pool]
    rows = [model.draw().row for _ in range(2000)]
    assert abs(statistics.correlation([r[0] for r in rows], [r[1] for r in rows]) - statistics.correlation(xs, ys)) < 0.2
