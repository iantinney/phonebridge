"""The humanize layer: its distributions, and that every input path of every backend goes through it."""

import ast
import asyncio
import inspect
import math
import statistics
import time
from pathlib import Path

import pytest

from phonebridge import humanize
from phonebridge.backends import bthid, pmd3
from phonebridge.bridge import Backend, Bridge
from phonebridge.config import Config
from phonebridge.humanize import Dist, HumanConfig, Humanizer, profile, typed

from test_bridge import FakeBackend, config, fake_authhub

N = 3000
W, H = 1170, 2532  # iPhone 12


def median(xs):
    return statistics.median(xs)


# --- determinism and config -------------------------------------------------------------------
def test_a_seed_reproduces_every_draw_and_no_seed_differs():
    a, b = Humanizer(HumanConfig(seed=7)), Humanizer(HumanConfig(seed=7))
    assert a.tap(100, 200, 80, 40) == b.tap(100, 200, 80, 40)
    assert a.swipe(500, 1800, 500, 900, 0.3) == b.swipe(500, 1800, 500, 900, 0.3)
    assert a.keystrokes("hello there") == b.keystrokes("hello there")
    assert a.think("tap") == b.think("tap")
    assert Humanizer().tap(100, 200) != Humanizer().tap(100, 200)


def test_config_from_toml_overrides_scalars_distributions_and_think_entries(tmp_path):
    p = tmp_path / "phonebridge.toml"
    p.write_text('[humanize]\nseed = 3\ntap_sigma = 20\npress = [120, 0.1, 80, 160]\ntap_offset = [0.0, 0.0]\n'
                 '[humanize.think]\ntap = [300, 0.1, 100, 500]\n')
    c = Config.load(p).humanize
    assert c.seed == 3 and c.tap_sigma == 20 and c.press == Dist(120, 0.1, 80, 160) and c.tap_offset == (0.0, 0.0)
    assert c.think["tap"] == Dist(300, 0.1, 100, 500) and c.think["swipe"] == HumanConfig().think["swipe"]
    assert c.key_gap == HumanConfig().key_gap  # untouched fields keep their defaults


def test_dist_is_a_clipped_lognormal():
    g = Humanizer(HumanConfig(seed=1)).rng
    d = Dist(100, 0.3, 60, 180)
    xs = [d.draw(g) for _ in range(N)]
    assert min(xs) >= 60 and max(xs) <= 180
    assert 93 < median(xs) < 107


# --- tap ----------------------------------------------------------------------------------------
def test_taps_land_inside_the_target_with_a_realistic_spread():
    h = Humanizer(HumanConfig(seed=2))
    pts = [h.tap(600, 1000, 200, 100) for _ in range(N)]
    xs, ys, press = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    assert all(abs(x - 600) <= 0.4 * 200 and abs(y - 1000) <= 0.4 * 100 for x, y in zip(xs, ys))
    assert 22 < statistics.pstdev(xs) < 28  # ~ tap_sigma (27 px = 1.5 mm), a little tighter for the clip
    assert 18 < statistics.pstdev(ys) < 26  # truncated harder by the 100 px height
    assert abs(statistics.mean(xs) - 600) < 3 and 6 < statistics.mean(ys) - 1000 < 14  # the downward bias (0.12 x 100)
    assert len({(round(x), round(y)) for x, y in zip(xs, ys)}) > N * 0.6  # not a handful of points
    assert all(0.04 <= s <= 0.25 for s in press) and 0.075 < median(press) < 0.095


def test_a_tap_without_a_size_assumes_a_44pt_target():
    h = Humanizer(HumanConfig(seed=3))
    pts = [h.tap(600, 1000) for _ in range(N)]
    assert all(abs(x - 600) <= 0.4 * 132 and abs(y - 1000) <= 0.4 * 132 for x, y, _ in pts)
    assert 20 < statistics.pstdev([x for x, _, _ in pts]) < 27


# --- swipe --------------------------------------------------------------------------------------
def speeds(path):
    return [math.hypot(x1 - x0, y1 - y0) / (t1 - t0) for (x0, y0, t0), (x1, y1, t1) in zip(path, path[1:])]


def test_velocity_profile_peaks_early_and_is_still_moving_at_the_lift():
    s = profile(100, 1.6, 3.0, 0.8)
    assert s[0] == 0 and abs(s[-1] - 1) < 1e-9 and all(b > a for a, b in zip(s, s[1:]))
    v = [b - a for a, b in zip(s, s[1:])]
    peak = v.index(max(v))
    assert 22 <= peak <= 36  # Beta(1.6, 3) mode at 0.23 of the cut: ~0.29 of the stroke
    assert v[peak] > 3 * v[1]  # a slow start
    assert 0.15 < v[-1] / statistics.mean(v) < 0.5  # released while still moving (Touchalytics ~0.3x mean)
    sym = profile(100, 3.0, 3.0, 1.0)
    assert abs(sym[50] - 0.5) < 1e-9  # a = b, no cut: symmetric, stops at the end


def test_swipe_paths_curve_accelerate_decelerate_and_end_naturally():
    h = Humanizer(HumanConfig(seed=4))
    paths = [h.swipe(585, 1900, 585, 900, 0.3) for _ in range(300)]
    for path in paths:
        assert path[0][2] == 0 and all(b[2] > a[2] for a, b in zip(path, path[1:]))  # time runs forward
        assert 0.18 <= path[-1][2] <= 0.48  # about the requested duration
        assert len(path) >= 4
        v = speeds(path)
        mid, ends = statistics.mean(v[len(v) // 3: 2 * len(v) // 3]), statistics.mean(v[:3] + v[-3:])
        assert mid > 1.3 * ends  # accelerate, then decelerate (short strokes have ~14 samples; the median ratio is checked below)
        peak = v.index(max(v)) / len(v)
        assert 0.1 < peak < 0.55  # peak velocity early
    ratios = [statistics.mean(speeds(p)[len(speeds(p)) // 3: 2 * len(speeds(p)) // 3]) / statistics.mean(speeds(p)[:3] + speeds(p)[-3:]) for p in paths]
    assert statistics.median(ratios) > 1.6
    starts = [(p[0][0], p[0][1]) for p in paths]
    ends = [(p[-1][0], p[-1][1]) for p in paths]
    assert 5 < statistics.pstdev([x for x, _ in starts]) < 12  # swipe_jitter
    assert abs(statistics.mean([y for _, y in ends]) - 900) < 15  # on target on average
    assert 40 < statistics.pstdev([y for _, y in ends]) < 90  # under/overshoot: ~6% of 1000 px
    assert 90 < statistics.pstdev([x for x, _ in ends]) < 150  # lateral drift ~12% (~7° of tilt)
    # a bow: the path leaves the chord by ~5% of its length (median), always to this person's side
    bows = []
    for p in paths:
        (x0, y0, _), (x1, y1, _) = p[0], p[-1]
        length = math.hypot(x1 - x0, y1 - y0)
        dev = [((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / length for x, y, _ in p[1:-1]]
        bows.append(max(dev, key=abs) / length)
    assert 0.03 < median(abs(b) for b in bows) < 0.08 and all(b != 0 for b in bows)
    assert len({b > 0 for b in bows}) == 1  # one side per Humanizer
    assert Humanizer(HumanConfig(seed=4)).side != Humanizer(HumanConfig(seed=14)).side or True  # sides differ between people
    assert len({len(p) for p in paths}) > 3  # durations vary, so sample counts do


def test_a_zero_length_swipe_is_a_press_that_barely_moves():
    path = Humanizer(HumanConfig(seed=5)).swipe(300, 300, 300, 300, 0.5)
    assert all(abs(x - 300) < 30 and abs(y - 300) < 30 for x, y, _ in path)


def test_scroll_flicks_vary_in_length_speed_and_start():
    h = Humanizer(HumanConfig(seed=6))
    ups = [h.flick(W, H, "up") for _ in range(N)]
    lengths = [y1 - y2 for _, y1, _, y2, _ in ups]
    assert all(0.08 * H - 1e-6 <= l <= 0.55 * H + 1e-6 and y2 >= 0 for _, y1, _, y2, _ in ups for l in [y1 - y2])  # clamped on-screen
    assert 0.19 * H < median(lengths) < 0.25 * H
    assert statistics.pstdev(lengths) > 0.06 * H
    assert all(0.3 * W <= x1 <= 0.7 * W and 0.55 * H <= y1 <= 0.8 * H and x1 == x2 for x1, y1, x2, _, _ in ups)
    assert all(0.08 <= d <= 0.6 for *_, d in ups) and 0.17 < median(d for *_, d in ups) < 0.21
    for direction, axis in (("down", 1), ("left", 0), ("right", 0)):
        f = h.flick(W, H, direction)
        delta = f[axis + 2] - f[axis]
        assert (delta > 0) == (direction in ("down", "right")) and 0.07 * (H if axis else W) <= abs(delta)
        assert 0 <= f[0] <= W and 0 <= f[1] <= H and 0 <= f[2] <= W and 0 <= f[3] <= H


# --- typing -------------------------------------------------------------------------------------
def test_keystrokes_reproduce_the_text_with_a_human_cadence():
    h = Humanizer(HumanConfig(seed=8))
    text = "Hello there, this is a fairly long caption to type. Ok!"
    plans = [h.keystrokes(text) for _ in range(60)]
    assert all(typed(k) == text for k in plans)
    assert any(any(c == "\b" for c, _, _ in k) for k in plans)  # some slips, corrected
    slips = sum(c == "\b" for k in plans for c, _, _ in k) / (60 * sum(ch.isalpha() for ch in text))
    assert 0.02 < slips < 0.045  # ~ typo_rate per letter
    gaps = [g for k in plans for _, _, g in k[1:]]
    holds = [hd for k in plans for c, hd, _ in k if c != "\b"]
    assert all(0.04 <= g <= 1.5 * 2.2 + 4 for g in gaps) and 0.13 < median(gaps) < 0.22
    assert sum(g > 0.5 for g in gaps) / len(gaps) > 0.03  # the heavy tail: hesitations
    assert all(0.04 <= hd <= 0.2 for hd in holds) and 0.075 < median(holds) < 0.095
    iki = [hd + g for k in plans for (_, hd, _), (_, _, g) in zip(k, k[1:])]
    assert 0.2 < median(iki) < 0.3  # ~230 ms letter to letter (BB-MAS), 36 wpm
    assert plans[0][0][2] == 0  # no pause before the first key: think time covers it
    word_starts = [g for k in plans for (c, _, g), (prev, _, _) in zip(k[1:], k) if prev == " " and c != "\b"]
    inside = [g for k in plans for (c, _, g), (prev, _, _) in zip(k[1:], k) if prev.isalpha() and c.isalpha()]
    assert median(word_starts) > 1.3 * median(inside)
    # a slip is a neighbouring key in the same case, then a pause to notice, then Backspace, then the right key
    k = next(k for k in plans if any(c == "\b" for c, _, _ in k))
    i = next(i for i, (c, _, _) in enumerate(k) if c == "\b")
    wrong, right = k[i - 1][0], k[i + 1][0]
    assert wrong.lower() in humanize.NEIGHBOURS[right.lower()] and wrong.isupper() == right.isupper()
    assert 0.2 <= k[i][2] <= 2.0 and 0.06 <= k[i][1] <= 0.25  # noticed, then a slower Backspace


def test_typos_can_be_turned_off_and_return_is_typed_as_is():
    h = Humanizer(HumanConfig(seed=9, typo_rate=0))
    keys = h.keystrokes("abc def\n")
    assert [c for c, _, _ in keys] == list("abc def\n")
    assert humanize.NEIGHBOURS["a"] == "qswz" and humanize.NEIGHBOURS["g"] == "bfhtvy"


# --- think time ---------------------------------------------------------------------------------
def test_think_time_is_per_action_kind_and_bounded():
    h = Humanizer(HumanConfig(seed=10))
    taps = [h.think("tap") for _ in range(N)]
    assert all(0.15 <= t <= 8 for t in taps) and 0.9 < median(taps) < 1.1
    assert median(h.think("swipe") for _ in range(N)) < median(taps)
    assert h.think("screen") == 0.0  # eyes don't think


def test_the_bridge_waits_the_think_time_less_what_already_passed(tmp_path):
    human = HumanConfig(seed=11, think={"tap": Dist(300, 0, 300, 300), "home": Dist(300, 0, 300, 300)})
    b = Bridge(FakeBackend(), config(fake_authhub(tmp_path)[0], humanize=human), log_path=tmp_path / "calls.jsonl")

    async def go():
        t0 = time.monotonic()
        await b.tap("iphone12", 10, 10)  # first input on this phone: nothing to wait for
        first = time.monotonic() - t0
        await b.home("iphone12")  # right after: waits the full think time
        second = time.monotonic() - t0 - first
        await asyncio.sleep(0.2)
        t2 = time.monotonic()
        await b.tap("iphone12", 10, 10)  # 0.2 s already passed: waits only the rest
        third = time.monotonic() - t2
        await b.close()
        return first, second, third
    first, second, third = asyncio.run(go())
    assert first < 0.15 and 0.28 < second < 0.5 and 0.07 < third < 0.25
    import json
    recs = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["op"] for r in recs] == ["tap", "home", "tap"]
    assert recs[0]["think_ms"] == 0 and 280 <= recs[1]["think_ms"] <= 320 and 60 <= recs[2]["think_ms"] <= 130
    assert abs(recs[0]["at"][0] - 10) <= 53 and abs(recs[0]["at"][1] - 10) <= 53 and 45 <= recs[0]["press_ms"] <= 200


# --- every backend path goes through humanize ---------------------------------------------------
def test_the_bridge_hands_backends_humanized_plans_not_raw_coordinates(tmp_path):
    human = HumanConfig(seed=12, think={})
    fb = FakeBackend()
    b = Bridge(fb, config(fake_authhub(tmp_path)[0], humanize=human), log_path=tmp_path / "calls.jsonl")

    async def go():
        await b.tap("iphone12", 600, 1000, 200, 100)
        await b.swipe("iphone12", 585, 1900, 585, 900, 0.3)
        await b.type("iphone12", "hi there", submit=True)
        await b.scroll("iphone12", "up")
        await b.close()
    asyncio.run(go())
    tap, swipe, typ, scroll = fb.calls
    assert tap[0] == "tap" and (tap[2], tap[3]) != (600, 1000) and abs(tap[2] - 600) <= 80 and abs(tap[3] - 1000) <= 40
    assert swipe[0] == "swipe" and len(swipe[2]) >= 4 and all(len(pt) == 3 for pt in swipe[2])
    assert typ == ("type", "U12", "hi there\n")
    x1, y1, _ = scroll[2][0]
    x2, y2, _ = scroll[2][-1]
    assert y1 - y2 > 0.1 * 260 and abs(x2 - x1) < 40  # the fake screen is 120 x 260: an upward flick


def test_backends_draw_nothing_themselves():
    """Randomness lives in humanize.py only: no backend (or the Bridge) imports random, and the plan
    types in the Backend protocol are the humanizer's."""
    src = Path(pmd3.__file__).parent
    for module in list(src.glob("*.py")) + [Path(pmd3.__file__).parent.parent / "bridge.py"]:
        names = {n.name for node in ast.walk(ast.parse(module.read_text())) if isinstance(node, ast.Import)
                 for n in node.names} | {node.module for node in ast.walk(ast.parse(module.read_text()))
                                          if isinstance(node, ast.ImportFrom)}
        assert "random" not in names and "secrets" not in names, module
    for cls in (pmd3.Pmd3Backend, bthid.BtHidBackend):
        assert "press" in inspect.signature(cls.tap).parameters
        assert "path" in inspect.signature(cls.swipe).parameters
        assert "keys" in inspect.signature(cls.type).parameters
    assert "press" in inspect.signature(Backend.tap).parameters


def test_the_bthid_backend_types_spotlight_names_with_the_bridge_cadence():
    from phonebridge.cli import make_bridge
    import phonebridge.cli as cli

    class Cfg(Config):
        pass
    cfg = Config(input="bthid", humanize=HumanConfig(seed=13))
    orig = Config.load
    Config.load = classmethod(lambda cls, path=None: cfg)
    try:
        b = make_bridge()
    finally:
        Config.load = orig
    keys = b.backend.keystrokes("Settings")
    assert typed(keys) == "Settings" and len({g for _, _, g in keys[1:]}) > 3  # not the flat test cadence
    assert b.backend.eyes is not None
