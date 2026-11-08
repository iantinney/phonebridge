"""Named gestures (INP-05) and sampling from the persona (BEH-03/INP-03): double-tap, long-press, edge-back,
drag-to-value; taps never identical; flicks rendered from a persona row."""

import math
import random

from phonebridge.director.persona import default_persona
from phonebridge.humanize import HumanConfig, Humanizer

W, H = 1170, 2532


def test_taps_never_identical_inside_box():
    h = Humanizer(HumanConfig(seed=11))
    pts = [h.tap(600, 1000, 200, 100) for _ in range(500)]
    assert len({(round(x, 3), round(y, 3), round(p, 4)) for x, y, p in pts}) == 500
    assert all(500 <= x <= 700 and 950 <= y <= 1050 for x, y, _ in pts)


def test_double_tap_is_two_nearby_taps_in_quick_succession():
    h = Humanizer(HumanConfig(seed=3))
    taps = [h.double_tap(600, 1200) for _ in range(300)]
    for (x1, y1, p1), (x2, y2, p2), gap in taps:
        assert 0.08 <= gap <= 0.35 and (x1, y1) != (x2, y2) and math.hypot(x2 - x1, y2 - y1) < 60
        assert p1 < 0.2 and p2 < 0.2
    assert len({round(g, 4) for _, _, g in taps}) > 250  # varies (0.1 ms bins: birthday collisions)


def test_long_press_holds_and_edge_back_starts_at_the_edge():
    h = Humanizer(HumanConfig(seed=4))
    x, y, press = h.long_press(500, 900)
    assert 0.5 <= press <= 1.5
    x1, y1, x2, y2, d = h.edge_back(W, H)
    assert x1 < 0.03 * W and x2 - x1 > 0.25 * W and 0.2 * H < y1 < 0.8 * H and 0.15 <= d <= 0.6


def test_drag_to_value_ends_with_a_hold_and_moves_slowly():
    h = Humanizer(HumanConfig(seed=5))
    path = h.drag_to_value(300, 1800, 800, 1800, duration=0.8)
    assert path[0][2] == 0 and path[-1][2] >= 0.8
    xs = [p[0] for p in path]
    assert abs(xs[-1] - 800) < 40 and max(abs(b - a) for a, b in zip(xs, xs[1:])) < 60  # no jump, a slow drag
    tail = [p for p in path if p[2] > path[-1][2] - 0.1]
    assert max(abs(p[0] - path[-1][0]) for p in tail) < 6  # held still at the end before the lift


def test_flick_from_persona_row_renders_length_and_duration():
    h = Humanizer(HumanConfig(seed=6))
    p = default_persona("public", seed=6)
    for row in p.flicks["instagram"][:50]:
        length_frac, duration, _ = row
        x1, y1, x2, y2, d = h.flick_from(W, H, row)
        assert abs((y1 - y2) / H - min(length_frac, 0.71)) < 0.02 and d == duration  # long rows are shortened, never moved onto a control
        assert 0.3 * W <= x1 <= 0.7 * W and 0.15 * H <= y2 < y1 <= 0.88 * H
    for row in ([0.9, 0.2, 1.0], [0.5, 0.2, 1.0]):  # S2: no touch-down or lift on the tabs/tab bar, either direction
        for direction in ("up", "down"):
            x1, y1, x2, y2, d = h.flick_from(W, H, row, direction)
            assert 0.15 * H <= min(y1, y2) and max(y1, y2) <= 0.88 * H


def test_glide_path_keeps_length_duration_and_release_exactly():
    from phonebridge.humanize import _cut_for_release, profile
    h = Humanizer(HumanConfig(seed=8))
    for release in (0.1, 0.25, 0.5):
        path = h.glide_path(600, 2000, 600, 800, 0.45, release=release)
        assert abs(path[-1][2] - 0.45) < 1e-9 and len(path) >= 4
        assert abs((path[0][1] - path[-1][1]) - 1200) < 60  # the axis length is the plan's (texture only across it)
        a, b, _ = HumanConfig().swipe_profile
        cut = _cut_for_release(a, b, release)
        pos = profile(60, a, b, cut)
        v = [(pos[i + 1] - pos[i]) for i in range(len(pos) - 1)]
        assert abs(v[-1] / (sum(v) / len(v)) - release) < 0.08  # the lift speed over the mean is the release asked for
    lo = h.glide_path(600, 2000, 600, 800, 0.45, release=0.1)
    hi = h.glide_path(600, 2000, 600, 800, 0.45, release=0.5)
    end_speed = lambda p: abs(p[-1][1] - p[-2][1]) / (p[-1][2] - p[-2][2])  # noqa: E731
    assert end_speed(lo) < end_speed(hi)


def test_swipe_still_varies_and_uses_the_same_generator():
    h = Humanizer(HumanConfig(seed=9))
    paths = [h.swipe(500, 1800, 500, 900, 0.3) for _ in range(50)]
    lens = {round(p[0][1] - p[-1][1]) for p in paths}
    durs = {round(p[-1][2], 3) for p in paths}
    assert len(lens) > 40 and len(durs) > 40
