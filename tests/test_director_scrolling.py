"""Post-by-post scrolling: the next post's header sets the distance, the coast is compensated, alignment is checked."""

import random

from phonebridge.director import scrolling as sc

H = 2532


def ig(*rows):
    return [{"text": t, "x": 40, "y": y, "w": 600, "h": 40, "conf": 0.9} for t, y in rows]


def test_next_header_sets_the_distance():
    items = ig(("riddhishpatel7 How did I not know this existed... more", 1230), ("2 days ago", 1360), ("aetreum_3d_visualizers", 1630), ("Follow", 1660), ("Suggested for you", 1730))
    d, how = sc.next_post_distance(items, H, random.Random(1))
    assert how == "next header" and abs(d - (1630 - sc.CONTENT_TOP * H)) < 1  # the next post's header to the content top


def test_default_when_no_next_header_is_visible():
    d, how = sc.next_post_distance(ig(("4,497", 500), ("1,065", 500)), H, random.Random(2))
    assert how == "default post height" and 0.4 * H <= d <= 0.9 * H


def test_the_planned_drag_travels_its_target_under_the_measured_model():
    """R1-IG attempts 7/8/10: the page stops with the finger below ~250 px/s of lift speed and flings ≈ 1.0 s × the lift speed
    above ~450. plan_drag inverts that and keeps the lift out of the band in between, where the phone may or may not fling."""
    for target in (400, 900, 1600, -1500):
        for T in (0.05, 0.12, 0.3, 0.6, 1.2):
            for rel in (0.05, 0.15, 0.3, 0.6):
                d, r = sc.plan_drag(target, T, rel)
                v = abs(r * d / T)
                assert v <= sc.FLING_V[0] or v >= sc.FLING_V[1], (target, T, rel, v)  # never inside the ambiguous band
                assert abs(sc.travel(d, T, r) - target) < 1.0, (target, T, rel, sc.travel(d, T, r))
    assert sc.travel(1000, 0.5, 0.1) == sc.STICK * 1000  # a slow lift: no coast
    assert sc.travel(300, 0.1, 0.5) > 1500  # a flick coasts


def test_alignment_and_nudge():
    ok, off = sc.alignment(ig(("neethashetty_official", int(0.08 * H)), ("Suggested for you", int(0.1 * H)), ("Follow", int(0.09 * H))), H)
    assert ok and abs(off) < 0.01
    ok, off = sc.alignment(ig(("neethashetty_official", int(0.30 * H)), ("Follow", int(0.31 * H))), H)
    assert not ok and 0.2 < off < 0.25 and sc.nudge_for(off, H) > 0.2 * H
    assert sc.alignment(ig(("4,497", 500),), H) == (True, None)


def test_next_post_distance_uses_the_time_line_when_the_next_header_is_off_screen():
    import random as _random

    from phonebridge.director import scrolling as sc
    h = 2532
    items = [{"text": "someone_x", "x": 300, "y": 250, "w": 300, "h": 40}, {"text": "Suggested for you", "x": 300, "y": 290, "w": 300, "h": 30},
             {"text": "a long caption about the thing... more", "x": 500, "y": 1900, "w": 900, "h": 36},
             {"text": "2 days ago", "x": 120, "y": 1960, "w": 160, "h": 30}]
    d, how = sc.next_post_distance(items, h, _random.Random(0))
    assert how == "after the post's time line" and abs(d - (1960 + sc.NEXT_AFTER_TIME * h - sc.CONTENT_TOP * h)) < 1
