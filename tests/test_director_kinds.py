"""Kinds are data: affordances decide probability 0, steps declare screens, the allowed-screen set follows the active step."""

from phonebridge.director import kinds as kd


def look(*texts, **extra):
    return {"items": [{"text": t, "x": 300, "y": 900, "w": 60, "h": 30} for t in texts], "width": 1170, "height": 2532, **extra}


def test_affordances_gate_each_kind():
    assert kd.KINDS["expand"].affordance(look("gympolar Strong... more"), "instagram")
    assert not kd.KINDS["expand"].affordance(look("gympolar Strong"), "instagram")
    assert kd.KINDS["carousel"].affordance(look("1/4"), "instagram") and not kd.KINDS["carousel"].affordance(look("4,497"), "instagram")
    assert kd.KINDS["carousel"].affordance(look("01 / 10 →"), "instagram") and kd.cards_left(look("01 / 10 →")) == 9  # IG's current counter
    assert not kd.KINDS["carousel"].affordance(look("10 / 10 →"), "instagram")
    assert kd.KINDS["comments_peek"].affordance(look("1,065"), "instagram")
    assert kd.KINDS["profile"].affordance(look(header=[0.08, 0.26, 0.8, 0.33]), "instagram") and not kd.KINDS["profile"].affordance(look(), "instagram")


def test_allowed_screens_follow_the_active_step():
    assert kd.allowed_screens(None, None, "instagram") == ("instagram.home", "instagram.reels")
    c = kd.KINDS["comments_peek"]
    assert "instagram.comments" in kd.allowed_screens(c, 0, "instagram") and "instagram.home" not in kd.allowed_screens(c, 0, "instagram")
    assert kd.allowed_screens(c, 2, "instagram") == ("instagram.home", "instagram.reels")
    assert all(k.steps and k.verify in ("feed_back", "heart_red", "saved", "follow_state", "none") for k in kd.KINDS.values())
    assert kd.KINDS["like"].modes == ("intentional",) and kd.KINDS["comments_peek"].modes == ("intentional",)
