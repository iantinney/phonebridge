"""The semantic cue: an interest score from his taste corpus (text space, image space on a Mac), the nearest item named;
the dwell policy conditioned on interest."""

import random

from phonebridge.director import dwell as dw
from phonebridge.director.cues import Cue, decide
from phonebridge.director.semantic import TasteModel, TextSpace

TASTE = {"source": "test", "items": [
    {"title": "Alex Hormozi business advisor gives rapid-fire advice on revenue and marketing", "summary": "a talking head on entrepreneurship", "tags": ["business-growth", "marketing", "entrepreneurship"], "creator": "ahormozi"},
    {"title": "A founder lists daily habits: showing up early, preparing the night before", "summary": "productivity and discipline", "tags": ["productivity", "habits", "mindset"], "creator": "ahormozi"},
]}


def test_scores_rank_his_kind_of_post_above_an_unrelated_one():
    m = TasteModel(TASTE)
    biz = m.score("entrepreneur explains how to grow revenue with better marketing")
    cat = m.score("my cat knocked over a plant again")
    assert biz.score > cat.score and biz.interest >= 2 and cat.interest <= 1
    assert "Hormozi" in biz.nearest and biz.parts["text_backend"] in ("bag-of-words", "nl-sentence")
    assert m.topics[:3] and "business-growth" in m.topics


def test_empty_taste_is_neutral_not_a_guess():
    m = TasteModel({"items": []})
    r = m.score("anything")
    assert r.interest == 1 and r.nearest == "" and r.parts["note"] == "no taste data"


def test_decide_cites_the_nearest_taste_item():
    m = TasteModel(TASTE)
    match = m.score("founder shares the daily habits and discipline behind his productivity")
    d = decide(Cue("video", "stay", "motion", 5.0), [{"text": "x"}], {}, match=match)
    assert d.action == "stay" and "≈ '" in d.reason and d.interest == min(2, match.interest)  # the glance never likes; understanding may
    from phonebridge.director.content_vlm import Proposal
    told = Proposal("video", False, ["habits"], 3, "", False, "a founder lists his daily habits")
    d2 = decide(Cue("video", "stay", "motion", 5.0), [{"text": "x"}], {}, described=told, match=match)
    assert d2.interest == match.interest and d2.reason.startswith("a founder lists his daily habits")  # the sentence is cited; its 3/3 is ignored
    strong = m.score("a founder lists his daily habits for productivity and discipline", topics=["habits", "productivity", "mindset"])
    d3 = decide(Cue("video", "stay", "motion", 5.0), [{"text": "x"}], {}, described=told, match=strong)
    assert d3.interest == strong.interest and (d3.action == "like_candidate") == (strong.interest == 3)


def test_dwell_conditioned_on_interest_uses_his_distribution_ends():
    rng = random.Random(3)
    pool = sorted(rng.lognormvariate(1.8, 0.7) for _ in range(400))
    model = dw.DwellModel({"video": pool}, rng=random.Random(1))
    low = [model.draw_for("video", 0) for _ in range(500)]
    high = [model.draw_for("video", 3) for _ in range(500)]
    mid = [model.draw_for("video", 2) for _ in range(500)]
    assert sorted(low)[250] < sorted(mid)[250] < sorted(high)[250]
    assert min(high) >= pool[int(0.6 * 399)] * 0.5  # the long end of his pool, smoothed


def test_clean_strips_the_apps_chrome():
    from phonebridge.director.semantic import clean
    items = [{"text": "For You", "y": 150}, {"text": "LIVE", "y": 150}, {"text": "6,567", "y": 900}, {"text": "•oll", "y": 20},
             {"text": "When Magnus Carlsen chooses you", "y": 1800}, {"text": "Following Friends For You", "y": 140}]
    assert clean(items, 2532) == "when magnus carlsen chooses you"


def test_nl_percentile_calibration_uses_the_background(monkeypatch):
    """With a fake embedding distance, a post closer than most neutral sentences scores high and one farther scores low."""
    from phonebridge.director import semantic as sm
    ts = sm.TextSpace(background=sm.BACKGROUND)
    ts.nl = object()  # pretend the framework is there
    def dist(a, b):  # distances FROM the post: to the taste doc, and to each neutral sentence (1.1)
        if b in sm.BACKGROUND:
            return 1.1
        return 0.9 if "business" in a else 1.3
    monkeypatch.setattr(ts, "_dist", dist)
    assert ts.similarity("business advice on revenue", "doc") == 1.0 and ts.similarity("my cat", "doc") == 0.0


def test_topic_tags_are_a_precise_channel():
    m = TasteModel(TASTE)
    assert m.tag_overlap(["mindset", "entrepreneur", "video"], ["mindset", "entrepreneurship", "habits"]) == 1.0
    assert m.tag_overlap(["chess", "esports"], ["mindset", "entrepreneurship"]) == 0.0
    with_tags = m.score("a man talks about building discipline", topics=["mindset", "productivity"])
    without = m.score("a man talks about building discipline", topics=["chess"])
    assert with_tags.score > without.score and with_tags.parts["tag_overlap"] == 1.0
