"""Fast local cues (PER-04): OCR markers, motion, an optional classifier, under a latency budget; a late cue is
a content-blind state."""

import asyncio
import time

import pytest

from phonebridge.director import cues
from phonebridge.director.cues import Cue, CueEngine

SPONSORED = [{"text": "Sponsored", "x": 60, "y": 900, "w": 90, "h": 24, "conf": 0.98},
             {"text": "nike  Just do it", "x": 60, "y": 2100, "w": 500, "h": 30, "conf": 0.9}]
SUGGESTED = [{"text": "Suggested for you", "x": 60, "y": 700, "w": 200, "h": 24, "conf": 0.95}]
LIVE = [{"text": "LIVE", "x": 90, "y": 300, "w": 60, "h": 24, "conf": 0.9}]
POST = [{"text": "gympolar Strong nutrition is about more than... more", "x": 40, "y": 2000, "w": 900, "h": 30, "conf": 0.9},
        {"text": "1,239", "x": 200, "y": 1900, "w": 60, "h": 24, "conf": 0.9}]


def test_ocr_markers_classify_ads_suggestions_and_live():
    assert cues.markers(SPONSORED) == {"ad"}
    assert cues.markers(SUGGESTED) == {"suggested"}
    assert cues.markers(LIVE) == {"live"}
    assert cues.markers(POST) == set()


def test_skip_decided_by_local_cues_under_200ms():
    """OCR items are already in hand (the bridge OCRs every look); the verdict needs no model and is fast."""
    eng = CueEngine()
    t = time.perf_counter()
    cue = asyncio.run(eng.judge(SPONSORED, moving=False))
    assert (time.perf_counter() - t) < 0.2
    assert cue.cls == "ad" and cue.decision == "skip" and cue.source == "ocr"
    cue = asyncio.run(eng.judge(POST, moving=True))
    assert cue.cls == "video" and cue.decision == "stay"
    assert cue.latency_ms < 200


def test_content_blind_state_when_cue_late():
    """A classifier that answers after the budget doesn't get to decide: the state is content-blind and counted."""
    async def slow(png):
        await asyncio.sleep(0.5)
        return {"cls": "photo", "topics": ["gym"], "interest": 3}
    eng = CueEngine(classify=slow, budget_ms=120)
    cue = asyncio.run(eng.judge(POST, moving=False, png=b"png"))
    assert cue.cls == cues.BLIND and cue.source == "late" and cue.decision == "stay"
    assert cue.latency_ms >= 120
    assert eng.counts["late"] == 1 and eng.counts["content"] == 0


def test_classifier_in_budget_adds_topics_and_interest_and_counts_as_content():
    async def fast(png):
        return {"cls": "photo", "topics": ["gym", "food"], "interest": 2}
    eng = CueEngine(classify=fast, budget_ms=200)
    cue = asyncio.run(eng.judge(POST, moving=False, png=b"png"))
    assert cue.cls == "photo" and cue.topics == ["gym", "food"] and cue.interest == 2 and cue.source == "classifier"
    assert eng.counts["content"] == 1


def test_interest_zero_skips_and_high_interest_stays():
    async def bored(png):
        return {"cls": "photo", "topics": [], "interest": 0}
    assert asyncio.run(CueEngine(classify=bored).judge(POST, moving=False, png=b"p")).decision == "skip"
    async def keen(png):
        return {"cls": "video", "topics": ["gym"], "interest": 3}
    assert asyncio.run(CueEngine(classify=keen).judge(POST, moving=True, png=b"p")).decision == "stay"


def test_a_failing_classifier_is_a_blind_state_not_a_crash():
    async def boom(png):
        raise RuntimeError("no model")
    cue = asyncio.run(CueEngine(classify=boom).judge(POST, moving=False, png=b"p"))
    assert cue.cls == cues.BLIND and cue.source == "late"


def test_cue_is_a_closed_record():
    c = Cue(cls="ad", decision="skip", source="ocr", latency_ms=3.0)
    assert c.as_dict() == {"cls": "ad", "decision": "skip", "source": "ocr", "latency_ms": 3.0, "topics": [], "interest": None,
                           "markers": []}


def test_media_crop_keeps_the_middle_band():
    from PIL import Image
    import io
    from test_bridge import tiny_png
    out = cues.media_crop(tiny_png(10, 100), (0.12, 0.72))
    assert Image.open(io.BytesIO(out)).size == (10, 60)


def test_on_feed_needs_a_feed_marker():
    assert cues.on_feed("instagram", [{"text": "Your story"}, {"text": "trails_by_sample"}])
    assert cues.on_feed("tiktok", [{"text": "Following Friends For You"}])
    assert not cues.on_feed("instagram", [{"text": "Settings"}, {"text": "Calendar"}, {"text": "Photos"}])  # the home screen
    assert not cues.on_feed("instagram", [{"text": "samplecreator.co"}, {"text": "3 followers"}, {"text": "Following"}, {"text": "Edit profile"}])  # a profile (autopilot)
    assert not cues.on_feed("instagram", [{"text": "Follow and invite friends"}, {"text": "Settings"}])
    assert not cues.on_feed("tiktok", [{"text": "Friends"}, {"text": "Shop"}, {"text": "Explore more"}])  # common words aren't markers
    assert not cues.on_feed("tiktok", [{"text": "GHI"}, {"text": "JKL"}])


def test_tiktok_live_tab_is_not_a_live_badge():
    tab = [{"text": "LIVE", "x": 60, "y": 150, "w": 60, "h": 24, "conf": 0.9}, {"text": "For You", "x": 600, "y": 150, "w": 100, "h": 24, "conf": 0.9}]
    badge = [{"text": "LIVE", "x": 60, "y": 420, "w": 60, "h": 24, "conf": 0.9}]
    assert cues.markers(tab, h=2532) == set() and cues.markers(badge, h=2532) == {"live"}
    assert cues.markers(tab) == {"live"}  # without the screen height the old rule stands
