"""The background vision model for writes only (BEH-02, PER-05): closed schema, low effort, cached in replay,
proposals gated by dwell fraction, first-write time and the pacer."""

import asyncio
import json

import pytest

from phonebridge.director import content_vlm as cv


def test_schema_is_closed_and_validated():
    ok = {"kind": "video", "is_ad": False, "topics": ["gym", "food"], "interest": 2, "text_to_read": "", "quick_skip": False, "summary": "a cook plates a steak"}
    assert cv.validate(ok) == cv.Proposal(**ok)
    assert cv.validate({k: v for k, v in ok.items() if k != "summary"}).summary == ""  # summary is optional on the way in
    with pytest.raises(ValueError):
        cv.validate({**ok, "topics": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError):
        cv.validate({**ok, "interest": 5})
    with pytest.raises(ValueError):
        cv.validate({**ok, "extra": 1})
    with pytest.raises(ValueError):
        cv.validate({k: v for k, v in ok.items() if k != "kind"})
    assert set(cv.Proposal.__dataclass_fields__) == set(cv.FIELDS)


def test_prompt_names_the_profile_and_the_schema():
    p = cv.prompt(profile={"topics": ["gym", "cooking"], "likes": ["calisthenics"]})
    assert "gym" in p and "interest" in p and "quick_skip" in p and "topics" in p
    assert cv.EFFORT == "low"


def test_calls_are_cached_by_frame_hash_for_replay(tmp_path):
    calls = []
    async def client(png, prompt, effort):
        calls.append(effort)
        return {"kind": "photo", "is_ad": False, "topics": ["gym"], "interest": 3, "text_to_read": "", "quick_skip": False}
    vlm = cv.CachedVLM(client, tmp_path / "vlm-cache.jsonl", profile={"topics": ["gym"]})
    a = asyncio.run(vlm.describe(b"frame-1"))
    b = asyncio.run(vlm.describe(b"frame-1"))
    assert a == b and calls == ["low"]
    again = cv.CachedVLM(client, tmp_path / "vlm-cache.jsonl", profile={"topics": ["gym"]})
    assert asyncio.run(again.describe(b"frame-1")) == a and calls == ["low"]  # replay: served from disk


def test_like_only_after_dwell_fraction():
    prop = cv.Proposal(kind="video", is_ad=False, topics=["gym"], interest=3, text_to_read="", quick_skip=False)
    gate = cv.WriteGate(first_write_after_s=60, like_after_fraction=0.5)
    assert gate.candidates(prop, elapsed_fraction=0.2, session_s=400, pacer_ok=lambda a: True) == []
    assert "like" in gate.candidates(prop, elapsed_fraction=0.6, session_s=400, pacer_ok=lambda a: True)
    assert gate.candidates(prop, elapsed_fraction=0.9, session_s=400, pacer_ok=lambda a: a != "like") == []


def test_first_write_after_60s():
    prop = cv.Proposal(kind="video", is_ad=False, topics=["gym"], interest=3, text_to_read="", quick_skip=False)
    gate = cv.WriteGate(first_write_after_s=60, like_after_fraction=0.5)
    assert gate.candidates(prop, elapsed_fraction=0.9, session_s=59, pacer_ok=lambda a: True) == []
    assert gate.candidates(prop, elapsed_fraction=0.9, session_s=61, pacer_ok=lambda a: True)


def test_ads_and_low_interest_never_propose():
    gate = cv.WriteGate(first_write_after_s=0, like_after_fraction=0.0)
    ad = cv.Proposal(kind="video", is_ad=True, topics=["gym"], interest=3, text_to_read="", quick_skip=False)
    meh = cv.Proposal(kind="video", is_ad=False, topics=["gym"], interest=1, text_to_read="", quick_skip=False)
    assert gate.candidates(ad, 1.0, 999, lambda a: True) == [] and gate.candidates(meh, 1.0, 999, lambda a: True) == []


def test_comments_are_preview_only_until_the_voice_is_tested():
    """DECISIONS G3-G17: comments auto-post only after Ian tailors and tests his voice; until then a preview."""
    prop = cv.Proposal(kind="video", is_ad=False, topics=["gym"], interest=3, text_to_read="great form", quick_skip=False)
    preview = cv.WriteGate(first_write_after_s=0, like_after_fraction=0.0)
    assert preview.candidates(prop, 1.0, 999, lambda a: True) == ["like", "comment_preview"]
    assert preview.candidates(prop, 1.0, 999, lambda a: a != "comment") == ["like", "comment_preview"]  # the pacer doesn't gate a preview
    auto = cv.WriteGate(first_write_after_s=0, like_after_fraction=0.0, comment_write="auto")
    assert auto.candidates(prop, 1.0, 999, lambda a: True) == ["like", "comment"]


def test_daemon_understanding_client_maps_the_tools_answer():
    from phonebridge.director.content_vlm import daemon_understanding_client, understanding_chain

    class Phone:
        async def _offered(self, tool):
            return tool == "understand"
        async def understand(self, device, prompt=None, max_tokens=None):
            return {"summary": "GothamChess posts a Magnus Carlsen clip", "interest": 1, "parsed": True, "raw": "{}", "ms": 1900.0, "model": "qwen3.5-4b",
                    "kind": "video", "is_ad": False, "topics": ["chess"], "frame": {"seq": 42, "age_ms": 30}}
    client = daemon_understanding_client(Phone(), "iphone12")
    out = asyncio.run(client(b"png", "prompt", "low"))
    assert out["summary"].startswith("GothamChess") and out["topics"] == ["chess"] and out["kind"] == "video" and cv.validate(out).summary
    c, name = asyncio.run(understanding_chain(Phone(), "iphone12"))
    assert name.startswith("daemon:understand")


def test_budget_and_escalation():
    from phonebridge.director.content_vlm import Budget, escalation_client
    t = [0.0]
    b = Budget(2, clock=lambda: t[0])
    assert b.allow() and not b.allow()  # spread evenly: 2/min = one per 30 s
    t[0] = 31.0
    assert b.allow() and not b.allow()
    t[0] = 45.0
    assert not b.allow()  # two in the last minute
    t[0] = 61.0
    assert b.allow()

    async def local(png, prompt, effort):
        return {"kind": "video", "is_ad": False, "topics": [], "interest": 1, "text_to_read": "", "quick_skip": False, "summary": ""}
    calls = []
    async def strong(png, prompt, effort):
        calls.append(1)
        return {"kind": "video", "is_ad": False, "topics": ["chess"], "interest": 1, "text_to_read": "", "quick_skip": False, "summary": "GothamChess posts a chess clip"}
    c = escalation_client(local, strong)
    out = asyncio.run(c(b"png", "p", "low"))
    assert out["summary"].startswith("GothamChess") and out["_escalated"] and len(calls) == 1
    async def good(png, prompt, effort):
        return {"kind": "video", "is_ad": False, "topics": ["chess"], "interest": 1, "text_to_read": "", "quick_skip": False, "summary": "x posts y"}
    assert "_escalated" not in asyncio.run(escalation_client(good, strong)(b"png", "p", "low")) and len(calls) == 1


def test_warm_claude_restarts_its_conversation_every_max_turns(monkeypatch):
    """One stream-json process is one growing conversation: after max_turns answers the next call starts a fresh one."""
    import json as _json

    from phonebridge.director import content_vlm as cv

    starts = []

    class Proc:
        def __init__(self):
            self.lines = []
            self.stdin = self
            self.stdout = self

        def write(self, s):
            self.lines.append(_json.dumps({"type": "result", "result": '{"kind": "photo"}'}) + "\n")

        def flush(self):
            pass

        def readline(self):
            return self.lines.pop(0)

        def poll(self):
            return None

        def close(self):
            pass

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    w = cv.WarmClaude(max_turns=3)
    monkeypatch.setattr(w, "_start", lambda: (starts.append(1), setattr(w, "proc", Proc())))
    for _ in range(7):
        assert w._ask(b"png", "p") == {"kind": "photo"}
    assert len(starts) == 3 and w.calls == 7  # turns 1-3, 4-6, 7
