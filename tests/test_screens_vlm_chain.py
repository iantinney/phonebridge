"""The slow model on Ian's Claude plan (DECISIONS "Models via his plans", 04:3xZ 10-19): the director's warm `claude`
process (Haiku, Sonnet on escalation) answers; Bedrock only when the plan reports a usage limit; no API key path."""
import io
import json

import pytest
from PIL import Image

from phonebridge.director import content_vlm as cv
from phonebridge.screens import vlm as V

ANSWER = {"screen_name": "comments_sheet", "kind": "sheet", "action": "back", "element_id": -1, "target_box": [0, 0, 0, 0],
          "label_text": "", "confidence": 0.8, "reason": "a comments sheet"}


def png(color="gray") -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (1170, 2532), color).save(b, "PNG")
    return b.getvalue()


class FakeWarm:
    """Stands in for content_vlm.WarmClaude: records how it was built and what it was asked."""
    built: list = []
    fail: Exception | None = None

    def __init__(self, model="haiku", timeout=60.0, system=None, schema=None):
        self.model, self.system, self.schema, self.asked = model, system, schema, []
        FakeWarm.built.append(self)

    def _ask(self, png_bytes: bytes, text: str) -> dict:
        self.asked.append((png_bytes, text))
        if FakeWarm.fail is not None:
            raise FakeWarm.fail
        return dict(ANSWER)


@pytest.fixture
def warm(monkeypatch):
    FakeWarm.built, FakeWarm.fail = [], None
    monkeypatch.setenv("PHONEBRIDGE_VLM", "claude-cli")
    monkeypatch.setattr(cv, "WarmClaude", FakeWarm)
    V._warm_pool.clear()
    bedrock = []
    monkeypatch.setattr(V, "bedrock_transport", lambda payload, timeout=60.0: bedrock.append(payload) or {
        "content": [{"type": "text", "text": json.dumps(ANSWER)}], "usage": {"input_tokens": 900, "output_tokens": 60},
        "cost_usd": 0.0012, "provider": "bedrock"})
    yield bedrock
    V._warm_pool.clear()


def test_slow_model_is_the_warm_claude_process_on_the_plan(warm, tmp_path):
    vlm = V.VLM(state_dir=tmp_path)
    ans = vlm.resolve(png(), [{"text": "Add comment...", "x": 100, "y": 2300, "w": 300, "h": 40}], ["tiktok.feed"], "tiktok")
    assert ans["screen_name"] == "comments_sheet" and not warm
    (w,) = FakeWarm.built
    assert w.model == "haiku" and w.system == V.SYSTEM and w.schema == V.RESOLVE_SCHEMA
    sent, text = w.asked[0]
    assert Image.open(io.BytesIO(sent)).size == (585, 1266) and "[1] 'Add comment...'" in text
    log = [json.loads(l) for l in (tmp_path / "vlm.jsonl").read_text().splitlines()]
    assert log[-1]["provider"] == "claude-plan" and log[-1]["cost_usd"] == 0  # the plan is not metered spend


def test_plan_usage_limit_falls_back_to_bedrock(warm, tmp_path):
    FakeWarm.fail = RuntimeError("claude: You've hit your limit · resets 9am (UTC)")
    ans = V.VLM(state_dir=tmp_path).resolve(png("white"), [], [], "x")
    assert ans["screen_name"] == "comments_sheet" and len(warm) == 1 and warm[0]["model"] == "claude-haiku-5-5"
    log = [json.loads(l) for l in (tmp_path / "vlm.jsonl").read_text().splitlines()]
    assert any("plan limit" in (r.get("note") or "") for r in log) and log[-1]["provider"] == "bedrock" and log[-1]["cost_usd"] > 0


def test_other_plan_failures_do_not_reach_bedrock(warm, tmp_path):
    FakeWarm.fail = RuntimeError("claude stream closed")
    with pytest.raises(V.VLMError, match="stream closed"):
        V.VLM(state_dir=tmp_path).resolve(png("black"), [], [], "x")
    assert not warm


def test_escalation_reuses_one_warm_sonnet_process(warm, tmp_path):
    vlm = V.VLM(state_dir=tmp_path, cache=False)
    for color in ("red", "blue"):
        vlm.resolve(png(color), [], [], "instagram", escalate=True)
    assert [w.model for w in FakeWarm.built] == ["sonnet"] and len(FakeWarm.built[0].asked) == 2


def test_no_api_key_path_left():
    assert not any(hasattr(V, n) for n in ("find_key", "http_transport", "endpoint", "cli_transport", "OPENROUTER"))


class FakeProc:
    def __init__(self, argv, **_):
        self.argv, self.lines = argv, []
        FakeProc.last = self
        self.stdin = self

    def write(self, s):
        msg = json.loads(s)
        assert msg["message"]["content"][1]["type"] == "image"
        self.lines += [json.dumps({"type": "system"}) + "\n", json.dumps(FakeProc.result) + "\n"]

    def flush(self):
        pass

    @property
    def stdout(self):
        return self

    def readline(self):
        return self.lines.pop(0) if self.lines else ""

    def poll(self):
        return None


def test_warm_claude_takes_a_system_prompt_and_schema_and_surfaces_errors(monkeypatch):
    """The director's client, extended (not copied): --system-prompt / --json-schema when given; the structured answer;
    an error result (a plan limit) raises with its text instead of a JSON parse error."""
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    w = cv.WarmClaude("haiku", system="sys", schema={"type": "object"})
    FakeProc.result = {"type": "result", "is_error": False, "result": "ok", "structured_output": {"a": 1}}
    assert w._ask(b"png", "q") == {"a": 1}
    argv = FakeProc.last.argv
    assert argv[argv.index("--system-prompt") + 1] == "sys" and json.loads(argv[argv.index("--json-schema") + 1]) == {"type": "object"}
    FakeProc.result = {"type": "result", "is_error": True, "result": "You've hit your limit · resets 9am"}
    with pytest.raises(RuntimeError, match="hit your limit"):
        w._ask(b"png", "q")
    plain = cv.WarmClaude("haiku")
    FakeProc.result = {"type": "result", "is_error": False, "result": 'x {"b": 2} y'}
    assert plain._ask(b"png", "q") == {"b": 2} and "--json-schema" not in FakeProc.last.argv


def test_no_model_unless_one_is_chosen(monkeypatch, tmp_path):
    """Off by default (no _deployment default, no PHONEBRIDGE_VLM): no screenshot leaves the Mac."""
    monkeypatch.delenv("PHONEBRIDGE_VLM", raising=False)
    monkeypatch.setattr(V, "site", lambda name, default: default)
    monkeypatch.setattr(V, "plan_transport", lambda *a, **k: pytest.fail("the claude CLI was called"))
    monkeypatch.setattr(V, "bedrock_transport", lambda *a, **k: pytest.fail("Bedrock was called"))
    with pytest.raises(V.VLMError, match="PHONEBRIDGE_VLM"):
        V.VLM(state_dir=tmp_path, cache=False).resolve(png("white"), [], [], "x")
