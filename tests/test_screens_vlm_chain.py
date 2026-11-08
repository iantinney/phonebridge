"""The slow model on Ian's Claude plan (DECISIONS "Models via his plans", 04:3xZ 10-19): a `claude` process of its own per
ask (Haiku, Sonnet on escalation) answers; Bedrock only when the plan reports a usage limit; no API key path."""
import base64
from concurrent.futures import ThreadPoolExecutor, wait
import io
import json
import signal
import subprocess
import sys
import threading

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
        self.model, self.system, self.schema, self.asked, self.closed = model, system, schema, [], False
        FakeWarm.built.append(self)

    def _ask(self, png_bytes: bytes, text: str) -> dict:
        self.asked.append((png_bytes, text))
        if FakeWarm.fail is not None:
            raise FakeWarm.fail
        return dict(ANSWER)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def warm(monkeypatch):
    FakeWarm.built, FakeWarm.fail = [], None
    monkeypatch.setenv("PHONEBRIDGE_VLM", "claude-cli")
    monkeypatch.setattr(cv, "WarmClaude", FakeWarm)
    bedrock = []
    monkeypatch.setattr(V, "bedrock_transport", lambda payload, timeout=60.0: bedrock.append(payload) or {
        "content": [{"type": "text", "text": json.dumps(ANSWER)}], "usage": {"input_tokens": 900, "output_tokens": 60},
        "cost_usd": 0.0012, "provider": "bedrock"})
    return bedrock


def test_slow_model_is_one_claude_process_per_ask_on_the_plan(warm, tmp_path):
    vlm = V.VLM(state_dir=tmp_path)
    ans = vlm.resolve(png(), [{"text": "Add comment...", "x": 100, "y": 2300, "w": 300, "h": 40}], ["tiktok.feed"], "tiktok")
    assert ans["screen_name"] == "comments_sheet" and not warm
    (w,) = FakeWarm.built
    assert w.model == "haiku" and w.system == V.SYSTEM and w.schema == V.RESOLVE_SCHEMA and w.closed
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
    assert not warm and [w.closed for w in FakeWarm.built] == [True]  # a failed ask closes its process too


def test_escalations_share_no_process(warm, tmp_path):
    vlm = V.VLM(state_dir=tmp_path, cache=False)
    for color in ("red", "blue"):
        vlm.resolve(png(color), [], [], "instagram", escalate=True)
    assert [(w.model, len(w.asked), w.closed) for w in FakeWarm.built] == [("sonnet", 1, True)] * 2


def test_a_cached_screen_starts_no_process_and_the_plan_stays_unmetered(warm, tmp_path):
    vlm = V.VLM(state_dir=tmp_path)
    first = vlm.resolve(png("red"), [], [], "x")
    assert vlm.resolve(png("red"), [], [], "x") == first and len(FakeWarm.built) == 1
    vlm.resolve(png("blue"), [], [], "x")
    assert len(FakeWarm.built) == 2 and vlm.spend_today() == 0 and vlm.last_cost == 0
    log = [json.loads(l) for l in (tmp_path / "vlm.jsonl").read_text().splitlines()]
    assert [r["cached"] for r in log] == [False, True, False] and all(r["cost_usd"] == 0 for r in log)


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


class GatedProc:
    """A stream-json process that answers the newest question once `gate` opens: with two writers in flight, the first
    reader takes the second's answer."""

    def __init__(self, gate):
        self.gate, self.asked, self.wrote, self.dead, self.terminated = gate, [], threading.Semaphore(0), False, False
        self.stdin = self.stdout = self

    def write(self, s):
        self.asked.append(json.loads(s)["message"]["content"][0]["text"])
        self.wrote.release()

    def flush(self):
        pass

    def readline(self):
        assert self.gate.wait(5)
        return json.dumps({"type": "result", "result": "x", "structured_output": {"answer": self.asked.pop()}}) + "\n"

    def poll(self):
        return 1 if self.dead else None

    def close(self):
        pass

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def gated() -> tuple[cv.WarmClaude, GatedProc, threading.Event]:
    gate = threading.Event()
    w = cv.WarmClaude(timeout=30)
    w.proc = GatedProc(gate)
    return w, w.proc, gate


def test_concurrent_asks_cannot_cross_answers():
    w, proc, gate = gated()
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(w._ask, b"", "phone1")
        assert proc.wrote.acquire(timeout=2)
        second = pool.submit(w._ask, b"", "phone2")
        assert not proc.wrote.acquire(timeout=0.2)  # phone2's question waits for phone1's answer
        gate.set()
    assert first.result(2) == {"answer": "phone1"} and second.result(2) == {"answer": "phone2"}


def test_close_waits_for_the_turn_in_progress():
    w, proc, gate = gated()
    with ThreadPoolExecutor(2) as pool:
        asked = pool.submit(w._ask, b"", "phone1")
        assert proc.wrote.acquire(timeout=2)
        closing = pool.submit(w.close)
        wait([closing], timeout=0.2)
        assert not closing.done() and not proc.terminated
        gate.set()
    assert asked.result(2) == {"answer": "phone1"} and proc.terminated and w.proc is None


def test_restart_waits_for_the_turn_in_progress(monkeypatch):
    w, proc, gate = gated()
    procs = [proc]
    monkeypatch.setattr(w, "_start", lambda: (procs.append(GatedProc(gate)), setattr(w, "proc", procs[-1])))
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(w._ask, b"", "phone1")
        assert proc.wrote.acquire(timeout=2)
        proc.dead = True  # the process is seen dying while its turn is still being read
        second = pool.submit(w._ask, b"", "phone2")
        wait([second], timeout=0.2)
        assert procs == [proc]  # the restart waits: replacing the process now would hand phone1 the new process's stream
        gate.set()
    assert first.result(2) == {"answer": "phone1"} and second.result(2) == {"answer": "phone2"} and len(procs) == 2


REMEMBERING_CLI = """#!{python}
import json, sys
heard = []
for line in sys.stdin:
    heard.append(json.loads(line)["message"]["content"][0]["text"])
    print(json.dumps({{"type": "result", "result": "x", "structured_output": {{"heard": heard}}}}), flush=True)
"""
IGNORES_TERM_CLI = """#!{python}
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("ready", flush=True)
time.sleep(60)
"""


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Real child processes that speak the CLI's stream-json and remember their conversation: no claude, no network."""
    def install(source=REMEMBERING_CLI):
        script = tmp_path / "claude"
        script.write_text(source.format(python=sys.executable))
        script.chmod(0o755)
        monkeypatch.setattr(cv, "claude_bin", lambda: str(script))
    install()
    monkeypatch.setitem(V._remote, "on", False)
    procs = []
    real = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: procs.append(real(*a, **k)) or procs[-1])
    return procs, install


def heard_by_transport(text: str) -> dict:
    frame = {"type": "image", "source": {"data": base64.b64encode(text.encode()).decode()}}
    payload = {"model": "haiku", "system": "s", "messages": [{"role": "user", "content": [frame, {"type": "text", "text": text}]}]}
    return json.loads(V.plan_transport(payload, timeout=20)["content"][0]["text"])


def test_unrelated_screen_asks_share_no_history_and_leave_no_process(cli):
    procs, _ = cli
    assert [heard_by_transport(f"screen{i}") for i in range(3)] == [{"heard": [f"screen{i}"]} for i in range(3)]
    assert len(procs) == 3 and all(p.returncode is not None for p in procs)


def test_seven_phones_asking_at_once_hear_only_themselves(cli):
    procs, _ = cli
    phones = [f"phone{i}" for i in range(1, 8)]
    with ThreadPoolExecutor(7) as pool:
        assert list(pool.map(heard_by_transport, phones)) == [{"heard": [p]} for p in phones]
    assert len(procs) == 7 and all(p.returncode is not None for p in procs)


def test_close_kills_a_process_that_ignores_it(cli, monkeypatch):
    procs, install = cli
    install(IGNORES_TERM_CLI)
    monkeypatch.setattr(cv, "CLOSE_WAIT_S", 0.2)
    w = cv.WarmClaude()
    w._start()
    assert w.proc.stdout.readline() == "ready\n"
    w.close()
    assert procs[0].returncode == -signal.SIGKILL


def test_no_model_unless_one_is_chosen(monkeypatch, tmp_path):
    """Off by default (no _deployment default, no PHONEBRIDGE_VLM): no screenshot leaves the Mac."""
    monkeypatch.delenv("PHONEBRIDGE_VLM", raising=False)
    monkeypatch.setattr(V, "site", lambda name, default: default)
    monkeypatch.setattr(V, "plan_transport", lambda *a, **k: pytest.fail("the claude CLI was called"))
    monkeypatch.setattr(V, "bedrock_transport", lambda *a, **k: pytest.fail("Bedrock was called"))
    with pytest.raises(V.VLMError, match="PHONEBRIDGE_VLM"):
        V.VLM(state_dir=tmp_path, cache=False).resolve(png("white"), [], [], "x")
