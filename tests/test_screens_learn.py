"""SCR-06 (Q3): unknown screens are solved by the VLM, saved as candidates, guarded, promoted after 3 matches."""
import asyncio
import json

import pytest

from phonebridge.backends.replay import Fault, ReplayPhone, write_sequence
from phonebridge.screens.cascade import Cascade
from phonebridge.screens.engine import Engine, ListJournal, Stop
from phonebridge.screens.learn import Learner, GuardError, guard
from phonebridge.screens.schema import Record
from phonebridge.screens.store import Store
from phonebridge.screens.vlm import VLM
from screens_fixtures.synth import frame as synth_frame, PALETTE
from test_screens_engine import FRAMES, HOME, PICKER, FINAL, PASTE, IG_REEL, item


def fake_vlm(answers):
    """A VLM whose transport replays scripted answers; every call is still logged and cached."""
    calls = []

    def transport(payload):
        calls.append(payload)
        a = answers.pop(0)
        return {"content": [{"type": "text", "text": json.dumps(a)}],
                "usage": {"input_tokens": 1000, "output_tokens": 50}}
    return calls, transport


@pytest.fixture
def world(tmp_path):
    d = tmp_path / "sequences" / "ig_reel" / "1"
    write_sequence(d, [(n, synth_frame(top=PALETTE[i], seed=i), ocr, {"screen": s, "action": a})
                       for i, (n, s, a, ocr) in enumerate(FRAMES)])
    extra = tmp_path / "extra"
    write_sequence(extra, [("wall", synth_frame(top="gray", seed=9),
                            [item("Enable library access", 585, 1200, 400), item("Not now", 585, 1700, 120),
                             item("Settings", 585, 1800, 120)], {"screen": None})])
    store = Store(records=[Record.from_dict(r) for r in (HOME, PICKER, FINAL, PASTE)], corpus=tmp_path / "corpus")
    return d, extra, store


def make(store, phone, vlm, state):
    learner = Learner(vlm=vlm, store=store)
    return Engine(store=store, cascade=Cascade(store), phone=phone, eyes=phone.ocr, journal=ListJournal(),
                  learner=learner, sleep=lambda s: asyncio.sleep(0))


WALL_ANSWER = {"screen_name": "library_access_wall", "kind": "popup", "action": "tap_named", "label_text": "Not now",
               "element_id": -1, "confidence": 0.9, "target_box": [0.4, 0.66, 0.6, 0.69], "reason": "a permissions upsell; Not now dismisses it"}


def test_unknown_resolved_by_vlm_saved_and_fast_next_time(world, tmp_path):
    d, extra, store = world
    calls, transport = fake_vlm([dict(WALL_ANSWER)])
    vlm = VLM(transport=transport, state_dir=tmp_path / "state")
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "wall.png", answer="Not now")])
    out = asyncio.run(make(store, phone, vlm, tmp_path).run(IG_REEL, rehearse=True))
    assert out.status == "rehearsed" and len(calls) == 1
    cand = [r for r in store.records.values() if r.source == "vlm"]
    assert len(cand) == 1 and cand[0].id == "instagram.popup.library_access_wall" and cand[0].answer and cand[0].successes == 1
    assert cand[0].evidence and (store.corpus / cand[0].evidence[0]).exists()
    logged = [json.loads(l) for l in (tmp_path / "state" / "vlm.jsonl").read_text().splitlines()]
    assert logged and {"frame", "prompt_hash", "answer"} <= set(logged[0])

    # the same wall a second time: the candidate record matches, no VLM call
    phone2 = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "wall.png", answer="Not now")])
    out2 = asyncio.run(make(store, phone2, vlm, tmp_path).run(IG_REEL, rehearse=True))
    assert out2.status == "rehearsed" and len(calls) == 1


def test_guard_is_schema_sanity_only():
    """Caution is prompt guidance (Ian: no code-level never-list); only a malformed action is unusable."""
    from phonebridge.screens.vlm import SYSTEM
    guard({**WALL_ANSWER, "label_text": "Post"})  # the task's approval gates publishing, not a word list
    with pytest.raises(GuardError):
        guard({**WALL_ANSWER, "action": "type"})
    assert "Never choose Post, Share" in SYSTEM  # the guidance lives in the prompt


def test_publish_tap_requires_approved_task(world, tmp_path):
    """The irreversible Share tap happens only inside an approved task (`ops approved`), never in rehearse."""
    d, _, store = world
    phone = ReplayPhone(d)
    out = asyncio.run(make(store, phone, None, tmp_path).run(IG_REEL))
    assert out.status == "stopped" and "not approved" in out.reason
    assert not any(t.get("target") == "share" for t in phone.log if t["op"] == "tap")
    phone2 = ReplayPhone(d)
    out2 = asyncio.run(make(store, phone2, None, tmp_path).run(IG_REEL.__class__(**{**IG_REEL.__dict__, "approved": True})))
    assert out2.status == "done" and sum(1 for t in phone2.log if t["op"] == "tap" and t.get("target") == "share") == 1


def test_counters_update_on_success_and_failure(world, tmp_path):
    d, extra, store = world
    calls, transport = fake_vlm([dict(WALL_ANSWER)])
    vlm = VLM(transport=transport, state_dir=tmp_path / "state")
    for i in range(4):
        phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "wall.png", answer="Not now")])
        asyncio.run(make(store, phone, vlm, tmp_path).run(IG_REEL, rehearse=True))
        rec = store.records["instagram.popup.library_access_wall"]
        assert rec.successes == i + 1 and rec.failures == 0, i  # the slow model's own success, then one per run
    assert len(calls) == 1
    store.failure(rec.id, "a run where the popup stayed")
    assert store.records[rec.id].failures == 1 and store.records[rec.id].successes == 4


def test_failing_response_returns_to_slow_model(world, tmp_path):
    """A response that keeps failing goes back to the slow model (the stronger one from the second try); the
    operator only after three misses on one screen."""
    d, extra, store = world
    bad = {**WALL_ANSWER, "label_text": "Settings", "target_box": [0.4, 0.70, 0.6, 0.73]}
    calls, transport = fake_vlm([dict(bad), dict(bad), dict(bad)])
    vlm = VLM(transport=transport, state_dir=tmp_path / "state")
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "wall.png", answer="Not now")])
    out = asyncio.run(make(store, phone, vlm, tmp_path).run(IG_REEL, rehearse=True))
    assert out.status == "stopped" and "operator" in out.reason and len(calls) == 3
    assert [c["model"] for c in calls] == ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-5-5"]
    rec = store.records["instagram.popup.library_access_wall"]
    assert rec.failures >= 3 and rec.successes == 0


def test_stop_kinds_park_without_a_tap(world, tmp_path):
    d, extra, store = world
    calls, transport = fake_vlm([{**WALL_ANSWER, "kind": "captcha", "action": "none", "label_text": ""}])
    vlm = VLM(transport=transport, state_dir=tmp_path / "state")
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "wall.png", answer="Not now")])
    out = asyncio.run(make(store, phone, vlm, tmp_path).run(IG_REEL, rehearse=True))
    assert out.status == "stopped" and "captcha" in out.reason
    assert not any(t["matched"] and t["fault"] for t in phone.log if t["op"] == "tap") and phone.pending is not None


def test_vlm_cap_and_cache(tmp_path):
    calls, transport = fake_vlm([dict(WALL_ANSWER), dict(WALL_ANSWER)])
    alerts = []
    vlm = VLM(transport=transport, state_dir=tmp_path / "state", cap_usd=0.00014, notify=alerts.append)
    png = synth_frame(top="gray", seed=9)
    a = vlm.resolve(png, [], known=["instagram.home"], platform="instagram")
    assert a["screen_name"] == "library_access_wall" and len(calls) == 1
    assert vlm.resolve(png, [], known=["instagram.home"], platform="instagram") == a and len(calls) == 1  # cached
    assert alerts and "80" in alerts[0]
    from phonebridge.screens.vlm import CapReached
    with pytest.raises(CapReached):
        vlm.resolve(synth_frame(top="pink", seed=1), [], known=[], platform="instagram")
    assert len(calls) == 1


def test_element_grounding_and_escalation(world, tmp_path):
    """Haiku answers by element number; when its action fails the next-screen check the retry goes to Sonnet."""
    d, extra, store = world
    models = []

    def transport(payload):
        models.append(payload["model"])
        # 1st: Haiku points at element [3] (Settings) — wrong; 2nd: Sonnet points at element [2] (Not now)
        a = {**WALL_ANSWER, "label_text": "", "element_id": 3 if len(models) == 1 else 2}
        return {"content": [{"type": "text", "text": json.dumps(a)}], "usage": {"input_tokens": 1000, "output_tokens": 50}}
    vlm = VLM(transport=transport, state_dir=tmp_path / "state")
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "wall.png", answer="Not now")])
    out = asyncio.run(make(store, phone, vlm, tmp_path).run(IG_REEL, rehearse=True))
    assert out.status == "rehearsed" and models == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    taps = [t for t in phone.log if t["op"] == "tap" and t["fault"]]
    assert taps[-1]["matched"] and taps[-1]["target"] == "Not now"


def test_learned_screen_stands_in_for_the_flow_placeholder(world, tmp_path):
    """The flow names `instagram.search` before any record exists; the slow model names the screen differently.
    The learned record stands in for the placeholder and the flow goes on from it."""
    from phonebridge.screens.engine import Flow, Step
    d, extra, store = world
    answer = {**WALL_ANSWER, "screen_name": "explore_grid", "kind": "screen", "action": "none", "label_text": "", "confidence": 0.8}
    calls, transport = fake_vlm([dict(answer)])
    vlm = VLM(transport=transport, state_dir=tmp_path / "state")
    flow = Flow(name="t", platform="instagram", bundle="com.burbn.instagram", final="instagram.new_post.picker", steps=(
        Step(at="instagram.home", do="tap", target="create", next=("instagram.search",)),  # placeholder id, no record
        Step(at="instagram.search", do="tap", target="not_now", args={"ocr": "Not now"}, next=("instagram.new_post.picker",))))
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "wall.png", answer="Not now")])
    out = asyncio.run(make(store, phone, vlm, tmp_path).run(flow, rehearse=True))
    assert out.status == "done", (out.reason, [e["event"] for e in out.journal.events])
    assert len(calls) == 1 and "instagram.search" in store.records  # named with the flow's own id (one dangling hint)


def test_learned_screen_stands_in_at_the_expectation_check(world, tmp_path):
    """Live 23:27Z: a record learned in an earlier run is recognised at once on a fresh run; the step that expected
    the placeholder id must count it as the expected next screen, and the next step runs from it."""
    from phonebridge.screens.engine import Flow, Step
    d, extra, store = world
    learned = Record.from_dict({"id": "instagram.screen.explore_grid", "platform": "instagram", "bundle": "com.burbn.instagram",
                                "match": {"ocr_any": ["Enable library access", "Not now"], "at_least": 2}, "source": "vlm", "successes": 1,
                                "appearance": "any", "targets": {"not_now": {"ocr": "Not now"}}})
    store.records[learned.id] = learned
    flow = Flow(name="t", platform="instagram", bundle="com.burbn.instagram", final="instagram.new_post.picker", steps=(
        Step(at="instagram.home", do="tap", target="create", next=("instagram.search",)),  # placeholder, never recorded
        Step(at="instagram.search", do="tap", target="not_now", next=("instagram.new_post.picker",))))
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "wall.png", answer="Not now")])
    out = asyncio.run(make(store, phone, None, tmp_path).run(flow, rehearse=True))
    assert out.status == "done", (out.reason, [e["event"] for e in out.journal.events])
    assert any(e["event"] == "alias" for e in out.journal.events) and not any(e["event"] == "vlm" for e in out.journal.events)


def test_bedrock_payload_and_budget():
    """Bedrock: logical models map to the account's models, 4.x gets no effort, the per-minute budget waits."""
    from phonebridge.screens.vlm import bedrock_payload, bedrock_budget, extract_json, _bedrock_calls
    body = bedrock_payload({"model": "claude-haiku-5-5", "max_tokens": 10, "system": "s", "messages": [],
                            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}})
    assert body["model"].startswith("us.anthropic.claude-haiku-4-5") and "output_config" not in body  # 4.x ignores it
    assert body["system"].startswith("s\n\nAnswer with one JSON object only") and '{"type":"object"}' in body["system"]
    assert bedrock_payload({"model": "claude-sonnet-5-5", "max_tokens": 1, "messages": []})["model"] == "us.anthropic.claude-sonnet-4-6"
    _bedrock_calls.clear()
    assert all(bedrock_budget("m", now=1000.0 + i) == 0.0 for i in range(10))
    assert 49 < bedrock_budget("m", now=1010.0) <= 50.1  # the 11th call inside the minute waits for the first to age out
    assert extract_json('Sure: {"a": {"b": 1}} end')["a"]["b"] == 1



def test_bedrock_sends_the_schema_in_the_prompt_and_the_image_as_jpeg_585(monkeypatch):
    """controller-verify's bake-off: the forced tool cost +49 % latency for the same answers; JPEG q80 at 585 px saves
    ~25 %. The API path keeps structured output and PNG; the CLI path is untouched; the order stays api → bedrock → cli."""
    pytest.importorskip("anthropic")  # the vlm extra
    import base64, io
    from PIL import Image
    from phonebridge.screens import vlm
    from phonebridge.screens.vlm import bedrock_payload, bedrock_transport
    png = io.BytesIO(); Image.new("RGB", (1170, 2532), "navy").save(png, "PNG")
    payload = {"model": "claude-haiku-5-5", "max_tokens": 64, "system": "You name screens.",
               "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object", "properties": {"screen": {"type": "string"}}}}},
               "messages": [{"role": "user", "content": [
                   {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png.getvalue()).decode()}},
                   {"type": "text", "text": "Which screen?"}]}]}
    body = bedrock_payload(payload)
    img = body["messages"][0]["content"][0]["source"]
    im = Image.open(io.BytesIO(base64.b64decode(img["data"])))
    assert img["media_type"] == "image/jpeg" and im.format == "JPEG" and im.width == 585 and im.height <= 1266
    assert "tools" not in body and "output_config" not in body and '"screen"' in body["system"]
    assert payload["messages"][0]["content"][0]["source"]["media_type"] == "image/png"  # the caller's payload untouched

    calls = []

    class Resp:
        def model_dump(self):
            return {"content": [{"type": "text", "text": 'Here: {"screen": "instagram.home"}'}], "usage": {"input_tokens": 1, "output_tokens": 1}}

    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                calls.append(kw)
                return Resp()
    monkeypatch.setattr(vlm, "_bedrock_client", lambda anthropic, timeout: Client())
    monkeypatch.setattr(vlm, "bedrock_budget", lambda model: 0.0)
    d = bedrock_transport(payload)
    assert "tools" not in calls[0] and "tool_choice" not in calls[0] and calls[0]["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"
    assert vlm.extract_json(d["content"][0]["text"]) == {"screen": "instagram.home"}
