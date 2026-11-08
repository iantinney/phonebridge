"""Teaching (L1) on a labelled FIXTURE: tests/fixtures/teach_demo.py renders a synthetic screen recording of a fictional
app (not a human demonstration, not a real app). The analysis sees only the encoded mp4 and the narration; the script
that drew it is the ground truth these tests compare against."""

import hashlib
import re
import importlib.util
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from teachable_worker import agency, teaching
from teachable_worker.store import query

spec = importlib.util.spec_from_file_location("teach_demo", Path(__file__).parent / "fixtures" / "teach_demo.py")
teach_demo = importlib.util.module_from_spec(spec)
sys.modules["teach_demo"] = teach_demo
spec.loader.exec_module(teach_demo)

OCR = teaching.OCR_PYTHON.exists() or shutil.which("tesseract") or os.environ.get("TW_OCR_CMD")
video_ok = pytest.mark.skipif(not (teach_demo.available() and OCR), reason="needs ffmpeg, DejaVu fonts and an OCR engine")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if not (teach_demo.available() and OCR):
        pytest.skip("needs ffmpeg, DejaVu fonts and an OCR engine")
    d = tmp_path_factory.mktemp("teach_demo")
    r = teach_demo.render(d)
    r["new_clip"] = teach_demo.clip(d / "bakery.mp4", 22)
    return r


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def org_with(handle="acme_studio", platform="studio"):
    org = agency.create_org("Fixture agency", "agency", "test")
    owner = agency.add_person(org["id"], "Fixture owner", role="owner", cohort="test")
    acct = agency.add_account(org["id"], platform, handle, owner["id"])
    return org, owner, acct


def live(steps):
    return [(s["op"], s.get("label")) for s in steps if s["op"] != "scroll"]


@video_ok
def test_recording_becomes_a_routine_that_plans_a_fresh_job_and_keeps_corrections(demo, monkeypatch):
    monkeypatch.setenv("TW_TEACH_MODEL", "off")  # the measured layer alone; the model route is tested on its own
    org, owner, acme = org_with()
    original = sha(demo["video"])
    rec = teaching.ingest_recording(org["id"], demo["video"], demo["narration"], owner["id"], acme["id"], label="fixture")
    r = teaching.derive_routine(rec, "Client post")

    # what the recording shows, in order, with evidence; nothing was scripted into the analyser
    assert r["provenance"] == "operator_demo"  # a fixture never counts as a customer demonstration
    steps = json.loads(r["steps_json"])
    assert live(steps) == [("tap", "Create"), ("select_media", None), ("tap", "Next"), ("tap", "Next"),
                           ("focus", "Write a caption..."), ("type", None), ("dismiss_keyboard", "Done"), ("tap", "Save draft")]
    assert sum(s["op"] == "scroll" and s["optional"] for s in steps) == 3
    by = {(s["op"], s["screen"]["title"]): s for s in steps}
    assert "press_highlight" in [p["kind"] for p in by[("tap", "Studio")]["proof"]]
    pick = by[("select_media", "New post")]
    assert pick["pick"] == "newest" and pick["duration"] == "0:15" and pick["param"] == "video"
    typed = by[("type", "Details")]
    assert typed["slot"] == "caption" and typed["param"] == "caption" and "drop is live" in typed["value"]
    unseen = by[("tap", "Edit")]  # the quick tap with no visible highlight: inferred, flagged, asked once
    assert unseen["uncertain"] and unseen["question"] and "doesn't show what you touched" in unseen["question"]
    assert all(not s["uncertain"] for s in steps if s is not unseen)
    doc = teaching.doc_of(r)
    assert {(x["kind"], x["value"]) for x in doc["rules"]} >= {("stop_before", "Share"), ("caption_suffix", "#acme")}
    assert [a["label"] for a in doc["avoid"]] == ["Share"]
    pace = doc["behaviour"]["humanize"]
    assert 600 < pace["think"]["tap"][0] < 1800 and 0.15 < pace["flick_len"][0] < 0.45 and "key_gap" in pace
    assert "Here is the job as I understood it" in r["summary"]

    # a guess cannot be confirmed; the person's answer is a correction, and the question is never asked again
    with pytest.raises(teaching.TeachError, match="Answer this first"):
        teaching.confirm(r["id"], owner["id"])
    n = next(i for i, s in enumerate(steps, 1) if s is unseen)
    r = teaching.correct(r["id"], f"Step {n} (question): yes", owner["id"])
    s = json.loads(r["steps_json"])[n - 1]
    assert (s["label"], s["confidence"], s["question"]) == ("Next", 1.0, None) and r["version"] == 2
    r = teaching.confirm(r["id"], owner["id"])
    assert r["confirmed_at"]

    # a fresh job: another account, a new video, a new caption; nothing of the demonstration's content is reused
    bakery = agency.add_account(org["id"], "studio", "bright_bakery", owner["id"])
    agency.grant(bakery["id"], "approve_each", owner["id"], "test grant")
    media = agency.save_media(org["id"], "bakery.mp4", demo["new_clip"].read_bytes())
    caption = "Fresh croissants every morning #acme"
    job = agency.submit_job(org["id"], bakery["id"], "publish", {"media": [{"sha256": media["sha256"]}], "caption": caption},
                            owner["id"], routine_id=r["id"])
    p = teaching.plan(teaching._routine(r["id"]), job)
    assert p["status"] == "needs_input" and "Share" in p["needs"]  # publishing was never demonstrated: ask once
    r = teaching.correct(r["id"], "To publish, tap Share.", owner["id"])
    p = teaching.plan(r, job)
    assert p["status"] == "ready", p["needs_all"]
    assert (p["v"], p["caption"], p["media"]["sha256"]) == (1, caption, media["sha256"])
    assert p["checks"][0]["handle"] == "@bright_bakery"
    assert "New drop" not in json.dumps(p) and "acme_studio" not in json.dumps(p["steps"])
    pick = next(x for x in p["steps"] if x["op"] == "select_media")
    assert pick["target"] == {"slot": "media", "pick": "newest", "sha256": media["sha256"], "duration": "0:22"}
    last = [x for x in p["steps"] if x["op"] == "tap"][-1]
    assert (last["target"]["label"], last["replaced"], last["requires_approved_sha"]) == ("Share", "Save draft", job["content_sha"])
    assert p["approval"] == {"required": True, "content_sha": job["content_sha"]}
    assert all("x" not in (x.get("target") or {}) and "y" not in (x.get("target") or {}) for x in p["steps"])
    assert all(x.get("pause_s", {}).get("median", 1) > 0 for x in p["steps"])
    # the phone lane's taught flow (phonebridge run/taught.py): the way in, ending where the protocol taps the final button
    dos = [st["do"] for st in p["flow"]["states"]]
    assert dos[:3] == [{"tap": "Create"}, {"pick": "first", "next": "Next", "select": "body"}, {"tap": "Next"}]
    assert dos[3] == {"caption": ["Write a caption"], "close_at": [0.5, 0.06]} and dos[4] == "done"
    assert p["flow"]["states"][-1]["bottom"] == ["Share"]

    # the routine's caption rule is applied before approval, never silently at plan time
    job2 = agency.submit_job(org["id"], bakery["id"], "publish", {"media": [{"sha256": media["sha256"]}], "caption": "Doors open at 7"},
                             owner["id"], routine_id=r["id"])
    # integration: agency.submit_job applies the rule before hashing, so the customer approves the ruled caption
    assert json.loads(job2["input_json"])["caption"] == "Doors open at 7 #acme"
    assert teaching.plan(r, job2)["status"] == "ready"
    assert teaching.apply_rules(r, {"caption": "Doors open at 7"})["caption"] == "Doors open at 7 #acme"

    # corrections persist: a note rides along with every plan and survives re-deriving from the same recording
    r = teaching.correct(r["id"], "Keep the tone calm; no emojis.", owner["id"])
    assert "Keep the tone calm; no emojis." in teaching.plan(r, job)["remembered"]
    v = r["version"]
    r = teaching.rederive(r["id"])
    doc = teaching.doc_of(r)
    assert r["version"] == v + 1 and all(q["status"] == "answered" for q in doc["questions"])
    assert doc["publish_via"]["label"] == "Share" and doc["notes"][0]["text"] == "Keep the tone calm; no emojis."
    assert json.loads(r["steps_json"])[n - 1]["label"] == "Next"

    # the original is preserved: untouched source, a read-only copy with the same hash
    stored = Path(teaching._recording(rec)["path"])
    assert sha(demo["video"]) == original == sha(stored) and stat.S_IMODE(stored.stat().st_mode) == 0o400
    assert teaching.recording(rec)["preserved"] and teaching.recording(rec)["duration_s"] > 20
    names = [e["name"] for e in query("SELECT name FROM events ORDER BY id")]
    assert {"recording_ingested", "routine_derived", "routine_corrected", "routine_confirmed"} <= set(names)
    ing = query("SELECT data_json FROM events WHERE name = 'recording_ingested'")[0]
    assert json.loads(ing["data_json"])["recording_id"] == rec


@video_ok
def test_teach_routes_upload_derive_correct_delete(demo, http, monkeypatch):
    monkeypatch.setenv("TW_TEACH_MODEL", "claude-plan")  # a customer upload never goes to an external model
    org, owner, acme = org_with()
    tok = owner["token"]
    st, m = http("PUT", "/api/media?name=demo.mp4", raw=demo["video"].read_bytes(), token=tok, ctype="video/mp4")
    assert st == 200
    st, out = http("POST", "/api/teach/recordings", {"sha256": m["sha256"], "narration": demo["narration"], "account_id": acme["id"]}, token=tok)
    assert st == 200
    rec = out["recording_id"]
    st, view = http("POST", f"/api/teach/recordings/{rec}/routine", {"name": "Client post"}, token=tok)
    assert st == 200, view
    assert view["provenance"] == "customer_demo" and view["interpreter"] == "off" and view["recording"]["duration_s"] > 20
    assert all({"text", "at", "evidence", "param", "question", "uncertain"} <= set(s) for s in view["steps"])
    n = next(i for i, s in enumerate(view["steps"], 1) if s["question"])
    st, view = http("POST", f"/api/routines/{view['id']}/corrections", {"text": f"Step {n} (question): Next"}, token=tok)
    assert st == 200 and not any(s["question"] for s in view["steps"])
    st, other = http("POST", "/api/teach/recordings", {"sha256": "0" * 64}, token=tok)
    assert st == 404  # a path is never taken from the client: unknown media is refused
    st, gone = http("DELETE", f"/api/teach/recordings/{rec}", token=tok)
    assert st == 200 and gone == {"deleted": True}
    assert not teaching.recording(rec)["preserved"] and teaching._routine(view["id"])["summary"]


def test_private_recordings_never_reach_an_external_model(monkeypatch):
    monkeypatch.setenv("TW_TEACH_MODEL", "claude-plan")
    assert teaching.interpreter({"label": "customer", "external_model_ok": 0}) is None
    assert teaching.interpreter({"label": "agent", "external_model_ok": 0}) is None
    assert isinstance(teaching.interpreter({"label": "fixture", "external_model_ok": 0}), teaching.ClaudePlan)
    assert isinstance(teaching.interpreter({"label": "customer", "external_model_ok": 1}), teaching.ClaudePlan)
    monkeypatch.setenv("TW_TEACH_MODEL", "local:http://127.0.0.1:8935")
    assert isinstance(teaching.interpreter({"label": "customer", "external_model_ok": 0}), teaching.LocalVLM)


def test_the_model_never_certifies_a_touch_it_could_not_see():
    a = {"items": [{"text": "Back", "fx": 0.1, "fy": 0.08}, {"text": "Next", "fx": 0.85, "fy": 0.08}]}
    guess = {"op": "tap", "label": "Next", "candidates": ["Next", "Back"], "confidence": 0.4, "proof": []}
    agree_unseen = {"action": "tap", "item": 1, "touch_visible": False, "purpose": "go on", "varies": "no", "confidence": 0.95, "reason": "r"}
    out = teaching._fuse(dict(guess, proof=[]), agree_unseen, a, "fake")
    assert out["label"] == "Next" and out["confidence"] < teaching.ACCEPT  # still a question for the person
    pressed = {"op": "tap", "label": "Back", "confidence": 0.85, "proof": [{"kind": "press_highlight"}]}
    out = teaching._fuse(pressed, agree_unseen, a, "fake")
    assert out["confidence"] < teaching.ACCEPT and out["candidates"] == ["Back", "Next"]  # disagreement is asked, not resolved
    seen = dict(agree_unseen, item=0, touch_visible=True)
    out = teaching._fuse({"op": "tap", "label": "Back", "confidence": 0.85, "proof": []}, seen, a, "fake")
    assert out["confidence"] > 0.85 and out["proof"][-1]["kind"] == "model"


def test_corrections_are_read_into_answers_rules_and_edits():
    doc = {"steps": [{"id": "s1", "op": "tap"}, {"id": "s2", "op": "tap"}],
           "questions": [{"id": "q1", "step": "s2", "status": "open", "options": ["Next", "Back", "something else"]}]}
    assert teaching.interpret("Step 2 (question): it was Back", doc) == [{"kind": "answer", "question": "q1", "value": "Back"}]
    assert teaching.interpret("q1: yes", doc)[0]["value"] == "Next"
    assert teaching.interpret("Never press Share without my OK", doc)[0]["kind"] == "stop_before"  # not an answer
    assert teaching.interpret("Always end it with #brand", doc)[0] == {"kind": "caption_suffix", "value": "#brand", "quote": "Always end it with #brand"}
    assert teaching.interpret("To publish tap Post now.", doc)[0] == {"kind": "publish_via", "value": "Post now", "quote": "To publish tap Post now."}
    assert teaching.interpret('{"step": "s1", "set": {"optional": true}}', doc)[0]["kind"] == "edit"
    assert teaching.interpret("Wait for the upload bar to finish", doc) == [{"kind": "note"}]


def _png(path: Path, w: int = 1170, h: int = 2532) -> None:
    """A blank PNG of the phone's size (the session route reads the size; the OCR comes with the frame)."""
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = zlib.compress(b"".join(b"\0" + b"\0" * ((w + 7) // 8) for _ in range(h)), 9)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 1, 0, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def session_fixture(d: Path) -> Path:
    """FIXTURE: a phone session of the fictional Studio app as phonebridge journals it (looks with their frames and OCR,
    actions with their touch points and times). Hand-written; not a real account or app."""
    def it(text, x, y, h=56):
        return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": h, "conf": 1.0}
    looks = {
        "001-in-home": [it("9:41", 120, 50), it("Studio", 585, 220), it("Your story", 160, 620), it("Home", 120, 2400), it("Inbox", 700, 2400)],
        "002-in-new post": [it("Cancel", 120, 230), it("New post", 585, 230), it("Recents", 160, 560), it("0:15", 330, 1100), it("0:42", 720, 1100),
                            it("1:03", 1100, 1100)],
        "003-in-edit": [it("Back", 110, 230), it("Edit", 585, 230), it("Next", 1050, 230), it("Trim", 120, 2200), it("Music", 600, 2200)],
        "004-in-details": [it("Back", 110, 230), it("Details", 585, 230), it("Write a caption...", 520, 520), it("Tag people", 220, 1100),
                           it("Save draft", 300, 2380), it("Share", 860, 2380)],
        "005-in-caption": [it("Caption", 585, 230), it("OK", 1060, 230), it("Spring menu is here #cafe", 480, 420), it("Tag people", 220, 1100)],
        "006-in-details": [it("Back", 110, 230), it("Details", 585, 230), it("Spring menu is here #cafe", 480, 520), it("Tag people", 220, 1100),
                           it("Save draft", 300, 2380), it("Share", 860, 2380)],
    }
    (d / "frames").mkdir(parents=True)
    for name, items in looks.items():
        _png(d / "frames" / f"{name}.png")
        (d / "frames" / f"{name}.ocr.json").write_text(json.dumps(items))
    t, ev = 1000.0, [{"seq": 1, "ts": 999.0, "step": "stage", "outcome": "started"}]
    def add(**e):
        ev.append({"seq": len(ev) + 1, "ts": t, **e})
    for name, acts in (("001-in-home", ["tap create (+, top-left) (84,218)"]),
                       ("002-in-new post", ["tap video cell 0:15 (330,1050)", 'tap "Next" (1050,230)']),
                       ("003-in-edit", ['tap "Next" (1050,230)']),
                       ("004-in-details", ['tap "Write a caption..." (520,520)', "paste caption (25 chars)"]),
                       ("005-in-caption", ['tap "OK" (close editor) (1060,230)']), ("006-in-details", [])):
        add(msg=f"in: {name[7:]}", frame_ref=f"frames/{name}.png")
        for a in acts:
            t += 1.2
            add(msg=a)
        t += 2.0
    add(step="stage", outcome="ok")
    (d / "journal.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    return d


def test_a_phone_session_becomes_a_runnable_flow_once_publishing_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("TW_TEACH_MODEL", "claude-plan")  # an agent's session is private: it never reaches an external model
    org, owner, acme = org_with()
    ev = session_fixture(tmp_path / "session")
    narration = "Tap + at the top left, pick the newest video, then Next, Next, add the caption and tap OK. Never tap Share without my approval."
    rec = teaching.ingest_session(org["id"], ev, narration, owner["id"], acme["id"], label="agent")
    r = teaching.derive_routine(rec, "Studio post (session)")
    doc = teaching.doc_of(r)
    assert r["provenance"] == "operator_demo" and doc["interpreter"] == "off" and doc["recording"]["kind"] == "session"
    assert [(s["op"], s.get("label")) for s in doc["steps"]] == [
        ("tap", ""), ("select_media", None), ("tap", "Next"), ("tap", "Next"), ("focus", "Write a caption..."), ("type", None),
        ("dismiss_keyboard", "OK")]
    icon, pick, typed = doc["steps"][0], doc["steps"][1], doc["steps"][5]
    assert icon["icon_at"] == [0.072, 0.086] and "touch" in [p["kind"] for p in icon["proof"]]
    assert (pick["pick"], pick["duration"]) == ("newest", "0:15") and typed["value"] == "Spring menu is here #cafe"
    assert [a["label"] for a in doc["avoid"]] == ["Share"] and doc["behaviour"]["humanize"]["think"]["tap"][0] == pytest.approx(1200, rel=0.2)
    media = agency.save_media(org["id"], "next.mp4", b"\0" * 64)
    agency.grant(acme["id"], "approve_each", owner["id"], "test grant")
    job = agency.submit_job(org["id"], acme["id"], "publish", {"media": [{"sha256": media["sha256"]}], "caption": "Summer hours start Monday"},
                            owner["id"], routine_id=r["id"])
    p = teaching.plan(r, job)
    assert p["status"] == "needs_input" and p["flow"] is None and "Share" in p["needs"]
    r = teaching.correct(r["id"], "q1: yes", owner["id"])  # Share is the publishing button
    p = teaching.plan(r, job)
    assert p["status"] == "ready" and p["caption"] == "Summer hours start Monday" and not teaching.recording(rec)["frames"] == []
    states = p["flow"]["states"]
    assert [st["do"] for st in states] == [
        {"tap_at": [0.072, 0.086], "what": "create (+, top-left)"}, {"pick": "first", "next": "Next", "select": "body"}, {"tap": "Next"},
        {"caption": ["Write a caption"], "close_at": [0.5, 0.06]}, {"caption": ["Write a caption"], "close_at": [0.5, 0.06]}, "done"]
    assert states[3]["all_of"] == ["Write a caption"] and states[4]["all_of"] == ["OK"] and states[-1]["bottom"] == ["Share"]
    assert teaching.recording(rec)["preserved"]


def account_check_fixture(d: Path, handle: str = "studio_fixture", app: str = "app") -> Path:
    """FIXTURE: a read-only account check as a phone session journals it (a fictional account and app layout modelled on
    a profile screen: search tab, search field, typed handle, the matching row, the profile with its counts). app=
    "instagram": the same fictional account on Instagram's screen ids, its Search tab touched at the tab's own point."""
    def it(text, x, y, h=56):
        return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": h, "conf": 1.0}
    looks = {
        "001-in-home": (f"{app}.home", [it("Studio", 585, 220), it("Your story", 160, 620)]),
        "002-in-search": (f"{app}.search", [it("Search", 585, 230), it("For you", 200, 420), it("Accounts", 520, 420)]),
        "003-in-search typing": (f"{app}.search.typing", [it(handle, 300, 230), it("Cancel", 1060, 230), it(handle, 330, 420), it("Fixture Studio", 380, 470)]),
        "004-in-profile": (f"{app}.profile.other", [it(handle, 585, 220), it("1,204", 450, 520), it("posts", 450, 580), it("38.5K", 700, 520),
                                                 it("followers", 700, 580), it("312", 950, 520), it("following", 950, 580), it("Message", 700, 760)]),
    }
    (d / "frames").mkdir(parents=True)
    for name, (sid, items) in looks.items():
        _png(d / "frames" / f"{name}.png")
        (d / "frames" / f"{name}.ocr.json").write_text(json.dumps(items))
    t, ev = 1000.0, [{"seq": 1, "ts": 999.0, "step": "stage", "outcome": "started"}]
    for name, acts in (("001-in-home", ["tap search tab (819,2357)" if app == "instagram" else "tap search tab (700,2420)"]), ("002-in-search", ['tap "Search" (585,230)', "type handle (14 chars)"]),
                       ("003-in-search typing", [f'tap "{handle}" (330,420)']), ("004-in-profile", [])):
        ev.append({"seq": len(ev) + 1, "ts": t, "msg": f"in: {name[7:]}", "screen_id": looks[name][0], "frame_ref": f"frames/{name}.png"})
        for a in acts:
            t += 1.4
            ev.append({"seq": len(ev) + 1, "ts": t, "msg": a})
        t += 3.0
    ev.append({"seq": len(ev) + 1, "ts": t, "step": "stage", "outcome": "ok"})
    (d / "journal.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    return d


def test_a_read_only_account_check_exports_a_parameterised_workflow(tmp_path):
    org, owner, acme = org_with()
    narration = "Check @studio_fixture's profile: the handle, posts, followers and following."
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s"), narration, owner["id"], acme["id"], label="agent")
    r = teaching.derive_routine(rec, "Account check")
    doc = teaching.doc_of(r)
    ops = [(s["op"], s.get("label_slot") or s.get("slot") or s.get("label")) for s in doc["steps"]]
    assert ops == [("tap", ""), ("focus", "Search"), ("type", "handle"), ("tap", "handle"),
                   ("read", "posts"), ("read", "followers"), ("read", "following")]
    assert doc["slots"]["handle"]["type"] == "handle" and doc["steps"][0]["screen"]["id"] == "app.home"
    path = teaching.learned_path(doc, "Account check")
    assert path["goal"] == "Check @{handle}'s profile" and path["slots"] == {"handle": {"type": "handle"}}
    assert [x["do"] for x in path["steps"]] == ["tap:icon:search_tab", "tap:text:Search", "type:handle", "tap:text:{handle}",
                                                "read:posts@+0.000,-0.024", "read:followers@+0.000,-0.024", "read:following@+0.000,-0.024"]
    assert path["steps"][3] == {"at": "app.search.typing", "do": "tap:text:{handle}", "expect": ["app.profile.other"]}
    assert [u["need"] for u in path["unresolved"]] == ["the record target for this icon"]  # every screen has its id from the session
    wf = teaching.export_workflow(r)
    assert "studio_fixture" not in json.dumps(wf) and "frames" not in json.dumps(wf) and wf["pacing"]["source"].startswith("an agent")
    # imported elsewhere (another org, no recording): it plans a fresh account the same way, and is a template, not a taught routine
    org2, owner2, other = org_with("bright_bakery")
    r2 = teaching.import_workflow(org2["id"], wf, owner2["id"], other["id"])
    assert r2["provenance"] == "template" and r2["recording_id"] is None
    assert [x["do"] for x in teaching.learned_path(teaching.doc_of(r2), "Account check")["steps"]] == [x["do"] for x in path["steps"]]
    with pytest.raises(teaching.TeachError):
        teaching.import_workflow(org2["id"], {"format": "something else"}, owner2["id"])


def strict_act(record):
    """missions.act's contract, for fakes: a resolver gives a (kind, what, tool, args) spec or None, and anything else is
    unpacked as the real act does (live 11-08 a resolver's False crashed the start recovery)."""
    def act(ctx, o, stops, resolve, note):
        spec = resolve(o)
        if spec is None:
            return None
        kind, what, tool, args = spec
        record(kind, what, args)
        return o
    return act


class FakeCtx:
    """SIMULATED workspace context (INTERFACES v2 ctx): canned observations, recorded steps. Proves the executor's wiring
    and its checks, never a phone run."""

    def __init__(self, task, observations, answers=()):
        self.task, self.device, self.obs, self.answers, self.steps = task, {"id": "dev_fake"}, list(observations), list(answers), []

    def observe(self, note=None):
        o = self.obs.pop(0)
        self.steps.append(("observe", note, o["frame_id"]))
        return o

    def step(self, kind, text, frame_id=None, pace_ms=None, **data):
        self.steps.append((kind, text, frame_id))

    def check(self):
        pass

    def needs(self, text):
        self.steps.append(("needs", text, None))
        return self.answers.pop(0)


def test_taught_run_verifies_each_read_on_its_own_result_frame(tmp_path, monkeypatch):
    from teachable_worker import phone

    org, owner, acme = org_with(platform="instagram")
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s", app="instagram"), "Check @studio_fixture's profile: posts, followers and following.",
                                  owner["id"], acme["id"], label="agent")
    r = teaching.confirm(teaching.derive_routine(rec, "Account check")["id"], owner["id"])
    calls = []

    def run_learned(device, path, goal, slots, ctx):  # SIMULATED runner: what tw-phone's do runtime returns
        calls.append((path["goal"], goal, slots, [s["do"] for s in path["steps"]]))
        return {"status": "done", "answers": {"followers": "2.1M"}, "steps": [{"tier": 0, "pace_ms": 2100}, {"tier": 0, "pace_ms": 1800},
                                                                              {"tier": 1, "pace_ms": 2500}, {"tier": 0, "pace_ms": 1900}]}
    monkeypatch.setattr(phone, "run_learned", run_learned, raising=False)
    assert teaching.TASKS["taught_run"].available() == (True, "")
    def it(text, x, y):
        return {"text": text, "x": x, "y": y, "w": 80, "h": 56}
    home = {"frame_id": "f1", "captured_at": "t1", "width": 1170, "height": 2532, "items": [it("Studio", 585, 220), it("Your story", 160, 620)]}
    prof = {"frame_id": "f2", "captured_at": "t2", "width": 1170, "height": 2532,
            "items": [it("nasa", 585, 220), it("4,511", 450, 520), it("posts", 450, 580), it("2.1M", 700, 520), it("followers", 700, 580)]}
    ctx = FakeCtx({"input": {"routine_id": r["id"], "params": {}}}, [home, prof], answers=["@nasa"])
    with pytest.raises(teaching.TeachError, match="the following count could not be read on the result frame"):  # never 'done'
        teaching.TASKS["taught_run"].run(ctx)
    assert calls[0][1] == "Check @nasa's profile" and calls[0][2] == {"handle": "nasa"} and "tap:text:{handle}" in calls[0][3]
    assert ("needs", "Which handle should I use?", None) in ctx.steps  # asked once, then used
    assert ("note", "Read: posts 4,511, followers 2.1M; not read: the following count", "f2") in ctx.steps  # what was found, with its frame
    prof["items"].append(it("312", 950, 520)) or prof["items"].append(it("following", 950, 580))
    ctx = FakeCtx({"input": {"routine_id": r["id"], "params": {"handle": "nasa"}}}, [home, prof])
    out = teaching.TASKS["taught_run"].run(ctx)
    assert out["reads"]["followers"] == {"value": "2.1M", "frame_id": "f2", "observed_at": "t2", "runtime_said": "2.1M"}
    assert out["handle_shown"] is True and out["steps_by_tier"] == {"0": 3, "1": 1} and out["learned_steps_used"] == 3
    assert out["pace"][0] == {"step": 1, "taught_ms": 1400, "drawn_ms": 2100}
    assert "At run time: 2 run(s) logged" in teaching._routine(r["id"])["summary"]  # SIMULATED runs: the line now reports them


@video_ok
def test_a_recorded_run_lays_its_logged_touches_and_screen_ids_on_the_video(demo, tmp_path):
    """FIXTURE: the synthetic clip as a strip, with a SYNTHETIC run log of the same scripted touches (what the daemon's
    events.jsonl holds: `tap` points, `task_step` screens). The unseen tap needs no question once its touch is logged."""
    from datetime import datetime, timezone

    t0 = 1_800_000_000.0
    iso = lambda s: datetime.fromtimestamp(t0 + s, timezone.utc).isoformat()  # noqa: E731
    points = {("tap", "Create"): (210, 1206), ("tap", "Next"): (505, 112), ("dismiss_keyboard", "Done"): (515, 790), ("tap", "Save draft"): (150, 1170),
              ("focus", "Write a caption..."): (330, 215)}
    screens = {"Studio": "studio.home", "New post": "studio.new_post", "Edit": "studio.edit", "Details": "studio.details"}
    ev = [{"t": iso(0.1), "ev": "look", "width": 590, "height": 1276}]
    for m in demo["truth"]:
        if (m["op"], m.get("label")) in points:
            x, y = points[(m["op"], m.get("label"))]
            ev.append({"t": iso(m["t"] + 0.05), "ev": "tap", "x": x, "y": y, "what": m.get("label")})
        if m.get("screen") in screens:
            ev.append({"t": iso(m["t"] - 0.1), "ev": "task_step", "screen": screens[m["screen"]], "op": m["op"]})
    run = tmp_path / "run"
    run.mkdir()
    shutil.copyfile(demo["video"], run / "strip.mp4")
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    (run / "strip.json").write_text(json.dumps({"t0": t0, "fps": 30}))
    org, owner, acme = org_with()
    rec = teaching.ingest_session(org["id"], run, demo["narration"], owner["id"], acme["id"], label="agent")
    r = teaching.derive_routine(rec, "Client post (recorded run)")
    doc = teaching.doc_of(r)
    assert doc["recording"]["kind"] == "video+events" and teaching.analyse(rec)["alignment"] == "strip t0"
    edit = next(s for s in doc["steps"] if s["screen"]["title"] == "Edit")
    assert edit["label"] == "Next" and edit["confidence"] >= 0.85 and "touch" in [p["kind"] for p in edit["proof"]]
    assert not [q for q in doc["questions"] if q.get("blocking")]  # the logged touch answered what the video alone could not
    assert edit["screen"].get("id") == "studio.edit" and doc["steps"][0]["screen"].get("id") == "studio.home"


def test_an_operator_picks_a_captured_session_to_teach_from(tmp_path, monkeypatch, http):
    """FIXTURE session folder under $TW_SESSIONS; the routes list it to a founder, never to a customer, and refuse a path."""
    root = tmp_path / "sessions"
    account_check_fixture(root / "20261108T052707Z-account-check")
    monkeypatch.setenv("TW_SESSIONS", str(root))
    org = agency.create_org("Founder", "agency", "founder")
    founder = agency.add_person(org["id"], "Founder", role="founder", cohort="founder")
    st, lst = http("GET", "/api/teach/sessions", token=founder["token"])
    assert st == 200 and lst == [{"id": "20261108T052707Z-account-check", "label": "account check", "captured_at": "2026-11-08T05:27:07+00:00",
                                  "device": None, "kind": "looks+journal", "frames": 4, "duration_s": 18.6}]
    st, out = http("POST", "/api/teach/sessions", {"session_id": lst[0]["id"], "narration": "Check @studio_fixture's profile: posts, followers."},
                   token=founder["token"])
    assert st == 200 and out["recording_id"].startswith("rec_")
    st, view = http("POST", f"/api/teach/recordings/{out['recording_id']}/routine", {"name": "Account check"}, token=founder["token"])
    assert st == 200 and view["provenance"] == "operator_demo" and view["slots"]["handle"]["type"] == "handle"
    st, again = http("GET", f"/api/teach/routines/{view['id']}", token=founder["token"])  # reopening a saved routine keeps its inputs
    assert st == 200 and again["slots"]["handle"]["type"] == "handle" and again["questions"] == [] and len(again["steps"]) == len(view["steps"])
    assert teaching.routine_extras(teaching._routine(view["id"]))["slots"] == view["slots"]  # what agency.routine_view can merge in
    assert http("POST", "/api/teach/sessions", {"session_id": "../etc"}, token=founder["token"])[0] == 400
    cust_org, cust, _ = org_with()
    assert http("GET", "/api/teach/sessions", token=cust["token"])[0] == 400


def test_taught_steps_drive_missions_primitives_with_the_taught_pauses(tmp_path, monkeypatch):
    """SIMULATED: a fake missions module (tw-phone's primitives) and fake looks. Shows the interpreter's own logic: the
    routine's steps choose what is tapped and typed, its pauses are what is waited, a wrong screen stops the run."""
    import contextlib
    import time as _time
    import types

    from teachable_worker import missions

    monkeypatch.setattr(_time, "sleep", lambda s: waited.append(s))
    waited = []
    org, owner, acme = org_with(platform="instagram")
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s", app="instagram"), "Check @studio_fixture's profile: posts, followers and following.",
                                  owner["id"], acme["id"], label="agent")
    r = teaching.confirm(teaching.derive_routine(rec, "Account check")["id"], owner["id"])

    def it(text, x, y, h=56):
        return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": h}
    def look(fid, *items):
        return {"frame_id": fid, "captured_at": fid, "width": 1170, "height": 2532, "items": list(items)}
    looks = [look("f1", it("Studio", 585, 220), it("Your story", 160, 620)) | {"screen": "instagram.home"},
             look("f2", it("Search", 585, 230), it("For you", 200, 420), it("Accounts", 520, 420)) | {"screen": "instagram.screen.instagram_explore_search_grid"},
             look("f3", it("Search", 585, 230), it("Cancel", 1060, 230), it("Q W E R T Y U I O P", 585, 1900), it("space", 585, 2300)) | {"screen": "instagram.search.typing"},
             look("f4", it("nasa", 330, 420), it("NASA", 380, 470), it("Cancel", 1060, 230), it("nasa", 300, 230)) | {"screen": "instagram.search.typing"},
             look("f5", it("nasa", 585, 220), it("4,511", 450, 520), it("posts", 450, 580), it("98.1M", 700, 520), it("followers", 700, 580),
                  it("201", 950, 520), it("following", 950, 580))]
    taps, typed = [], []

    class Stop(RuntimeError):
        pass
    m = types.SimpleNamespace(
        Stop=Stop, NEVER=re.compile(r"^(follow|message|like)\b", re.I), TASKS={},
        _stops=lambda: [], _open_ig=lambda ctx, stops: looks.pop(0), _look=lambda ctx, note, stops: looks.pop(0),
        _line=missions._line, lead=missions.lead, TAB_SCREENS=missions.TAB_SCREENS, tab_screen=missions.tab_screen, where=missions.where,
        act=strict_act(lambda kind, what, a: typed.append(a["text"]) if kind == "type" else taps.append(a | {"what": what})),
        keyboard_up=lambda items, h: any(teaching.norm(i["text"]) == "space" for i in items))

    class Ctx:
        task, device = {"id": "t1", "input": {"routine_id": r["id"], "inputs": {"handle": "@nasa"}}}, {"id": "iphone12"}
        steps = []

        @contextlib.contextmanager
        def session(self):
            yield self

        def hands(self, ctx, kind, what, tool, **args):
            typed.append(args.get("text"))

        def observe(self, note=None):
            return looks.pop(0) if note == "before the taught run" else None

        def step(self, *a, **k):
            self.steps.append((a, k))

        def check(self):
            pass
    looks.insert(0, look("f0", it("Studio", 585, 220), it("Your story", 160, 620)))
    monkeypatch.setattr(teaching.TaughtRun, "_missions", staticmethod(lambda: m))
    out = teaching.TASKS["taught_run"].run(Ctx())
    assert out["status"] == "done" and typed == ["nasa"], out
    assert [t["what"] for t in taps] == ["Tap search tab (taught point)", "Tap 'Search'", "Tap 'nasa'"]  # the routine's steps, in order
    assert round(taps[0]["x"] / 1170, 3) == 0.7  # the demonstration's own touch for the text-less tab, nothing hard-coded
    assert out["reads"]["followers"]["value"] == "98.1M" and out["reads"]["followers"]["frame_id"] == "f5"
    assert out["learned_steps_used"] == 4 and all(p["drawn_ms"] >= 600 for p in out["pace"])
    assert [p["taught_ms"] for p in out["pace"]] == [1400, 1400, 1400, 1400]  # the demonstration's pause before each step
    assert [round(w * 1000) for w in waited[::2]][:4] == [p["drawn_ms"] for p in out["pace"]]  # and what was actually waited (each then a 1 s settle)


def test_a_workspace_task_becomes_a_demonstration(tmp_path, monkeypatch):
    """FIXTURE workspace task rows (INTERFACES v2 tasks/task_steps/frames) and FIXTURE OCR (a stand-in for the OCR
    engine, by frame name): the import keeps the run's order, times, logged touch points and typed text. The first look
    carries the bridge's own read (frames.items_json): it is used as is, and only the other frames are OCR'd."""
    from teachable_worker.store import tx

    with tx() as con:
        for ddl in ("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, device_id TEXT, kind TEXT, input_json TEXT, state TEXT, origin TEXT, "
                    "created TEXT, updated TEXT)",
                    "CREATE TABLE IF NOT EXISTS task_steps (task_id TEXT, idx INTEGER, at TEXT, kind TEXT, text TEXT, frame_id TEXT, pace_ms INTEGER, data_json TEXT)",
                    "CREATE TABLE IF NOT EXISTS frames (id TEXT PRIMARY KEY, device_id TEXT, task_id TEXT, captured_at TEXT, path TEXT, content_type TEXT, "
                    "width INTEGER, height INTEGER, source TEXT, simulated INTEGER)"):
            con.execute(ddl)
    ocr = {"search": [("Search", 585, 230), ("For you", 200, 420)],  # (no "instagram": that look is never OCR'd)
           "results": [("studio_fixture", 330, 420), ("Cancel", 1060, 230)],
           "profile": [("studio_fixture", 585, 220), ("1,204", 450, 520), ("posts", 450, 580), ("38.5K", 700, 520), ("followers", 700, 580)]}
    script = tmp_path / "ocr.py"
    script.write_text("import json, sys\nO = " + json.dumps(ocr) + "\nprint(json.dumps([[{'text': t, 'x': x, 'y': y, 'w': 22 * len(t), 'h': 56, 'conf': 1.0}"
                      " for t, x, y in O[p.rsplit('-', 1)[-1].split('.')[0]]] for p in sys.argv[1:]]))\n")
    monkeypatch.setenv("TW_OCR_CMD", f"{sys.executable} {script}")
    rows = [("observe", "Instagram", "home", {}), ("tap", "Tap the Search tab", None, {"at": [0.598, 0.956]}),
            ("observe", "Search", "search", {"screen": "app.search"}), ("tap", "Tap the search field", None, {"at": [0.5, 0.091]}),
            ("type", "Type studio_fixture", None, {}), ("observe", "Results", "typed", {}), ("tap", "Open studio_fixture", None, {}),
            ("observe", "Profile", "profile", {}), ("result", "@studio_fixture: 1,204 posts", None, {}), ("tap", "Tap the Home tab", None, {})]
    with tx() as con:
        con.execute("INSERT INTO tasks (id, device_id, kind, input_json, state, created, origin, updated) VALUES "  # the workspace's 19-column
                    "('tsk_fixture1', 'iphone12', 'account_check', '{}', 'done', '2026-11-08T06:12:30+00:00', 'test', "  # table takes this too
                    "'2026-11-08T06:12:30+00:00')")
        for n, (kind, text, shot, data) in enumerate(rows):
            fid = None
            if shot:
                fid = f"frm_{n}"
                _png(tmp_path / f"{n}-{shot}.png")
                read = json.dumps([{"text": t, "x": x, "y": y, "w": 22 * len(t), "h": 56} for t, x, y in (("Studio", 585, 220), ("Your story", 160, 620))]) if n == 0 else None
                con.execute("INSERT INTO frames (id, device_id, task_id, captured_at, path, content_type, width, height, source, "
                            "simulated, items_json) VALUES (?, 'iphone12', 'tsk_fixture1', ?, ?, 'image/png', 1170, 2532, 'bridge look', 0, ?)",
                            (fid, f"2026-11-08T06:12:{30 + 2 * n:02d}+00:00", str(tmp_path / f"{n}-{shot}.png"), read))
            con.execute("INSERT INTO task_steps (task_id, idx, at, kind, text, frame_id, pace_ms, data_json) VALUES ('tsk_fixture1', ?, ?, ?, ?, ?, NULL, ?)",
                        (n, f"2026-11-08T06:12:{30 + 2 * n:02d}+00:00", kind, text, fid, json.dumps(data)))
    org, owner, acme = org_with()
    rec = teaching.ingest_task(org["id"], "tsk_fixture1", "Check @studio_fixture's profile: posts and followers.", owner["id"])
    doc = teaching.doc_of(teaching.derive_routine(rec, "Account check (workspace run)"))
    assert [(s["op"], s.get("label_slot") or s.get("slot") or s.get("label") or s.get("read")) for s in doc["steps"]] == [
        ("tap", None), ("focus", "Search"), ("type", "handle"), ("tap", None), ("read", "posts"), ("read", "followers")]
    assert doc["steps"][0]["icon_at"] == [0.598, 0.956] and doc["steps"][2]["value"] == "studio_fixture"  # logged point, logged text
    assert doc["steps"][0]["screen"]["title"] == "Studio"  # read from the bridge's own items, not OCR
    assert doc["steps"][3]["confidence"] < teaching.ACCEPT and doc["steps"][3]["question"]  # no point logged: asked, not guessed
    assert "Tap the Home tab" not in json.dumps(doc["steps"])  # the import stops at the run's first result


def test_a_routine_from_an_older_derivation_is_not_run(tmp_path, monkeypatch):
    """A routine derived before the current teaching (stamped lower, or not at all) is refused by taught_run: derive it again."""
    from teachable_worker.store import tx

    org, owner, acme = org_with()
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s"), "Check @studio_fixture's profile: posts.", owner["id"],
                                  acme["id"], label="agent")
    r = teaching.confirm(teaching.derive_routine(rec, "Account check")["id"], owner["id"])
    assert teaching.doc_of(r)["teaching"] == teaching.TEACHING
    with tx() as con:
        row = json.loads(con.execute("SELECT doc_json FROM teaching_routines WHERE routine_id = ?", (r["id"],)).fetchone()[0])
        row.pop("teaching")
        con.execute("UPDATE teaching_routines SET doc_json = ? WHERE routine_id = ?", (json.dumps(row), r["id"]))
    with pytest.raises(teaching.TeachError, match="derived by an older teaching"):
        teaching.TASKS["taught_run"].run(FakeCtx({"input": {"routine_id": r["id"], "params": {"handle": "nasa"}}}, []))


def test_an_unconfirmed_routine_or_bad_params_never_touch_the_phone(tmp_path):
    """Refused before any look, session or input (the wrong pre-merge import, unconfirmed, was still runnable by API)."""
    org, owner, acme = org_with()
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s"), "Check @studio_fixture's profile: posts.", owner["id"],
                                  acme["id"], label="agent")
    r = teaching.derive_routine(rec, "Account check")

    class Untouched(FakeCtx):
        def observe(self, note=None):
            raise AssertionError("looked at the phone")

        def session(self):
            raise AssertionError("opened a session")

        def hands(self, *a, **k):
            raise AssertionError("sent an input")
    for inp, why in (({"routine_id": r["id"], "params": {"handle": "nasa"}}, "isn't confirmed"),):
        ctx = Untouched({"input": inp}, [])
        with pytest.raises(teaching.TeachError, match=why):
            teaching.TASKS["taught_run"].run(ctx)
        assert ctx.steps == []
    teaching.confirm(r["id"], owner["id"])
    for bad in (["nasa"], "nasa"):
        ctx = Untouched({"input": {"routine_id": r["id"], "params": bad}}, [])
        with pytest.raises(teaching.TeachError, match="params must be an object"):
            teaching.TASKS["taught_run"].run(ctx)
        assert ctx.steps == []


def test_taught_run_through_the_real_workspace_ctx_and_run_task(tmp_path, monkeypatch):
    """The real workspace.run_task and workspace.Ctx (input from the task's input_json, the session hook, observe storing
    frames), with a fake bridge (scripted looks) and fake missions primitives (taps recorded, none sent): an unconfirmed
    routine ends failed with zero steps; a confirmed one runs its taught steps and its reads rest on stored frames."""
    import contextlib
    import types

    from teachable_worker import workspace

    org, owner, acme = org_with(platform="instagram")
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s", app="instagram"), "Check @studio_fixture's profile: posts and followers.",
                                  owner["id"], acme["id"], label="agent")
    r = teaching.derive_routine(rec, "Account check")

    def it(text, x, y):
        return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": 56}
    G, T = "instagram.screen.instagram_explore_search_grid", "instagram.search.typing"
    looks = [("instagram.home", [it("Studio", 585, 220), it("Your story", 160, 620)]), ("instagram.home", [it("Studio", 585, 220), it("Your story", 160, 620)]),
             (G, [it("Search", 585, 230), it("For you", 200, 420), it("Accounts", 520, 420)]),
             (T, [it("Search", 585, 230), it("Cancel", 1060, 230), it("space", 585, 2300)]),
             (T, [it("nasa", 300, 230), it("nasa", 330, 420), it("NASA", 380, 470), it("Cancel", 1060, 230)]),
             ("instagram.profile.other", [it("nasa", 585, 220), it("4,511", 450, 520), it("posts", 450, 580), it("98.1M", 700, 520), it("followers", 700, 580)]),
             ("instagram.profile.other", [it("nasa", 585, 220), it("4,511", 450, 520), it("posts", 450, 580), it("98.1M", 700, 520), it("followers", 700, 580)])]

    class Bridge:
        simulated = True

        def observe(self, device):
            scr, items = looks.pop(0) if len(looks) > 1 else looks[0]
            return {"jpeg": b"\xff\xd8\xff\xe0 fake", "width": 1170, "height": 2532, "items": items, "screen": scr}
    sent = []

    @contextlib.contextmanager
    def session(ctx):  # hands write their input row through the ctx they are given, as missions.Hands does
        ctx.hands = lambda c, kind, what, tool, **a: sent.append((kind, a.get("text"))) or c.step(kind, what, pace_ms=900, tool=tool)
        yield
    from teachable_worker import missions
    m = types.SimpleNamespace(
        Stop=RuntimeError, NEVER=re.compile(r"^(follow|message)\b", re.I), TASKS={}, _stops=lambda: [],
        _open_ig=lambda ctx, stops: ctx.observe("Instagram"), _look=lambda ctx, note, stops: ctx.observe(note),
        _line=missions._line, lead=missions.lead, tab_screen=missions.tab_screen, TAB_SCREENS=missions.TAB_SCREENS, where=missions.where,
        keyboard_up=lambda items, h: any(teaching.norm(i["text"]) == "space" for i in items),
        act=missions.act)  # the real one: it hands the spec to ctx.hands
    monkeypatch.setattr(workspace, "bridge", Bridge())
    monkeypatch.setattr(workspace, "session_hook", lambda: session)
    monkeypatch.setattr(workspace, "executors", lambda: {"taught_run": teaching.TASKS["taught_run"]})
    monkeypatch.setattr(teaching.TaughtRun, "_missions", staticmethod(lambda: m))
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    def run(params):
        from teachable_worker.store import new_id, now, tx
        tid = new_id("tsk")
        with tx() as con:
            con.execute("INSERT INTO tasks (id, device_id, kind, input_json, state, origin, created, updated) VALUES (?, 'iphone12', "
                        "'taught_run', ?, 'queued', 'test', ?, ?)", (tid, json.dumps({"routine_id": r["id"], "params": params}), now(), now()))
        return workspace.run_task(workspace.get_task(tid))
    t = run({"handle": "nasa"})  # not confirmed yet
    assert t["state"] == "failed" and "isn't confirmed" in t["error"] and not sent
    assert query("SELECT COUNT(*) AS n FROM task_steps WHERE task_id = ?", (t["id"],))[0]["n"] == 0
    teaching.confirm(r["id"], owner["id"])
    t = run({"handle": "@nasa"})
    assert t["state"] == "done", t["error"]
    res = json.loads(t["result_json"])
    assert res["params"] == {"handle": "nasa"} and res["reads"]["followers"]["value"] == "98.1M" and res["handle_shown"]
    frames = {f["id"] for f in query("SELECT id FROM frames WHERE task_id = ?", (t["id"],))}
    assert res["reads"]["followers"]["frame_id"] in frames  # the read rests on a frame the workspace stored
    assert ("type", "nasa") in sent and any("taught pause" in s["text"] for s in query("SELECT text FROM task_steps WHERE task_id = ?", (t["id"],)))
    rows = [(r["kind"], json.loads(r["data_json"] or "{}")) for r in query("SELECT kind, data_json FROM task_steps WHERE task_id = ? AND kind IN "
                                                                          "('tap', 'type') ORDER BY idx", (t["id"],))]
    assert rows and all(d["phase"] == "taught" and d["step"] >= 1 and d["step_id"] for _, d in rows), rows  # each input row says
    assert [d["step"] for k, d in rows if k == "type"] == [3]  # which taught step made it (the fixture's typing is step 3)


def test_teach_frames_serve_session_jpgs_and_stay_in_their_folder(tmp_path, http):
    from teachable_worker.store import tx

    org = agency.create_org("Founder", "agency", "founder")
    founder = agency.add_person(org["id"], "Founder", role="founder", cohort="founder")
    d = account_check_fixture(tmp_path / "s")
    jpg = d / "frames" / "005-in-search.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    rec = teaching.ingest_session(org["id"], d, "", founder["id"], label="agent")
    st, body = http("GET", f"/api/teach/recordings/{rec}/frames/005-in-search.jpg", token=founder["token"])
    assert st == 200 and body == b"\xff\xd8\xff\xe0 fake jpeg"
    assert http("GET", f"/api/teach/recordings/{rec}/frames/001-in-home.png", token=founder["token"])[0] == 200
    for bad in ("..%2Fjournal.jsonl", "journal.jsonl", "x.gif"):
        assert http("GET", f"/api/teach/recordings/{rec}/frames/{bad}", token=founder["token"])[0] in (400, 404)


def test_a_taught_run_that_starts_on_search_typing_never_taps_the_tab_point(monkeypatch):
    """REAL-FRAME regression (root's run tsk_94270f11ea25, 11-08; the screen id given is the curated record's, the run's
    own was a generated record now retired): Instagram started on search typing; the daemon's look
    had no keyboard keys in it but named the screen ...search_typing; step 1 (the Search tab at its taught point) was
    tapped into the keyboard. Steps 1-5 are the real routine's (rtn_2d9f9f50e6d0; its own-profile title and feed anchors
    removed); the look is the stored frame frm_c6b1c6651ff1 read by RapidOCR with its keyboard keys taken out, as the
    daemon's 12-item look had them. Recognised: the navigation already done is skipped, the old-query clear isn't
    needed, the handle is typed into the search field. Not recognised: nothing is sent."""
    import contextlib
    import time as _time
    import types

    from teachable_worker import missions

    monkeypatch.setattr(_time, "sleep", lambda s: None)

    def it(text, x, y):
        return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": 50}
    real = [it("CSearchwith Meta Al", 454, 216), it("Recent", 131, 390), it("Seeall", 1054, 389), it("nasa", 269, 526),
            it("NASA", 116, 547), it("NASA", 270, 574), it("nasa", 270, 700), it("NASA", 974, 719), it("nasa·104Mfollowers", 408, 748),
            it("natgeo", 293, 874), it("National Geographic·268Mfollowers", 544, 921), it("national geographic", 426, 1048),
            it("natgeo·268Mfollowers", 429, 1096)]
    S, P = {"id": "search", "frame": "007-in-search.jpg", "key": "ci bearch with meta al", "title": "Ci bearch with Meta Al",
            "anchors": ["Recent", "See all"]}, lambda y, x: [{"kind": "touch", "at": [x, y]}]
    steps = [
        {"op": "tap", "icon_at": [0.7, 0.931], "what": "Tap the Search tab", "think_s": 0.413, "proof": P(0.9309, 0.7),
         "screen": {"id": "own_profile", "frame": "004-in-own_profile.jpg", "key": "", "title": "", "anchors": []},
         "expect": {"id": "search", "frame": "005-in-search.jpg", "key": "g search with meta al", "title": "G Search with Meta Al", "anchors": []}},
        {"op": "tap", "label": "G Search with Meta Al", "think_s": 0.948, "proof": P(0.0841, 0.4496),
         "screen": {"id": "search", "frame": "005-in-search.jpg", "key": "g search with meta al", "title": "G Search with Meta Al", "anchors": []},
         "expect": dict(S, frame="006-in-search.jpg")},
        {"op": "tap", "icon_at": [0.795, 0.084], "what": "Clear the previous search", "think_s": 0.469, "proof": P(0.0837, 0.7949),
         "screen": dict(S, frame="006-in-search.jpg"), "expect": S},
        {"op": "type", "slot": "handle", "think_s": 2.357, "screen": S,
         "expect": {"id": "search", "frame": "008-in-search.jpg", "key": "", "title": "", "anchors": ["TV"]}},
        {"op": "tap", "label": "national geographic o", "label_slot": "handle", "think_s": 1.234, "proof": P(0.1706, 0.4393),
         "screen": {"id": "search", "frame": "008-in-search.jpg", "key": "", "title": "", "anchors": ["TV"]},
         "expect": {"id": "keyword", "frame": "009-in-keyword.jpg", "key": "ci national geographic", "title": "Ci national geographic",
                    "anchors": ["For you", "Profiles", "Audio", "Tags"]}},
        {"op": "tap", "label": "Profiles", "think_s": 0.9, "screen": {"id": "keyword", "frame": "009-in-keyword.jpg", "key": "ci national geographic",
         "title": "Ci national geographic", "anchors": ["For you", "Profiles", "Audio", "Tags"]}, "expect": {"id": "keyword", "frame": "010-in-keyword.jpg"}},
        {"op": "read", "read": "followers@+0.000,-0.024", "label": "followers",
         "screen": {"id": "other_profile", "frame": "011-in-other_profile.jpg", "key": "", "title": "", "anchors": ["posts", "followers", "following"]}}]
    doc = {"platform": "instagram", "slots": {"handle": {"type": "handle"}}, "steps": steps}
    assert teaching.supported(doc) is None

    def run(first, then):
        sent, notes, looks = [], [], [first, *then]

        class Ctx:
            device = {"id": "iphone12"}

            def session(self):
                return contextlib.nullcontext()

            def step(self, kind, text, frame_id=None, **k):
                notes.append(text)
        m = types.SimpleNamespace(
            Stop=RuntimeError, NEVER=missions.NEVER, _stops=lambda: [], _open_ig=lambda ctx, stops: looks.pop(0),
            _look=lambda ctx, note, stops: looks.pop(0), _line=missions._line, lead=missions.lead, where=missions.where,
            keyboard_up=missions.keyboard_up, tab_screen=missions.tab_screen, TAB_SCREENS=missions.TAB_SCREENS, old_query=missions.old_query,
            act=strict_act(lambda kind, what, a: sent.append((kind, a.get("text"), a.get("y")))))
        try:
            return sent, notes, teaching.run_taught_steps(Ctx(), doc, {"handle": "examplestudio"}, m)["status"]
        except RuntimeError as e:
            return sent, notes, str(e)
    look = lambda items, scr=None: {"frame_id": "frm_c6b1c6651ff1", "width": 1170, "height": 2532, "items": items, **({"screen": scr} if scr else {})}  # noqa: E731
    typed = [it("examplestudio", 380, 216), it("Cancel", 1060, 216)]
    sent, notes, stop = run(look(real, "instagram.search.typing"), [look(typed, "instagram.search.typing")])
    assert sent == [("type", "examplestudio", None)], sent  # no tap at all: not at the tab point, not the field, not its clear
    assert any("steps 1–2 only lead there, skipped" in t for t in notes) and any("step 3 skipped" in t for t in notes)
    assert "step 5" in stop  # the typed query's results aren't in this look (the query in the field is no row): it stops there
    rows = typed + [it("examplestudio", 330, 430), it("Example Studio · 1,234 followers", 420, 470)]
    prof = [it("examplestudio", 585, 220), it("57", 450, 520), it("posts", 450, 580), it("1,234", 700, 520), it("followers", 700, 580),
            it("80", 950, 520), it("following", 950, 580), it("Follow", 400, 760)]  # FIXTURE profile (not a real read)
    T = "instagram.search.typing"
    sent, notes, stop = run(look(real, T), [look(rows, T), look(prof, "instagram.profile.other"), look(prof, "instagram.profile.other")])
    assert stop == "done" and [x[0] for x in sent] == ["type", "tap"] and sent[1][2] == 430  # the result row, not the field's query
    assert "already on the screen of step read: steps 6–6 skipped" in " | ".join(notes)  # the app went straight to the profile
    sent, notes, stop = run(look(real), [look(real)])  # the daemon recognised nothing: no positive tab screen, no input
    assert sent == [] and "step 1" in stop and "not recognised" in stop


def test_a_portable_export_never_carries_the_account_or_the_demonstration(tmp_path, monkeypatch):
    """tw-verifier's finding on rtn_2d9f9f50e6d0 (its export carried the founder's handle as the own profile's title, a
    friend's name, Explore-feed lines, a second demo handle in the goal). FIXTURE routine shaped like it: the own
    profile exports only its id, other screens only Instagram's chrome or a {param} title, the goal only parameters;
    a private value left anywhere else refuses the export. The runner's own use (portable=False) is unaffected."""
    org, owner, acme = org_with("acme_studio", platform="instagram")
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s", app="instagram"),
                                  "Check @studio_fixture, @other_fixture's profile: posts.", owner["id"], acme["id"], label="agent")
    r = teaching.derive_routine(rec, "Account check")
    doc_of_real = teaching.doc_of
    doc = teaching.doc_of(r)
    doc["steps"][0]["screen"].update(id="own_profile", title="acme_studio v", anchors=["Robin Fixture", "Edit profile", "Ask friends"])
    doc["steps"][1]["screen"].update(title="G Search with Meta Al", anchors=["A feed post line", "For you", "Social media"])
    monkeypatch.setattr(teaching, "doc_of", lambda r: doc)
    wf = teaching.export_workflow(r)
    text = json.dumps(wf)
    assert wf["steps"][0]["screen"] == {"id": "own_profile"} and wf["steps"][1]["screen"] == {"id": "instagram.search", "anchors": ["For you"]}
    assert wf["goal"] == "Check @{handle}'s profile" and wf["learned_path"]["goal"] == "Check @{handle}'s profile"
    assert wf["steps"][1]["target"] == {"label": "Search"} and wf["learned_path"]["steps"][1]["do"] == "tap:text:Search"  # the
    # field, not its OCR'd placeholder, in the learned path the runner drives too
    for private in ("acme_studio", "Robin", "Ask friends", "feed post", "Social media", "other_fixture", "studio_fixture", "Meta Al"):
        assert private not in text, private
    row = dict(doc["steps"][3])
    doc["steps"][3] = {**row, "label_slot": None, "label": "Robin's birthday party"}  # a tap on free text (a caption): not portable
    with pytest.raises(teaching.TeachError, match="tap text that isn't the app's own interface"):
        teaching.export_workflow(r)
    doc["steps"][3] = row
    doc["avoid"] = [{"label": "acme_studio", "screen": "", "anchors": []}]  # a private value where no screen rule reaches
    with pytest.raises(teaching.TeachError, match="Not exported"):
        teaching.export_workflow(r)
    assert "acme_studio" in json.dumps(teaching.export_workflow(r, portable=False))  # the runner's own copy, never shared
    org2, owner2, acme2 = org_with("brighter_fixture")  # a platform without a chrome list: no OCR'd screen text at all
    r2 = teaching.derive_routine(teaching.ingest_session(org2["id"], account_check_fixture(tmp_path / "s2"), "Check @studio_fixture's profile: posts.",
                                                         owner2["id"], acme2["id"], label="agent"), "Account check")
    monkeypatch.setattr(teaching, "doc_of", doc_of_real)
    screens = [sc for st in teaching.export_workflow(r2)["steps"] for sc in (st["screen"], st["expect"])]
    assert not any(sc.get("anchors") for sc in screens) and all("{" in sc.get("title", "{") for sc in screens), screens


def test_the_clear_is_tapped_only_on_a_field_that_holds_an_old_query(monkeypatch):
    """REAL-FRAME regression (root's run tsk_dccf35570abb, 11-08; the screen id given is the curated record's, the frame's
    own was a generated record now retired): after step 2 the field was empty and focused, its
    placeholder read by the daemon as 'Ci bearch with Meta Al'; the clear's gate took that for an old query, tapped the
    empty field, and iOS's AutoFill menu left a screen the daemon couldn't name: step 4 typed nothing. The look below is
    that frame's (frm_5ed6d345a92a) own items as the daemon stored them. Empty: the clear is skipped and the handle typed.
    The same look holding an old query: the clear is tapped at its taught point."""
    import contextlib
    import time as _time
    import types

    from teachable_worker import missions

    monkeypatch.setattr(_time, "sleep", lambda s: None)
    real = [{"text": "Ci bearch with Meta Al", "x": 452, "y": 212, "w": 521, "h": 66}, {"text": "Recent", "x": 130, "y": 388, "w": 157, "h": 35},
            {"text": "See all", "x": 1053, "y": 387, "w": 128, "h": 32}, {"text": "natgeo •", "x": 313, "y": 526, "w": 176, "h": 37},
            {"text": "National Geographic . 268M followers", "x": 545, "y": 575, "w": 640, "h": 34},
            {"text": "national geographic •", "x": 443, "y": 699, "w": 436, "h": 40}, {"text": "natgeo . 268M followers", "x": 430, "y": 749, "w": 411, "h": 34},
            {"text": "nasa •", "x": 290, "y": 871, "w": 131, "h": 30}, {"text": "NASA", "x": 272, "y": 920, "w": 93, "h": 27},
            {"text": "nasa •", "x": 290, "y": 1045, "w": 131, "h": 29}, {"text": "nasa 104M followers", "x": 409, "y": 1094, "w": 368, "h": 28},
            {"text": "123 e", "x": 149, "y": 2294, "w": 212, "h": 57}, {"text": "Ip", "x": 128, "y": 2480, "w": 23, "h": 13}]
    for placeholder in ("Ci bearch with Meta Al", "Q Search with Meta Al", "SearchwithMetaAl", "CSearchwith Meta Al", "Search"):
        assert missions.old_query([{"text": placeholder, "x": 452, "y": 212, "w": 500}], 2532) is None, placeholder
    assert missions.old_query([{"text": "Ci natgeo", "x": 300, "y": 212, "w": 200}], 2532) == "natgeo"
    stored = {"frm_5ed6d345a92a": ("Ci bearch with Meta Al", 452, 521, None),  # the top-bar line of stored frames' OCR:
              "demo 006": ("Ci bearch with Meta Al", 452, 521, None),          # the demonstration's own no-op clear (006 = 007)
              "demo 007": ("Ci bearch with Meta Al", 452, 521, None),
              "demo 016": ("Ci nasa", 280, 184, "nasa")}                        # an old query, its clear (x) beside it
    for frame, (text, x, w, query) in stored.items():
        assert missions.old_query([{"text": text, "x": x, "y": 212, "w": w, "h": 66}], 2532) == query, frame
    rows, real_ctx = [], types.SimpleNamespace(step=lambda kind, text, frame_id=None, **d: rows.append((kind, d)))
    phased = teaching._Phased(real_ctx)
    phased.hands = "set through the wrapper"
    phased.step("tap", "Tap the Profile tab", pace_ms=900, tool="tap")  # missions' account check, before any taught step
    phased.tag = {"phase": "taught", "step": 1, "step_id": "s1"}
    phased.step("tap", "Tap the Search tab", tool="tap") or phased.step("note", "step 1: …")
    assert real_ctx.hands == "set through the wrapper" and [d.get("phase") for _, d in rows] == ["start", "taught", None]
    cancel = {"text": "Cancel", "x": 1076, "y": 212, "w": 130, "h": 40}  # tw-phone-review: the typing screen's Cancel beside the field
    assert missions.old_query([cancel], 2532) is None  # the placeholder unread (faint grey): no query, no clear
    assert missions.old_query([{"text": "Ci nasa", "x": 280, "y": 212, "w": 120, "h": 66}, cancel], 2532) == "nasa"
    S = {"id": "search", "frame": "006-in-search.jpg", "key": "ci bearch with meta al", "title": "Ci bearch with Meta Al", "anchors": ["Recent", "See all"]}
    doc = {"platform": "instagram", "slots": {"handle": {"type": "handle"}}, "steps": [
        {"op": "tap", "icon_at": [0.795, 0.084], "what": "Clear the previous search", "think_s": 0.469, "screen": S, "expect": dict(S, frame="007-in-search.jpg"),
         "proof": [{"kind": "touch", "at": [0.7949, 0.0837]}]},
        {"op": "type", "slot": "handle", "think_s": 2.357, "screen": dict(S, frame="007-in-search.jpg"),
         "expect": {"id": "search", "frame": "008-in-search.jpg", "key": "", "title": "", "anchors": ["TV"]}}]}
    T = "instagram.search.typing"
    look = lambda items: {"frame_id": "frm_5ed6d345a92a", "width": 1170, "height": 2532, "items": items, "screen": T}  # noqa: E731
    typed = look([{"text": "Q examplestudio", "x": 380, "y": 212, "w": 360, "h": 60}])

    def run(first, *then):
        sent, notes, looks = [], [], [first, *then]

        class Ctx:
            device = {"id": "iphone12"}

            def session(self):
                return contextlib.nullcontext()

            def step(self, kind, text, frame_id=None, **k):
                notes.append(text)
        m = types.SimpleNamespace(
            Stop=RuntimeError, NEVER=missions.NEVER, _stops=lambda: [], _open_ig=lambda ctx, stops: looks.pop(0),
            _look=lambda ctx, note, stops: looks.pop(0), _line=missions._line, lead=missions.lead, where=missions.where,
            keyboard_up=missions.keyboard_up, tab_screen=missions.tab_screen, TAB_SCREENS=missions.TAB_SCREENS, old_query=missions.old_query,
            recognised=lambda dev: T,  # a fresh recognition (a later frame) would say typing: never used for this look
            act=strict_act(lambda kind, what, a: sent.append((kind, a.get("text"), round(a.get("y") or 0)))))
        return sent, notes, teaching.run_taught_steps(Ctx(), doc, {"handle": "examplestudio"}, m)["status"]
    sent, notes, status = run(look(real), typed)
    assert status == "done" and sent == [("type", "examplestudio", 0)], sent  # the empty field: no tap at all
    retired = dict(look(real), screen="instagram.screen.instagram_search_typing")  # the frame's own id: a retired generated record
    with pytest.raises(RuntimeError, match="search field isn't focused"):
        run(retired, retired, retired)
    unseen = {k: v for k, v in look(real).items() if k != "screen"}  # stored without a screen: unrecognised, whatever is later
    with pytest.raises(RuntimeError, match="not recognised"):
        run(unseen, unseen, unseen)
    assert any(t.startswith("step 1 skipped") for t in notes)
    held = [dict(real[0], text="Ci natgeo", w=200)] + real[1:]
    sent, notes, status = run(look(held), look(real), typed)
    assert status == "done" and sent == [("tap", None, 213), ("type", "examplestudio", 0)], sent  # the old query: cleared first


def test_a_recordings_files_are_private_from_creation(tmp_path, monkeypatch):
    """Root's file-mode rule (store.write_private): under a permissive umask, every copy of a recording's file is 0600
    before its data streams in, and the kept session is the app's own (dirs 0700, files 0400)."""
    import os
    import shutil
    import stat

    org, owner, acme = org_with()
    d = account_check_fixture(tmp_path / "s")
    at_copy, real = [], shutil.copyfile
    monkeypatch.setattr(teaching.shutil, "copyfile", lambda s, dst: at_copy.append(stat.S_IMODE(os.stat(dst).st_mode)) or real(s, dst))
    old = os.umask(0)
    try:
        rid = teaching.ingest_session(org["id"], d, "Check @studio_fixture's profile: posts.", owner["id"], acme["id"], label="agent")
    finally:
        os.umask(old)
    kept = Path(teaching._recording(rid)["path"])
    assert at_copy and set(at_copy) == {0o600}, at_copy  # private before the first byte
    assert all(stat.S_IMODE(f.stat().st_mode) == (0o700 if f.is_dir() else 0o400) for f in [kept, *kept.rglob("*")])


def test_a_count_fast_ocr_dropped_is_read_once_more_accurately_on_one_frame(tmp_path, monkeypatch):
    """Root's artifact run tsk_ab7297bd7e30 (11-08): every taught step was right, but the daemon's fast OCR dropped the lone
    '9' above 'following' on @apple's profile. FIXTURE looks shaped like it: ONE accurate look of the same screen, inside
    the session, no input; every read and the header are taken from that frame alone. A different profile on the reread
    fails on its header; without an accurate look (an older workspace) the failure says what couldn't be read."""
    import contextlib
    import time as _time
    import types

    from teachable_worker import missions

    monkeypatch.setattr(_time, "sleep", lambda s: None)
    org, owner, acme = org_with(platform="instagram")
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s", app="instagram"),
                                  "Check @studio_fixture's profile: posts, followers and following.", owner["id"], acme["id"], label="agent")
    r = teaching.confirm(teaching.derive_routine(rec, "Account check")["id"], owner["id"])

    def it(text, x, y):
        return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": 56}

    def look(fid, scr, *items):
        return {"frame_id": fid, "captured_at": fid, "width": 1170, "height": 2532, "items": list(items), "screen": scr}
    home = look("f1", "instagram.home", it("Studio", 585, 220), it("Your story", 160, 620))
    profile = lambda fid, head, *extra: look(fid, None, it(head, 585, 220), it("1,413", 450, 520), it("posts", 450, 580),  # noqa: E731
                                             it("38.1M", 700, 520), it("followers", 700, 580), it("following", 950, 580), *extra)
    steps_looks = [home, home, look("f2", "instagram.screen.instagram_explore_search_grid", it("Search", 585, 230), it("For you", 200, 420)),
                   look("f3", "instagram.search.typing", it("Search", 585, 230), it("Cancel", 1060, 230)),
                   look("f4", "instagram.search.typing", it("apple", 300, 230), it("apple", 330, 420), it("Cancel", 1060, 230)),
                   profile("fast", "apple")]  # the lone '9' dropped
    sent = []

    def ctx_for(accurate_look):
        looks = list(steps_looks)

        class Ctx:
            task, device, input = {"id": "t1"}, {"id": "iphone12"}, {"routine_id": r["id"], "params": {"handle": "apple"}}
            steps = []

            @contextlib.contextmanager
            def session(self):
                yield self

            def step(self, kind, text, frame_id=None, **k):
                self.steps.append((kind, text, frame_id))

            def check(self):
                pass

            def hands(self, ctx, kind, what, tool, **a):
                sent.append((kind, a.get("text")))

            if accurate_look is not None:
                def observe(self, note=None, accurate=False):
                    if accurate:
                        self.accurate_calls = getattr(self, "accurate_calls", 0) + 1
                        return accurate_look
                    return looks.pop(0)
            else:
                def observe(self, note=None):
                    return looks.pop(0)
        return Ctx()
    m = types.SimpleNamespace(
        Stop=RuntimeError, NEVER=missions.NEVER, TASKS={}, _stops=lambda: [], _open_ig=lambda ctx, stops: ctx.observe("Instagram"),
        _look=lambda ctx, note, stops: ctx.observe(note), _line=missions._line, lead=missions.lead, where=missions.where,
        keyboard_up=missions.keyboard_up, tab_screen=missions.tab_screen, TAB_SCREENS=missions.TAB_SCREENS, old_query=missions.old_query,
        read_profile=missions.read_profile, act=strict_act(lambda kind, what, a: sent.append((kind, a.get("text")))))
    monkeypatch.setattr(teaching.TaughtRun, "_missions", staticmethod(lambda: m))
    nine = it("9", 950, 520)
    ctx = ctx_for(profile("accurate", "apple", nine))
    out = teaching.TASKS["taught_run"].run(ctx)
    assert ctx.accurate_calls == 1 and out["read_accurately"] and out["result_frame"]["frame_id"] == "accurate"
    assert {k: (v["value"], v["frame_id"]) for k, v in out["reads"].items()} == {
        "posts": ("1,413", "accurate"), "followers": ("38.1M", "accurate"), "following": ("9", "accurate")}  # one frame, never a mix
    assert any(t.startswith("Not read on the result frame: the following count") and f == "fast" for _, t, f in ctx.steps)
    assert [k for k, _ in sent] == ["tap", "tap", "type", "tap"]  # the taught inputs only: the reread sent nothing
    ctx = ctx_for(profile("accurate", "applesupport", nine))  # another profile by the reread: its counts are not apple's
    with pytest.raises(teaching.TeachError, match="the result frame's header isn't @apple"):
        teaching.TASKS["taught_run"].run(ctx)
    ctx = ctx_for(None)  # a workspace without an accurate look: the honest failure, and what couldn't be read
    with pytest.raises(teaching.TeachError, match="the following count could not be read on the result frame"):
        teaching.TASKS["taught_run"].run(ctx)
