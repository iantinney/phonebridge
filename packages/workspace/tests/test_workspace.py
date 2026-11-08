"""The workspace through the real HTTP server, with a fake bridge (simulated) and a fake work session."""

import contextlib
import threading
import time

import pytest

from teachable_worker import workspace
from teachable_worker.store import query

ADMIN = "admin-test-token"
JPEG = b"\xff\xd8\xff\xe0 fake jpeg"


class FakeBridge:
    simulated = True

    def __init__(self):
        self.frames = 0

    def devices(self):
        return [{"id": "iphone12", "name": "iPhone 12"}]

    def presence(self, device):
        return {"attached": True, "lockdown": True, "tunnel": True, "locked": None, "ios": "27.2",
                "foreground": "com.burbn.instagram"}

    def frame(self, device):
        self.frames += 1
        return {"jpeg": JPEG, "source": "stream", "seq": self.frames, "age_ms": None}  # like the real frame tool

    def observe(self, device):
        return {"jpeg": JPEG, "width": 1170, "height": 2532, "items": [{"text": "natgeo", "x": 1, "y": 2}],
                "age_ms": 30, "frame_seq": 7}


@pytest.fixture
def phone(monkeypatch):
    b = FakeBridge()
    monkeypatch.setattr(workspace, "bridge", b)
    monkeypatch.setattr(workspace, "_beat", lambda: None)
    monkeypatch.setattr(workspace, "_lease", lambda d: None)
    held = {"busy": 0, "entered": 0, "exited": 0}

    @contextlib.contextmanager
    def session(ctx):
        if held["busy"]:
            held["busy"] -= 1
            raise workspace.Busy("held by autopilot ig:public")
        held["entered"] += 1
        try:
            yield
        finally:
            held["exited"] += 1

    monkeypatch.setattr(workspace, "session_hook", lambda: session)
    b.held = held
    return b


def wait_state(tid, *states, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        t = workspace.get_task(tid)
        if t["state"] in states:
            return t
        time.sleep(0.05)
    raise AssertionError(f"{tid} stayed {workspace.get_task(tid)['state']}")


def customer(http):
    s, out = http("POST", "/api/ops/orgs", {"name": "Client", "track": "agency", "cohort": "external",
                                            "person": {"name": "C"}}, ADMIN)
    return out["login_link"].rsplit("/", 1)[1]


def test_only_the_operator_sees_the_phone(http, phone):
    tok = customer(http)
    for method, path in (("GET", "/api/devices"), ("GET", "/api/devices/iphone12/frame"), ("GET", "/api/tasks"),
                         ("POST", "/api/devices/iphone12/observe")):
        assert http(method, path, {} if method == "POST" else None, tok)[0] == 403
    assert http("GET", "/api/devices")[0] == 401
    s, devs = http("GET", "/api/devices", token=ADMIN)
    assert s == 200 and devs[0]["connection"]["state"] == "connected" and devs[0]["ios"] == "27.2"


def test_live_view_is_lease_free_cached_audited_and_honest_about_age(http, phone):
    s, body = http("GET", "/api/devices/iphone12/frame", token=ADMIN)
    assert s == 200 and body == JPEG
    http("GET", "/api/devices/iphone12/frame", token=ADMIN)
    assert phone.frames == 1  # ~1 s cache: one capture shared by viewers
    assert query("SELECT COUNT(*) AS n FROM tasks")[0]["n"] == 0  # viewing is never a task
    assert query("SELECT COUNT(*) AS n FROM frames")[0]["n"] == 0  # and never stored
    assert query("SELECT COUNT(*) AS n FROM events WHERE name = 'live_view'")[0]["n"] == 1  # one audited burst
    s, devs = http("GET", "/api/devices", token=ADMIN)
    f = devs[0]["frame"]
    assert f["available"] and f["source"] == "stream" and f["pixel_age_s"] is None and f["fetched_at"]


def test_live_view_headers(tw, phone):
    import urllib.request
    from http.server import ThreadingHTTPServer
    from teachable_worker import server
    server.Handler.ROUTES = server.routes()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/api/devices/iphone12/frame",
                                 headers={"Authorization": f"Bearer {ADMIN}"})
    with urllib.request.urlopen(req) as r:
        h = r.headers
    httpd.shutdown()
    assert h["X-Pixel-Age-S"] == "unknown" and h["X-Source"] == "stream" and h["Cache-Control"] == "no-store"
    assert h["X-Fetched-At"] and float(h["X-Cache-Age-S"]) >= 0


class Recorder:
    """An executor that asks the person, records an input, and notes concurrency on its device."""

    read_only, input, route = True, {"q": "a question"}, "test"
    running, max_running, stops = 0, 0, []
    lock = threading.Lock()

    def available(self):
        return True, ""

    def stop(self, task):
        Recorder.stops.append(task["id"])

    def run(self, ctx):
        with Recorder.lock:
            Recorder.running += 1
            Recorder.max_running = max(Recorder.max_running, Recorder.running)
        try:
            with ctx.session():
                answer = ctx.needs("Which account?", poll_s=0.05) if ctx.input.get("q") else None
                ctx.step("tap", "Tapped Search", pace_ms=180)
                time.sleep(0.2)
                ctx.check()
            return {"answer": answer}
        finally:
            with Recorder.lock:
                Recorder.running -= 1


def test_message_answers_needs_and_one_task_per_device(http, phone, monkeypatch):
    monkeypatch.setattr(workspace, "executors", lambda: {"rec": Recorder()})
    s, a = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "rec", "input": {"q": 1}}, ADMIN)
    s, b = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "rec", "input": {}}, ADMIN)
    assert wait_state(a["id"], "needs")["needs"] == "Which account?"
    assert workspace.get_task(b["id"])["state"] == "queued"  # FIFO: b waits for a
    t0 = time.time()
    s, polled = http("GET", f"/api/tasks/{a['id']}?after=99&wait=3", token=ADMIN)  # long-poll ends on the message
    http("POST", f"/api/tasks/{a['id']}/message", {"text": "@natgeo"}, ADMIN)
    assert wait_state(a["id"], "done")["result_json"] == '{"answer": "@natgeo"}'
    wait_state(b["id"], "done")
    assert Recorder.max_running == 1
    s, view = http("GET", f"/api/tasks/{a['id']}?after=-1", token=ADMIN)
    assert [x["kind"] for x in view["steps"]] == ["message", "tap", "result"] and view["steps"][1]["pace_ms"] == 180
    assert view["messages"][0]["text"] == "@natgeo"


def test_stop_is_forwarded_to_the_executor(http, phone, monkeypatch):
    monkeypatch.setattr(workspace, "executors", lambda: {"rec": Recorder()})
    s, a = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "rec", "input": {"q": 1}}, ADMIN)
    wait_state(a["id"], "needs")
    http("POST", f"/api/tasks/{a['id']}/handoff", {"note": "I'll take it"}, ADMIN)
    assert wait_state(a["id"], "handed_off")["handoff_note"] == "I'll take it"
    for _ in range(50):
        if a["id"] in Recorder.stops:
            break
        time.sleep(0.02)
    assert a["id"] in Recorder.stops


def test_a_task_running_at_restart_fails_instead_of_resuming(tw):
    from teachable_worker.store import tx, now
    workspace.get_task  # schema via import
    with tx() as con:
        con.execute("INSERT INTO tasks (id, device_id, kind, input_json, state, origin, created, updated, session_open) "
                    "VALUES ('tsk_1', 'iphone12', 'watch', '{}', 'running', 'operator', ?, ?, 1)", (now(), now()))
    assert workspace.recover() == 1
    t = workspace.get_task("tsk_1")
    assert t["state"] == "failed" and t["session_open"] == 0 and "restart" in t["error"]


def test_old_evidence_frames_are_purged(tw, phone):
    f = workspace.save_frame("iphone12", JPEG, None, "test", captured_at="2020-01-01T00:00:00+00:00")
    assert workspace.purge_frames() == 1 and not query("SELECT 1 FROM frames WHERE id = ?", (f["id"],))


LEASE = """#!/bin/sh
# a lease CLI with the local-lease protocol: acquire prints the token; renew fails once $DIR/lose exists
case "$1" in
  acquire) [ -f "$DIR/held" ] && { echo "LeaseDenied: already leased by someone" >&2; exit 1; }
           touch "$DIR/held"; echo "tok-123";;
  renew)   [ -f "$DIR/lose" ] && { echo "lease expired" >&2; exit 1; }; exit 0;;
  release) rm -f "$DIR/held"; echo released >> "$DIR/log";;
  *) echo "usage: lease acquire|renew|release" >&2; exit 2;;
esac
"""


@pytest.fixture
def lease_cli(tw, monkeypatch):
    from types import SimpleNamespace
    from teachable_worker import missions
    monkeypatch.setenv("TW_PHONE_ADAPTER", "local")
    monkeypatch.setattr(missions, "Hands", lambda *args: SimpleNamespace(release=lambda: None))
    p = tw / "lease"
    p.write_text(LEASE)
    p.chmod(0o755)
    monkeypatch.setenv("DIR", str(tw))
    monkeypatch.setenv("TW_LEASE_CMD", str(p))
    return tw


def test_standalone_lease_session_takes_releases_and_loses_the_device(http, monkeypatch, lease_cli):
    monkeypatch.setattr(workspace, "bridge", FakeBridge())
    monkeypatch.setattr(workspace.LeaseSession, "TTL", 0.3)
    monkeypatch.setattr(workspace, "executors", lambda: {"rec": Recorder()})
    s, a = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "rec", "input": {"q": 1}}, ADMIN)
    wait_state(a["id"], "needs")
    assert (lease_cli / "held").exists() and workspace.get_task(a["id"])["session_open"] == 1
    (lease_cli / "lose").touch()  # the next renew fails: the phone may be someone else's now
    t = wait_state(a["id"], "stopped")
    assert t["session_open"] == 0 and "lease lost" in t["error"]


def test_bridge_refuses_inputs_on_the_read_path():
    with pytest.raises(ValueError):
        workspace.Bridge("http://127.0.0.1:9").tool("tap", device="iphone12", x=1, y=1)


def test_observe_is_an_immediate_stored_look(http, phone):
    s, look = http("POST", "/api/devices/iphone12/observe", {}, ADMIN)
    assert s == 201 and look["items"][0]["text"] == "natgeo" and look["observed_at"]
    assert query("SELECT COUNT(*) AS n FROM tasks")[0]["n"] == 0  # not a task: nothing waits in a queue
    s, body = http("GET", look["frame_url"], token=ADMIN)
    assert s == 200 and body == JPEG
    assert http("POST", "/api/devices/iphone12/observe", {}, ADMIN)[0] == 429  # looked a moment ago


def test_busy_device_waits_in_the_queue_naming_the_holder(http, phone, monkeypatch):
    monkeypatch.setattr(workspace, "BUSY_RETRY_S", 0.3)
    monkeypatch.setattr(workspace, "executors", lambda: {"rec": Recorder()})
    phone.held["busy"] = 1
    s, t = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "rec", "input": {}}, ADMIN)
    for _ in range(100):
        cur = workspace.get_task(t["id"])
        if cur["needs"]:
            break
        time.sleep(0.02)
    assert cur["state"] == "queued" and "held by autopilot" in cur["needs"]
    assert wait_state(t["id"], "done")["needs"] is None


def test_unavailable_kinds_name_their_prerequisite(http, monkeypatch):
    monkeypatch.setattr(workspace, "executors", lambda: {})
    s, ks = http("GET", "/api/tasks/kinds", token=ADMIN)
    by = {k["kind"]: k for k in ks}
    assert not by["account_check"]["available"] and "missions.py" in by["account_check"]["reason"]
    s, err = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "feed_scout"}, ADMIN)
    assert s == 400 and "not available yet" in err["error"]


def test_standalone_lease_denied_is_busy_and_usage_errors_fail(http, monkeypatch, lease_cli):
    monkeypatch.setattr(workspace, "bridge", FakeBridge())
    monkeypatch.setattr(workspace, "BUSY_RETRY_S", 30)
    monkeypatch.setattr(workspace, "executors", lambda: {"rec": Recorder()})
    (lease_cli / "held").touch()
    s, t = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "rec", "input": {}}, ADMIN)
    for _ in range(100):
        cur = workspace.get_task(t["id"])
        if cur["needs"]:
            break
        time.sleep(0.02)
    assert cur["state"] == "queued" and "LeaseDenied" in cur["needs"]
    workspace.stop(t["id"], None)
    (lease_cli / "held").unlink()
    monkeypatch.setenv("TW_LEASE_CMD", f"{lease_cli / 'lease'} bogus-group")  # misconfigured: usage error, exit 2
    s, t2 = http("POST", "/api/tasks", {"device_id": "iphone13", "kind": "rec", "input": {}}, ADMIN)
    assert "misconfigured" in wait_state(t2["id"], "failed")["error"]


@pytest.mark.parametrize("presence,state", [
    ({"attached": True, "lockdown": None, "tunnel": True}, "connected"),  # trust unread this moment: not "untrusted"
    ({"attached": True, "lockdown": None, "tunnel": None}, "unknown"),
    ({"attached": True, "lockdown": False, "tunnel": False}, "untrusted"),
    ({"attached": False}, "unplugged"),
    ({"attached": True, "lockdown": True, "tunnel": True, "locked": True}, "locked")])
def test_connection_never_guesses_an_instruction(monkeypatch, presence, state):
    monkeypatch.setattr(workspace, "_beat", lambda: None)
    workspace._conn.clear()
    monkeypatch.setattr(workspace.bridge, "presence", lambda d: presence)
    assert workspace.connection("iphone12")["state"] == state


def test_forget_task_removes_what_it_read(http, phone, monkeypatch):
    monkeypatch.setattr(workspace, "executors", lambda: {"rec": Recorder()})
    s, a = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "rec", "input": {"q": 1}}, ADMIN)
    wait_state(a["id"], "needs")
    workspace.message(a["id"], "the thread with Sam", None)
    wait_state(a["id"], "done")
    workspace.save_frame("iphone12", JPEG, a["id"], "test")
    with pytest.raises(workspace.NotFound):
        workspace.forget_task(a["id"], "per_someone_else")
    out = workspace.forget_task(a["id"], None)
    assert out["frames_deleted"] == 1
    for table in ("task_steps", "task_messages", "frames"):
        assert query(f"SELECT COUNT(*) AS n FROM {table} WHERE task_id = ?", (a["id"],))[0]["n"] == 0
    assert workspace.get_task(a["id"])["input_json"] == "{}"


def test_taught_run_reads_its_input_through_the_real_ctx(tw, phone, monkeypatch):
    """The real seam: a tasks row (input_json) → workspace.run_task → Ctx → TaughtRun. The routine id must arrive:
    a confirmed routine derived by an older teaching is refused for that reason, never as 'does not exist'."""
    from teachable_worker import agency, teaching
    org = agency.create_org("Founder", "agency", "founder")
    r = agency.create_routine(org["id"], "old routine", "operator_demo", None, steps=[{"op": "tap"}])
    agency.confirm_routine(r["id"], org["id"], None)
    import json
    from teachable_worker.store import now, tx
    with tx() as con:  # the row exactly as POST /api/tasks stores it (input_json), run by the real run_task
        con.execute("INSERT INTO tasks (id, device_id, kind, input_json, state, origin, created, updated) VALUES "
                    "('tsk_t1', 'iphone12', 'taught_run', ?, 'queued', 'test', ?, ?)",
                    (json.dumps({"routine_id": r["id"], "params": {"handle": "natgeo"}}), now(), now()))
    t = workspace.get_task("tsk_t1")
    done = workspace.run_task(t)
    assert done["state"] == "failed" and "does not exist" not in (done["error"] or "")
    assert query("SELECT COUNT(*) AS n FROM task_steps WHERE task_id = ? AND kind IN ('tap', 'type')", (t["id"],))[0]["n"] == 0


def test_a_stored_look_keeps_the_text_read_on_that_frame(http, phone):
    """What the bridge read (Apple Vision on the Mac) sits in the frame's own row: teaching needs no OCR install,
    and deleting the frame deletes the text."""
    import json
    s, look = http("POST", "/api/devices/iphone12/observe", {}, ADMIN)
    row = query("SELECT items_json FROM frames WHERE id = ?", (look["frame_id"],))[0]
    assert json.loads(row["items_json"])[0]["text"] == "natgeo"
    workspace.purge_frames(days=-1)
    assert query("SELECT COUNT(*) AS n FROM frames")[0]["n"] == 0


def test_frames_get_their_size_from_the_image_when_the_bridge_omits_it(tw):
    import io, struct, zlib
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1170, 2532) + b"\x08\x02\x00\x00\x00"
    assert workspace.image_size(png) == (1170, 2532)
    jpeg = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9 +
            b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 2532, 1170) + b"\x00" * 10)
    assert workspace.image_size(jpeg) == (1170, 2532)
    f = workspace.save_frame("iphone12", jpeg, None, "test")
    assert (f["width"], f["height"]) == (1170, 2532)
    assert workspace.image_size(b"not an image") == (None, None)


def test_a_finished_task_long_poll_answers_at_once(http, phone, monkeypatch):
    """Polling a task that is already over, after its last step, must not wait out the long-poll."""
    monkeypatch.setattr(workspace, "executors", lambda: {"rec": Recorder()})
    s, a = http("POST", "/api/tasks", {"device_id": "iphone12", "kind": "rec", "input": {}}, ADMIN)
    done = wait_state(a["id"], "done")
    last = query("SELECT MAX(idx) AS i FROM task_steps WHERE task_id = ?", (a["id"],))[0]["i"]
    t0 = time.time()
    s, view = http("GET", f"/api/tasks/{a['id']}?after={last}&wait=20", token=ADMIN)
    assert s == 200 and view["state"] == "done" and view["steps"] == [] and time.time() - t0 < 2


def test_a_look_is_scoped_to_the_foreground_app(monkeypatch):
    """Instagram in front → look(platform="instagram"): TikTok's records can't make a confident match ambiguous."""
    b = workspace.Bridge("http://127.0.0.1:1")
    calls = []
    monkeypatch.setattr(b, "presence", lambda d: {"foreground": "com.burbn.instagram"})
    monkeypatch.setattr(b, "tool", lambda name, **a: calls.append((name, a)) or {"structuredContent": {
        "frame_jpeg": "AAAA", "items": [], "screen": "instagram.profile.other"}})
    o = b.observe("iphone12")
    assert calls == [("look", {"device": "iphone12", "with_frame": True, "platform": "instagram"})]
    assert o["screen"] == "instagram.profile.other"
    monkeypatch.setattr(b, "presence", lambda d: {"foreground": "com.apple.springboard"})
    calls.clear()
    b.observe("iphone12")
    assert calls == [("look", {"device": "iphone12", "with_frame": True})]  # unknown app: unscoped, as before


def test_an_accurate_read_stores_exactly_the_image_vision_read_and_its_text(tw, monkeypatch):
    """ocr(fast=False, with_image=True) as phonebridge returns it: {items, image_jpeg, width, height}. The stored frame
    is that image with those lines (the single-digit count the fast pass dropped), never the earlier look's pixels."""
    import base64, json
    from teachable_worker.store import now, tx
    jpg = b"\xff\xd8\xff\xe0 accurate still"
    reply = {"items": [{"text": "9", "x": 700, "y": 520, "w": 20, "h": 40, "conf": 1.0},
                       {"text": "following", "x": 700, "y": 580, "w": 120, "h": 30, "conf": 1.0}],
             "image_jpeg": base64.b64encode(jpg).decode(), "width": 1170, "height": 2532}
    b = workspace.Bridge("http://127.0.0.1:1")
    calls = []
    monkeypatch.setattr(b, "tool", lambda name, **a: calls.append((name, a)) or {"structuredContent": {"result": reply}})
    monkeypatch.setattr(workspace, "bridge", b)
    with tx() as con:
        con.execute("INSERT INTO tasks (id, device_id, kind, input_json, state, origin, created, updated) VALUES "
                    "('tsk_acc', 'iphone12', 'taught_run', '{}', 'running', 'test', ?, ?)", (now(), now()))
    ctx = workspace.Ctx(workspace.get_task("tsk_acc"))
    o = ctx.observe("Final read (accurate)", accurate=True)
    assert calls == [("ocr", {"device": "iphone12", "fast": False, "with_image": True})]  # no look, no input
    f = query("SELECT * FROM frames WHERE id = ?", (o["frame_id"],))[0]
    assert open(f["path"], "rb").read() == jpg and (f["width"], f["height"]) == (1170, 2532)
    assert [i["text"] for i in json.loads(f["items_json"])] == ["9", "following"] and f["source"] == "bridge ocr (accurate)"
    step = query("SELECT data_json FROM task_steps WHERE task_id = 'tsk_acc'")[0]
    assert json.loads(step["data_json"])["accurate"] is True and json.loads(step["data_json"])["screen"] is None
    monkeypatch.setattr(b, "tool", lambda name, **a: {"structuredContent": {"result": {"items": []}}})
    with pytest.raises(RuntimeError):
        ctx.observe(accurate=True)  # text without its image is never stored
