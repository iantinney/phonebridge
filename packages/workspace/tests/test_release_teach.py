"""The released workspace teaches and runs a workflow through phonebridge-workspace's own route set (not the private
server's): a FIXTURE captured session becomes a routine through /api/teach, it is listed and confirmed through
/api/routines, and a taught_run task started through /api/tasks runs it in the real workspace (Ctx, run_task, stored frames) over a
SIMULATED bridge, session and missions primitives. No phone, network or private module."""

import json
import shlex
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from teachable_worker import httpd, standalone, teaching, workspace
from teachable_worker.missions import IG
from test_teaching import account_check_fixture, strict_act
from test_workspace import JPEG, wait_state


@pytest.fixture
def launcher(tw, monkeypatch):
    monkeypatch.setenv("TW_PHONE_ADAPTER", "local")
    monkeypatch.setattr(httpd.Handler, "ROUTES", httpd.compile_routes(standalone.routes()))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), httpd.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base, token = f"http://127.0.0.1:{srv.server_address[1]}", workspace.ensure_operator()["token"]

    def call(method, path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    yield call
    srv.shutdown()


@pytest.mark.parametrize("start", ["on the taught start screen", "off it"])
def test_the_release_teaches_a_session_lists_confirms_and_runs_it(launcher, tw, monkeypatch, start):
    monkeypatch.setenv("TW_SESSIONS", str(tw / "sessions"))
    account_check_fixture(tw / "sessions" / "20261108T052707Z-account-check", app="instagram")
    st, sessions = launcher("GET", "/api/teach/sessions")
    assert st == 200 and [s["id"] for s in sessions] == ["20261108T052707Z-account-check"]
    st, out = launcher("POST", "/api/teach/sessions", {"session_id": sessions[0]["id"],
                                                       "narration": "Check @studio_fixture's profile: posts, followers and following."})
    assert st == 200
    st, derived = launcher("POST", f"/api/teach/recordings/{out['recording_id']}/routine", {"name": "Account check"})
    assert st == 200 and derived["slots"]["handle"]["type"] == "handle"
    st, saved = launcher("GET", "/api/routines")
    assert st == 200 and [r["name"] for r in saved] == ["Account check"]
    st, confirmed = launcher("POST", f"/api/routines/{saved[0]['id']}/confirm", {})
    assert st == 200 and confirmed["confirmed_at"]
    st, reopened = launcher("GET", f"/api/routines/{saved[0]['id']}")  # what the Teach screen opens a saved workflow with
    assert st == 200 and reopened["slots"]["handle"]["type"] == "handle" and reopened["slots"] == derived["slots"]
    assert {"questions", "rules", "behaviour"} <= set(reopened)
    st, kinds = launcher("GET", "/api/tasks/kinds")
    assert st == 200 and "taught_run" in {k["kind"] for k in kinds}

    def it(text, x, y, h=56):
        return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": h}

    def look(*items):
        return {"jpeg": JPEG, "width": 1170, "height": 2532, "items": list(items), "age_ms": 30}
    first = (look(it("Studio", 585, 220), it("Your story", 160, 620)) if start == "on the taught start screen" else
             look(it("natgeo", 585, 220), it("Message", 700, 760)) | {"screen": "instagram.profile.other"})
    looks = [first,  # the look before the run: Instagram may be anywhere (live 11-08: off the start screen)
             look(it("Studio", 585, 220), it("Your story", 160, 620)) | {"screen": "instagram.home"},
             look(it("Search", 585, 230), it("For you", 200, 420), it("Accounts", 520, 420))
             | {"screen": "instagram.screen.instagram_explore_search_grid"},
             look(it("Search", 585, 230), it("Cancel", 1060, 230), it("Q W E R T Y U I O P", 585, 1900), it("space", 585, 2300))
             | {"screen": "instagram.search.typing"},
             look(it("nasa", 330, 420), it("NASA", 380, 470), it("Cancel", 1060, 230), it("nasa", 300, 230))
             | {"screen": "instagram.search.typing"},
             look(it("nasa", 585, 220), it("4,511", 450, 520), it("posts", 450, 580), it("98.1M", 700, 520),
                  it("followers", 700, 580), it("201", 950, 520), it("following", 950, 580))]
    taps, typed = [], []

    class Bridge:  # SIMULATED phonebridge: each look answers the next canned screen (its text, a real JPEG)
        simulated = True

        def devices(self):
            return [{"id": "phone", "name": "iPhone"}]

        def presence(self, device):
            return {"attached": True, "lockdown": True, "tunnel": True, "locked": None, "ios": "27.2", "foreground": IG}

        def observe(self, device):
            return looks.pop(0)

    from teachable_worker import missions

    m = types.SimpleNamespace(  # SIMULATED primitives where they touch the phone; missions' own reading helpers
        Stop=missions.Stop, NEVER=missions.NEVER, TASKS={}, _stops=lambda: [],
        _open_ig=lambda ctx, stops: ctx.observe("look"), _look=lambda ctx, note, stops: ctx.observe(note),  # the REAL Ctx
        _line=missions._line, lead=missions.lead, TAB_SCREENS=missions.TAB_SCREENS, tab_screen=missions.tab_screen,
        where=missions.where, keyboard_up=lambda items, h: any(teaching.norm(i["text"]) == "space" for i in items),
        act=strict_act(lambda kind, what, a: typed.append(a["text"]) if kind == "type" else taps.append(a | {"what": what})))
    monkeypatch.setattr(workspace, "bridge", Bridge())
    # the REAL standalone session (phonebridge-workspace's): the local lease CLI as a subprocess, missions.Hands on the
    # (dead) bridge; only the bridge's looks and missions' touch primitives are simulated
    monkeypatch.setenv("PHONEBRIDGE_STATE", str(tw / "phonebridge"))
    monkeypatch.setenv("TW_LEASE_CMD", shlex.join([sys.executable, "-m", "phonebridge.local_lease", "lease"]))
    assert workspace.session_hook() is standalone.session
    monkeypatch.setattr(teaching.TaughtRun, "_missions", staticmethod(lambda: m))
    monkeypatch.setattr(time, "sleep", lambda s: None)  # the taught pauses (their values are test_teaching's)
    st, task = launcher("POST", "/api/tasks", {"device_id": "phone", "kind": "taught_run",
                                               "input": {"routine_id": saved[0]["id"], "params": {"handle": "nasa"}}})
    assert st == 201, task
    done = wait_state(task["id"], "done", "failed", timeout=20)
    assert done["state"] == "done", done["error"]
    result = json.loads(done["result_json"])
    assert typed == ["nasa"] and result["reads"]["followers"]["value"] == "98.1M"
    assert result["reads"]["followers"]["frame_id"].startswith("frm_")  # read off a frame the task stored
