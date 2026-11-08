"""The desktop workspace's first task, "Read the screen", through the real HTTP server exactly as the page calls it,
with a simulated bridge: no phone, no network, no browser. It ships with the app (the standalone release runs it in
CI), so it imports only workspace, store and server, and checks the page's own files and calls, not a copy of them."""

import base64
import re
from pathlib import Path

import pytest

from teachable_worker import server, store, workspace

ADMIN = "admin-test-token"
WEB = Path(__file__).resolve().parents[1] / "web"
# a real 117x253 JPEG (made with ffmpeg), so whatever reads it sees an image, not a placeholder
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYwLjMxLjEwMgD/2wBDAAgQEBMQExYWFhYWFhoYGhsbGxoaGhobGxsdHR0iIiId"
    "HR0bGx0dICAiIiUmJSMjIiMmJigoKDAwLi44ODpFRVP/xABMAAEBAAAAAAAAAAAAAAAAAAAABgEBAQAAAAAAAAAAAAAAAAAAAAQQ"
    "AQAAAAAAAAAAAAAAAAAAAAARAQAAAAAAAAAAAAAAAAAAAAD/wAARCAD8AHQDASIAAhEAAxEA/9oADAMBAAIRAxEAPwCAAVpQAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH//2Q==")


class SimulatedBridge:
    """Stands in for the Mac's phonebridge (`phonebridge serve --http`); frames it produces are stored simulated=1."""
    simulated = True

    def devices(self):
        return [{"id": "iphone12", "name": "iPhone", "model": "iPhone 12"}]

    def presence(self, device):
        return {"attached": True, "lockdown": True, "tunnel": True, "locked": None, "ios": "27.2",
                "foreground": "com.burbn.instagram"}

    def frame(self, device):
        return {"jpeg": JPEG, "source": "still", "seq": None, "age_ms": None}

    def observe(self, device):
        return {"jpeg": JPEG, "width": 8, "height": 8, "age_ms": 40, "foreground": "com.burbn.instagram",
                "items": [{"text": "natgeo", "x": 1, "y": 1, "w": 4, "h": 1}]}


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("TW_LEASE_CMD", "phonebridge-local-lease lease")  # the standalone release's lease command
    monkeypatch.setattr(workspace, "bridge", SimulatedBridge())
    for name in ("_beat", "_lease"):  # the Mac heartbeat and authhub lease are deployment adapters; absent here
        if hasattr(workspace, name):
            monkeypatch.setattr(workspace, name, lambda *a: None)


def ui_calls(js: str) -> str:
    return (WEB / "assets" / js).read_text()


def test_the_page_and_its_modules_are_served(http):
    status, body = http("GET", "/")
    assert status == 200 and b"/assets/workspace.js" in body and b"/assets/workspace.css" in body
    for name in ("workspace.js", "workspace-teach.js", "workspace.css", "api.js", "ui.js"):
        s, data = http("GET", f"/assets/{name}")
        assert s == 200 and data, name
    for js in ("workspace.js", "workspace-teach.js"):
        for imp in re.findall(r'from "\./([\w.-]+)"', ui_calls(js)):
            assert (WEB / "assets" / imp).is_file(), f"{js} imports {imp}"
    assert not re.search(r"https?://", ui_calls("workspace.js") + (WEB / "index.html").read_text()), \
        "the workspace loads nothing from the network"


def test_read_the_screen_is_one_stored_look(http, bridge):
    js = ui_calls("workspace.js")
    assert "/api/devices/${encodeURIComponent(d.id)}/observe" in js and "/frame`" in js  # the calls tested below

    s, devices = http("GET", "/api/devices", token=ADMIN)
    assert s == 200 and devices[0]["connection"]["state"] == "connected", devices  # the button's precondition

    s, live = http("GET", "/api/devices/iphone12/frame", token=ADMIN)
    assert s == 200 and live[:2] == b"\xff\xd8", "the live viewport gets a JPEG"

    s, look = http("POST", "/api/devices/iphone12/observe", {}, token=ADMIN)
    assert s == 201, look
    assert look["frame_url"] == f"/api/frames/{look['frame_id']}" and look["observed_at"]
    assert [i["text"] for i in look["items"]] == ["natgeo"]  # the card's lines, with positions for the marker
    assert {"x", "y", "w", "h"} <= set(look["items"][0])

    s, stored = http("GET", look["frame_url"], token=ADMIN)
    assert s == 200 and stored[:2] == b"\xff\xd8", "the evidence frame the card shows"

    s, tasks = http("GET", "/api/tasks?device_id=iphone12", token=ADMIN)
    assert s == 200 and tasks == [], "a look is not a task: nothing queued, no lease taken"
    assert store.query("SELECT task_id, simulated FROM frames") == [{"task_id": None, "simulated": 1}]

    s, again = http("POST", "/api/devices/iphone12/observe", {}, token=ADMIN)
    assert s == 429 and again["error"], "a second look within seconds is refused with a sentence the UI shows"


def test_customers_never_reach_the_phone(http, bridge):
    s, out = http("POST", "/api/ops/orgs", {"name": "Client", "track": "agency", "cohort": "external",
                                            "person": {"name": "C"}}, token=ADMIN)
    token = out["login_link"].rsplit("/", 1)[1]
    for method, path in (("GET", "/api/devices"), ("GET", "/api/devices/iphone12/frame"),
                         ("POST", "/api/devices/iphone12/observe")):
        s, _ = http(method, path, {} if method == "POST" else None, token=token)
        assert s in (401, 403), f"{method} {path} answered a customer with {s}"


def test_server_routes_cover_the_workspace_calls():
    routes = server.routes()
    for method, path in (("GET", "/api/devices"), ("GET", "/api/devices/x/frame"), ("POST", "/api/devices/x/observe"),
                         ("GET", "/api/frames/x"), ("GET", "/api/tasks/kinds"), ("POST", "/api/tasks"),
                         ("GET", "/api/tasks"), ("GET", "/api/tasks/x"), ("POST", "/api/tasks/x/stop")):
        assert any(m == method and rx.match(path) for m, rx, _, _ in routes), f"{method} {path}"
