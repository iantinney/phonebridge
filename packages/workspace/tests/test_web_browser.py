"""The customer UI in a real browser against the real server (temp DB, test cohort), checked through the API
afterwards: what the person did in the page is what the server recorded. Opt-in, because it needs Playwright:

    uv run --with playwright pytest -q tests/test_web_browser.py   # once: uvx playwright install chromium
"""

import base64
import json
import os
import re
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright  # noqa: E402

from teachable_worker import agency, server, store  # noqa: E402

ADMIN = "admin-test-token"


@pytest.fixture
def base(tw):
    server.Handler.ROUTES = server.routes()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def page():
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # browser binary not installed
            pytest.skip(f"no Chromium for Playwright: {e}")
        # a phone's own user agent: growth drops "Headless" visits as bots, and this test checks the visit beacon
        ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, user_agent=ua)
        pg = ctx.new_page()
        pg.js_errors = []
        pg.fits = lambda: pg.evaluate("document.documentElement.scrollWidth") <= 390  # no sideways scroll on a phone
        pg.on("pageerror", lambda e: pg.js_errors.append(str(e)))
        yield pg
        browser.close()


def call(base, method, path, body=None, token=ADMIN):
    req = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def test_agency_teach_send_approve_accept(base, page, tmp_path):
    made = call(base, "POST", "/api/ops/orgs", {"name": "UI test studio", "track": "agency", "cohort": "test",
                                                "person": {"name": "Pat"}})
    token = made["login_link"].rsplit("/t/", 1)[1]
    routine = call(base, "POST", "/api/ops/routines", {"org_id": made["org"]["id"], "name": "Test routine",
                                                       "provenance": "founder_config", "summary": "Test.",
                                                       "steps": [{"text": "Open the app"}]})
    clips = []
    for name in ("one.mp4", "two.mp4"):
        (tmp_path / name).write_bytes(name.encode() * 64)
        clips.append(str(tmp_path / name))

    page.goto(base + "/jobs/?src=qa")
    expect(page.get_by_role("button", name="Ask for a pilot")).to_be_visible()
    assert page.fits()
    if agency.lane("growth"):
        page.get_by_role("button", name="Ask for a pilot").click()
        expect(page.get_by_label("Your name")).to_be_focused()
        page.get_by_label("Your name").fill("Robin")
        page.get_by_label("Work email").fill("robin@agency.example")
        page.get_by_role("button", name="Send request").click()
        expect(page.get_by_text("Thanks. We'll reply by email.")).to_be_visible()
        assert [(r["track"], r["source"]) for r in store.query("SELECT track, source FROM growth_interest")] == [("agency", "qa")]
        assert any('"agency_landing"' in e["data_json"] for e in store.query("SELECT data_json FROM events WHERE name = 'visit'"))
    page.goto(f"{base}/t/{token}")
    page.goto(base + "/jobs/#/")
    expect(page.get_by_role("heading", name="Let's get your first job done")).to_be_visible()

    page.get_by_role("link", name="Add account").click()
    page.get_by_label("Handle").fill("@ui.test.account")
    page.get_by_role("button", name="Add account").click()
    page.get_by_text("Ask me before each post").click()
    page.get_by_role("button", name="Save permission").click()
    expect(page.get_by_role("radio", name="Ask me before each post")).to_be_checked()

    page.goto(base + "/jobs/#/new")
    page.locator("#nw-files").set_input_files(clips)
    page.locator("#cap-0").fill("Exact caption")
    page.get_by_role("button", name="Use this caption for all").click()
    page.get_by_role("button", name="Send 2 jobs").click()
    expect(page.get_by_role("heading", name="2 things need you")).to_be_visible()

    page.get_by_role("link", name="Draft for @ui.test.account").first.click()
    page.get_by_role("button", name="Looks right").click()
    expect(page.get_by_text("You accepted this")).to_be_visible()
    page.goto(base + "/jobs/#/")
    page.get_by_role("link", name="Draft for @ui.test.account").first.click()
    page.get_by_role("button", name="Something's wrong").click()
    page.get_by_label("What's wrong?").fill("Use the shop hashtag")
    page.get_by_role("button", name="Send back").click()
    expect(page.get_by_text("Sent back").first).to_be_visible()

    page.goto(base + "/jobs/#/new")
    page.get_by_text("Post them").click()
    page.locator("#nw-files").set_input_files(clips[0])
    page.locator("#cap-0").fill("Before edit")
    page.get_by_role("button", name="Send 1 job").click()
    page.get_by_role("link", name="Post to @ui.test.account").first.click()
    expect(page.get_by_text("@ui.test.account on TikTok").first).to_be_visible()
    page.get_by_role("button", name="Change caption").click()
    page.locator("#cap-edit").fill("After edit")
    page.get_by_role("button", name="Save caption").click()
    expect(page.locator(".panel", has_text="After edit").first).to_be_visible()
    page.get_by_role("button", name="Approve this post").click()
    expect(page.get_by_text("Approved, in line").first).to_be_visible()
    assert page.fits()

    jobs = call(base, "GET", "/api/jobs", token=token)
    by = {(j["kind"], j["status"]) for j in jobs}
    assert by == {("prepare", "accepted"), ("prepare", "rejected"), ("publish", "queued")}, by
    pub = next(j for j in jobs if j["kind"] == "publish")
    assert pub["input"]["caption"] == "After edit" and pub["approved_sha"] == pub["content_sha"]
    assert all(j["origin"] == "customer" for j in jobs)
    r = call(base, "GET", f"/api/routines/{routine['id']}", token=token)
    assert [c["text"] for c in r["corrections"]] == ["Use the shop hashtag"]
    assert not page.js_errors


def test_game_review_loop_and_deletion(base, page, monkeypatch):
    coaching = agency.lane("coaching")
    if coaching is None:
        pytest.skip("coaching lane not installed in this checkout")
    dims = list(coaching.DIMENSIONS)

    voiced = {t: f"({t})" for t in coaching.TONES}

    def stand_in(prompt, schema):  # answers in the real schema; this checks the page, not coaching quality
        if "changes" in schema["properties"]:  # coaching's reconcile pass: nothing to reconcile
            return {"changes": []}
        if "lessons" in schema["properties"]:
            # coaching keeps a lesson only when the marks support it: clarity helps on every message here
            return {"lessons": [{"kind": "keep", "title": "Answer, then ask", "why": voiced, "practice": "p",
                                 "focus": "clarity", "turns": [1]}], "summary": voiced}
        return {"dimensions": {d: {"mark": "helps" if d == "clarity" else "neutral", "note": "n", "evidence": ""}
                               for d in dims}, "reason": voiced, "alternative_kind": "rewrite",
                "alternative": "alt", "alternative_why": "why"}

    monkeypatch.setitem(coaching.PROVIDERS, "stand_in", stand_in)
    monkeypatch.setenv("TW_COACH_PROVIDER", "stand_in")

    page.goto(base + "/review/")
    page.get_by_role("button", name="Try a made-up conversation").click()
    expect(page.get_by_role("heading", name="Message by message")).to_be_visible(timeout=15000)
    page.wait_for_function("!document.body.innerText.includes('This page updates by itself')", timeout=30000)
    assert page.fits()
    page.get_by_role("button", name="Encouraging").click()
    expect(page.get_by_text("(encouraging)").first).to_be_visible()
    page.get_by_role("button", name="Answer, then ask").click()
    expect(page.get_by_text("Your lesson for next time")).to_be_visible()
    rid = page.evaluate("location.hash.split('/')[2]")
    page.get_by_role("button", name="Delete this review").click()
    page.get_by_role("button", name="Yes, delete").click()
    expect(page.get_by_role("heading", name="Go through your messages one at a time.")).to_be_visible()

    cookie = next(c["value"] for c in page.context.cookies() if c["name"] == "tw")
    with pytest.raises(urllib.error.HTTPError) as gone:
        call(base, "GET", f"/api/coach/reviews/{rid}", token=cookie)
    assert gone.value.code == 404

    # their own drafts: nothing is sent until processing consent and age are ticked
    page.get_by_role("button", name="Keep the conversation going").click()
    page.get_by_role("button", name="My drafts").click()
    page.get_by_label("Paste your messages").fill("so what's the best thing you've eaten here?\n\nok you're picking the restaurant")
    page.get_by_role("button", name="Review my messages").click()
    expect(page.get_by_text("Read how your messages are processed")).to_be_visible()
    assert call(base, "GET", "/api/coach/reviews", token=cookie) == []
    page.get_by_label("I agree to my messages being processed this way").check()
    page.get_by_label("I'm 18 or older, and so is everyone in the conversation").check()
    page.get_by_role("button", name="Review my messages").click()
    expect(page.get_by_role("heading", name="Keep the conversation going")).to_be_visible(timeout=15000)
    mine = call(base, "GET", "/api/coach/reviews", token=cookie)
    assert len(mine) == 1 and not mine[0]["synthetic"]
    detail = call(base, "GET", f"/api/coach/reviews/{mine[0]['id']}", token=cookie)
    assert detail["consent"]["processing"] and [i["who"] for i in detail["items"]] == ["me", "me"]
    assert not page.js_errors


TINY_JPEG = base64.b64decode("/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYwLjMxLjEwMgD/2wBDAAgQEBMQExYWFhYWFhoYGhsbGxoaGhobGxsdHR0iIiIdHR0bGx0dICAiIiUmJSMjIiMmJigoKDAwLi44ODpFRVP/xABMAAEBAAAAAAAAAAAAAAAAAAAABgEBAQAAAAAAAAAAAAAAAAAAAAQQAQAAAAAAAAAAAAAAAAAAAAARAQAAAAAAAAAAAAAAAAAAAAD/wAARCAD8AHQDASIAAhEAAxEA/9oADAMBAAIRAxEAPwCAAVpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH//2Q==")


class SimulatedBridge:
    """A stand-in for the Mac's phonebridge (workspace.bridge): a real tiny JPEG and OCR lines. Frames it produces
    are stored simulated=1, so nothing here can be mistaken for the phone."""
    simulated = True

    def devices(self):
        return [{"id": "iphone12", "name": "iPhone", "model": "iPhone 12"}]

    def presence(self, device):
        return {"attached": True, "lockdown": True, "tunnel": True, "locked": None, "ios": "27.2",
                "foreground": "com.burbn.instagram"}

    def frame(self, device):
        return {"jpeg": TINY_JPEG, "source": "still", "seq": None, "age_ms": None}

    def observe(self, device):
        return {"jpeg": TINY_JPEG, "width": 117, "height": 253, "age_ms": 40, "foreground": "com.burbn.instagram",
                "items": [{"text": "natgeo", "x": 40, "y": 20, "w": 30, "h": 8},
                          {"text": "268M followers", "x": 40, "y": 32, "w": 50, "h": 8}]}


def test_workspace_read_the_screen_is_an_immediate_look(base, page, monkeypatch):
    workspace = agency.lane("workspace")
    if workspace is None:
        pytest.skip("workspace lane not installed in this checkout")
    monkeypatch.setattr(workspace, "bridge", SimulatedBridge())
    monkeypatch.setattr(workspace, "_beat", lambda: None, raising=False)
    monkeypatch.setattr(workspace, "_lease", lambda d: None, raising=False)

    page.goto(f"{base}/t/{ADMIN}")
    page.goto(base + "/")
    expect(page.get_by_role("heading", name="iPhone 12")).to_be_visible(timeout=15000)
    expect(page.get_by_text("Live · updated")).to_be_visible(timeout=15000)
    assert not page.get_by_role("radio", name="Read the screen").count()  # not a queued recipe
    page.get_by_role("button", name="Read the screen").click()
    expect(page.get_by_role("heading", name="What's on the screen")).to_be_visible()
    expect(page.locator(".fresh")).to_contain_text("Screen read at")  # not "from the recording": it's a live look
    page.get_by_role("button", name="268M followers").click()
    expect(page.locator(".tapmark")).to_be_visible()
    assert page.fits()

    assert call(base, "GET", "/api/tasks?device_id=iphone12") == []  # no task, no queue, no lease
    frames = store.query("SELECT task_id, simulated FROM frames")
    assert frames == [{"task_id": None, "simulated": 1}], frames
    assert not page.js_errors


class DemoLook:
    """A read-only stand-in executor: two stored looks and a tap between them, through the real workspace Ctx."""
    read_only, input, route = True, {"handle": "the profile to look at"}, "test"

    def available(self):
        return True, ""

    def run(self, ctx):
        with ctx.session():
            first = ctx.observe("Looking at the search screen")
            ctx.step("tap", "Tapped the first result", frame_id=first["frame_id"], pace_ms=420, x=40, y=20, width=117, height=253)
            second = ctx.observe("The profile opened")
        return {"profile": {"handle_observed": ctx.input.get("handle"), "frame_id": second["frame_id"],
                            "observed_at": second["captured_at"]}}


class RealShapedBridge(SimulatedBridge):
    """For the UI path only, in this test's temporary DB: frames stored as not simulated, because teaching (rightly)
    never lists a run on a simulated phone as a demonstration. That guard is checked below with SimulatedBridge."""
    simulated = False


def test_do_it_once_then_teach_from_that_task(base, page, monkeypatch):
    workspace, teaching = agency.lane("workspace"), agency.lane("teaching")
    if workspace is None or teaching is None or not hasattr(teaching, "workspace_runs"):
        pytest.skip("workspace and teaching-from-workspace-runs aren't both installed in this checkout")
    import contextlib
    monkeypatch.setattr(workspace, "_beat", lambda: None, raising=False)
    monkeypatch.setattr(workspace, "_lease", lambda d: None, raising=False)
    monkeypatch.setattr(workspace, "session_hook", lambda: (lambda ctx: contextlib.nullcontext()))
    monkeypatch.setattr(workspace, "executors", lambda: {"demo_look": DemoLook()})
    made = call(base, "POST", "/api/ops/orgs", {"name": "Operator (test)", "track": "agency", "cohort": "founder",
                                                "person": {"name": "Operator", "role": "founder"}})
    token = made["login_link"].rsplit("/t/", 1)[1]
    page.goto(f"{base}/t/{token}")

    def run_demo():
        page.goto(base + "/")
        expect(page.get_by_role("heading", name="iPhone 12")).to_be_visible(timeout=15000)
        page.get_by_role("radio", name="Demo look").click()
        page.get_by_label("Handle").fill("natgeo")
        page.get_by_role("button", name="Start").click()
        expect(page.get_by_text("What it found")).to_be_visible(timeout=20000)  # the answer, above the steps

    monkeypatch.setattr(workspace, "bridge", SimulatedBridge())  # a simulated run is never offered for teaching
    run_demo()
    expect(page.get_by_text("Simulated, not the phone")).to_be_visible()
    assert not page.get_by_role("button", name="Teach a workflow from this task").count()

    monkeypatch.setattr(workspace, "bridge", RealShapedBridge())
    run_demo()
    page.get_by_role("button", name="Teach a workflow from this task").click()
    expect(page.get_by_role("heading", name="Teach a workflow")).to_be_visible()
    expect(page.locator("#teach-session")).to_have_value(re.compile(r"^tsk_"))
    page.get_by_label("Name").fill("Look up a profile")
    page.get_by_label("Describe the job").fill("Search for the handle I give you and open its profile. Read only.")
    page.get_by_role("button", name="Learn this workflow").click()
    expect(page.get_by_text("Steps ·")).to_be_visible(timeout=60000)
    assert page.fits()

    routines = call(base, "GET", "/api/routines", token=token)
    assert [r["name"] for r in routines] == ["Look up a profile"], routines
    assert not page.js_errors


class KeepsLooking:
    """A read-only stand-in that works until stopped; `ctx.check()` raises once a stop or hand-off is requested."""
    read_only, input, route = True, {}, "test"

    def available(self):
        return True, ""

    def run(self, ctx):
        with ctx.session():
            ctx.step("note", "Looking around")
            ctx.step("note", "StaleInput: frame_age_ms=1840 > 1500 (seq 4411, tap@(351,2420)) after 3 tries; timings {'look': 812}")
            ctx.step("type", "step 4: On Ci bearch with Meta Al, type the handle you give me (taught pause 2.357 s, drawn 3.1 s)")
            for _ in range(300):
                time.sleep(0.05)
                ctx.check()
        return {}


def test_take_over_waits_for_the_phone_to_be_released(base, page, monkeypatch):
    workspace = agency.lane("workspace")
    if workspace is None:
        pytest.skip("workspace lane not installed in this checkout")
    import contextlib

    @contextlib.contextmanager
    def slow_release(ctx):  # like the real session: letting go of the phone takes a moment after the stop
        try:
            yield
        finally:
            time.sleep(1.5)

    monkeypatch.setattr(workspace, "bridge", SimulatedBridge())
    monkeypatch.setattr(workspace, "_beat", lambda: None, raising=False)
    monkeypatch.setattr(workspace, "_lease", lambda d: None, raising=False)
    monkeypatch.setattr(workspace, "session_hook", lambda: slow_release)
    monkeypatch.setattr(workspace, "executors", lambda: {"keeps_looking": KeepsLooking()})

    page.goto(f"{base}/t/{ADMIN}")
    page.goto(base + "/")
    page.get_by_role("radio", name="Keeps looking").click()
    page.get_by_role("button", name="Start").click()
    take_over = page.locator(".stage-foot").get_by_role("button", name="Take over the phone")
    expect(take_over).to_be_visible(timeout=15000)
    # the step list starts collapsed (the answer leads); opened, a driver's refusal reads as one plain line
    steps = page.locator(".steps-box > summary")
    expect(steps).to_be_visible(timeout=10000)
    expect(page.get_by_text("The phone refused an input based on an old screen")).to_be_hidden()
    steps.click()
    expect(page.get_by_text("The phone refused an input based on an old screen")).to_be_visible()
    # a taught step: the step and what it does; the screen as read and its timings only under Details
    expect(page.get_by_text("Step 4 · Type the handle you give me")).to_be_visible()
    expect(page.get_by_text("Ci bearch", exact=False)).to_be_hidden()
    expect(page.get_by_text("frame_age_ms=1840", exact=False)).to_be_hidden()
    take_over.click()

    # requested, not done: one control, disabled, saying what it waits for; nobody is told the phone is free yet
    expect(page.locator(".stage-foot").get_by_role("button", name="Handing back…")).to_be_disabled()
    expect(page.locator(".toast")).to_contain_text("Handing the phone back")
    assert not page.get_by_role("button", name="Take over the phone").count()
    assert "The phone is yours" not in page.locator("body").inner_text()

    expect(page.locator(".toast")).to_contain_text("The phone is yours now", timeout=15000)
    # the stage stops saying "working" at once, not at the next device poll
    expect(page.locator(".stage-foot").get_by_role("button", name="Read the screen")).to_be_visible(timeout=1500)
    expect(page.locator(".taskbox .state.handed_off")).to_be_visible()
    tasks = call(base, "GET", "/api/tasks?device_id=iphone12")
    assert [t["state"] for t in tasks] == ["handed_off"], tasks
    assert not page.js_errors


class TaughtRunStandIn:
    """Only its input schema matters here: the raw fields a person must never be asked to type."""
    read_only, input, route = True, {"routine_id": "the routine to run", "params": "JSON object of slot values"}, "test"

    def available(self):
        return True, ""

    def run(self, ctx):
        return {}


def test_saved_workflows_are_chosen_not_typed(base, page, monkeypatch):
    workspace = agency.lane("workspace")
    if workspace is None:
        pytest.skip("workspace lane not installed in this checkout")
    monkeypatch.setattr(workspace, "bridge", SimulatedBridge())
    monkeypatch.setattr(workspace, "_beat", lambda: None, raising=False)
    monkeypatch.setattr(workspace, "_lease", lambda d: None, raising=False)
    monkeypatch.setattr(workspace, "executors", lambda: {"taught_run": TaughtRunStandIn()})
    page.goto(f"{base}/t/{ADMIN}")
    page.goto(base + "/")
    page.get_by_role("radio", name="Run a saved workflow").click()
    expect(page.get_by_role("tab", name="Teach")).to_have_attribute("aria-selected", "true")
    expect(page.locator(".teach")).to_be_visible()
    assert not page.get_by_label(re.compile("routine", re.I)).count() and not page.get_by_label(re.compile("params", re.I)).count()

    t = call(base, "POST", "/api/tasks", {"device_id": "iphone12", "kind": "taught_run",
                                          "input": {"routine_id": "rtn_0123456789ab", "params": {"handle": "examplestudio"}}})
    page.get_by_role("tab", name="Run").click()
    page.locator(".narrow-tasks").get_by_role("button", name=re.compile("examplestudio")).click()  # phone width: the sidebar folds into this list
    asked = page.locator(".asked")
    expect(asked).to_contain_text("@examplestudio")
    expect(asked).to_contain_text("Workflow")
    assert "[object Object]" not in page.locator(".taskbox").inner_text()
    assert "rtn_0123456789ab" not in asked.inner_text()  # the id stays under Details
    assert t["id"]
    assert not page.js_errors


def test_saved_workflow_run_form_catches_up_with_late_kinds(base, page):
    """UI-only (API answers stubbed in the browser): Teach opened before /api/tasks/kinds answers must not be stuck
    on 'no runner'; when kinds arrive, the run form updates and keeps what was typed and the open workflow."""
    routine = {"id": "rtn_x", "name": "Look up a profile", "confirmed_at": "2026-11-08T06:55:20+00:00", "version": 3,
               "recording_id": None, "steps": [{"text": "Search for the handle", "param": "handle"}],
               "slots": {"handle": {"type": "handle", "example": "natgeo"}}, "questions": [], "rules": []}
    device = [{"id": "iphone12", "name": "iPhone", "model": "iPhone 12", "connection": {"state": "connected"},
               "frame": {"available": False, "reason": "test"}, "task_id": None}]
    kinds = [{"kind": "taught_run", "available": True, "reason": "", "read_only": True,
              "input": {"routine_id": "x", "params": "y"}}]

    held = []  # the race: kinds answer only after the person has already opened the workflow

    def stub(route):
        path = "/" + route.request.url.split("://", 1)[1].split("/", 1)[1].split("?")[0]
        body = {"/api/devices": device, "/api/routines": [routine], "/api/teach/routines/rtn_x": routine,
                "/api/tasks": [], "/api/teach/sessions": []}.get(path)
        if path == "/api/tasks/kinds":
            return held.append(route)
        if body is None:
            return route.continue_()
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

    page.route("**/api/**", stub)
    page.goto(f"{base}/t/{ADMIN}")
    page.goto(base + "/")
    page.get_by_role("tab", name="Teach").click()
    page.get_by_role("button", name="Look up a profile").click()
    run = page.locator(".teach-run")
    expect(run).to_contain_text("Checking what this phone can do")
    run.get_by_label("Handle").fill("nasa")
    page.locator("body").click(position={"x": 5, "y": 5})  # the cursor leaves the field: the form may now redraw
    for r in held:
        r.fulfill(status=200, content_type="application/json", body=json.dumps(kinds))
    held.clear()
    page.wait_for_timeout(500)
    expect(run.get_by_role("button", name="Run on the phone")).to_be_enabled(timeout=10000)
    expect(page.locator(".teach-run").get_by_label("Handle")).to_have_value("nasa")
    expect(page.get_by_role("heading", name="Look up a profile")).to_be_visible()
    assert not page.js_errors


# The exact shape of a real taught_run task (workspace task view of a done run, Nov 8), SANITIZED: a fictional handle,
# made-up counts and ids, generic step texts. Keys, nesting and value types are as the runner returned them.
TAUGHT_DONE = {
    "id": "tsk_0000000000aa", "device_id": "iphone12", "kind": "taught_run", "state": "done", "needs": None,
    "origin": "operator", "requested_by": "per_000000000001", "created": "2026-11-08T08:18:05+00:00",
    "updated": "2026-11-08T08:18:39+00:00", "started": "2026-11-08T08:18:06+00:00", "finished": "2026-11-08T08:18:39+00:00",
    "stop_requested": False, "handoff_note": None, "error": None, "session_open": 0, "not_before": None,
    "input": {"routine_id": "rtn_000000000abc", "params": {"handle": "example.creator"}},
    "execution": {"route": "phone", "executor": "teachable_worker.teaching.TaughtRun", "simulated": False},
    "result": {
        "routine": {"id": "rtn_000000000abc", "version": 1, "provenance": "operator_demo"},
        "params": {"handle": "example.creator"}, "status": "done",
        "reads": {k: {"value": v, "frame_id": "frm_00000000000f", "observed_at": "2026-11-08T08:18:38+00:00", "runtime_said": None}
                  for k, v in (("posts", "128"), ("followers", "3.4K"), ("following", "1,234"))},
        "handle_shown": True, "result_frame": {"frame_id": "frm_00000000000f", "observed_at": "2026-11-08T08:18:38+00:00"},
        "steps_by_tier": {"taught": 4}, "learned_steps_used": 4,
        "pace": [{"step": i, "taught_ms": 400 + i, "drawn_ms": 600 + i} for i in range(1, 5)],
        "limits": ["pace taught by an agent's or synthetic run (not a person's pace)", "screens named by the recording or matched by text"],
    },
    "steps": [{"idx": i, "at": "2026-11-08T08:18:%02d+00:00" % (10 + i), "kind": k, "text": txt, "frame_id": None, "frame_url": None,
               "pace_ms": None, "data": {"phase": ph} if ph else {}}
              for i, (k, txt, ph) in enumerate([("note", "Taught routine started", None), ("tap", "Back", "start"),
                                                ("tap", "Search", "start"), ("tap", "step 1: tap Search", "taught"),
                                                ("type", "step 2: type the handle", "taught"), ("tap", "step 3: open the profile", "taught"),
                                                ("tap", "step 4: read the counts", "taught"), ("result", "Done", None)])],
    "messages": [],
}


def stub_workspace(page, task, extra=None):
    device = [{"id": "iphone12", "name": "iPhone", "model": "iPhone 12", "connection": {"state": "connected"},
               "frame": {"available": False, "reason": "test"}, "task_id": None}]
    row = {k: v for k, v in task.items() if k != "steps"}

    def stub(route):
        path = "/" + route.request.url.split("://", 1)[1].split("/", 1)[1].split("?")[0]
        if extra and extra(route, path):
            return
        if path.startswith("/api/frames/"):
            return route.fulfill(status=200, content_type="image/jpeg", body=TINY_JPEG)
        body = {"/api/devices": device, "/api/tasks/kinds": [], "/api/tasks": [row], f"/api/tasks/{task['id']}": task,
                "/api/routines/rtn_000000000abc": {"id": "rtn_000000000abc", "name": "Account check (taught)"}}.get(path)
        if body is None:
            return route.fulfill(status=404, content_type="application/json", body='{"error": "not in this stub"}')
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

    page.route("**/api/**", stub)


def test_taught_run_result_reads_as_a_profile_card(base, page):
    page.set_viewport_size({"width": 1440, "height": 900})
    stub_workspace(page, TAUGHT_DONE)
    page.goto(f"{base}/t/{ADMIN}")
    page.goto(base + "/")
    card = page.locator(".profile")
    expect(card).to_contain_text("@example.creator")
    for value, label in (("128", "Posts"), ("3.4K", "Followers"), ("1,234", "Following")):
        metric = card.locator(".metric", has_text=label)
        expect(metric.locator(".metric-value")).to_have_text(value)
        assert metric.locator(".metric-value").evaluate("e => e.getBoundingClientRect().height") < 40  # one line
    expect(card).to_contain_text("4 taught inputs, 2 to reach its start screen")
    expect(card.get_by_role("button", name="View evidence")).to_have_count(1)
    shown = card.inner_text()
    for raw in ("rtn_", "frm_", "runtime_said", "operator_demo", "taught_ms", "[object Object]"):
        assert raw not in shown.split("Details")[0], raw  # only under Details
    assert page.locator(".panel").evaluate("e => e.scrollWidth <= e.clientWidth + 1"), "the right column overflows"
    card.get_by_role("button", name="View evidence").click()
    expect(page.locator(".fresh")).to_contain_text("The profile as read at")
    if os.environ.get("TW_SHOTS"):  # for review: the card on the sanitized fixture (never real account data)
        page.screenshot(path=os.path.join(os.environ["TW_SHOTS"], "taught-result-card-1440-FIXTURE.png"))
    expect(page.locator(".steps-box")).not_to_have_attribute("open", "")  # the steps stay folded under the answer
    assert not page.js_errors


def test_a_task_that_ends_between_two_polls_shows_its_end(base, page):
    """workspace.wait_task wakes on a change from the state it began with. If the run writes its final step, the UI
    gets that (still 'running'), and the state flips before the next poll begins, a long wait never wakes. Here the
    stub holds every long wait (as the server would) and answers short polls with the finished task."""
    running = {**TAUGHT_DONE, "state": "running", "finished": None, "result": None}
    held = []

    def extra(route, path):
        if path != f"/api/tasks/{TAUGHT_DONE['id']}":
            return False
        url = route.request.url
        if "after=-1" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(running))
        elif "wait=0" in url:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({**TAUGHT_DONE, "steps": []}))
        else:
            held.append(route)  # a long wait that would sit out its full timeout
        return True

    stub_workspace(page, running, extra)
    page.goto(f"{base}/t/{ADMIN}")
    page.goto(base + "/")
    expect(page.locator(".taskbox .state.done")).to_be_visible(timeout=3000)
    expect(page.locator(".taskbox")).not_to_contain_text("Stop")
    assert not page.js_errors
