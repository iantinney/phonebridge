"""The `browse` task kind (M5 autopilot): one session-director visit on the feed under the task API's rules (leases,
journal, stop texts, park), read-only. The director itself is lane director's (`phonebridge.director`); here a
scripted stand-in takes its seat through the flow's `visit` seam."""

# ruff: noqa: F811  (fixtures imported from test_run_taskapi are redefined as test parameters, by design)
import asyncio

import pytest

from phonebridge import bridge as bridge_mod
from phonebridge.run import taskapi
from phonebridge.run.journal import Journal
from test_run_taskapi import IG_FEED, IG_PROFILE, IG_STATUS, STOP, drain, run, task  # noqa: F401
from test_run_taskapi import backend, clock, make  # noqa: F401 (fixtures)
from test_bridge import authhub, tiny_png  # noqa: F401

TT_PROFILE = [{"text": "samplemaker42", "x": 500, "y": 200, "w": 200, "h": 40, "conf": 1.0},
              {"text": "Followers", "x": 600, "y": 450, "w": 120, "h": 30, "conf": 1.0},
              {"text": "Following", "x": 400, "y": 450, "w": 120, "h": 30, "conf": 1.0},
              {"text": "Home", "x": 90, "y": 2440, "w": 70, "h": 24, "conf": 1.0},
              {"text": "Profile", "x": 1050, "y": 2440, "w": 80, "h": 24, "conf": 1.0}]
TT_FEED = [{"text": "Following", "x": 400, "y": 230, "w": 150, "h": 40, "conf": 1.0},
           {"text": "For You", "x": 620, "y": 230, "w": 150, "h": 40, "conf": 1.0}]


def feed_markers(app, items):
    words = {"instagram": ("your story",), "tiktok": ("for you",)}[app]
    return any(w in i["text"].lower() for i in items for w in words)


class Visit:
    """A stand-in director: looks and flicks `posts` times through the phone it is handed, then reports."""
    calls: list = []

    def __init__(self, posts=3, raise_at=None):
        self.posts, self.raise_at = posts, raise_at

    async def __call__(self, phone, device, *, app, identity, minutes, evidence, lease, knobs=None):
        Visit.calls.append({"app": app, "identity": identity, "minutes": minutes, "evidence": evidence, "lease": lease,
                            "knobs": knobs or {}})
        for n in range(self.posts):
            await phone.screen(device)
            await phone.ocr(device)
            if self.raise_at == n:
                raise self.exc
            await phone.swipe(device, 585, 1900, 600, 700, 0.25)
        return {"status": "ok", "label": "done (read-only)", "seconds": minutes * 60.0, "writes": 0,
                "counts": {"posts": self.posts, "skipped": 1, "flicks": self.posts, "habits": 0, "writes": 0}}


@pytest.fixture
def director(monkeypatch):
    Visit.calls = []
    monkeypatch.setattr(taskapi.Browse, "on_feed", staticmethod(feed_markers))
    v = Visit()
    monkeypatch.setattr(taskapi.Browse, "visit", v)
    return v


@pytest.fixture
def screens(monkeypatch, backend):
    """OCR from a script: `seq` is consumed one look at a time, then `then` forever."""
    state = {"seq": [], "then": IG_FEED}

    def recognize(png, fast=False):
        return state["seq"].pop(0) if state["seq"] else state["then"]
    monkeypatch.setattr(bridge_mod.vision, "recognize", recognize)
    monkeypatch.setattr(taskapi.vision, "recognize", recognize)
    backend.current = tiny_png(1170, 2532, b"\x80\x80\x80")
    return state


def browse(identity="ig:public", minutes=10, **kw):
    return task(kind="browse", payload={"minutes": minutes}, **kw) | {"identity": identity}


def test_a_visit_opens_the_app_runs_the_director_read_only_and_goes_home(make, backend, director, screens):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        events, r = await drain(api, tid)
        assert r["status"] == "done", r["reason"]
        assert r["result"]["counts"]["posts"] == 3 and r["result"]["writes"] == 0
        kinds = [c[0] for c in backend.calls]
        assert kinds[0] == "launch" and kinds[-1] == "home" and kinds.count("swipe") == 3 and "tap" not in kinds
        [call] = Visit.calls
        assert (call["app"], call["identity"], call["minutes"]) == ("instagram", "ig:public", 10)
        assert call["lease"] == "callertok" and call["evidence"] == api.runs_dir / tid / "director"
        assert [e["step"] for e in events if e.get("outcome") == "ok"] == ["open", "feed", "home"]
    run(go())


def test_a_visit_finds_the_feed_from_where_the_last_task_left_the_app(make, backend, director, screens):
    """After the day's look-only read IG reopens on Account Status: back out to a tab screen, then Home tab."""
    screens["seq"] = [IG_STATUS] * 4 + [IG_PROFILE]  # lock check, before open, open, feed_before; then one back

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        events, r = await drain(api, tid)
        assert r["status"] == "done", r["reason"]
        inputs = [c for c in backend.calls if c[0] in ("tap", "swipe")]
        back = inputs[0]
        assert back[0] == "swipe" and back[2][0][0] < 40 and back[2][-1][0] > 600  # an edge-back from the left edge
        tab = inputs[1]
        assert tab[0] == "tap" and tab[2] < 250 and tab[3] > 2250  # IG's Home tab, bottom left (never "+", top left)
    run(go())


def test_tiktok_taps_its_home_label_from_the_profile(make, backend, director, screens):
    screens["seq"] = [TT_PROFILE] * 4
    screens["then"] = TT_FEED
    backend.front = "com.zhiliaoapp.musically"

    async def go():
        api = make()
        api.verify = lambda token, identity: 600.0 if (identity, token) in (
            ("device:iphone12", "callertok"), ("tiktok:public", "accttok")) else 1 / 0
        tid = (await api.submit("iphone12", browse("tiktok:public")))["task_id"]
        events, r = await drain(api, tid)
        assert r["status"] == "done", r["reason"]
        tab = next(c for c in backend.calls if c[0] == "tap")
        assert abs(tab[2] - 90) < 50 and abs(tab[3] - 2440) < 30  # OCR boxes are tap-ready centres
        assert Visit.calls[0]["app"] == "tiktok"
    run(go())


def test_never_a_gesture_on_a_composer_and_no_director_off_the_feed(make, backend, director, screens):
    screens["then"] = [{"text": "Share", "x": 585, "y": 2300, "w": 900, "h": 80, "conf": 1.0}]

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        events, r = await drain(api, tid)
        assert r["status"] == "failed" and "posting screen" in r["reason"]
        assert not [c for c in backend.calls if c[0] in ("tap", "swipe")] and not Visit.calls
    run(go())


def test_a_stop_text_mid_visit_stops_the_task_and_goes_home(make, backend, director, screens):
    screens["seq"] = [IG_FEED] * 5 + [[{"text": "Try Again Later", "x": 100, "y": 900, "w": 400, "h": 50, "conf": 1.0}]]

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "stopped" and "Try Again Later" in r["reason"]
        assert [c[0] for c in backend.calls][-1] == "home"
    run(go())


def test_a_vm_park_mid_visit_stops_at_the_next_look_and_resume_runs_a_fresh_visit(make, backend, director, screens):
    async def go():
        api = make()
        director.posts = 50
        tid = (await api.submit("iphone12", browse()))["task_id"]
        while not any(c[0] == "swipe" for c in backend.calls):
            await asyncio.sleep(0.01)
        await api.park(tid)
        _, r = await drain(api, tid)
        assert r["status"] == "parked"
        n = sum(c[0] == "swipe" for c in backend.calls)
        assert n < 50
        await asyncio.sleep(0.05)
        assert sum(c[0] == "swipe" for c in backend.calls) == n  # nothing after the park
    run(go())


def test_the_lease_copy_running_out_parks_the_visit(make, backend, director, screens, clock):
    async def go():
        api = make()
        director.posts = 50
        tid = (await api.submit("iphone12", browse() | {"lease_s": 5}))["task_id"]
        while not any(c[0] == "swipe" for c in backend.calls):
            await asyncio.sleep(0.01)
        clock.t += 60  # the VM stopped renewing (tunnel down)
        _, r = await drain(api, tid)
        assert r["status"] == "parked" and "lease copy expired" in r["reason"]
    run(go())


@pytest.mark.parametrize("minutes", [0, -1, "ten", None, 90])
def test_a_visit_needs_sane_minutes(make, minutes):
    async def go():
        api = make()
        with pytest.raises(taskapi.Refused):
            await api.submit("iphone12", task(kind="browse", payload={"minutes": minutes}))
    run(go())


def test_a_director_that_stops_itself_fails_the_task_with_its_reason(make, backend, monkeypatch, screens):
    async def stopped(phone, device, **kw):
        await phone.screen(device)
        return {"status": "stopped", "reason": "unknown: the feed didn't move after 3 looks", "label": "stopped",
                "seconds": 12.0, "writes": 0, "counts": {"posts": 3}}
    monkeypatch.setattr(taskapi.Browse, "on_feed", staticmethod(feed_markers))
    monkeypatch.setattr(taskapi.Browse, "visit", staticmethod(stopped))

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "failed" and "didn't move" in r["reason"]
    run(go())


def test_the_real_director_drives_a_short_read_only_visit(make, backend, screens, monkeypatch, tmp_path):
    """The seam against lane director's v1 (skipped until it is merged): a few seconds on a moving feed, flicks only."""
    pytest.importorskip("phonebridge.director.session")
    import phonebridge.director.persona as persona_mod
    monkeypatch.setattr(persona_mod, "ROOT", tmp_path / "personas")
    backend.animate = True  # a video playing: every look differs, so the feed is never "stuck"
    looks = []

    async def look(self, device, platform=None, expect=None, settle=True):  # fast-eyes' look (no stream in FakeBackend)
        looks.append(platform)
        return {"screen": "instagram.home", "settled": True, "frame_age_ms": 20, "items": IG_FEED}
    monkeypatch.setattr(bridge_mod.Bridge, "look", look, raising=False)

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse(minutes=0.1)))["task_id"]
        _, r = await drain(api, tid, timeout=60)
        assert r["status"] == "done", r["reason"]
        assert r["result"]["writes"] == 0 and r["result"]["counts"]["posts"] >= 1
        assert "tap" not in [c[0] for c in backend.calls]  # read-only: no touch on a post
        assert (api.runs_dir / tid / "director" / "events.jsonl").exists()
        assert looks and set(looks) == {"instagram"}  # live visits gate gestures on the daemon's look (S3)
    run(go())


def loose_markers(app, items):
    """Like lane director's FEED_MARKERS: substrings ("follow" is in "followers", "Follow and invite friends")."""
    words = {"instagram": ("your story", "follow"), "tiktok": ("for you", "friends", "live")}[app]
    return any(w in i["text"].lower() for i in items for w in words)


def test_a_profile_is_never_the_feed_even_when_the_marker_check_says_so(make, backend, director, screens, monkeypatch):
    monkeypatch.setattr(taskapi.Browse, "on_feed", staticmethod(loose_markers))
    screens["seq"] = [IG_PROFILE] * 4  # lock check, before open, open, feed_before; then the feed

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "done", r["reason"]
        inputs = [c for c in backend.calls if c[0] in ("tap", "swipe")]
        assert inputs[0][0] == "tap" and inputs[0][2] < 250 and inputs[0][3] > 2250  # Home tab first, then the visit
    run(go())


def test_a_composer_with_feed_words_is_still_a_composer(make, backend, director, screens, monkeypatch):
    monkeypatch.setattr(taskapi.Browse, "on_feed", staticmethod(loose_markers))
    screens["then"] = [{"text": "Your story", "x": 300, "y": 800, "w": 200, "h": 40, "conf": 1.0},
                       {"text": "Share", "x": 585, "y": 2300, "w": 900, "h": 80, "conf": 1.0}]

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "failed" and "posting screen" in r["reason"]
        assert not [c for c in backend.calls if c[0] in ("tap", "swipe")] and not Visit.calls
    run(go())


def test_an_unrecognised_screen_gets_no_gesture(make, backend, director, screens):
    screens["then"] = [{"text": "Something new and unknown", "x": 500, "y": 900, "w": 600, "h": 40, "conf": 1.0},
                       {"text": "Continue", "x": 585, "y": 2200, "w": 300, "h": 60, "conf": 1.0}]

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "failed" and "unrecognised" in r["reason"]
        assert not [c for c in backend.calls if c[0] in ("tap", "swipe")] and not Visit.calls
    run(go())


def test_no_input_after_a_park_even_between_two_looks(make, backend, screens, monkeypatch):
    async def parks_then_flicks(phone, device, **kw):
        await phone.screen(device)
        phone.ctx.api.live[phone.tid]["park"] = True  # the VM parks right after the look
        await phone.swipe(device, 585, 1900, 600, 700, 0.25)
        return {"status": "ok", "counts": {"posts": 1}}
    monkeypatch.setattr(taskapi.Browse, "on_feed", staticmethod(feed_markers))
    monkeypatch.setattr(taskapi.Browse, "visit", staticmethod(parks_then_flicks))

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "parked" and not [c for c in backend.calls if c[0] == "swipe"]
    run(go())


def test_the_director_gets_the_daemons_look_with_stop_texts_and_the_lease_check(make, backend, screens, monkeypatch):
    """Live, the director gates every gesture on fast-eyes' recognised screen (`look`, S3); through the task it is the
    same look, with the account's stop texts checked on its items and the lease copy before it."""
    seen = []

    async def fake_look(self, device, platform=None, expect=None, settle=True):
        seen.append(platform)
        items = IG_FEED if len(seen) < 3 else [{"text": "Try Again Later", "x": 100, "y": 900, "w": 400, "h": 50, "conf": 1.0}]
        return {"screen": "instagram.home", "settled": True, "items": items}
    monkeypatch.setattr(bridge_mod.Bridge, "look", fake_look)

    async def uses_look(phone, device, *, app, **kw):
        for _ in range(5):
            meta = await phone.look(device, app)
            await phone.screen(device)
            assert meta["screen"] == "instagram.home"
            await phone.swipe(device, 585, 1900, 600, 700, 0.25)
        return {"status": "ok", "counts": {"posts": 5}}
    monkeypatch.setattr(taskapi.Browse, "on_feed", staticmethod(feed_markers))
    monkeypatch.setattr(taskapi.Browse, "visit", staticmethod(uses_look))

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "stopped" and "Try Again Later" in r["reason"]
        assert seen[0] == "instagram" and sum(c[0] == "swipe" for c in backend.calls) == 2
    run(go())


def test_frames_come_from_the_stream_and_a_failing_stream_is_counted(make, backend, screens, monkeypatch):
    calls = []

    async def fake_frame(self, device, format="jpeg", quality=80):
        calls.append(format)
        if len(calls) > 1:
            raise RuntimeError("stream down")
        return b"PNGDATA", {"source": "stream", "seq": 1}
    monkeypatch.setattr(bridge_mod.Bridge, "frame", fake_frame)
    out = {}

    async def visit(phone, device, **kw):
        out["first"] = await phone.screen(device)
        out["second"] = await phone.screen(device)  # the stream failed: the DVT still answers
        out["fallbacks"] = phone.stream_fallbacks
        return {"status": "ok", "counts": {"posts": 0}}
    monkeypatch.setattr(taskapi.Browse, "on_feed", staticmethod(feed_markers))
    monkeypatch.setattr(taskapi.Browse, "visit", staticmethod(visit))

    async def go():
        api = make()
        tid = (await api.submit("iphone12", browse()))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "done", r["reason"]
        assert out["first"] == b"PNGDATA" and out["second"] != b"PNGDATA" and out["fallbacks"] == 1
        assert calls == ["png", "png"]
    run(go())



def test_ians_knobs_travel_with_the_visit_and_capture_off_means_no_visit(make, backend, director, screens):
    """operator.toml (VM): exploration rate and comment voice go to the director; with screen capture off the Mac takes
    no visit at all (the director never acts without seeing): refused at submit, zero input."""
    async def go():
        api = make()
        knobs = {"exploration_rate": 0.2, "comments": "preview"}
        tid = (await api.submit("iphone12", task(kind="browse", payload={"minutes": 10, "settings": knobs})))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "done" and Visit.calls[0]["knobs"] == knobs
        with pytest.raises(taskapi.Refused, match="capture is off"):
            await api.submit("iphone12", task(kind="browse", task_id="t2", payload={"minutes": 10,
                                                                                  "settings": {"capture": False}}))
        assert len([c for c in backend.calls if c[0] == "launch"]) == 1  # only the first visit's
    run(go())


def test_a_visit_in_the_one_loop_runs_the_same_through_its_phones_queue(make, backend, director, screens):
    """Task protocol 2 (lane one-loop): the same visit, after the post that holds the phone; never both at once."""
    from test_run_queue import v2

    async def go():
        api = make()
        api.flows["slow_post"] = __import__("test_run_taskapi").SlowStage
        await api.submit("iphone12", v2("post1"))
        tid = (await api.submit("iphone12", browse(task_id="visit1") | {"v": 2}))["task_id"]
        assert (await api.status(tid))["status"] == "queued"
        (_, rp), (events, r) = await asyncio.gather(drain(api, "post1"), drain(api, tid))
        assert rp["status"] == "CONFIRMED" and r["status"] == "done", r["reason"]
        assert r["result"]["counts"]["posts"] == 3 and r["result"]["writes"] == 0
        assert [e["step"] for e in events if e.get("outcome") == "ok"] == ["open", "feed", "home"]
        j = Journal.open(api.runs_dir, "post1").events()
        start = next(e["mono"] for e in events if e.get("state") == "running")
        assert j[-1]["mono"] <= start
    run(go())
