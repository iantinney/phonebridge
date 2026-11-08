"""The device operator's task API on the Mac (ARC-01, REC-01, REC-02): fake backend, fake authhub, fake clock."""

import asyncio

import pytest

from phonebridge import bridge as bridge_mod
from phonebridge.bridge import Bridge
from phonebridge.proto import PROTO, hello
from phonebridge.run import taskapi
from phonebridge.run.journal import Journal
from phonebridge.run.taskapi import PostFlow, Refused, TaskAPI
from test_bridge import FakeBackend, authhub, config  # noqa: F401 (authhub: the fake CLI fixture)

LEASES = {"device": "callertok", "account": "accttok"}  # fake authhub: only these are live


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Killed(BaseException):
    """The daemon process dying mid-step (SIGKILL): nothing after it runs."""


def verify(token, identity):
    if {"device:iphone12": "callertok", "ig:public": "accttok"}.get(identity) != token:
        raise taskapi.LeaseError(f"no live lease on {identity}")
    return 600.0


class TapPost(PostFlow):
    """A post flow whose Post tap is a real (fake-backend) tap: counts taps across restarts."""
    kind = "tap_post"
    kill_at: str | None = None
    verify_found = True

    async def run_step(self, name, ctx):
        if TapPost.kill_at == name:
            TapPost.kill_at = None
            raise Killed()
        return await super().run_step(name, ctx)

    async def do_tap(self, ctx):
        await ctx.bridge.tap(ctx.device, 500, 2400, lease=ctx.lease)
        if TapPost.kill_at == "after_tap":
            TapPost.kill_at = None
            raise Killed()

    async def do_verify(self, ctx):
        return TapPost.verify_found


@pytest.fixture
def backend():
    b = FakeBackend()
    b.front = "com.burbn.instagram"
    return b


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make(tmp_path, backend, clock, authhub):
    def make(**kw):
        b = Bridge(backend, config(authhub[0]), log_path=tmp_path / "calls.jsonl")

        async def foreground(device):  # lane host's frontmost-app query; Instagram unless a test says otherwise
            return backend.front
        b.foreground = foreground
        return TaskAPI(b, runs_dir=tmp_path / "runs", verify=verify, clock=clock, ack_timeout=0.3, poll=0.01,
                       settle=0, park_wait=kw.pop("park_wait", 0.2), flows={**taskapi.FLOWS, "tap_post": TapPost},
                       **kw)
    return make


STOP = ["Try again later", "Action Blocked", "Log In", "We suspect automated behavior"]  # from the VM's pacing config


def task(kind="tap_post", **kw):
    return {"task_id": kw.pop("task_id", "t1"), "identity": "ig:public", "kind": kind,
            "payload": kw.pop("payload", {"content_key": "sha-1"}), "leases": dict(LEASES), "lease_s": 600,
            "stop_texts": STOP, **kw}


async def killed(api, tid):
    """Acknowledge like the VM until the runner dies; it must have died of Killed."""
    after = 0
    while not api.runner_done(tid):
        for e in (await api.events(tid, after_seq=after))["events"]:
            after = e["seq"]
            if e.get("need_ack"):
                await api.ack(tid, e["need_ack"])
        await asyncio.sleep(0.01)
    with pytest.raises(Killed):
        await api.wait_runner(tid)


async def drain(api, tid, ack=True, after=0, timeout=5.0):
    """The VM's side: pull events by seq; acknowledge `publishing` when the task asks."""
    got = []
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        r = await api.events(tid, after_seq=after)
        for e in r["events"]:
            assert e["seq"] == after + 1, f"gap after {after}: {e}"
            after = e["seq"]
            got.append(e)
            if ack and e.get("need_ack"):
                await api.ack(tid, e["need_ack"])
        if r["final"] or r["status"] == "parked":
            return got, r
        await asyncio.sleep(0.01)
    raise AssertionError(f"task {tid} never settled: {got[-3:]}")


def taps(backend):
    return [c for c in backend.calls if c[0] == "tap"]


def run(coro):
    return asyncio.run(coro)


def test_submit_refuses_without_live_leases_and_unknown_kinds(make):
    async def go():
        api = make()
        with pytest.raises(Refused, match="lease"):
            await api.submit("iphone12", {**task(), "leases": {}})
        with pytest.raises(Refused, match="no live lease on ig:public"):
            await api.submit("iphone12", {**task(), "leases": {"device": "callertok", "account": "stale"}})
        with pytest.raises(Refused, match="not wired"):
            await api.submit("iphone12", task(kind="dm"))
        assert Journal.all(api.runs_dir) == []
    run(go())


def test_a_post_runs_once_with_an_ack_and_events_are_contiguous(make, backend):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        events, r = await drain(api, tid)
        states = [e["state"] for e in events if "state" in e]
        assert states == ["submitted", "running", "PICKED", "STAGED", "ARMED", "POST_TAPPED", "CONFIRMED"]
        assert r["final"] and len(taps(backend)) == 1
        assert await api.submit("iphone12", task()) == {"task_id": tid, "status": "CONFIRMED", **hello()}  # idempotent
        assert r["proto"] == PROTO  # every reply carries the protocol version (authhub's driver checks it)
    run(go())


def test_events_pulled_by_seq_no_gaps(make):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        everything, _ = await drain(api, tid)
        chunks, after = [], 0
        while True:
            got = (await api.events(tid, after_seq=after, limit=2))["events"]
            if not got:
                break
            chunks += got
            after = got[-1]["seq"]
        assert chunks == everything and [e["seq"] for e in chunks] == list(range(1, len(chunks) + 1))
    run(go())


def test_park_is_idempotent_and_resume_continues(make, backend):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        await api.park(tid)
        first = await api.park(tid)
        assert first["status"] == "parked" and await api.park(tid) == first
        parks = [e for e in (await api.events(tid))["events"] if e.get("state") == "parked"]
        assert len(parks) == 1
        await api.resume(tid, LEASES, 600)
        _, r = await drain(api, tid)
        assert r["status"] == "CONFIRMED" and len(taps(backend)) == 1
    run(go())


@pytest.mark.parametrize("step", ["pick", "stage", "arm", "verify"])
def test_resume_after_kill_at_each_step(make, backend, step):
    async def go():
        TapPost.kill_at = step
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        await killed(api, tid)
        api2 = make()  # a new daemon process on the same journal directory
        assert (await api2.status(tid))["status"] == "parked"
        await api2.resume(tid, LEASES, 600)
        events, r = await drain(api2, tid)
        j = Journal.open(api2.runs_dir, tid, clock=api2.clock)
        done = j.done_steps
        if step == "arm":
            assert done == ["pick", "stage", "back_out"]
            assert r["status"] == "backed_out" and taps(backend) == []
            return
        assert done == ["pick", "stage", "arm", "tap", "verify"]  # every step completed exactly once
        assert len(taps(backend)) == 1 and r["status"] == "CONFIRMED"
        assert [e["seq"] for e in j.events()] == list(range(1, len(j.events()) + 1))
        assert any(e.get("state") == "resumed" for e in j.events())
    run(go())


@pytest.mark.parametrize("kill", ["tap", "after_tap"])
def test_post_tapped_then_crash_never_second_tap(make, backend, kill):
    """Killed after POST_TAPPED is journaled (before or after the physical tap): never a second tap."""
    async def go():
        TapPost.kill_at = kill
        TapPost.verify_found = kill == "after_tap"
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        await killed(api, tid)
        j = Journal.open(api.runs_dir, tid)
        assert j.post_state in ("ARMED", "POST_TAPPED")
        api2 = make()
        await api2.resume(tid, LEASES, 600)
        _, r = await drain(api2, tid)
        assert len(taps(backend)) <= 1
        if Journal.open(api2.runs_dir, tid).post_state == "POST_TAPPED" or kill == "after_tap":
            assert r["status"] in ("CONFIRMED", "POST_UNCONFIRMED")
        TapPost.verify_found = True
    run(go())


def test_post_tapped_is_journaled_before_the_tap(make, backend):
    async def go():
        TapPost.kill_at = "after_tap"
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        await killed(api, tid)
        assert Journal.open(api.runs_dir, tid).post_state == "POST_TAPPED" and len(taps(backend)) == 1
        TapPost.verify_found = False
        api2 = make()
        await api2.resume(tid, LEASES, 600)
        _, r = await drain(api2, tid)
        assert r["status"] == "POST_UNCONFIRMED" and len(taps(backend)) == 1  # verify, never re-tap
        TapPost.verify_found = True
    run(go())


def test_stale_armed_backs_out(make, backend, clock):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        _, r = await drain(api, tid, ack=False)  # nobody acknowledges: parks at ARMED
        assert r["status"] == "parked" and Journal.open(api.runs_dir, tid).post_state == "ARMED"
        clock.t += 31 * 60
        api2 = make()
        assert (await api2.status(tid))["status"] == "parked"  # no saved tokens after restart
        assert (await api2.resume(tid, LEASES, 600))["status"] == "backed_out"
        j = Journal.open(api2.runs_dir, tid, clock=api2.clock)
        assert j.final and any(e.get("step") == "back_out" for e in j.events()) and taps(backend) == []
    run(go())


def test_tunnel_drop_parks_before_tap(make, backend):
    """The VM stops answering between ARMED and the tap: no acknowledgement, no tap, the task parks."""
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        _, r = await drain(api, tid, ack=False)
        assert r["status"] == "parked" and "acknowledg" in r["reason"] and taps(backend) == []
    run(go())


def test_an_ack_for_another_content_key_is_refused(make, backend):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        for _ in range(200):
            if any(e.get("need_ack") for e in (await api.events(tid))["events"]):
                break
            await asyncio.sleep(0.01)
        with pytest.raises(Refused):
            await api.ack(tid, "sha-other")
        _, r = await drain(api, tid, ack=False)
        assert r["status"] == "parked" and taps(backend) == []
    run(go())


def test_lease_copy_expiry_parks_and_extend_keeps_it(make, clock):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", {**task(), "lease_s": 5}))["task_id"]
        clock.t += 6
        _, r = await drain(api, tid)
        assert r["status"] == "parked" and "lease" in r["reason"]
    run(go())


def test_rehearse_post_is_mock_and_never_touches_the_phone(make, backend):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task(kind="rehearse_post")))["task_id"]
        events, r = await drain(api, tid)
        assert r["status"] == "CONFIRMED" and backend.calls == [] and all(e.get("mock") for e in events[1:])
    run(go())


IG_FEED = [{"text": "Instagram", "x": 60, "y": 90, "w": 200, "h": 40, "conf": 1.0},
           {"text": "Your story", "x": 40, "y": 300, "w": 120, "h": 20, "conf": 1.0}]
IG_PROFILE = [{"text": "samplecreator.co v", "x": 589, "y": 221, "w": 559, "h": 59, "conf": 1.0},  # real layout
              {"text": "0", "x": 420, "y": 400, "w": 20, "h": 40, "conf": 1.0},
              {"text": "posts", "x": 400, "y": 450, "w": 70, "h": 30, "conf": 1.0},
              {"text": "12", "x": 640, "y": 400, "w": 40, "h": 40, "conf": 1.0},
              {"text": "followers", "x": 600, "y": 450, "w": 120, "h": 30, "conf": 1.0},
              {"text": "34", "x": 860, "y": 400, "w": 40, "h": 40, "conf": 1.0},
              {"text": "following", "x": 820, "y": 450, "w": 120, "h": 30, "conf": 1.0},
              {"text": "Edit profile", "x": 200, "y": 700, "w": 200, "h": 40, "conf": 1.0}]
IG_MENU = [{"text": "Settings and activity", "x": 585, "y": 212, "w": 486, "h": 55, "conf": 1.0},
           {"text": "Q Search", "x": 186, "y": 388, "w": 232, "h": 48, "conf": 1.0}]
IG_SEARCHED = [{"text": "Settings and activity", "x": 585, "y": 212, "w": 486, "h": 55, "conf": 1.0},
               {"text": "Q account status", "x": 200, "y": 386, "w": 300, "h": 68, "conf": 1.0},
               {"text": "More info and support", "x": 270, "y": 526, "w": 445, "h": 44, "conf": 1.0},
               {"text": "Account Status", "x": 150, "y": 1900, "w": 260, "h": 40, "conf": 1.0}]
IG_STATUS = [{"text": "Account Status", "x": 400, "y": 120, "w": 300, "h": 40, "conf": 1.0},
             {"text": "You haven't posted anything that goes against our Community Guidelines.",
              "x": 100, "y": 600, "w": 900, "h": 60, "conf": 1.0}]


@pytest.fixture
def ig_screens(monkeypatch, backend):
    """OCR follows what was tapped: launch -> feed, profile tab -> profile, menu -> settings, entry -> status."""
    def recognize(png, fast=False):
        last = next((c for c in reversed(backend.calls) if c[0] in ("tap", "launch", "home", "type")), None)
        if last is None or last[0] in ("launch", "home"):
            return IG_FEED
        if last[0] == "type":
            return IG_SEARCHED
        _, _, x, y = last
        if y > 2300 and x > 900:
            return IG_PROFILE
        if y < 300 and x > 900:
            return IG_MENU
        if 350 < y < 430 and x < 400:
            return IG_MENU  # the search field focused, keyboard up
        if 1850 < y < 1990:
            return IG_STATUS
        return IG_FEED
    monkeypatch.setattr(bridge_mod.vision, "recognize", recognize)
    backend.current = __import__("test_bridge").tiny_png(1170, 2532, b"\x80\x80\x80")  # a lit iPhone 12 screen


def test_snapshot_status_reads_profile_and_status_and_goes_home(make, backend, ig_screens, tmp_path):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task(kind="snapshot_status", payload={})))["task_id"]
        events, r = await drain(api, tid)
        assert r["status"] == "done", events[-3:]
        res = r["result"]
        assert res["handle"] == "samplecreator.co"
        assert (res["posts"], res["followers"], res["following"]) == (0, 12, 34)
        assert res["status"] == "healthy" and "Community Guidelines" in res["status_text"]
        kinds = [c[0] for c in backend.calls]
        assert kinds[0] == "launch" and kinds[-1] == "home" and "swipe" not in kinds and "type" in kinds
        assert not any(c[0] == "tap" and c[2] < 300 and c[3] < 300 for c in backend.calls)  # never near "+" (84, 216)
        assert [e["screen_id"] for e in events if e.get("outcome") == "ok" and e.get("screen_id")] == [
            "ig_feed", "ig_profile", "ig_account_status", "home"]
        frames = [e["frame_ref"] for e in events if e.get("frame_ref")]
        assert frames and all((api.runs_dir / tid / f).exists() for f in frames)
    run(go())


def test_snapshot_status_stops_on_a_stop_screen(make, backend, monkeypatch):
    monkeypatch.setattr(bridge_mod.vision, "recognize", lambda png, fast=False: [
        {"text": "Try Again Later", "x": 100, "y": 900, "w": 400, "h": 50, "conf": 1.0}])

    async def go():
        api = make()
        tid = (await api.submit("iphone12", task(kind="snapshot_status", payload={})))["task_id"]
        _, r = await drain(api, tid)
        assert r["status"] == "stopped" and "Try Again Later" in r["reason"]
        assert [c[0] for c in backend.calls][-1] == "home"
    run(go())



@pytest.mark.parametrize("screen", [
    [{"text": "Share", "x": 585, "y": 2300, "w": 900, "h": 80, "conf": 1.0}],  # IG's final share page
    [{"text": "Save draft", "x": 300, "y": 2350, "w": 300, "h": 60, "conf": 1.0},
     {"text": "Next", "x": 880, "y": 2350, "w": 300, "h": 60, "conf": 1.0}],  # the caption page
])
def test_snapshot_never_taps_blind_on_a_composer_screen(make, backend, monkeypatch, screen):
    monkeypatch.setattr(bridge_mod.vision, "recognize", lambda png, fast=False: screen)
    backend.current = __import__("test_bridge").tiny_png(1170, 2532, b"\x80\x80\x80")

    async def go():
        api = make()
        tid = (await api.submit("iphone12", task(kind="snapshot_status", payload={})))["task_id"]
        events, r = await drain(api, tid)
        assert r["status"] == "done" and [c[0] for c in backend.calls] == ["launch", "home"]
        assert any("no tap" in (e.get("skipped") or "") for e in events)
    run(go())


def test_two_concurrent_resumes_start_one_runner_and_tap_once(make, backend):
    async def go():
        TapPost.kill_at = "stage"
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        await killed(api, tid)  # died before staging: no relinquished composer to trust
        await asyncio.gather(api.resume(tid, LEASES, 600), api.resume(tid, LEASES, 600))
        events, r = await drain(api, tid)
        assert r["status"] == "CONFIRMED" and len(taps(backend)) == 1
        seqs = [e["seq"] for e in Journal.open(api.runs_dir, tid).events()]
        assert seqs == list(range(1, len(seqs) + 1))
        assert sum(e.get("state") == "resumed" for e in Journal.open(api.runs_dir, tid).events()) == 1
    run(go())


class SlowStage(TapPost):
    kind = "slow_post"

    async def do_stage(self, ctx):
        await asyncio.sleep(0.5)


def test_park_during_a_long_step_never_duplicates_a_seq(make, backend):
    async def go():
        api = make(park_wait=0.05)
        api.flows["slow_post"] = SlowStage
        tid = (await api.submit("iphone12", task(kind="slow_post")))["task_id"]
        await asyncio.sleep(0.1)  # inside stage
        await api.park(tid)  # returns before the step ends; the runner writes `parked` at its next boundary
        await api.wait_runner(tid)
        evs = Journal.open(api.runs_dir, tid).events()
        assert [e["seq"] for e in evs] == list(range(1, len(evs) + 1))
        assert sum(e.get("state") == "parked" for e in evs) == 1 and taps(backend) == []
    run(go())


def test_the_sweeper_backs_out_a_stale_armed_task_nobody_asks_about(make, backend, clock):
    async def go():
        api = make(sweep_s=0.05)
        tid = (await api.submit("iphone12", task()))["task_id"]
        _, r = await drain(api, tid, ack=False)
        assert r["status"] == "parked"
        clock.t += 31 * 60
        await asyncio.sleep(0.3)  # no call about this task: the sweeper alone
        assert Journal.open(api.runs_dir, tid).status == "backed_out" and taps(backend) == []
    run(go())


def test_an_ack_racing_a_park_never_taps(make, backend):
    async def go():
        api = make()
        tid = (await api.submit("iphone12", task()))["task_id"]
        for _ in range(200):
            if any(e.get("need_ack") for e in (await api.events(tid))["events"]):
                break
            await asyncio.sleep(0.01)
        api.live[tid]["park"] = True  # the park request lands ...
        await api.ack(tid, "sha-1")  # ... in the same wake as the ack
        await api.wait_runner(tid)
        assert Journal.open(api.runs_dir, tid).status == "parked" and taps(backend) == []
        assert Journal.open(api.runs_dir, tid).post_state == "ARMED"
    run(go())


def test_snapshot_never_taps_blind_outside_the_app(make, backend, ig_screens):
    backend.front = "com.apple.springboard"  # launch didn't bring Instagram up (or a resume found Home)

    async def go():
        api = make()
        tid = (await api.submit("iphone12", task(kind="snapshot_status", payload={})))["task_id"]
        events, r = await drain(api, tid)
        assert r["status"] == "parked" and "springboard is in front" in r["reason"] and taps(backend) == []
        assert [c[0] for c in backend.calls] == ["launch", "launch"]  # no tap, no Home: only launches
    run(go())


def test_account_status_reads_instagrams_real_wording():
    """The live screen on 2026-10-18 (evidence M2-orchestrator/20261018T195400Z, frame 010) is healthy, though it
    names removals and restrictions (as absent)."""
    real = ("Your account meets all our guidelines You don't have removed content or messages, feature restrictions "
            "or recommendation limits. Learn more about Account Status Features Content and message removals")
    assert taskapi.account_status(real) == "healthy"
    assert taskapi.account_status("Some of your content was removed for violating our guidelines") == "notice"
    assert taskapi.account_status("Account Status Features") == "unknown"


def test_extension_rechecks_both_tokens_and_clamps_broker_ttl(make, clock):
    async def go():
        api = make()
        calls = []
        def checked(token, identity):
            calls.append((token, identity))
            return 7 if identity == 'ig:public' else 20
        api.verify = checked
        await api.submit('iphone12', task())
        calls.clear()
        await api.events('t1', extend_s=9999)
        assert set(calls) == {('callertok', 'device:iphone12'), ('accttok', 'ig:public')}
        assert api.live['t1']['deadline'] <= clock() + 7
        def lost(*args):
            raise taskapi.LeaseError('replaced')
        api.verify = lost
        with pytest.raises(Refused):
            await api.events('t1', extend_s=60)
        assert api.live['t1']['park']
        await api.park('t1')
    run(go())


@pytest.mark.parametrize('seconds', [0, -1, float('inf'), float('nan')])
def test_invalid_lease_window_refused(make, seconds):
    async def go():
        with pytest.raises(Refused, match='finite and positive'):
            await make().submit('iphone12', task(lease_s=seconds))
    run(go())


def test_distinct_tasks_cannot_enter_same_phone_together(make):
    async def go():
        entered = []
        release = asyncio.Event()
        class Hold(taskapi.Flow):
            mock = True
            steps = ('hold',)
            async def run_step(self, name, ctx):
                entered.append(ctx.task['task_id'])
                await release.wait()
        api = make()
        api.flows['hold'] = Hold
        await api.submit('iphone12', task(kind='hold'))
        await api.submit('iphone12', task(kind='hold', task_id='t2'))
        await asyncio.sleep(.05)
        try:
            assert entered == ['t1']
        finally:
            release.set()
            await api.wait_runner('t1')
            await api.wait_runner('t2')
    run(go())


def test_extension_transport_failure_keeps_only_the_verified_lifetime(make, clock):
    from phonebridge.lease import LeaseUnavailable

    async def go():
        api = make()
        await api.submit('iphone12', task())
        deadline = api.live['t1']['deadline']
        def unreachable(*args):
            raise LeaseUnavailable('authhub transport unavailable')
        api.verify = unreachable
        await api.events('t1', extend_s=9999)
        assert api.live['t1']['deadline'] == deadline
        assert not api.live['t1']['park']
        clock.t = deadline
        with pytest.raises(taskapi.Park):
            api._check('t1')
        await api.park('t1')
    run(go())


def test_stale_draft_defers_once_until_resume_supplies_fresh_leases(make, clock):
    async def go():
        api = make(stale_s=1)
        await api.submit('iphone12', task())
        await drain(api, 't1', ack=False)
        api.live.clear()  # the VM ended its visit and released both tokens
        clock.t += 2
        await api.status('t1')
        journal = api._open('t1')
        assert journal.last(lambda e: e.get('step') == 'back_out')['outcome'] == 'deferred'
        before = len(journal.events())
        await api.status('t1')
        assert len(journal.events()) == before
        assert (await api.resume('t1', LEASES, 60))['status'] == 'backed_out'
    run(go())
