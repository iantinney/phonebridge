"""Task protocol 2 (lane one-loop): one runner per phone, the rest wait in its queue in submit order; v1 tasks behave
exactly as before (test_run_taskapi.py is the proof), and a v1 task never starts on a phone a v2 task holds."""

# ruff: noqa: F811  (fixtures imported from test_run_taskapi are redefined as test parameters, by design)
import asyncio
import contextlib

import pytest

from phonebridge.proto import TASKS, hello
from phonebridge.run.journal import Journal
from phonebridge.run.taskapi import Refused
from test_run_taskapi import LEASES, SlowStage, TapPost, drain, run, task, taps  # noqa: F401
from test_run_taskapi import backend, clock, make  # noqa: F401 (fixtures)
from test_bridge import authhub  # noqa: F401


def v2(tid, kind="slow_post", **kw):
    return task(kind=kind, task_id=tid, v=2, **kw)


def states(api, tid):
    return [e["state"] for e in Journal.open(api.runs_dir, tid).events() if "state" in e]


@pytest.fixture
def slow(make):
    def mk(**kw):
        api = make(**kw)
        api.flows["slow_post"] = SlowStage
        return api
    return mk


def test_hello_says_the_daemon_runs_task_protocol_2():
    assert hello()["task_proto"] == TASKS == 2


def test_v2_tasks_on_one_phone_run_one_at_a_time_in_order(slow, backend):
    async def go():
        api = slow()
        for tid in ("a", "b", "c"):
            await api.submit("iphone12", v2(tid))
        assert api.holders and api.queues[next(iter(api.queues))] == ["b", "c"]
        assert states(api, "b")[-1] == "queued" and (await api.status("c"))["status"] == "queued"
        done = await asyncio.gather(*(drain(api, t) for t in ("a", "b", "c")))
        assert [r["status"] for _, r in done] == ["CONFIRMED"] * 3 and len(taps(backend)) == 3
        starts = {t: next(e["mono"] for e in Journal.open(api.runs_dir, t).events() if e.get("state") == "running")
                  for t in "abc"}
        ends = {t: Journal.open(api.runs_dir, t).events()[-1]["mono"] for t in "abc"}
        assert ends["a"] <= starts["b"] and ends["b"] <= starts["c"]  # never two runners on the phone
        assert not api.holders and not any(api.queues.values())
    run(go())


def test_a_v1_task_is_refused_while_a_v2_task_holds_the_phone(slow, backend):
    async def go():
        api = slow()
        await api.submit("iphone12", v2("a"))
        with pytest.raises(Refused, match="busy: task a"):
            await api.submit("iphone12", task(task_id="old"))
        await drain(api, "a")
        await api.submit("iphone12", task(task_id="old"))  # free again: v1 as before
        _, r = await drain(api, "old")
        assert r["status"] == "CONFIRMED"
    run(go())


def test_a_v2_task_waits_for_a_running_v1_task(slow, backend):
    async def go():
        api = slow()
        await api.submit("iphone12", task(kind="slow_post", task_id="old"))  # v1: runs at once
        await api.submit("iphone12", v2("new", kind="tap_post"))
        assert (await api.status("new"))["status"] == "queued"
        (_, r1), (_, r2) = await asyncio.gather(drain(api, "old"), drain(api, "new"))
        assert r1["status"] == r2["status"] == "CONFIRMED"
        old_end = Journal.open(api.runs_dir, "old").events()[-1]["mono"]
        assert old_end <= next(e["mono"] for e in Journal.open(api.runs_dir, "new").events() if e.get("state") == "running")
    run(go())


def test_parking_a_queued_task_takes_it_out_and_resume_puts_it_at_the_back(slow, backend):
    async def go():
        api = slow()
        for tid in ("a", "b", "c"):
            await api.submit("iphone12", v2(tid))
        assert (await api.park("b"))["status"] == "parked"
        assert api.queues[next(iter(api.queues))] == ["c"]
        await api.resume("b", LEASES, 600)
        assert api.queues[next(iter(api.queues))] == ["c", "b"]
        await api.resume("b", LEASES, 600)  # a second resume while queued changes nothing
        assert api.queues[next(iter(api.queues))] == ["c", "b"]
        done = await asyncio.gather(*(drain(api, t) for t in ("a", "c", "b")))
        assert [r["status"] for _, r in done] == ["CONFIRMED"] * 3 and len(taps(backend)) == 3
    run(go())


def test_a_restarted_daemon_parks_queued_tasks_until_the_vm_resumes_them(slow, backend):
    async def go():
        api = slow()
        await api.submit("iphone12", v2("a"))
        await api.submit("iphone12", v2("b"))
        api2 = slow()  # a new process on the same journals: nothing in memory
        api.queues.clear()  # simulate process death: no old pump may start another task
        api.runners["a"].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await api.runners["a"]
        if api._sweeper:
            api._sweeper.cancel()
        assert (await api2.status("b"))["status"] == "parked"
        assert "daemon restarted" in (await api2.status("b"))["reason"]
        await api2.resume("a", LEASES, 600)
        await api2.resume("b", LEASES, 600)
        (_, ra), (_, rb) = await asyncio.gather(drain(api2, "a"), drain(api2, "b"))
        assert {ra["status"], rb["status"]} == {"CONFIRMED"}
    run(go())


def test_task_protocol_is_checked_and_post_needs_v2(make):
    pytest.importorskip("phonebridge.run.post")  # the publishing adapter (not in every install)

    async def go():
        api = make()
        with pytest.raises(Refused, match="task protocol 3"):
            await api.submit("iphone12", task(v=3))
        with pytest.raises(Refused, match="needs task protocol 2"):
            await api.submit("iphone12", task(kind="post", payload={"content_key": "k", "caption": "c", "duration": 3}))
        assert Journal.all(api.runs_dir) == []
    run(go())
