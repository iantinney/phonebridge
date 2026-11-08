"""Independent review (azure/tw-phone-review) of missions.session()'s cleanup with a recorded demonstration: the
recorder stops while the phone is still ours, every release runs whatever failed before it, and the video copy waits
until the phone is released. authhub, focusgate, the daemon and scp are fakes recording the order of events; no
phone, Mac or account is reached."""

import subprocess
import types

import pytest

from teachable_worker import missions

site = missions._site  # the private deployment's session (authhub leases + Focus): what this reviews
pytestmark = pytest.mark.skipif(site is None, reason="the private deployment's work session isn't in this install")


@pytest.fixture
def world(tw, monkeypatch):
    events, w = [], types.SimpleNamespace(stop_fails=0, focus_end=None)

    def lease(op, payload):
        events.append(f"lease {op} {payload.get('identity') or payload['token']}")
        return {"token": payload["identity"]} if op == "acquire" else {}

    class Hands:
        def __init__(self, device, token):
            self.device, self.strip = device, None

        def tool(self, name, **args):
            events.append(name)
            if name == "record_stop" and w.stop_fails:
                w.stop_fails -= 1
                return {"error": "the daemon didn't answer"}
            return {"t0": 1.0} if name == "record_stop" else {"ok": True}

        def release(self):
            events.append("controller release")

    def run(cmd, *a, **kw):
        events.append(" ".join(str(c) for c in cmd[:4]))
        if cmd[0] == "focusgate" and cmd[2:4] == ["iphone12", "end"] and w.focus_end:
            raise w.focus_end
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(site, "_lease", lease)
    monkeypatch.setattr(missions, "Hands", Hands)
    monkeypatch.setattr(site, "FOCUSGATE", "focusgate")
    monkeypatch.setattr(site.subprocess, "run", run)
    monkeypatch.setattr(missions, "DEMO_ROOT", tw / "demos")
    w.events = events
    return w


def _ctx():
    ctx = types.SimpleNamespace(device={"id": "iphone12"}, task={"id": "task_1", "kind": "account_check"})
    ctx.demo = missions.Demo("task_1", "Check @nasa's profile")
    return ctx


def _releases_after(events, i):
    return [e for e in events[i:] if e.startswith("lease release")]


def test_the_focus_end_timing_out_still_releases_both_leases_and_the_copy_waits_for_them(world):
    world.focus_end = subprocess.TimeoutExpired("focusgate", 30)
    ctx = _ctx()
    with missions.session(ctx):
        pass
    ev = world.events
    stop, end = ev.index("record_stop"), ev.index("focusgate work iphone12 end")
    assert stop < ev.index("controller release") < end  # the recorder stops while the phone is still ours
    assert _releases_after(ev, end) == ["lease release device:iphone12", "lease release ig:public"]  # despite the timeout
    scp = next(i for i, e in enumerate(ev) if e.startswith("scp"))
    assert scp > ev.index("lease release ig:public")  # the copy never holds the phone


def test_a_recorder_that_wont_stop_is_retried_once_flagged_and_never_copied(world):
    world.stop_fails = 2
    ctx = _ctx()
    with missions.session(ctx):
        pass
    ev = world.events
    assert ev.count("record_stop") == 2 and not any(e.startswith("scp") for e in ev)
    assert ev[-2:] == ["lease release device:iphone12", "lease release ig:public"]  # the lease's end stops the strip
    journal = (ctx.demo.dir / "journal.jsonl").read_text()
    assert "the recording may still be running" in journal
    assert oct((ctx.demo.dir).stat().st_mode & 0o777) == "0o700"  # private phone evidence
