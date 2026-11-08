"""DEV-04 power: `caffeinate -i -w <our pid>` only while a task runs (no always-on assertion), plus the busy
marker the deploy script checks. A task is a live device lease used by phonebridge."""

import asyncio
import json
import os

from phonebridge.bridge import Bridge
from phonebridge.host.power import TaskPower
from test_bridge import config, fake_authhub
from test_bridge_guards import IG, Phone


class Proc:
    def __init__(self, args):
        self.args, self.alive = args, True

    def terminate(self):
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None if self.alive else 0


def power(tmp_path, spawned):
    return TaskPower(marker=tmp_path / "busy.json", caffeinate="/usr/bin/caffeinate",
                     spawn=lambda args, **kw: spawned.append(Proc(args)) or spawned[-1])


def held(spawned):
    return [p for p in spawned if p.alive]


def test_power_assertion_held_only_while_task_runs(tmp_path):
    dead = tmp_path / "lease-ended"  # the fake authhub's `lease check` fails once this exists
    cmd, _ = fake_authhub(tmp_path)
    script = tmp_path / "authhub"
    script.write_text(script.read_text().replace("check) [", f"check) [ ! -e {dead} ] && ["))
    spawned = []
    marker = tmp_path / "busy.json"
    b = Bridge(Phone(), config(cmd), log_path=tmp_path / "calls.jsonl", power=power(tmp_path, spawned),
               lease_idle=0.15, task_poll=0.05)

    async def go():
        assert not held(spawned) and not marker.exists()  # no task: no assertion
        await b.launch("iphone12", IG)  # phonebridge holds device:iphone12 for this caller: a task
        assert [p.args for p in held(spawned)] == [["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]]
        busy = json.loads(marker.read_text())
        assert list(busy["tasks"]) == ["iphone12"] and busy["tasks"]["iphone12"]["app"] == IG
        await b.tap("iphone12", 5, 5)
        assert len(spawned) == 1  # one assertion per host, not per input
        await asyncio.sleep(0.3)  # idle: the lease is released, the task is over
        assert not held(spawned) and not marker.exists()

        await b.tap("iphone12", 5, 5, lease="callertok")  # a task that owns its lease (authhub run)
        assert len(held(spawned)) == 1 and marker.exists()
        await asyncio.sleep(0.2)
        assert len(held(spawned)) == 1  # its lease is live: still held between inputs
        dead.touch()  # authhub run exits and releases the lease
        await asyncio.sleep(0.2)
        assert not held(spawned) and not marker.exists()
        await b.close()

    asyncio.run(go())


def test_restart_restores_a_live_task_from_the_marker(tmp_path):
    """A daemon restart (deploy, kickstart) mid-task keeps the assertion and the busy marker."""
    spawned = []
    first = power(tmp_path, spawned)
    first.begin("iphone12", owner="callertok", identity="device:iphone12", app=IG)
    del first  # the process dies; caffeinate -w would drop its assertion with it
    again = power(tmp_path, spawned)
    assert again.restore(lambda task: task["owner"] == "callertok") == ["iphone12"]
    assert len(held(spawned)) == 2 and again.held
    again.end("iphone12")
    assert not (tmp_path / "busy.json").exists() and not again.held

    third = power(tmp_path, spawned)
    third.begin("iphone12", owner="gone")
    assert power(tmp_path, spawned).restore(lambda task: False) == []  # a dead lease: cleared, nothing held
    assert not (tmp_path / "busy.json").exists()


def test_without_caffeinate_only_the_marker_is_kept(tmp_path):
    p = TaskPower(marker=tmp_path / "busy.json", caffeinate=None)  # Linux CI, the VM
    p.begin("iphone12", owner="x")
    assert p.held and (tmp_path / "busy.json").exists()
    p.end("iphone12")
    assert not p.held
