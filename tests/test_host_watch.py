"""DEV-04 sleep/wake, REC-05 USB watchdog, DEV-08 thermal/battery: park with zero input, re-check, resume."""

import asyncio
import json

import pytest

from phonebridge.bridge import Bridge, Parked, TaskEnded
from phonebridge.host.watch import Watch
from test_bridge import authhub, config  # noqa: F401  (authhub: a fixture)
from test_bridge_guards import IG, Phone, inputs


class Clock:
    def __init__(self):
        self.t = 1_000.0

    def __call__(self):
        return self.t


class Watched(Phone):
    """Records the order of the wake re-checks; `present` is what usbmux lists."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.order, self.present = [], {"U12"}

    async def attached(self):
        return sorted(self.present)

    async def devices(self):
        self.order.append("devices")
        return [d for d in await super().devices() if d["udid"] in self.present]

    async def lock_state(self, udid):
        self.order.append("lock_state")
        return {"locked": False}

    async def foreground(self, udid):
        self.order.append("foreground")
        return await super().foreground(udid)

    async def tap(self, udid, x, y, press):
        self.order.append("tap")
        await super().tap(udid, x, y, press)


@pytest.fixture
def rig(tmp_path, authhub):
    phone, clock, handoffs = Watched(), Clock(), []
    b = Bridge(phone, config(authhub[0]), log_path=tmp_path / "calls.jsonl")
    w = Watch(b, clock=clock, wall=clock, lost_after=30, handoff=handoffs.append, events=tmp_path / "events.jsonl")
    return phone, b, w, clock, handoffs, tmp_path / "events.jsonl"


def events(path, kind=None):
    out = [json.loads(line) for line in path.open()] if path.exists() else []
    return [e for e in out if kind is None or e["event"] == kind]


def test_sleep_parks_and_wake_rechecks_before_resume(rig):
    phone, b, w, clock, handoffs, log = rig

    async def go():
        await b.launch("iphone12", IG)  # a task is at its STAGED step
        w.will_sleep()
        assert w.state() == "sleeping"
        with pytest.raises(Parked, match="sleep"):
            await b.tap("iphone12", 5, 5)
        assert inputs(phone) == ["launch"]  # no input after the will-sleep event
        phone.order.clear()
        clock.t += 600
        assert (await w.did_wake())["iphone12"]["resumed"]
        await b.tap("iphone12", 5, 5)
        assert phone.order == ["devices", "lock_state", "foreground", "foreground", "tap"]  # re-checked first

        w.will_sleep()
        phone.present.clear()  # unplugged while the lid was shut
        clock.t += 60
        assert not (await w.did_wake())["iphone12"]["resumed"]
        with pytest.raises(Parked, match="usb"):
            await b.tap("iphone12", 5, 5)
        assert w.state() == "no_usb"
        phone.present.add("U12")
        await w.poll_usb()
        await b.tap("iphone12", 5, 5)

        w.will_sleep()
        clock.t += 31 * 60  # asleep past the stale bound (orchestrator: 30 min): the task backs out
        await w.did_wake()
        with pytest.raises(TaskEnded, match="stale"):
            await b.tap("iphone12", 5, 5)
        await b.close()

    asyncio.run(go())
    assert len(events(log, "sleep_mid_task")) == 3 and handoffs == []  # a notice per sleep, never a page


def test_sleep_without_a_task_is_just_logged(rig):
    phone, b, w, clock, handoffs, log = rig

    async def go():
        w.will_sleep()
        clock.t += 3600
        await w.did_wake()
        await b.tap("iphone12", 5, 5)  # no task was parked: input simply flows
        await b.close()

    asyncio.run(go())
    assert [e["event"] for e in events(log) if e["event"] != "recheck"] == ["will_sleep", "did_wake"] and handoffs == []


def test_device_lost_resume_at_T_minus_1_fail_at_T_plus_1(rig):
    phone, b, w, clock, handoffs, log = rig

    async def go():
        await b.launch("iphone12", IG)
        w.battery("U12", {"temperature": 31.5, "level": 64, "charging": True})
        await w.poll_usb()
        phone.present.clear()  # cable out
        await w.poll_usb()
        with pytest.raises(Parked, match="usb"):
            await b.tap("iphone12", 5, 5)
        clock.t += 29  # T - 1
        phone.present.add("U12")
        await w.poll_usb()  # re-enumerated in time: re-check and resume
        await b.tap("iphone12", 5, 5)

        phone.present.clear()
        await w.poll_usb()
        clock.t += 31  # T + 1
        await w.poll_usb()
        clock.t += 120
        await w.poll_usb()
        assert handoffs == []  # Ian unplugs the phone to use it (grill G3-G17): logged, the task backs out, no page
        phone.present.add("U12")
        await w.poll_usb()
        with pytest.raises(TaskEnded, match="device_lost"):
            await b.tap("iphone12", 5, 5)  # that task is over; a new one may start
        await b.close()

    asyncio.run(go())
    assert inputs(phone) == ["launch", "tap"]
    detaches = events(log, "detach")
    assert len(detaches) == 2 and detaches[0]["battery"] == 64 and detaches[0]["temperature"] == 31.5
    assert [e["event"] for e in events(log) if e["event"] in ("attach", "device_lost")] == ["attach", "device_lost", "attach"]


def test_thermal_pause_above_40C(rig):
    phone, b, w, clock, handoffs, log = rig

    async def go():
        await b.launch("iphone12", IG)
        w.battery("U12", {"temperature": 35.0, "level": 80, "charging": True})
        assert events(log) == []  # 35 °C: nothing
        w.battery("U12", {"temperature": 41.0, "level": 80, "charging": True})
        with pytest.raises(Parked, match="thermal"):
            await b.tap("iphone12", 5, 5)
        w.battery("U12", {"temperature": 39.5, "level": 80, "charging": True})  # cooling, not cool yet
        with pytest.raises(Parked, match="thermal"):
            await b.tap("iphone12", 5, 5)
        w.battery("U12", {"temperature": 37.0, "level": 80, "charging": True})
        await asyncio.sleep(0)
        await b.tap("iphone12", 5, 5)
        await b.close()

    asyncio.run(go())
    assert [e["event"] for e in events(log)] == ["thermal_pause", "thermal_resume"]


def test_low_phone_battery_mid_task_parks_with_one_notice(rig):
    phone, b, w, clock, handoffs, log = rig

    async def go():
        await b.launch("iphone12", IG)
        for level in (19, 18, 17):
            w.battery("U12", {"temperature": 30.0, "level": level, "charging": False})
        with pytest.raises(Parked, match="battery"):
            await b.tap("iphone12", 5, 5)
        await b.close()

    asyncio.run(go())
    assert len(handoffs) == 1 and "battery" in handoffs[0]


def test_the_clock_gap_catches_a_missed_sleep_once(tmp_path, authhub):
    phone, slept = Watched(), {"s": 0.0}
    b = Bridge(phone, config(authhub[0]), log_path=tmp_path / "calls.jsonl")
    w = Watch(b, events=tmp_path / "events.jsonl", gap=lambda: slept["s"])

    async def go():
        await b.launch("iphone12", IG)
        slept["s"] += 600  # IOKit's will-sleep never came: the clock gap shows the sleep
        await w.tick()
        await w.tick()
        w.will_sleep()  # a sleep IOKit did report, wake included...
        slept["s"] += 300
        await w.did_wake()
        await w.tick()  # ...is not handled twice by the fallback
        await b.close()

    asyncio.run(go())
    assert [e["event"] for e in events(tmp_path / "events.jsonl") if e["event"] in ("will_sleep", "did_wake")] == [
        "will_sleep", "did_wake", "will_sleep", "did_wake"]


def test_a_flapping_cable_alerts_once_an_hour(rig):
    phone, b, w, clock, handoffs, log = rig

    async def go():
        await w.poll_usb()
        for _ in range(7):  # out and back within seconds, over and over: not a person using the phone
            phone.present.clear()
            await w.poll_usb()
            clock.t += 8
            phone.present.add("U12")
            await w.poll_usb()
            clock.t += 300
        await b.close()

    asyncio.run(go())
    assert len(handoffs) == 1 and "bad cable" in handoffs[0] and len(events(log, "usb_flapping")) == 1


def test_one_second_blips_count_as_flapping(rig):
    phone, b, w, clock, handoffs, log = rig

    async def go():
        await w.poll_usb()
        for _ in range(8):  # 1 s data-link drops, cable in (15 of them 19:05-19:20Z): a bad cable or port
            phone.present.clear()
            await w.poll_usb()
            clock.t += 1
            phone.present.add("U12")
            await w.poll_usb()
            clock.t += 120
        await b.close()

    asyncio.run(go())
    assert len(handoffs) == 1 and "bad cable" in handoffs[0] and len(events(log, "attach")) == 8


def test_gone_with_the_cable_in_is_logged_never_paged(tmp_path, authhub):
    """Pulling the Lightning end leaves the USB-C plug in the Mac, which still presents a device (Ian, 01:06Z
    2026-10-19): indistinguishable from a link failure, so it is logged once and never paged."""
    phone, clock, handoffs = Watched(), Clock(), []
    b = Bridge(phone, config(authhub[0]), log_path=tmp_path / "c.jsonl")
    w = Watch(b, clock=clock, wall=clock, handoff=handoffs.append, events=tmp_path / "e.jsonl", plugged=lambda: True)

    async def go():
        await w.poll_usb()
        phone.present.clear()
        for _ in range(10):  # 10 min off usbmux, cable "in"
            await w.poll_usb()
            clock.t += 60
        await b.close()

    asyncio.run(go())
    assert handoffs == [] and len(events(tmp_path / "e.jsonl", "usb_gone_cable_in")) == 1


def test_two_wake_signals_are_handled_once(rig):
    phone, b, w, clock, handoffs, log = rig

    async def go():
        w.will_sleep()
        clock.t += 134
        await asyncio.gather(w._woke(), w._woke())  # IOKit's powered-on and the clock-gap fallback, together
        await b.close()

    asyncio.run(go())
    assert len(events(log, "did_wake")) == 1 and len(events(log, "recheck")) == len(b.config.devices)  # once each
