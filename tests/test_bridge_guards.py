"""DEV-03 bridge guards: the foreground-app rule, stop after one unrecognised flick, the operator guard off the
gesture path. The keypad incident (R5, 15:34-15:37Z): Instagram lost the foreground and ~65 s of flicks
landed on the Phone keypad."""

import asyncio
import dataclasses
import json
import time

import pytest

from phonebridge.bridge import Bridge, NotFrontmost, OperatorActive, UnfamiliarScreen
from phonebridge.humanize import Dist
from test_bridge import EXACT, OTHER, FakeBackend, authhub, config, quick  # noqa: F401  (authhub: a fixture)

IG, HOME_APP, PHONE = "com.burbn.instagram", "com.apple.springboard", "com.apple.mobilephone"


class Phone(FakeBackend):
    """FakeBackend plus the frontmost-app query; `fg` is what is frontmost, `fg_log` when it was asked."""

    def __init__(self, fg=IG, **kw):
        super().__init__(**kw)
        self.fg, self.fg_log, self.input_log = fg, [], []

    async def foreground(self, udid):
        self.fg_log.append(time.monotonic())
        return self.fg

    async def launch(self, udid, bundle_id, url=None):
        await super().launch(udid, bundle_id, url)
        self.fg = bundle_id

    async def tap(self, udid, x, y, press):
        self.input_log.append(time.monotonic())
        await super().tap(udid, x, y, press)

    async def swipe(self, udid, path):
        self.input_log.append(time.monotonic())
        await super().swipe(udid, path)


def inputs(phone):
    return [c[0] for c in phone.calls]


def records(b):
    return [json.loads(r) for r in b.log_path.open()]


@pytest.fixture
def phone():
    return Phone()


@pytest.fixture
def bridge(tmp_path, authhub, phone):
    return Bridge(phone, config(authhub[0]), log_path=tmp_path / "calls.jsonl")


async def every_input(b):
    """Try each kind of input once; return the refusals."""
    out = []
    for op, args in (("tap", (5, 5)), ("swipe", (60, 200, 60, 80)), ("type", ("hi",)), ("paste", ("hi",)), ("home", ())):
        with pytest.raises(NotFrontmost) as e:
            await getattr(b, op)("iphone12", *args)
        out.append(e.value)
    return out


def test_no_input_when_leased_app_not_frontmost(bridge, phone):
    async def go():
        await bridge.launch("iphone12", IG)  # the task's app: the lease is now for Instagram
        phone.fg = HOME_APP  # Instagram exits to the home screen mid-feed
        refused = await every_input(bridge)
        phone.fg = PHONE  # ...or another app took the foreground
        refused += await every_input(bridge)
        await bridge.close()
        return refused

    refused = asyncio.run(go())
    assert inputs(phone) == ["launch"]  # zero input after the app lost the foreground
    assert {e.foreground for e in refused} == {HOME_APP, PHONE} and {e.app for e in refused} == {IG}
    assert "only launch(com.burbn.instagram)" in str(refused[0])
    refusals = [r for r in records(bridge) if not r["ok"]]
    assert len(refusals) == 10 and {r["foreground"] for r in refusals} == {HOME_APP, PHONE}  # the event names it


def test_launch_allowed_from_home(bridge, phone):
    async def go():
        await bridge.launch("iphone12", IG)
        phone.fg = HOME_APP
        with pytest.raises(NotFrontmost):
            await bridge.scroll("iphone12")
        await bridge.launch("iphone12", IG)  # the one allowed action: back into the leased app
        await bridge.scroll("iphone12")  # then input flows again
        await bridge.close()

    asyncio.run(go())
    assert inputs(phone) == ["launch", "launch", "swipe"]


def test_without_a_task_app_there_is_no_foreground_rule(bridge, phone):
    phone.fg = HOME_APP  # e.g. the deploy smoke opening Clock from the home screen
    asyncio.run(bridge.tap("iphone12", 5, 5))
    assert inputs(phone) == ["tap"]


def test_stop_after_one_unrecognised_flick(tmp_path, authhub, phone):
    """A flick that lands on an unfamiliar screen (glossary) stops flicking: the next flick is refused with zero input.
    Recognition is the frontmost app until the screens lane plugs in a recogniser (here: a fake one)."""
    seen = {"ok": True}
    b = Bridge(phone, config(authhub[0]), log_path=tmp_path / "calls.jsonl",
               recognise=lambda app, png: "feed" if seen["ok"] else None)

    async def go():
        await b.launch("iphone12", IG)
        await b.scroll("iphone12")
        seen["ok"] = False  # the flick opened something the recogniser doesn't know
        await b.screen("iphone12")
        with pytest.raises(UnfamiliarScreen, match="unfamiliar.*flick"):
            await b.scroll("iphone12")
        await b.tap("iphone12", 5, 5)  # a tap may resolve it (Q3: the VLM prescribes one) ...
        with pytest.raises(UnfamiliarScreen):
            await b.scroll("iphone12")  # ... but no more flicks until the screen is recognised
        seen["ok"] = True
        await b.screen("iphone12")
        await b.scroll("iphone12")

        phone.fg = PHONE  # the keypad incident: the flick itself left Instagram
        with pytest.raises(NotFrontmost):
            await b.scroll("iphone12")
        phone.fg = IG  # even if Instagram comes back on its own, flicking stays stopped ...
        with pytest.raises(UnfamiliarScreen):
            await b.scroll("iphone12")
        await b.launch("iphone12", IG)  # ... until the task launches it again
        await b.scroll("iphone12")
        await b.close()

    asyncio.run(go())
    assert inputs(phone) == ["launch", "swipe", "tap", "swipe", "launch", "swipe"]


def test_foreground_is_checked_at_most_1s_before_each_input(tmp_path, authhub, phone):
    slow = dataclasses.replace(EXACT, think={"tap": Dist(1800, 0, 1800, 1800)})  # think longer than the 1 s bound
    b = Bridge(phone, config(authhub[0], humanize=slow), log_path=tmp_path / "calls.jsonl")

    async def go():
        await b.launch("iphone12", IG)
        for _ in range(2):
            await b.tap("iphone12", 5, 5)
        await b.close()

    asyncio.run(go())
    for at in phone.input_log:
        assert 0 <= at - max(t for t in phone.fg_log if t <= at) <= 1.0
    taps = [r for r in records(b) if r["op"] == "tap"]
    assert all(r["foreground"] == IG and r["fg_age_ms"] <= 1000 for r in taps)


def test_the_operator_guard_judges_the_last_look_not_a_new_screenshot(tmp_path, authhub, phone):
    """DEV-03: no screenshot round-trip before a gesture. A look after our input settled is the evidence."""
    b = quick(Bridge(phone, config(authhub[0]), log_path=tmp_path / "calls.jsonl"))
    b.op_backoff = 5.0  # outlasts the confirming frame below

    async def go():
        await b.launch("iphone12", IG)
        await asyncio.sleep(0.1)  # settled: the baseline
        await b.screen("iphone12")  # the agent looks (unchanged)
        shots = phone.shots
        await b.tap("iphone12", 5, 5)
        assert phone.shots == shots  # nothing taken on the gesture path
        await asyncio.sleep(0.1)
        phone.current = OTHER  # the operator opens something else
        await b.screen("iphone12")  # the agent's look shows it
        await asyncio.sleep(0.6)  # (the verdict's confirming frame is taken off the gesture path)
        shots = phone.shots
        with pytest.raises(OperatorActive):
            await b.tap("iphone12", 6, 6)
        assert phone.shots == shots
        await b.close()

    asyncio.run(go())
    assert inputs(phone) == ["launch", "tap"]


def test_foreground_is_a_read_only_call(bridge, phone):
    phone.fg = PHONE
    assert asyncio.run(bridge.foreground("iphone12")) == PHONE and inputs(phone) == []
    assert records(bridge)[-1]["foreground"] == PHONE


def test_frontmost_from_the_process_list():
    """Recorded on the iPhone 12 (iOS 27.0.1, 18:5xZ): SpringBoard, InputUI and a widget renderer also run in
    the foreground; the app is the one that isn't a UI service."""
    from phonebridge.backends.pmd3 import frontmost

    def procs(*apps):
        rows = [("com.apple.springboard", False, "/System/Library/CoreServices/SpringBoard.app/SpringBoard"),
                ("com.apple.InputUI", True, "/Applications/InputUI.app/InputUI"), *apps]
        return [{"bundleIdentifier": b, "isApplication": a, "realAppName": r, "foregroundRunning": True} for b, a, r in rows] + [
            {"bundleIdentifier": "com.apple.mobilephone", "isApplication": True, "foregroundRunning": False}]

    assert frontmost(procs(("com.apple.chrono.WidgetRenderer-Default", True, "/Applications/WidgetRenderer_Default.app/x"))) == HOME_APP
    assert frontmost(procs(("com.apple.Preferences", True, "/Applications/Preferences.app/Preferences"))) == "com.apple.Preferences"
    assert frontmost(procs((IG, True, "/var/containers/Bundle/Application/DE58/Instagram.app/Instagram"),
                           ("com.apple.Preferences", True, "/Applications/Preferences.app/Preferences"))) == IG
    with pytest.raises(NotImplementedError):
        frontmost([{"bundleIdentifier": IG, "isApplication": True}])


def test_without_the_operator_watch_no_screen_is_judged(bridge, phone):
    """The served daemon's default (grill G3-G17): Ian unplugs the phone to use it; USB loss parks instead."""
    bridge.op_watch, bridge.op_gap = False, 0.0

    async def go():
        await bridge.launch("iphone12", IG)
        phone.current = OTHER  # would have been "the operator"
        await bridge.screen("iphone12")
        shots = phone.shots
        await bridge.tap("iphone12", 5, 5)
        assert phone.shots == shots  # nothing taken, nothing judged
        await bridge.close()

    asyncio.run(go())
    assert inputs(phone) == ["launch", "tap"]
    assert not any("operator_check" in r for r in records(bridge))


def test_every_input_names_its_backend(bridge, phone):
    asyncio.run(bridge.tap("iphone12", 5, 5))
    assert records(bridge)[-1]["backend"] == "Phone"  # INP-01: each production touch tagged with its hands
