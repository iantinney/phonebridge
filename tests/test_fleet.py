"""Seven simulated phones: real Bridge/leases, no claim of seven-device hardware validation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import subprocess
import sys
import threading
import time

import pytest
from PIL import Image

from phonebridge.bridge import Bridge, Parked, StaleInput
from phonebridge.config import Config, Device
from phonebridge.humanize import HumanConfig
from phonebridge.lease import LeaseError, Leases
from phonebridge.mcp_server import build
from test_bridge import EXACT, FakeBackend, tiny_png
from test_seven_phones import fleet as driver_fleet, UDIDS, COLORS, DISPLAYS, open_all, paint, session, hid_point
from phonebridge.screens.store import Store


def fleet(tmp_path, monkeypatch, human=EXACT):
    monkeypatch.setenv("PHONEBRIDGE_STATE", str(tmp_path / "state"))
    devices = {f"phone{i}": Device(f"phone{i}", f"00008101-{i:016X}", f"device:phone{i}")
               for i in range(1, 8)}
    cfg = Config(devices=devices, lease_cmd=[sys.executable, "-m", "phonebridge.local_lease"],
                 humanize=human, tunnel="tunneld", eyes="stream")
    bridge = Bridge(FakeBackend(), cfg, log_path=tmp_path / "calls.jsonl")
    bridge.op_watch = False
    return bridge


def test_seven_phones_run_concurrently_with_real_local_leases(tmp_path, monkeypatch):
    b = fleet(tmp_path, monkeypatch)
    mcp = build(b)

    async def run():
        arrived, release = asyncio.Queue(), asyncio.Event()
        calls = []

        async def tap(udid, x, y, press):
            await arrived.put(udid)
            await release.wait()
            calls.append((udid, x, y))

        b.backend.tap = tap
        for d in b.config.devices.values():
            b._awake[d.udid] = time.monotonic()
        jobs = [asyncio.create_task(mcp.call_tool("tap", {"device": alias, "x": i, "y": i + 10,
                                                        "controller": f"agent{i}"}))
                for i, alias in enumerate(b.config.devices, 1)]
        try:
            # All seven reach the backend before any is allowed to finish: proves overlap, not elapsed-time luck.
            started = [await asyncio.wait_for(arrived.get(), 5) for _ in jobs]
            assert set(started) == {d.udid for d in b.config.devices.values()}
            assert len(b.tasks) == len(b.leases._held) == 7
            with pytest.raises(LeaseError, match="controller"):
                await b.tap("phone1", 99, 99, caller="other-agent")
            release.set()
            results = await asyncio.gather(*jobs)
            assert all(not r.is_error for r in results)
            assert sorted(calls) == [(d.udid, i, i + 10)
                                     for i, d in enumerate(b.config.devices.values(), 1)]
        finally:
            release.set()
            await asyncio.gather(*jobs, return_exceptions=True)
            await b.close()
        assert not b.tasks and not b.leases._held and not b._reservations
    asyncio.run(run())


def test_seven_screens_and_a_disconnected_phone_are_isolated(tmp_path, monkeypatch):
    b = fleet(tmp_path, monkeypatch)
    shots = {d.udid: tiny_png(100 + i, 200 + i, bytes((i, 0, 0)))
             for i, d in enumerate(b.config.devices.values(), 1)}
    failed = b.config.resolve("phone4").udid

    async def screen(udid):
        if udid == failed:
            raise ConnectionError("unplugged")
        return shots[udid]
    b.backend.screen = screen

    async def run():
        try:
            results = await asyncio.gather(*(b.screen(a) for a in b.config.devices), return_exceptions=True)
            for i, (d, result) in enumerate(zip(b.config.devices.values(), results), 1):
                if d.udid == failed:
                    assert isinstance(result, ConnectionError)
                else:
                    assert result == shots[d.udid]
                    assert b._size[d.udid] == (100 + i, 200 + i)
            b.park("usb_lost", "phone4")
            with pytest.raises(Parked):
                await b.home("phone4", caller="agent4")
            await b.home("phone5", caller="agent5")
            assert b.backend.calls == [("home", b.config.resolve("phone5").udid)]
        finally:
            await b.close()
    asyncio.run(run())


def test_same_phone_inputs_serialize_checks_and_gestures(tmp_path, monkeypatch):
    b = fleet(tmp_path, monkeypatch)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        active = 0
        peak = 0

        async def home(udid):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            entered.set()
            await release.wait()
            active -= 1
        b.backend.home = home
        first = asyncio.create_task(b.home("phone1", caller="agent1"))
        await entered.wait()
        second = asyncio.create_task(b.home("phone1", caller="agent1"))
        try:
            async def queued():
                while True:
                    lock = b._inputs.get(b.config.resolve("phone1").udid)
                    if lock and lock._waiters and len(lock._waiters) == 1:
                        return
                    await asyncio.sleep(0)
            await asyncio.wait_for(queued(), 2)  # prove the second call reached the busy phone
            assert active == 1
            release.set()
            await asyncio.gather(first, second)
            assert peak == 1
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)
            await b.close()
    asyncio.run(run())


def test_one_slow_lease_does_not_block_the_other_six():
    leases = Leases(["fake-lease"])
    entered, release = threading.Event(), threading.Event()

    def command(verb, *args):
        if verb == "acquire" and args[0] == "device:phone1":
            entered.set()
            assert release.wait(5)
        return subprocess.CompletedProcess([], 0, stdout="test-token", stderr="")
    leases._run = command
    with ThreadPoolExecutor(max_workers=7) as pool:
        slow = pool.submit(leases.ensure, "device:phone1")
        try:
            assert entered.wait(2)
            peers = [pool.submit(leases.ensure, f"device:phone{i}") for i in range(2, 8)]
            for peer in peers:
                peer.result(timeout=2)
        finally:
            release.set()
        slow.result(timeout=2)
    assert len(leases._held) == 7


def test_a_peer_recovery_cannot_stale_this_phones_input(tmp_path, monkeypatch):
    b = fleet(tmp_path, monkeypatch)
    b.backend.recoveries = []
    requested = b.config.resolve("phone1").udid
    peer = b.config.resolve("phone2").udid

    async def run():
        async def check(device, udid, identity, rec):
            b.backend.recoveries.append({"n": len(b.backend.recoveries) + 1, "udid": peer,
                                        "call": "screen", "error": "peer disconnected"})
        b._operator_check = check
        b._awake[requested] = time.monotonic()
        try:
            await b.tap("phone1", 10, 20)
            assert b.backend.calls == [("tap", requested, 10, 20)]
            # Recovery of the actual target still refuses before input.
            nonlocal peer
            peer = requested
            with pytest.raises(StaleInput, match="recovered"):
                await b.tap("phone1", 30, 40)
            assert len(b.backend.calls) == 1
        finally:
            await b.close()
    asyncio.run(run())
    rows = [json.loads(line) for line in b.log_path.read_text().splitlines()]
    assert "recoveries" not in rows[0]


def test_each_phone_has_independent_bounded_gestures(tmp_path, monkeypatch):
    b = fleet(tmp_path, monkeypatch, HumanConfig(seed=123))
    first = b._human(b.config.resolve("phone1").udid)
    # A clean stream must reproduce phone1 even if six peers have consumed arbitrarily many draws.
    expected = [first.tap(500, 800, 44, 20) for _ in range(10)]
    b._humans.clear()
    for d in list(b.config.devices.values())[1:]:
        other = b._human(d.udid)
        for _ in range(50):
            other.tap(1, 2)
            other.keystrokes("example")
    actual = [b._human(b.config.resolve("phone1").udid).tap(500, 800, 44, 20) for _ in range(10)]
    assert actual == expected
    b._humans.clear()
    starts = [tuple(b._human(d.udid).tap(500, 800, 44, 20) for _ in range(10))
              for d in b.config.devices.values()]
    assert len(set(starts)) == 7  # independent seeds, not seven copies of the same sequence
    assert all(abs(x - 500) <= 0.4 * 44 and abs(y - 800) <= 0.4 * 20 and 0.04 <= p <= 0.25
               for x, y, p in actual)
    assert len(set(actual)) > 1


def test_changed_driver_settings_keep_refusing_until_restart(tmp_path):
    path = tmp_path / "phonebridge.toml"
    path.write_text('eyes = "stream"\ntunnel = "tunneld"\n')
    b = Bridge(FakeBackend(), config_path=path, log_path=tmp_path / "calls.jsonl")
    assert b.config.eyes == "stream"
    path.write_text('eyes = "dvt"\ntunnel = "tunneld"\n')
    for _ in range(2):
        with pytest.raises(ValueError, match="restart the bridge"):
            _ = b.config
    assert b.backend.calls == []


def test_config_edit_after_delivery_does_not_turn_a_tap_into_a_failure(tmp_path):
    path = tmp_path / "phonebridge.toml"
    udid = "00008101-0000000000000001"
    path.write_text(f'[devices.phone]\nudid = "{udid}"\n[lease]\nenforce = false\n')
    b = Bridge(FakeBackend(), config_path=path, log_path=tmp_path / "calls.jsonl")
    b.op_watch = False

    async def tap(target, x, y, press):
        b.backend.calls.append(("tap", target))
        path.write_text("[devices.phone\n")  # an unrelated editor writes invalid TOML during delivery
    b.backend.tap = tap

    async def run():
        b._awake[udid] = time.monotonic()
        try:
            await b.tap("phone", 10, 20, think=False)
            assert b.backend.calls == [("tap", udid)]
            with pytest.raises(ValueError):
                await b.tap("phone", 30, 40)
            assert len(b.backend.calls) == 1
        finally:
            await b.close()  # cleanup also uses the identity acquired before the edit
    asyncio.run(run())


def test_look_pixels_taps_and_a_lost_lease_stay_on_their_phone(driver_fleet, tmp_path, monkeypatch):
    """Actual Bridge/Eyes/Pmd3 routing with fake USB/tunnels/HID, and real local lease subprocesses."""
    monkeypatch.setenv("PHONEBRIDGE_STATE", str(tmp_path / "state"))
    cfg = Config(devices={f"phone{i}": Device(f"phone{i}", u, f"device:phone{i}") for i, u in enumerate(UDIDS)},
                 lease_cmd=[sys.executable, "-m", "phonebridge.local_lease"], humanize=EXACT,
                 tunnel="tunneld", eyes="stream")
    b = Bridge(driver_fleet.backend, cfg, log_path=tmp_path / "calls.jsonl")
    b.op_watch = False
    b.eyes_from(Store(records=[], corpus=tmp_path / "corpus"), ocr=lambda *a, **kw: [])

    async def run():
        try:
            frames = await open_all(b.backend)
            for i, frame in enumerate(frames):
                paint(frame, i)
                b._awake[UDIDS[i]] = time.monotonic()
            looks = await asyncio.gather(*(b.look(a, settle=False) for a in cfg.devices))
            for i, (alias, look) in enumerate(zip(cfg.devices, looks)):
                png, meta = await b.frame(alias, format="png", seq=look["frame_seq"])
                im = Image.open(io.BytesIO(png)).convert("RGB")
                assert im.size == DISPLAYS[i] and im.getpixel((1, 1)) == COLORS[i]
                assert (meta["width"], meta["height"]) == DISPLAYS[i]
            await asyncio.gather(*(b.tap(a, 10 + i, 20 + i, caller=a, think=False)
                                   for i, a in enumerate(cfg.devices)))
            for i, u in enumerate(UDIDS):
                reports = session(b.backend, u).hid.reports
                assert len(reports) == 2
                assert all(r[1:] == hid_point(10 + i, 20 + i, i) for r in reports)
            identity = cfg.resolve("phone3").lease
            token, _ = b.leases._held[identity]
            b.leases._held[identity] = (token, -math.inf)  # a renewal was lost
            with pytest.raises(LeaseError, match="expired"):
                await b.tap("phone3", 12, 25, caller="phone3", think=False)
            assert len(session(b.backend, UDIDS[3]).hid.reports) == 2
            await b.tap("phone4", 12, 25, caller="phone4", think=False)
            assert len(session(b.backend, UDIDS[4]).hid.reports) == 4
        finally:
            await b.close()
    asyncio.run(run())


def test_device_alias_cannot_switch_physical_phones_until_restart(tmp_path):
    path = tmp_path / "phonebridge.toml"
    original = '[devices.phone]\nudid = "00008101-0000000000000001"\n'
    path.write_text(original)
    b = Bridge(FakeBackend(), config_path=path, log_path=tmp_path / "calls.jsonl")
    assert b.config.resolve("phone").udid.endswith("1")
    path.write_text(original.replace("0000000000000001", "0000000000000002"))
    for _ in range(2):
        with pytest.raises(ValueError, match="restart the bridge"):
            _ = b.config
    assert b.backend.calls == []
