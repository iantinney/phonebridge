"""Fleet regressions from independent review: a faulty phone must not disable the host watchdog."""

import asyncio

from phonebridge.host.watch import Watch
from test_bridge import FakeBackend
from test_fleet import fleet


class Phones(FakeBackend):
    def __init__(self, udids):
        super().__init__()
        self.attached_now = set(udids)
        self.hang_lock, self.hang_battery = set(), set()
        self.battery_started = asyncio.Event()
        self.lock_started = asyncio.Event()

    async def devices(self):
        return [{"udid": u, "paired": True} for u in self.attached_now]

    async def attached(self):
        return list(self.attached_now)

    async def lock_state(self, udid):
        if udid in self.hang_lock:
            self.lock_started.set()
            await asyncio.Event().wait()
        return {"locked": False}

    async def battery(self, udid):
        if udid in self.hang_battery:
            self.battery_started.set()
            await asyncio.Event().wait()
        return {"temperature": 30, "level": 90, "charging": True}


def rig(tmp_path, monkeypatch):
    b = fleet(tmp_path, monkeypatch)
    b.backend = Phones(d.udid for d in b.config.devices.values())
    now, notices = [0.0], []
    w = Watch(b, events=tmp_path / "events.jsonl", gap=lambda: 0, clock=lambda: now[0], handoff=notices.append)
    return b, w, now, notices


def test_one_hung_phone_times_out_on_wake_without_leaving_peers_parked(tmp_path, monkeypatch):
    b, w, _, _ = rig(tmp_path, monkeypatch)
    monkeypatch.setattr("phonebridge.host.watch.PROBE_TIMEOUT_S", 0.05)
    b.backend.hang_lock.add(b.config.resolve("phone3").udid)

    async def run():
        try:
            w.will_sleep()
            result = await asyncio.wait_for(w.did_wake(0), 1)
            assert not result["phone3"]["resumed"] and b.parked("phone3") == {"usb"}
            assert all(not b.parked(a) for a in b.config.devices if a != "phone3")
        finally:
            await b.close()
    asyncio.run(run())


def test_usb_loss_is_seen_while_another_phones_battery_probe_hangs(tmp_path, monkeypatch):
    b, w, _, _ = rig(tmp_path, monkeypatch)
    b.backend.hang_battery.add(b.config.resolve("phone3").udid)

    async def run():
        job = asyncio.create_task(w.run(usb_every=0.01, battery_every=1000))
        try:
            await asyncio.wait_for(b.backend.battery_started.wait(), 1)
            b.backend.attached_now.remove(b.config.resolve("phone5").udid)
            async def parked():
                while "usb" not in b.parked("phone5"):
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(parked(), 1)
            assert not b.parked("phone6")
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
            await b.close()
    asyncio.run(run())


def test_one_blip_per_phone_does_not_become_six_blips_on_one_phone(tmp_path, monkeypatch):
    b, w, now, notices = rig(tmp_path, monkeypatch)

    async def run():
        try:
            await w.poll_usb()
            for d in list(b.config.devices.values())[:6]:
                b.backend.attached_now.remove(d.udid)
                await w.poll_usb()
                now[0] += 1
                b.backend.attached_now.add(d.udid)
                await w.poll_usb()
                now[0] += 1
            assert not notices
            assert all(len(drops) == 1 for drops in w.drops.values())
        finally:
            await b.close()
    asyncio.run(run())


def test_a_hung_reconnect_does_not_delay_a_peer_detach(tmp_path, monkeypatch):
    b, w, _, _ = rig(tmp_path, monkeypatch)
    stuck = b.config.resolve("phone3").udid
    peer = b.config.resolve("phone5").udid
    b.backend.hang_lock.add(stuck)

    async def run():
        await w.poll_usb()
        b.backend.attached_now.remove(stuck)
        await w.poll_usb()
        b.backend.attached_now.add(stuck)
        job = asyncio.create_task(w.run(usb_every=0.01, battery_every=1000))
        try:
            await asyncio.wait_for(b.backend.lock_started.wait(), 1)
            b.backend.attached_now.remove(peer)
            async def parked():
                while "usb" not in b.parked("phone5"):
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(parked(), 1)
            assert b.parked("phone3") == {"usb"} and not b.parked("phone6")
            assert len(w.drops[stuck]) == 1  # retries aren't physical cable flaps
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
            assert all(task.done() for task in w._rechecks.values())
            await b.close()
    asyncio.run(run())


def test_usb_and_wake_checks_accept_the_same_phones_normalized_ids(tmp_path, monkeypatch):
    b, w, _, _ = rig(tmp_path, monkeypatch)
    b.backend.attached_now = {u.replace("-", "").lower() for u in b.backend.attached_now}

    async def run():
        try:
            await w.poll_usb()
            assert not w.gone
            w.will_sleep()
            result = await w.did_wake(0)
            assert all(r["resumed"] for r in result.values())
            assert all(not b.parked(a) for a in b.config.devices)
        finally:
            await b.close()
    asyncio.run(run())
