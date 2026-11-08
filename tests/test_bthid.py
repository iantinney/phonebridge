import asyncio
import json

import pytest

from phonebridge.backends import bthid
from phonebridge.backends.bthid import BtHidBackend, drag_path

from test_bridge import FakeBackend, tiny_png

ADDR = "34-fe-77-de-15-84"


class FakeDaemon:
    """bthidd's socket protocol, in-process."""

    def __init__(self, paired=True, linked=True):
        self.paired, self.linked, self.requests = paired, linked, []

    async def serve(self, path):
        async def client(r, w):
            while line := await r.readline():
                req = json.loads(line)
                self.requests.append(req)
                resp = {"ok": True}
                if req["op"] == "status":
                    resp["phones"] = [{"address": ADDR, "name": "iPhone", "paired": self.paired, "interrupt": self.linked}]
                elif req["op"] == "connect":
                    self.linked = True
                elif req["op"] == "send":
                    resp["sent"] = len(req["reports"])
                w.write((json.dumps(resp) + "\n").encode())
                await w.drain()
        return await asyncio.start_unix_server(client, path)

    def sent(self):
        return [bytes.fromhex(r[0]) for q in self.requests if q["op"] == "send" for r in q["reports"]]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    async def make(**kw):
        d = FakeDaemon(**kw)
        sock = tmp_path / "bthid.sock"
        server = await d.serve(str(sock))
        eyes = FakeBackend()
        eyes.screen = lambda udid: asyncio.sleep(0, tiny_png(1170, 2532))
        b = BtHidBackend(eyes, sock=sock, connect_timeout=2)

        async def usb_setup(udid):
            return ADDR, "iPhone13,2"
        monkeypatch.setattr(b, "_usb_setup", usb_setup)
        return b, d, server
    return make


def test_tap_moves_absolute_then_clicks_on_the_relative_report(rig):
    async def go():
        b, d, server = await rig()
        await b.tap("U12", 1169, 0, 0.08)
        await b.close(); server.close()
        return d.sent()
    reports = asyncio.run(go())
    assert reports[0] == bytes([2, 0, 0xFF, 0x7F, 0, 0])  # far right, top: (32767, 0)
    presses = [r for r in reports if r[0] == 3]
    assert [r[1] for r in presses] == [1, 0]  # press then release, no movement
    assert all(r[2:4] == b"\0\0" for r in presses)


def test_type_sends_shifted_keys_and_releases(rig):
    async def go():
        b, d, server = await rig()
        await b.type("U12", [("H", 0.05, 0.0), ("i", 0.06, 0.2)])
        await b.close(); server.close()
        return d.requests
    reqs = asyncio.run(go())
    reports = next(r["reports"] for r in reqs if r["op"] == "send")
    up = bytes([1] + [0] * 8).hex()
    assert reports == [[bytes([1, 2, 0, 0, 0, 0, 0, 0, 0]).hex(), 30],  # Shift first, on its own
                       [bytes([1, 2, 0, 0x0B, 0, 0, 0, 0, 0]).hex(), 50], [up, 200],  # hold, then the pause before "i"
                       [bytes([1, 0, 0, 0x0C, 0, 0, 0, 0, 0]).hex(), 60], [up, 0]]
    with pytest.raises(ValueError):
        asyncio.run(BtHidBackend(FakeBackend()).type("U12", [("é", 0.05, 0.0)]))


def test_unpaired_phone_opens_pairing_and_says_how(rig):
    async def go():
        b, d, server = await rig(paired=False, linked=False)
        with pytest.raises(RuntimeError, match="Settings > Bluetooth"):
            await b.tap("U12", 10, 10, 0.08)
        await b.close(); server.close()
        return d.requests
    ops = [r["op"] for r in asyncio.run(go())]
    assert "discoverable" in ops and "send" not in ops


def test_dropped_link_is_reconnected(rig):
    async def go():
        b, d, server = await rig(linked=False)
        await b.home("U12")
        await b.close(); server.close()
        return d.requests
    reqs = asyncio.run(go())
    ops = [r["op"] for r in reqs]
    assert ops.index("connect") < ops.index("send")
    assert next(r for r in reqs if r["op"] == "send")["awake"] == 60  # bthidd keeps the phone from auto-locking


def test_drag_path_covers_the_distance_and_keeps_the_timing():
    path = [(500, 1800 - 900 * i / 18, 0.3 * i / 18) for i in range(19)]  # 300 pt up on an @3x phone in 0.3 s
    steps = drag_path(path, 3)
    assert all(abs(dx) <= bthid.DRAG_STEP and abs(dy) <= bthid.DRAG_STEP for dx, dy, _ in steps)
    assert sum(dx for dx, _, _ in steps) == 0
    per = lambda d: d * (bthid.DRAG_A + bthid.DRAG_B * d)
    travel = sum(per(abs(dy)) for _, dy, _ in steps) * bthid.DRAG_GAIN
    assert 285 < travel < 315
    assert abs(sum(ms for _, _, ms in steps) - (300 - bthid.REPORT_COST_MS * len(steps))) < 1e-6  # each report's cost taken off
    assert drag_path([(0, 0, 0), (0, 0, 0.1)], 3) == [(0, 0, 100 - bthid.REPORT_COST_MS)]  # a hold keeps its time


def test_a_swipe_plays_the_path_as_a_drag_with_press_hold_release(rig):
    async def go():
        b, d, server = await rig()
        await b.swipe("U12", [(585, 1800, 0), (585, 1500, 0.1), (585, 900, 0.3)])
        await b.close(); server.close()
        return next(r["reports"] for r in d.requests if r["op"] == "send")
    reports = asyncio.run(go())
    rel = [(bytes.fromhex(r[0]), r[1]) for r in reports if bytes.fromhex(r[0])[0] == 3]
    assert rel[0][0][1] == 1 and rel[-1][0][1] == 0  # pressed first, released last
    drag = [ms for raw, ms in rel[1:-2] if raw[1] == 1]
    assert 295 < sum(drag) + bthid.REPORT_COST_MS * len(drag) < 305  # the path's own 300 ms, not a fixed gap


def test_assistive_touch_switched_on_by_bthid_is_switched_off_on_close(monkeypatch):
    """The AssistiveTouch floating button is a visible pointer and an app-readable flag (UIAccessibility.isAssistiveTouchRunning):
    bthid switches it on for its own use and must switch it off again when it is done (trial 1, 2026-10-19)."""
    import asyncio
    from phonebridge.backends import bthid as mod

    state = {"on": False, "sets": []}

    class LD:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get_assistive_touch(self):
            return state["on"]

        async def set_assistive_touch(self, v):
            state["on"] = v
            state["sets"].append(v)

        async def get_value(self, key):
            return {"BluetoothAddress": "AA:BB:CC:DD:EE:FF", "ProductType": "iPhone13,2"}[key]

    async def fake_create(udid, autopair=False):
        return LD()
    monkeypatch.setattr(mod, "create_using_usbmux", fake_create)

    class Eyes:
        async def close(self):
            pass
    be = mod.BtHidBackend(eyes=Eyes(), keystrokes=lambda t: [])
    asyncio.run(be._usb_setup("U12"))
    assert state["on"] is True and be._assistive_on == {"U12"}
    asyncio.run(be.close())
    assert state["on"] is False and state["sets"] == [True, False] and be._assistive_on == set()
