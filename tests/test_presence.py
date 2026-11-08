"""Phone presence, measured not inferred: a cheap real round trip (usbmux listing + tunneld listing + one lockdown
GetValue) within ~2 s, cached ~10 s; exposed by the bridge, the MCP tool and GET /presence on the daemon."""
import asyncio
import json

import pytest

from phonebridge import presence as pres

UDID = "00008101-AAAAAAAAAAAAAAAA"


class Dev:
    def __init__(self, serial, usb=True):
        self.serial, self.is_usb = serial, usb
        self.connection_type = "USB" if usb else "Network"


class LD:
    def __init__(self, name="iPhone", slow=0.0, fail=None):
        self.name, self.slow, self.fail = name, slow, fail

    async def __aenter__(self):
        if self.fail:
            raise self.fail
        await asyncio.sleep(self.slow)
        return self

    async def __aexit__(self, *exc):
        pass

    async def get_value(self, domain=None, key=None):
        return {"DeviceName": self.name, "ProductVersion": "27.0.1"}[key]


@pytest.fixture
def world(monkeypatch):
    state = {"devices": [Dev(UDID)], "tunneld": {UDID: [{"tunnel-address": "fd::1"}]}, "ld": LD(), "probes": 0}

    async def list_devices():
        state["probes"] += 1
        return state["devices"]

    async def create(udid, autopair=False, connection_type=None):
        return state["ld"]

    async def tunneld(url, timeout):
        if state["tunneld"] is None:
            raise OSError("tunneld down")
        return state["tunneld"]
    monkeypatch.setattr(pres, "list_devices", list_devices)
    monkeypatch.setattr(pres, "create_using_usbmux", create)
    monkeypatch.setattr(pres, "tunneld_listing", tunneld)
    return state


def test_attached_over_usb_with_a_tunnel_and_lockdown_answering(world):
    p = asyncio.run(pres.probe(UDID))
    assert p["attached"] is True and p["transport"] == "usb" and p["tunnel"] is True and p["lockdown"] is True
    assert p["name"] == "iPhone" and p["ios"] == "27.0.1" and p["ms"] < 2000 and p["ts"] > 1.7e9


def test_off_usb_is_reported_as_none_without_a_lockdown_attempt(world):
    world["devices"] = []
    world["ld"] = LD(fail=AssertionError("must not open lockdown for a phone that isn't listed"))
    p = asyncio.run(pres.probe(UDID))
    assert p["attached"] is False and p["transport"] == "none" and p["lockdown"] is False


def test_wifi_only_and_tunneld_down_are_told_apart(world):
    world["devices"] = [Dev(UDID, usb=False)]
    world["tunneld"] = None
    p = asyncio.run(pres.probe(UDID))
    assert p["transport"] == "wifi" and p["tunnel"] is None and "tunneld" in p["errors"]


def test_a_hung_lockdown_is_cut_at_the_budget(world):
    world["ld"] = LD(slow=5.0)
    p = asyncio.run(pres.probe(UDID, budget=0.3))
    assert p["attached"] is True and p["lockdown"] is False and p["ms"] < 1000 and "lockdown" in p["errors"]


def test_presence_is_cached_for_ten_seconds_unless_fresh(world):
    cache = pres.Presence(ttl=10.0)
    a = asyncio.run(cache.get(UDID))
    b = asyncio.run(cache.get(UDID))
    assert world["probes"] == 1 and b["cached"] is True and a["ts"] == b["ts"]
    asyncio.run(cache.get(UDID, fresh=True))
    assert world["probes"] == 2


def test_bridge_presence_adds_the_daemon_and_the_last_seen_state(world, tmp_path):
    from phonebridge.bridge import Bridge
    from test_bridge import FakeBackend, config
    b = Bridge(FakeBackend(), config([]), log_path=tmp_path / "calls.jsonl")
    p = asyncio.run(b.presence("iphone12"))
    assert p["device"] == "iphone12" and set(p["daemon"]) >= {"sha", "pid", "started"}
    assert "locked" in p and "foreground" in p
    assert '"op": "presence"' in b.log_path.read_text()


def test_http_presence_route(world, tmp_path):
    import httpx2 as httpx
    from phonebridge import mcp_server
    from phonebridge.bridge import Bridge
    from test_bridge import FakeBackend, config
    world["devices"].append(Dev("U12"))  # the test config's iphone12
    world["tunneld"]["U12"] = [{"tunnel-address": "fd::2"}]
    b = Bridge(FakeBackend(), config([]), log_path=tmp_path / "calls.jsonl")
    app = mcp_server.with_presence_route(mcp_server.build(b).streamable_http_app(host="127.0.0.1"), b)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as c:
            r = await c.get("/presence", params={"device": "iphone12"})
            assert r.status_code == 200 and r.json()["attached"] is True and r.json()["transport"] == "usb"
            bad = await c.get("/presence", params={"device": "nope"})
            assert bad.status_code == 404
    asyncio.run(go())
    assert "presence" in {t.name for t in asyncio.run(mcp_server.build(b).list_tools())}
