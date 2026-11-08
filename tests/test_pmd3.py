import pytest
import asyncio
import contextlib

from phonebridge.backends import pmd3

from test_bridge import tiny_png


class FakeCtx:
    def __init__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_screenshot(self):
        return tiny_png(1170, 2532)


def test_first_call_of_a_session_can_be_a_screenshot(monkeypatch):
    """Regression: screen() as the very first call used p.stack before _rsd() created it."""
    async def fake_rsd(self, p):
        if p.rsd is None:
            p.stack, p.rsd = contextlib.AsyncExitStack(), object()
        return p.rsd

    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", fake_rsd)
    monkeypatch.setattr(pmd3, "DvtProvider", FakeCtx)
    monkeypatch.setattr(pmd3, "Screenshot", FakeCtx)
    b = pmd3.Pmd3Backend()
    png = asyncio.run(b.screen("U12"))
    assert pmd3.png_size(png) == (1170, 2532) and b._phone("U12").size == (1170, 2532)


class FakeHid:
    def __init__(self):
        self.events = []

    async def send_report(self, service_id, report):  # the backend now sends its own 58-byte touch report
        import struct
        assert service_id == pmd3.TOUCHSCREEN_SERVICE and len(report) == 58
        await self.send_touchscreen(report[3], *struct.unpack("<HH", report[4:8]))

    async def send_touchscreen(self, state, x, y):
        self.events.append((state, x, y))


def test_a_cancelled_tap_still_lifts_the_finger(monkeypatch):
    """A finger left down reads as a long press: cancellation mid-tap must still send RELEASE."""
    hid = FakeHid()

    async def fake_hid(self, p):
        return hid

    async def fake_norm(self, p, x, y):
        return 100, 200

    async def slow_sleep(_):
        await asyncio.Event().wait()  # never returns: the call is cancelled here, finger down

    monkeypatch.setattr(pmd3.Pmd3Backend, "_hid", fake_hid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_norm", fake_norm)
    b = pmd3.Pmd3Backend()

    async def go():
        monkeypatch.setattr(pmd3.asyncio, "sleep", slow_sleep)
        task = asyncio.ensure_future(b.tap("U12", 1, 2, 0.08))
        for _ in range(5):
            await asyncio.wait([asyncio.ensure_future(asyncio.Event().wait())], timeout=0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert hid.events == [(pmd3.TOUCHSCREEN_STATE_CONTACT, 100, 200), (pmd3.TOUCHSCREEN_STATE_RELEASE, 100, 200)]



# --- the touch report, byte for byte (Apple's DigitizerReport as the phone reads it) --------------------------------

def test_touch_report_puts_the_timestamp_where_the_phone_reads_it():
    import struct
    from pymobiledevice3.remote.core_device.hid_service import build_touchscreen_report
    r = pmd3.touch_report(pmd3.TOUCHSCREEN_STATE_CONTACT, 0x1234, 0xABCD, timestamp=0x0102030405060708)
    assert len(r) == 58 and r[0] == 0x09 and r[1:3] == b"\x01\x05" and r[3] == 0xC2
    assert struct.unpack("<HH", r[4:8]) == (0x1234, 0xABCD)
    assert r[8:40] == bytes(32)  # contacts 2-5 and the scroll collection untouched
    assert r[40:45] == b"\x02\x00\x00\x00\x00"  # identities per slot, nothing stray in slot 4
    assert struct.unpack("<q", r[45:53])[0] == 0x0102030405060708  # remoteTimestamp at byte 45, Int64
    assert r[53:] == bytes(5)
    # pymobiledevice3 11.34.1 writes the same first 44 bytes but its timestamp one byte early (6 bytes at 44)
    old = build_touchscreen_report(pmd3.TOUCHSCREEN_STATE_CONTACT, 0x1234, 0xABCD, timestamp=0x060708)
    assert old[:44] == r[:44] and old[44:50] == (0x060708).to_bytes(6, "little")
    assert struct.unpack("<q", old[45:53])[0] == 0x060708 >> 8  # what the phone made of it: the timestamp / 256


def test_release_report_keeps_the_release_state_and_position():
    r = pmd3.touch_report(pmd3.TOUCHSCREEN_STATE_RELEASE, 7, 9, timestamp=1)
    assert r[3] == 0x02 and r[4:8] == b"\x07\x00\x09\x00" and r[45:53] == (1).to_bytes(8, "little")


def test_tap_sends_reports_to_the_touchscreen_surface(monkeypatch):
    sent = []

    class Hid:
        async def send_report(self, service_id, report):
            sent.append((service_id, report))

    async def fake_hid(self, p):
        return Hid()

    async def fake_norm(self, p, x, y):
        return int(x), int(y)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_hid", fake_hid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_norm", fake_norm)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", lambda self, p: None)
    asyncio.run(pmd3.Pmd3Backend().tap("udid", 10, 20, 0.01))
    assert [s for s, _ in sent] == [257, 257] and sent[0][1][3] == 0xC2 and sent[1][1][3] == 0x02
    assert len(sent[0][1]) == 58 and sent[0][1][45:53] != bytes(8)


def test_timestamps_are_mach_ticks_so_deltas_match_device_hubs(monkeypatch):
    """Device Hub writes the Mac's mach_absolute_time (24 MHz). Off macOS we convert the monotonic clock to the same
    scale: a 16.7 ms frame is ~400,000 ticks (41.67 ns each), never 16.7 million nanoseconds."""
    import time as _t
    monkeypatch.setattr(pmd3.sys, "platform", "linux")
    a = pmd3.mach_ticks(); _t.sleep(0.0167); b = pmd3.mach_ticks()
    assert 380_000 <= b - a <= 5_000_000  # ≥ 16.7 ms of ticks, with room for a loaded machine's late wake-up
    r1 = pmd3.touch_report(0xC2, 1, 1); r2 = pmd3.touch_report(0xC2, 1, 1)
    d = int.from_bytes(r2[45:53], "little") - int.from_bytes(r1[45:53], "little")
    assert 0 <= d < 24_000_000  # back to back: well under a second of ticks



def test_indigo_tap_sends_start_then_end_digitizer_events(monkeypatch):
    sent = []

    class Svc:
        async def send_request(self, msg):
            sent.append(msg)

    class Indigo:
        service = Svc()

    async def fake_hid(self, p):
        return object()

    async def fake_service(self, p, attr, cls):
        return Indigo()
    monkeypatch.setattr(pmd3.Pmd3Backend, "_hid", fake_hid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_service", fake_service)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", lambda self, p: None)
    asyncio.run(pmd3.Pmd3Backend().tap_indigo("udid", 583.0, 784.0, 0.01))
    assert [m["messageType"] for m in sent] == ["IndigoDigitizerEvent"] * 2
    assert [int(m["payload"]["eventType"]) for m in sent] == [0, 2]
    assert sent[0]["payload"]["pointOne"] == {"x": 583.0, "y": 784.0}
    assert sent[0]["featureIdentifier"] == "com.apple.coredevice.feature.remote.hid.digitizer"


# --- the tunnel and the eyes come from the config (tunneld installed on the Mac 2026-10-19) -----------------------

def test_tunnel_tunneld_uses_tunneld_and_never_falls_back(monkeypatch):
    calls = []

    class Rsd:
        async def close(self):
            pass

    async def tunneld(udid):
        calls.append(("tunneld", udid))
        return Rsd()

    class Userspace:
        def __init__(self, **kw):
            calls.append(("userspace",))
    monkeypatch.setattr(pmd3, "get_tunneld_device_by_udid", tunneld)
    monkeypatch.setattr(pmd3, "UserspaceRsdTunnel", Userspace)
    be = pmd3.Pmd3Backend(tunnel="tunneld")
    rsd = asyncio.run(be._open_rsd(be._phone("U12")))
    assert isinstance(rsd, Rsd) and calls == [("tunneld", "U12")]

    async def none(udid):
        return None
    monkeypatch.setattr(pmd3, "get_tunneld_device_by_udid", none)
    with pytest.raises(RuntimeError, match="tunneld"):
        asyncio.run(be._open_rsd(be._phone("U13")))
    assert ("userspace",) not in calls


def test_tunnel_userspace_skips_tunneld(monkeypatch):
    async def tunneld(udid):
        raise AssertionError("tunneld must not be asked")
    monkeypatch.setattr(pmd3, "get_tunneld_device_by_udid", tunneld)
    be = pmd3.Pmd3Backend(tunnel="userspace")
    assert be.tunnel == "userspace"
    with pytest.raises(ValueError):
        pmd3.Pmd3Backend(tunnel="wifi")


def test_config_carries_eyes_and_tunnel(tmp_path):
    from phonebridge.config import Config
    p = tmp_path / "pb.toml"
    p.write_text('eyes = "stream"\ntunnel = "tunneld"\n')
    c = Config.load(p)
    assert (c.eyes, c.tunnel) == ("stream", "tunneld")
    assert (Config().eyes, Config().tunnel) == ("auto", "auto")


def test_indigo_swipe_sends_start_positions_end_on_the_paths_clock(monkeypatch):
    sent = []

    class Svc:
        async def send_request(self, msg):
            sent.append((round(asyncio.get_running_loop().time(), 3), msg))

    class Indigo:
        service = Svc()

    async def fake_hid(self, p):
        return object()

    async def fake_service(self, p, attr, cls):
        return Indigo()
    monkeypatch.setattr(pmd3.Pmd3Backend, "_hid", fake_hid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_service", fake_service)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", lambda self, p: None)
    path = [(100.0, 900.0, 0.0), (100.0, 800.0, 0.05), (100.0, 600.0, 0.1)]
    asyncio.run(pmd3.Pmd3Backend().swipe_indigo("udid", path))
    kinds = [int(m["payload"]["eventType"]) for _, m in sent]
    assert kinds == [0, 1, 1, 2] and sent[-1][1]["payload"]["pointOne"] == {"x": 100.0, "y": 600.0}
    assert sent[2][0] - sent[0][0] >= 0.09  # the positions follow the path's clock


def test_pointer_surface_remove_asks_dtuhidd_and_reports_whether_it_is_gone():
    sent = []

    class Svc:
        async def send_request(self, msg):
            sent.append(msg)

    class Hid:
        service = Svc()

        async def list_connected_services(self):
            return {"connectedServices": [{"_ServiceID": 257}, {"_ServiceID": 512}]}
    be = pmd3.Pmd3Backend(pointer_surface="remove")
    assert asyncio.run(be._remove_pointer_surface(Hid())) is True
    assert sent[0]["payload"] == {"removeService": {"_0": pmd3.POINTER_SURFACE}}
    with pytest.raises(ValueError):
        pmd3.Pmd3Backend(pointer_surface="hide")


# --- a dead cached connection: home retries once, taps never (orchestrator, 2026-10-19 07:09Z) ----------------------


class Conn:
    """A button/HID connection the phone may have closed: `fail_at` = the send number (1-based) that raises."""

    def __init__(self, log, fail_at=None):
        self.log, self.fail_at, self.n = log, fail_at, 0

    async def send_button(self, *args):
        self.n += 1
        if self.fail_at == self.n:
            raise pmd3_exc().ConnectionTerminatedError()
        self.log.append(("button", args[-1]))


def pmd3_exc():
    from pymobiledevice3 import exceptions
    return exceptions


@pytest.fixture
def conns(monkeypatch):
    state = {"log": [], "conns": [], "resets": 0}

    async def fake_service(self, p, attr, cls):
        return state["conns"][min(state["resets"], len(state["conns"]) - 1)]

    async def fake_reset(self, p):
        state["resets"] += 1
    monkeypatch.setattr(pmd3.Pmd3Backend, "_service", fake_service)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_reset", fake_reset)
    return state


def test_home_on_a_dead_connection_presses_once_on_a_fresh_one(conns):
    conns["conns"] = [Conn(conns["log"], fail_at=1), Conn(conns["log"])]
    asyncio.run(pmd3.Pmd3Backend().home("udid"))
    assert conns["resets"] == 1
    assert conns["log"] == [("button", pmd3.HID_BUTTON_STATE_DOWN), ("button", pmd3.HID_BUTTON_STATE_UP)]


def test_home_whose_release_failed_only_releases_on_retry(conns):
    conns["conns"] = [Conn(conns["log"], fail_at=2), Conn(conns["log"])]
    asyncio.run(pmd3.Pmd3Backend().home("udid"))
    assert conns["log"] == [("button", pmd3.HID_BUTTON_STATE_DOWN), ("button", pmd3.HID_BUTTON_STATE_UP)]  # no second press


def test_home_gives_up_after_one_retry(conns):
    conns["conns"] = [Conn(conns["log"], fail_at=1), Conn(conns["log"], fail_at=1)]
    with pytest.raises(pmd3_exc().ConnectionTerminatedError):
        asyncio.run(pmd3.Pmd3Backend().home("udid"))
    assert conns["resets"] == 2 and conns["log"] == []


def test_a_tap_on_a_dead_connection_is_never_retried(monkeypatch):
    calls = {"hid": 0, "resets": 0}

    async def fake_hid(self, p):
        calls["hid"] += 1
        raise pmd3_exc().ConnectionTerminatedError()

    async def fake_reset(self, p):
        calls["resets"] += 1

    async def fake_norm(self, p, x, y):
        return round(x), round(y)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_norm", fake_norm)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_hid", fake_hid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_reset", fake_reset)
    with pytest.raises(pmd3_exc().ConnectionTerminatedError):
        asyncio.run(pmd3.Pmd3Backend().tap("udid", 100, 100, 0.05))
    assert calls == {"hid": 1, "resets": 1}


# --- a hung DVT call recycles only DVT (R1-IG attempt 2: a 5 s proclist hang + a full reset held a swipe 7.5 s) -------


def test_a_hung_frontmost_query_recycles_only_the_dvt_channel_quickly(monkeypatch):
    opened, full_resets = [], []
    procs = [{"foregroundRunning": True, "isApplication": True, "bundleIdentifier": "com.burbn.instagram",
              "realAppName": "/private/var/containers/Bundle/Application/X/Instagram.app"}]

    class Ctx:
        def __init__(self, *a):
            opened.append(type(self).__name__)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class FakeDvt(Ctx):
        pass

    class FakeInfo(Ctx):
        async def proclist(self):
            if opened.count("FakeInfo") == 1:
                await asyncio.sleep(30)  # the dead channel: never answers
            return procs

    async def fake_rsd(self, p):
        p.stack = p.stack or contextlib.AsyncExitStack()
        p.rsd = object()
        return p.rsd

    async def fake_reset(self, p):
        full_resets.append(1)
    monkeypatch.setattr(pmd3, "DvtProvider", FakeDvt)
    monkeypatch.setattr(pmd3, "DeviceInfo", FakeInfo)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", fake_rsd)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_reset", fake_reset)
    be = pmd3.Pmd3Backend()
    p = be._phone("udid")
    p.session = session = object()  # the touch session must survive a DVT hiccup

    async def go():
        t = asyncio.get_running_loop().time()
        fg = await be.foreground("udid")
        return fg, asyncio.get_running_loop().time() - t
    fg, took = asyncio.run(go())
    assert fg == "com.burbn.instagram"
    assert took < pmd3.FG_TIMEOUT_S + 0.5  # was 5 s + a full reset + a new touch session
    assert opened == ["FakeDvt", "FakeInfo", "FakeDvt", "FakeInfo"]  # a fresh DVT channel, nothing else reopened
    assert full_resets == [] and p.session is session
    assert [(r["call"], r["reset"]) for r in be.recoveries] == [("fg", "dvt")] and "TimeoutError" in be.recoveries[0]["error"]


# --- the capture's start recreates tunneld's tunnel (R1-IG attempts 2 and 3) ------------------------------------------


def test_a_recreated_tunnel_is_noticed_before_a_call_and_reconnects_keeping_the_capture(monkeypatch):
    be = pmd3.Pmd3Backend()
    p = be._phone("udid")
    p.rsd, p.rsd_addr, p.stack = object(), "fd64::1", contextlib.AsyncExitStack()
    stopped, seen = [], []
    monkeypatch.setattr(be, "stop_eyes", lambda reason="": stopped.append(reason))

    async def addr(udid):
        return "fd70::1"  # tunneld made a new tunnel
    monkeypatch.setattr(be, "_tunnel_addr", addr)

    async def call(p):
        seen.append((p.rsd, p.rsd_addr))
        return "ok"
    assert asyncio.run(be._run("udid", call, retry=False)) == "ok"
    assert seen == [(None, None)]  # the stale RSD was dropped before the call, which reconnects
    assert stopped == []  # the capture (USB) is kept


def test_a_failed_call_keeps_the_capture_while_the_phone_is_on_usb_and_stops_it_when_it_left(monkeypatch):
    be = pmd3.Pmd3Backend()
    stopped = []
    monkeypatch.setattr(be, "stop_eyes", lambda reason="": stopped.append(reason))
    on_usb = {"udid"}

    async def attached():
        return list(on_usb)
    monkeypatch.setattr(be, "attached", attached)

    async def boom(p):
        raise ConnectionError("dead")
    with pytest.raises(ConnectionError):
        asyncio.run(be._run("udid", boom, retry=False))
    assert stopped == []
    on_usb.clear()
    with pytest.raises(ConnectionError):
        asyncio.run(be._run("udid", boom, retry=False))
    assert stopped == ["reset: the phone left USB"]


def test_after_the_capture_starts_the_new_tunnel_is_awaited_and_dvt_warmed(monkeypatch):
    be = pmd3.Pmd3Backend()
    p = be._phone("udid")
    p.rsd, p.rsd_addr, p.stack = object(), "fd64::1", contextlib.AsyncExitStack()
    answers = iter(["fd64::1", "fd64::1", "fd70::1"])  # the old tunnel, then tunneld's new one
    warmed = []

    async def addr(udid):
        return next(answers, "fd70::1")

    class Info:
        def __init__(self, dvt):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

        async def proclist(self):
            warmed.append(p.rsd_addr)
            return []

    async def fake_dvt(self, p):
        p.dvt_stack = p.dvt_stack or contextlib.AsyncExitStack()
        p.dvt = p.dvt or object()
        return p.dvt

    async def fake_reset(self, p, keep_eyes=False):
        p.rsd = p.rsd_addr = None
        warmed.append(("reset", keep_eyes))
    monkeypatch.setattr(be, "_tunnel_addr", addr)
    monkeypatch.setattr(pmd3, "DeviceInfo", Info)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_dvt", fake_dvt)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_reset", fake_reset)

    async def fake_session(self, p):
        warmed.append("touch session")
    monkeypatch.setattr(pmd3.Pmd3Backend, "_session", fake_session)
    asyncio.run(be._rewarm_after_capture({"udid": "fd64::1"}))
    assert warmed[0] == ("reset", True) and warmed[2] == "touch session" and len(warmed) == 3  # reconnect, DVT, touch


def test_keep_warm_restarts_the_idle_timers_without_opening_anything(monkeypatch):
    """R1-IG attempt 7: a 70 s hold let the touch session and the capture idle out (45 s), so the next input paid a
    cold start. keep_warm restarts the timers of what is open; nothing closed is reopened."""
    be = pmd3.Pmd3Backend(idle=0.2)
    closed, stopped = [], []

    async def fake_close(self, p):
        closed.append(p.udid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_close_session", fake_close)
    monkeypatch.setattr(be, "_stop_avf", lambda: stopped.append(1))

    async def go():
        loop = asyncio.get_running_loop()
        p = be._phone("udid")
        p.session = object()
        p.idle_timer = loop.call_later(0.2, lambda: closed.append("first"))
        be._avf = object()
        be._avf_timer = loop.call_later(0.2, lambda: stopped.append("first"))
        await asyncio.sleep(0.12)
        be.keep_warm("udid")
        await asyncio.sleep(0.15)  # past the first deadline: nothing idled out
        assert closed == [] and stopped == []
        await asyncio.sleep(0.15)  # past the restarted one: both idle out
        assert closed == ["udid"] and stopped == [1]
        be._avf = None
        q = be._phone("other")  # nothing open: keep_warm opens nothing
        be.keep_warm("other")
        assert q.session is None and q.idle_timer is None
    asyncio.run(go())


def _input_fields(desc: bytes) -> dict:
    """{(usage_page, usage): (bit offset after the report id, bits)} for every variable Input of a HID descriptor."""
    fields, page, size, count, usages, bit, i = {}, 0, 0, 0, [], 0, 0
    while i < len(desc):
        b = desc[i]; n = (0, 1, 2, 4)[b & 3]; v = int.from_bytes(desc[i + 1:i + 1 + n], "little"); i += 1 + n
        kind, tag = (b >> 2) & 3, b >> 4
        if kind == 1:
            page, size, count = (v if tag == 0 else page), (v if tag == 7 else size), (v if tag == 9 else count)
        elif kind == 2 and tag == 0:
            usages.append(v)
        elif kind == 0:
            if tag == 8 and not v & 1:
                for k in range(count):
                    fields[(page, usages[min(k, len(usages) - 1)])] = (bit + k * size, size)
            bit += size * count if tag == 8 else 0
            usages = []
    fields["total"] = bit
    return fields


def test_the_virtual_digitizer_descriptor_lays_out_apples_struct_plus_a_contact_size():
    """The probe only means something if the descriptor's X/Y sit where the 58-byte DigitizerReport has them and its
    Width/Height in bytes 9-12 (the struct's unused contact 2)."""
    f = _input_fields(pmd3.VDIG_DESCRIPTOR)
    assert f["total"] == 57 * 8  # 58 bytes with the report id
    assert f[(0x0D, 0x54)] == (0, 8) and f[(0x0D, 0x55)] == (8, 8)  # contact count, maximum: bytes 1, 2
    assert f[(0x0D, 0x51)] == (16, 5) and f[(0x0D, 0x42)] == (22, 1) and f[(0x0D, 0x32)] == (23, 1)  # 0xC2 = touch + range
    assert f[(0x01, 0x30)] == (24, 16) and f[(0x01, 0x31)] == (40, 16)  # X, Y: bytes 4-7
    assert f[(0x0D, 0x47)] == (56, 1) and f[(0x0D, 0x48)] == (64, 16) and f[(0x0D, 0x49)] == (80, 16)
    r = pmd3.sized_touch_report(0xC2, 1000, 2000, 9.5)
    base = pmd3.touch_report(0xC2, 1000, 2000)
    assert len(r) == 58 and r[:8] == base[:8] and r[13:45] == base[13:45]
    assert r[8] == 1 and r[9:13] == bytes([95, 0, 95, 0])


def test_vdigitizer_tap_registers_the_service_once_then_sends_sized_contact_and_release(monkeypatch):
    sent, created = [], []

    class Svc:
        async def send_receive_request(self, msg):
            created.append(msg["payload"]["createService"]["_0"])
            return {"serviceID": 0x100002002}

    class Hid:
        service = Svc()

        async def send_report(self, sid, report):
            sent.append((sid, report))

    async def fake_hid(self, p):
        return Hid()

    async def fake_norm(self, p, x, y):
        return 1000, 2000
    monkeypatch.setattr(pmd3.Pmd3Backend, "_hid", fake_hid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_norm", fake_norm)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", lambda self, p: None)
    monkeypatch.setattr(pmd3, "KEYBOARD_ATTACH_SETTLE_SECONDS", 0)
    b = pmd3.Pmd3Backend()

    async def go():
        await b.tap_vdigitizer("udid", 100.0, 200.0, 0.01, 8.0)
        await b.tap_vdigitizer("udid", 100.0, 200.0, 0.01, 11.0)
    asyncio.run(go())
    assert len(created) == 1 and int(created[0]["PrimaryUsagePage"]) == 0x0D and int(created[0]["PrimaryUsage"]) == 0x04
    assert created[0]["_CoreDevice_codablePropertyStorage"]["ReportDescriptor"]["data"] == pmd3.VDIG_DESCRIPTOR
    assert [(sid, r[3], r[11]) for sid, r in sent] == [(0x100002002, 0xC2, 80), (0x100002002, 0x02, 80),
                                                       (0x100002002, 0xC2, 110), (0x100002002, 0x02, 110)]


def test_the_disk_image_is_mounted_when_coredevice_services_are_missing_even_with_dvt_listed():
    """iOS 27.2 (live 2026-11-08): DVT listed, no CoreDevice services, nothing mounted; every input failed."""
    from phonebridge.backends.pmd3 import needs_ddi

    dvt = "com.apple.instruments.dtservicehub"
    assert needs_ddi({"Services": {dvt: {}, "com.apple.dt.remoteFetchSymbols": {}}})
    assert needs_ddi({"Services": {}})
    assert not needs_ddi({"Services": {dvt: {}, "com.apple.coredevice.appservice": {}, "com.apple.coredevice.deviceinfo": {}}})
