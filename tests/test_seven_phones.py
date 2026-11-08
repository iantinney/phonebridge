"""Seven phones on one Mac, offline: the supported path is tunneld + each phone's own display stream. Real asyncio
concurrency and (where it says so) the real receiver and decoder; the phones, tunnels and USB are fakes, so this proves
the driver's isolation and failure handling, not how seven physical iPhones behave."""
import asyncio
import contextlib
import io
import random
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from phonebridge import avf, stream
from phonebridge.backends import pmd3
from pymobiledevice3.exceptions import ConnectionTerminatedError
from stream_helpers import hevc_aus, rtp_packets, solid
from test_bridge import tiny_png
from test_stream import FakeTransport

UDIDS = [f"00008101-0000000000000{i:03d}" for i in range(7)]
DISPLAYS = [(64 + 4 * i, 128 + 6 * i) for i in range(7)]  # no two phones share a size
COLORS = [(10 + 30 * i, 200 - 25 * i, 40 + 5 * i) for i in range(7)]
SERVICES = {"Services": {s: {} for s in (pmd3.DvtProvider.RSD_SERVICE_NAME, *pmd3.DDI_SERVICES)}}


def index(udid):
    return UDIDS.index(udid)


class Rsd:
    def __init__(self, udid):
        self.udid, self.peer_info, self.closed = udid, SERVICES, False

    async def close(self):
        self.closed = True


class Hid:
    """One phone's touch surface: what reached it, plus the ways a phone goes wrong."""

    def __init__(self):
        self.reports, self.gate, self.fail = [], None, False

    async def send_report(self, service_id, report):
        if self.fail:
            self.reports.append("failed")
            raise ConnectionTerminatedError()
        if self.gate:
            await self.gate.wait()
        await asyncio.sleep(0)  # let the other phones' gestures interleave with this one's
        self.reports.append((report[3], int.from_bytes(report[4:6], "little"), int.from_bytes(report[6:8], "little")))


class Stream:
    """stream.ScreenStream for one phone: opened on that phone's RSD, with its own frames and touch surface."""

    def __init__(self, rsd, display=None, **kw):
        self.rsd, self.frames, self.hid, self.closed = rsd, stream.Frames(), Hid(), False
        self.receiver = SimpleNamespace(display=display, set_display=lambda d: setattr(self.receiver, "display", d))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True


class Dvt:
    RSD_SERVICE_NAME = pmd3.DvtProvider.RSD_SERVICE_NAME

    def __init__(self, target):
        self.udid = target.udid

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_screenshot(self):
        return tiny_png(*DISPLAYS[index(self.udid)])


@pytest.fixture
def fleet(monkeypatch):
    state = SimpleNamespace(rsds={}, opened=[], on_usb=list(UDIDS))

    async def tunneld(udid):
        state.opened.append(udid)
        state.rsds[udid] = Rsd(udid)
        return state.rsds[udid]

    async def addr(self, udid):
        return f"fd{index(udid)}::1"

    async def attached(self):
        return list(state.on_usb)
    monkeypatch.setattr(pmd3, "tunneld_rsd", tunneld)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_tunnel_addr", addr)
    monkeypatch.setattr(pmd3.Pmd3Backend, "attached", attached)
    monkeypatch.setattr(pmd3, "ScreenStream", Stream)
    monkeypatch.setattr(pmd3, "DvtProvider", Dvt)
    monkeypatch.setattr(pmd3, "Screenshot", Dvt)
    state.backend = pmd3.Pmd3Backend(idle=60, eyes="stream", tunnel="tunneld")
    return state


def session(be, udid):
    return be._phone(udid).session


def paint(frames, i):
    w, h = DISPLAYS[i]
    r, g, b = COLORS[i]
    frames.push(bytes((b, g, r, 255)) * (w * h), w, h)


def hid_point(x, y, i):
    w, h = DISPLAYS[i]
    return round(x * pmd3.HID_MAX / w), round(y * pmd3.HID_MAX / h)


async def open_all(be):
    return await asyncio.gather(*(be.frames(u) for u in UDIDS))


# --- the screens and the hands stay with their phone ---------------------------------------------------------------

def test_seven_phones_at_once_each_see_only_their_own_screen(fleet):
    async def go():
        be = fleet.backend
        frames = await open_all(be)
        assert len({id(f) for f in frames}) == 7 and sorted(fleet.opened) == UDIDS  # one RSD and one stream per phone
        for u in UDIDS:
            assert session(be, u).rsd.udid == u and session(be, u).receiver.display == DISPLAYS[index(u)]
        order = list(range(7))
        random.Random(7).shuffle(order)
        for i in order:
            paint(frames[i], i)
        pngs = await asyncio.gather(*(be.screen(u) for u in UDIDS))
        for i, png in enumerate(pngs):
            im = Image.open(io.BytesIO(png)).convert("RGB")
            assert im.size == DISPLAYS[i] and im.getpixel((1, 1)) == COLORS[i]
        await be.close()
    asyncio.run(go())


def test_seven_concurrent_swipes_arrive_whole_on_their_own_phone_by_its_own_geometry(fleet):
    async def go():
        be = fleet.backend
        await open_all(be)
        paths = [[(10 + i, 20 + 5 * k, 0.002 * k) for k in range(6)] for i in range(7)]
        await asyncio.gather(*(be.swipe(u, paths[i]) for i, u in enumerate(UDIDS)))
        for i, u in enumerate(UDIDS):
            got = session(be, u).hid.reports
            want = [(pmd3.TOUCHSCREEN_STATE_CONTACT, *hid_point(x, y, i)) for x, y, _ in paths[i]]
            assert got[:-1] == want and got[-1] == (pmd3.TOUCHSCREEN_STATE_RELEASE, *want[-1][1:])
        await be.close()
    asyncio.run(go())


def test_seven_real_receivers_and_decoders_at_once_keep_their_own_pictures_and_rtcp():
    sizes = [(64 + 16 * i, 128) for i in range(7)]
    streams = [rtp_packets(hevc_aus([solid(*sizes[i], COLORS[i])] * 3, keyint=3)) for i in range(7)]

    async def go():
        parts = []
        for i in range(7):
            tr, frames = FakeTransport(streams[i], idle=0.3), stream.Frames()
            rx = stream.Receiver(tr, frames, decoder="av", local_ssrc=100 + i, remote_ssrc=200 + i,
                                 rtcp_dest=(f"fd{i}::2", 6000 + i), rr_interval=0.05)
            parts.append((tr, frames, rx))
        await asyncio.gather(*(rx.run() for _, _, rx in parts))
        await asyncio.sleep(0.2)
        for i, (tr, frames, _) in enumerate(parts):
            last = await frames.wait(after=2, timeout=2.0)
            assert (last.width, last.height) == sizes[i] and frames.stats["frames"] == 3
            assert last.image.getpixel((sizes[i][0] // 2, 64)) == pytest.approx(COLORS[i], abs=14)
            assert tr.sent and {(ip, port) for _, ip, port in tr.sent} == {(f"fd{i}::2", 6000 + i)}
            assert all(int.from_bytes(d[4:8], "big") == 100 + i for d, *_ in tr.sent)
    asyncio.run(go())


def test_a_new_display_size_reaches_only_that_phones_decoder(fleet):
    async def go():
        be = fleet.backend
        await open_all(be)
        p = be._phone(UDIDS[0])
        await be._dvt_png(p)
        session(be, UDIDS[0]).receiver.set_display((1, 1))
        assert session(be, UDIDS[1]).receiver.display == DISPLAYS[1]
        await be._dvt_png(p)  # a still restores this phone's own size, nobody else's
        assert session(be, UDIDS[0]).receiver.display == DISPLAYS[0] and be._phone(UDIDS[1]).display == DISPLAYS[1]
        await be.close()
    asyncio.run(go())


def test_a_frame_of_the_wrong_geometry_refuses_that_phones_input_alone(fleet):
    async def go():
        be = fleet.backend
        frames = await open_all(be)
        for i, f in enumerate(frames):
            paint(f, i)
        w, h = DISPLAYS[2]
        frames[2].push(b"\xff" * ((w + 40) * h * 4), w + 40, h)  # bright margin: not padding
        results = await asyncio.gather(*(be.tap(u, 5, 5, 0.001) for u in UDIDS), return_exceptions=True)
        assert isinstance(results[2], RuntimeError) and "zero input" in str(results[2])
        assert [r for i, r in enumerate(results) if i != 2] == [None] * 6
        for i, u in enumerate(UDIDS):
            if i != 2:
                assert len(session(be, u).hid.reports) == 2
        await be.close()
    asyncio.run(go())


# --- a slow or gone phone is that phone's problem --------------------------------------------------------------------

def test_a_wedged_touch_does_not_hold_up_the_other_six(fleet):
    async def go():
        be = fleet.backend
        await open_all(be)
        stuck = UDIDS[3]
        session(be, stuck).hid.gate = asyncio.Event()
        wedged = asyncio.ensure_future(be.tap(stuck, 1, 1, 0.001))
        await asyncio.sleep(0.02)
        others = [u for u in UDIDS if u != stuck]
        await asyncio.wait_for(asyncio.gather(*(be.tap(u, 1, 1, 0.001) for u in others)), 1.0)
        assert not wedged.done() and all(len(session(be, u).hid.reports) == 2 for u in others)
        wedged.cancel()
        session(be, stuck).hid.gate.set()  # the finger's release is shielded: it finishes before the cancel does
        with contextlib.suppress(asyncio.CancelledError):
            await wedged
    asyncio.run(go())


def test_a_tunnel_that_never_opens_does_not_hold_up_the_other_six(fleet, monkeypatch):
    never = asyncio.Event
    stuck = UDIDS[5]
    original = pmd3.tunneld_rsd

    async def tunneld(udid):
        if udid == stuck:
            await never().wait()
        return await original(udid)
    monkeypatch.setattr(pmd3, "tunneld_rsd", tunneld)

    async def go():
        be = fleet.backend
        opening = asyncio.ensure_future(be.frames(stuck))
        await asyncio.sleep(0.01)
        others = [u for u in UDIDS if u != stuck]
        await asyncio.wait_for(asyncio.gather(*(be.frames(u) for u in others)), 1.0)
        assert not opening.done()
        opening.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await opening
    asyncio.run(go())


def test_a_phone_that_drops_out_resets_alone_and_its_input_is_not_retried(fleet):
    async def go():
        be = fleet.backend
        await open_all(be)
        gone = UDIDS[2]
        peers = {u: session(be, u) for u in UDIDS if u != gone}
        dead = session(be, gone)
        dead.hid.fail = True
        fleet.on_usb.remove(gone)
        results = await asyncio.gather(*(be.tap(u, 1, 1, 0.001) for u in UDIDS), return_exceptions=True)
        assert isinstance(results[2], ConnectionTerminatedError)
        assert dead.hid.reports.count("failed") == 2  # the one contact and its release: no second attempt at the tap
        assert dead.closed and fleet.rsds[gone].closed and gone not in be._routes
        for u, s in peers.items():
            assert session(be, u) is s and not s.closed and not fleet.rsds[u].closed and be._routes[u] == "tunneld"
        assert [r["udid"] for r in be.recoveries] == [gone]
        await be.close()
    asyncio.run(go())


def test_every_input_is_run_without_a_retry(monkeypatch):
    seen = []

    async def spy(self, udid, fn, retry, narrow=None):
        seen.append(retry)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_run", spy)
    be = pmd3.Pmd3Backend()
    path = [(1, 1, 0.0)]

    async def go():
        await be.tap("u", 1, 1, 0.01)
        await be.tap_indigo("u", 1, 1, 0.01)
        await be.tap_vdigitizer("u", 1, 1, 0.01, 7.0)
        await be.swipe("u", path)
        await be.swipe_indigo("u", path)
        await be.type("u", [("a", 0.01, 0.01)])
        await be.paste("u", "x")
        await be.home("u")
        await be.launch("u", "com.example.app")
    asyncio.run(go())
    assert seen == [False] * 9


def test_a_recovery_names_its_phone_at_both_call_sites(monkeypatch):
    async def reset(self, p, keep_eyes=False):
        pass

    async def narrow(p):
        pass

    async def boom(p):
        raise ConnectionError("dead")
    monkeypatch.setattr(pmd3.Pmd3Backend, "_reset", reset)
    be = pmd3.Pmd3Backend()
    with pytest.raises(ConnectionError):
        asyncio.run(be._run(UDIDS[4], boom, retry=True, narrow=narrow))
    assert [(r["udid"], r["reset"]) for r in be.recoveries] == [(UDIDS[4], "dvt"), (UDIDS[4], "all")]


def test_one_phones_idle_timer_is_its_own(fleet):
    async def go():
        be = fleet.backend
        be.idle = 0.08
        await open_all(be)
        kept, other = session(be, UDIDS[0]), session(be, UDIDS[1])
        for _ in range(4):
            await asyncio.sleep(0.04)
            be.keep_warm(UDIDS[0])
        await asyncio.sleep(0.02)
        assert session(be, UDIDS[0]) is kept and not kept.closed and other.closed and session(be, UDIDS[1]) is None
        await be.close()
    asyncio.run(go())


@pytest.fixture
def usb(monkeypatch):
    state = SimpleNamespace(wedged=set(), asked=[], wireless=set())

    class Lockdown:
        short_info, display_name, paired = {"DeviceName": "iPhone", "ProductVersion": "27.2", "ProductType": "iPhone13,2"}, "iPhone 12", True

        async def __aenter__(self):
            await asyncio.sleep(0.05)
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_developer_mode_status(self):
            return True

    async def create(serial, **kw):
        state.asked.append(serial)
        if serial in state.wedged:
            await asyncio.Event().wait()
        return Lockdown()

    async def listed():
        return [SimpleNamespace(serial=u, is_usb=u not in state.wireless, connection_type="USB") for u in UDIDS]
    monkeypatch.setattr(pmd3, "create_using_usbmux", create)
    monkeypatch.setattr(pmd3.usbmux, "list_devices", listed)
    monkeypatch.setattr(pmd3, "DEVICE_INFO_S", 0.15)
    return state


def timed(coro):
    async def go():
        t0 = time.monotonic()
        return await coro, time.monotonic() - t0
    return asyncio.run(go())


def test_listing_seven_phones_is_concurrent_and_a_wedged_one_is_listed_with_the_reason(usb):
    usb.wedged = {UDIDS[1]}
    out, took = timed(pmd3.Pmd3Backend().devices())
    assert [d["udid"] for d in out] == UDIDS and took < 0.3  # sequential: 0.15 + 6 × 0.05
    assert "error" in out[1] and all(d["developer_mode"] for i, d in enumerate(out) if i != 1)


def test_one_phones_info_asks_only_that_phone(usb):
    usb.wedged = set(UDIDS[1:])  # every other phone is wedged: this one's recheck must not notice
    info, took = timed(pmd3.Pmd3Backend().device_info(UDIDS[0]))
    assert usb.asked == UDIDS[:1] and took < 0.14 and info["udid"] == UDIDS[0] and info["developer_mode"] is True


def test_one_wedged_phones_info_is_its_own_error_within_the_bound(usb):
    usb.wedged = {UDIDS[2]}
    info, took = timed(pmd3.Pmd3Backend().device_info(UDIDS[2]))
    assert info["udid"] == UDIDS[2] and "error" in info and "name" not in info and took < 0.3


@pytest.mark.parametrize("absent", ["00008101-FFFFFFFFFFFFFFFF", UDIDS[3]])
def test_a_phone_not_on_usb_is_its_own_error_entry_never_another_phones(usb, absent):
    usb.wireless = {UDIDS[3]}  # still listed by usbmux, over Wi-Fi only
    info = asyncio.run(pmd3.Pmd3Backend().device_info(absent))
    assert info == {"udid": absent, "error": "LookupError: not on USB"} and usb.asked == []


def test_close_resets_the_phones_at_once_and_one_failure_stops_nobody(monkeypatch):
    done = []

    async def reset(self, p, keep_eyes=False):
        await asyncio.sleep(0.1)
        if p.udid == UDIDS[0]:
            raise RuntimeError("wedged")
        done.append(p.udid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_reset", reset)

    async def go():
        be = pmd3.Pmd3Backend()
        for u in UDIDS:
            be._phone(u)
        t0 = time.monotonic()
        await be.close()
        return time.monotonic() - t0
    assert asyncio.run(go()) < 0.3 and sorted(done) == UDIDS[1:]


# --- AVFoundation is one phone's, positively bound, or nobody's ----------------------------------------------------

class ScreenDevice:
    def __init__(self, unique, muxed=True):
        self.unique, self.muxed = unique, muxed

    def uniqueID(self):
        return self.unique

    def hasMediaType_(self, kind):
        return self.muxed and kind == "muxed"

    def localizedName(self):
        return "iPhone"  # every one of them


def test_a_capture_binds_only_to_the_device_whose_unique_id_is_the_udid():
    devices = [ScreenDevice(u.replace("-", "").lower()) for u in UDIDS]
    for i, u in enumerate(UDIDS):
        assert avf.bound_device(devices, u, "muxed") is devices[i]
    assert avf.bound_device(devices, "00008101-FFFFFFFFFFFFFFFF", "muxed") is None  # never the first "iPhone"
    assert avf.bound_device([ScreenDevice(UDIDS[0], muxed=False)], UDIDS[0], "muxed") is None
    assert avf.bound_device([ScreenDevice(UDIDS[0]), ScreenDevice(UDIDS[0])], UDIDS[0], "muxed") is None  # ambiguous


class Capture:
    def __init__(self, push, udid):
        self.push, self.udid, self.stopped = push, udid, False

    def stop(self):
        self.stopped = True


@pytest.fixture
def eyes(fleet, monkeypatch):
    fleet.opens = []

    async def opener(push, udid, wait_s=10.0):
        fleet.opens.append(udid)
        if fleet.__dict__.get("refuse"):
            raise RuntimeError("no screen device bound")
        return Capture(push, udid)
    monkeypatch.setattr(avf, "open_capture", opener)
    fleet.backend = pmd3.Pmd3Backend(idle=60, eyes="avf", tunnel="tunneld")
    fleet.on_usb[:] = UDIDS[:1]
    return fleet


def test_several_phones_on_usb_never_open_a_capture(eyes):
    eyes.on_usb[:] = UDIDS

    async def go():
        be = eyes.backend
        frames = await open_all(be)
        assert eyes.opens == [] and be._avf is None
        assert all(frames[i] is session(be, u).frames for i, u in enumerate(UDIDS))
        await be.close()
    asyncio.run(go())


def test_the_only_phone_gets_a_capture_bound_to_it_and_no_other_phone_ever_sees_it(eyes):
    async def go():
        be = eyes.backend
        frames = await be.frames(UDIDS[0])
        assert eyes.opens == [UDIDS[0]] and be._avf.udid == UDIDS[0]
        paint(frames, 0)
        assert be._live(be._phone(UDIDS[0])).width == DISPLAYS[0][0]
        assert be._live(be._phone(UDIDS[1])) is None  # the old code handed every phone this frame
        await be.close()
    asyncio.run(go())


def test_a_second_phone_on_usb_ends_the_capture_and_both_use_their_streams(eyes):
    async def go():
        be = eyes.backend
        await be.frames(UDIDS[0])
        cap = be._avf.device
        eyes.on_usb[:] = UDIDS[:2]
        b = await be.frames(UDIDS[1])
        assert cap.stopped and be._avf is None and b is session(be, UDIDS[1]).frames
        a = await be.frames(UDIDS[0])
        assert a is session(be, UDIDS[0]).frames and a is not b and eyes.opens == [UDIDS[0]]
        await be.close()
    asyncio.run(go())


def test_a_phone_missing_from_usb_for_a_moment_keeps_its_capture(eyes):
    """Starting a capture re-enumerates the phone's USB: an empty listing is not another phone."""
    async def go():
        be = eyes.backend
        frames = await be.frames(UDIDS[0])
        cap = be._avf.device
        eyes.on_usb.clear()
        assert await be.frames(UDIDS[0]) is frames and not cap.stopped
        await be.close()
    asyncio.run(go())


def test_stopping_or_warming_another_phone_leaves_the_capture_alone(eyes):
    async def go():
        be = eyes.backend
        await be.frames(UDIDS[0])
        cap, timer = be._avf.device, be._avf_timer
        be.stop_eyes("park", UDIDS[1])
        be.keep_warm(UDIDS[1])
        assert not cap.stopped and be._avf_timer is timer
        be.keep_warm(UDIDS[0])
        assert be._avf_timer is not timer
        be.stop_eyes("park", UDIDS[0])
        assert cap.stopped and be._avf is None
        await be.frames(UDIDS[0])
        cap = be._avf.device
        be.stop_eyes("shutdown")  # no phone named: whatever runs
        assert cap.stopped
        await be.close()
    asyncio.run(go())


def test_a_capture_that_cannot_bind_falls_back_and_is_not_rediscovered_on_every_look(eyes):
    eyes.refuse = True

    async def go():
        be = eyes.backend
        frames = await be.frames(UDIDS[0])
        assert frames is session(be, UDIDS[0]).frames and eyes.opens == [UDIDS[0]]
        await be.frames(UDIDS[0])
        assert eyes.opens == [UDIDS[0]]
        be._avf_retry_at = 0.0  # the window passed
        await be.frames(UDIDS[0])
        assert eyes.opens == UDIDS[:1] * 2
        await be.close()
    asyncio.run(go())


# --- routes: tunneld for several phones, a userspace tunnel only alone, never mixed ---------------------------------

@pytest.fixture
def routes(monkeypatch):
    state = SimpleNamespace(userspace=[], listed=set(), tunnels={})

    class Userspace:
        def __init__(self, serial, autopair):
            state.userspace.append(serial)
            self.serial = serial

        async def __aenter__(self):
            await asyncio.sleep(0)
            return Rsd(self.serial)

        async def __aexit__(self, *exc):
            return False

    async def tunneld(udid):
        await asyncio.sleep(0)
        if udid not in state.listed:
            return None
        state.tunnels[udid] = Rsd(udid)
        return state.tunnels[udid]

    async def addr(self, udid):
        return None
    monkeypatch.setattr(pmd3, "UserspaceRsdTunnel", Userspace)
    monkeypatch.setattr(pmd3, "tunneld_rsd", tunneld)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_tunnel_addr", addr)
    return state


def test_seven_phones_on_tunneld_each_keep_their_route(routes):
    routes.listed = set(UDIDS)
    be = pmd3.Pmd3Backend(tunnel="tunneld")

    async def go():
        rsds = await asyncio.gather(*(be._open_rsd(be._phone(u)) for u in UDIDS))
        assert [r.udid for r in rsds] == UDIDS and be._routes == dict.fromkeys(UDIDS, "tunneld") and routes.userspace == []
    asyncio.run(go())


@pytest.mark.parametrize("tunnel", ["userspace", "auto"])
def test_phones_opening_at_once_get_at_most_one_userspace_tunnel_and_the_rest_refuse(routes, tunnel):
    be = pmd3.Pmd3Backend(tunnel=tunnel)

    async def go():
        return await asyncio.gather(*(be._open_rsd(be._phone(u)) for u in UDIDS), return_exceptions=True)
    results = asyncio.run(go())
    won = [r for r in results if isinstance(r, Rsd)]
    assert len(won) == 1 and len(routes.userspace) == 1 and routes.userspace == [won[0].udid]
    assert all("several phones need tunneld" in str(r) for r in results if not isinstance(r, Rsd))
    assert be._routes == {won[0].udid: "userspace"}


def test_a_userspace_phone_refuses_a_tunneld_phone_and_the_reverse(routes):
    routes.listed = {UDIDS[1]}

    async def go():
        be = pmd3.Pmd3Backend(tunnel="auto")
        await be._open_rsd(be._phone(UDIDS[0]))  # not listed: userspace
        with pytest.raises(RuntimeError, match="several phones need tunneld"):
            await be._open_rsd(be._phone(UDIDS[1]))  # listed, but its RTP receiver would land on the userspace stack
        assert be._routes == {UDIDS[0]: "userspace"}
        routes.userspace.clear()
        other = pmd3.Pmd3Backend(tunnel="auto")
        await other._open_rsd(other._phone(UDIDS[1]))  # listed: tunneld
        with pytest.raises(RuntimeError, match="several phones need tunneld"):
            await other._open_rsd(other._phone(UDIDS[0]))  # not listed: no userspace beside it
        assert routes.userspace == [] and other._routes == {UDIDS[1]: "tunneld"}
    asyncio.run(go())


def test_a_route_is_given_back_on_reset_and_on_a_failed_open(routes, monkeypatch):
    async def go():
        be = pmd3.Pmd3Backend(tunnel="userspace")
        p = be._phone(UDIDS[0])
        p.rsd = await be._open_rsd(p)
        monkeypatch.setattr(be, "attached", lambda: asyncio.sleep(0, list(UDIDS)))
        await be._reset(p)
        assert be._routes == {}
        await be._open_rsd(be._phone(UDIDS[1]))  # the tunnel is free for the next phone
        assert be._routes == {UDIDS[1]: "userspace"}

        class Broken:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                raise OSError("no such device")

            async def __aexit__(self, *exc):
                return False
        monkeypatch.setattr(pmd3, "UserspaceRsdTunnel", Broken)
        fresh = pmd3.Pmd3Backend(tunnel="userspace")
        with pytest.raises(OSError):
            await fresh._open_rsd(fresh._phone(UDIDS[0]))
        assert fresh._routes == {}
    asyncio.run(go())


def test_tunneld_is_refused_while_a_userspace_tunnel_is_still_up_in_the_process(routes, monkeypatch):
    routes.listed = set(UDIDS)
    monkeypatch.setattr(pmd3.userspace_tunnel, "USERSPACE_ACTIVE", True)  # e.g. a close that timed out
    be = pmd3.Pmd3Backend(tunnel="tunneld")
    with pytest.raises(RuntimeError, match="several phones need tunneld"):
        asyncio.run(be._open_rsd(be._phone(UDIDS[0])))
    assert routes.tunnels == {} and be._routes == {}


@pytest.mark.parametrize("tunnel", ["tunneld", "userspace"])
def test_a_tunnel_that_reaches_a_different_phone_is_refused_and_closed(routes, monkeypatch, tunnel):
    wrong = Rsd(UDIDS[6])

    async def tunneld(udid):
        return wrong

    class Userspace:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return wrong

        async def __aexit__(self, *exc):
            await wrong.close()
            return False
    monkeypatch.setattr(pmd3, "tunneld_rsd", tunneld)
    monkeypatch.setattr(pmd3, "UserspaceRsdTunnel", Userspace)
    be = pmd3.Pmd3Backend(tunnel=tunnel)
    with pytest.raises(RuntimeError, match="zero input"):
        asyncio.run(be._open_rsd(be._phone(UDIDS[0])))
    assert wrong.closed and be._routes == {}


def test_the_same_udid_written_another_way_is_still_that_phone(routes, monkeypatch):
    async def tunneld(udid):
        return Rsd(udid.replace("-", "").lower())
    monkeypatch.setattr(pmd3, "tunneld_rsd", tunneld)
    be = pmd3.Pmd3Backend(tunnel="tunneld")
    assert asyncio.run(be._open_rsd(be._phone(UDIDS[0]))).udid == UDIDS[0].replace("-", "").lower()


# --- the real tunneld lookup, its network faked ---------------------------------------------------------------------

@pytest.fixture
def lookup(monkeypatch):
    state = SimpleNamespace(tunnels={}, delay=0.0, connects=[], dead=set())

    def listing(*a):
        time.sleep(state.delay)  # pymobiledevice3's blocking requests.get
        return state.tunnels

    class Service:
        def __init__(self, address, name=None, auxiliary_metadata=None):
            self.address, self.udid, self.closed = address, "from-handshake", False

        async def connect(self):
            state.connects.append(self.address[0])
            if self.address[0] in state.dead:
                raise ConnectionError("gone")

        async def close(self):
            self.closed = True
    monkeypatch.setattr("pymobiledevice3.tunneld.api._list_tunnels", listing)
    monkeypatch.setattr(pmd3, "RemoteServiceDiscoveryService", Service)
    return state


def entry(address):
    return {"tunnel-address": address, "tunnel-port": 1234, "interface": "utun7"}


def test_a_slow_tunneld_listing_does_not_stall_the_event_loop(lookup):
    lookup.tunnels, lookup.delay = {UDIDS[0]: [entry("fd0::1")]}, 0.3

    async def go():
        gaps, last = [], time.monotonic()

        async def beat():
            nonlocal last
            while True:
                await asyncio.sleep(0.01)
                gaps.append(time.monotonic() - last)
                last = time.monotonic()
        heart = asyncio.ensure_future(beat())
        rsd = await pmd3.tunneld_rsd(UDIDS[0])
        heart.cancel()
        return rsd, max(gaps)
    rsd, worst = asyncio.run(go())
    assert rsd.address == ("fd0::1", 1234) and worst < 0.15


def test_tunneld_lookup_returns_none_only_for_an_unlisted_phone(lookup):
    assert asyncio.run(pmd3.tunneld_rsd(UDIDS[0])) is None and lookup.connects == []


def test_a_listed_phone_tries_its_tunnels_then_raises_and_auto_does_not_fall_back(lookup, monkeypatch):
    lookup.tunnels = {UDIDS[0]: [entry("fd0::1"), entry("fd0::2")]}
    lookup.dead = {"fd0::1"}
    assert asyncio.run(pmd3.tunneld_rsd(UDIDS[0])).address[0] == "fd0::2"
    lookup.dead = {"fd0::1", "fd0::2"}
    with pytest.raises(ConnectionError, match="none of its tunnels"):
        asyncio.run(pmd3.tunneld_rsd(UDIDS[0]))

    class Userspace:
        def __init__(self, **kw):
            raise AssertionError("a listed phone never falls back to the userspace tunnel")
    monkeypatch.setattr(pmd3, "UserspaceRsdTunnel", Userspace)
    be = pmd3.Pmd3Backend(tunnel="auto")
    with pytest.raises(ConnectionError):
        asyncio.run(be._open_rsd(be._phone(UDIDS[0])))
    assert be._routes == {}
