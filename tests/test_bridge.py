import asyncio
import io
import json
import os
import struct
import time
import zlib

import pytest
from PIL import Image

from phonebridge import humanize, ocr
from phonebridge.bridge import Bridge
from phonebridge.config import Config, Device
from phonebridge.lease import LeaseError, Leases

UNCONFIGURED = "00008140-AAAAAAAAAAAAAAAA"


def tiny_png(w=3, h=5, rgb=b"\0\0\0") -> bytes:
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\0" + rgb * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


W, H = 120, 260  # a phone screen at 1/10 scale
LIT, DARK = tiny_png(W, H, b"\x40\x60\x90"), tiny_png(W, H)
OTHER = tiny_png(W, H, b"\xd0\xd0\xd0")  # a different (still) screen
from conftest import HOME  # noqa: F401  (the fake OCR fixture's home screen)
LOCKED = [{"text": "9:41", "x": 60, "y": 55, "w": 50, "h": 25, "conf": 1.0}]  # the big centred clock


class FakeBackend:
    def __init__(self, tap_s=0.0, screens=None, tap_shows=None):
        self.calls, self.tap_s, self.shots, self.shot_s = [], tap_s, 0, 0.0
        self.screens = screens  # list of PNGs to return in turn, then `current`
        self.current, self.animate = LIT, False  # animate: every screenshot differs (a video playing)
        self.tap_shows = tap_shows  # what the screen shows after our tap (the UI responding)

    async def devices(self):
        return [{"udid": "U12", "name": "iPhone", "ios": "27.0.1", "product": "iPhone13,2"},
                {"udid": UNCONFIGURED, "name": "iPhone", "ios": "27.0.1", "product": "iPhone17,3"}]

    async def screen(self, udid):
        await asyncio.sleep(self.shot_s)
        self.shots += 1
        if self.screens:
            return self.screens.pop(0)
        if self.animate:
            return (LIT, OTHER)[self.shots % 2]
        return self.current

    async def tap(self, udid, x, y, press):
        await asyncio.sleep(self.tap_s)
        self.calls.append(("tap", udid, round(x), round(y)))  # the humanizer's landing point
        if self.tap_shows:
            self.current = self.tap_shows

    async def swipe(self, udid, path):
        self.calls.append(("swipe", udid, path))

    async def type(self, udid, keys):
        self.calls.append(("type", udid, humanize.typed(keys)))

    async def paste(self, udid, text):
        self.calls.append(("paste", udid, text))

    async def home(self, udid):
        self.calls.append(("home", udid))

    async def launch(self, udid, bundle_id, url=None):
        if bundle_id == "com.nope":
            raise RuntimeError("no such app")
        self.calls.append(("launch", udid, bundle_id, url))

    async def lock_state(self, udid):
        return {"locked": False}

    async def close(self):
        pass


def fake_authhub(tmp_path, acquire="echo tok123"):
    """A stand-in `authhub` CLI: logs its args; `lease check` accepts only the token 'callertok'."""
    log = tmp_path / "authhub.log"
    script = tmp_path / "authhub"
    script.write_text(f"""#!/bin/sh
echo "$@" >> {log}
case "$2" in
  acquire) {acquire} ;;
  check) [ "$3" = callertok ] || {{ echo "no live lease on $4 for that token; acquire it first" >&2; exit 1; }}; echo 600 ;;
esac
""")
    script.chmod(0o755)
    return [str(script)], log


def via_fake_ssh(tmp_path, cmd):
    """Like `ssh host authhub …`: args are re-joined with spaces and parsed by a shell on the far side."""
    ssh = tmp_path / "ssh"
    ssh.write_text(f'#!/bin/sh\nexec sh -c "{cmd[0]} $*"\n')
    ssh.chmod(0o755)
    return [str(ssh)]


def swipe_of(call, tol=40):
    """(x1, y1, x2, y2) a recorded swipe path roughly covers (the humanizer jitters the ends), else the path's ends."""
    assert call[0] == "swipe"
    path = call[2]
    ends = (*path[0][:2], *path[-1][:2])

    class Near(tuple):
        def __eq__(self, other):
            return all(abs(a - b) <= tol for a, b in zip(self, other))
    return Near(ends)


def authhub_calls(log):
    return [line.split()[1] for line in log.read_text().splitlines()] if log.exists() else []


EXACT = humanize.HumanConfig(seed=1, tap_sigma=0, tap_offset=(0, 0), think={}, typo_rate=0, pause_rate=0)  # land where asked, no pauses


def config(cmd, **kw):
    return Config(devices={"iphone12": Device("iphone12", "U12", lease="device:iphone12", supervision="auto"),
                           "iphone16": Device("iphone16", "U16", lease="device:iphone16", supervision="hold-outward")},
                  lease_cmd=cmd, humanize=kw.pop("humanize", EXACT), **kw)


@pytest.fixture
def authhub(tmp_path):
    return fake_authhub(tmp_path)


@pytest.fixture
def bridge(tmp_path, authhub):
    return Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / "calls.jsonl")


def test_png_size():
    assert ocr.png_size(tiny_png(1170, 2532)) == (1170, 2532)
    with pytest.raises(ValueError):
        ocr.png_size(b"GIF89a" + b"\0" * 30)


def test_find_prefers_exact_then_shortest():
    items = [{"text": "Settings and Privacy", "conf": 1.0}, {"text": "settings", "conf": 0.5},
             {"text": "My Settings", "conf": 0.9}]
    assert ocr.find(items, "Settings")["text"] == "settings"
    assert ocr.find(items[::2], "settings")["text"] == "My Settings"
    assert ocr.find(items, "Mail") is None


def test_alias_resolution_and_supervision(bridge):
    devs = asyncio.run(bridge.devices())
    assert devs[0]["id"] == "iphone12" and devs[0]["udid"] == "U12" and devs[0]["supervision"] == "auto"
    assert devs[1]["id"] == UNCONFIGURED and devs[1]["supervision"] == "read-only"
    assert bridge.config.resolve("U16").alias == "iphone16"
    with pytest.raises(ValueError, match="unknown device 'iphone13'; aliases: iphone12, iphone16"):
        bridge.config.resolve("iphone13")


def test_unconfigured_phones_are_read_only(bridge):
    asyncio.run(bridge.screen(UNCONFIGURED))  # eyes work
    for call in (bridge.tap(UNCONFIGURED, 1, 2), bridge.home(UNCONFIGURED), bridge.launch(UNCONFIGURED, "com.x")):
        with pytest.raises(PermissionError, match="not in the phonebridge config"):
            asyncio.run(call)
    assert bridge.backend.calls == [] and bridge.backend.shots == 1


def test_without_a_lease_cmd_input_is_refused_unless_enforce_is_off(tmp_path):
    b = Bridge(FakeBackend(), config([]), log_path=tmp_path / "calls.jsonl")
    with pytest.raises(LeaseError, match=r"no \[lease\] cmd"):
        asyncio.run(b.tap("iphone12", 1, 2))
    b = Bridge(FakeBackend(), config([], lease_enforce=False), log_path=tmp_path / "calls2.jsonl")
    asyncio.run(b.tap("iphone12", 1, 2))
    assert b.backend.calls == [("tap", "U12", 1, 2)]


def test_config_file_is_reloaded_and_a_new_authhub_drops_held_leases(tmp_path):
    cmd, log = fake_authhub(tmp_path)
    path = tmp_path / "phonebridge.toml"
    b = Bridge(FakeBackend(), log_path=tmp_path / "calls.jsonl", config_path=path)
    assert asyncio.run(b.devices())[0]["id"] == "U12"  # no file yet: UDID as id, read-only
    path.write_text(f'[devices.iphone12]\nudid = "U12"\n[lease]\ncmd = {json.dumps(cmd)}\n')
    assert asyncio.run(b.devices())[0]["id"] == "iphone12"
    assert b.config.devices["iphone12"].lease == "device:iphone12" and b.config.devices["iphone12"].supervision == "auto"
    asyncio.run(b.tap("iphone12", 1, 2))
    assert b.leases.holds("device:iphone12")
    path.write_text('[devices.iphone12]\nudid = "U12"\n[lease]\nenforce = false\n')
    os.utime(path, (1, 1))  # force a new mtime
    assert b.config.lease_cmd == [] and not b.leases.holds("device:iphone12")
    assert authhub_calls(log) == ["acquire", "release"]


def test_calls_route_to_udid_are_timed_and_the_log_is_private(bridge):
    asyncio.run(bridge.tap("iphone12", 10, 20))
    asyncio.run(bridge.screen("iphone16"))
    assert bridge.backend.calls == [("tap", "U12", 10, 20)]
    recs = [json.loads(l) for l in bridge.log_path.read_text().splitlines()]
    assert [r["op"] for r in recs] == ["tap", "screen"]
    assert all(r["ok"] and r["ms"] >= 0 for r in recs)
    assert os.stat(bridge.log_path).st_mode & 0o777 == 0o600


def test_failures_are_logged_and_raised(bridge):
    with pytest.raises(RuntimeError):
        asyncio.run(bridge.launch("iphone12", "com.nope"))
    rec = json.loads(bridge.log_path.read_text().splitlines()[-1])
    assert rec["ok"] is False and "no such app" in rec["error"]


def test_open_url_policy_and_scheme_apps(tmp_path, authhub):
    urls = ["https://*", "shortcuts://run-shortcut?name=ReadyFeed", "instagram://*"]
    b = Bridge(FakeBackend(), config(authhub[0], urls=urls), log_path=tmp_path / "calls.jsonl")
    asyncio.run(b.open_url("iphone12", "shortcuts://run-shortcut?name=ReadyFeed"))
    asyncio.run(b.open_url("iphone12", "https://example.com/?token=s3cret"))
    asyncio.run(b.open_url("iphone12", "instagram://camera", bundle_id="com.burbn.instagram"))
    assert [c[2] for c in b.backend.calls] == ["com.apple.shortcuts", "com.apple.mobilesafari", "com.burbn.instagram"]
    assert b.backend.calls[0][3] == "shortcuts://run-shortcut?name=ReadyFeed"
    assert "s3cret" not in b.log_path.read_text()  # only scheme://host is logged
    for url in ("shortcuts://run-shortcut?name=Other", "tel:+15555550100", "prefs:root=General",
                "shortcuts://x-callback-url/run-shortcut?name=ReadyFeed&x-success=tel:1"):
        with pytest.raises(PermissionError, match="not allowed by \\[policy\\] urls"):
            asyncio.run(b.open_url("iphone12", url, bundle_id="com.apple.mobilephone"))
    for url in ("https://example.com/\nx", "https://a b", "https://" + "a" * 2050):
        with pytest.raises(ValueError):
            asyncio.run(b.open_url("iphone12", url))
    assert len(b.backend.calls) == 3


def test_default_url_policy_is_https_only(bridge):
    with pytest.raises(PermissionError):
        asyncio.run(bridge.open_url("iphone12", "shortcuts://run-shortcut?name=ReadyFeed"))
    asyncio.run(bridge.open_url("iphone12", "https://example.com"))


def test_input_is_bounded_and_typing_never_presses_return_by_accident(bridge):
    bad = [bridge.tap("iphone12", float("nan"), 1), bridge.swipe("iphone12", 0, 0, 1, float("inf")),
           bridge.swipe("iphone12", 0, 0, 1, 1, duration=0), bridge.swipe("iphone12", 0, 0, 1, 1, duration=86400),
           bridge.type("iphone12", "x" * 301), bridge.type("iphone12", "hello\nsend"),
           bridge.type("iphone12", "tab\there"), bridge.paste("iphone12", "x" * 64_001)]
    for call in bad:
        with pytest.raises(ValueError):
            asyncio.run(call)
    asyncio.run(bridge.type("iphone12", "hello", submit=True))
    asyncio.run(bridge.paste("iphone12", "multi\nline ok"))
    assert bridge.backend.calls == [("type", "U12", "hello\n"), ("paste", "U12", "multi\nline ok")]


def test_the_device_lock_is_per_caller_until_idle(tmp_path, authhub):
    cmd, log = authhub
    b = Bridge(FakeBackend(), config(cmd), log_path=tmp_path / "calls.jsonl", lease_idle=0.1)

    async def go():
        await b.tap("iphone12", 1, 2, caller="mcp:A")
        await b.home("iphone12", caller="mcp:A")  # same session: fine
        with pytest.raises(LeaseError, match="busy: controller"):
            await b.tap("iphone12", 3, 4, caller="mcp:B")
        await asyncio.sleep(0.3)  # A goes idle → released
        await b.tap("iphone12", 5, 6, caller="mcp:B")
        await b.close()

    asyncio.run(go())
    assert authhub_calls(log) == ["acquire", "release", "acquire", "release"]
    holders = [l for l in log.read_text().splitlines() if "acquire" in l]
    assert holders[0].endswith("mcp:A") and holders[1].endswith("mcp:B")
    assert [c[2:] for c in b.backend.calls if c[0] == "tap"] == [(1, 2), (5, 6)]


def test_idle_release_waits_for_the_last_input_to_end(tmp_path, authhub):
    cmd, log = authhub
    b = Bridge(FakeBackend(tap_s=0.4), config(cmd), log_path=tmp_path / "calls.jsonl", lease_idle=0.1)

    async def go():
        await b.tap("iphone12", 1, 2)  # a gesture longer than lease_idle
        assert authhub_calls(log) == ["acquire"]  # not released mid-gesture
        await asyncio.sleep(0.3)
        assert authhub_calls(log) == ["acquire", "release"]

    asyncio.run(go())


def test_a_caller_lease_token_is_verified_and_never_acquired_or_released(tmp_path, authhub):
    cmd, log = authhub
    b = Bridge(FakeBackend(), config(cmd), log_path=tmp_path / "calls.jsonl", lease_idle=0.05)

    async def go():
        await b.tap("iphone12", 1, 2, lease="callertok")
        await b.type("iphone12", "hi", lease="callertok")  # cached: one check for both
        await asyncio.sleep(0.2)
        with pytest.raises(LeaseError, match="not a live lease on device:iphone12") as e:
            await b.home("iphone12", lease="stolen")
        assert "stolen" not in str(e.value)

    asyncio.run(go())
    assert authhub_calls(log) == ["check", "check"]  # no acquire, renew or release
    assert "callertok" not in b.log_path.read_text()
    assert [c[0] for c in b.backend.calls] == ["tap", "type"]


def test_lease_refusal_names_the_holder_and_never_echoes_stdout(tmp_path):
    cmd, _ = fake_authhub(tmp_path, acquire="echo 'held by lane-bthid until 09:12' >&2; exit 75")
    with pytest.raises(LeaseError, match="held by lane-bthid"):
        Leases(cmd).ensure("device:iphone12")
    cmd, _ = fake_authhub(tmp_path, acquire="echo tok-on-stdout; exit 3")
    with pytest.raises(LeaseError, match="lease command exited 3") as e:
        Leases(cmd).ensure("device:iphone12")
    assert "tok-on-stdout" not in str(e.value)
    Leases(cmd).ensure(None)  # no identity, no lease


def test_values_sent_over_ssh_stay_single_shell_words(tmp_path, authhub):
    cmd, log = authhub
    b = Bridge(FakeBackend(), config(via_fake_ssh(tmp_path, cmd)), log_path=tmp_path / "calls.jsonl")
    pwned = tmp_path / "pwned"

    async def go():
        await b.tap("iphone12", 1, 2, caller="mcp:9f2c 1b;x")  # odd session ids can't split the holder
        for token in (f"x; touch {pwned}", "$(touch /tmp/x)", "a b", "callertok\n"):
            with pytest.raises(LeaseError, match="malformed") as e:
                await b.home("iphone12", lease=token, caller="mcp:9f2c 1b;x")
            assert token not in str(e.value)
        await b.home("iphone12", lease="callertok", caller="mcp:9f2c 1b;x")

    asyncio.run(go())
    assert not pwned.exists()
    acquire = log.read_text().splitlines()[0].split()
    assert acquire[:2] == ["lease", "acquire"] and acquire[-1].endswith("/mcp:9f2c_1b_x") and len(acquire) == 9


def test_lease_acquires_once_renews_by_token_and_releases(tmp_path, authhub):
    cmd, log = authhub
    leases = Leases(cmd, ttl=120, holder="t")
    leases.ensure("device:iphone12")
    leases.ensure("device:iphone12")  # fresh: no call
    leases._held["device:iphone12"] = ("tok123", time.monotonic() - 70)  # pretend half the TTL has passed
    leases.ensure("device:iphone12")
    leases.release_all()
    assert log.read_text().splitlines() == [
        "lease acquire device:iphone12 --ttl 120 --wait 0 --holder t",
        "lease renew tok123 --ttl 120",
        "lease release tok123",
    ]


def test_mcp_tools_take_the_session_as_caller_and_an_optional_lease(bridge):
    from mcp import Client

    from phonebridge.mcp_server import build

    async def go():
        async with Client(build(bridge)) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            r = await c.call_tool("tap", {"device": "iphone12", "x": 1, "y": 2, "controller": "test-client"})
            t = await c.call_tool("type", {"device": "iphone12", "text": "ok", "submit": True, "lease": "callertok", "controller": "test-client"})
            img = await c.call_tool("screen", {"device": "iphone12"})
            bad = await c.call_tool("home", {"device": "iphone13", "controller": "test-client"})
            return tools, r, t, img, bad

    tools, r, t, img, bad = asyncio.run(go())
    assert {"devices", "screen", "ocr", "find_text", "tap", "swipe", "type", "paste", "launch",
            "open_url", "lock_state", "home"} <= set(tools)
    for name in ("tap", "swipe", "type", "paste", "home", "unlock", "launch", "open_url"):
        props = tools[name].input_schema["properties"]
        assert "lease" in props and "ctx" not in props, name
    assert bad.is_error and "unknown device 'iphone13'" in bad.content[0].text
    assert not r.is_error and not t.is_error, (r.content, t.content)
    assert bridge.backend.calls[:2] == [("tap", "U12", 1, 2), ("type", "U12", "ok\n")]
    assert img.content[0].type == "image" and img.content[0].mime_type == "image/jpeg"


def test_bearer_auth_guards_http():
    from phonebridge.mcp_server import BearerAuth

    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    async def call(headers):
        sent = []

        async def send(msg):
            sent.append(msg)
        await BearerAuth(app, "t0ken")({"type": "http", "headers": headers}, None, send)
        return sent

    assert asyncio.run(call([]))[0]["status"] == 401
    assert asyncio.run(call([(b"authorization", b"Bearer wrong")]))[0]["status"] == 401
    assert asyncio.run(call([(b"authorization", b"Bearer t0ken")])) == [] and seen == ["http"]
    asyncio.run(BearerAuth(app, "t0ken")({"type": "lifespan"}, None, None))  # lifespan passes through
    assert seen == ["http", "lifespan"]


def test_while_a_caller_holds_the_device_lease_tokenless_input_is_refused(tmp_path):
    """feedagent holds device:iphone12 through its ReadyFeed wait: nobody without the token slips in."""
    cmd, log = fake_authhub(tmp_path, acquire="echo 'busy: device:iphone12 is held by feedagent (expires in 600s)' >&2; exit 75")
    b = Bridge(FakeBackend(), config(cmd), log_path=tmp_path / "calls.jsonl")
    with pytest.raises(LeaseError, match="held by feedagent"):
        asyncio.run(b.tap("iphone12", 1, 2))
    asyncio.run(b.tap("iphone12", 1, 2, lease="callertok"))
    assert b.backend.calls == [("tap", "U12", 1, 2)] and authhub_calls(log) == ["acquire", "check"]



def test_lock_screen_detection():
    assert ocr.lock_screen(DARK, []) == "the display is off"
    assert ocr.lock_screen(LIT, LOCKED) == "the lock-screen clock is showing"
    assert "Swipe up to open" in ocr.lock_screen(LIT, [{"text": "Swipe up to open", "x": 60, "y": 250, "w": 40, "h": 4}])
    assert ocr.lock_screen(LIT, HOME) is None  # the small status-bar time is not the lock clock
    assert "Sun Oct 18" in ocr.lock_screen(LIT, HOME + [{"text": "Sun Oct 18", "x": 60, "y": 30, "w": 30, "h": 4}])
    settings_rows = HOME + [{"text": "Notification Center", "x": 30, "y": 130, "w": 30, "h": 4},  # Face ID & Passcode page
                            {"text": "Sunday, October 18", "x": 25, "y": 40, "w": 30, "h": 4}]  # a left-aligned header
    assert ocr.lock_screen(LIT, settings_rows) is None
    status_bar_only = [{"text": "4:35", "x": 15, "y": 7, "w": 8, "h": 4}]  # iOS 27 lock screen with just a lock glyph
    assert ocr.lock_screen(LIT, status_bar_only) == "nothing but the status bar is readable"
    big_but_off_centre = [{"text": "4:58", "x": 100, "y": 60, "w": 30, "h": 20}]  # e.g. the Clock app's list
    assert ocr.lock_screen(LIT, big_but_off_centre) is None


def test_taps_and_swipes_are_refused_on_the_lock_screen_except_the_unlock_swipe(bridge, fake_ocr):
    from phonebridge.bridge import PhoneLocked

    fake_ocr.extend([LOCKED, LOCKED, LOCKED])
    with pytest.raises(PhoneLocked, match="phone is locked .*lock-screen clock.*unlock"):
        asyncio.run(bridge.tap("iphone12", 60, 130))
    with pytest.raises(PhoneLocked):  # a sideways swipe here opened the Camera (bthid run, 09:24Z)
        asyncio.run(bridge.swipe("iphone12", 100, 150, 10, 150))
    asyncio.run(bridge.swipe("iphone12", 60, 258, 60, 120))  # swipe up from the bottom edge = unlock
    assert swipe_of(bridge.backend.calls[0]) == (60, 256, 60, 117)


def test_a_recently_touched_phone_is_not_rechecked(bridge):
    asyncio.run(bridge.tap("iphone12", 1, 2))
    asyncio.run(bridge.tap("iphone12", 3, 4))
    asyncio.run(bridge.swipe("iphone12", 1, 2, 3, 4))
    assert bridge.backend.shots == 1  # one guard check, then the phone counts as awake for 20 s


def test_unlock_wakes_swipes_up_and_confirms(tmp_path, authhub, fake_ocr):
    from phonebridge.bridge import PhoneLocked

    b = Bridge(FakeBackend(screens=[DARK, LIT]), config(authhub[0]), log_path=tmp_path / "calls.jsonl")
    fake_ocr.extend([[], HOME])  # dark display, then the home screen
    assert asyncio.run(b.unlock("iphone12")) == {"was_locked": True, "unlocked": True}
    assert b.backend.calls[0] == ("home", "U12") and swipe_of(b.backend.calls[1]) == (60, 256, 60, 117)

    asyncio.run(b.close())
    b = Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / "calls2.jsonl")
    fake_ocr.extend([LOCKED, LOCKED])  # still locked after the swipe: a passcode is set
    with pytest.raises(PhoneLocked, match="passcode"):
        asyncio.run(b.unlock("iphone12"))
    asyncio.run(b.close())
    assert asyncio.run(Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / "c3.jsonl").unlock("iphone12")) \
        == {"was_locked": False, "unlocked": True}



def quick(bridge):
    bridge.op_gap, bridge.op_settle, bridge.op_backoff = 0.05, 0.02, 0.3
    return bridge


def test_operator_activity_releases_the_lease_and_backs_off(tmp_path, authhub):
    from phonebridge.bridge import OperatorActive

    cmd, log = authhub
    b = quick(Bridge(FakeBackend(), config(cmd), log_path=tmp_path / "calls.jsonl"))

    async def go():
        await b.tap("iphone12", 1, 2)
        await asyncio.sleep(0.1)  # our tap settles: that screen is the baseline
        b.backend.current = OTHER  # the operator navigates somewhere else
        await asyncio.sleep(0.1)
        with pytest.raises(OperatorActive, match="operator appears active.*backing off"):
            await b.tap("iphone12", 3, 4)
        shots = b.backend.shots
        with pytest.raises(OperatorActive, match="backs off for"):
            await b.home("iphone12")  # during the backoff: refused without looking
        assert b.backend.shots == shots
        await asyncio.sleep(0.35)  # backoff over, operator gone (screen unchanged since)
        await b.tap("iphone12", 5, 6)
        await b.close()

    asyncio.run(go())
    assert [c[2:] for c in b.backend.calls if c[0] == "tap"] == [(1, 2), (5, 6)]
    assert authhub_calls(log) == ["acquire", "release", "acquire", "release"]  # dropped the lease on detection
    checks = [(r["op"], r.get("operator_check"), r.get("change")) for r in map(json.loads, b.log_path.open())]
    assert checks == [("tap", "no-baseline", None), ("tap", "operator", 1.0), ("home", "backing-off", None),
                      ("tap", "same", 0.0)]  # each verdict and score is in the call log
    evidence = [tmp_path / "phones" / f"U12.operator-{n}.png" for n in ("before", "after")]
    assert [p.read_bytes() for p in evidence] == [LIT, OTHER]  # what it saw, to tell the operator from a false alarm
    assert {p.stat().st_mode & 0o777 for p in evidence} == {0o600}


def test_a_slow_screenshot_does_not_make_the_baseline_look_stale(tmp_path, authhub):
    """Live miss 17:58Z: the check dated our last input after its own screenshot (0.5-2 s on the phone), so
    the baseline taken op_settle after that input looked too early ('stale-baseline'): the operator went unseen."""
    from phonebridge.bridge import OperatorActive

    cmd, _ = authhub
    b = quick(Bridge(FakeBackend(), config(cmd), log_path=tmp_path / "calls.jsonl"))

    async def go():
        await b.launch("iphone12", "com.apple.mobiletimer")
        await asyncio.sleep(0.1)  # our launch settles: that screen is the baseline
        b.backend.current, b.backend.shot_s = OTHER, 0.1  # the operator presses Home; the next screenshot is slow
        await asyncio.sleep(0.1)
        with pytest.raises(OperatorActive):
            await b.tap("iphone12", 1, 2)
        await b.close()

    asyncio.run(go())
    assert [json.loads(r).get("operator_check") for r in b.log_path.open()] == ["no-baseline", "operator"]


def test_two_dark_mode_screens_are_told_apart():
    def dark(rows):  # a dark-mode screen: black, a few thin lines of white text
        im = Image.new("L", (W, H))
        for y in rows:
            im.paste(255, (10, y, 80, y + 3))
        out = io.BytesIO()
        im.save(out, "PNG")
        return out.getvalue()

    a, b = dark(range(60, 240, 40)), dark(range(80, 240, 40))
    assert ocr.change(a, a) == 0 and ocr.change(a, b) > ocr.CHANGED and ocr.change(a, DARK) > ocr.CHANGED


def test_between_sessions_other_lanes_are_not_the_operator_but_the_operator_is(tmp_path, authhub):
    """Our lease hold ends; another lane drives the phone through its own phonebridge process (post-live's
    11:07Z false alarm), then our next session starts clean. Then the operator uses the phone between
    sessions: the next session's first input is refused (the lead's "before each input session")."""
    from phonebridge.bridge import OperatorActive

    phone = FakeBackend(tap_shows=OTHER)
    ours = quick(Bridge(phone, config(authhub[0]), log_path=tmp_path / "a.jsonl", lease_idle=0.15))
    other_lane = quick(Bridge(phone, config(authhub[0]), log_path=tmp_path / "b.jsonl"))

    async def go():
        phone.tap_shows = None
        await ours.tap("iphone12", 1, 2)
        await asyncio.sleep(0.25)  # settled on LIT; idle, so our hold ends and the lease is released
        assert not ours.leases.holds("device:iphone12")
        phone.tap_shows = OTHER
        await other_lane.tap("iphone12", 3, 4, lease="callertok")  # leaves the phone on OTHER
        await other_lane.release_reservation("iphone12", "local")
        await asyncio.sleep(0.1)
        await ours.tap("iphone12", 5, 6)  # a new hold: the screen changed, but by phonebridge
        await asyncio.sleep(0.25)  # settled on OTHER; the hold ends again
        phone.current = LIT  # the operator, between our sessions
        await asyncio.sleep(0.1)
        with pytest.raises(OperatorActive, match="operator appears active"):
            await ours.tap("iphone12", 7, 8, lease="callertok")  # a new session, with a token this time
        await ours.close()
        await other_lane.close()

    asyncio.run(go())
    assert len([c for c in phone.calls if c[0] == "tap"]) == 3  # the operator's session-start tap never landed
    checks = [r.get("operator_check") for r in map(json.loads, ours.log_path.open())]
    assert checks == ["no-baseline", "stale-baseline", "operator"]


def test_our_own_changes_and_videos_are_not_the_operator(tmp_path, authhub):
    b = quick(Bridge(FakeBackend(tap_shows=OTHER), config(authhub[0]), log_path=tmp_path / "calls.jsonl"))

    async def go():
        await b.tap("iphone12", 1, 2)  # our tap changes the screen (LIT -> OTHER)
        await asyncio.sleep(0.15)
        await b.tap("iphone12", 3, 4)  # baseline was taken after ours settled: no operator
        b.backend.animate = True  # a video plays: every screenshot differs
        await asyncio.sleep(0.15)
        await b.tap("iphone12", 5, 6)  # inconclusive, carry on
        await b.close()

    asyncio.run(go())
    assert len([c for c in b.backend.calls if c[0] == "tap"]) == 3


def test_quick_inputs_are_not_rechecked(tmp_path, authhub):
    b = Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / "calls.jsonl")  # op_gap 5 s

    async def go():
        for i in range(4):
            await b.tap("iphone12", i, i)
        await b.close()

    asyncio.run(go())
    assert b.backend.shots == 1  # one check before the first input only


def test_a_caller_lease_is_refused_but_not_released_when_the_operator_is_active(tmp_path, authhub):
    from phonebridge.bridge import OperatorActive

    cmd, log = authhub
    b = quick(Bridge(FakeBackend(), config(cmd), log_path=tmp_path / "calls.jsonl"))

    async def go():
        await b.tap("iphone12", 1, 2, lease="callertok")
        await asyncio.sleep(0.1)
        b.backend.current = OTHER
        await asyncio.sleep(0.1)
        with pytest.raises(OperatorActive):
            await b.tap("iphone12", 3, 4, lease="callertok")
        await b.close()

    asyncio.run(go())
    assert "release" not in authhub_calls(log)  # the caller owns its lease


def test_auto_lock_is_not_the_operator(tmp_path, authhub, fake_ocr):
    from phonebridge.bridge import PhoneLocked

    b = quick(Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / "calls.jsonl"))

    async def go():
        await b.tap("iphone12", 1, 2)
        await asyncio.sleep(0.1)
        b.backend.current = DARK  # idle: iOS turns the display off and locks
        await asyncio.sleep(0.1)
        with pytest.raises(PhoneLocked, match="display is off"):  # not "operator appears active"
            await b.tap("iphone12", 3, 4)
        b.backend.screens = [DARK, LIT]
        assert await b.unlock("iphone12") == {"was_locked": True, "unlocked": True}  # no backoff to wait out
        await b.close()

    asyncio.run(go())


def test_input_from_another_phonebridge_process_is_not_the_operator(tmp_path, authhub):
    from phonebridge.bridge import OperatorActive

    phone = FakeBackend()  # one phone, two phonebridge processes (say the MCP server and a CLI run)
    server, cli = (quick(Bridge(phone, config(authhub[0]), log_path=tmp_path / f"{n}.jsonl")) for n in ("a", "b"))

    async def go():
        await server.tap("iphone12", 1, 2, lease="callertok")
        await asyncio.sleep(0.1)  # the server's baseline: LIT
        phone.tap_shows = OTHER
        await server.release_reservation("iphone12", "local")
        await cli.tap("iphone12", 3, 4, lease="callertok")  # the CLI run leaves the phone on another screen
        await asyncio.sleep(0.1)
        await cli.release_reservation("iphone12", "local")
        await server.tap("iphone12", 5, 6, lease="callertok")  # changed, but by phonebridge: carry on
        await asyncio.sleep(0.1)
        phone.current = LIT  # now the operator
        await asyncio.sleep(0.1)
        with pytest.raises(OperatorActive, match="operator appears active"):
            await server.tap("iphone12", 7, 8, lease="callertok")
        await server.release_reservation("iphone12", "local")
        with pytest.raises(OperatorActive, match="backs off for"):  # the other process backs off too
            await cli.home("iphone12", lease="callertok")
        await server.close()
        await cli.close()

    asyncio.run(go())
    assert [c[2:] for c in phone.calls if c[0] == "tap"] == [(1, 2), (3, 4), (5, 6)]



def test_tap_path_indigo_needs_a_backend_that_has_it(bridge):
    with pytest.raises(RuntimeError, match="indigo"):
        asyncio.run(bridge.tap("iphone12", 10, 10, path="indigo"))
    with pytest.raises(ValueError):
        asyncio.run(bridge.tap("iphone12", 10, 10, path="mouse"))


def test_swipe_path_indigo_needs_a_backend_that_has_it(bridge):
    with pytest.raises(RuntimeError, match="indigo"):
        asyncio.run(bridge.swipe("iphone12", 10, 500, 10, 100, path="indigo"))


def test_swipe_rows_carry_start_end_and_lift_velocity(bridge):
    asyncio.run(bridge.swipe("iphone12", 6, 1200, 700, 1200, duration=0.3))
    row = [json.loads(l) for l in bridge.log_path.read_text().splitlines() if '"op": "swipe"' in l][-1]
    assert abs(row["start"][0] - 6) < 80 and abs(row["end"][0] - 700) < 120
    assert row["lift_v"] is not None and row["lift_v"] >= 0 and len(row["lift_dir"]) == 2


def test_swipe_shape_from_a_known_path():
    from phonebridge.bridge import swipe_shape
    pts = [(0, 0, 0.0), (100, 0, 0.1), (130, 0, 0.11), (160, 0, 0.12), (190, 0, 0.13)]
    s = swipe_shape(pts)
    assert s["start"] == [0, 0] and s["end"] == [190, 0] and s["lift_v"] == 3.0 and s["lift_dir"] == [90, 0]


# --- pre-input checks stay off the input path (R1-IG attempt 2: a swipe held 7.5 s before its 228 ms path) ------------


def test_a_caller_lease_is_checked_once_then_kept_fresh_in_the_background(tmp_path):
    log, revoked = tmp_path / "authhub.log", tmp_path / "revoked"
    script = tmp_path / "authhub"
    script.write_text(f"""#!/bin/sh
echo "$@" >> {log}
case "$2" in
  check) sleep 0.3; [ "$3" = callertok ] && [ ! -e {revoked} ] || {{ echo "no live lease on $4" >&2; exit 1; }}; echo 600 ;;
esac
""")
    script.chmod(0o755)
    b = Bridge(FakeBackend(), config([str(script)]), log_path=tmp_path / "calls.jsonl", task_poll=0.2)

    async def go():
        for _ in range(5):
            await b.tap("iphone12", 1, 2, lease="callertok")
            await asyncio.sleep(0.15)
        revoked.write_text("")  # the holder released it: the watcher sees that within one refresh
        await asyncio.sleep(0.9)
        from phonebridge.bridge import TaskEnded
        with pytest.raises((LeaseError, TaskEnded)):
            await b.tap("iphone12", 1, 2, lease="callertok")
    asyncio.run(go())
    rows = [json.loads(l) for l in b.log_path.read_text().splitlines() if '"op": "tap"' in l]
    holds = [r["hold_ms"] for r in rows if "hold_ms" in r]
    assert holds[0] >= 250  # the first input verifies by ssh (0.3 s here)
    assert all(h < 50 for h in holds[1:5]), holds  # the rest never wait on it
    assert all({"hold_ms", "operator_ms"} <= set(r) for r in rows[:5])
    assert authhub_calls(log).count("check") >= 3  # the background refreshes did the checking


def test_backend_recoveries_inside_an_input_land_in_its_call_row(bridge):
    import collections
    be = bridge.backend
    be.recoveries = collections.deque([{"n": 1, "call": "fg", "error": "old", "after_ms": 1, "reset": "dvt"}])
    real = be.tap

    async def tap(udid, x, y, press):
        be.recoveries.append({"n": 2, "call": "fg", "error": "TimeoutError: ", "after_ms": 1500, "reset": "dvt"})
        return await real(udid, x, y, press)
    be.tap = tap
    asyncio.run(bridge.tap("iphone12", 1, 2))
    row = [json.loads(l) for l in bridge.log_path.read_text().splitlines() if '"op": "tap"' in l][-1]
    assert row["recoveries"] == [{"n": 2, "call": "fg", "error": "TimeoutError: ", "after_ms": 1500, "reset": "dvt"}]


def test_a_tap_held_too_long_before_it_starts_is_refused_as_stale(tmp_path):
    """controller-verify S3: an input lands on a fresh look. Pre-input checks over 1 s (here a slow first lease
    check) refuse the tap with zero input sent; the next one, checked, goes through."""
    from phonebridge.bridge import StaleInput
    log = tmp_path / "authhub.log"
    script = tmp_path / "authhub"
    script.write_text(f"""#!/bin/sh
echo "$@" >> {log}
case "$2" in check) sleep 1.2; [ "$3" = callertok ] || exit 1; echo 600 ;; esac
""")
    script.chmod(0o755)
    b = Bridge(FakeBackend(), config([str(script)]), log_path=tmp_path / "calls.jsonl")

    async def go():
        with pytest.raises(StaleInput, match="stale: re-look.*hold_ms.*zero input sent"):
            await b.tap("iphone12", 1, 2, lease="callertok")
        assert b.backend.calls == []  # zero input
        await b.tap("iphone12", 1, 2, lease="callertok")  # vouched now: no wait
    asyncio.run(go())
    assert [c[0] for c in b.backend.calls] == ["tap"]
    rows = [json.loads(l) for l in b.log_path.read_text().splitlines() if '"op": "tap"' in l]
    assert rows[0]["ok"] is False and rows[0]["stale"] and rows[0]["pre_ms"] > 1000 and rows[1]["pre_ms"] < 1000


def test_a_tap_after_a_backend_recovery_is_refused_as_stale(bridge, monkeypatch):
    import collections
    from phonebridge.bridge import StaleInput
    be = bridge.backend
    be.recoveries = collections.deque()
    real = bridge._operator_check

    async def check(*a, **k):
        be.recoveries.append({"n": 1, "call": "fg", "error": "TimeoutError: ", "after_ms": 1500, "reset": "dvt"})
        return await real(*a, **k)
    monkeypatch.setattr(bridge, "_operator_check", check)
    with pytest.raises(StaleInput, match="a backend call recovered \\(fg: TimeoutError"):
        asyncio.run(bridge.tap("iphone12", 1, 2))
    assert be.calls == []


def test_think_false_skips_the_drawn_pause_and_keeps_the_guards(bridge, monkeypatch):
    """The director paces its own inputs (dwell draws); the bridge's think pause would stack on them."""
    import time as _t
    monkeypatch.setattr(bridge.human, "think", lambda op: 0.4)

    async def go():
        await bridge.tap("iphone12", 1, 2)
        t = _t.monotonic()
        await bridge.tap("iphone12", 1, 2)
        paused = _t.monotonic() - t
        t = _t.monotonic()
        await bridge.tap("iphone12", 1, 2, think=False)
        return paused, _t.monotonic() - t
    paused, fast = asyncio.run(go())
    rows = [json.loads(l) for l in bridge.log_path.read_text().splitlines() if '"op": "tap"' in l]
    assert rows[1]["think_ms"] >= 300 and rows[2]["think_ms"] == 0 and rows[2]["think"] == "caller"
    assert fast < paused and "hold_ms" in rows[2] and "pre_ms" in rows[2]


def test_double_tap_takes_think_false_and_is_refused_when_stale(tmp_path, monkeypatch):
    """Likes (R2 on) get the tap/swipe guarantees: think=False skips the drawn pause; a pre-input stall refuses it."""
    from phonebridge.bridge import StaleInput
    log = tmp_path / "authhub.log"
    script = tmp_path / "authhub"
    script.write_text(f"""#!/bin/sh
echo "$@" >> {log}
case "$2" in check) sleep 1.2; [ "$3" = callertok ] || exit 1; echo 600 ;; esac
""")
    script.chmod(0o755)
    b = Bridge(FakeBackend(), config([str(script)]), log_path=tmp_path / "calls.jsonl")
    monkeypatch.setattr(b.human, "think", lambda op: 0.4)

    async def go():
        with pytest.raises(StaleInput, match="stale: re-look"):
            await b.double_tap("iphone12", 300, 600, lease="callertok", think=False)
        assert b.backend.calls == []  # zero input
        await b.double_tap("iphone12", 300, 600, lease="callertok", think=False)
    asyncio.run(go())
    rows = [json.loads(l) for l in b.log_path.read_text().splitlines() if '"op": "double_tap"' in l]
    assert [c[0] for c in b.backend.calls] == ["tap", "tap"]
    assert rows[1]["think_ms"] == 0 and rows[1]["think"] == "caller" and rows[1]["pre_ms"] < 1000


def test_a_running_task_keeps_the_backend_warm_until_it_ends_and_not_while_parked(bridge, monkeypatch):
    from phonebridge import bridge as bridge_mod
    monkeypatch.setattr(bridge_mod, "WARM_EVERY_S", 0.05)
    warm = []
    bridge.backend.keep_warm = lambda udid: warm.append(udid)

    async def go():
        await bridge.tap("iphone12", 1, 2)  # a task begins with the first input
        await asyncio.sleep(0.3)
        n = len(warm)
        assert n >= 3 and set(warm) == {"U12"}
        bridge.park("operator")
        await asyncio.sleep(0.2)
        assert len(warm) == n  # parked: nothing kept warm
        bridge.unpark("operator")
        bridge.end_task("iphone12", "test")
        await asyncio.sleep(0.2)
        m = len(warm)
        await asyncio.sleep(0.2)
        assert len(warm) == m  # the task ended: the loop is gone
    asyncio.run(go())


def test_expiry_during_pacing_sends_no_input(tmp_path, authhub, monkeypatch):
    b = Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / 'calls.jsonl')
    async def expires(*args, **kw):
        b.leases._expires[('callertok', 'device:iphone12')] = 0
    monkeypatch.setattr(b, '_think', expires)
    async def go():
        with pytest.raises(LeaseError, match='expired before input'):
            await b.tap('iphone12', 1, 2, lease='callertok')
        assert b.backend.calls == []
        await b.close()
    asyncio.run(go())


def test_explicit_token_does_not_allow_two_controllers(tmp_path, authhub):
    b = Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / 'calls.jsonl')
    async def go():
        await b.tap('iphone12', 1, 2, caller='first', lease='callertok', think=False)
        with pytest.raises(LeaseError, match='controller|reserved|busy'):
            await b.tap('U12', 1, 2, caller='second', lease='callertok', think=False)
        await b.close()
    asyncio.run(go())


def test_reservation_release_waits_for_input_and_is_owner_checked(tmp_path, authhub):
    b = Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / 'calls.jsonl')
    entered, finish = asyncio.Event(), asyncio.Event()
    async def tap(*args):
        entered.set()
        await finish.wait()
    b.backend.tap = tap
    async def go():
        running = asyncio.create_task(b.tap('iphone12', 1, 2, caller='a', lease='callertok', think=False))
        await entered.wait()
        with pytest.raises(LeaseError, match='only the controller'):
            await b.release_reservation('U12', 'b')
        freeing = asyncio.create_task(b.release_reservation('U12', 'a'))
        await asyncio.sleep(0)
        with pytest.raises(LeaseError, match='busy'):
            await b.reserve('iphone12', 'b')
        assert not freeing.done()
        finish.set()
        await running
        await freeing
        await b.reserve('iphone12', 'b')
        await b.close()
    asyncio.run(go())


def test_reservations_exclude_other_process_instances_but_not_other_phones(tmp_path, authhub):
    a, b = [Bridge(FakeBackend(), config(authhub[0]), log_path=tmp_path / n / 'calls.jsonl') for n in 'ab']
    async def go():
        await a.reserve('iphone12', 'one')
        with pytest.raises(LeaseError, match='another process'):
            await b.reserve('U12', 'two')
        await b.reserve('iphone16', 'two')
        await a.release_reservation('iphone12', 'one')
        await b.reserve('iphone12', 'two')
        await a.close()
        await b.close()
    asyncio.run(go())


def test_implicit_expired_lease_is_reported_before_reacquiring(authhub):
    cmd, log = authhub
    leases = Leases(cmd, ttl=120)
    leases.ensure('device:iphone12')
    leases._held['device:iphone12'] = ('tok123', 0)
    with pytest.raises(LeaseError, match='expired'):
        leases.ensure('device:iphone12')
    assert authhub_calls(log) == ['acquire']


def test_failed_renewal_never_silently_acquires_a_successor(authhub, monkeypatch):
    import subprocess

    cmd, log = authhub
    leases = Leases(cmd, ttl=120)
    leases.ensure("device:iphone12")
    leases._held["device:iphone12"] = ("tok123", time.monotonic() - 70)
    calls = []

    def fail(verb, *args):
        calls.append(verb)
        return subprocess.CompletedProcess([], 1, "", "lease lost")

    monkeypatch.setattr(leases, "_run", fail)
    with pytest.raises(LeaseError, match="lost"):
        leases.ensure("device:iphone12")
    with pytest.raises(LeaseError, match="expired"):
        leases.ensure("device:iphone12")
    assert calls == ["renew"]
    assert authhub_calls(log) == ["acquire"]


def test_expired_controller_cannot_resume_after_another_controller(bridge):
    async def go():
        await bridge.tap("iphone12", 1, 2, caller="a", lease="callertok")
        old = bridge._reservations["U12"]
        bridge.leases._expires[("callertok", "device:iphone12")] = 0
        await bridge._expire_controller("iphone12", "a", old, "callertok")
        await bridge.tap("iphone12", 3, 4, caller="b", lease="callertok")
        await bridge.release_reservation("iphone12", "b")
        with pytest.raises(LeaseError, match="reservation expired"):
            await bridge.tap("iphone12", 5, 6, caller="a", lease="callertok")
        assert [c for c in bridge.backend.calls if c[0] == "tap"] == [
            ("tap", "U12", 1, 2), ("tap", "U12", 3, 4)]
        await bridge.close()

    asyncio.run(go())


def test_controller_expiry_timer_honors_refreshed_verification(bridge):
    async def go():
        await bridge.tap('iphone12', 1, 2, caller='a', lease='callertok')
        held = bridge._reservations['U12']
        held['timer'].cancel()
        old_timer = held['timer'] = asyncio.get_running_loop().call_later(
            0, lambda: asyncio.create_task(bridge._expire_controller(
                'iphone12', 'a', held, 'callertok')))
        # A heartbeat has reverified the same token before its earlier timer fires.
        bridge.leases._expires[('callertok', 'device:iphone12')] = time.monotonic() + 600
        for _ in range(100):
            if held['timer'] is not old_timer or 'U12' not in bridge._reservations:
                break
            await asyncio.sleep(.01)
        assert bridge._reservations.get('U12') is held
        assert held['timer'] is not old_timer
        await bridge.tap('iphone12', 3, 4, caller='a', lease='callertok')
        await bridge.close()
    asyncio.run(go())


def test_controller_input_idle_expires_even_while_lease_renews(bridge):
    async def go():
        bridge.controller_idle = .03
        await bridge.tap('iphone12', 1, 2, caller='a', lease='callertok')
        bridge.leases._expires[('callertok', 'device:iphone12')] = time.monotonic() + 600
        for _ in range(100):
            if 'U12' not in bridge._reservations:
                break
            await asyncio.sleep(.01)
        assert 'U12' not in bridge._reservations
        with pytest.raises(LeaseError, match='reservation expired'):
            await bridge.tap('iphone12', 3, 4, caller='a', lease='callertok')
        await bridge.tap('iphone12', 5, 6, caller='b', lease='callertok')
        await bridge.close()
    asyncio.run(go())


def test_tap_path_vdigitizer_passes_the_contact_size(bridge):
    got = []

    async def tap_vdigitizer(udid, x, y, press, contact_mm):
        got.append(contact_mm)
    bridge.backend.tap_vdigitizer = tap_vdigitizer
    asyncio.run(bridge.tap("iphone12", 10, 10, path="vdigitizer", contact_mm=10.5, think=False))
    assert got == [10.5]
    with pytest.raises(ValueError, match="report \\| indigo \\| vdigitizer"):
        asyncio.run(bridge.tap("iphone12", 10, 10, path="nope"))


def test_ax_probe_needs_a_backend_that_has_it_and_is_held_like_an_input(bridge):
    with pytest.raises(RuntimeError, match="ax_probe"):
        asyncio.run(bridge.ax_probe("iphone12", ["caps"]))
    got = []

    async def ax_probe(udid, ops, max_nodes):
        got.append((ops, max_nodes))
        return {"op_ms": {"caps": 1.0}, "caps": {}}
    bridge.backend.ax_probe = ax_probe
    out = asyncio.run(bridge.ax_probe("iphone12", ["caps", "tree"], 30))
    assert got == [(["caps", "tree"], 30)] and out["op_ms"] == {"caps": 1.0}
    rows = [json.loads(l) for l in bridge.log_path.read_text().splitlines()]
    assert any(r["op"] == "ax_probe" for r in rows)


def test_a_slow_frontmost_query_alone_does_not_make_a_tap_stale(bridge, monkeypatch):
    """iOS 27.2 (live 2026-11-08): DVT's process list answers in 0.4-2.3 s, so the frontmost check outlasted the 1 s
    stale bound on every tap and refused it (zero input, but no input ever). The query re-verifies the app at the
    input, so its own time no longer counts as staleness; a slow hold still does (the test above)."""
    import time as _t
    be = bridge.backend
    asyncio.run(bridge.launch("iphone12", "com.burbn.instagram"))  # the task's app: every tap is checked frontmost

    async def slow_foreground(udid):
        await asyncio.sleep(1.6)
        return "com.burbn.instagram"
    monkeypatch.setattr(be, "foreground", slow_foreground, raising=False)
    monkeypatch.setattr(bridge, "_seen_app", {}, raising=False)
    asyncio.run(bridge.tap("iphone12", 1, 2))
    row = [json.loads(l) for l in bridge.log_path.read_text().splitlines() if '"op": "tap"' in l][-1]
    assert row["ok"] and row["fg_ms"] >= 1500 and row["pre_ms"] < 1000 and row["foreground"] == "com.burbn.instagram"
    assert [c[0] for c in be.calls if c[0] == "tap"] == ["tap"]


def test_a_frontmost_query_far_slower_than_ios_27_2s_usual_still_refuses_the_tap(bridge, monkeypatch):
    """tw-phone-review: only ≤2 s of query overrun is excused; a 3 s query (an outlier) is a zero-input re-look."""
    from phonebridge.bridge import StaleInput
    be = bridge.backend
    asyncio.run(bridge.launch("iphone12", "com.burbn.instagram"))

    async def slower(udid):
        await asyncio.sleep(3.0)
        return "com.burbn.instagram"
    monkeypatch.setattr(be, "foreground", slower, raising=False)
    monkeypatch.setattr(bridge, "_seen_app", {}, raising=False)
    with pytest.raises(StaleInput, match="zero input sent"):
        asyncio.run(bridge.tap("iphone12", 1, 2))
    assert [c for c in be.calls if c[0] == "tap"] == []
