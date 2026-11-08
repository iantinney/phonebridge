"""Mac power, USB and phone-battery events -> park the bridge, re-check, resume (DEV-04, REC-05, DEV-08).

- **Sleep/wake.** On will-sleep every phone is parked (no input after the event) and a running task gets one
  `sleep_mid_task` notice; on did-wake each phone is re-checked (USB, trust, lock, frontmost) before input
  flows again. A task asleep longer than `stale_after` is ended (`stale`): it backs out (orchestrator's rule).
  Sources: IOKit's system-power notifications (darwin; the will-sleep callback holds sleep until the phones
  are parked and the `sleeping` beat is out), and a clock gap as the fallback (`time.monotonic()` stops
  during sleep on macOS, CLOCK_MONOTONIC does not).
- **USB.** A detach parks the phone (logged with its last battery/temperature reading); re-enumeration
  within `lost_after` s re-checks and resumes; past it, `device_lost`: the task is ended (it backs out). Ian
  unplugs the phone to use it (grill G3-G17), so a detach is never a page by itself; only flapping is (`flaky`
  quick drops in an hour, each back within `lost_after` s: too short for a person). Gone while the cable's USB-C
  end is still in the Mac is logged (`usb_gone_cable_in`) but not paged: pulling the Lightning end looks the same.
- **Battery.** Above `hot` °C the phone is parked until it cools to `cool`; below `low` % and not charging
  mid-task it is parked with one notice.
Nothing here pages Ian because the Mac slept (DECISIONS Q1).
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..config import STATE_DIR, site, udid_key

EVENTS = STATE_DIR / "host" / "events.jsonl"
PROBE_TIMEOUT_S = 10.0  # bounds a single phone, including the driver's 8 s lockdown-info budget


class Watch:
    def __init__(self, bridge, *, clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 lost_after: float = 30.0, stale_after: float = 1800.0, hot: float = 40.0, cool: float = 38.0,
                 low: float = 20.0, handoff: Callable[[str], object] | None = None, events: Path = EVENTS,
                 cable: str | None = None, gap: Callable[[], float] | None = None, flaky: int = 6,
                 plugged: Callable[[], bool | None] | None = None, dead_after: float = 60.0):
        self.bridge, self.clock, self.wall = bridge, clock, wall
        self.gap = gap or _sleep_gap  # seconds slept since boot
        self._gap = self.gap()  # ...as of the last wake handled
        self.lost_after, self.stale_after, self.hot, self.cool, self.low = lost_after, stale_after, hot, cool, low
        self.handoff, self.events, self.cable = handoff or (lambda text: None), events, cable
        self.asleep_since: float | None = None  # wall time of the last will-sleep
        self._waking = False  # a wake is being handled (its re-check awaits the phone)
        self.gone: dict[str, float] = {}  # udid -> clock time it left USB
        self.lost: set[str] = set()  # udids past lost_after (handed off once)
        self.seen: set[str] = set()  # udids seen on USB since this daemon started (a never-plugged phone isn't lost)
        self.readings: dict[str, dict] = {}  # udid -> last battery reading
        self.on_sleep: list[Callable[[], object]] = []  # e.g. the heartbeat's "sleeping" beat
        self.flaky = flaky
        self.drops: dict[str, list[float]] = {}  # quick USB drops, per physical phone
        self.flap_alerted: dict[str, float] = {}
        # is the phone's cable physically connected (Type-C CC layer), whatever usbmux says? None: can't tell
        self.plugged, self.dead_after, self.dead_told = plugged or (lambda: None), dead_after, set()
        self._plug_asked: dict[str, float] = {}  # udid -> clock time of the last port query (one per 30 s)
        self._present: set[str] = set()
        self._rechecks: dict[str, asyncio.Task] = {}

    def _log(self, event: str, **kw) -> None:
        self.events.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.events.touch(mode=0o600, exist_ok=True)
        with self.events.open("a") as f:
            f.write(json.dumps({"ts": round(self.wall(), 3), "event": event, **kw}) + "\n")

    def _phones(self) -> dict[str, str]:
        return {d.udid: alias for alias, d in self.bridge.config.devices.items()}

    def state(self) -> str:
        if self.asleep_since is not None:
            return "sleeping"
        return "no_usb" if any(udid in self.gone for udid in self._phones()) else "available"

    # --- sleep / wake -----------------------------------------------------------------------
    def will_sleep(self) -> None:
        self.bridge.park("sleep")
        self.asleep_since = self.wall()
        tasks = sorted(t.alias for t in self.bridge.tasks.values())
        self._log("will_sleep", tasks=tasks)
        if tasks:
            self._log("sleep_mid_task", tasks=tasks, steps="parked: no input until the wake re-check")
        for hook in self.on_sleep:
            with contextlib.suppress(Exception):
                hook()

    async def did_wake(self, slept: float | None = None) -> dict[str, dict]:
        if self._waking:  # IOKit's powered-on and the clock-gap fallback can both arrive (01:31:23Z 10-19, 0.18 s apart)
            return {}
        self._waking = True
        try:
            return await self._did_wake(slept)
        finally:
            self._waking = False

    async def _did_wake(self, slept: float | None) -> dict[str, dict]:
        if slept is None:
            slept = self.wall() - self.asleep_since if self.asleep_since is not None else 0.0
        self._log("did_wake", slept_s=round(slept))
        out, found = {}, None
        if not hasattr(self.bridge.backend, "device_info"):
            with contextlib.suppress(Exception):
                found = await asyncio.wait_for(self.bridge.backend.devices(), PROBE_TIMEOUT_S)
        phones = self._phones()
        for alias in phones.values():
            self.bridge.park("wake", alias)
        self.bridge.unpark("sleep")  # each phone now resumes only after its own bounded check

        async def check(udid, alias):
            r = out[alias] = await self._recheck(alias, found)
            if not r["resumed"] and udid not in self.gone:
                self.gone[udid] = self.clock()  # T counts from the wake
                self.bridge.park("usb", alias)
            if slept > self.stale_after and udid in self.bridge.tasks:
                self.bridge.end_task(alias, "stale")
                self._log("task_stale", device=alias, slept_s=round(slept))
            self.bridge.unpark("wake", alias)
        await asyncio.gather(*(check(udid, alias) for udid, alias in phones.items()))
        self.asleep_since = None
        self._gap = self.gap()  # this sleep is handled: the clock-gap fallback must not see it again
        self.bridge.unpark("sleep")
        return out

    async def tick(self) -> None:
        """The fallback for a missed IOKit event: the sleep clock jumped since the last wake handled."""
        if (slept := self.gap() - self._gap) > 5.0:
            if self.asleep_since is None:
                self.will_sleep()
                self.asleep_since = self.wall() - slept
            await self.did_wake()

    async def _recheck(self, alias: str, found: list[dict] | None = None) -> dict:
        try:
            r = await asyncio.wait_for(self.bridge.recheck(alias, found), PROBE_TIMEOUT_S)
        except Exception as e:
            r = {"usb": False, "error": f"{type(e).__name__}: {e}"}
        r["resumed"] = bool(r.get("usb") and r.get("trusted"))
        self._log("recheck", device=alias, **r)
        return r

    # --- USB --------------------------------------------------------------------------------
    async def _reattached(self, udid: str, alias: str) -> None:
        r = await self._recheck(alias)
        if r["resumed"] and udid in self._present and self.asleep_since is None:
            self.gone.pop(udid, None)
            self.bridge.unpark("usb", alias)
            self.lost.discard(udid)

    async def poll_usb(self, *, wait: bool = True) -> None:
        backend = self.bridge.backend
        listed = await backend.attached() if hasattr(backend, "attached") else [d["udid"] for d in await backend.devices()]
        keys = {udid_key(u) for u in listed}
        listed = {u for u in self._phones() if udid_key(u) in keys}
        previous, self._present = self._present, set(listed)
        now = self.clock()
        for udid, alias in self._phones().items():
            if udid in listed:
                self.seen.add(udid)
                if udid in self.gone:
                    away = now - self.gone[udid]
                    attached = udid not in previous
                    if attached:
                        self.dead_told.discard(udid)
                        self._log("attach", device=alias, after_s=round(away, 1))
                    # a quick drop: the USB data link failing (15 such blips 19:05-19:20Z on 2026-10-18, the
                    # cable connected throughout; the phone's port had 34 enumeration failures), not a person
                    if attached and away <= self.lost_after:
                        drops = self.drops[udid] = [t for t in self.drops.get(udid, []) if now - t < 3600] + [now]
                        if len(drops) >= self.flaky and now - self.flap_alerted.get(udid, -1e18) >= 3600:
                            self.flap_alerted[udid] = now
                            self._log("usb_flapping", device=alias, drops_1h=len(drops))
                            self.handoff(f"[host] {alias} dropped off USB {len(drops)} times in the last hour, "
                                         f"back within {self.lost_after:.0f} s each: likely a bad cable or port.")
                    if udid not in self._rechecks or self._rechecks[udid].done():
                        self._rechecks[udid] = asyncio.create_task(self._reattached(udid, alias))
            elif udid not in self.gone:
                self.gone[udid] = now
                self.bridge.park("usb", alias)
                if udid not in self.seen:
                    continue  # never plugged in since we started: not a detach
                last = self.readings.get(udid, {})
                self._log("detach", device=alias, cable=self.cable, battery=last.get("level"),
                          temperature=last.get("temperature"), charging=last.get("charging"), reading_at=last.get("at"))
            elif (now - self.gone[udid] > self.dead_after and udid in self.seen and udid not in self.dead_told
                  and now - self._plug_asked.get(udid, -1e18) >= 30 and self._ask_plugged(udid, now)):
                # gone from usbmux while the cable's USB-C end is still in the Mac: a link failure OR Ian pulling the
                # Lightning end to use the phone (the C plug still presents a device; Ian, 01:06Z 2026-10-19). The
                # two look the same here, so this is logged for diagnosis, never paged.
                self.dead_told.add(udid)
                self._log("usb_gone_cable_in", device=alias, away_s=round(now - self.gone[udid]))
            if udid not in self._present and (pending := self._rechecks.pop(udid, None)):
                pending.cancel()  # a stale successful check must never unpark a phone that left again
            gone = self.gone.get(udid)
            if gone is not None and now - gone > self.lost_after and udid not in self.lost and udid in self.seen:
                self.lost.add(udid)
                task = self.bridge.tasks.get(udid)
                self._log("device_lost", device=alias, away_s=round(now - gone), task=bool(task))
                if task:  # unplugged (Ian using it) or the link died: the task backs out; no page for either
                    self.bridge.end_task(alias, "device_lost")
        if wait:
            await asyncio.gather(*self._rechecks.values())

    def _ask_plugged(self, udid: str, now: float) -> bool:
        self._plug_asked[udid] = now
        return bool(self.plugged())

    # --- phone battery ----------------------------------------------------------------------
    def battery(self, udid: str, reading: dict) -> None:
        self.readings[udid] = {**reading, "at": round(self.wall(), 3)}
        alias = self._phones().get(udid)
        if alias is None:
            return
        parked = self.bridge.parked(alias)
        temp, level, charging = reading.get("temperature"), reading.get("level"), reading.get("charging")
        if temp is not None:
            if temp > self.hot and "thermal" not in parked:
                self.bridge.park("thermal", alias)
                self._log("thermal_pause", device=alias, temperature=temp)
            elif temp <= self.cool and "thermal" in parked:
                self.bridge.unpark("thermal", alias)
                self._log("thermal_resume", device=alias, temperature=temp)
        if level is not None:
            if level < self.low and not charging and udid in self.bridge.tasks and "battery" not in parked:
                self.bridge.park("battery", alias)
                self._log("battery_low", device=alias, level=level)
                self.handoff(f"[host] {alias} battery {level}% and not charging mid-task: parked. "
                             f"Check the USB cable/charger.")
            elif "battery" in parked and (charging or level >= self.low + 5):
                self.bridge.unpark("battery", alias)
                self._log("battery_ok", device=alias, level=level)

    # --- the daemon -------------------------------------------------------------------------
    async def run(self, usb_every: float = 1.0, battery_every: float = 60.0) -> None:
        """Poll USB, the phones' batteries and the clock gap; take IOKit sleep/wake events on darwin."""
        loop = asyncio.get_running_loop()
        if sys.platform == "darwin":
            with contextlib.suppress(Exception):  # the gap fallback still catches wakes
                power_events(loop, self._iokit_will_sleep, lambda: asyncio.run_coroutine_threadsafe(self._woke(), loop))
        async def battery(udid):
            # A wedged diagnostic service must not suspend USB-loss detection for the whole fleet.
            with contextlib.suppress(Exception):
                reading = await asyncio.wait_for(self.bridge.backend.battery(udid), PROBE_TIMEOUT_S)
                if udid not in self.gone:
                    self.battery(udid, reading)

        next_battery, readings = 0.0, {}
        try:
            while True:
                await asyncio.sleep(usb_every)
                await self.tick()
                with contextlib.suppress(Exception):
                    await self.poll_usb(wait=False)
                if self.clock() >= next_battery:
                    next_battery = self.clock() + battery_every
                    for udid in self._phones():
                        if (udid not in self.gone and hasattr(self.bridge.backend, "battery")
                                and (udid not in readings or readings[udid].done())):
                            readings[udid] = asyncio.create_task(battery(udid))
        finally:
            pending = [*readings.values(), *self._rechecks.values()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _woke(self) -> None:
        if self.asleep_since is not None:
            await self.did_wake()

    def _iokit_will_sleep(self, loop: asyncio.AbstractEventLoop) -> None:
        """On the IOKit thread: park on the event loop (and send the sleeping beat), then let the Mac sleep."""
        done = threading.Event()
        loop.call_soon_threadsafe(lambda: (self.will_sleep(), done.set()))
        done.wait(10)


def _sleep_gap() -> float:
    """Seconds this Mac has slept since boot: CLOCK_MONOTONIC counts sleep on macOS, time.monotonic() doesn't."""
    return time.clock_gettime(time.CLOCK_MONOTONIC) - time.monotonic() if sys.platform == "darwin" else 0.0


_KEEP: list = []  # ctypes callbacks must outlive the thread that calls them


def power_events(loop: asyncio.AbstractEventLoop, will_sleep: Callable[[asyncio.AbstractEventLoop], None],
                 did_wake: Callable[[], object]) -> threading.Thread:
    """IORegisterForSystemPower on a CFRunLoop thread. will_sleep(loop) runs before the Mac is allowed to sleep
    (macOS waits up to 30 s for the acknowledgement); did_wake() runs once it has powered on."""
    iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    iokit.IORegisterForSystemPower.restype = ctypes.c_uint32
    iokit.IORegisterForSystemPower.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), callback,
                                               ctypes.POINTER(ctypes.c_uint32)]
    iokit.IONotificationPortGetRunLoopSource.restype = ctypes.c_void_p
    iokit.IONotificationPortGetRunLoopSource.argtypes = [ctypes.c_void_p]
    iokit.IOAllowPowerChange.argtypes = [ctypes.c_uint32, ctypes.c_long]
    cf.CFRunLoopGetCurrent.restype = ctypes.c_void_p
    cf.CFRunLoopAddSource.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    root = ctypes.c_uint32(0)
    CAN_SLEEP, WILL_SLEEP, POWERED_ON = 0xE0000270, 0xE0000280, 0xE0000300

    def on_message(refcon, service, message, argument):
        try:
            if message == WILL_SLEEP:
                will_sleep(loop)
            elif message == POWERED_ON:
                did_wake()
        finally:
            if message in (CAN_SLEEP, WILL_SLEEP):
                iokit.IOAllowPowerChange(root.value, argument or 0)

    handler = callback(on_message)
    _KEEP.append(handler)
    ready = threading.Event()

    def run():
        port, notifier = ctypes.c_void_p(), ctypes.c_uint32()
        root.value = iokit.IORegisterForSystemPower(None, ctypes.byref(port), handler, ctypes.byref(notifier))
        mode = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
        cf.CFRunLoopAddSource(cf.CFRunLoopGetCurrent(), iokit.IONotificationPortGetRunLoopSource(port), mode)
        ready.set()
        cf.CFRunLoopRun()

    thread = threading.Thread(target=run, name="iokit-power", daemon=True)
    thread.start()
    if not ready.wait(5) or not root.value:
        raise OSError("IORegisterForSystemPower failed")
    return thread


# --- the Mac daemon: `phonebridge serve --http` runs these beside the MCP server ------------------------
DEFAULT_LOGS = ["/tmp/phonebridge.log", "~/.local/state/phonebridge/bthid.log"]  # [host] cap_logs replaces the list


def _remote(cmd: list[str], *args: str) -> list[str]:
    """`cmd args` where cmd may be `ssh host authhub`: ssh re-joins args for a shell, so quote them."""
    import shlex

    return [*cmd, *(map(shlex.quote, args) if cmd and Path(cmd[0]).name == "ssh" else args)]


def _held(cmd: list[str], identity: str | None) -> bool:
    import subprocess

    try:
        r = subprocess.run(_remote(cmd, "status", "--json"), capture_output=True, text=True, timeout=30)
        return any(i["id"] == identity and i.get("holder") for i in json.loads(r.stdout)["identities"])
    except Exception:
        return False


def _cable_connected(port: str) -> bool | None:
    """The Type-C controller's view of the phone's port (`ioreg`): a cable with a device on it, whether or not USB
    data works. None when it can't tell."""
    import subprocess

    try:
        out = subprocess.run(["ioreg", "-r", "-n", port, "-w0", "-d1"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    if '"ConnectionActive"' not in out:
        return None
    return '"ConnectionActive" = Yes' in out


def _mullvad_connected() -> bool:
    """While Mullvad is connected the VM tunnel is down (it counts as a closed window, not a fault)."""
    import shutil
    import subprocess

    if not (exe := shutil.which("mullvad")):
        return False
    with contextlib.suppress(Exception):
        return subprocess.run([exe, "status"], capture_output=True, text=True, timeout=5).stdout.startswith("Connected")
    return False


async def run_host(bridge, settings: dict | None = None) -> None:
    """Restore live tasks from the busy marker, then watch (sleep/wake, USB, battery), beat, and cap logs."""
    import subprocess

    from .heartbeat import Heartbeat, cap_logs, mac_status, screen_p95, send_over

    s = settings if settings is not None else _settings()
    cmd = s.get("authhub_cmd") or bridge.config.lease_cmd
    try:
        restored = await asyncio.to_thread(bridge.restore_tasks, lambda task: _held(cmd, task.get("identity")))
    except Exception:  # a broken marker: start clean
        restored = []

    alert = s.get("alert_cmd") or site("ALERT_CMD", None)  # [host] alert_cmd: a command that takes the text; none: the log only

    def handoff(text: str) -> None:
        watch._log("alert", text=text)
        if alert:
            with contextlib.suppress(Exception):
                subprocess.run(_remote(alert, text), capture_output=True, timeout=30)

    port = s.get("usb_port", "Port-USB-C@3")  # where the iPhone 12 is plugged (ioreg, 2026-10-18)
    watch = Watch(bridge, handoff=handoff, cable=s.get("cable"), lost_after=s.get("lost_after", 30.0),
                  plugged=lambda: _cable_connected(port))
    watch._log("daemon_start", restored=restored, pid=os.getpid())

    def state() -> str:
        st = watch.state()
        return "tunnel_down" if st == "available" and _mullvad_connected() else st

    tasks = lambda: {t.alias: {"app": t.app, "identity": t.identity} for t in bridge.tasks.values()}  # noqa: E731
    planned = STATE_DIR / "host" / "planned-restart"  # deploy-mac.sh touches it before its kickstart
    try:
        fresh = time.time() - planned.stat().st_mtime < 600
        planned.unlink()
    except OSError:
        fresh = False
    watch._log("boot", planned=fresh)
    with contextlib.suppress(Exception):
        await watch.poll_usb()  # so the first beat already knows whether the phone is there
    services = [watch.run()]
    if cmd and s.get("heartbeat", site("HEARTBEAT", False)):  # [host] heartbeat: the lease service takes beats (a deadman)
        beat = Heartbeat(send_over(cmd), state=state, tasks=tasks, boot=time.time(),
                         log=STATE_DIR / "host" / "heartbeat.jsonl", every=s.get("heartbeat_s", 60.0), mac=mac_status,
                         extra={"planned": fresh})
        watch.on_sleep.append(lambda: beat.beat("sleeping"))
        services.append(beat.run())
    logs = [Path(p).expanduser() for p in s.get("cap_logs", site("CAP_LOGS", DEFAULT_LOGS))]
    cap = int(s.get("cap_log_mb", 50)) * 2**20

    slow_told = [-1e18]

    async def caps():  # every 5 min: log caps, and REC-08's screen latency (p95 > 2 s: one alert an hour)
        while True:
            await asyncio.to_thread(cap_logs, logs, cap)
            p95, n = await asyncio.to_thread(screen_p95, bridge.log_path)
            if n >= 5 and p95 > 2.0 and time.time() - slow_told[0] > 3600:
                slow_told[0] = time.time()
                watch._log("screen_slow", p95_s=round(p95, 2), samples=n)
                handoff(f"[host] phone screenshots are slow: p95 {p95:.1f} s over {n} in the last 5 min (USB link "
                        f"or a wedged DVT session; L3 resets it)")
            await asyncio.sleep(300)

    async def guarded(name, coro):  # one service failing must not stop the others, nor the MCP server
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            watch._log("service_failed", service=name, error=f"{type(e).__name__}: {e}")

    await asyncio.gather(*(guarded(n, c) for n, c in zip(("watch", "heartbeat"), services)), guarded("caps", caps()))


def _settings() -> dict:
    import tomllib

    from ..config import CONFIG_PATH

    with contextlib.suppress(OSError, ValueError):
        return tomllib.loads(CONFIG_PATH.read_text()).get("host", {})
    return {}


def with_host(app, bridge):
    """Wrap a Starlette app's lifespan so the host services run while the server does (macOS only): the
    per-task power assertion with the busy marker, sleep/wake, the USB watchdog, the heartbeat, log caps."""
    if sys.platform != "darwin":
        return app
    from .power import TaskPower

    bridge.power = TaskPower()
    # Two false alarms in 10 min on 2026-10-18 (a slow video, a page still loading) and Ian's word that he unplugs
    # the phone to use it (grill G3-G17): USB loss is the operator signal; the screen-change guard is opt-in.
    bridge.op_watch = bool(_settings().get("operator_watch", False))
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(a):
        async with inner(a) as state:
            task = asyncio.get_running_loop().create_task(run_host(bridge))
            try:
                yield state
            finally:
                task.cancel()
                for t in list(bridge.tasks):
                    bridge._end_task(t)

    app.router.lifespan_context = lifespan
    return app
