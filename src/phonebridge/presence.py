"""Is the phone really there? A cheap, real round trip instead of an inference (lanes misread the phone's attachment
all night on 2026-10-18; `system_profiler` doesn't list it at all).

`probe(udid)` within ~2 s: usbmux's listing (attached, over USB or Wi-Fi), tunneld's listing (does a CoreDevice tunnel
to it exist; None when tunneld itself doesn't answer), and one lockdown GetValue (the phone answers; its name and iOS).
`Presence` caches a result for ~10 s per phone so monitors can ask often without touching the phone each time.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request

from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.usbmux import list_devices

TUNNELD_URL = "http://127.0.0.1:49151/"
BUDGET_S = 2.0
TTL_S = 10.0


async def tunneld_listing(url: str, timeout: float) -> dict:
    def get():
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    return await asyncio.to_thread(get)


async def probe(udid: str, budget: float = BUDGET_S, tunneld_url: str = TUNNELD_URL) -> dict:
    t0 = time.perf_counter()
    left = lambda: max(0.05, budget - (time.perf_counter() - t0))  # noqa: E731
    out = {"udid": udid, "attached": False, "transport": "none", "tunnel": None, "lockdown": False,
           "name": None, "ios": None, "errors": {}}
    devs = []
    try:
        devs = [d for d in await asyncio.wait_for(list_devices(), left()) if d.serial == udid]
    except Exception as e:
        out["errors"]["usbmux"] = f"{type(e).__name__}: {e}"[:200]
    if devs:
        out["attached"] = True
        usb = [d for d in devs if d.is_usb]
        out["transport"] = "usb" if usb else "wifi"
    try:
        listing = await asyncio.wait_for(tunneld_listing(tunneld_url, min(0.5, left())), min(0.6, left()))
        out["tunnel"] = bool(listing.get(udid))
    except Exception as e:
        out["errors"]["tunneld"] = f"{type(e).__name__}: {e}"[:200]
    if devs:
        d = (usb or devs)[0]

        async def ask():
            async with await create_using_usbmux(udid, autopair=False, connection_type=d.connection_type) as ld:
                return await ld.get_value(key="DeviceName"), await ld.get_value(key="ProductVersion")
        try:
            out["name"], out["ios"] = await asyncio.wait_for(ask(), left())
            out["lockdown"] = True
        except asyncio.TimeoutError:
            out["errors"]["lockdown"] = f"no answer within {budget:.1f} s"
        except Exception as e:
            out["errors"]["lockdown"] = f"{type(e).__name__}: {e}"[:200]
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["ts"] = round(time.time(), 3)
    return out


class Presence:
    """One probe per phone per `ttl` seconds; concurrent askers share it. `extra(udid, probe)` adds fields (the
    bridge adds the frontmost app and what it last saw) inside the same cached result."""

    def __init__(self, ttl: float = TTL_S, budget: float = BUDGET_S, extra=None, tunneld_url: str = TUNNELD_URL):
        self.ttl, self.budget, self.extra, self.tunneld_url = ttl, budget, extra, tunneld_url
        self._cache: dict[str, tuple[float, dict]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, udid: str, fresh: bool = False) -> dict:
        lock = self._locks.setdefault(udid, asyncio.Lock())
        async with lock:
            hit = self._cache.get(udid)
            if hit and not fresh and time.monotonic() - hit[0] < self.ttl:
                return {**hit[1], "cached": True, "age_s": round(time.monotonic() - hit[0], 1)}
            p = await probe(udid, self.budget, self.tunneld_url)
            if self.extra is not None:
                try:
                    p.update(await self.extra(udid, p))
                except Exception as e:
                    p["errors"]["extra"] = f"{type(e).__name__}: {e}"[:200]
            self._cache[udid] = (time.monotonic(), p)
            return {**p, "cached": False, "age_s": 0.0}
