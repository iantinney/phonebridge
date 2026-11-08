"""DEV-01/02/08, NET-01 preflight: a session starts only when nothing required fails or is unknown.

A probe table (CHECKS) names the source of every check: the phone over lockdown / diagnostics, the Mac, OCR,
or an operator attestation (a value with date and expiry, for what no host-side API can read: the DEV-02 settings
checklist, the egress IP). A missing reading is `unknown`, never `pass`. Every result carries the value it judged
and a fix hint. State files are mode 600 in 700 directories.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import STATE_DIR

DAY = 86400.0


@dataclass(frozen=True)
class Check:
    source: str  # lockdown | diagnostics | host | ocr | attestation
    good: object  # a value that passes (what attestations record)
    bad: object  # a value that fails
    hint: str  # how to fix it
    test: Callable[[object], bool]


def _attested(hint: str) -> Check:
    return Check("attestation", True, False, hint, lambda v: v is True)


def _on(source: str, hint: str) -> Check:
    return Check(source, True, False, hint, lambda v: v is True)


CHECKS: dict[str, Check] = {
    "unlocked": _on("lockdown", "Unlock the phone: wake it, swipe up from the bottom edge (no passcode is set)"),
    "developer_mode": _on("lockdown", "Settings › Privacy & Security › Developer Mode › On, then restart the phone"),
    "charging": _on("diagnostics", "Plug the phone into the Mac's USB cable; it must be charging"),
    "battery_temp_c": Check("diagnostics", 32.0, 41.0, "Battery above 40 °C: pause, take it out of the sun, let it cool",
                            lambda v: v <= 40.0),
    "battery_level": Check("diagnostics", 80, 15, "Battery under 20 %: charge it before a session",
                           lambda v: v >= 20),
    "free_space_gb": Check("lockdown", 50.0, 1.0, "Phone has under 4 GB free: Settings › General › iPhone Storage, "
                           "clear space", lambda v: v >= 4.0),
    "mac_disk_free_gb": Check("host", 600.0, 4.0, "Mac has under 10 GB free: clear disk space (screens and logs "
                              "land there)", lambda v: v >= 10.0),
    "app_installed": Check("lockdown", "412.0", "", "The target app is not installed: install it from the App Store",
                           lambda v: isinstance(v, str) and bool(v)),
    "auto_lock_never": _attested("Settings › Display & Brightness › Auto-Lock › Never, then "
                                 "`phonebridge attest auto_lock_never`"),
    "standby_off": _attested("Settings › StandBy › off, then `phonebridge attest standby_off`"),
    "appearance_light": _attested("Settings › Display & Brightness › Light, then `phonebridge attest appearance_light`"),
    "assistivetouch_menu_off": _attested("Settings › Accessibility › Touch › AssistiveTouch › Always Show Menu off "
                                         "(no floating button), then `phonebridge attest assistivetouch_menu_off`"),
    "reduce_motion_on": _attested("Settings › Accessibility › Motion › Reduce Motion › On, then "
                                  "`phonebridge attest reduce_motion_on`"),
    "auto_updates_off": _attested("Settings › General › Software Update › Automatic Updates › all off, then "
                                  "`phonebridge attest auto_updates_off`"),
    "low_power_off": _attested("Settings › Battery › Low Power Mode › Off, then `phonebridge attest low_power_off`"),
    "icloud_photos_off": _attested("Settings › [your name] › iCloud › Photos › Sync this iPhone off, then "
                                   "`phonebridge attest icloud_photos_off`"),
    "paste_allow": _attested("Settings › Apps › Instagram › Paste from Other Apps › Allow, then "
                             "`phonebridge attest paste_allow`"),
    "photos_full_access": _attested("Settings › Privacy & Security › Photos › Instagram › Full Access, then "
                                    "`phonebridge attest photos_full_access`"),
    "upright_on_stand": _attested("Stand the phone upright (portrait) on its stand, then "
                                  "`phonebridge attest upright_on_stand`"),
    "agent_focus": _attested("Settings › Focus › Agent (allows no people and no apps) › on, then "
                             "`phonebridge attest agent_focus`"),
    "egress_confirmed": _on("ocr", "The phone's exit IP (an IP-echo page in Safari) is not the registered IPRoyal ISP "
                            "IP ([host] egress_ip): the account stays read-only. The proxy is never changed or "
                            "rotated, here or by an agent: tell the operator"),
}


def _open(path: Path, flags: int) -> int:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, flags, 0o600)
    os.fchmod(fd, 0o600)
    return fd


def write_json(path: Path, obj) -> None:
    """Replace `path` with `obj` as JSON (mode 600, via a temp file so a crash never leaves half a file)."""
    tmp = path.with_name(path.name + ".tmp")
    with os.fdopen(_open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC), "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj) -> None:
    """Append one JSON line to `path` (mode 600)."""
    with os.fdopen(_open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND), "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


class Attestations:
    """The operator's word for checks no host-side API can read: {check: {value, by, at, expires}}."""

    def __init__(self, path: Path | None = None):
        self.path = path or STATE_DIR / "attestations.json"
        try:
            self.entries: dict[str, dict] = json.loads(self.path.read_text())
        except (OSError, ValueError):  # unreadable means unattested, which preflight reports as unknown
            self.entries = {}

    def attest(self, check: str, value: bool = True, days: float = 30, by: str = "ian", now: float | None = None) -> None:
        if check not in CHECKS or CHECKS[check].source != "attestation":
            raise KeyError(f"{check!r} is not an attestable check; attestable: {', '.join(n for n, c in CHECKS.items() if c.source == 'attestation')}")
        now = time.time() if now is None else now
        self.entries[check] = {"value": value, "by": by, "at": now, "expires": now + days * DAY}
        write_json(self.path, self.entries)

    def get(self, check: str, now: float) -> tuple[object | None, dict | None]:
        """(value, entry); an expired entry has no value, only itself."""
        entry = self.entries.get(check)
        if entry is None or entry["expires"] <= now:
            return None, entry
        return entry["value"], entry


@dataclass
class Result:
    check: str
    status: str  # pass | fail | unknown
    value: object  # the evidence: what was read or attested (None when nothing was)
    source: str
    hint: str  # "" on a pass


def _judge(name: str, check: Check, reading: dict, attestations: Attestations, now: float) -> Result:
    if check.source == "attestation":
        value, entry = attestations.get(name, now)
        if entry is None:
            return Result(name, "unknown", None, check.source, f"not attested: {check.hint}")
        if value is None and entry["expires"] <= now:
            when = datetime.fromtimestamp(entry["expires"], timezone.utc).strftime("%Y-%m-%d")
            return Result(name, "unknown", None, check.source, f"attestation expired {when}; re-attest: {check.hint}")
    else:
        value = reading.get(name)
    if value is None:
        return Result(name, "unknown", None, check.source, f"no {check.source} reading: {check.hint}")
    try:
        passed = bool(check.test(value))
    except (TypeError, ValueError):  # a reading of the wrong shape is no pass
        passed = False
    return Result(name, "pass" if passed else "fail", value, check.source, "" if passed else check.hint)


def ok(results) -> bool:
    """A session may start: no required check failed or is unknown."""
    return all(r.status == "pass" for r in results)


def preflight(reading: dict, attestations: Attestations, now: float | None = None, log: Path | None = None) -> list[Result]:
    """Judge every check in CHECKS order; with `log`, append the verdict as one JSON line."""
    now = time.time() if now is None else now
    results = [_judge(n, c, reading, attestations, now) for n, c in CHECKS.items()]
    if log is not None:
        append_jsonl(log, {"ts": now, "ok": ok(results),
                           "results": {r.check: {"status": r.status, "value": r.value} for r in results}})
    return results


IP = re.compile(r'"ip"\s*:\s*"([0-9a-fA-F.:]+)"')


async def egress(bridge, device: str, lease: str | None = None, url: str = "https://ipinfo.io/json") -> str | None:
    """NET-01 (method per E24): the phone's exit IP, read off an IP-echo page opened in Safari. Never touches the
    proxy configuration. Needs the device lease (open_url is input)."""
    await bridge.open_url(device, url, lease=lease)
    for _ in range(4):
        await asyncio.sleep(2.0)
        text = " ".join(i["text"] for i in await bridge.ocr(device))
        if m := IP.search(text.replace("“", '"').replace("”", '"')):
            return m[1]
    return None


async def read(backend, udid: str, app: str | None = None, egress_ip: str | None = None,
               seen_ip: str | None = None) -> dict:
    """What the host can probe itself; anything it can't (or that errors) is absent, so preflight says unknown.
    The backend is duck-typed: `lock_state`, `battery` -> {temperature, level, charging}, `devices`."""
    async def probe(name: str, *args):
        try:
            return await getattr(backend, name)(*args)
        except Exception:  # absent method, USB hiccup, locked phone: all just "no reading"
            return None

    lock, battery, devices = await probe("lock_state", udid), await probe("battery", udid), await probe("devices")
    storage = await probe("storage", udid)
    version = await probe("app_version", udid, app) if app else None
    phone = next((d for d in devices or [] if d.get("udid") == udid), {})
    try:
        disk = shutil.disk_usage(Path.home()).free / 2**30
    except OSError:
        disk = None
    reading = {"unlocked": None if not lock or lock.get("locked") is None else not lock["locked"],
               "developer_mode": phone.get("developer_mode"),
               "battery_temp_c": (battery or {}).get("temperature"), "battery_level": (battery or {}).get("level"),
               "charging": (battery or {}).get("charging"), "mac_disk_free_gb": disk,
               "free_space_gb": (storage or {}).get("free_gb"), "app_installed": version,
               "egress_confirmed": None if not (egress_ip and seen_ip) else seen_ip == egress_ip}
    return {k: v for k, v in reading.items() if v is not None}
