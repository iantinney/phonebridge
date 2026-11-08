"""REC-05: the Mac beats to the VM every 60 s while awake; the VM's dead-man (authhub.deadman) judges the silence.

A failed send backs off (2x, 4x, ... capped) and is logged once when it starts and once when it ends, never
once per try: bthid.log reached 13 GB on 2026-10-18 from a reconnect loop. `cap_logs` is the other half of that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from ..proto import hello


class Heartbeat:
    def __init__(self, send: Callable[[dict], None], *, state: Callable[[], str], tasks: Callable[[], dict],
                 boot: float, log: Path, every: float = 60.0, mac: Callable[[], dict] | None = None,
                 max_backoff: float = 600.0, extra: dict | None = None):
        self.send, self.state, self.tasks, self.mac = send, state, tasks, mac
        self.boot, self.log, self.every, self.max_backoff = boot, Path(log), every, max_backoff
        self.missed, self.broken = 0, None  # consecutive failed sends; the last state()/tasks() error logged
        self.extra = extra or {}  # sent in every beat, e.g. {"planned": True} after a deploy's restart

    def beat(self, state: str | None = None) -> float:
        """Send one beat (`state` overrides, e.g. "sleeping" from the will-sleep handler); return the delay to the next."""
        beat = {"ts": time.time(), "state": state or self.state(), "tasks": self.tasks(), "boot": self.boot,
                "every": self.every, "mac": self.mac() if self.mac else {}, **hello(), **self.extra}
        try:
            self.send(beat)
        except Exception as e:
            if not self.missed:
                self._log(event="send_failed", state="tunnel_down", error=str(e))
            self.missed += 1
            return min(self.every * 2 ** min(self.missed, 16), self.max_backoff)
        if self.missed:
            self._log(event="send_ok", missed=self.missed)
            self.missed = 0
        return self.every

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            start = loop.time()  # the cadence is start to start, so a slow send does not stretch it
            try:
                delay, self.broken = await asyncio.to_thread(self.beat), None
            except Exception as e:  # state()/tasks()/mac() broke; a send failure is beat()'s own
                if self.broken != (err := f"{type(e).__name__}: {e}"):
                    self.broken = err
                    self._log(event="beat_error", error=err)
                delay = self.every
            await asyncio.sleep(max(0.0, start + delay - loop.time()))

    def _log(self, **event) -> None:
        with contextlib.suppress(OSError):  # a full disk must not stop the beats
            self.log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with os.fdopen(os.open(self.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a") as f:
                f.write(json.dumps({**event, "ts": time.time()}) + "\n")


def cap_logs(paths, max_bytes: int = 50 * 2**20, keep: int = 2**20) -> list[Path]:
    """Keep the last `keep` bytes of each log over `max_bytes` in <name>.1 and truncate it IN PLACE: launchd holds
    the daemon logs open O_APPEND, so renaming would leave the daemon writing to the moved file."""
    capped = []
    for path in map(Path, paths):
        try:
            size = path.stat().st_size
            if size <= max_bytes:
                continue
            with path.open("rb") as f:
                f.seek(max(0, size - keep))
                path.with_name(path.name + ".1").write_bytes(f.read())
            os.truncate(path, 0)
        except OSError:  # missing, or not ours to cap: the rest still are
            continue
        capped.append(path)
    return capped


def send_over(cmd: list[str], timeout: float = 20) -> Callable[[dict], None]:
    """A `send(beat)` that pipes the beat to `<cmd> deadman beat`, cmd like ["ssh", "lease-host", "authhub"]."""
    cmd = list(cmd)
    if cmd[0] == "ssh":  # never prompt or hang on a dead tunnel
        for key, value in (("ConnectTimeout", "10"), ("BatchMode", "yes")):
            if not any(a.split("=")[0] == key for a in cmd):
                cmd[1:1] = ["-o", f"{key}={value}"]

    def send(beat: dict) -> None:
        try:
            subprocess.run([*cmd, "deadman", "beat"], input=json.dumps(beat), text=True, capture_output=True,
                           check=True, timeout=timeout)
        except subprocess.CalledProcessError as e:
            raise OSError(e.stderr.strip() or str(e)) from e

    return send


def understand_health(url: str = "http://127.0.0.1:8935/health", timeout: float = 1.0) -> dict | None:
    """The local understanding service's /health (understand.py), or None when it isn't running."""
    import urllib.request

    with contextlib.suppress(Exception):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    return None


def mac_status() -> dict:
    """Battery percent, AC, free disk (GB), the understanding service's health: best effort, what the Mac can tell
    without a daemon of its own."""
    if sys.platform != "darwin":
        return {}
    out = {}
    if (uh := understand_health()) is not None:
        out["understand"] = {k: uh.get(k) for k in ("loaded", "idle_s", "last_ms") if k in uh}
    with contextlib.suppress(Exception):
        out["disk_free_gb"] = round(shutil.disk_usage(Path.home()).free / 1e9)
    with contextlib.suppress(Exception):
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        if m := re.search(r"(\d+)%", batt):
            out["battery"] = int(m[1])
        out["ac"] = "AC Power" in batt
    return out


def screen_p95(calls: Path, now: float | None = None, window: float = 300.0, tail: int = 512_000) -> tuple[float, int]:
    """REC-08: p95 of `screen` latency (s) over the last `window` s of the call log, and the sample count. Reads only
    the log's tail."""
    now = time.time() if now is None else now
    try:
        with open(calls, "rb") as f:
            f.seek(max(0, os.path.getsize(calls) - tail))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return 0.0, 0
    ms = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("op") == "screen" and r.get("ok") and now - r.get("ts", 0) <= window and "ms" in r:
            ms.append(r["ms"] / 1000)
    if not ms:
        return 0.0, 0
    ms.sort()
    return ms[min(len(ms) - 1, int(0.95 * len(ms)))], len(ms)
