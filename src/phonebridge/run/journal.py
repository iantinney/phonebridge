"""The per-task journal: the device operator's record of truth for work in flight (REC-01; PLAN §3.5 A7).

One directory per task, `<runs>/<task_id>/` (mode 700): `task.json`, written once (temp + fsync + rename) and
never holding lease tokens, and `journal.jsonl`, append-only, every line flushed and fsync'd before
`append()` returns. Lines are events numbered `seq` 1, 2, 3 ... with no gaps; the VM pulls them by `after_seq`
(OBS-01). A line torn by a crash mid-write is dropped when the journal is next opened, so the sequence goes on
from the last whole line.

Event fields: `seq`, `ts` (Mac wall clock, for people), `mono` (monotonic, for timeouts: VM time is the
authority for leases, ARC-04), then any of `state`, `step` + `outcome` (started | ok | ...), `screen_id`,
`frame_ref`, `reason`, `result`.

States. Run: submitted (→ queued, task protocol 2: waiting for the phone) → running → parked | resumed → done | failed |
stopped | backed_out. A post also walks
PICKED → STAGED → ARMED → POST_TAPPED → CONFIRMED | POST_UNCONFIRMED; POST_TAPPED is written before the tap.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

POST_STATES = ("PICKED", "STAGED", "ARMED", "POST_TAPPED", "CONFIRMED", "POST_UNCONFIRMED")
RUN_STATES = ("submitted", "queued", "running", "parked", "resumed", "done", "failed", "stopped", "backed_out")
FINAL = frozenset({"done", "failed", "stopped", "backed_out", "CONFIRMED", "POST_UNCONFIRMED"})
SECRET_KEYS = ("leases",)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Journal:
    def __init__(self, path: Path, clock=time.monotonic):
        self.path, self.clock = Path(path), clock
        self.file = self.path / "journal.jsonl"
        self._events = self._read()
        self.task = json.loads((self.path / "task.json").read_text())

    @classmethod
    def create(cls, runs: Path, task: dict, clock=time.monotonic) -> Journal:
        runs = Path(runs)
        runs.mkdir(parents=True, exist_ok=True, mode=0o700)
        d = runs / task["task_id"]
        d.mkdir(mode=0o700)  # FileExistsError: the task id is taken
        tmp = d / "task.json.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({k: v for k, v in task.items() if k not in SECRET_KEYS}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / "task.json")
        os.close(os.open(d / "journal.jsonl", os.O_WRONLY | os.O_CREAT, 0o600))
        _fsync_dir(d)
        _fsync_dir(runs)
        j = cls(d, clock)
        j.append(state="submitted", kind=task["kind"], identity=task["identity"])
        return j

    @classmethod
    def open(cls, runs: Path, task_id: str, clock=time.monotonic) -> Journal:
        d = Path(runs) / task_id
        if not (d / "task.json").exists():
            raise KeyError(f"no task {task_id!r} on this device operator")
        return cls(d, clock)

    @classmethod
    def all(cls, runs: Path, clock=time.monotonic) -> list[Journal]:
        runs = Path(runs)
        if not runs.is_dir():
            return []
        return [cls(d, clock) for d in sorted(runs.iterdir()) if (d / "task.json").exists()]

    def _read(self) -> list[dict]:
        if not self.file.exists():
            return []
        raw = self.file.read_bytes()
        whole = raw[: raw.rfind(b"\n") + 1]
        if len(whole) != len(raw):  # a torn last line: cut it so the next append starts clean
            with open(self.file, "r+b") as f:
                f.truncate(len(whole))
                f.flush()
                os.fsync(f.fileno())
        return [json.loads(line) for line in whole.decode().splitlines() if line.strip()]

    def append(self, **fields) -> dict:
        e = {"seq": (self._events[-1]["seq"] if self._events else 0) + 1, "ts": round(time.time(), 3),
             "mono": round(self.clock(), 3), **fields}
        line = (json.dumps(e, separators=(",", ":")) + "\n").encode()
        fd = os.open(self.file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
        self._events.append(e)
        return e

    def events(self, after_seq: int = 0, limit: int | None = None) -> list[dict]:
        out = [e for e in self._events if e["seq"] > after_seq]
        return out[:limit] if limit else out

    def last(self, pred) -> dict | None:
        return next((e for e in reversed(self._events) if pred(e)), None)

    @property
    def status(self) -> str:
        """The run status: the latest run state, or the post's final state; PICKED..POST_TAPPED read `running`."""
        e = self.last(lambda e: e.get("state") in RUN_STATES or e.get("state") in FINAL)
        s = e["state"] if e else "submitted"
        return "running" if s == "resumed" else s

    @property
    def post_state(self) -> str | None:
        e = self.last(lambda e: e.get("state") in POST_STATES)
        return e["state"] if e else None

    @property
    def done_steps(self) -> list[str]:
        return [e["step"] for e in self._events if e.get("step") and e.get("outcome") == "ok"]

    @property
    def final(self) -> bool:
        return self.status in FINAL
