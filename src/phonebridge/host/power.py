"""A power assertion only while a task runs (DEV-04, DECISIONS Q1): `caffeinate -i -w <our pid>` from the first
task's start to the last task's end, never an always-on `caffeinate -d` keeping Ian's display awake, and `pmset`
left alone. `-i` stops idle sleep only: a shut lid or a flat battery still sleeps the Mac, and `watch` parks the
task. `-w` drops the assertion with this process if it dies.

The busy marker (`<state>/host/busy.json`, the running tasks) is what `scripts/deploy-mac.sh` checks until
orchestrator's task API lands, and what a restarted daemon restores its tasks from.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from ..config import STATE_DIR

MARKER = STATE_DIR / "host" / "busy.json"


class TaskPower:
    def __init__(self, marker: Path | None = MARKER, spawn=subprocess.Popen, caffeinate: str | None = shutil.which("caffeinate")):
        self.marker, self.spawn, self.caffeinate = marker, spawn, caffeinate
        self.tasks: dict[str, dict] = {}  # device alias -> what the task is (owner, identity, app, since)
        self._proc = None

    @property
    def held(self) -> bool:
        return bool(self.tasks)

    def begin(self, key: str, **info) -> None:
        self.tasks[key] = {"since": round(time.time(), 3), **self.tasks.get(key, {}), **info}
        if self.caffeinate and (self._proc is None or self._proc.poll() is not None):
            self._proc = self.spawn([self.caffeinate, "-i", "-w", str(os.getpid())],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._save()

    def end(self, key: str) -> None:
        self.tasks.pop(key, None)
        if not self.tasks and self._proc is not None:
            self._proc.terminate()
            self._proc.wait(5)
            self._proc = None
        self._save()

    def restore(self, live: Callable[[dict], bool]) -> list[str]:
        """Take over the tasks a previous daemon left in the marker whose lease `live(task)` still is; drop the rest."""
        try:
            left = json.loads(self.marker.read_text()).get("tasks", {})
        except (AttributeError, OSError, ValueError):
            left = {}
        for key, task in left.items():
            if live(task):
                self.begin(key, **task)
        self._save()
        return list(self.tasks)

    def _save(self) -> None:
        if self.marker is None:  # a one-off CLI run: the assertion, but the server's marker is the server's
            return
        if not self.tasks:
            self.marker.unlink(missing_ok=True)
            return
        self.marker.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = self.marker.with_suffix(".tmp")
        tmp.touch(mode=0o600)
        tmp.write_text(json.dumps({"pid": os.getpid(), "tasks": self.tasks}))
        tmp.replace(self.marker)
