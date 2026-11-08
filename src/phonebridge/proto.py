"""The VM<->Mac task-API protocol version, sent in run_submit / run_status / run_events replies and every heartbeat.

authhub keeps its own copy by value (packages/authhub/src/authhub/drivers/phone.py PROTO): it never imports
phonebridge (GPL boundary). Bump both together on an incompatible change to the run_* tools or the heartbeat.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

PROTO = 1
# The highest task protocol the daemon runs (a task's `v`, default 1): 1 = each task runs on submit; 2 = every task
# waits its turn in its phone's queue (one runner per phone) and `post` is a kind (lane one-loop). Additive: a v1
# task behaves exactly as before. apps/autopilot sends v2, for the accounts in operator.toml [phone] one_loop.
TASKS = 2  # sent as hello()["task_proto"]: the heartbeat already has a `tasks` field (device -> task)


@functools.cache
def build() -> str | None:
    """This checkout's short git sha (the Mac runs a git checkout, scripts/deploy-mac.sh); None if unknown."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 - no git, no repo: the sha is a nicety
        return None


def hello() -> dict:
    return {"proto": PROTO, "build": build(), "task_proto": TASKS}
