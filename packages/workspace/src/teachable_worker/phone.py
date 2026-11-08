"""The execution adapter (owned by L2, tw-phone). This default is the honest "no route": nothing is qualified, so
every account reads not usable and a publish job blocks on one founder action instead of pretending to run.

    readiness(account) -> {usable, reason, checked, evidence}
    submit(job, plan, approved_sha, run_ref) -> None    idempotent per run_ref; refuses a sha that is not the job's
    poll(run_ref) -> {state: running|done|failed|needs|unknown, needs, result: {url, readback, verified_at, simulated}}
"""

from __future__ import annotations

from .store import now

NO_ROUTE = "No phone route is qualified yet: connect and requalify a phone for this account"


def readiness(account: dict) -> dict:
    return {"usable": False, "reason": NO_ROUTE, "checked": now(), "evidence": "phone.py default adapter"}


def submit(job: dict, plan: dict, approved_sha: str, run_ref: str) -> None:
    raise RuntimeError(NO_ROUTE)


def poll(run_ref: str) -> dict:
    return {"state": "unknown", "needs": None, "result": None}
