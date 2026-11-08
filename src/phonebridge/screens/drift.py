"""SCR-12 drift: the unknown-screen rate per day per app (A/B layouts change without a version bump) from the
live evidence dirs, plus the canary: `phonebridge screens walk <device> --no-vlm` is the daily read-only
rehearse over known screens (run when the phone is free and leased, and first on any new app version). A
rate step (today ≥ 2× yesterday with ≥ 3 unknowns) or a canary that stops on an unknown screen raises one
alert naming the screen."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .live import EVIDENCE

SUMMARY = re.compile(r"^- summary: (\{.*\})\s*$", re.M)


def read_runs(evidence: Path = EVIDENCE) -> list[dict]:
    runs = []
    for p in sorted(Path(evidence).glob("*/RESULT.md")):
        m = SUMMARY.search(p.read_text())
        if not m:
            continue
        try:
            s = json.loads(m.group(1))
        except ValueError:
            continue
        day = p.parent.name[:8]
        platform = next((l.split("platform: ")[1].split(",")[0] for l in p.read_text().splitlines() if "platform: " in l), "?")
        unknown_screens = []
        ev = p.parent / "events.jsonl"
        if ev.exists():
            for line in ev.read_text().splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("event") == "unknown":
                    unknown_screens.append(e.get("context", "")[:80])
        runs.append({"day": day, "platform": platform, "dir": p.parent.name, **s, "unknown_screens": unknown_screens})
    return runs


def report(runs: list[dict]) -> dict:
    days: dict[tuple[str, str], dict] = defaultdict(lambda: {"runs": 0, "recognitions": 0, "unknown": 0, "vlm_calls": 0, "screens": []})
    for r in runs:
        if r.get("frames", 0) == 0 and r.get("status") == "escalate":
            continue  # a USB-loss park, not a look at the app
        d = days[(r["day"], r["platform"])]
        d["runs"] += 1
        d["unknown"] += r.get("unknown_events", 0)
        d["vlm_calls"] += r.get("vlm_calls", 0)
        d["recognitions"] += max(1, r.get("steps", 0) + r.get("unknown_events", 0))
        d["screens"] += r.get("unknown_screens", [])
    out = {}
    for (day, platform), d in sorted(days.items()):
        out.setdefault(platform, {})[day] = {**d, "rate": round(d["unknown"] / max(1, d["recognitions"]), 3)}
    return out


def alerts(rep: dict) -> list[str]:
    out = []
    for platform, by_day in rep.items():
        days = sorted(by_day)
        for prev, cur in zip(days, days[1:]):
            a, b = by_day[prev], by_day[cur]
            if b["unknown"] >= 3 and b["rate"] >= 2 * max(a["rate"], 0.01):
                out.append(f"{platform}: unknown rate {a['rate']} ({prev}) -> {b['rate']} ({cur}); screens: {b['screens'][:3]}")
    return out


def render(rep: dict) -> str:
    lines = ["| app | day | runs | recognitions | unknown | rate | VLM calls |", "|---|---|---|---|---|---|---|"]
    for platform, by_day in rep.items():
        for day, d in by_day.items():
            lines.append(f"| {platform} | {day} | {d['runs']} | {d['recognitions']} | {d['unknown']} | {d['rate']} | {d['vlm_calls']} |")
    a = alerts(rep)
    return "\n".join(lines) + ("\n\nAlerts:\n" + "\n".join(f"- {x}" for x in a) if a else "\n\nNo drift alert.") + \
        "\n\nCanary: `phonebridge screens walk <device> --no-vlm` (read-only, known screens only; run daily when the phone is free, and first on a new app version).\n"
