"""SCR-03: the store accuracy gates on the corpus, with N per class and 95 % confidence intervals.
Zero false positives on every screen with an `irreversible` target (each with ≥ 5 examples, else
"insufficient", never "pass"); zero matches on ≥ 200 confusers; overall ≥ 98 %."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from .cascade import Cascade, Frame

MIN_IRREVERSIBLE = 5
MIN_CONFUSERS = 200
MIN_ACCURACY = 0.98


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def recognise_fixture(store, png: Path, cascade: Cascade | None = None) -> tuple[str | None, str | None]:
    """(expected screen id, recognised screen id) for one corpus frame; confusers expect None."""
    png = Path(png)
    platform = png.parent.parent.name
    want = None if platform == "confusers" or png.parent.name == "confusers" else png.parent.name
    cascade = cascade or Cascade(store)
    m = cascade.recognise(Frame.from_files(png), platform=None if want is None else platform)
    return want, m.screen_id


def run(store, cascade: Cascade | None = None) -> dict:
    cascade = cascade or Cascade(store)
    per: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": []})
    fp: dict[str, list] = defaultdict(list)
    overall_n = overall_ok = 0
    layers: dict[str, int] = defaultdict(int)
    ms = []
    for platform, sid, png in store.fixtures():
        m = cascade.recognise(Frame.from_files(png), platform=platform)
        ms.append(m.ms)
        layers[m.layer] += 1
        per[sid]["n"] += 1
        overall_n += 1
        if m.screen_id == sid:
            per[sid]["correct"] += 1
            overall_ok += 1
        else:
            per[sid]["wrong"].append((png.relative_to(store.corpus).as_posix(), m.screen_id))
            if m.screen_id:
                fp[m.screen_id].append(png.relative_to(store.corpus).as_posix())
    conf_matches = []
    confusers = store.confusers()
    for png in confusers:
        m = cascade.recognise(Frame.from_files(png), platform=None)
        ms.append(m.ms)
        if m.screen_id:
            conf_matches.append((png.relative_to(store.corpus).as_posix(), m.screen_id))
            fp[m.screen_id].append(png.relative_to(store.corpus).as_posix())
    irreversible = {}
    for r in store.records.values():
        if r.irreversible:
            n = per[r.id]["n"] if r.id in per else 0
            irreversible[r.id] = {"n": n, "false_positives": len(fp.get(r.id, [])), "sources": fp.get(r.id, []),
                                  "verdict": "insufficient" if n < MIN_IRREVERSIBLE else
                                  ("pass" if not fp.get(r.id) else "FAIL")}
    for sid, row in per.items():
        row["accuracy"] = round(row["correct"] / row["n"], 4) if row["n"] else 0.0
        row["ci95"] = wilson(row["correct"], row["n"])
    ms.sort()
    return {
        "overall": {"n": overall_n, "correct": overall_ok, "accuracy": round(overall_ok / overall_n, 4) if overall_n else 0.0,
                    "ci95": wilson(overall_ok, overall_n), "layers": dict(layers),
                    "p50_ms": ms[len(ms) // 2] if ms else 0, "p95_ms": ms[int(len(ms) * 0.95)] if ms else 0},
        "per_class": dict(per),
        "confusers": {"n": len(confusers), "matches": conf_matches},
        "irreversible": irreversible,
        "records": {"n": len(store.records), "confirmed": sum(1 for r in store.records.values() if r.status == "confirmed")},
    }


def verdict(rep: dict) -> tuple[bool, list[str]]:
    problems = []
    if rep["confusers"]["n"] < MIN_CONFUSERS:
        problems.append(f"confusers: {rep['confusers']['n']} < {MIN_CONFUSERS}")
    if rep["confusers"]["matches"]:
        problems.append(f"confusers matched: {rep['confusers']['matches'][:5]}")
    for sid, row in rep["irreversible"].items():
        if row["verdict"] != "pass":
            problems.append(f"irreversible {sid}: {row['verdict']} (n={row['n']}, fp={row['false_positives']})")
    if rep["overall"]["accuracy"] < MIN_ACCURACY:
        problems.append(f"overall accuracy {rep['overall']['accuracy']:.4f} < {MIN_ACCURACY}")
    return (not problems, problems)


def render(rep: dict) -> str:
    ok, problems = verdict(rep)
    o = rep["overall"]
    out = [f"# Store gates (SCR-03): {'PASS' if ok else 'FAIL'}", "",
           f"Overall: {o['correct']}/{o['n']} = {o['accuracy']:.4f} (95% CI {o['ci95'][0]:.4f}–{o['ci95'][1]:.4f}); "
           f"layers {o['layers']}; latency p50 {o['p50_ms']} ms, p95 {o['p95_ms']} ms",
           f"Confusers: {rep['confusers']['n']} (≥ {MIN_CONFUSERS}), matches: {len(rep['confusers']['matches'])}",
           f"Records: {rep['records']['n']} ({rep['records']['confirmed']} confirmed)", ""]
    if problems:
        out += ["Problems:"] + [f"- {p}" for p in problems] + [""]
    out += ["| screen | N | correct | accuracy | 95% CI |", "|---|---|---|---|---|"]
    for sid, row in sorted(rep["per_class"].items()):
        out.append(f"| {sid} | {row['n']} | {row['correct']} | {row['accuracy']:.3f} | {row['ci95'][0]:.3f}–{row['ci95'][1]:.3f} |")
    out += ["", "| irreversible screen | N | false positives | verdict |", "|---|---|---|---|"]
    for sid, row in sorted(rep["irreversible"].items()):
        out.append(f"| {sid} | {row['n']} | {row['false_positives']} | {row['verdict']} |")
    wrong = [(sid, w) for sid, row in rep["per_class"].items() for w in row["wrong"]]
    if wrong:
        out += ["", "Misrecognised fixtures:"] + [f"- {sid}: {p} → {got}" for sid, (p, got) in wrong]
    if rep["confusers"]["matches"]:
        out += ["", "Confusers that matched:"] + [f"- {p} → {got}" for p, got in rep["confusers"]["matches"]]
    return "\n".join(out) + "\n"
