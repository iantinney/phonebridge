"""`phonebridge screens …`: the store's operator commands (append-only registration in phonebridge.cli)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..config import DATA_DIR, site
from .store import CORPUS, RECORD_DIRS, Store


def add_parser(sub) -> None:
    s = sub.add_parser("screens", help="the known-screens store: list, stats, gates, harvest, add, promote, replay, walk")
    ss = s.add_subparsers(dest="screens_cmd", required=True)
    ss.add_parser("list", help="every record: id, kind, status, fixtures")
    ss.add_parser("stats", help="records, frames per screen, confusers, unknown queue")
    g = ss.add_parser("gates", help="SCR-03 accuracy gates on the corpus; writes the report")
    g.add_argument("--out", type=Path, default=None, help="report path (default: <corpus>/gates-report.md)")
    h = ss.add_parser("harvest", help="evidence directories -> corpus fixtures, sequences, confusers, manifest")
    h.add_argument("evidence", type=Path, nargs="?", default=DATA_DIR / "evidence")
    h.add_argument("--calls", type=Path, default=None, help="a copy of the Mac's phonebridge calls.jsonl (actions by timestamp)")
    h.add_argument("--no-synth", action="store_true", help="skip the derived confusers")
    a = ss.add_parser("add", help="draft a candidate record from a frame (VLM names it; on-demand, SCR-10)")
    a.add_argument("frame", type=Path); a.add_argument("--platform", required=True); a.add_argument("--bundle", default="*")
    a.add_argument("--ocr", type=Path, default=None, help="cached .ocr.json (default: beside the frame)")
    p = ss.add_parser("promote", help="cluster unknown/ and candidates; report what is ready to confirm (SCR-13)")
    p.add_argument("--apply", action="store_true", help="adopt working learned records into the repo records dir")
    p.add_argument("--draft", action="store_true", help="also draft records from the unknown queue (unreviewed; off by default)")
    r = ss.add_parser("replay", help="run a flow on a recorded sequence (rehearse)")
    r.add_argument("flow", choices=["ig_reel", "ig_walk", "tt_walk"]); r.add_argument("--n", type=int, default=1)
    dr = ss.add_parser("drift", help="SCR-12: unknown-screen rate per day per app from the live evidence; alerts on a step")
    if site("NOTIFY_CMD", None):
        dr.add_argument("--notify", action="store_true", help="send alerts through the deployment's notify command")
    b = ss.add_parser("bench", help="E1: per-look timings of the daemon's stream `look` on whatever is on screen (no input by default); evidence dir written")
    b.add_argument("--device", required=True, help="the phone's alias or UDID")
    b.add_argument("--platform", default="instagram", choices=("instagram", "tiktok", "reddit"))
    b.add_argument("--url", default=os.environ.get("PHONEBRIDGE_URL", "http://127.0.0.1:8934/mcp"))
    b.add_argument("--lease", default=None, help="a lease token for device:<alias> held by the caller")
    b.add_argument("--minutes", type=float, default=2.0)
    b.add_argument("--flick-every", type=int, default=0, help="0 (default): zero input; N: a read-only flick every N looks")
    b.add_argument("--launch", action="store_true", help="launch the app first (input); default: look at what is on screen")
    b.add_argument("--out", default=None)
    lk = ss.add_parser("looks", help="E1 numbers (in-daemon ms per layer, frame age p50/p95/p99, stream counters) from an events.jsonl of looks")
    lk.add_argument("events")
    w = ss.add_parser("walk", help="live read-only walk on the phone (home → search → profile → settings → back), evidence dir written")
    w.add_argument("device"); w.add_argument("--platform", default="instagram", choices=["instagram", "tiktok"])
    w.add_argument("--url", default="http://127.0.0.1:8934/mcp", help="the phonebridge MCP server (phonebridge serve --http)")
    w.add_argument("--lease", default=os.environ.get(site("LEASE_ENV", "PHONEBRIDGE_LEASE")), help="a lease token for device:<alias> (default: from the environment)")
    w.add_argument("--out", type=Path, default=None, help="evidence dir (default: <data>/evidence/M1-screens/<ts>)")
    w.add_argument("--no-vlm", action="store_true", help="never call the VLM (cascade only; unknowns park)")
    if site("SESSION_CMD", None):
        w.add_argument("--identity", help="the account signed in on the phone: a finished walk is recorded as its read-only "
                                          "session (the deployment's session command)")


def run(a) -> int:
    if a.screens_cmd == "looks":  # needs no store: numbers from an events.jsonl
        from .live import looks_from, summarise
        print(json.dumps(summarise(looks_from(Path(a.events))), indent=1))
        return 0
    store = Store(records_dirs=RECORD_DIRS, corpus=CORPUS)
    if a.screens_cmd == "list":
        fx = {}
        for _, sid, _p in store.fixtures():
            fx[sid] = fx.get(sid, 0) + 1
        for r in sorted(store.records.values(), key=lambda r: r.id):
            print(f"{r.id:45s} {r.kind:6s} {r.status:9s} n={fx.get(r.id, 0):<3d} {r.source or ''}")
        return 0
    if a.screens_cmd == "stats":
        print(json.dumps(store.stats(), indent=1))
        if gaps := store.manifest_gaps():
            print("manifest gaps:", *gaps, sep="\n  ")
        return 0
    if a.screens_cmd == "gates":
        from . import gates
        rep = gates.run(store)
        text = gates.render(rep)
        out = a.out or (CORPUS / "gates-report.md")
        out.write_text(text)
        print(text)
        return 0 if gates.verdict(rep)[0] else 1
    if a.screens_cmd == "harvest":
        from .harvest import harvest
        harvest(a.evidence, store, calls=a.calls, synth=not a.no_synth)
        return 0
    if a.screens_cmd == "add":
        from .cascade import Frame
        from .learn import draft_anchors, slug
        from .schema import Record
        from .vlm import VLM
        f = Frame.from_files(a.frame, a.ocr)
        ans = VLM().resolve(f.png, f.ocr, sorted(r.id for r in store.for_platform(a.platform)), a.platform, "screens add (draft only)",
                            f.width, f.height)
        anchors = draft_anchors(f, store, a.platform)
        rec = Record.from_dict({"id": f"{a.platform}.screen.{slug(ans['screen_name'])}", "platform": a.platform, "bundle": a.bundle,
                                "match": {"ocr_any": anchors, "at_least": min(2, len(anchors))}, "status": "candidate", "source": "vlm",
                                "note": f"screens add: {ans.get('kind')} — {ans.get('reason')}"})
        print(store.add_candidate(rec, f, {"source": str(a.frame), "vlm": ans}))
        return 0
    if a.screens_cmd == "promote":
        from .promote import promote
        for line in promote(store, apply=a.apply, draft=a.draft):
            print(line)
        return 0
    if a.screens_cmd == "replay":
        import asyncio
        from .flows import FLOWS
        from .live import replay
        out = asyncio.run(replay(store, FLOWS[a.flow], CORPUS / "sequences" / a.flow / str(a.n)))
        print(out.status, out.screen, out.reason)
        return 0 if out.status in ("done", "rehearsed") else 1
    if a.screens_cmd == "drift":
        from . import drift
        rep = drift.report(drift.read_runs())
        print(drift.render(rep))
        if getattr(a, "notify", False) and (al := drift.alerts(rep)):
            import subprocess
            subprocess.run([*site("NOTIFY_CMD", None), "[screens drift] " + "; ".join(al)], check=False)
        return 0
    if a.screens_cmd in ("walk", "bench"):
        import asyncio
        from . import live
        if a.screens_cmd == "walk":
            out = asyncio.run(live.walk(store, a.device, a.platform, a.url, a.lease, a.out, use_vlm=not a.no_vlm,
                                        identity=getattr(a, "identity", None)))
        else:
            out = asyncio.run(live.bench(a.device, a.platform, a.url, a.lease, a.out, minutes=a.minutes,
                                         flick_every=a.flick_every, launch=a.launch))
        print(out)
        return 0
    return 2
