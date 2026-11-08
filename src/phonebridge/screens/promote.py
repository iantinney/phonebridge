"""`screens promote`: cluster the unknown queue (chrome dHash + OCR-token Jaccard) and draft a record per
cluster; report the learned records (success/failure counters) and, with --apply, adopt the ones that worked
from the corpus into the repo records dir so they are versioned. There is no confirmation step (G3): a
Learned response runs from the moment it is saved; this only moves files into git."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .cascade import Frame, dhash, hamming, norm
from .learn import draft_anchors, slug
from .schema import Record
from .store import REPO_RECORDS, Store

CHROME = (0, 0, 1, 0.15)
MAX_BITS = 6
MIN_JACCARD = 0.5


def tokens(frame: Frame) -> set[str]:
    return {norm(i["text"]) for i in frame.ocr if i.get("conf", 1.0) >= 0.6 and norm(i["text"])}


def cluster(frames: list[tuple[Path, Frame]]) -> list[list[tuple[Path, Frame]]]:
    sigs = [(p, f, dhash(f.png, CHROME) if f.png else 0, tokens(f)) for p, f in frames]
    groups: list[list] = []
    for p, f, h, t in sigs:
        for g in groups:
            _, gf, gh, gt = g[0]
            j = len(t & gt) / len(t | gt) if (t | gt) else 0.0
            if hamming(h, gh) <= MAX_BITS and j >= MIN_JACCARD:
                g.append((p, f, h, t))
                break
        else:
            groups.append([(p, f, h, t)])
    return [[(p, f) for p, f, _, _ in g] for g in groups]


def promote(store: Store, apply: bool = False, draft: bool = False) -> list[str]:
    out = []
    unknown = store.corpus / "unknown"
    frames = []
    for png in (sorted(unknown.rglob("*.png")) if unknown.exists() and draft else []):
        try:
            frames.append((png, Frame.from_files(png)))
        except (OSError, ValueError):
            continue
    for g in cluster(frames):
        platform = g[0][0].relative_to(unknown).parts[0]
        anchors = draft_anchors(g[0][1], store, platform)
        name = slug(" ".join(anchors[:2])) if anchors else f"cluster_{g[0][0].stem}"
        rid = f"{platform}.screen.{name}"
        if rid in store.records or len(anchors) < 2:
            out.append(f"cluster {rid}: {len(g)} frames, {'exists' if rid in store.records else 'too few anchors'}")
            continue
        rec = Record.from_dict({"id": rid, "platform": platform, "bundle": "*", "match": {"ocr_any": anchors, "at_least": min(2, len(anchors))},
                                "status": "candidate", "source": "promote",
                                "note": f"drafted from {len(g)} unknown frames: {', '.join(p.name for p, _ in g[:3])}"})
        rec.evidence = [str(p.relative_to(store.corpus)) for p, _ in g[:3]]  # the frames stay in unknown/: a draft is not a labelled fixture
        store.save(rec)
        out.append(f"drafted {rid} from {len(g)} unknown frames (anchors {anchors}); frames left in unknown/")
    for r in sorted(store.records.values(), key=lambda r: r.id):
        if r.source not in ("vlm", "promote"):
            continue
        out.append(f"learned {r.id}: {r.successes} successes, {r.failures} failures"
                   + (" — works" if r.successes and not r.failures else " — keeps failing: the slow model takes over" if r.failures else ""))
    moved = 0
    for r in list(store.records.values()):
        p = store.paths.get(r.id)
        if r.successes and not r.failures and p and str(p).startswith(str(store.corpus)):
            dest = REPO_RECORDS / r.platform / f"{r.id}.yaml"
            if apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), dest)
                store.paths[r.id] = dest
                moved += 1
            out.append(f"{'adopted' if apply else 'would adopt'} {r.id} -> {dest}")
    out.append(f"{moved} record(s) adopted into the repo" if apply else "dry run: pass --apply to adopt working learned records into the repo")
    return out
