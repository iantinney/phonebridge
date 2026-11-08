"""SCR-08/13, TST-01: the records (repo + live candidates) and the corpus of frames outside git.

Corpus layout (PLAN §9.2): `<platform>/<screen_id>/<date>-<appver>-<n>.{png,ocr.json,meta.json}`,
`confusers/`, `sequences/<flow>/<n>/`, `unknown/<platform>/<date>/`, `candidates/<platform>/<id>.yaml`
(records learned live, adopted into the repo by `screens promote --apply`), `manifest.yaml`. Frames are mode 600,
deduplicated by perceptual hash and capped per screen.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

from ..config import DATA_DIR
from .cascade import Frame, hamming
from .schema import Record, SchemaError, load_records

CORPUS = Path(os.environ.get("SCREENS_CORPUS", DATA_DIR / "screens" / "corpus")).expanduser()
REPO_RECORDS = (Path(__file__).parent / "records").resolve()  # the known-screen records and templates, shipped with the package
RECORD_DIRS = [Path(p).expanduser() for p in os.environ["SCREENS_RECORDS"].split(":")] \
    if os.environ.get("SCREENS_RECORDS") else [REPO_RECORDS]
SPECIAL = ("confusers", "unknown", "sequences", "candidates", "templates")
DEMOTE_FAILURES = 2
DEDUPE_BITS = 2


def phash(png: bytes | Image.Image, size: int = 8) -> int:
    """Perceptual (average) hash of the whole frame at 8x8 grey: dedupe key for the corpus."""
    im = png if isinstance(png, Image.Image) else Image.open(io.BytesIO(png))
    px = im.convert("L").resize((size, size), Image.BILINEAR).tobytes()
    mean = sum(px) / len(px)
    bits = 0
    for p in px:
        bits = (bits << 1) | (p > mean)
    return bits


def _secret(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb" if isinstance(data, bytes) else "w", opener=lambda p, f: os.open(p, f, 0o600)) as fh:
        fh.write(data)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


SYNC_TO = os.environ.get("SCREENS_SYNC_TO")  # e.g. "mac:~/.local/state/agent-reach/screens/corpus/candidates" (rsync target)


class Store:
    """Records from the repo dirs plus the corpus `candidates/` (learned live). Every store that recognises must
    see a learned save at once (G3, no promote step): `refresh()` re-reads the record dirs when any of them
    changed (the cascade calls it on every recognise; one stat per dir), and `save()` of a learned record also
    pushes the file to `sync_to` (the Mac daemon's candidates dir over the tunnel), best effort, in the
    background."""

    def __init__(self, records: list[Record] | None = None, records_dirs: list[Path] | None = None,
                 corpus: Path = CORPUS, cap_per_screen: int = 50, sync_to: str | None = SYNC_TO):
        self.corpus = Path(corpus)
        self.cap = cap_per_screen
        self.sync_to = sync_to
        self.records_dirs = [Path(d) for d in (records_dirs or [])]
        self.records: dict[str, Record] = {}
        self.paths: dict[str, Path] = {}
        self._extra: list[Record] = list(records or [])
        self._stamp = None
        self._load()

    def _dirs(self) -> list[Path]:
        return self.records_dirs + [self.corpus / "candidates"]

    def _signature(self) -> tuple:
        sig = []
        for d in self._dirs():
            try:
                sig.append((str(d), max((p.stat().st_mtime_ns for p in d.rglob("*.yaml")), default=0), sum(1 for _ in d.rglob("*.yaml"))))
            except OSError:
                sig.append((str(d), 0, 0))
        return tuple(sig)

    def _load(self) -> None:
        records, paths = {}, {}
        for p, r in load_records(self._dirs(), lenient=(self.corpus,)).items():
            if r.id in records and not str(p).startswith(str(self.corpus)):
                raise ValueError(f"duplicate record id {r.id!r}: {p} and {paths[r.id]}")
            if r.id in records and str(p).startswith(str(self.corpus)):
                continue  # a promoted (versioned) record wins over a learned candidate of the same id: a stale copy must not shadow it
            records[r.id], paths[r.id] = r, p
        for r in self._extra:
            records[r.id] = r
        for rid, r in self.records.items():  # keep in-memory-only records (tests, learners mid-run)
            if rid not in records and rid not in paths and rid not in self.paths:
                records[rid] = r
        self.records, self.paths = records, paths
        self._stamp = self._signature()

    def refresh(self) -> bool:
        """Re-read the record dirs if any file changed since the last load (a learned save from another process
        or host). Returns whether a reload happened."""
        if self._signature() == self._stamp:
            return False
        self._load()
        return True

    def for_platform(self, platform: str) -> list[Record]:
        return [r for r in self.records.values() if r.platform in (platform, "ios")]

    # --- frames -----------------------------------------------------------------------------
    def _dir(self, platform: str, screen_id: str) -> Path:
        return self.corpus / platform / screen_id

    def _hashes(self, d: Path) -> list[int]:
        out = []
        for m in d.glob("*.meta.json"):
            try:
                out.append(int(json.loads(m.read_text()).get("phash", "0"), 16))
            except (ValueError, json.JSONDecodeError):
                pass
        return out

    def _write(self, d: Path, stem: str, frame: Frame, meta: dict, h: int | None = None) -> Path:
        if frame.png is None:
            raise ValueError("cannot save a frame without pixels")
        png = _secret(d / f"{stem}.png", frame.png)
        _secret(d / f"{stem}.ocr.json", json.dumps(frame.ocr, ensure_ascii=False))
        meta = {**meta, "phash": f"{h if h is not None else phash(frame.png):016x}", "sha256": frame.sha,
                "ts": meta.get("ts") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "width": frame.width, "height": frame.height}
        _secret(d / f"{stem}.meta.json", json.dumps(meta, ensure_ascii=False, indent=1))
        return png

    def save_frame(self, platform: str, screen_id: str, frame: Frame, meta: dict) -> Path | None:
        """A recognised frame into `<platform>/<screen_id>/`; None when a near-duplicate exists (phash
        within DEDUPE_BITS) or the screen's cap is reached."""
        d = self._dir(platform, screen_id)
        h = phash(frame.png)
        existing = self._hashes(d)
        if any(hamming(h, e) <= DEDUPE_BITS for e in existing) or len(existing) >= self.cap:
            return None
        n = len(list(d.glob("*.png"))) + 1
        stem = f"{datetime.now(timezone.utc):%Y-%m-%d}-{meta.get('app_version') or 'v'}-{n:02d}"
        while (d / f"{stem}.png").exists():
            n += 1
            stem = f"{datetime.now(timezone.utc):%Y-%m-%d}-{meta.get('app_version') or 'v'}-{n:02d}"
        return self._write(d, stem, frame, meta, h)

    def queue_unknown(self, platform: str, frame: Frame, meta: dict) -> Path:
        """An unrecognised frame into `unknown/<platform>/<date>/` for `screens promote` to cluster."""
        now = datetime.now(timezone.utc)
        d = self.corpus / "unknown" / platform / f"{now:%Y-%m-%d}"
        h = phash(frame.png)
        for other in (self.corpus / "unknown" / platform).glob("*/*.meta.json") if (self.corpus / "unknown" / platform).exists() else []:
            try:
                if hamming(h, int(json.loads(other.read_text()).get("phash", "0"), 16)) <= DEDUPE_BITS:
                    return other.with_suffix("").with_suffix(".png")
            except (ValueError, OSError):
                pass
        stem = f"{now:%H%M%S}-{int(time.time() * 1000) % 1000:03d}"
        return self._write(d, stem, frame, {**meta, "kind": "unknown"}, h)

    def save_confuser(self, frame: Frame, meta: dict, name: str | None = None) -> Path | None:
        d = self.corpus / "confusers"
        h = phash(frame.png)
        if any(hamming(h, e) <= DEDUPE_BITS for e in self._hashes(d)):
            return None
        stem = name or f"{datetime.now(timezone.utc):%Y-%m-%d}-{len(list(d.glob('*.png'))) + 1:04d}"
        return self._write(d, stem, frame, meta, h)

    def fixtures(self):
        """(platform, screen_id, png path) for every labelled frame in the corpus."""
        for png in sorted(self.corpus.glob("*/*/*.png")):
            platform = png.parent.parent.name
            if platform in SPECIAL:
                continue
            yield platform, png.parent.name, png

    def confusers(self) -> list[Path]:
        return sorted((self.corpus / "confusers").glob("*.png"))

    # --- records ----------------------------------------------------------------------------
    def save(self, record: Record) -> Path:
        p = self.paths.get(record.id) or self.corpus / "candidates" / record.platform / f"{record.id}.yaml"
        self.records[record.id], self.paths[record.id] = record, p
        out = record.dump(p)
        self._stamp = self._signature()  # our own write is not a reason to reload
        if self.sync_to and str(p).startswith(str(self.corpus)):
            self.sync(p, record.platform)
        return out

    # the record keys every deployed schema since the first merge (main 96d25b8) accepts; newer keys (`avoid`,
    # `header`, …) stay on this side until the target's deploy carries the tolerant loader (14e4fb3), because a
    # pushed record must never take the other store down (fast-eyes, 00:5xZ)
    PORTABLE_KEYS = ("id", "platform", "bundle", "match", "kind", "seen_versions", "appearance", "locale", "mask", "chrome",
                     "targets", "checks", "next", "popups", "way_out", "timeout_s", "evidence", "status", "answer",
                     "stop_reason", "source", "matches", "successes", "failures", "note")

    def portable(self, record: Record) -> dict:
        d = record.to_dict()
        dropped = sorted(set(d) - set(self.PORTABLE_KEYS))
        d = {k: v for k, v in d.items() if k in self.PORTABLE_KEYS}
        if dropped:
            d["note"] = ((d.get("note") or "") + f" | pushed without {dropped} (target schema may predate them)").strip(" |")
        return d

    def sync(self, path: Path, platform: str) -> None:
        """Push one learned record to the other store's candidates dir (rsync over ssh), without waiting. The
        pushed copy carries only PORTABLE_KEYS (full record with SCREENS_SYNC_FULL=1)."""
        target = f"{self.sync_to.rstrip('/')}/{platform}/"
        log = self.corpus / "sync.log"
        try:
            src = path
            if not os.environ.get("SCREENS_SYNC_FULL"):
                outbox = self.corpus / "sync-outbox" / platform
                outbox.mkdir(parents=True, exist_ok=True, mode=0o700)
                src = outbox / path.name
                src.write_text(yaml.safe_dump(self.portable(self.records[path.stem] if path.stem in self.records else Record.load(path)),
                                              sort_keys=False, allow_unicode=True, width=110))
            with open(log, "a") as fh:
                fh.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {src.name} -> {target}\n")
                self._sync_runner(["rsync", "-a", "--rsync-path=mkdir -p " + target.split(":", 1)[-1] + " && rsync",
                                   str(src), target], fh)
        except (OSError, SchemaError) as e:
            with open(log, "a") as fh:
                fh.write(f"  failed to start: {e}\n")

    @staticmethod
    def _sync_runner(argv: list[str], log) -> None:
        import subprocess
        subprocess.Popen(argv, stdout=log, stderr=log, start_new_session=True)

    def add_candidate(self, record: Record, frame: Frame | None, meta: dict) -> Path:
        """A learned record (status candidate) with its exemplar frame as evidence."""
        record.status = "candidate"
        if frame is not None and frame.png is not None:
            if p := self.save_frame(record.platform, record.id, frame, meta):
                record.evidence.append(p.relative_to(self.corpus).as_posix())
        return self.save(record)

    def success(self, screen_id: str, ref: str) -> bool:
        """The record's response led to the expected next screen (Learned response counter, G3: saved at once,
        no confirmation step; nothing gates on `status`). Returns False (kept for callers)."""
        r = self.records[screen_id]
        r.successes += 1
        r.failures = 0
        if ref not in r.matches:
            r.matches.append(ref)
        if screen_id in self.paths or r.source in ("vlm", "promote"):
            self.save(r)
        return False

    record_match = success

    def failure(self, screen_id: str, reason: str = "") -> int:
        r = self.records[screen_id]
        r.failures += 1
        if reason:
            r.note = (r.note + " | " if r.note else "") + f"failure: {reason}"[:200]
        if screen_id in self.paths or r.source in ("vlm", "promote"):
            self.save(r)
        return r.failures

    def demote(self, screen_id: str, reason: str = "") -> None:
        """SCR-09: a record that failed its expectation twice: its counters say so; the slow model takes over."""
        r = self.records[screen_id]
        r.failures = max(r.failures, DEMOTE_FAILURES)
        r.matches = []
        r.note = f"demoted: {reason}" if reason else r.note
        if screen_id in self.paths:
            self.save(r)

    # --- reports ----------------------------------------------------------------------------
    def stats(self) -> dict:
        per: dict[str, int] = {}
        for platform, sid, _ in self.fixtures():
            per[f"{platform}/{sid}"] = per.get(f"{platform}/{sid}", 0) + 1
        return {"records": len(self.records),
                "confirmed": sum(1 for r in self.records.values() if r.status == "confirmed"),
                "candidates": sum(1 for r in self.records.values() if r.status == "candidate"),
                "frames": per, "confusers": len(self.confusers()),
                "unknown": len(list((self.corpus / "unknown").rglob("*.png"))),
                "sequences": len(list((self.corpus / "sequences").glob("*/*"))) if (self.corpus / "sequences").exists() else 0}

    def manifest_gaps(self) -> list[str]:
        """TST-01: every manifest entry resolves to a fixture under the corpus (or a documented xfail)."""
        p = self.corpus / "manifest.yaml"
        if not p.exists():
            return ["manifest.yaml missing"]
        m = yaml.safe_load(p.read_text()) or {}
        gaps = []
        for section in ("screens", "stops", "alerts", "faults", "irreversible"):
            for key, paths in (m.get(section) or {}).items():
                if isinstance(paths, str):  # a documented xfail or a replay reference
                    continue
                for rel in (paths or []):
                    if not (self.corpus / rel).exists():
                        gaps.append(f"{key}: {rel} missing")
        return gaps
