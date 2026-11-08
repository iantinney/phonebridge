"""The shared store, one SQLite file ($TW_DB, default ~/.local/state/teachable-worker/tw.sqlite) and one file tree
($TW_DATA, default ~/.local/state/teachable-worker/data). Shared tables are below; lane-private tables come from
`extend()` at module import. Ids are `<prefix>_<12 hex>`, times ISO-8601 UTC, JSON columns end in `_json`.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, track TEXT NOT NULL CHECK (track IN ('agency', 'dating')),
  cohort TEXT NOT NULL CHECK (cohort IN ('external', 'founder', 'staff', 'test', 'friend')),
  source TEXT, referred_by TEXT REFERENCES orgs(id), created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS people (
  id TEXT PRIMARY KEY, org_id TEXT NOT NULL REFERENCES orgs(id), name TEXT NOT NULL, email TEXT,
  role TEXT NOT NULL DEFAULT 'owner' CHECK (role IN ('owner', 'member', 'founder')),
  cohort TEXT NOT NULL CHECK (cohort IN ('external', 'founder', 'staff', 'test', 'friend')),
  source TEXT, referred_by TEXT REFERENCES people(id), token TEXT NOT NULL UNIQUE, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS accounts (
  id TEXT PRIMARY KEY, org_id TEXT NOT NULL REFERENCES orgs(id), platform TEXT NOT NULL, handle TEXT NOT NULL,
  identity TEXT, authority TEXT NOT NULL DEFAULT 'claimed' CHECK (authority IN ('none', 'claimed', 'verified')),
  authority_evidence TEXT, access TEXT NOT NULL DEFAULT 'unknown' CHECK (access IN ('unknown', 'usable', 'broken')),
  access_checked TEXT, access_evidence TEXT, instructions TEXT, created TEXT NOT NULL,
  UNIQUE (org_id, platform, handle));
CREATE TABLE IF NOT EXISTS permissions (
  id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
  mode TEXT NOT NULL CHECK (mode IN ('prepare', 'approve_each', 'delegated')),
  granted_by TEXT NOT NULL REFERENCES people(id), evidence TEXT NOT NULL, created TEXT NOT NULL, revoked TEXT);
CREATE TABLE IF NOT EXISTS routines (
  id TEXT PRIMARY KEY, org_id TEXT NOT NULL REFERENCES orgs(id), account_id TEXT REFERENCES accounts(id),
  name TEXT NOT NULL,
  provenance TEXT NOT NULL CHECK (provenance IN ('customer_demo', 'operator_demo', 'founder_config', 'template')),
  recording_id TEXT, summary TEXT, steps_json TEXT NOT NULL DEFAULT '[]', version INTEGER NOT NULL DEFAULT 1,
  confirmed_at TEXT, confirmed_by TEXT, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS corrections (
  id TEXT PRIMARY KEY, routine_id TEXT REFERENCES routines(id), job_id TEXT REFERENCES jobs(id), text TEXT NOT NULL,
  by TEXT, created TEXT NOT NULL, applied_version INTEGER);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, org_id TEXT NOT NULL REFERENCES orgs(id), account_id TEXT NOT NULL REFERENCES accounts(id),
  routine_id TEXT REFERENCES routines(id), kind TEXT NOT NULL CHECK (kind IN ('prepare', 'publish')),
  input_json TEXT NOT NULL, origin TEXT NOT NULL CHECK (origin IN ('customer', 'founder', 'agent', 'test')),
  requested_by TEXT, requested_at TEXT NOT NULL, due_at TEXT, status TEXT NOT NULL, resume_status TEXT,
  needs TEXT, needs_from TEXT, content_sha TEXT, approved_sha TEXT, approved_by TEXT, approved_at TEXT,
  run_ref TEXT, attempts INTEGER NOT NULL DEFAULT 0, accepted_at TEXT,
  acceptance_method TEXT CHECK (acceptance_method IN ('customer_click', 'customer_message', 'objective_criteria')),
  accepted_by TEXT, updated TEXT NOT NULL, rev INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), kind TEXT NOT NULL, output_json TEXT NOT NULL,
  url TEXT, readback_json TEXT, verified_at TEXT, simulated INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS payments (
  id TEXT PRIMARY KEY, org_id TEXT REFERENCES orgs(id), person_id TEXT REFERENCES people(id),
  product TEXT NOT NULL CHECK (product IN ('pilot', 'subscription', 'review_pack')), amount_cents INTEGER NOT NULL,
  currency TEXT NOT NULL DEFAULT 'USD', status TEXT NOT NULL CHECK (status IN ('settled', 'pending', 'refunded')),
  processor TEXT NOT NULL, processor_ref TEXT, evidence TEXT NOT NULL, at TEXT NOT NULL,
  UNIQUE (processor, processor_ref));
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, at TEXT NOT NULL, name TEXT NOT NULL, org_id TEXT, person_id TEXT, account_id TEXT,
  job_id TEXT, routine_id TEXT, actor TEXT NOT NULL CHECK (actor IN ('customer', 'founder', 'agent', 'system')),
  data_json TEXT NOT NULL DEFAULT '{}');
CREATE INDEX IF NOT EXISTS events_name ON events (name, at);
CREATE INDEX IF NOT EXISTS jobs_org ON jobs (org_id, status);
"""
_EXTRA: list[str] = []
COLUMNS = [("jobs", "rev", "INTEGER NOT NULL DEFAULT 0"),  # added after v1: ALTERed into older files once
           ("jobs", "relayed_by", "TEXT"),  # a request the operator entered for the customer: never auto-approved
           ("people", "verified_at", "TEXT"),  # a distinct human we know (operator onboarding, payment, email)
           ("people", "verified_evidence", "TEXT"),
           ("payments", "refunded_cents", "INTEGER NOT NULL DEFAULT 0"),  # total refunded (record_refund)
           ("frames", "items_json", "TEXT")]  # the text the bridge read on that frame (deleted with the frame)
_migrated: set[str] = set()


def db_path() -> Path:
    return Path(os.environ.get("TW_DB") or "~/.local/state/teachable-worker/tw.sqlite").expanduser()


def data_dir() -> Path:
    return Path(os.environ.get("TW_DATA") or "~/.local/state/teachable-worker/data").expanduser()


def _private(path: Path, mode: int) -> None:
    """Tighten one of this app's own files or directories to `mode`: that path only, never its contents, and never
    through a symlink (chmod would follow it to a file this app doesn't own)."""
    with contextlib.suppress(FileNotFoundError):
        if not path.is_symlink() and path.stat().st_mode & 0o777 != mode:
            path.chmod(mode)


def private_dir(path: Path) -> Path:
    """One of this app's own directories (it holds sign-in tokens or phone screenshots): created 0700, an existing
    one tightened to 0700. Only for paths this app owns; a parent it merely writes into is left alone."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _private(path, 0o700)
    return path


def write_private(path: Path, data: bytes | str, exclusive: bool = False) -> None:
    """Write one of this app's private files (a token, a screenshot, its text): created 0600 from the start, so no
    umask window; an existing one is tightened. `exclusive`: the file must be new."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(data.encode() if isinstance(data, str) else data)


def _make_private(db: Path, created_parent: bool) -> None:
    """Once per database per process, on first use: the state this app owns becomes readable by its user only, so a new
    install starts private and an existing one is tightened. SQLite gives -wal/-shm the database file's mode, so the
    database is tightened first; the existing ones are tightened too. The data tree is tightened at its own levels only
    (data/, frames/, frames/<device>/, media/, recordings/ and the files directly in frames/<device>/ and media/), never
    by walking into paths it doesn't own."""
    if created_parent or db.parent == Path("~/.local/state/teachable-worker").expanduser():  # never a user's own dir
        _private(db.parent, 0o700)
    for f in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm"), data_dir().parent / "admin.token"):
        _private(f, 0o600)
    data = data_dir()
    created_data = not data.exists()
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    if created_data or data == Path("~/.local/state/teachable-worker/data").expanduser():  # never a user's own dir
        _private(data, 0o700)
    devices = [d for d in (data / "frames").iterdir() if d.is_dir() and not d.is_symlink()] if (data / "frames").is_dir() else []
    for d in [data / "frames", data / "media", data / "recordings", *devices]:
        _private(d, 0o700)
    for d in [data / "media", *devices]:
        for f in (d.iterdir() if d.is_dir() else ()):
            if f.is_file() and not f.is_symlink():
                _private(f, 0o600)


def extend(sql: str) -> None:
    """A lane's private tables (CREATE ... IF NOT EXISTS), applied on every connection."""
    if sql not in _EXTRA:
        _EXTRA.append(sql)


_local = threading.local()


@contextlib.contextmanager
def tx():
    """One write transaction per thread; a nested tx() (or query/one/event inside one) joins the outer one, so a
    flow's reads and writes commit or roll back together and never wait on their own lock."""
    if getattr(_local, "con", None) is not None:
        yield _local.con
        return
    p = db_path()
    created_parent = not p.parent.exists()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    con = sqlite3.connect(p, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        if str(p) not in _migrated:
            _make_private(p, created_parent)  # before the WAL files exist, so they take the database's 0600
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        con.executescript(SCHEMA + "".join(_EXTRA))
        if str(p) not in _migrated:
            for table, col, decl in COLUMNS:
                have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
                if have and col not in have:  # a lane's table absent here (its module not imported): nothing to alter
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            _recheck(con, "routines", "operator_demo")
            _migrated.add(str(p))
        con.execute("BEGIN IMMEDIATE")
        _local.con = con
        yield con
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        _local.con = None
        con.close()


def _recheck(con: sqlite3.Connection, table: str, needle: str) -> None:
    """A CHECK constraint widened after v1 (SQLite cannot ALTER one): rebuild `table` from SCHEMA, keeping rows."""
    old = con.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    if not old or needle in old[0]:
        return
    create = next(c for c in SCHEMA.split(";") if f"CREATE TABLE IF NOT EXISTS {table} (" in c)
    cols = ", ".join(r[1] for r in con.execute(f"PRAGMA table_info({table})"))
    con.execute("PRAGMA foreign_keys = OFF")
    con.executescript(f"BEGIN; {create.replace(f'IF NOT EXISTS {table} (', f'{table}_new (')}; "
                      f"INSERT INTO {table}_new ({cols}) SELECT {cols} FROM {table}; DROP TABLE {table}; "
                      f"ALTER TABLE {table}_new RENAME TO {table}; COMMIT;")
    con.execute("PRAGMA foreign_keys = ON")


def query(sql: str, args: tuple = ()) -> list[dict]:
    with tx() as con:
        return [dict(r) for r in con.execute(sql, args).fetchall()]


def one(sql: str, args: tuple = ()) -> dict | None:
    rows = query(sql, args)
    return rows[0] if rows else None


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def event(name: str, *, actor: str, org_id=None, person_id=None, account_id=None, job_id=None, routine_id=None,
          con: sqlite3.Connection | None = None, **data) -> int:
    """Append one metric event. Never put private conversation text in `data`."""
    row = (now(), name, org_id, person_id, account_id, job_id, routine_id, actor, json.dumps(data))
    sql = ("INSERT INTO events (at, name, org_id, person_id, account_id, job_id, routine_id, actor, data_json) "
           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
    if con is not None:
        return con.execute(sql, row).lastrowid
    with tx() as c:
        return c.execute(sql, row).lastrowid
