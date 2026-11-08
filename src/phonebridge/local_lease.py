"""Local, exclusive device leases for a standalone Mac installation.

Implements the lease CLI protocol consumed by phonebridge.Leases. Configure
``[lease] cmd = ["phonebridge-local-lease"]``; keep ``enforce = true``.
This backend coordinates processes on one host, not multiple computers. It
does not replace a deployment's account policy, Focus grants, or authhub.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import time


class LeaseDenied(RuntimeError):
    pass


@contextmanager
def database(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS leases (identity TEXT PRIMARY KEY, "
                     "token_hash TEXT UNIQUE NOT NULL, holder TEXT NOT NULL, expires REAL NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def operate(path: Path, verb: str, value: str, *, identity: str | None = None,
            ttl: float = 120, holder: str = "local", now: float | None = None) -> str:
    """Return a token on acquire and remaining seconds on check/renew.

Only token hashes are persisted. Expired tokens never renew or release a new
holder's lease. The transaction serializes admission across CLI processes.
"""
    if not math.isfinite(ttl) or not 1 <= ttl <= 3600:
        raise LeaseDenied("TTL must be between 1 and 3600 seconds")
    if not value or len(value) > 200 or not holder or len(holder) > 200:
        raise LeaseDenied("invalid lease argument")
    now = time.time() if now is None else now
    with database(path) as conn:
        conn.execute("DELETE FROM leases WHERE expires <= ?", (now,))
        if verb == "acquire":
            if conn.execute("SELECT 1 FROM leases WHERE identity = ?", (value,)).fetchone():
                raise LeaseDenied("device is already leased by another session")
            token = secrets.token_urlsafe(32)
            digest = hashlib.sha256(token.encode()).hexdigest()
            conn.execute("INSERT INTO leases VALUES (?, ?, ?, ?)", (value, digest, holder, now + ttl))
            return token
        digest = hashlib.sha256(value.encode()).hexdigest()
        row = conn.execute("SELECT identity, expires FROM leases WHERE token_hash = ?", (digest,)).fetchone()
        if verb == "release":
            conn.execute("DELETE FROM leases WHERE token_hash = ?", (digest,))
            return "released"
        if not row or (verb == "check" and row[0] != identity):
            raise LeaseDenied("lease is expired, unknown, or belongs to another device")
        if verb == "check":
            return str(row[1] - now)
        if verb == "renew":
            conn.execute("UPDATE leases SET expires = ? WHERE token_hash = ?", (now + ttl, digest))
            return str(ttl)
        raise LeaseDenied("unknown lease operation")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("group", choices=["lease"])
    commands = p.add_subparsers(dest="verb", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("value", metavar="identity")
    acquire.add_argument("--ttl", type=float, default=120)
    acquire.add_argument("--wait", type=float, choices=[0], default=0,
                         help="only immediate admission is supported")
    acquire.add_argument("--holder", default=f"local:{os.getpid()}")
    check = commands.add_parser("check")
    check.add_argument("value", metavar="token")
    check.add_argument("identity")
    renew = commands.add_parser("renew")
    renew.add_argument("value", metavar="token")
    renew.add_argument("--ttl", type=float, default=120)
    release = commands.add_parser("release")
    release.add_argument("value", metavar="token")
    args = p.parse_args(argv)
    state = Path(os.environ.get("PHONEBRIDGE_STATE", "~/.local/state/phonebridge")).expanduser()
    try:
        result = operate(state / "local-leases.sqlite", args.verb, args.value,
                         identity=getattr(args, "identity", None),
                         ttl=getattr(args, "ttl", 120), holder=getattr(args, "holder", "local"))
    except (LeaseDenied, OSError, sqlite3.Error) as exc:
        # Tokens never enter errors, logs, or the database in plaintext.
        print(f"local lease: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
