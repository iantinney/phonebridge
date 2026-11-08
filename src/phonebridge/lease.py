"""Lease guard: input calls hold the phone's `device:<alias>` lease, so one driver at a time.

Shells out to a lease CLI (`lease acquire <identity>` prints a token; `lease check|renew|release
<token>`) through `[lease] cmd` in the config: `phonebridge-local-lease` on one Mac, or a lease
service with the same protocol (the maintainers use authhub's, over ssh). With
no cmd it is a stub that allows everything (and logs that once per input session); the Bridge only
uses the stub when the config says `enforce = false`.
"""

from __future__ import annotations

import logging
import math
import os
import re
import socket
import subprocess
import threading
import time

log = logging.getLogger(__name__)


# `ssh host cmd args…` hands the args to the remote shell as one string, so every value must be inert there.
SAFE = re.compile(r"[A-Za-z0-9@:._/=+-]{1,200}")


def holder_id(*parts: str) -> str:
    """Join holder parts into one shell-safe word, e.g. phonebridge@mac:123/mcp:9f2c."""
    return "/".join(re.sub(r"[^A-Za-z0-9@:._=+-]", "_", p) for p in parts)


class LeaseError(RuntimeError):
    pass


class LeaseUnavailable(LeaseError):
    pass


class Leases:
    def __init__(self, cmd: list[str], ttl: int = 120, holder: str | None = None):
        self.cmd, self.ttl = cmd, ttl
        self.holder = holder or holder_id(f"phonebridge@{socket.gethostname()}:{os.getpid()}")
        self._held: dict[str, tuple[str, float]] = {}  # identity -> (token, monotonic time of acquire/renew)
        self._verified: dict[tuple[str, str], float] = {}  # (caller token, identity) -> monotonic time checked
        self._expires: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()  # ensure/release run in worker threads

    def ensure(self, identity: str | None, holder: str | None = None) -> None:
        """Hold `identity`: acquire (as `holder`), or renew past half the TTL. Raises LeaseError if someone
        else holds it."""
        if not identity:
            return
        with self._lock:
            if not self.cmd:
                if identity not in self._held:
                    log.warning("lease stub: not enforcing %s", identity)
                    self._held[identity] = ("", time.monotonic())
                return
            now = time.monotonic()
            held = self._held.get(identity)
            if held and now - held[1] < self.ttl / 2:
                self._held[identity] = held
                return
            if held:
                self.require_held(identity)
                if self._run("renew", held[0], "--ttl", str(self.ttl)).returncode:
                    self._held[identity] = (held[0], 0)
                    raise LeaseError(f"lease {identity} lost; release and re-observe before acquiring again")
                self._held[identity] = (held[0], now)
                self.require_held(identity)
                return
            r = self._run("acquire", identity, "--ttl", str(self.ttl), "--wait", "0", "--holder", holder or self.holder)
            if r.returncode or not r.stdout.split():  # never echo stdout: on success it is the token
                raise LeaseError(f"lease {identity}: {r.stderr.strip() or f'the lease command exited {r.returncode}'}")
            self._held[identity] = (r.stdout.split()[-1], now)

    def holds(self, identity: str) -> bool:
        return identity in self._held

    def vouched(self, token: str, identity: str, within_s: float) -> bool:
        """A live answer for this token is this recent (from verify() or a background alive() via vouch())."""
        ok = self._verified.get((token, identity))
        return (ok is not None and time.monotonic() - ok < within_s
                and time.monotonic() < self._expires.get((token, identity), 0))

    def vouch(self, token: str, identity: str) -> None:
        self._verified[(token, identity)] = time.monotonic()

    def forget(self, token: str, identity: str) -> None:
        self._verified.pop((token, identity), None)

    def verify(self, token: str, identity: str, cache_s: float = 10.0) -> float:
        """Check a caller's own lease (`<cmd> lease check <token> <identity>`); never acquires, renews or
        releases it. A positive answer is reused for `cache_s` (each check is an ssh round-trip)."""
        if not self.cmd:
            raise LeaseError("can't verify a lease without [lease] cmd in the config")
        now = time.monotonic()
        if self.vouched(token, identity, cache_s):
            return self._expires[(token, identity)] - now
        r = self._run("check", token, identity)
        if r.returncode == 255:
            raise LeaseUnavailable("lease service unreachable")
        if r.returncode:  # the token is never echoed
            raise LeaseError(f"the given lease is not a live lease on {identity}: "
                             f"{r.stderr.strip() or f'the lease command exited {r.returncode}'}")
        try:
            remaining = float(r.stdout.strip())
        except ValueError:
            raise LeaseError("lease check did not return a remaining lifetime") from None
        if not math.isfinite(remaining) or remaining <= 0:
            raise LeaseError("lease check returned an expired or invalid lifetime")
        self._expires[(token, identity)] = now + remaining
        self._verified[(token, identity)] = now
        left = now + remaining - time.monotonic()
        if left <= 0:
            raise LeaseError("lease expired during verification")
        return left

    def alive(self, token: str, identity: str) -> bool | None:
        """Is a caller's lease still live? None when the lease service can't be asked (ssh down, timeout): don't know."""
        try:
            self.verify(token, identity, cache_s=0)
            return True
        except (subprocess.TimeoutExpired, LeaseUnavailable):
            return None
        except LeaseError:
            return False

    def require_held(self, identity: str) -> None:
        if not self.cmd:
            return
        held = self._held.get(identity)
        if held is None or time.monotonic() >= held[1] + self.ttl:
            raise LeaseError(f"lease {identity} expired before input; release and re-observe")

    def require_live(self, token: str, identity: str) -> None:
        """Recheck the verified expiry immediately before input, after any pacing/guard awaits."""
        if time.monotonic() >= self._expires.get((token, identity), 0):
            raise LeaseError(f"verified lease on {identity} expired before input")

    def release(self, identity: str) -> None:
        with self._lock:
            held = self._held.pop(identity, None)
            if held and held[0] and self.cmd:
                self._run("release", held[0])

    def release_all(self) -> None:
        for identity in list(self._held):
            self.release(identity)

    def _run(self, verb: str, *args: str) -> subprocess.CompletedProcess:
        if not all(SAFE.fullmatch(a) for a in args):  # never echo the value: it may be a token, or an attack
            raise LeaseError(f"refusing to pass a malformed value to the lease command's {verb}")
        return subprocess.run([*self.cmd, "lease", verb, *args], capture_output=True, text=True, timeout=30)
