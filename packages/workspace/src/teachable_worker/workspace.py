"""The desktop phone workspace (INTERFACES.md §Workspace): devices, frames, and tasks a person launches, follows,
messages, stops or takes over.

Reads come from the phone bridge (`$TW_BRIDGE_URL`, the Mac's phonebridge reverse-forwarded to the VM) over its MCP
endpoint with plain JSON-RPC (stdlib; tw-web's verified call shape): device list, presence, the live `frame`/`screen`,
and `look` (a frame with the text read from that same frame).

The live view is the operator looking at their own phone: lease-free, outside the task queue, never stored, ~1 s cache,
one capture in flight; each viewing burst is audited (`live_view` event with who). Its pixel age is unknown (the bridge's
frame tool reports none), so it is labelled with fetch time and cache age, never called fresh. Stored evidence is
different: a frame is kept only when a task holding a work session reads it (the lane's `session` hook: the device
lease, plus Focus in our deployment). Inputs happen only inside lane executors, which record each one as a step with
the pace the bridge actually drew. One task runs per device at a time, and only while its session holds the device
lease; a busy device leaves it queued with the holder named. Evidence frames live under `$TW_DATA/frames/` (kept
TW_FRAME_DAYS, default 7), operator-only.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .errors import Conflict, NotFound, Refused, TooMany, lane
from .store import data_dir, extend, new_id, now, one, private_dir, query, tx, write_private

extend("""
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, device_id TEXT NOT NULL, kind TEXT NOT NULL, input_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'needs', 'done', 'failed', 'stopped', 'handed_off')),
  needs TEXT, origin TEXT NOT NULL CHECK (origin IN ('operator', 'agent', 'test')), requested_by TEXT, created TEXT NOT NULL,
  updated TEXT NOT NULL, started TEXT, finished TEXT, stop_requested INTEGER NOT NULL DEFAULT 0, handoff_note TEXT,
  execution_json TEXT, result_json TEXT, error TEXT, session_open INTEGER NOT NULL DEFAULT 0, not_before REAL);
CREATE INDEX IF NOT EXISTS tasks_device ON tasks (device_id, state, created);
CREATE TABLE IF NOT EXISTS task_steps (
  task_id TEXT NOT NULL REFERENCES tasks(id), idx INTEGER NOT NULL, at TEXT NOT NULL, kind TEXT NOT NULL, text TEXT,
  frame_id TEXT, pace_ms INTEGER, data_json TEXT NOT NULL DEFAULT '{}', PRIMARY KEY (task_id, idx));
CREATE TABLE IF NOT EXISTS task_messages (
  task_id TEXT NOT NULL REFERENCES tasks(id), idx INTEGER NOT NULL, at TEXT NOT NULL, by TEXT, text TEXT NOT NULL,
  PRIMARY KEY (task_id, idx));
CREATE TABLE IF NOT EXISTS frames (
  id TEXT PRIMARY KEY, device_id TEXT NOT NULL, task_id TEXT, captured_at TEXT NOT NULL, path TEXT NOT NULL,
  content_type TEXT NOT NULL, width INTEGER, height INTEGER, source TEXT NOT NULL, simulated INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS frames_device ON frames (device_id, captured_at);
""")

OPEN = ("queued", "running", "needs")
FINAL = ("done", "failed", "stopped", "handed_off")
STEP_KINDS = ("observe", "tap", "swipe", "type", "open_app", "wait", "note", "result", "message")
OBSERVE_EVERY_S = 5
BUSY_RETRY_S = 30
NEEDS_TIMEOUT_S = 180  # a task waiting on the person ends `stopped` after this long, freeing the phone and the queue
BEAT_STALE_S = 180
LANES = ("missions", "teaching", "coaching", "phone_coach", "phone")  # modules that may export TASKS


class Stopped(Exception):
    """The person stopped the task or took the phone; the executor ends at once, sending no further input."""


class Busy(Exception):
    """The executor's session could not take the device (another holder): the task goes back to the queue."""


class Unavailable(Refused):
    status = 503


# ---- the bridge (reads only) ----------------------------------------------------------------------------------

class Bridge:
    """Reads from the phone bridge. Replaceable in tests (`workspace.bridge = Fake()`)."""

    simulated = False

    READS = ("devices", "frame", "screen", "look", "ocr", "presence", "foreground", "lock_state", "understand")

    def __init__(self, url: str | None = None):
        self.url = (url or os.environ.get("TW_BRIDGE_URL") or "http://127.0.0.1:8934").rstrip("/").removesuffix("/mcp")
        self.token = os.environ.get("TW_BRIDGE_TOKEN") or None  # the Mac daemon's bearer token, when it runs with one
        self._session, self._lock = None, threading.Lock()

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _rpc(self, body: dict) -> dict | None:
        req = urllib.request.Request(f"{self.url}/mcp", json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self._auth(),
            **({"Mcp-Session-Id": self._session} if self._session else {})})
        with urllib.request.urlopen(req, timeout=30) as r:
            self._session, raw = r.headers.get("Mcp-Session-Id") or self._session, r.read().decode()
        if not raw.strip():
            return None
        if not raw.lstrip().startswith(("event:", "data:")):
            return json.loads(raw)
        for ln in reversed(raw.splitlines()):  # the JSON-RPC answer; skip empty data lines and pings
            if ln.startswith("data:") and ln[5:].strip():
                msg = json.loads(ln[5:].strip())
                if "result" in msg or "error" in msg:
                    return msg
        return None

    def tool(self, name: str, **args) -> dict:
        """One read-only MCP tool call (one session kept; on any transport error the session is renewed and the read
        retried once). Inputs never go through here: a retry after the bridge acted would double an input."""
        if name not in self.READS:
            raise ValueError(f"{name} is not a read; inputs go through the lane executors' leased path")
        with self._lock:
            for attempt in (0, 1):
                try:
                    if not self._session:
                        self._rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                            "protocolVersion": "2025-03-26", "capabilities": {},
                            "clientInfo": {"name": "teachable-worker", "version": "1"}}})
                        self._rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
                    out = self._rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                     "params": {"name": name, "arguments": args}})
                except (urllib.error.URLError, OSError, ValueError):  # HTTPError is a URLError
                    self._session = None
                    if attempt:
                        raise
                    continue
                if out and "error" in out:
                    raise RuntimeError(f"phonebridge {name}: {out['error'].get('message')}")
                result = (out or {}).get("result") or {}
                if result.get("isError"):
                    raise RuntimeError(f"phonebridge {name}: " +
                                       " ".join(c.get("text", "") for c in result.get("content", []))[:300])
                return result
        raise RuntimeError(f"phonebridge {name}: no answer")

    @staticmethod
    def _value(result: dict):
        sc = result.get("structuredContent")
        if sc is not None:
            return sc["result"] if set(sc) == {"result"} else sc
        text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        return json.loads(text) if text else None

    def devices(self) -> list[dict]:
        v = self._value(self.tool("devices"))
        return [d if isinstance(d, dict) else {"id": d} for d in (v or [])]

    def presence(self, device: str) -> dict:
        q = urllib.parse.urlencode({"device": device})  # the daemon's cached presence (age_s says how old)
        req = urllib.request.Request(f"{self.url}/presence?{q}", headers=self._auth())
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())

    def frame(self, device: str) -> dict:
        """The current screen for the live view: a still (`screen`, ~1 s on the Mac's DVT path), else the stream's
        newest frame (`frame`, ~10 s when it has to open the stream). {jpeg, source, seq, age_ms}."""
        for name, source in (("screen", "still"), ("frame", "stream")):
            try:
                result = self.tool(name, device=device, format="jpeg")
            except (RuntimeError, OSError, ValueError):
                if name == "screen":
                    continue
                raise
            img = next((c for c in result.get("content", []) if c.get("type") == "image"), None)
            if img:  # the frame tool reports no capture time: pixel age is unknown, never assumed 0
                return {"jpeg": base64.b64decode(img["data"]), "source": source, "seq": None, "age_ms": None}
        raise RuntimeError("phonebridge answered without an image")

    PLATFORMS = {"com.burbn.instagram": "instagram", "com.zhiliaoapp.musically": "tiktok", "com.atebits.Tweetie2": "x"}

    def observe(self, device: str, platform: str | None = None, accurate: bool = False) -> dict:
        """One coherent look: the frame and the text read from that same frame. The platform (given, else from the
        foreground app) scopes the daemon's screen match, so another app's records don't cancel a confident one.
        `accurate`: the bridge's own ocr(fast=False, with_image=True), Vision's accurate pass on a still it takes
        itself, returned with that very image; for verifying a final read (it names no screen and has no pixel age)."""
        if accurate:
            r = self._value(self.tool("ocr", device=device, fast=False, with_image=True))
            if not isinstance(r, dict) or not r.get("image_jpeg"):
                raise RuntimeError("phonebridge's accurate ocr returned no image for its text")
            return {"jpeg": base64.b64decode(r["image_jpeg"]), "width": r.get("width"), "height": r.get("height"),
                    "items": r.get("items") or [], "age_ms": None, "frame_seq": None, "screen": None,
                    "foreground": None, "accurate": True}
        if platform is None:
            try:
                platform = self.PLATFORMS.get(self.presence(device).get("foreground") or "")
            except Exception:  # noqa: BLE001 - unscoped look, as before
                platform = None
        args = {"device": device, "with_frame": True, **({"platform": platform} if platform else {})}
        r = self._value(self.tool("look", **args))
        encoded = r.get("frame_jpeg") or r.get("image_jpeg")
        if not encoded:
            raise RuntimeError("phonebridge's look returned no pixels for its text")
        return {"jpeg": base64.b64decode(encoded), "width": r.get("width"), "height": r.get("height"),
                "items": r.get("items") or [], "age_ms": r.get("frame_age_ms"), "frame_seq": r.get("frame_seq"),
                "screen": r.get("screen") or r.get("screen_id"), "foreground": r.get("foreground") or r.get("bundle")}


bridge = Bridge()


def _beat() -> dict | None:
    """The Mac's last heartbeat as the VM received it (authhub deadman, deployment-specific; None when absent)."""
    d = os.environ.get("TW_HEARTBEAT_DIR") or os.environ.get("AUTHHUB_DEADMAN_DIR")
    if not d:
        return None
    d = Path(d).expanduser()
    try:
        with open(d / "beats.jsonl", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            b = json.loads(f.read().splitlines()[-1])
    except (OSError, IndexError, ValueError):
        return None
    t = b.get("rx", b.get("ts", 0))
    return {"at": datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds"), "age_s": round(time.time() - t)}


_leases: dict = {"at": 0.0, "rows": None}


def _refresh_leases() -> None:
    cmd = os.environ.get("TW_LEASE_STATUS_CMD")  # e.g. "authhub status --json" in our deployment; unset elsewhere
    if not cmd:
        _leases.update(rows=None, refreshing=False)
        return
    try:
        r = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=20)
        _leases["rows"] = {x["id"]: x for x in json.loads(r.stdout)["identities"]} if r.returncode == 0 else None
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        _leases["rows"] = None
    _leases["refreshing"] = False


def _lease(device: str) -> dict | None:
    """Who holds `device:<id>` in authhub (deployment-specific; None when authhub isn't here or not read yet).
    Refreshed in the background every ~10 s so the device list never waits on it."""
    if time.time() - _leases["at"] > 10 and not _leases.get("refreshing"):
        _leases.update(at=time.time(), refreshing=True)
        threading.Thread(target=_refresh_leases, daemon=True).start()
    row = (_leases["rows"] or {}).get(f"device:{device}")
    if row is None:
        return None
    return {"holder": row.get("holder"), "since": row.get("since") or row.get("acquired"), "mine": False}


_conn: dict[str, tuple[float, dict]] = {}
CONN_CACHE_S = 5


def connection(device: str) -> dict:
    """{state, reason, heartbeat_at, heartbeat_age_s, checked_at}, cached a few seconds (the list is polled)."""
    key = (id(bridge), device)  # a different bridge (reconfigured, or a test's fake) never reads another's cache
    hit = _conn.get(key)
    if hit and time.time() - hit[0] < CONN_CACHE_S:
        return dict(hit[1])
    c = {**_connection(device), "checked_at": now()}
    _conn[key] = (time.time(), c)
    return dict(c)


def _connection(device: str) -> dict:
    """{state, reason, heartbeat_at, heartbeat_age_s}: one plain sentence saying what to do when not connected."""
    beat = _beat()
    out = {"heartbeat_at": beat and beat["at"], "heartbeat_age_s": beat and beat["age_s"]}
    if beat and beat["age_s"] > BEAT_STALE_S:
        return {**out, "state": "asleep", "reason": f"The Mac is asleep or offline (last heartbeat "
                f"{beat['age_s'] // 60} min ago): open it and keep it awake.", "presence": None}
    try:
        p = bridge.presence(device)
    except Exception as e:  # noqa: BLE001 - unknown is never connected
        return {**out, "state": "unknown", "reason": f"The phone bridge isn't answering ({str(e)[:120]}).",
                "presence": None}
    if p.get("attached") is False:
        state, reason = "unplugged", "Plug the iPhone into the Mac with a data cable."
    elif p.get("lockdown") is False and not p.get("tunnel"):  # known untrusted, not just unread this moment
        state, reason = "untrusted", "Unlock the iPhone and tap Trust for this Mac."
    elif p.get("locked") is True:
        state, reason = "locked", "Unlock the iPhone."
    elif p.get("attached") and (p.get("tunnel") or p.get("lockdown")):
        state, reason = "connected", ""
    else:  # fields not read yet (the phone is busy): say so, never guess an instruction
        state, reason = "unknown", "The phone's state isn't readable this moment; it usually clears in a few seconds."
    return {**out, "state": state, "reason": reason, "presence": p}


def _frame_view(f: dict | None) -> dict | None:
    if not f:
        return None
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(f["captured_at"])).total_seconds()
    return {"url": f"/api/frames/{f['id']}", "frame_id": f["id"], "captured_at": f["captured_at"], "age_s": round(age, 1),
            "source": f["source"], "simulated": bool(f["simulated"])}


def latest_frame(device: str) -> dict | None:
    return one("SELECT * FROM frames WHERE device_id = ? ORDER BY captured_at DESC, rowid DESC LIMIT 1", (device,))


_listed: dict = {"t": 0.0, "v": None}


def devices() -> list[dict]:
    try:
        if time.time() - _listed["t"] > 30 or _listed["v"] is None or _listed.get("b") != id(bridge):
            _listed.update(t=time.time(), v=bridge.devices(), b=id(bridge))
        listed = _listed["v"]
    except Exception:  # noqa: BLE001 - the bridge down: still show what we know from tasks/frames
        listed = [{"id": d["device_id"]} for d in query("SELECT DISTINCT device_id FROM frames UNION "
                                                         "SELECT DISTINCT device_id FROM tasks")]
    out = []
    for d in listed:
        did = d.get("id") or d.get("alias") or d.get("device")
        c = connection(did)
        p = c.pop("presence") or {}
        t = one("SELECT id FROM tasks WHERE device_id = ? AND state IN ('running', 'needs') ORDER BY started DESC "
                "LIMIT 1", (did,))
        live = _live.get(did)
        out.append({"id": did, "name": d.get("name") or p.get("name") or did, "model": d.get("model"),
                    "ios": p.get("ios") or d.get("ios"), "connection": c, "foreground_app": p.get("foreground"),
                    "lease": _lease(did), "focus": None,
                    "frame": {"url": f"/api/devices/{did}/frame", "available": c["state"] in ("connected", "locked"),
                              "reason": c["reason"], "fetched_at": live and live["fetched_at"],
                              "cache_age_s": live and round(time.time() - live["t"], 1), "pixel_age_s": None,
                              "source": live and live["source"]},
                    "last_evidence": _frame_view(latest_frame(did)), "task_id": t and t["id"]})
    return out


def look_now(device: str, viewer: str | None) -> dict:
    """The operator's read-only coherent snapshot, stored as evidence at once: no task, no queue, no lease."""
    f = latest_frame(device)
    if f and f["task_id"] is None and \
            (datetime.now(timezone.utc) - datetime.fromisoformat(f["captured_at"])).total_seconds() < OBSERVE_EVERY_S:
        raise TooMany("This phone was looked at a moment ago; use that look or try again in a few seconds.")
    try:
        o = bridge.observe(device)
    except Exception as e:  # noqa: BLE001
        raise Unavailable(f"The phone's screen can't be read right now ({str(e)[:120]}).")
    saved = save_frame(device, o["jpeg"], None, "bridge look", o.get("width"), o.get("height"),
                       simulated=getattr(bridge, "simulated", False), items=o.get("items"))
    from .store import event
    event("phone_look", actor="founder", person_id=viewer, device=device, frame_id=saved["id"])
    return {"frame_id": saved["id"], "frame_url": f"/api/frames/{saved['id']}", "observed_at": saved["captured_at"],
            "pixel_age_ms": o.get("age_ms"), "foreground": o.get("foreground"), "screen": o.get("screen"),
            "items": [{k: i.get(k) for k in ("text", "x", "y", "w", "h")} for i in o.get("items") or []]}


def image_size(data: bytes) -> tuple[int | None, int | None]:
    """(width, height) from a PNG's IHDR or a JPEG's SOF marker; (None, None) if neither is readable."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    i = 2
    while data[:2] == b"\xff\xd8" and i + 9 < len(data):
        if data[i] != 0xFF:
            break
        marker, size = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # a start-of-frame segment
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + size
    return None, None


def save_frame(device: str, data: bytes, task_id: str | None, source: str, width=None, height=None,
               simulated: bool = False, captured_at: str | None = None, items: list | None = None) -> dict:
    """An immutable evidence frame (jpeg or png by its bytes), with the text the bridge read on that same frame
    (`items`: [{text, x, y, w, h, conf?}] in its pixels) in the same row, so purge and forget_task delete both."""
    png = data.startswith(b"\x89PNG")
    if width is None or height is None:  # the bridge's look reply carries no size: read it from the image
        width, height = image_size(data)
    fid = new_id("frm")
    path = private_dir(private_dir(data_dir() / "frames") / device) / f"{fid}.{'png' if png else 'jpg'}"
    write_private(path, data, exclusive=True)  # a phone screenshot
    with tx() as con:
        con.execute("INSERT INTO frames (id, device_id, task_id, captured_at, path, content_type, width, height, source, "
                    "simulated, items_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (fid, device, task_id, captured_at or now(), str(path), "image/png" if png else "image/jpeg",
                     width, height, source, int(simulated), json.dumps(items) if items is not None else None))
    return one("SELECT * FROM frames WHERE id = ?", (fid,))


def purge_frames(days: float | None = None) -> int:
    """Evidence frames older than TW_FRAME_DAYS (default 7) are deleted with their files."""
    days = float(os.environ.get("TW_FRAME_DAYS", "7")) if days is None else days
    cutoff = datetime.fromtimestamp(time.time() - days * 86400, timezone.utc).isoformat(timespec="seconds")
    old = query("SELECT id, path FROM frames WHERE captured_at < ?", (cutoff,))
    for f in old:
        Path(f["path"]).unlink(missing_ok=True)
    with tx() as con:
        con.execute("DELETE FROM frames WHERE captured_at < ?", (cutoff,))
    return len(old)


# ---- the live view: a current frame, only inside a work session -----------------------------------------------

LIVE_CACHE_S = 1.0
_live: dict[str, dict] = {}
_live_locks: dict[str, threading.Lock] = {}


VIEW_BURST_S = 120  # a gap longer than this starts a new audited viewing burst
_viewed: dict[str, float] = {}


def session_task(device: str) -> dict | None:
    """The workspace task holding a work session (device lease) on this phone, if any."""
    return one("SELECT * FROM tasks WHERE device_id = ? AND state IN ('running', 'needs') AND session_open = 1 "
               "ORDER BY started DESC LIMIT 1", (device,))


def live_frame(device: str, viewer: str | None) -> dict:
    """The current screen for the operator, cached ~1 s per device, one capture in flight (viewers share it); no
    lease, no queue, never stored. Pixel age unknown: `fetched_at` is when the VM got it."""
    t = time.time()
    if t - _viewed.get(device, 0) > VIEW_BURST_S:
        from .store import event
        event("live_view", actor="founder", person_id=viewer, device=device)
    _viewed[device] = t
    with _live_locks.setdefault(device, threading.Lock()):
        f = _live.get(device)
        if f and time.time() - f["t"] < LIVE_CACHE_S:
            return f
        try:
            got = bridge.frame(device)
        except Exception as e:  # noqa: BLE001
            raise Unavailable(f"The phone's screen can't be read right now ({str(e)[:120]}).")
        _live[device] = f = {**got, "t": time.time(), "fetched_at": now()}
        return f


# ---- tasks ----------------------------------------------------------------------------------------------------

class LeaseSession:
    """The standalone work session: the device lease only. `$TW_LEASE_CMD` is the lease command group, e.g.
    `phonebridge-local-lease lease` or `authhub lease`: `acquire device:<id> --holder H --ttl S` prints the token,
    `renew TOKEN --ttl S`, `release TOKEN`. No Focus (that is our deployment's adapter). A denied lease raises Busy; a
    usage/config error fails the task; a failed renew closes the session and stops the task (the lease is lost)."""

    TTL = 300

    def __init__(self, ctx, command: str | None = None):
        self.cmd = shlex.split(command or os.environ["TW_LEASE_CMD"])
        self.ctx, self.token, self._stop = ctx, None, threading.Event()

    def _run(self, *args):
        return subprocess.run([*self.cmd, *args], capture_output=True, text=True, timeout=30)

    def __enter__(self):
        r = self._run("acquire", f"device:{self.ctx.device['id']}", "--holder", f"teachable-worker {self.ctx.task['id']}",
                      "--ttl", str(self.TTL))
        why = (r.stderr or r.stdout).strip()[-200:]
        if r.returncode and ("usage:" in why or r.returncode == 2):
            raise RuntimeError(f"the lease command is misconfigured (TW_LEASE_CMD): {why}")
        if r.returncode:
            raise Busy(why or "the device lease is held")
        self.token = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
        if not self.token:
            raise RuntimeError("the lease command printed no token")
        threading.Thread(target=self._renew, daemon=True).start()
        return self

    def _renew(self):
        while not self._stop.wait(self.TTL / 3):
            try:
                r = self._run("renew", self.token, "--ttl", str(self.TTL))
                why = (r.stderr or r.stdout).strip()[-200:] if r.returncode else None
            except (OSError, subprocess.TimeoutExpired) as exc:
                why = f"lease renewal failed ({type(exc).__name__})"  # never include token-bearing argv
            if why is not None:  # lost: the phone may be someone else's now; no further reads or inputs
                self.token = None
                lose_session(self.ctx.task["id"], why)
                return

    def __exit__(self, *exc):
        self._stop.set()
        if self.token:
            try:
                result = self._run("release", self.token)
                if result.returncode:
                    logging.getLogger(__name__).warning("Device lease release failed; waiting for its TTL to expire")
            except (OSError, subprocess.TimeoutExpired) as error:
                # TimeoutExpired includes the command's token in str(error).
                # Cleanup must neither leak it nor replace the executor's outcome.
                logging.getLogger(__name__).warning("Device lease release failed (%s); waiting for its TTL to expire",
                                                    type(error).__name__)
            finally:
                self.token = None
        return False


def lose_session(task_id: str, why: str) -> None:
    """The work session's lease was lost: close it and stop the task (the executor ends at its next check)."""
    _set(task_id, session_open=0, stop_requested=1, error=f"device lease lost: {why}"[:500])
    _forward_stop(get_task(task_id))


def session_hook():
    """The work-session context manager `session(ctx)`: our deployment's (tw-phone's missions.session: authhub device
    lease + Focus work window, closing only what it opened), else the standalone LeaseSession when TW_LEASE_CMD is set."""
    from . import standalone
    if standalone.enabled():
        return standalone.session
    mod = lane("missions")
    hook = getattr(mod, "session", None) if mod else None
    return hook or (LeaseSession if os.environ.get("TW_LEASE_CMD") else None)


NO_SESSION = ("no work session is configured on this server: install tw-phone's missions.session (device lease + "
              "Focus) or set TW_LEASE_CMD to a lease command")


def executors() -> dict:
    out = {}
    for name in LANES:
        mod = lane(name)
        out.update(getattr(mod, "TASKS", {}) if mod else {})
    return out


PREREQ = {"account_check": "tw-phone's account_check executor (teachable_worker/missions.py) isn't installed on this "
                           "server",
          "feed_scout": "tw-phone's feed_scout executor (teachable_worker/missions.py) isn't installed on this server",
          "taught_run": "tw-teach's taught_run executor (teaching.py TASKS) isn't installed on this server",
          "coach_observe": "tw-coaching's coach_observe executor (phone_coach.py TASKS) isn't installed on this server"}


def kinds() -> list[dict]:
    have = executors()
    want = ("account_check", "feed_scout", "taught_run", "coach_observe")
    out = []
    for k in dict.fromkeys([*want, *have]):
        ex = have.get(k)
        ok, why = ex.available() if ex else (False, PREREQ.get(k, "no executor is installed for it on this server"))
        out.append({"kind": k, "available": bool(ok), "reason": why, "read_only": getattr(ex, "read_only", None),
                    "input": getattr(ex, "input", {})})
    return out


def _task_view(t: dict, after: int | None = None) -> dict:
    out = {**{k: v for k, v in t.items() if not k.endswith("_json")}, "input": json.loads(t["input_json"]),
           "execution": json.loads(t["execution_json"] or "null"), "result": json.loads(t["result_json"] or "null"),
           "stop_requested": bool(t["stop_requested"])}
    if after is not None:
        out["steps"] = [{**{k: s[k] for k in ("idx", "at", "kind", "text", "frame_id", "pace_ms")},
                         "frame_url": f"/api/frames/{s['frame_id']}" if s["frame_id"] else None,
                         "data": json.loads(s["data_json"])}
                        for s in query("SELECT * FROM task_steps WHERE task_id = ? AND idx > ? ORDER BY idx",
                                       (t["id"], after))]
        out["messages"] = query("SELECT idx, at, by, text FROM task_messages WHERE task_id = ? ORDER BY idx", (t["id"],))
    return out


def get_task(task_id: str) -> dict:
    t = one("SELECT * FROM tasks WHERE id = ?", (task_id,))
    if not t:
        raise NotFound("That task does not exist.")
    return t


def create_task(device_id: str, kind: str, inputs: dict, requested_by: str | None, origin: str = "operator") -> dict:
    ex = executors().get(kind)
    ok, why = ex.available() if ex else (False, PREREQ.get(kind, "no executor is installed for it on this server"))
    if not ok:
        raise Refused(f"{kind} is not available yet: {why}")
    if origin not in ("operator", "agent", "test"):
        raise Refused("origin is operator, agent or test.")
    tid = new_id("tsk")
    with tx() as con:
        con.execute("INSERT INTO tasks (id, device_id, kind, input_json, state, origin, requested_by, created, updated) "
                    "VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
                    (tid, device_id, kind, json.dumps(inputs or {}), origin, requested_by, now(), now()))
    _ensure_runner(device_id)
    return get_task(tid)


def _set(task_id: str, **cols) -> None:
    cols["updated"] = now()
    with tx() as con:
        con.execute(f"UPDATE tasks SET {', '.join(f'{k} = ?' for k in cols)} WHERE id = ?", (*cols.values(), task_id))


def _step(task_id: str, kind: str, text: str, frame_id=None, pace_ms=None, **data) -> int:
    if kind not in STEP_KINDS:
        raise ValueError(f"unknown step kind {kind!r}")
    with tx() as con:
        idx = con.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM task_steps WHERE task_id = ?", (task_id,)).fetchone()[0]
        con.execute("INSERT INTO task_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (task_id, idx, now(), kind, text, frame_id, pace_ms, json.dumps(data)))
        con.execute("UPDATE tasks SET updated = ? WHERE id = ?", (now(), task_id))
    return idx


def stop(task_id: str, by: str | None) -> dict:
    t = get_task(task_id)
    if t["state"] in FINAL:
        raise Conflict("This task already ended.")
    if t["state"] == "queued":
        _set(task_id, state="stopped", stop_requested=1, finished=now())
    else:
        _set(task_id, stop_requested=1)
        _forward_stop(t)
    return get_task(task_id)


def _forward_stop(t: dict) -> None:
    """An executor that handed work to the Mac forwards the stop there (its `stop(task)` parks/cancels the Mac task
    so the Mac checks before its next input); ctx.check() alone only stops between VM-side steps."""
    ex = executors().get(t["kind"])
    if hasattr(ex, "stop"):
        threading.Thread(target=lambda: _quiet(ex.stop, get_task(t["id"])), daemon=True).start()


def _quiet(fn, *a):
    try:
        fn(*a)
    except Exception:  # noqa: BLE001 - best effort; the VM-side check still ends the task
        pass


def handoff(task_id: str, note: str, by: str | None) -> dict:
    t = get_task(task_id)
    if t["state"] in FINAL:
        raise Conflict("This task already ended.")
    _set(task_id, stop_requested=1, handoff_note=note or "taken over by the person",
         **({"state": "handed_off", "finished": now()} if t["state"] == "queued" else {}))
    if t["state"] != "queued":
        _forward_stop(t)
    return get_task(task_id)


def forget_task(task_id: str, by: str | None) -> dict:
    """Delete what a task read and said (steps, messages, input, result, needs, error) and its evidence frames with
    their files; the bare row stays (id, kind, device, state, times) as an audit that something ran and was deleted.
    Only the person who asked for the task (`by` = its requested_by; None = the admin console). Refused while running."""
    t = get_task(task_id)
    if t["requested_by"] != by:
        raise NotFound("That task does not exist.")
    if t["state"] in OPEN:
        raise Conflict("Stop the task before deleting what it read.")
    frames = query("SELECT path FROM frames WHERE task_id = ?", (task_id,))
    for f in frames:
        Path(f["path"]).unlink(missing_ok=True)
    with tx() as con:
        con.execute("DELETE FROM task_steps WHERE task_id = ?", (task_id,))
        con.execute("DELETE FROM task_messages WHERE task_id = ?", (task_id,))
        con.execute("DELETE FROM frames WHERE task_id = ?", (task_id,))
        con.execute("UPDATE tasks SET input_json = '{}', result_json = NULL, needs = NULL, error = NULL, "
                    "handoff_note = NULL, execution_json = NULL, updated = ? WHERE id = ?", (now(), task_id))
    left = query("SELECT COUNT(*) AS n FROM frames WHERE task_id = ?", (task_id,))[0]["n"] + \
        query("SELECT COUNT(*) AS n FROM task_steps WHERE task_id = ?", (task_id,))[0]["n"]
    if left:
        raise RuntimeError(f"forget_task left {left} rows for {task_id}")
    return {"task_id": task_id, "frames_deleted": len(frames)}


def message(task_id: str, text: str, by: str | None) -> dict:
    t = get_task(task_id)
    if not text or not text.strip():
        raise Refused("Write the message.")
    if t["state"] in FINAL:
        raise Conflict("This task already ended; start a new one.")
    with tx() as con:  # the message and its timeline step together: a waiting task can't act between them
        idx = con.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM task_messages WHERE task_id = ?",
                          (task_id,)).fetchone()[0]
        con.execute("INSERT INTO task_messages VALUES (?, ?, ?, ?, ?)", (task_id, idx, now(), by, text.strip()))
        _step(task_id, "message", text.strip(), by=by)
    return get_task(task_id)


class Ctx:
    """What an executor gets: the task, the device, a read-only look, step recording, messages, stop."""

    def __init__(self, task: dict):
        self.task, self.device = task, {"id": task["device_id"]}
        self.input = json.loads(task["input_json"])

    def stopped(self) -> bool:
        return bool(get_task(self.task["id"])["stop_requested"])

    def check(self) -> None:
        if self.stopped():
            raise Stopped()

    def step(self, kind: str, text: str, frame_id=None, pace_ms=None, **data) -> int:
        self.check()
        return _step(self.task["id"], kind, text, frame_id, pace_ms, **data)

    def session(self):
        """The lane's work session (device lease + Focus) around the task's phone work; marks the task as holding
        the phone so the live view opens, and closes it after."""
        hook = session_hook()
        if hook is None:
            raise RuntimeError(NO_SESSION)
        ctx = self

        class _S:
            def __enter__(s):
                s.cm = hook(ctx)
                s.cm.__enter__()
                _set(ctx.task["id"], session_open=1)
                return s

            def __exit__(s, *exc):
                _set(ctx.task["id"], session_open=0)
                return s.cm.__exit__(*exc)
        return _S()

    def observe(self, note: str | None = None, accurate: bool = False) -> dict:
        """A coherent look (frame + its text), stored as evidence. `accurate=True`: a new still read by Vision's
        accurate pass, stored as its own frame with exactly that image and its text (for verifying a final read;
        screen is None, never borrowed from another frame). Read-only; no retry."""
        self.check()
        o = bridge.observe(self.device["id"], accurate=True) if accurate else bridge.observe(self.device["id"])
        f = save_frame(self.device["id"], o["jpeg"], self.task["id"], "bridge ocr (accurate)" if accurate else
                       "bridge look", o.get("width"), o.get("height"), simulated=getattr(bridge, "simulated", False),
                       items=o.get("items"))
        _step(self.task["id"], "observe", note or "Looked at the phone", f["id"],
              items=len(o.get("items") or []), age_ms=o.get("age_ms"), screen=o.get("screen"),
              **({"accurate": True} if accurate else {}))
        return {**o, "frame_id": f["id"], "frame_url": f"/api/frames/{f['id']}", "captured_at": f["captured_at"]}

    def attach_frame(self, data: bytes, observed_at: str, source: str, note: str | None = None,
                     items: list | None = None, **data_) -> dict:
        """Evidence a claim was actually read from (a Mac run's journal frame): stored as-is with its own time, and
        the text read on it when the caller has it (Vision OCR from the Mac journal)."""
        f = save_frame(self.device["id"], data, self.task["id"], source,
                       simulated=getattr(bridge, "simulated", False), captured_at=observed_at, items=items)
        _step(self.task["id"], "observe", note or f"Frame from {source}", f["id"], **data_)
        return {"frame_id": f["id"], "frame_url": f"/api/frames/{f['id']}", "captured_at": f["captured_at"]}

    def messages(self, after: int = -1) -> list[dict]:
        return query("SELECT idx, at, by, text FROM task_messages WHERE task_id = ? AND idx > ? ORDER BY idx",
                     (self.task["id"], after))

    def needs(self, text: str, poll_s: float = 1.0, timeout_s: float = NEEDS_TIMEOUT_S) -> str:
        """Wait for the person: returns their next message; raises Stopped on stop, hand-off, or no answer within
        timeout_s (the phone is not held indefinitely). An executor that can, leaves its session before asking."""
        seen = max([m["idx"] for m in self.messages()] or [-1])
        _set(self.task["id"], state="needs", needs=text)
        until = time.time() + timeout_s
        while True:
            self.check()
            if time.time() > until:
                _set(self.task["id"], stop_requested=1, error=f"no answer within {int(timeout_s // 60)} min; released")
                raise Stopped()
            new = self.messages(seen)
            if new:
                _set(self.task["id"], state="running", needs=None)
                return new[0]["text"]
            time.sleep(poll_s)


_runners: dict[str, threading.Thread] = {}
_runners_lock = threading.Lock()


def _ensure_runner(device_id: str) -> None:
    """One runner thread per device: tasks on a phone run one at a time, in order."""
    with _runners_lock:
        t = _runners.get(device_id)
        if t and t.is_alive():
            return
        t = threading.Thread(target=_run_device, args=(device_id,), daemon=True, name=f"tw-device-{device_id}")
        _runners[device_id] = t
        t.start()


def _run_device(device_id: str) -> None:
    while True:
        t = one("SELECT * FROM tasks WHERE device_id = ? AND state = 'queued' AND (not_before IS NULL OR not_before <= ?) "
                "ORDER BY created, rowid LIMIT 1", (device_id, time.time()))
        if not t and one("SELECT 1 FROM tasks WHERE device_id = ? AND state = 'queued'", (device_id,)):
            time.sleep(1)  # only tasks waiting for a busy device: wait for their retry time
            continue
        if not t:
            with _runners_lock:  # re-check under the lock so a task queued meanwhile is not stranded
                if not one("SELECT 1 FROM tasks WHERE device_id = ? AND state = 'queued'", (device_id,)):
                    _runners.pop(device_id, None)
                    return
            continue
        run_task(t)


def run_task(t: dict) -> dict:
    ex = executors().get(t["kind"])
    execution = {"route": getattr(ex, "route", "bridge"), "executor": type(ex).__module__ + "." + type(ex).__name__,
                 "simulated": bool(getattr(ex, "simulated", False) or getattr(bridge, "simulated", False))}
    _set(t["id"], state="running", started=now(), execution_json=json.dumps(execution))
    try:
        if ex is None:
            raise RuntimeError(f"no executor for {t['kind']}")
        result = ex.run(Ctx(get_task(t["id"])))
        with tx():  # the Done step and the done state are one write: no poll sees the step with the task still running
            _step(t["id"], "result", "Done")
            _set(t["id"], state="done", finished=now(), result_json=json.dumps(result, default=str), needs=None,
                 session_open=0)
    except Busy as e:  # the device is someone else's right now: wait in the queue, say who holds it
        _set(t["id"], state="queued", needs=f"Device busy: {e}", not_before=time.time() + BUSY_RETRY_S,
             session_open=0)
    except Stopped:
        cur = get_task(t["id"])
        _set(t["id"], state="handed_off" if cur["handoff_note"] else "stopped", finished=now(), needs=None,
             session_open=0)
    except Exception as e:  # noqa: BLE001 - a task's failure is its result, never the server's
        _set(t["id"], state="failed", finished=now(), error=str(e)[:500], needs=None, session_open=0)
    return get_task(t["id"])


def recover() -> int:
    """At start: a task that was running when the process died is failed, never silently resumed (inputs may have
    been mid-way); queued ones get their runner back."""
    stale = query("SELECT id FROM tasks WHERE state IN ('running', 'needs')")
    for t in stale:
        _set(t["id"], state="failed", finished=now(), error="interrupted by a server restart; start it again",
             session_open=0)
        _forward_stop(get_task(t["id"]))  # a Mac-side run must not keep sending input after the VM says failed
    for d in query("SELECT DISTINCT device_id FROM tasks WHERE state = 'queued'"):
        _ensure_runner(d["device_id"])
    return len(stale)


def wait_task(task_id: str, after: int, wait: float) -> dict:
    """Long-poll: return when a step after `after` exists or the state changes, or after `wait` seconds."""
    t0, first = time.time(), get_task(task_id)
    if first["state"] in FINAL:  # already over: nothing more will come, answer now (the UI would sit on "Working")
        return first
    while time.time() - t0 < wait:
        t = get_task(task_id)
        if t["state"] in FINAL:
            return t
        newer = one("SELECT 1 FROM task_steps WHERE task_id = ? AND idx > ?", (task_id, after))
        if newer or (t["state"], t["needs"]) != (first["state"], first["needs"]):
            return t
        time.sleep(0.4)
    return get_task(task_id)


# ---- routes -----------------------------------------------------------------------------------------------------

def _who(req) -> str | None:
    return req.person["id"] if req.person else None


def _api_devices(req):
    return devices()


def _api_device_frame(req):
    """The live screen (not stored): headers say when it was captured, its age, stream or still."""
    from .httpd import Response
    f = live_frame(req.params["id"], _who(req))
    return Response(200, f["jpeg"], "image/jpeg", {
        "X-Fetched-At": f["fetched_at"], "X-Cache-Age-S": f"{time.time() - f['t']:.2f}",
        "X-Pixel-Age-S": "unknown" if f.get("age_ms") is None else f"{f['age_ms'] / 1000:.2f}",
        "X-Source": f["source"], "Cache-Control": "no-store",
        **({"X-Frame-Seq": str(f["seq"])} if f.get("seq") else {})})


def _frame_response(f, Response):
    v = _frame_view(f)
    return Response(200, Path(f["path"]).read_bytes(), f["content_type"],
                    {"X-Captured-At": f["captured_at"], "X-Frame-Id": f["id"], "X-Age-S": str(v["age_s"]),
                     "Cache-Control": "private, no-store"})


def _api_frame(req):
    from .httpd import Response
    f = one("SELECT * FROM frames WHERE id = ?", (req.params["id"],))
    if not f:
        raise NotFound("No such frame.")
    return _frame_response(f, Response)


def _api_observe(req):
    return 201, look_now(req.params["id"], _who(req))


def _origin(req) -> str:
    o = (req.json() or {}).get("origin") if req.body else None
    return o or "operator"


def _api_create(req):
    b = req.json()
    device_id, kind = b.get("device_id"), b.get("kind")
    if not device_id or not kind:
        raise Refused("Give device_id and kind.")
    return 201, _task_view(create_task(device_id, kind, b.get("input") or {}, _who(req), b.get("origin") or "operator"), -1)


def _api_list(req):
    where, args = ["1"], []
    if req.query.get("device_id"):
        where.append("device_id = ?")
        args.append(req.query["device_id"])
    if req.query.get("open") in ("1", "true"):
        where.append(f"state IN ({','.join('?' * len(OPEN))})")
        args += OPEN
    return [_task_view(t) for t in query(f"SELECT * FROM tasks WHERE {' AND '.join(where)} ORDER BY created DESC "
                                         "LIMIT 200", tuple(args))]


def _api_get(req):
    after = int(req.query.get("after", -1))
    wait = min(float(req.query.get("wait", 0) or 0), 25.0)
    t = wait_task(req.params["id"], after, wait) if wait > 0 else get_task(req.params["id"])
    return _task_view(t, after)


def _api_action(req):
    tid, act, b = req.params["id"], req.params["action"], req.json()
    if act == "stop":
        t = stop(tid, _who(req))
    elif act == "handoff":
        t = handoff(tid, b.get("note") or "", _who(req))
    elif act == "message":
        t = message(tid, b.get("text") or "", _who(req))
    else:
        raise NotFound("Unknown action.")
    return _task_view(t, -1)


ROUTES = [
    ("GET", "/api/devices", _api_devices, "operator"),
    ("GET", "/api/devices/{id}/frame", _api_device_frame, "operator"),
    ("POST", "/api/devices/{id}/observe", _api_observe, "operator"),
    ("GET", "/api/frames/{id}", _api_frame, "operator"),
    ("GET", "/api/tasks/kinds", lambda req: kinds(), "operator"),
    ("POST", "/api/tasks", _api_create, "operator"),
    ("GET", "/api/tasks", _api_list, "operator"),
    ("GET", "/api/tasks/{id}", _api_get, "operator"),
    ("POST", "/api/tasks/{id}/{action}", _api_action, "operator"),
]


WORKSPACE_MODULES = ("workspace", "missions")  # what `tw workspace` serves: no agency, coaching, growth or metrics


def ensure_operator() -> dict:
    """The local operator (a founder person in a founder org), created on first launch so sign-in just works."""
    p = one("SELECT * FROM people WHERE role = 'founder' AND cohort = 'founder' ORDER BY created LIMIT 1")
    if p:
        return p
    import secrets
    oid, pid = new_id("org"), new_id("per")
    with tx() as con:
        con.execute("INSERT INTO orgs (id, name, track, cohort, source, created) VALUES (?, 'This computer', "
                    "'agency', 'founder', 'local', ?)", (oid, now()))
        con.execute("INSERT INTO people (id, org_id, name, role, cohort, source, token, created, verified_at, "
                    "verified_evidence) VALUES (?, ?, 'Operator', 'founder', 'founder', 'local', ?, ?, ?, "
                    "'the local operator')", (pid, oid, secrets.token_urlsafe(24), now(), now()))
    return one("SELECT * FROM people WHERE id = ?", (pid,))


def serve_workspace(host: str = "127.0.0.1", port: int = 8780) -> None:
    """The workspace on its own: devices, live view, reads and read-only missions."""
    import sys
    from . import httpd
    p = ensure_operator()
    sys.stderr.write(f"Open http://{host}:{port}/t/{p['token']} on this computer to sign in.\n")
    httpd.serve(httpd.HEALTH + httpd.module_routes(WORKSPACE_MODULES), host, port, 60,
                lambda: (purge_frames,), banner="phonebridge workspace")
