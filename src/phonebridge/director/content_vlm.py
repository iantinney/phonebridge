"""The background vision model, for writes only (BEH-02, PER-05). One call per post the director stays long on,
through screens' `vlm.py` client (`client(png, prompt, effort) -> dict`), `effort: low`, closed schema, cached by
frame hash so replays never call out. It proposes; `WriteGate` applies the dwell fraction, the first-write time and
the pacer; the caller's action does the write and verifies it on the next look."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable

EFFORT = "low"
KINDS = ("video", "photo", "carousel", "text", "live", "unknown")
FIELDS = ("kind", "is_ad", "topics", "interest", "text_to_read", "quick_skip", "summary")
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(FIELDS), "properties": {
    "summary": {"type": "string", "maxLength": 240},
    "kind": {"enum": list(KINDS)}, "is_ad": {"type": "boolean"},
    "topics": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    "interest": {"type": "integer", "minimum": 0, "maximum": 3},
    "text_to_read": {"type": "string"}, "quick_skip": {"type": "boolean"}}}
CLOSE_WAIT_S = 2.0  # a process asked to end gets this long before its group is killed
Client = Callable[[bytes, str, str], Awaitable[dict]]


@dataclass(frozen=True)
class Proposal:
    kind: str
    is_ad: bool
    topics: list
    interest: int
    text_to_read: str
    quick_skip: bool
    summary: str = ""  # one sentence: who, what, the hook, the topic (what the decision cites)


def validate(d: dict) -> Proposal:
    summary = str(d.get("summary", ""))[:240]
    if summary.lstrip().startswith("{") or "maxLength" in summary:  # the schema echoed back (seen on a cold first call)
        summary = ""
    d = {**d, "summary": summary}
    if set(d) != set(FIELDS):
        raise ValueError(f"schema: fields {sorted(set(d) ^ set(FIELDS))} off")
    if d["kind"] not in KINDS or not isinstance(d["is_ad"], bool) or not isinstance(d["quick_skip"], bool):
        raise ValueError("schema: kind/is_ad/quick_skip")
    if not isinstance(d["topics"], list) or len(d["topics"]) > 3 or not all(isinstance(t, str) for t in d["topics"]):
        raise ValueError("schema: topics ≤ 3 strings")
    if not isinstance(d["interest"], int) or isinstance(d["interest"], bool) or not 0 <= d["interest"] <= 3:
        raise ValueError("schema: interest 0–3")
    if not isinstance(d["text_to_read"], str):
        raise ValueError("schema: text_to_read")
    return Proposal(**d)


def prompt(profile: dict | None = None) -> str:
    p = profile or {}
    return ("You see one phone screenshot of a social feed. Answer with JSON only, exactly these fields: "
            f"{json.dumps(SCHEMA['properties'])}. summary: one sentence — who posted, what it shows, its hook, its topic (e.g. "
            "'Magnus Carlsen chess clip: a streamer reacts to a blunder'). kind: what the post is. is_ad: Sponsored/paid. topics: up to 3 short "
            f"tags. interest 0-3: how much this viewer would care, given they like {', '.join(p.get('topics', [])) or 'nothing in particular'}"
            f"{' and liked ' + ', '.join(p['likes']) if p.get('likes') else ''}. text_to_read: the caption or on-image text "
            "worth reading, else empty. quick_skip: true if a person would flick past at a glance.")


class CachedVLM:
    def __init__(self, client: Client, cache_path: Path | None, profile: dict | None = None):
        self.client, self.path, self.profile = client, Path(cache_path) if cache_path else None, profile
        self.cache: dict[str, dict] = {}
        self.last_cached = False
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.cache[e["key"]] = e["proposal"]

    async def describe(self, png: bytes) -> Proposal:
        key = hashlib.sha256(png).hexdigest()
        if key in self.cache:
            self.last_cached = True
            return Proposal(**self.cache[key])
        self.last_cached = False
        raw = await self.client(png, prompt(self.profile), EFFORT)
        prop = validate(raw)
        if raw.get("summary") and not prop.summary:  # the schema echoed back (a cold first call): one more ask
            prop = validate(await self.client(png, prompt(self.profile), EFFORT))
        self.cache[key] = asdict(prop)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "proposal": asdict(prop)}) + "\n")
        return prop


class WriteGate:
    """Which writes a proposal may become: after `first_write_after_s` into the session, after `like_after_fraction`
    of the post's dwell, interest ≥ 2 (a like) or 3 with something read (a comment), never on an ad, and only what
    the pacer allows (`pacer_ok(action)`)."""

    def __init__(self, first_write_after_s: float, like_after_fraction: float = 0.5, comment_write: str = "preview"):
        """`comment_write`: 'preview' (DECISIONS G3-G17: a comment is drafted and shown, never posted, until Ian has tailored
        and tested his voice) or 'auto'."""
        self.first_write_after_s, self.like_after_fraction, self.comment_write = first_write_after_s, like_after_fraction, comment_write

    def candidates(self, prop: Proposal, elapsed_fraction: float, session_s: float, pacer_ok) -> list[str]:
        if prop.is_ad or prop.quick_skip or prop.interest < 2 or session_s < self.first_write_after_s or elapsed_fraction < self.like_after_fraction:
            return []
        out = ["like"]
        if prop.interest == 3 and prop.text_to_read.strip():
            out.append("comment" if self.comment_write == "auto" else "comment_preview")
        return [a for a in out if a == "comment_preview" or pacer_ok(a)]


def claude_p_client(model: str = "haiku", timeout: int = 120) -> Client:
    """The stopgap vision client until the API key lands: one `claude -p` per call (the CLI reads the frame with its Read
    tool), JSON out. ~9 s model time on the VM (measured 2026-10-18): background use only — the fast cue covers the glance."""
    import subprocess
    import tempfile

    async def client(png: bytes, prompt_text: str, effort: str) -> dict:
        import asyncio

        def run() -> dict:
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "frame.png"
                f.write_bytes(png)
                cmd = [claude_bin(), "-p", "--model", model, "--output-format", "json", "--allowedTools", "Read", "--permission-mode",
                       "bypassPermissions", "--strict-mcp-config", "--setting-sources", "",
                       f"Read the image file {f} (a phone screenshot). {prompt_text}"]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=d)
            if r.returncode:
                raise RuntimeError(f"claude -p exit {r.returncode}: {r.stderr[-200:]}")
            out = json.loads(r.stdout)
            text = str(out.get("result", "")).strip()
            text = text[text.find("{"): text.rfind("}") + 1]
            return json.loads(text)
        return await asyncio.to_thread(run)
    return client


def claude_bin() -> str:
    """The claude CLI, also under launchd's bare PATH on the Mac (the daemon found no ffmpeg that way: integrate 10-19)."""
    import os
    import shutil

    found = shutil.which("claude")
    if found:
        return found
    for p in ("~/.local/bin/claude", "/opt/homebrew/bin/claude", "/usr/local/bin/claude", "~/.claude/local/claude"):
        if os.access(os.path.expanduser(p), os.X_OK):
            return os.path.expanduser(p)
    return "claude"


def _kill_group(proc) -> None:
    """The process's own group (start_new_session): a stalled one goes whole, ssh and its children with it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - a test double or a process already gone
        (getattr(proc, "kill", None) or proc.terminate)()


class WarmClaude:
    """The understanding model until the API key lands: one persistent `claude -p --input-format stream-json
    --output-format stream-json` process (Haiku) kept warm, fed user messages with the frame as an image block. Measured
    on the VM 2026-10-18: 9 s for the first answer (6.5 s of it startup), 1.3 s for the next ones. One request at a time,
    across threads too (the process answers in order, so a second writer would read the first's answer; a restart or close
    waits for the turn in progress); the process is restarted if it dies. One instance is ONE conversation: never share it
    between phones. Call `client()` for the `Client` shape."""

    def __init__(self, model: str = "haiku", timeout: float = 60.0, system: str | None = None, schema: dict | None = None, max_turns: int = 25,
                 via: list[str] | None = None):
        import asyncio

        self.model, self.timeout, self.system, self.schema = model, timeout, system, schema  # system/schema: screens' slow model
        self.via = via  # e.g. ["ssh", "-o", "BatchMode=yes", "myhost"]: the same CLI on a host where it is logged in to a plan
        self.proc = None
        self.lock = asyncio.Lock()
        self._serial = threading.RLock()  # a whole turn, restart and close included (reentrant: a turn closes a dead stream)
        self.calls = 0
        self.max_turns, self.turns = max_turns, 0  # one stream-json process is ONE growing conversation (~1.3K tokens per image turn, surfaces' measure): restart it every max_turns

    def _start(self):
        import shlex

        argv = ["claude" if self.via else claude_bin(), "-p", "--model", self.model, "--input-format", "stream-json", "--output-format", "stream-json",
                "--verbose", "--tools", "", "--strict-mcp-config", "--setting-sources", ""] \
            + (["--system-prompt", self.system] if self.system else []) + (["--json-schema", json.dumps(self.schema)] if self.schema else [])
        if self.via:  # the VM keeps no copy of the screenshots it was shown (the CLI's sessions are not saved)
            argv.insert(2, "--no-session-persistence")
        cmd = [*self.via, "--", shlex.join(argv)] if self.via else argv  # over ssh the remote shell parses one quoted command line
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                     start_new_session=True)  # its own process group: a stalled one is killed whole (ssh and its children)

    def _ask(self, png: bytes, text: str) -> dict:
        with self._serial:
            return self._turn(png, text)

    def _turn(self, png: bytes, text: str) -> dict:
        import base64

        if self.proc is not None and self.turns >= self.max_turns:
            self.close()  # bound the conversation's context (and its per-call latency); the next start costs ~6 s once
        if self.proc is None or self.proc.poll() is not None:
            self._start()
            self.turns = 0
        image = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg" if png[:3] == b"\xff\xd8\xff" else "image/png",
                                              "data": base64.b64encode(png).decode()}}] if png else []  # no image: a text-only ask (voice.py)
        msg = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}, *image]}}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        expired = threading.Event()  # a stalled stream (a dead tunnel) is killed at the deadline: the blocked readline returns

        def expire(proc=self.proc):
            expired.set()
            _kill_group(proc)
        timer = threading.Timer(self.timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            return self._read(expired)
        finally:
            timer.cancel()

    def _read(self, expired) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                self.close()
                if expired.is_set():
                    raise TimeoutError("claude stream: no result in time")
                raise RuntimeError("claude stream closed")
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                self.calls += 1
                self.turns += 1
                out = str(ev.get("result", ""))
                if ev.get("is_error"):  # a plan usage limit, an auth failure: the CLI's own words, not a parse error
                    raise RuntimeError(f"claude: {out[:300]}")
                if isinstance(ev.get("structured_output"), dict):
                    return ev["structured_output"]
                return json.loads(out[out.find("{"): out.rfind("}") + 1])

    def close(self) -> None:
        with self._serial:
            proc, self.proc = self.proc, None
            if proc is None:
                return
            try:
                proc.stdin.close()
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=CLOSE_WAIT_S)  # reaped here: a process started for one ask leaves no zombie
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=CLOSE_WAIT_S)

    def client(self) -> Client:
        import asyncio

        async def call(png: bytes, prompt_text: str, effort: str) -> dict:
            async with self.lock:
                return await asyncio.to_thread(self._ask, png, prompt_text)
        return call


def understanding_client(model_cli: str = "haiku") -> tuple[Client, str]:
    """The understanding model behind the local one: the warm claude process on Ian's Claude plan. Ian 2026-10-19 04:3xZ:
    no Anthropic API key; model calls beyond the free local model go through his Claude or Codex plan, whichever measures
    better (a Codex client joins when that comparison is run). Returns (client, name)."""
    warm = WarmClaude(model_cli)
    return warm.client(), f"warm-claude:{model_cli}"


def daemon_understanding_client(phone, device: str) -> Client:
    """The daemon's `understand` (local Qwen on the Mac, fast-eyes) as a `Client`: the daemon describes its latest frame,
    so the png passed in is not sent; the result's `frame.seq` is kept on the proposal dict as `_frame` for the journal."""
    async def call(png: bytes, prompt_text: str, effort: str) -> dict:
        r = await phone.understand(device, prompt=prompt_text, max_tokens=300)  # bake-off: local max_tokens ≥ 300 (a cut-off JSON was the main local failure)
        if r is None:
            raise RuntimeError("the daemon offers no understand tool")
        if not r.get("parsed", True) and r.get("raw"):
            raw = str(r["raw"])
            try:
                r = {**r, **json.loads(raw[raw.find("{"): raw.rfind("}") + 1])}
            except Exception:
                pass
        out = {k: r.get(k) for k in FIELDS if k in r}
        out.setdefault("kind", "unknown")
        out.setdefault("is_ad", False)
        out.setdefault("topics", [])
        out.setdefault("interest", r.get("interest") if isinstance(r.get("interest"), int) else 1)
        out.setdefault("text_to_read", "")
        out.setdefault("quick_skip", False)
        out["summary"] = str(r.get("summary") or "")
        return out
    return call


async def understanding_chain(phone=None, device: str | None = None, model_cli: str = "haiku") -> tuple[Client, str]:
    """The understanding path in order: the daemon's local model when the daemon offers `understand`, else the warm claude
    process on Ian's plan (no API key, Ian 04:3xZ)."""
    if phone is not None and device and hasattr(phone, "understand"):
        try:
            if await phone._offered("understand"):
                return daemon_understanding_client(phone, device), "daemon:understand (local Qwen)"
        except Exception:
            pass
    return understanding_client(model_cli)


class Budget:
    """At most `per_min` calls in any rolling minute, spread evenly (a rate-limited remote model behind the local one)."""

    def __init__(self, per_min: int = 8, clock=None):
        import time

        self.per_min, self.clock, self.calls = per_min, clock or time.monotonic, []
        self.min_gap_s = 60.0 / per_min  # spread evenly: 10 req/min = at most one post per 6 s (bake-off, supervisor 02:2xZ)

    def allow(self) -> bool:
        now = self.clock()
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= self.per_min or (self.calls and now - self.calls[-1] < self.min_gap_s):
            return False
        self.calls.append(now)
        return True


def jpeg_585(png: bytes, width: int = 585, quality: int = 80) -> bytes:
    """The frame for a remote model: 585 px wide JPEG q80 (controller-verify's bake-off: same answers as the full PNG at a
    fraction of the bytes; 585 × 1266 = 966 image tokens)."""
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality)
    return out.getvalue()


def escalation_client(primary: Client, secondary: Client, when=None) -> Client:
    """Primary (the local model) answers; the secondary (the warm claude process) is asked only when `when(answer)` says so (low confidence:
    nameless summary, no topics, parse trouble) and the budget allows; its answer replaces the primary's."""
    when = when or (lambda d: not d.get("summary") or not d.get("topics"))

    async def call(png: bytes, prompt_text: str, effort: str) -> dict:
        d = await primary(png, prompt_text, effort)
        if when(d):
            try:
                d2 = await secondary(png, prompt_text, effort)
                return {**d2, "_escalated": True}
            except Exception:
                return d
        return d
    return call
