"""Running flows: on a recorded sequence (`replay`) or on the phone through phonebridge's MCP server
(`walk`, read-only). A live run writes `~/.local/state/agent-reach/evidence/M1-screens/<ts>/` with every
recognised frame + OCR, `events.jsonl` (the journal), a `calls.jsonl` slice pulled from the Mac when it is
reachable, and `RESULT.md` with the git SHA, the modules exercised, the claim and the label (PLAN §9.6)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import time
from phonebridge_client import MCPClient
from datetime import datetime, timezone
from pathlib import Path

from ..backends.replay import ReplayPhone
from ..config import DATA_DIR, site
from .cascade import Cascade, Frame
from .engine import Engine, Flow, ListJournal, Step
from .store import RECORD_DIRS, Store

EVIDENCE = Path(os.environ.get("SCREENS_EVIDENCE", DATA_DIR / "evidence" / "M1-screens")).expanduser()

# the read-only canary walk (SCR-12; the M1 live exit): home → search → home → profile → settings → back →
# home → search again (the second visit must need no VLM) → home. `instagram.search` has no record until
# the VLM names it live; the step after it taps the home tab by region (a flow-level, region-bounded target).
HOME_TAB = {"region": [0.0, 0.9, 0.2, 0.97]}
IG_LIVE_WALK = Flow(name="ig_walk_live", platform="instagram", bundle="com.burbn.instagram", final="instagram.home", steps=(
    Step(at="instagram.home", do="tap", target="search_tab", next=("instagram.search",)),
    Step(at="instagram.search", do="tap", target="home_tab", args=HOME_TAB, next=("instagram.home",)),
    Step(at="instagram.home", do="tap", target="profile_tab"),
    Step(at="instagram.profile", do="tap", target="menu"),
    Step(at="instagram.settings", do="tap", target="back"),
    Step(at="instagram.profile", do="tap", target="home_tab"),
    Step(at="instagram.home", do="tap", target="search_tab", next=("instagram.search",)),
    Step(at="instagram.search", do="tap", target="home_tab", args=HOME_TAB, next=("instagram.home",)),
))
# without a vision-model key the walk stays on known screens: home → profile → settings → back → home
IG_KNOWN_WALK = Flow(name="ig_walk_known", platform="instagram", bundle="com.burbn.instagram", final="instagram.home", steps=(
    Step(at="instagram.home", do="tap", target="profile_tab"),
    Step(at="instagram.profile", do="tap", target="menu"),
    Step(at="instagram.settings", do="tap", target="back"),
    Step(at="instagram.profile", do="tap", target="home_tab"),
))
# TikTok (lane tiktok): For You → the magnifier → Search (learned live; back by its top-left region) → profile → ≡ →
# Settings and privacy → back → home → Search again (fast this time) → back. Read-only throughout.
TT_BACK = {"region": [0.0, 0.04, 0.16, 0.13]}
# Back from Settings: the profile, or the ≡ drawer still open over it (then close it by the dimmed strip)
TT_SETTINGS_BACK = (Step(at="tiktok.settings", do="tap", target="back", next=("tiktok.profile", "tiktok.menu")),
                    Step(at="tiktok.menu", do="tap", target="close", next=("tiktok.profile",)))
TT_LIVE_WALK = Flow(name="tt_walk_live", platform="tiktok", bundle="com.zhiliaoapp.musically", final="tiktok.feed", steps=(
    Step(at="tiktok.feed", do="tap", target="search", next=("tiktok.search",)),
    Step(at="tiktok.search", do="tap", target="back", args=TT_BACK, next=("tiktok.feed",)),
    Step(at="tiktok.feed", do="tap", target="profile_tab"),
    Step(at="tiktok.profile", do="tap", target="menu"),
    Step(at="tiktok.menu", do="tap", target="settings"),
    *TT_SETTINGS_BACK,
    Step(at="tiktok.profile", do="tap", target="home_tab"),
    Step(at="tiktok.feed", do="tap", target="search", next=("tiktok.search",)),
    Step(at="tiktok.search", do="tap", target="back", args=TT_BACK, next=("tiktok.feed",)),
))
TT_KNOWN_WALK = Flow(name="tt_walk_known", platform="tiktok", bundle="com.zhiliaoapp.musically", final="tiktok.feed", steps=(
    Step(at="tiktok.feed", do="tap", target="profile_tab"),
    Step(at="tiktok.profile", do="tap", target="menu"),
    Step(at="tiktok.menu", do="tap", target="settings"),
    *TT_SETTINGS_BACK,
    Step(at="tiktok.profile", do="tap", target="home_tab"),
))
WALKS = {"instagram": IG_LIVE_WALK, "tiktok": TT_LIVE_WALK}
KNOWN_WALKS = {"instagram": IG_KNOWN_WALK, "tiktok": TT_KNOWN_WALK}


class MCPPhone(MCPClient):
    """phonebridge's MCP server (on the Mac, reverse-forwarded to the VM) as the engine's phone."""

    def __init__(self, url: str, device: str, lease: str | None = None):
        super().__init__(url)
        self.device, self.lease = device, lease
        self.observation = None

    async def screen(self, udid=None) -> bytes:
        self.observation = await self.observe(self.device)
        return self.observation.png

    async def ocr(self, png: bytes | None = None) -> list[dict]:
        if png is None:
            self.observation = await self.observe(self.device)
        elif self.observation is None or png != self.observation.png:
            raise ValueError("OCR requested for a frame other than this observation")
        return self.observation.items

    async def tap(self, udid, x, y, press=0.08, w=0, h=0, target=None, **_):
        await self._call("tap", device=self.device, x=x, y=y, w=w or None, h=h or None, lease=self.lease)

    async def swipe(self, udid, x1, y1, x2, y2, duration=0.3, **_):
        await self._call("swipe", device=self.device, x1=x1, y1=y1, x2=x2, y2=y2, duration=duration, lease=self.lease)

    async def home(self, udid=None, **_):
        await self._call("home", device=self.device, lease=self.lease)

    async def launch(self, udid, bundle_id, url=None, **_):
        await self._call("launch", device=self.device, bundle_id=bundle_id, lease=self.lease)

    async def paste(self, udid, text, **_):
        await self._call("paste", device=self.device, text=text, lease=self.lease)

    async def type(self, udid, text, **_):
        await self._call("type", device=self.device, text=text, lease=self.lease)

    async def lock_state(self, udid=None) -> dict:
        return await self._data("lock_state", device=self.device)

    async def foreground(self, udid=None) -> str | None:
        return await self._data("foreground", device=self.device)

    # --- the stream eyes (lane fast-eyes): the daemon judges the live frame, only the verdict crosses the tunnel ---
    async def look(self, platform: str | None = None, expect: list[str] | None = None, settle: bool | str = True) -> dict:
        return await self._data("look", device=self.device, platform=platform, expect=expect, settle=settle)

    async def frame(self, format: str = "jpeg") -> bytes:
        r = await self._call("frame", device=self.device, format=format)
        return base64.b64decode(next(c.data for c in r.content if getattr(c, "type", "") == "image"))

    async def scroll(self, direction: str = "up") -> None:
        await self._call("scroll", device=self.device, direction=direction, lease=self.lease)


async def replay(store: Store, flow: Flow, seq_dir: Path, learner=None):
    phone = ReplayPhone(seq_dir)
    e = Engine(store=store, cascade=Cascade(store), phone=phone, eyes=phone.ocr, journal=ListJournal(), learner=learner,
               sleep=lambda s: asyncio.sleep(0), templates_dir=RECORD_DIRS[0] / "templates")
    return await e.run(flow, rehearse=True)


def git_sha() -> str:
    try:
        sha, branch = (subprocess.run(["git", "rev-parse", *args, "HEAD"], capture_output=True, text=True,
                                      cwd=Path(__file__).parent).stdout.strip() for args in (["--short"], ["--abbrev-ref"]))
        return f"{sha} ({branch})"
    except OSError:
        return "unknown"


def record_session(identity: str) -> None:
    """A finished walk is a read-only session on the account, through the deployment's session command."""
    subprocess.run([*site("SESSION_CMD", None), identity], check=True, capture_output=True, timeout=30)


def calls_slice(t0: float, t1: float) -> list[str]:
    """The Mac's phonebridge calls.jsonl lines inside the run window (best effort over ssh)."""
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", "mac",
                            "tail -n 2000 ~/.local/state/phonebridge/calls.jsonl"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in r.stdout.splitlines():
        try:
            if t0 - 5 <= json.loads(line).get("ts", 0) <= t1 + 5:
                out.append(line)
        except ValueError:
            pass
    return out


async def walk(store: Store, device: str, platform: str, url: str, lease: str | None, out: Path | None,
               use_vlm: bool = True, notify=None, identity: str | None = None) -> dict:
    """`identity`: the account signed in on the phone; a walk that finishes is recorded as its session."""
    flow = WALKS[platform] if use_vlm else KNOWN_WALKS[platform]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out) if out else EVIDENCE / ts
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    n = [0]
    seen = set()
    handles: list[str] = []  # "@samplemaker42" on the account's own profile: which account is signed in

    def on_frame(frame: Frame, label: str, match=None):
        if label.endswith(".profile"):
            handles.extend(t for i in frame.ocr if (t := i["text"].strip()).startswith("@") and t not in handles)
        if frame.sha in seen:
            return
        seen.add(frame.sha)
        n[0] += 1
        stem = out / f"{n[0]:02d}-{label.replace('.', '_')}"
        stem.with_suffix(".png").write_bytes(frame.png)
        os.chmod(stem.with_suffix(".png"), 0o600)
        stem.with_suffix(".ocr.json").write_text(json.dumps(frame.ocr, ensure_ascii=False))
        if match is not None:
            stem.with_suffix(".match.json").write_text(json.dumps(match.__dict__, default=str))

    learner = None
    if use_vlm:
        from .learn import Learner
        from .vlm import VLM
        learner = Learner(vlm=VLM(notify=notify), store=store)
    t0 = time.time()
    journal = ListJournal()
    async with MCPPhone(url, device, lease) as phone:
        e = Engine(store=store, cascade=Cascade(store), phone=phone, eyes=phone.ocr, journal=journal, learner=learner,
                   udid=device, templates_dir=RECORD_DIRS[0] / "templates", on_frame=on_frame)
        try:
            if (await phone.lock_state()).get("locked") is not False:  # iOS 27 reports None: let the bridge look and wake it
                try:
                    await phone._call("unlock", device=device, lease=lease)
                    journal.step(event="unlock")
                except Exception as ex:
                    journal.step(event="unlock_failed", error=str(ex)[:200])
            for attempt in range(2):  # launch, then wait for a known screen of the app (iOS switches apps slowly)
                await phone.launch(device, flow.bundle)
                journal.step(event="launch", bundle=flow.bundle, attempt=attempt)
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    await asyncio.sleep(1.0)
                    frame = await e.look()
                    m = e.recognise(frame, flow)
                    if not m.unknown:
                        break
                else:
                    continue
                break
        except Exception as ex:  # already frontmost, or the lease is someone else's: the engine will see it
            journal.step(event="launch_failed", error=str(ex))
        try:
            result = await e.run(flow, rehearse=True)
        except Exception as ex:  # Escalate (L1) or a phone error: the evidence is still written
            from .engine import Outcome
            journal.step(event="escalate", reason=f"{type(ex).__name__}: {ex}")
            result = Outcome("escalate", None, f"{type(ex).__name__}: {ex}", 0, journal, e.run_id)
    t1 = time.time()
    (out / "events.jsonl").write_text("\n".join(json.dumps(ev, default=str) for ev in journal.events) + "\n")
    (out / "calls.jsonl").write_text("\n".join(calls_slice(t0, t1)) + "\n")
    recognitions = [ev for ev in journal.events if ev["event"] in ("step", "unknown", "popup", "done", "rehearsed")]
    unknown = [ev for ev in journal.events if ev["event"] == "unknown"]
    vlm_calls = [ev for ev in journal.events if ev["event"] == "vlm"]
    cands = [ev for ev in journal.events if ev["event"] == "candidate"]
    summary = {"evidence": str(out), "status": result.status, "screen": result.screen, "reason": result.reason,
               "frames": n[0], "steps": sum(1 for ev in journal.events if ev["event"] == "step"),
               "unknown_events": len(unknown), "vlm_calls": len(vlm_calls), "candidates": [c["screen"] for c in cands],
               "unknown_rate": round(len(unknown) / max(1, len(recognitions)), 3), "seconds": round(t1 - t0, 1),
               "handle": handles[0] if handles else None}
    summary["session"] = None
    if identity and result.status == "done":
        try:
            record_session(identity)
            summary["session"] = identity
        except (OSError, subprocess.SubprocessError) as ex:  # the walk still happened; the report says it wasn't counted
            summary["session_error"] = str(ex)[:200]
    label = "done (read-only)" if result.status in ("done", "rehearsed") else f"{result.status}: {result.reason}"
    claim = (f"read-only walk {flow.name} ({' → '.join(s.at for s in flow.steps)} → {flow.final}); every frame named in the strip "
             f"({n[0]} frames, unknown rate {summary['unknown_rate']}); ")
    claim += (f"{len(vlm_calls)} VLM call(s), candidates saved: {summary['candidates']}" if use_vlm
              else "VLM not exercised (no key: --no-vlm), so no unknown screen was learned in this run")
    (out / "RESULT.md").write_text(
        f"# M1 live walk {ts}\n\n- git: {git_sha()}\n"
        f"- modules: phonebridge.screens.{{cascade,store,engine,learn,vlm,live}}, phonebridge MCP over the Mac tunnel\n"
        f"- device: {device}, platform: {platform}, account: {identity or 'not given'}, flow: {flow.name}\n"
        f"- claim: {claim}\n- label: **{label}**\n"
        f"- signed-in handle (own profile, OCR): {summary['handle'] or 'not seen'}\n"
        f"- OBSERVING session recorded: {summary['session'] or 'no'}\n"
        f"- summary: {json.dumps(summary)}\n")
    return summary


# --- E1 live proof: looks through the stream while a read-only feed scroll runs -----------------------------------

BENCH_EVIDENCE = Path(os.environ.get("EYES_EVIDENCE", DATA_DIR / "evidence" / "E1-fast-eyes")).expanduser()
BUNDLES = {"instagram": "com.burbn.instagram", "tiktok": "com.zhiliaoapp.musically", "reddit": "com.reddit.Reddit"}


def summarise(looks: list[dict]) -> dict:
    """Per-look rows (event=look, as bench and the controller write them) → the E1 numbers: in-daemon ms per layer
    and frame age (CONTROLLER_BAR L1) over recognised looks, stream counters, layers, OCR levels."""
    rec = [l for l in looks if l.get("screen")]

    def dist(values):
        values = [v for v in values if v is not None]
        return {"p50": pct(values, 0.5), "p95": pct(values, 0.95), "p99": pct(values, 0.99)}
    last = looks[-1].get("stream", {}) if looks else {}
    return {"looks": len(looks), "recognised": len(rec), "unknown": len(looks) - len(rec),
            "settled": sum(1 for l in looks if l.get("settled")),
            "in_daemon_ms": dist([l["ms"]["total"] for l in rec if "ms" in l]),
            "settle_ms": dist([l["ms"]["settle"] for l in rec if "ms" in l]), "ocr_ms": dist([l["ms"]["ocr"] for l in rec if "ms" in l]),
            "pixels_ms": dist([l["ms"]["pixels"] for l in rec if "ms" in l]), "round_trip_ms": dist([l.get("rt_ms") for l in rec]),
            "frame_age_ms": dist([l.get("frame_age_ms") for l in looks]),  # L1: every look, not only recognised
            "stream": {"fps": dist([l.get("stream", {}).get("fps") for l in looks]),
                       **{k: last.get(k) for k in ("frames", "aus", "dropped_aus", "decode_errors", "pli", "rr")}},
            "layers": {k: sum(1 for l in rec if l.get("layer") == k) for k in ("R2", "R3", "R4")},
            "ocr_levels": {"fast_only": sum(1 for l in rec if l.get("ocr") == ["fast"]), "accurate": sum(1 for l in rec if "accurate" in (l.get("ocr") or []))}}


def looks_from(events: Path) -> list[dict]:
    """The look rows of an events.jsonl (bench's, or a controller session's that journals its looks the same way)."""
    rows = []
    for line in Path(events).read_text().splitlines():
        if line.strip():
            ev = json.loads(line)
            if ev.get("event") == "look" or ev.get("ev") == "look":
                rows.append(ev)
    return rows


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 1)


async def bench(device: str, platform: str, url: str, lease: str | None, out: Path | None, minutes: float = 2.0,
                flick_every: int = 0, frames_every: int = 5, launch: bool = False, rng=None) -> dict:
    """Per-look timings of the daemon's `look` (in-daemon ms per layer, frame age at the verdict, the stream's fps and
    loss counters, the tunnel round trip) on whatever is on screen. No input by default (Ian: no agent scrolls that
    only exist to test something); `flick_every` > 0 adds a read-only flick every N looks and `launch` opens the app.
    Pauses are drawn at random (no metronome). Writes events.jsonl (one `look` row per look), sampled frames,
    calls.jsonl and RESULT.md; the label is done (read-only) when the recognised looks' in-daemon p95 is under 300 ms,
    else the numbers speak for themselves."""
    import random
    rng = rng or random.Random()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out) if out else BENCH_EVIDENCE / ts
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    looks: list[dict] = []
    t0 = time.time()
    async with MCPPhone(url, device, lease) as phone:
        if not await phone._offered("look"):
            raise RuntimeError("the daemon has no `look` tool: deploy lane/fast-eyes first")
        if launch:
            await phone.launch(device, BUNDLES[platform])
            await asyncio.sleep(2.0)
        deadline = time.monotonic() + minutes * 60
        i = 0
        while time.monotonic() < deadline:
            i += 1
            t = time.perf_counter()
            v = await phone.look(platform=platform)
            rt = round((time.perf_counter() - t) * 1000, 1)
            row = {"event": "look", "i": i, "ts": round(time.time(), 3), "rt_ms": rt, "n_items": len(v.get("items") or []),
                   **{k: v[k] for k in v if k != "items"}}  # frame_age_ms + ms per look: CONTROLLER_BAR L1/L2 inputs
            looks.append(row)
            with (out / "events.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if i % frames_every == 1:
                (out / f"{i:03d}-{(v.get('screen') or 'unknown').replace('.', '_')}.jpg").write_bytes(await phone.frame())
            if flick_every and i % flick_every == 0:
                await phone.scroll("up")
                await asyncio.sleep(rng.uniform(0.3, 0.8))  # the feed moves: the next look settles on it
            else:
                await asyncio.sleep(rng.uniform(0.4, 1.6))
    t1 = time.time()
    (out / "calls.jsonl").write_text("\n".join(calls_slice(t0, t1)) + "\n")
    summary = {"evidence": str(out), "seconds": round(t1 - t0, 1), **summarise(looks),
               "input": {"flicks": len(looks) // flick_every if flick_every else 0, "launch": launch}}
    p95, rec = summary["in_daemon_ms"]["p95"], summary["recognised"]
    label = "done (read-only)" if rec and p95 is not None and p95 < 300 else ("rehearsed (read-only): p95 over budget" if rec else "no recognised look")
    how = (f"a read-only {platform} scroll ({summary['input']['flicks']} flicks)" if flick_every else
           f"whatever {platform} showed, zero input")
    claim = (f"{len(looks)} looks through the live stream on {how} ({rec} recognised, "
             f"{summary['unknown']} unknown); recognised-screen look in-daemon p50 {summary['in_daemon_ms']['p50']} ms, "
             f"p95 {p95} ms, p99 {summary['in_daemon_ms']['p99']} ms (settle/ocr/pixels p95 {summary['settle_ms']['p95']}/"
             f"{summary['ocr_ms']['p95']}/{summary['pixels_ms']['p95']}); frame age at the verdict p95 {summary['frame_age_ms']['p95']} ms, "
             f"p99 {summary['frame_age_ms']['p99']} ms (L1); stream fps p50 {summary['stream']['fps']['p50']}, dropped AUs "
             f"{summary['stream']['dropped_aus']}, PLIs {summary['stream']['pli']}; MCP round trip p95 {summary['round_trip_ms']['p95']} ms")
    (out / "RESULT.md").write_text(
        f"# E1 fast-eyes live bench {ts}\n\n- git: {git_sha()} (lane/fast-eyes)\n"
        f"- modules: phonebridge.stream (CoreDevice display stream, VideoToolbox), phonebridge.screens.fast (Eyes), "
        f"phonebridge MCP look/frame over the Mac tunnel\n- device: {device}, platform: {platform}, input: {summary['input']}\n"
        f"- claim: {claim}\n- label: **{label}**\n- summary: {json.dumps(summary)}\n")
    return summary
