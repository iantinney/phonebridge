"""TST-01 / PLAN §9.3: `ReplayPhone`, a bridge `Backend` over a recorded sequence of frames.

A sequence directory holds `<name>.png` + `<name>.ocr.json` per frame and `steps.yaml`, one entry per frame
with the action that produced it: `{frame, screen, action: {kind: tap, x, y, target} | {kind: swipe,
direction} | {kind: home|launch|paste|type} | null}`. `screen()` serves the current frame; an input that
matches the next frame's recorded action advances; anything else does nothing (the tap missed). Scripted
faults branch at a step: popup, system_alert, lock, crash, background, banner_tap, wrong_account,
lost_link, stale_layout. OCR comes from the cached `.ocr.json`, so the engine runs without Apple Vision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import yaml

TAP_TOLERANCE = 80  # px: a tap within this of the recorded point (or inside the recorded box) matches
WAIT_KINDS = ("crash", "background", "banner_tap", "wrong_account")  # cleared by a relaunch


@dataclass
class RFrame:
    name: str
    png: bytes
    ocr: list[dict]
    meta: dict = field(default_factory=dict)

    @property
    def sha(self) -> str:
        return sha256(self.png).hexdigest()

    @property
    def action(self) -> dict | None:
        return self.meta.get("action")


@dataclass
class Fault:
    at: int  # the step (frame index) the fault is served at, when the sequence advances into it
    kind: str
    frame: Path | RFrame | None = None  # the fault's own frame (popup, alert, lock, other app, stale layout)
    answer: str | None = None  # popup/system alert: the OCR label whose tap clears it

    def load(self) -> RFrame | None:
        if self.frame is None or isinstance(self.frame, RFrame):
            return self.frame
        p = Path(self.frame)
        ocr = p.with_suffix(".ocr.json")
        return RFrame(p.stem, p.read_bytes(), json.loads(ocr.read_text()) if ocr.exists() else [], {"fault": self.kind})


def write_sequence(d: Path, frames: list[tuple[str, bytes, list[dict], dict]]) -> Path:
    """Write `<name>.png`, `<name>.ocr.json` and `steps.yaml` (one entry per frame, in order)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    steps = []
    for name, png, ocr, meta in frames:
        (d / f"{name}.png").write_bytes(png)
        (d / f"{name}.ocr.json").write_text(json.dumps(ocr, ensure_ascii=False))
        steps.append({"frame": name, **meta})
    (d / "steps.yaml").write_text(yaml.safe_dump(steps, sort_keys=False, allow_unicode=True))
    return d


def read_sequence(d: Path) -> list[RFrame]:
    d = Path(d)
    steps = yaml.safe_load((d / "steps.yaml").read_text()) or []
    out = []
    for s in steps:
        p = d / f"{s['frame']}.png"
        ocr = p.with_suffix(".ocr.json")
        out.append(RFrame(s["frame"], p.read_bytes(), json.loads(ocr.read_text()) if ocr.exists() else [],
                          {k: v for k, v in s.items() if k != "frame"}))
    return out


class ReplayPhone:
    """Implements `phonebridge.bridge.Backend` (and the Bridge-shaped `tap(udid, x, y, w, h)` the engine
    uses) over recorded frames."""

    def __init__(self, sequence: Path | list[RFrame], faults: list[Fault] = (), udid: str = "replay", start: int = 0):
        self.frames = read_sequence(sequence) if isinstance(sequence, (str, Path)) else list(sequence)
        if not self.frames:
            raise ValueError("empty sequence")
        self.udid = udid
        self.index = start
        self.faults: dict[int, list[Fault]] = {}
        for f in faults:
            self.faults.setdefault(f.at, []).append(f)
        self.pending: Fault | None = None  # a fault frame currently on screen
        self._pending_frame: RFrame | None = None
        self.lost = False
        self.locked = False
        self.log: list[dict] = []
        self._by_sha = {f.sha: f.ocr for f in self.frames}
        for fs in self.faults.values():
            for f in fs:
                if (rf := f.load()) is not None:
                    self._by_sha[rf.sha] = rf.ocr
        self._arm(self.index)

    # --- state -------------------------------------------------------------------------------
    @property
    def current(self) -> RFrame:
        return self._pending_frame if self.pending and self._pending_frame else self.frames[self.index]

    @property
    def at_end(self) -> bool:
        return self.index >= len(self.frames) - 1

    def ocr(self, png: bytes) -> list[dict]:
        """The cached OCR for a frame this phone served (the engine's eyes on replay)."""
        return list(self._by_sha.get(sha256(png).hexdigest(), []))

    def _rec(self, op: str, **kw) -> dict:
        e = {"op": op, "ts": time.time(), "index": self.index, "fault": self.pending.kind if self.pending else None, **kw}
        self.log.append(e)
        return e

    def _arm(self, index: int) -> None:
        for f in self.faults.pop(index, []):
            rf = f.load()
            if f.kind == "stale_layout" and rf is not None:
                self.frames[index] = RFrame(rf.name, rf.png, rf.ocr, {**self.frames[index].meta, "fault": "stale_layout"})
                self._by_sha[rf.sha] = rf.ocr
            elif f.kind == "lost_link":
                self.lost = True
            else:
                self.pending, self._pending_frame = f, rf
                self.locked = f.kind == "lock"

    def _advance(self) -> None:
        self.index += 1
        self._arm(self.index)

    def _next_action(self) -> dict | None:
        return None if self.at_end else self.frames[self.index + 1].action

    def _hit(self, x: float, y: float, label: str, frame: RFrame) -> bool:
        from ..screens.cascade import ocr_find
        i = ocr_find(frame.ocr, label)
        return bool(i) and abs(x - i["x"]) <= max(i.get("w", 0) / 2, TAP_TOLERANCE) \
            and abs(y - i["y"]) <= max(i.get("h", 0) / 2, TAP_TOLERANCE)

    # --- Backend -----------------------------------------------------------------------------
    async def devices(self) -> list[dict]:
        return [{"udid": self.udid, "name": "replay", "ios": "0", "product": "iPhone12,1"}]

    async def screen(self, udid: str) -> bytes:
        if self.lost:
            raise ConnectionError("replay: link lost")
        return self.current.png

    async def lock_state(self, udid: str) -> dict:
        return {"locked": self.locked, "passcode": False}

    async def tap(self, udid: str, x: float, y: float, press: float = 0.08, w: float = 0, h: float = 0,
                  target: str | None = None, **_) -> None:
        e = self._rec("tap", x=x, y=y, matched=False)
        if self.pending:
            if self.pending.answer and self._pending_frame and self._hit(x, y, self.pending.answer, self._pending_frame):
                e.update(matched=True, target=self.pending.answer, cleared=self.pending.kind)
                self.pending, self._pending_frame, self.locked = None, None, False
            return
        a = self._next_action()
        if a and a.get("kind") == "tap":
            if a.get("x") is not None and a.get("y") is not None:
                ok = abs(x - a["x"]) <= max(a.get("w", 0) / 2, TAP_TOLERANCE) and \
                    abs(y - a["y"]) <= max(a.get("h", 0) / 2, TAP_TOLERANCE)
            else:  # recorded without coordinates (no call log): match by the target's name
                ok = bool(target) and a.get("target") == target
            if ok:
                e.update(matched=True, target=a.get("target"))
                self._advance()

    async def swipe(self, udid: str, *args, **kw) -> None:
        if args and isinstance(args[0], list):  # Backend shape: path of (x, y, t)
            (x1, y1, _), (x2, y2, _) = args[0][0], args[0][-1]
        else:  # Bridge shape: x1, y1, x2, y2[, duration]
            x1, y1, x2, y2 = args[:4]
        dx, dy = x2 - x1, y2 - y1
        direction = ("up" if dy < 0 else "down") if abs(dy) >= abs(dx) else ("left" if dx < 0 else "right")
        e = self._rec("swipe", direction=direction, matched=False)
        if self.pending:
            return
        a = self._next_action()
        if a and a.get("kind") == "swipe" and a.get("direction", direction) == direction:
            e["matched"] = True
            self._advance()

    async def scroll(self, udid: str, direction: str = "up", **_) -> None:
        dy = -600 if direction == "up" else 600
        await self.swipe(udid, 585, 1400, 585, 1400 + dy)

    async def type(self, udid: str, keys, **_) -> None:
        await self._simple("type")

    async def paste(self, udid: str, text: str, **_) -> None:
        await self._simple("paste")

    async def home(self, udid: str, **_) -> None:
        await self._simple("home")

    async def launch(self, udid: str, bundle_id: str, url: str | None = None, **_) -> None:
        e = self._rec("launch", bundle=bundle_id, matched=False)
        if self.pending and self.pending.kind in WAIT_KINDS:
            self.pending, self._pending_frame = None, None
            e["cleared"] = True
            return
        a = self._next_action()
        if a and a.get("kind") == "launch" and not self.pending:
            e["matched"] = True
            self._advance()

    async def unlock(self, udid: str, **_) -> dict:
        was = self.locked
        if self.pending and self.pending.kind == "lock":
            self.pending, self._pending_frame = None, None
        self.locked = False
        return {"was_locked": was}

    async def close(self) -> None:
        return None

    async def _simple(self, kind: str) -> None:
        e = self._rec(kind, matched=False)
        if self.pending:
            return
        a = self._next_action()
        if a and a.get("kind") == kind:
            e["matched"] = True
            self._advance()
