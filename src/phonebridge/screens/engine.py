"""SCR-07/09 and the Q3 triggers: the flow engine over records and their `next` transitions.

From the recognised screen pick the step whose `at` matches, locate its target (OCR anchor, template, region;
a logged fallback only for icons), act, poll `settled` (no fixed sleeps), recognise against `next`. Popups on
the current record get their recorded answer (≤ 5 per step). An `irreversible` target (publish) is tapped only
inside an approved task (`flow.approved`, Ian's `ops approved`) whose record `checks` pass, never in `rehearse`. A record that fails its expectation twice is
demoted and the run escalates (L1). The two learning triggers (DECISIONS 18:10Z): the cascade does not
recognise the chrome, or progress halts (the expected transition is missing, an action had no effect, the
same screen after N actions) → the learner. Every step is journaled through the orchestrator's interface,
faked here until it is merged.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable, Protocol

from .cascade import CHANGE_EPS, Cascade, Frame, Match, changed, locate, system_alert
from .schema import Record, Target

POLL_S = 0.25
SETTLE_CHROME = [(0.0, 0.04, 1.0, 0.12), (0.0, 0.88, 1.0, 1.0)]  # below the status bar (its clock and indicators tick)
MAX_POPUPS_PER_STEP = 5
DEMOTE_AFTER = 2
NO_EFFECT_RETRIES = 1  # re-tap once when a tap did nothing; then the screen counts as halted


class Escalate(RuntimeError):
    """L1: the app layer could not continue (demoted record, unknown with no learner)."""


class Stop(RuntimeError):
    """Park the task: a stop screen, a guard, the operator (R7)."""


@dataclass(frozen=True)
class Step:
    at: str  # the record id this step acts on
    do: str = "tap"  # tap | swipe | home | launch | paste | type | none
    target: str | None = None
    args: dict = field(default_factory=dict)
    next: tuple[str, ...] | None = None  # overrides record.next[target]
    arms: bool = False  # after this step the flow is ARMED (caption pasted): no VLM until the irreversible tap


@dataclass(frozen=True)
class Flow:
    name: str
    platform: str
    bundle: str
    steps: tuple[Step, ...]
    final: str
    armed_from: str | None = None
    approved: bool = False  # the publish task carries Ian's approval (`ops approved`): the only way to an irreversible tap

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))

    def armed_after(self, screen_id: str) -> "Flow":
        return replace(self, armed_from=screen_id)


class Journal(Protocol):
    def step(self, **event) -> None: ...


class ListJournal:
    """The orchestrator's journal interface, faked: append-only events {seq, ts, event, ...}."""

    def __init__(self):
        self.events: list[dict] = []

    def step(self, **event) -> None:
        self.events.append({"seq": len(self.events) + 1, "ts": time.time(), **event})


@dataclass
class Outcome:
    status: str  # done | rehearsed | stopped | escalate
    screen: str | None
    reason: str = ""
    steps: int = 0
    journal: ListJournal | None = None
    run_id: str = ""


@dataclass
class Resolution:
    status: str  # resolved | failed | park | stop
    screen_id: str | None = None
    reason: str = ""


class Engine:
    def __init__(self, store, cascade: Cascade, phone, eyes: Callable[[bytes], list[dict]], journal=None,
                 learner=None, sleep: Callable[[float], Awaitable] | None = None, udid: str = "replay",
                 templates_dir=None, app_version: str | None = None, press: float = 0.08, on_frame=None):
        self.store, self.cascade, self.phone, self.eyes = store, cascade, phone, eyes
        self.journal = journal or ListJournal()
        self.learner = learner
        self.sleep = sleep or asyncio.sleep
        self.udid = udid
        self.templates_dir = templates_dir
        self.app_version = app_version
        self.press = press
        self.run_id = uuid.uuid4().hex[:8]
        self.on_frame = on_frame  # callback(frame, label) for evidence (every recognised look)
        self.coverage = None  # Coverage: set by the caller to feed the exploration map
        self.armed = False
        self.last_action_frame: Frame | None = None

    # --- eyes ---------------------------------------------------------------------------------
    async def look(self) -> Frame:
        png = await self.phone.screen(self.udid)
        items = self.eyes(png)
        if inspect.isawaitable(items):
            items = await items
        return Frame.from_png(png, items)

    async def settle(self, timeout_s: float = 8.0, chrome=None) -> Frame:
        """PER-01: return at the first pair of consecutive frames whose chrome differs below eps; the last
        frame after `timeout_s` (journaled as not settled)."""
        prev = await self.look()
        t0 = time.monotonic()
        chrome = chrome or SETTLE_CHROME  # the chrome only: a feed or a grid of thumbnails never stops moving
        while True:
            await self.sleep(POLL_S)
            cur = await self.look()
            if changed(prev.png, cur.png, chrome=chrome) < CHANGE_EPS:
                return cur
            if time.monotonic() - t0 > timeout_s:
                self.journal.step(event="not_settled", timeout_s=timeout_s)
                return cur
            prev = cur

    def recognise(self, frame: Frame, flow: Flow, expect=None) -> Match:
        return self.cascade.recognise(frame, expect=expect, platform=flow.platform, bundle=flow.bundle)

    # --- hands --------------------------------------------------------------------------------
    async def act(self, step: Step, record: Record, frame: Frame) -> dict:
        if step.do == "none":
            return {"do": "none"}
        if step.do == "home":
            await self.phone.home(self.udid)
            return {"do": "home"}
        if step.do == "launch":
            await self.phone.launch(self.udid, step.args.get("bundle") or record.bundle)
            return {"do": "launch"}
        if step.do == "paste":
            await self.phone.paste(self.udid, step.args.get("text", ""))
            return {"do": "paste"}
        if step.do == "swipe":
            d = step.args.get("direction", "up")
            x, y = frame.width // 2, int(frame.height * 0.55)
            dy = -int(frame.height * 0.3) if d == "up" else int(frame.height * 0.3)
            await self.phone.swipe(self.udid, x, y, x, y + dy, 0.4)
            return {"do": "swipe", "direction": d}
        if step.do == "type":
            await self.phone.type(self.udid, step.args.get("text", ""))
            return {"do": "type"}
        target = record.targets.get(step.target or "")
        if target is None and any(k in step.args for k in ("region", "ocr", "ocr_any", "template")):
            target = Target.from_dict(step.target or "step", {k: v for k, v in step.args.items()
                                                                if k in ("region", "ocr", "ocr_any", "template", "fallback")})
        if target is None:
            raise Stop(f"{record.id}: no target {step.target!r}")
        hit = locate(frame, target, self.templates_dir)
        if hit is None:
            return {"do": "tap", "target": step.target, "missing": True}
        x, y, w, h, how = hit
        if how == "fallback":
            self.journal.step(event="target_fallback", screen=record.id, target=step.target, frame=frame.sha[:12])
        await self.tap(x, y, w, h, target=step.target)
        return {"do": "tap", "target": step.target, "x": x, "y": y, "how": how}

    async def tap(self, x: int, y: int, w: int = 0, h: int = 0, target: str | None = None) -> None:
        self.last_action_frame = None
        if _accepts_press(self.phone.tap):
            await self.phone.tap(self.udid, x, y, self.press, w=w, h=h, target=target)
        else:
            await self.phone.tap(self.udid, x, y, w=w, h=h)

    async def dismiss(self, popup: Record, frame: Frame) -> bool:
        target = popup.targets.get(popup.answer or "")
        hit = locate(frame, target, self.templates_dir) if target else None
        if not hit:
            return False
        x, y, w, h, _ = hit
        await self.tap(x, y, w, h, target=popup.answer)
        self.journal.step(event="popup", screen=popup.id, answer=popup.answer, frame=frame.sha[:12])
        return True

    # --- the loop -----------------------------------------------------------------------------
    async def run(self, flow: Flow, rehearse: bool = False) -> Outcome:
        self.armed = False
        done = 0
        budget = len(flow.steps) * 4 + 6
        frame = await self.settle()
        popups = 0
        failures: dict[str, int] = {}
        last_rec: str | None = None
        no_effect = 0
        try:
            while budget > 0:
                budget -= 1
                step = flow.steps[done] if done < len(flow.steps) else None
                expect = [s.at for s in flow.steps[done:done + 2]] + [flow.final]
                m = self.recognise(frame, flow, expect=expect)
                if self.on_frame:
                    self.on_frame(frame, m.screen_id or "unknown", m)
                if m.unknown:
                    hint = [i for i in (flow.steps[done - 1].next or ()) if i not in self.store.records] if done else []
                    res = await self.resolve_unknown(frame, flow, context="unrecognised chrome", expected=hint or None)
                    if res.status == "resolved":
                        frame = await self.settle()
                        continue
                    if res.status in ("stop", "park"):
                        return self._out("stopped", None, res.reason, done)
                    if last_rec is not None:  # the last acted record led here twice: demote it (SCR-09)
                        failures[last_rec] = failures.get(last_rec, 0) + 1
                        if failures[last_rec] >= DEMOTE_AFTER:
                            self.store.demote(last_rec, f"led to an unknown screen: {res.reason}")
                            self.journal.step(event="demoted", screen=last_rec)
                    raise Escalate(res.reason)
                rec = self.store.records[m.screen_id]
                self._learned(rec, frame)
                if self.coverage and rec.kind == "screen":
                    self.coverage.saw(flow.platform, self.app_version, rec.id)
                if rec.kind == "stop":
                    self.journal.step(event="stop", screen=rec.id, reason=rec.stop_reason, frame=frame.sha[:12])
                    return self._out("stopped", rec.id, f"stop screen {rec.id}: {rec.stop_reason}", done)
                if rec.kind in ("popup", "alert"):
                    if rec.kind == "alert" or popups >= MAX_POPUPS_PER_STEP or not await self.dismiss(rec, frame):
                        text = system_alert(frame) or rec.id
                        return self._out("stopped", rec.id, f"parked on alert {rec.id}: {text}", done)
                    popups += 1
                    frame = await self.settle(rec.timeout_s)
                    if rec.source in ("vlm", "promote"):  # a Learned response: did the popup go away?
                        gone = self.recognise(frame, flow).screen_id != rec.id
                        if gone:
                            if self.store.success(rec.id, f"{self.run_id}:{frame.sha[:12]}"):
                                self.journal.step(event="promoted", screen=rec.id)
                        else:
                            self.store.failure(rec.id, "popup still on screen after its answer")
                    continue
                if self.failed_candidate(rec.id):
                    # a candidate whose resolution failed before: R7 unless the learner may try again (SCR-13)
                    res = await self.resolve_unknown(frame, flow, context=f"{rec.id}: earlier resolution failed")
                    if res.status in ("stop", "park"):
                        return self._out("stopped", rec.id, res.reason, done)
                    frame = await self.settle()
                    continue
                if step is None or (m.screen_id == flow.final and done >= len(flow.steps)):
                    self.journal.step(event="done", screen=m.screen_id)
                    return self._out("done", m.screen_id, "", done)
                if m.screen_id != step.at:
                    # not where the flow expects: a later step's screen (a resolution carried us there), an
                    # earlier one (retry from there), the final screen, or a halt
                    where = next((i for i, s in enumerate(flow.steps) if i >= done and s.at == m.screen_id), None)
                    if where is None:
                        where = next((i for i, s in enumerate(flow.steps[:done]) if s.at == m.screen_id), None)
                    if where is None and m.screen_id == flow.final:
                        self.journal.step(event="done", screen=m.screen_id)
                        return self._out("done", m.screen_id, "", done)
                    if where is None and rec.source in ("vlm", "promote") and step.at not in self.store.records:
                        # the flow names a screen nobody has recorded yet; the slow model just named this one: it stands in
                        self.journal.step(event="alias", screen=rec.id, step_at=step.at)
                        where = done
                    if where is None:
                        res = await self.resolve_unknown(frame, flow, context=f"expected {step.at}, on {m.screen_id}")
                        if res.status in ("stop", "park"):
                            return self._out("stopped", m.screen_id, res.reason, done)
                        frame = await self.settle()
                        continue
                    done = where
                    step = flow.steps[done]
                target = rec.targets.get(step.target) if step.target else None
                if target and target.irreversible:
                    if rehearse:
                        self.journal.step(event="rehearsed", screen=rec.id, target=step.target)
                        return self._out("rehearsed", rec.id, f"stopped before {step.target}", done)
                    if not flow.approved:
                        return self._out("stopped", rec.id, f"{step.target} on {rec.id} publishes: the task is not approved (ops approved)", done)
                    if failed := self.failed_checks(rec, frame):
                        return self._out("stopped", rec.id, f"checks failed: {failed}", done)
                    self.armed = False
                before = frame
                last_rec = rec.id
                action = await self.act(step, rec, frame)
                self.journal.step(event="step", screen=rec.id, step=done, **action, frame=frame.sha[:12])
                if action.get("missing"):
                    res = await self.resolve_unknown(frame, flow, context=f"target {step.target} not on {rec.id}")
                    if res.status in ("stop", "park"):
                        return self._out("stopped", rec.id, res.reason, done)
                    frame = await self.settle()
                    continue
                if step.arms or (flow.armed_from and step.at == flow.armed_from):
                    self.armed = True
                nxt = list(step.next) if step.next else list(rec.next.get(step.target or step.do, []))
                frame = await self.settle(rec.timeout_s, chrome=rec.chrome)
                popups = 0
                after = self.recognise(frame, flow, expect=nxt or None)
                if not nxt and not after.unknown:  # no expectation recorded: whatever is recognised is fine
                    done += 1
                    continue
                if self.coverage and step.target:
                    self.coverage.used(flow.platform, self.app_version, rec.id, step.target, after.screen_id)
                if after.screen_id and after.screen_id not in nxt and self.store.records[after.screen_id].source in ("vlm", "promote") \
                        and any(i not in self.store.records for i in nxt):
                    # the step expected a placeholder nobody recorded; the learned screen we landed on stands in for it
                    self.journal.step(event="alias", screen=after.screen_id, step_at=[i for i in nxt if i not in self.store.records])
                    nxt = list(nxt) + [after.screen_id]
                if after.screen_id in nxt or (after.screen_id and self.store.records[after.screen_id].kind != "screen"):
                    done += 1
                    failures.pop(rec.id, None)
                    no_effect = 0
                    if rec.source in ("vlm", "promote") and after.screen_id in nxt:
                        if self.store.success(rec.id, f"{self.run_id}:{frame.sha[:12]}"):
                            self.journal.step(event="promoted", screen=rec.id)
                    continue
                if changed(before.png, frame.png, chrome=rec.chrome) < CHANGE_EPS and after.screen_id == rec.id:
                    no_effect += 1
                    self.journal.step(event="no_effect", screen=rec.id, step=done, n=no_effect)
                    if no_effect <= NO_EFFECT_RETRIES:
                        continue  # a human re-taps once
                    res = await self.resolve_unknown(frame, flow, context=f"{action.get('do')} {step.target} on {rec.id} had no effect")
                    if res.status in ("stop", "park"):
                        return self._out("stopped", rec.id, res.reason, done)
                    no_effect = 0
                    frame = await self.settle()
                    continue
                # the expected transition did not happen
                failures[rec.id] = failures.get(rec.id, 0) + 1
                self.journal.step(event="expectation_failed", screen=rec.id, expected=nxt, got=after.screen_id,
                                  n=failures[rec.id], frame=frame.sha[:12])
                if rec.source in ("vlm", "promote"):
                    self.store.failure(rec.id, f"expected {nxt}, got {after.screen_id}")
                if after.unknown:
                    self.store.queue_unknown(flow.platform, frame, {"prev": rec.id, "action": action,
                                                                    "run": self.run_id, "app_version": self.app_version})
                    res = await self.resolve_unknown(frame, flow, context=f"after {action.get('do')} {step.target} on {rec.id}, expected {nxt}",
                                                     expected=nxt)
                    if res.status in ("stop", "park"):
                        return self._out("stopped", rec.id, res.reason, done)
                    if res.status == "resolved":
                        failures.pop(rec.id, None)
                        done += 1  # the action was spent; the resolution carried us somewhere known
                        frame = await self.settle()
                        continue
                if failures[rec.id] >= DEMOTE_AFTER:
                    self.store.demote(rec.id, f"expected {nxt} after {step.target}, got {after.screen_id}")
                    self.journal.step(event="demoted", screen=rec.id)
                    raise Escalate(f"L1: {rec.id} demoted after {failures[rec.id]} failed expectations")
                frame = await self.settle()
            raise Escalate("step budget exhausted")
        except Stop as e:
            self.journal.step(event="stop", reason=str(e))
            return self._out("stopped", None, str(e), done)

    def _out(self, status, screen, reason, done) -> Outcome:
        return Outcome(status, screen, reason, done, self.journal, self.run_id)

    def _learned(self, rec: Record, frame: Frame) -> None:
        """A recognised frame joins the corpus (successes are counted at the expectation check)."""
        if frame.png is not None and rec.kind == "screen":
            try:
                self.store.save_frame(rec.platform, rec.id, frame, {"app_version": self.app_version, "run": self.run_id})
            except OSError:
                pass

    def failed_checks(self, rec: Record, frame: Frame) -> list[str]:
        from .cascade import ocr_find
        bad = []
        for name, c in rec.checks.items():
            near = c.get("ocr_near") or c.get("ocr")
            if near and not ocr_find(frame.ocr, near, None, frame.width, frame.height):
                bad.append(name)
            elif (exp := c.get("expect")) and not ocr_find(frame.ocr, exp, None, frame.width, frame.height):
                bad.append(name)
        return bad

    async def resolve_unknown(self, frame: Frame, flow: Flow, context: str, expected=None) -> Resolution:
        """Q3: settle → re-recognise → the learner (VLM) when there is one."""
        self.journal.step(event="unknown", context=context, frame=frame.sha[:12])
        frame = await self.settle()
        m = self.recognise(frame, flow)
        if not m.unknown and not self.failed_candidate(m.screen_id):
            return Resolution("resolved", m.screen_id, "settled into a known screen")
        if self.learner is None:
            self.store.queue_unknown(flow.platform, frame, {"context": context, "run": self.run_id})
            if (text := system_alert(frame)) is not None:
                return Resolution("park", None, f"parked on a system alert: {text}")
            return Resolution("failed", None, f"unknown screen, no learner: {context}")
        if self.armed:  # between the caption paste and the publish tap: noted for the journal, not a block (G3)
            self.journal.step(event="armed_unknown", context=context)
        for _ in range(3):
            res = await self.learner.resolve(self, frame, flow, context, expected=expected)
            if res.status != "failed":
                return res
            frame = await self.settle()
            m = self.recognise(frame, flow)
            if not m.unknown and not self.failed_candidate(m.screen_id):
                return Resolution("resolved", m.screen_id, "known after a failed resolution")
        return res

    def failed_candidate(self, screen_id: str) -> bool:
        """A VLM candidate whose resolution failed before: not a way forward (SCR-13 routes it to R7)."""
        r = self.store.records[screen_id]
        return r.source == "vlm" and bool(r.failures) and not r.successes and r.kind == "screen" and not r.next


def _accepts_press(fn) -> bool:
    try:
        return "press" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
