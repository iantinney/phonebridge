"""SCR-06 (Q3, G3): the self-learning loop. On an Unfamiliar screen the engine settles, re-recognises, then
asks the slow model (Haiku; Sonnet when unsure or after a failed action); the engine executes the answer,
recognises the result, and the whole resolution (frame, OCR, answer, action, outcome, next screen) is saved
at once as a record with drafted anchors and success/failure counters: the Learned response runs fast from
then on, no confirmation step. A response that keeps failing goes back to the slow model. Stop + operator
only for captcha, 2FA, payment, warnings, action blocks, or a screen the slow model missed three times.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .cascade import Frame, dhash, hamming, norm, ocr_find
from .engine import Resolution
from .schema import Record

ALLOWED_ACTIONS = {"dismiss", "back", "scroll", "tap_named", "open", "none"}
STOP_KINDS = {"captcha": "captcha", "two_factor": "2fa", "payment": "payment", "warning": "warning",
              "action_block": "action_block", "login": "login_wall"}
NUMERIC = re.compile(r"^[\d:.,%\s]+$")


class GuardError(ValueError):
    pass


def guard(answer: dict) -> None:
    """Schema sanity only. Caution (avoid deleting, sending, following, changing settings unless the task calls
    for it) is prompt guidance for the model, not enforcement code (Ian: don't over-engineer); publishing is
    gated by the task's approval in the engine."""
    if answer.get("action") not in ALLOWED_ACTIONS:
        raise GuardError(f"action {answer.get('action')!r} is not one of {sorted(ALLOWED_ACTIONS)}")


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "screen").casefold()).strip("_")
    return s[:40] or "screen"


def draft_anchors(frame: Frame, store, platform: str, exclude: tuple[str, ...] = ()) -> list[str]:
    """Anchor lines for a new record: short, confident OCR lines on this frame that are not on the nearest
    known screen of the platform (content-free by construction: numbers, times and handles dropped)."""
    lines, content = [], []
    for i in frame.ocr:
        t = i["text"].strip()
        if i.get("conf", 1.0) < 0.8 or len(t) < 2 or len(t.split()) > 5 or NUMERIC.match(t) or t.startswith("@") or "•" in t:
            continue
        if norm(t) in exclude or not norm(t) or i["y"] < 0.04 * frame.height:  # the status bar
            continue
        # the chrome first (title band, search field, tab bar); the content area only to fill up
        (lines if i["y"] < 0.2 * frame.height or i["y"] > 0.86 * frame.height else content).append(t)
    lines += content
    known = Counter()
    for r in store.for_platform(platform):
        for a in r.match.ocr_any + r.match.ocr_all:
            known[norm(a)] += 1
    fresh = [t for t in lines if norm(t) not in known]
    out = fresh[:6] if len(fresh) >= 2 else (fresh + [t for t in lines if t not in fresh])[:6]
    seen, uniq = set(), []
    for t in out:
        if norm(t) not in seen:
            seen.add(norm(t))
            uniq.append(t)
    return uniq


@dataclass
class Learner:
    vlm: object
    store: object
    max_failures: int = 3  # slow-model attempts on one screen (the second and later escalate) before parking

    def __post_init__(self):
        self.failures: dict[str, int] = {}
        self.answers: list[dict] = []

    def key(self, frame: Frame) -> str:
        return f"{dhash(frame.png, (0, 0, 1, 0.15)):016x}" if frame.png else frame.sha[:16]

    async def resolve(self, engine, frame: Frame, flow, context: str, expected=None) -> Resolution:
        key = self.key(frame)
        m0 = engine.recognise(frame, flow)
        if not m0.unknown and self.store.records[m0.screen_id].source == "vlm":  # a candidate that failed before
            self.failures[key] = max(self.failures.get(key, 0), self.store.records[m0.screen_id].failures)
        if self.failures.get(key, 0) >= self.max_failures:
            return Resolution("stop", None, f"operator: {self.failures[key]} resolutions failed on this screen ({context})")
        known = sorted(r.id for r in self.store.for_platform(flow.platform))
        escalate = self.failures.get(key, 0) > 0  # the fast model's action already failed here: the stronger one
        try:
            import asyncio  # the slow model off the event loop (a daemon keeps serving looks meanwhile)

            answer = await asyncio.to_thread(self.vlm.resolve, frame.png, frame.ocr, known, flow.platform, context, frame.width, frame.height, escalate=escalate)
            if not escalate and answer.get("confidence", 1.0) < 0.5 and getattr(self.vlm, "escalate_model", None):
                engine.journal.step(event="vlm_unsure", answer=answer, frame=frame.sha[:12])
                answer = await asyncio.to_thread(self.vlm.resolve, frame.png, frame.ocr, known, flow.platform, context, frame.width, frame.height, escalate=True)
        except Exception as e:  # cap, network, refusal: degrade to the operator, never guess
            return Resolution("park", None, f"VLM unavailable: {e}")
        self.answers.append(answer)
        engine.journal.step(event="vlm", answer=answer, frame=frame.sha[:12], context=context)
        try:
            guard(answer)
        except GuardError as e:
            return Resolution("failed", None, f"unusable VLM answer ({e})")
        if (stop := STOP_KINDS.get(answer.get("kind"))) is not None:
            return Resolution("stop", None, f"{stop}: {answer.get('screen_name')} ({answer.get('reason')})")
        dangling = [i for i in (expected or []) if i not in self.store.records]
        if answer.get("kind") == "screen" and answer.get("action") == "none":
            # a plain screen the store did not know: naming it is the resolution (the flow goes on from it)
            record = self.save(frame, answer, False, True, flow.platform, flow.bundle, None, engine,
                               rid=dangling[0] if len(dangling) == 1 else None)
            if record is not None:
                self.failures.pop(key, None)
                return Resolution("resolved", record.id, f"VLM named the screen {record.id}")
            return Resolution("failed", None, "VLM named the screen but no anchors could be drafted")
        acted = await self.execute(engine, frame, answer)
        after = await engine.settle()
        m = engine.recognise(after, flow)
        moved = after.sha != frame.sha and (m.screen_id is not None or self.key(after) != key)
        ok = acted and moved
        record = self.save(frame, answer, acted, ok, flow.platform, flow.bundle, m.screen_id, engine,
                           rid=dangling[0] if len(dangling) == 1 else None)
        if ok:
            self.failures.pop(key, None)
            return Resolution("resolved", m.screen_id, f"VLM: {answer.get('screen_name')} → {answer.get('action')} {answer.get('label_text')!r}")
        self.failures[key] = self.failures.get(key, 0) + 1
        if record is not None:
            self.store.failure(record.id, f"VLM action {answer.get('action')} {label!r} did not change the screen"
                               if (label := answer.get("label_text")) is not None else "VLM action failed")
        return Resolution("failed", None, f"VLM action {answer.get('action')} {answer.get('label_text')!r} did not change the screen")

    async def execute(self, engine, frame: Frame, answer: dict) -> bool:
        action, label = answer.get("action"), (answer.get("label_text") or "").strip()
        if action == "none":
            return False
        if action == "scroll":
            x, y = frame.width // 2, int(frame.height * 0.6)
            await engine.phone.swipe(engine.udid, x, y, x, y - int(frame.height * 0.3), 0.4)
            return True
        from .vlm import numbered_elements
        by_id = dict(numbered_elements(frame.ocr))
        eid = answer.get("element_id", -1)
        if isinstance(eid, int) and eid in by_id:  # grounded by element number: tap its centre (humanized by the bridge)
            i = by_id[eid]
            await engine.tap(int(i["x"]), int(i["y"]), int(i.get("w", 40)), int(i.get("h", 40)), target=label or i["text"].strip())
            return True
        if label and (i := ocr_find(frame.ocr, label, None, frame.width, frame.height)):
            await engine.tap(int(i["x"]), int(i["y"]), int(i.get("w", 40)), int(i.get("h", 40)), target=label)
            return True
        box = answer.get("target_box") or []
        if len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1:
            x, y = int((box[0] + box[2]) / 2 * frame.width), int((box[1] + box[3]) / 2 * frame.height)
            w, h = int((box[2] - box[0]) * frame.width), int((box[3] - box[1]) * frame.height)
            engine.journal.step(event="target_fallback", screen="vlm", target=label or action, frame=frame.sha[:12])
            await engine.tap(x, y, w, h, target=label or action)
            return True
        if action == "back":
            engine.journal.step(event="target_fallback", screen="vlm", target="back", frame=frame.sha[:12])
            await engine.tap(int(0.06 * frame.width), int(0.065 * frame.height), 40, 40, target="back")
            return True
        return False

    def save(self, frame: Frame, answer: dict, acted: bool, ok: bool, platform: str, bundle: str,
             next_screen: str | None, engine, rid: str | None = None) -> Record | None:
        kind = "popup" if answer.get("kind") in ("popup", "sheet", "alert") else "screen"
        rid = rid or f"{platform}.{'popup' if kind == 'popup' else 'screen'}.{slug(answer.get('screen_name'))}"
        if rid in self.store.records:
            return self.store.records[rid]
        label = (answer.get("label_text") or "").strip()
        anchors = draft_anchors(frame, self.store, platform)
        if label and norm(label) not in {norm(a) for a in anchors}:
            anchors = [label] + anchors
        if len(anchors) < 2:
            return None
        targets, answer_name, nxt = {}, None, {}
        if label and ok:
            answer_name = slug(label)
            i = ocr_find(frame.ocr, label, None, frame.width, frame.height)
            region = None
            if i:
                region = [max(0, (i["x"] - 3 * i.get("w", 40)) / frame.width), max(0, (i["y"] - 3 * i.get("h", 40)) / frame.height),
                          min(1, (i["x"] + 3 * i.get("w", 40)) / frame.width), min(1, (i["y"] + 3 * i.get("h", 40)) / frame.height)]
            targets[answer_name] = {"ocr": label, **({"region": region} if region else {})}
            if next_screen:
                nxt[answer_name] = [next_screen]
        if kind == "popup" and not answer_name:
            kind = "screen"
        rec = Record.from_dict({
            "id": rid, "platform": platform, "bundle": bundle, "kind": kind,
            "appearance": frame.appearance if frame.png is not None else "any",  # as observed, never assumed
            "match": {"ocr_any": anchors, "at_least": min(2, len(anchors)),
                      **({"chrome_dhash": {"region": [0, 0, 1, 0.15], "value": f"{dhash(frame.png, (0, 0, 1, 0.15)):016x}",
                                           "max_bits": 4}} if frame.png and False else {})},
            "targets": targets, **({"answer": answer_name} if kind == "popup" else {}), "next": nxt,
            "status": "candidate", "source": "vlm", "seen_versions": [engine.app_version] if engine.app_version else [],
            "note": f"VLM {answer.get('kind')}: {answer.get('reason')} | action {answer.get('action')} {label!r} "
                    f"{'worked' if ok else 'failed'}; context: run {engine.run_id}",
        })
        if ok:  # the slow model's own working response is the first success (saved immediately, no review)
            rec.successes, rec.matches = 1, [f"{engine.run_id}:{frame.sha[:12]}"]
        self.store.add_candidate(rec, frame, {"app_version": engine.app_version, "run": engine.run_id,
                                              "vlm": answer, "outcome": "ok" if ok else "failed", "next": next_screen})
        engine.journal.step(event="candidate", screen=rid, ok=ok)
        return rec
