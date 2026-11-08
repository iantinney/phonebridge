"""The `do` task kind: "do X on this account" (docs/design/task-agent.md, approved 10-19, DECISIONS "Task agent go").

    payload: {goal, slots{handle?, tag?, ...}, scope[write classes], urgency: now|next_session, budget{action: n},
              path?: hint, minutes?: the visit's length}

The task is a detour inside a normal visit (the director's `task_detour`: orient on the feed, a few posts, the task,
a short glance, leave), never a teleport. Each step tries, in order (§3.2):

    T0  the Learned path's step, when the live look names the step's Screen and its target resolves
    T1  the planner (screens/planner.py: Claude-plan Haiku, Sonnet on low confidence or a failed expect)
    T2  an unfamiliar Screen: the Learner names it (or dismisses a sheet) and saves a Learned response
    T3  the operator: a stop record, ask_operator, or the third failure on one Screen ends the task `operator`; the
        VM's hand-off reaches Ian on Telegram with the frame

Every input goes through the director's gestures (`Director.tap/swipe/type`: the daemon's humaniser, lease, frontmost,
operator and freshness guards), with mobile-jev's mutation rules (§3.4) as rules of `TaskActor.act`: re-observe right
before every input and re-resolve the target's meaning on that look; the expected next Screen after every step (a miss
is a divergence, never a blind repeat); a write whose outcome the next look can't confirm ends the task `unverified`
and is never retried; typed text is read back; the same (screen, op, target) failing twice is `stuck` → escalation.
Writes spend the budget the VM granted from the pacer and must be in the task's declared scope (a write outside it is
refused and journaled); each verified write is a `write` event, so the director's ONE digest lists it with its undo.
A successful run is compiled into a Learned path (screens/paths.py), saved at once, replayed model-free next time."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..director import primitives as pr
from ..director.session import Stopped
from ..screens.paths import LearnedPath, PathStore, compile_path
from ..screens.planner import Planner, Turn

WRITES = ("save", "like", "follow", "unfollow", "comment", "reply", "dm", "post", "repost")  # the write classes a control can be
TASK_WRITES = ("save", "like", "follow", "unfollow")  # the ones a task may declare: speech (comment/reply/dm) waits for a real
# composer read-back and correctness's comment guard; posting is the post kind's
MAX_STEPS = 25
PAIR_TRIES = 3  # looks for a paired frame to read a write back on, before the write ends the task unverified  # the human-time budget of the visit is the real limit (§3.2)
UNTIL_S = 4.0  # an app transition (search results, a profile load): the until() after a step
CONFIDENT = 0.6  # below: the planner's step is re-asked of the stronger model
APP = {"ig": "instagram", "tiktok": "tiktok", "x": "x"}
DO_APPS = {"x": ("com.atebits.Tweetie2", "x")}  # apps a task may open that taskapi.APPS (browse) doesn't list yet
CARDS = (Path(__file__).parent.parent / "screens" / "records").resolve()  # <app>/CARD.md
OWN_BAND = 0.16  # the profile title / the post's author line: the top band of the screen
OWN_BANDS = {"x.profile": 0.45}  # X's expanded profile header: banner, name, then @handle at ~0.36 h
HANDLE_LINE = re.compile(r"[a-z0-9._]{2,30}")
PROFILES = ("instagram.profile.other", "tiktok.profile.other", "x.profile")
LOOK_AT = ("instagram.profile.other", "instagram.post", "instagram.tag", "x.profile", "x.post")  # a drawn reading dwell on arrival
POST_SCREENS = {"instagram": ("instagram.home", "instagram.post"), "x": ("x.post", "x.profile")}  # screens showing a post's action row (X: a profile lists posts with their own rows)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("’", "'")).strip().casefold().lstrip("@#")


@dataclass
class Action:
    op: str  # tap | type | back | scroll | read | done | ask_operator | dwell
    target: str = ""  # a named target, `text:<shown text>`, `tile[first_unpinned]`, or "" with `item`
    item: int = -1
    text: str = ""
    write: str = ""
    expect: tuple[str, ...] = ()
    tier: int = 1
    reason: str = ""
    confidence: float = 1.0
    step: int | None = None  # the path step this is (T0)

    @classmethod
    def from_planner(cls, a: dict, tier: int = 1) -> Action:
        return cls(a.get("op", "ask_operator"), (a.get("target") or "").split(" (→")[0].strip(), int(a.get("item", -1) if a.get("item") is not None else -1),
                   a.get("text") or "", (a.get("write") or "").strip().lower(), (a["expect"],) if a.get("expect") else (), tier,
                   a.get("reason") or "", float(a.get("confidence", 1.0) or 0.0))

    @property
    def sig(self) -> tuple:
        return (self.op, self.target, self.item, norm(self.text))


@dataclass
class Outcome:
    ok: bool
    why: str = ""
    screen: str | None = None
    reached: bool = False  # the expected Screen showed (or, without an expectation, the screen changed)
    sent: bool = False  # an input went to the phone
    value: str | None = None  # read: the text read; a write: verified True/False/None
    target: str = ""  # the target as resolved (an item becomes text:<its text> for the compiler)


class TaskActor:
    """The one way a task touches the phone (bench-adapter's visual-remote injects this same object into Mobilerun's
    loop: no second input path). `look()` is a fresh, journaled look with its own frame's pixels, the Screen named
    (the daemon's cascade, or ours for screens learned during this task) and the named targets resolved to boxes;
    `act(action, look)` performs one action under the §3.4 rules and returns what came of it."""

    def __init__(self, d, store=None, scope=(), budget: dict | None = None, slots: dict | None = None, on_write=None):
        from ..screens.cascade import Cascade
        from ..screens.store import RECORD_DIRS, Store

        self.d = d
        self.store = store if store is not None else Store(records_dirs=RECORD_DIRS)
        self.cascade = Cascade(self.store)
        self.scope, self.budget, self.slots = set(scope), dict(budget or {}), dict(slots or {})
        self.writes: list[dict] = []
        self.refused: list[dict] = []
        self.already: set[str] = set()  # declared writes found already done (no input sent, nothing spent)
        self.n = 0  # inputs sent by this task
        self.on_write = on_write  # the runtime's fsync'd journal: a sent write reaches the VM even if the visit dies after it

    @property
    def app(self) -> str:
        return self.d.app

    async def look(self) -> dict:
        d = self.d
        if not hasattr(d, "size"):
            await d.warm()
        pair = await d.paired_frame()
        if pair is None:  # a moving screen (a playing video) and no kept frame: the look's text, the latest pixels, unpaired;
            m = await d.look_dict()  # navigation may go on, a write never does (it needs pixels_seq: act refuses without)
            if not m.get("items"):
                raise Stopped("no_look", "no look: no input without one")
            pair = (m["items"], await d.pixels(m.get("png")), None)
        items, png, seq = pair  # seq: the frame both the text and the pixels came from (paired_frame), None when unpaired
        look = {**d.last_look, "items": items, "png": png, "frame_seq": seq if seq is not None else d.last_look.get("frame_seq"), "pixels_seq": seq}
        d.screen_ok(items)  # a stop record / iOS alert ends the task here (S3: never a gesture on it)
        w, h = d.size
        look["width"], look["height"] = w, h
        if not look.get("screen"):  # unknown to the daemon: maybe a screen this task's T2 just learned
            look["screen"], look["screen_by"] = self.learned(items, png), "task-cascade"
        look["targets"] = self.targets(look)
        return look

    def targets(self, look: dict) -> dict[str, list[float]]:
        """{name: [x, y, w, h]} (centre, px) for every named control on this look: the Screen record's targets (OCR,
        template, region), surfaces' measured resolvers on post screens, and the task's selectors (grid tiles)."""
        from ..screens.cascade import Frame, locate
        from ..screens.store import REPO_RECORDS

        w, h = look["width"], look["height"]
        out: dict[str, list[float]] = {}
        rec = self.store.records.get(look.get("screen") or "")
        if rec is not None:
            frame = Frame(look.get("items") or [], look.get("png"), w, h)
            for name, t in rec.targets.items():
                try:
                    hit = locate(frame, t, REPO_RECORDS / "templates")
                except Exception:  # noqa: BLE001 - a template that won't load is a target that isn't there
                    hit = None
                if hit:
                    out[name] = [hit[0], hit[1], hit[2], hit[3]]
        if look.get("screen") in POST_SCREENS.get(self.app, ()):
            from ..director.feedmaps import FEEDMAPS

            for name in (FEEDMAPS[self.app].targets if self.app in FEEDMAPS else {}):
                if name != "follow_button" and (fm := pr.feedmap_target(self.app, name, look, w, h)) is not None:
                    out[name] = list(fm)
        if look.get("screen") in PROFILES and (fm := pr.feedmap_target(self.app, "follow_button", look, w, h)) is not None:
            out["follow_button"] = list(fm)  # surfaces' resolver: the profile's own button row only, never a suggestion card
        for name, box in selectors(look, self.app).items():
            out.setdefault(name, box)
        return out

    def resolve(self, a: Action, look: dict) -> tuple[list[float], str] | None:
        """The action's box on this look, and its meaning (what the compiler stores): a named target, a text selector
        (`text:<shown text>`, slot names already filled), or a numbered item (→ `text:<its text>`)."""
        from ..screens.vlm import numbered_elements

        items = look.get("items") or []
        if a.target.startswith("text:"):
            want = norm(fill(a.target[5:], self.slots))
            fields = [b for n, b in look.get("targets", {}).items() if "field" in n]  # the typed text shows in its field too: never the field
            items = [i for i in items if not any(abs(i["x"] - b[0]) <= b[2] / 2 and abs(i["y"] - b[1]) <= b[3] / 2 for b in fields)]
            hits = [i for i in items if norm(i.get("text", "")) == want] or [i for i in items if want and first_word(i.get("text", "")) == want]
            if not hits:
                return None
            i = min(hits, key=lambda i: i["y"])
            return [i["x"], i["y"], i.get("w", 40), i.get("h", 40)], f"text:{i['text'].strip()}"
        if a.target and a.target in look.get("targets", {}):
            return look["targets"][a.target], a.target
        if a.item >= 0:
            by_id = dict(numbered_elements(items))
            if a.item in by_id:
                i = by_id[a.item]
                return [i["x"], i["y"], i.get("w", 40), i.get("h", 40)], f"text:{i['text'].strip()}"
        return None

    def allow_write(self, write: str) -> str | None:
        """Why this write may not happen now (None: it may): the task's declared scope, then the VM's grant."""
        if write not in self.scope:
            return f"{write} is outside this task's scope ({', '.join(sorted(self.scope)) or 'read-only'})"
        if self.budget.get(write, 0) <= 0:
            return f"no {write} budget left for this task (the pacer's grant is spent)"
        return None

    def refuse(self, a: Action, why: str, screen) -> Outcome:
        self.refused.append({"op": a.op, "target": a.target, "write": a.write, "why": why, "screen": screen})
        self.d.ev("task_refused", op=a.op, target=a.target, write=a.write, why=why, screen=screen)
        return Outcome(False, f"refused: {why}", screen)

    def shows_handle(self, look: dict) -> bool:
        """The task's account is the screen's own: on a profile its title, on a post its author line under the title (the
        top band), never a handle elsewhere on screen (a comment, a suggestion) nor a longer one (nasa ≠ nasahubble)."""
        want = norm(self.slots.get("handle", ""))
        h, band = look.get("height") or 2532, OWN_BANDS.get(look.get("screen") or "", OWN_BAND)
        return not want or any(first_word(i.get("text", "")) == want and i["y"] < band * h for i in look.get("items") or [])

    def owns(self, look: dict, box) -> str | None:
        """Why the control at `box` is not the task's account's (None: it is): it must sit above any 'Suggested for you'
        block and below its own account line, with no other account's line between them."""
        items = look.get("items") or []
        sug = min((i["y"] for i in items if norm(i.get("text", "")).startswith("suggested")), default=None)
        if sug is not None and box[1] > sug:
            return "the control belongs to a suggested account"
        want = norm(self.slots.get("handle", ""))
        if want and look.get("screen") in POST_SCREENS.get(self.app, ()):
            above = [i for i in items if i["y"] < box[1] and HANDLE_LINE.fullmatch(first_word(i.get("text", ""))) and i["x"] < 0.6 * look["width"]
                     and len(norm(i.get("text", "")).split(" ")) <= 3]
            near = max(above, key=lambda i: i["y"], default=None)  # the post this control belongs to: the nearest author line above
            if near is None or first_word(near.get("text", "")) != want:
                return f"the nearest post above is {near and near.get('text')!r}, not @{want}'s"
        return None

    async def act(self, a: Action, look: dict, step_no: int = 0) -> Outcome:
        d = self.d
        before = look.get("screen")
        if a.op not in ("tap", "type"):
            a.write = ""  # only a tap or typing can write
        if a.op == "dwell":  # a person looks at a profile / a post before acting: the drawn reading time
            await d.sleep(d.read_seconds(min(60, sum(len((i.get("text") or "").split()) for i in look.get("items") or []))))
            return Outcome(True, "dwell", before, True)
        if a.op == "type" and norm(a.text) not in {norm(str(v)) for v in self.slots.values()}:
            return self.refuse(a, "typed text must be text the task gave (a slot value)", before)
        if a.op == "type" and not fields_of(look):
            return self.refuse(a, f"no text field declared on {before}", before)
        r0 = self.resolve(a, look) if a.op == "tap" else None
        if a.op == "tap" and r0 is None:  # the decision named a control its own frame doesn't show: no meaning to check, no tap
            return Outcome(False, f"no {a.target or f'item {a.item}'} on the decision's frame", before)
        # re-observe right before the input (mobile-jev assertFresh): same Screen, the target still means the same thing
        fresh = await self.look()
        if fresh.get("screen") != before:
            return Outcome(False, f"the screen changed before the input ({before} → {fresh.get('screen')})", fresh.get("screen"))
        w, h = fresh["width"], fresh["height"]
        sent_target, box = a.target, None
        try:
            if a.op == "tap":
                r1 = self.resolve(a, fresh)
                if r1 is None:
                    return Outcome(False, f"no {a.target or f'item {a.item}'} on screen", before)
                if r0[1] != r1[1]:
                    return Outcome(False, f"the target's meaning changed ({r0[1]!r} → {r1[1]!r})", before)
                box, sent_target = r1
                if sent_target.startswith("text:") and COUNT_TEXT.match(sent_target[5:].strip()) and before in POST_SCREENS.get(self.app, ()):
                    return self.refuse(a, f"{sent_target[5:]!r} is a count on a post: it labels a like/repost control, not a link", before)
                a.write = write_of(sent_target, before) or a.write  # what the tap does is read from the control tapped, not the planner's word
                if a.write:
                    if why := self.allow_write(a.write):
                        return self.refuse(a, why, before)
                    if fresh.get("pixels_seq") is None and d.require_look:  # live: a write's before-state is read on ONE frame
                        return Outcome(False, "no paired frame for the write's before-state (text and pixels of one frame)", before)
                    if not self.shows_handle(fresh):
                        return self.refuse(a, f"this is not @{self.slots.get('handle')}'s screen", before)
                    if why := self.owns(fresh, box):
                        return self.refuse(a, why, before)
                    if a.write in ("follow", "unfollow") and sent_target != "follow_button":
                        return self.refuse(a, "a follow goes only through the profile's own follow_button", before)
                    if write_state(self.app, a.write, fresh, box) is True:  # already done: the toggle would undo it; nothing sent or spent
                        self.already.add(a.write)
                        return Outcome(True, f"already {a.write}d", before, True, value="already", target=sent_target)
                    self.budget[a.write] = self.budget.get(a.write, 0) - 1
                    self.writes.append({"action": a.write, "verified": None, "verify": "sent", "who": self.slots.get("handle"), "screen": before,
                                        "target": sent_target, "before_frame_seq": fresh.get("frame_seq"), "before_pixels_seq": fresh.get("pixels_seq")})
                    d.ev("write_sent", action=a.write, target=sent_target, screen=before, kind="task")  # journaled BEFORE the input: a stop
                    if self.on_write:
                        self.on_write(self.writes[-1])
                declared = not sent_target.startswith("text:")  # a record's target or a selector: the control itself, inside an avoid
                await d.tap(box[0], box[1], sent_target, kind="task", step=step_no, expect=a.expect or None, w=box[2], h=box[3],  # band
                            zone_exempt=declared)  # (IG's search bar); a plain text line obeys the avoid zones like any gesture
            elif a.op == "type":
                await d.type(a.text, f"type:{a.target or 'field'}", kind="task", step=step_no)
            elif a.op == "back":
                t = fresh.get("targets", {}).get("back")
                if t:
                    await d.tap(t[0], t[1], "back", kind="task", step=step_no, expect=a.expect or None, w=t[2], h=t[3], zone_exempt=True)
                    sent_target = "back"
                else:
                    x1, y1, x2, y2, T = d.human.edge_back(w, h)
                    await d.swipe(x1, y1, x2, y2, T, "edge back", kind="task", step=step_no, expect=a.expect or None, zone_exempt=True)
            elif a.op in ("scroll", "scroll_up"):
                lo, hi = d.free_band(0.5 * w)
                T, _ = d.params.glide["nudge_s"].draw(d.rng)
                x = w * d.rng.uniform(0.4, 0.6)
                if a.op == "scroll":
                    await d.swipe(x, hi - 0.05 * h, x + d.rng.uniform(-12, 12), max(lo, hi - 0.45 * h), T, "scroll", kind="task", step=step_no)
                else:  # a short drag down: the page comes back a little and the app shows its hidden bars (X's tab bar)
                    y0 = lo + d.rng.uniform(0.05, 0.15) * h
                    await d.swipe(x, y0, x + d.rng.uniform(-12, 12), y0 + d.rng.uniform(0.12, 0.2) * h, T, "scroll up", kind="task", step=step_no)
            elif a.op == "read":
                val = read_value(a, fresh, self)
                where = a.target if "@" in a.target else (read_selector(fresh, val, self.store, self.slots) if val else None)  # for the path
                return Outcome(val is not None, "read" if val is not None else "the value is not on screen", before, val is not None,
                               value=val, target=where or a.target)
            else:
                return Outcome(False, f"unknown op {a.op}", before)
        except pr_refusals() as e:
            if a.write and self.writes and self.writes[-1].get("verify") == "sent":  # zero input sent: the write didn't happen
                self.writes.pop()
                self.budget[a.write] = self.budget.get(a.write, 0) + 1
            return Outcome(False, f"refused by the guards ({type(e).__name__}): zero input sent", before)
        self.n += 1
        return await self.after(a, look, fresh, before, sent_target, box)

    async def after(self, a: Action, look: dict, fresh: dict, before: str | None, target: str, box) -> Outcome:
        """`until` the expected Screen (or, with no expectation, any change); a write is verified on the next look."""
        d = self.d
        reaction, _ = d.params.glide["reaction_s"].draw(d.rng)
        if a.op == "type":  # read-back (jev input-verification): the declared field's own text, suggestions tolerated
            boxes = fields_of(fresh)
            got = await pr.until(self._look_quiet, lambda m: typed_shown(a.text, m, boxes), UNTIL_S, reaction, d.sleep, d.now)
            ok = got is not None
            return Outcome(ok, "typed text read back" if ok else "the typed text is not in the field (read-back failed)",
                           (got or {}).get("screen", before), ok, True, target=target)
        if a.write:
            row = self.writes[-1]
            d.counts["writes"] = d.counts.get("writes", 0) + 1
            await d.sleep(reaction)
            for _ in range(PAIR_TRIES):  # the write is read back on ONE frame (text and pixels): never on an unpaired fallback
                after = await self.look()  # a stop screen here raises: the write row above already says it was sent
                if after.get("pixels_seq") is not None or not d.require_look:
                    break
                await d.sleep(d.params.glide["reaction_s"].draw(d.rng)[0])
            paired = after.get("pixels_seq") is not None or not d.require_look
            state = write_state(self.app, a.write, after, box) if paired else None  # where the control was (a filled bookmark no
            same = after.get("screen") in (before, *a.expect) and self.shows_handle(after)  # longer matches the outline template)
            verified = bool(state) and same
            row.update({"verified": verified if state is not None else None, "verify": "paired" if paired else "unpaired",
                        "frame_seq": after.get("frame_seq"), "pixels_seq": after.get("pixels_seq"), "same_post": same})
            d.ev("write", **row, kind="task")
            return Outcome(verified, "write verified" if verified else "write NOT verified on the next look", after.get("screen"), verified,
                           True, value=verified, target=target)
        def landed(m):  # a path step must land on its expected Screen (a miss is a divergence); the planner's step must move
            s = m.get("screen")  # something (its guess of the next Screen id is journaled, not enforced: it sees where it landed)
            moved = s != before or pr.text_shift(look.get("items") or [], m.get("items") or []) not in (None, 0.0)
            return s is None or ((s in a.expect) if (a.expect and a.tier == 0) else (s in a.expect or moved))
        got = await pr.until(self._look_quiet, landed, UNTIL_S, reaction, d.sleep, d.now)
        if got is not None and got.get("screen") is None:  # an unfamiliar screen over the landing (a popup): T2 next, then the
            return Outcome(False, "landed on an unfamiliar screen", None, False, True, target=target)  # step's `at` checks resume
        if got is None:
            return Outcome(False, f"expected {list(a.expect) or 'a change'}; still {d.last_look.get('screen') or before}", d.last_look.get("screen"),
                           False, True, target=target)
        return Outcome(True, "ok", got.get("screen"), True, True, target=target)

    def learned(self, items: list[dict], png) -> str | None:
        """A Screen the daemon could not name but this task's T2 learned (a Learned response saved minutes ago, not yet in
        the daemon's store); the repo's records are the daemon's to apply, never second-guessed here."""
        from ..screens.cascade import Frame

        m = self.cascade.recognise(Frame(items, png, *self.d.size), platform=self.app)
        rec = self.store.records.get(m.screen_id or "")
        return m.screen_id if rec is not None and rec.source == "vlm" else None

    async def _look_quiet(self) -> dict:
        m = await self.d.look_dict()
        if not m.get("screen") and m.get("items"):
            m["screen"] = self.learned(m["items"], m.get("png"))
        return m


WRITE_TARGETS = {"save_icon": "save", "heart": "like", "follow": "follow", "follow_button": "follow", "share": "post", "post": "post",
                 "send": "dm", "comment_field": "comment"}
WRITE_LABELS = {"follow": "follow", "follow back": "follow", "following": "unfollow", "unfollow": "unfollow", "requested": "unfollow",
                "send": "dm", "post": "comment", "reply": "reply", "share": "post", "save": "save", "like": "like",
                "repost": "repost", "undo repost": "repost", "quote": "repost", "bookmark": "save", "remove bookmark": "save"}
COUNT_TEXT = re.compile(r"^[\d.,]+[kKmM]?$")  # a bare count on a post screen labels a control (X's like / repost counts)


def write_of(meaning: str, screen: str | None) -> str | None:
    """The write class of a control, from its own name or label (so a tap the planner did not declare is still counted
    against the task's scope): the bookmark saves, the heart likes, "Follow" follows, "Following" unfollows, "Send" sends.
    Anything else is navigation; the task's caution beyond its scope is the planner prompt's (DECISIONS 10-18)."""
    if meaning.startswith("text:"):
        return WRITE_LABELS.get(norm(meaning[5:]))
    return WRITE_TARGETS.get(meaning)


def first_word(t: str) -> str:
    """The line's first word, normalised: a handle row 'nasa · Following' or 'nasa ✓' is nasa."""
    return (norm(t).split(" ") or [""])[0].rstrip("✓·")


def pr_refusals():
    from ..director.session import GestureRefused, StaleInput

    return (GestureRefused, StaleInput)


def fill(template: str, slots: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(slots.get(m.group(1), m.group(0))), template)


def fields_of(look: dict) -> list[list[float]]:
    """The text fields the Screen record declares (targets named *field), resolved on this look."""
    return [b for n, b in (look.get("targets") or {}).items() if "field" in n]


def inside(i: dict, b: list[float]) -> bool:
    return abs(i["x"] - b[0]) <= b[2] / 2 and abs(i["y"] - b[1]) <= b[3] / 2


def typed_shown(text: str, look: dict, boxes: list[list[float]]) -> bool:
    """The typed text is the declared field's own text: an OCR line inside the field's box equal to it, never a header,
    the keyboard's suggestion bar or a result row below it."""
    want = norm(text)
    return bool(boxes) and any(norm(i.get("text", "")) == want and any(inside(i, b) for b in boxes) for i in look.get("items") or [])


def read_value(a: Action, look: dict, actor: TaskActor) -> str | None:
    """read: the planner's value must be on this very frame (a count read from the profile is that profile's OCR); a
    path's read names where the value sits relative to a fixed label (`<label>@dx,dy`, fractions of the screen)."""
    items = look.get("items") or []
    if a.text:
        want = norm(a.text)
        hit = next((i for i in items if want and want in norm(i.get("text", ""))), None)
        return a.text if hit else None
    if "@" in a.target:
        label, _, off = a.target.rpartition("@")
        dx, dy = (float(v) for v in off.split(","))
        w, h = look.get("width") or 1170, look.get("height") or 2532
        lab = fill(label, actor.slots)
        anchor = next((i for i in items if norm(i.get("text", "")) == norm(lab)), None) or \
            next((i for i in items if norm(i.get("text", "")).startswith(norm(lab))), None)
        if anchor is None:
            return None
        x, y = anchor["x"] + dx * w, anchor["y"] + dy * h
        near = min(items, key=lambda i: (i["x"] - x) ** 2 + (i["y"] - y) ** 2, default=None)
        return near["text"].strip() if near is not None and abs(near["y"] - y) < 0.012 * h and abs(near["x"] - x) < 0.15 * w else None
    if (a.target or a.item >= 0) and (r := actor.resolve(a, look)):
        return r[1].removeprefix("text:")
    return None


def read_selector(look: dict, answer: str, store, slots: dict) -> str | None:
    """Where an answer sits, for the path: the nearest fixed label (the Screen record's own anchor words, or a slot's
    value shown alone) and the offset to the answer's line, `<label>@dx,dy`. None when the answer is no line here."""
    items = look.get("items") or []
    want = norm(answer)
    hit = next((i for i in items if norm(i.get("text", "")) == want), None) or next((i for i in items if want and norm(i.get("text", "")).startswith(want)), None)
    if hit is None:
        return None
    rec = store.records.get(look.get("screen") or "")
    words = {norm(t) for t in ((rec.match.ocr_any + rec.match.ocr_all) if rec else [])}
    vals = {norm(str(v)): "{" + k + "}" for k, v in slots.items() if v}
    labels = []
    for i in items:
        t = norm(i.get("text", ""))
        if i is hit or not t:
            continue
        if t in vals:
            labels.append((vals[t], i))
        elif any(t == w_ or t.startswith(w_ + " ") for w_ in words):
            labels.append((i["text"].strip() if t in words else next(w_ for w_ in words if t.startswith(w_ + " ")), i))
    if not labels:
        return None
    w, h = look.get("width") or 1170, look.get("height") or 2532
    lab, i = min(labels, key=lambda li: (li[1]["x"] - hit["x"]) ** 2 + (li[1]["y"] - hit["y"]) ** 2)
    return f"{lab}@{(hit['x'] - i['x']) / w:.3f},{(hit['y'] - i['y']) / h:.3f}"


def write_state(app: str, write: str, look: dict, box) -> bool | None:
    """The write's state on one frame: True done (bookmark filled, heart red, 'Following'), False not, None unreadable."""
    png, items = look.get("png"), look.get("items") or []
    w, h = look.get("width") or 1170, look.get("height") or 2532
    if write == "save":
        from ..director.feedmaps import FEEDMAPS

        heart = None if box else pr.feedmap_target(app, "heart", look, w, h)  # a FILLED bookmark no longer matches the outline template:
        at = box or pr.feedmap_target(app, "save_icon", look, w, h) or (heart and (0.938 * w, heart[1]))  # its row is the heart's
        if at is None or png is None:
            sheet = any(norm(i.get("text", "")) in ("saved", "save to collection", "saved to collection") for i in items)
            return True if sheet else None
        fm = FEEDMAPS.get(app)
        return bool(fm.saved_state(png, at[1], w, h)) if fm and fm.saved_state else None
    if write == "like":
        from ..director.actions import red_share
        from ..director.feedmaps import FEEDMAPS

        heart = box or pr.feedmap_target(app, "heart", look, w, h)
        if heart is None or png is None:
            return None
        fm = FEEDMAPS.get(app)
        if fm is not None and fm.like_state is not None:  # the app's own reading (X: pink around the heart on that row)
            return bool(fm.like_state(png, heart[1], w, h))
        return red_share(png, heart[0], heart[1]) > 0.12
    if write in ("follow", "unfollow") and look.get("screen") in PROFILES:
        from ..director.feedmaps import follow_state

        st = follow_state(app, items, png, w, h)
        if st is not None:
            return (st in ("following", "requested")) if write == "follow" else st == "not_following"
    if write == "follow":  # the button, never the "following" count label (a number sits right above that one)
        def label(i):
            return any(re.fullmatch(r"[\d.,]+[kKmM]?", (j.get("text") or "").strip()) and 0 < i["y"] - j["y"] < 0.04 * h and abs(i["x"] - j["x"]) < 0.08 * w
                       for j in items)
        buttons = [i for i in items if not label(i)]
        if any(norm(i.get("text", "")) in ("following", "requested") for i in buttons):
            return True
        return False if any(norm(i.get("text", "")) in ("follow", "follow back") for i in buttons) else None
    return None


X_TAB_BAR_PLUS = 0.885  # X's compose "+" sits at 0.857 h while the tab bar shows, 0.916 h once a scroll hid it (10-19 frames)
X_SEARCH_ICON = (0.30, 0.931)  # the bar's second icon (x-20261019T063445Z/01-home: home 0.1, search 0.3, Grok 0.5, bell 0.7, DMs 0.9)


def x_search_tab(look: dict) -> list[float] | None:
    """X's Search tab, only while the tab bar is on screen (the compose '+' high, not low): never a tap where the bar was."""
    w, h, items = look["width"], look["height"], look.get("items") or []
    plus = next((i for i in items if (i.get("text") or "").strip() == "+" and i["x"] > 0.84 * w), None)
    if plus is None or plus["y"] > X_TAB_BAR_PLUS * h:
        return None
    return [X_SEARCH_ICON[0] * w, X_SEARCH_ICON[1] * h, 0.09 * w, 0.03 * h]


def selectors(look: dict, app: str) -> dict[str, list[float]]:
    """Task selectors resolved on the live look (no stored coordinates): a profile's grid tiles `tile_1..tile_6`, and
    `tile[first_unpinned]` when a pin template exists to tell pinned tiles (IG puts them first). The grid's top is
    anchored on the profile's button row (Follow / Message), designed on the sim and unmeasured live (surfaces)."""
    if look.get("screen") in ("x.home", "x.home.tabs"):
        return {"search_tab": b} if (b := x_search_tab(look)) else {}
    if look.get("screen") == "x.profile":
        return x_profile_posts(look)
    if look.get("screen") not in ("instagram.profile.other", "instagram.profile", "instagram.tag"):
        return {}
    w, h, items = look["width"], look["height"], look.get("items") or []
    anchor = max((i["y"] + i.get("h", 30) / 2 for i in items if norm(i.get("text", "")) in
                  ("message", "follow", "following", "follow back", "edit profile", "share profile", "top posts")), default=None)
    if anchor is None:
        return {}
    top, tw, th = anchor + GRID_GAP * h, w / 3, w / 3 * 4 / 3  # IG's portrait 3:4 tiles, three across
    out = {f"tile_{r * 3 + c + 1}": [tw * (c + 0.5), top + th * (r + 0.5), tw * 0.8, th * 0.8] for r in range(2) for c in range(3)
           if top + th * (r + 1) < 0.9 * h}
    pinned = pinned_tiles(look, out)
    if pinned is not None:
        first = next((n for n in sorted(out, key=lambda n: int(n.split("_")[1])) if n not in pinned), None)
        if first:
            out["tile[first_unpinned]"] = out[first]
    return out


def x_profile_posts(look: dict) -> dict[str, list[float]]:
    """The X profile's posts below its "Posts" tab: `post[first_unpinned]` = the first post body not under a "Pinned" label
    (X shows pinned posts first). Designed: no real frame of the list yet (surfaces' capture plan); the planner picks when
    it does not resolve."""
    w, h, items = look["width"], look["height"], look.get("items") or []
    tab = next((i for i in items if norm(i.get("text", "")) == "posts" and i["y"] < 0.75 * h), None)
    if tab is None:
        return {}
    pins = [i["y"] for i in items if norm(i.get("text", "")).startswith("pinned") and i["y"] > tab["y"]]
    heads = sorted((i for i in items if i["y"] > tab["y"] + 0.02 * h and i["x"] > 0.15 * w and "@" in (i.get("text") or "")), key=lambda i: i["y"])
    for hd in heads:  # each post's header line ("Name @handle · 2h"); a "Pinned" label sits just above a pinned one
        if not any(0 < hd["y"] - p < 0.05 * h for p in pins):
            body = next((i for i in sorted(items, key=lambda i: i["y"]) if hd["y"] + 0.005 * h < i["y"] < hd["y"] + 0.08 * h and i["x"] > 0.15 * w), None)
            if body is not None:
                return {"post[first_unpinned]": [body["x"], body["y"], body.get("w", 200), body.get("h", 40)]}
    return {}


GRID_GAP = 0.125  # button row → the grid's first row (highlights + the grid/reels/tagged tab row); designed on the sim
PIN_TEMPLATE = CARDS / "templates" / "instagram" / "pin.png"


def pinned_tiles(look: dict, tiles: dict) -> set[str] | None:
    """The tiles showing the pin glyph in their top-right corner (template match on the pixels); None when no pin
    template exists yet (then `first_unpinned` does not resolve and the planner decides: a divergence, not a guess)."""
    png = look.get("png")
    template = look.get("pin_template") or PIN_TEMPLATE
    if png is None or not Path(template).exists():
        return None
    from ..screens.cascade import find_template

    w, h = look["width"], look["height"]
    out = set()
    for name, (x, y, bw, bh) in tiles.items():
        region = ((x + 0.15 * bw) / w, (y - 0.62 * bh) / h, min(1.0, (x + 0.62 * bw) / w), (y - 0.25 * bh) / h)
        if find_template(png, Path(template), region, 0.85):
            out.add(name)
    return out


DO_MINUTES = 6.0  # the visit when the payload names none: the director ends it a drawn glance after the task


# --- the loop -----------------------------------------------------------------------------------------------------------------
@dataclass
class Task:
    goal: str
    slots: dict = field(default_factory=dict)
    scope: list[str] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    path: str | None = None
    task_id: str = "task"
    max_steps: int = MAX_STEPS
    on_write: object = None  # the runtime's journal hook (run/do.py Do)


class Runner:
    """`await Runner(task, planner, ...)(director)`: the task as a detour of the director's visit. Returns the result
    (status done | failed | unverified | operator, answer, writes, tier trace, path used/compiled) and journals every
    step as a `task_step` event in the visit's events.jsonl (tier, screen, op, target, outcome, decide_ms)."""

    def __init__(self, task: Task, planner: Planner | None, paths: PathStore | None = None, learner=None, store=None,
                 compile: bool = True):
        self.t, self.planner, self.paths, self.learner, self.store = task, planner, paths, learner, store
        self.compile = compile
        self.rows: list[dict] = []
        self.answers: list[str] = []
        self.memory = ""
        self.used: LearnedPath | None = None  # the Learned path this run started on
        self.diverged = False

    def diverge(self, d, idx: int, why: str) -> str:
        self.diverged = True
        d.ev("task_diverged", path=self.used.id if self.used else None, step=idx, why=why)
        return f"the known route broke at its step {idx + 1} ({why}); continue from this screen"

    def known(self, a: Action) -> Action:
        """The planner's expectation counts only when it names a Screen the store has (a guessed id is no check)."""
        a.expect = tuple(e for e in a.expect if e in self.actor.store.records)
        return a

    async def decide(self, look: dict, note: str, escalate: bool) -> tuple[Action, float]:
        t0 = time.perf_counter()
        rec = self.actor.store.records.get(look.get("screen") or "")
        named = [f"{n} (→ {', '.join(rec.next[n])})" if rec is not None and n in rec.next else n for n in sorted(look.get("targets", {}))]
        args = (look.get("png") or b"", self.t.goal, self.actor.slots, look, named, self.memory, note)
        ans = await asyncio.to_thread(self.planner.next, *args, escalate=escalate)  # off the daemon's loop: looks and the stream go on
        a = self.known(Action.from_planner(ans))
        if a.confidence < CONFIDENT and not escalate and a.op != "done":  # unsure: the stronger model, same frame
            ans = await asyncio.to_thread(self.planner.next, *args, escalate=True)
            a = self.known(Action.from_planner(ans))
        self.memory = ans.get("memory") or self.memory
        return a, round((time.perf_counter() - t0) * 1000)

    async def __call__(self, d) -> dict:
        t = self.t
        actor = self.actor = TaskActor(d, store=self.store, scope=t.scope, budget=t.budget, slots=t.slots, on_write=t.on_write)
        if self.learner is None and self.planner is not None:
            from ..screens.learn import Learner
            self.learner = Learner(self.planner.vlm, actor.store)
        d.ev("task_start", goal=t.goal, slots=t.slots, scope=t.scope, budget=t.budget, task_id=t.task_id)
        idx, fails, last_fail, escalate, note, seen, assist = 0, {}, None, False, "", None, None
        try:
            for n in range(t.max_steps):
                look = await actor.look()
                screen = look.get("screen")
                if screen in LOOK_AT and screen != seen:  # people look at a profile or a post before acting on it (§4.4)
                    t_dwell = d.now()
                    await actor.act(Action("dwell"), look)
                    d.ev("task_dwell", screen=screen, s_spent=round(d.now() - t_dwell, 2))
                    look = await actor.look()
                    screen = look.get("screen")
                seen = screen
                if n == 0 and self.paths is not None and (m := self.paths.match(t.goal, d.app, screen, t.path)):
                    self.used, slots = m
                    actor.slots.update({k: v for k, v in slots.items() if k not in actor.slots})
                    d.ev("task_path", path=self.used.id, slots=slots)
                if not screen:  # T2: unfamiliar
                    res = await self.t2(d, actor, look)
                    self.row(n, 2, None, Action("learn", tier=2), Outcome(res.status == "resolved", res.reason, res.screen_id, res.status == "resolved"), 0)
                    if res.status == "stop" and res.reason.split(":")[0] in LEARNER_STOPS:  # an action block, a login, 2FA...: the platform
                        return self.finish(d, {"status": "operator", "reason": f"an unfamiliar screen: {res.reason}", "stopped_by": "stop_screen"})
                    if res.status == "stop" or (res.status != "resolved" and fails.setdefault("?", 0) >= 1):
                        return self.finish(d, {"status": "operator", "reason": f"an unfamiliar screen: {res.reason}"})
                    fails["?"] = fails.get("?", 0) + (res.status != "resolved")
                    continue
                a, decide_ms = None, 0.0
                path = self.used if not self.diverged else None
                if path is not None and idx < len(path.steps):  # T0
                    st = path.steps[idx]
                    if st.at != screen:
                        note = self.diverge(d, idx, f"expected {st.at}, on {screen}")
                    else:
                        a = step_action(st, idx, actor.slots)
                        if a.op == "tap" and actor.resolve(a, look) is None:
                            if "[" in a.target:  # a selector this look can't decide (no pin template yet): the planner picks
                                assist = idx    # this one control, the route goes on after it (not a divergence)
                                a, note = None, f"the known route's next step here: {st.do} ({', '.join(st.expect)} next)"
                                d.ev("task_assist", path=self.used.id, step=idx, why=f"{st.do} unresolved on {screen}")
                            else:
                                a, note = None, self.diverge(d, idx, f"{st.do} does not resolve on {screen}")
                elif path is not None:  # the route is walked: is the task visibly done?
                    if (done := self.done(actor, look, Action("done", text=self.answers[-1] if self.answers else "", tier=0))) is not None:
                        return self.finish(d, done)
                    note = self.diverge(d, idx, "the route ended but the task is not visibly done")
                if a is None:  # T1
                    if self.planner is None:
                        return self.finish(d, {"status": "operator", "reason": "no path step and no planner"})
                    try:
                        a, decide_ms = await self.decide(look, note, escalate)
                    except Exception as e:  # noqa: BLE001 - the plan's limit, no claude CLI, a cap: no guess, Ian decides (T3)
                        return self.finish(d, {"status": "operator", "reason": f"the planner is unavailable: {type(e).__name__}: {str(e)[:160]}"})
                    escalate, note = False, ""
                if a.op == "ask_operator":
                    self.row(n, a.tier, screen, a, Outcome(False, a.reason), decide_ms)
                    return self.finish(d, {"status": "operator", "reason": f"the planner asks the operator: {a.reason}"})
                if a.op == "done":
                    done = self.done(actor, look, a)
                    self.row(n, a.tier, screen, a, Outcome(done is not None, "done" if done else "not done"), decide_ms)
                    if done is not None:
                        return self.finish(d, done)
                    note = "you said done, but " + ("the task's write is not done yet" if t.scope and not self.wrote() else
                                                    "your answer is not on the screen") + ": look again"
                    continue
                out = await actor.act(a, look, n)
                self.row(n, a.tier, screen, a, out, decide_ms)
                if self.planner is not None:
                    self.planner.history.append(Turn(screen, a.op, out.target or a.target or (f"item {a.item}" if a.item >= 0 else ""), out.why))
                if a.op == "read" and out.ok and out.value:
                    self.answers.append(out.value)
                if out.sent and out.why == "landed on an unfamiliar screen":
                    idx += a.tier == 0  # the next step's `at` decides once T2 has dealt with what covers the screen
                    continue
                if a.write and out.sent and not out.ok:  # never retry an uncertain write (§3.4 rule 3)
                    return self.finish(d, {"status": "unverified", "reason": f"{a.write} sent but not verified on the next look"})
                if out.ok:
                    idx += a.tier == 0 or (assist == idx and out.screen in path_expect(self.used, idx))
                    assist, last_fail = None, None
                    continue
                if out.why.startswith("refused"):
                    note = out.why
                    if len(actor.refused) >= 3:
                        return self.finish(d, {"status": "failed", "reason": f"the planner kept choosing refused actions: {out.why}"})
                    continue
                if a.tier == 0:  # a path step that failed: the planner takes over from this state
                    note = self.diverge(d, idx, out.why)
                    continue
                fails[screen] = fails.get(screen, 0) + 1
                stuck, last_fail = last_fail == (screen, a.sig), (screen, a.sig)
                if fails[screen] >= 3:
                    return self.finish(d, {"status": "operator", "reason": f"three failed steps on {screen}: {out.why}"})
                escalate = fails[screen] >= 2 or stuck  # the potentially-stuck rule: the stronger model, told what failed
                note = f"your last step here failed ({out.why}); choose differently" if escalate else f"that did not work ({out.why})"
        except Stopped as e:
            e.task_result = self.finish(d, {"status": "operator", "reason": f"{e.kind}: {e.reason}", "stopped_by": e.kind})
            raise
        except Exception as e:  # a park, a lost phone, a guard: the task is journaled and counted, then the visit handles it
            e.task_result = self.finish(d, {"status": "failed", "reason": f"{type(e).__name__}: {str(e)[:200]}", "stopped_by": type(e).__name__})
            raise
        return self.finish(d, {"status": "failed", "reason": f"not done in {t.max_steps} steps"})

    async def t2(self, d, actor: TaskActor, look: dict):
        """T2 (Q3): the Learner names this unfamiliar screen (a plain app screen is named and the task goes on from it; a
        sheet or popup is dismissed) and saves a Learned response at once."""
        from ..screens.cascade import Cascade, Frame
        from ..screens.engine import Resolution

        if self.learner is None:
            return Resolution("stop", None, "an unfamiliar screen and no learner")
        frame = Frame(look.get("items") or [], look.get("png"), look["width"], look["height"])
        flow = type("TaskFlow", (), {"platform": d.app, "bundle": BUNDLES.get(d.app, "")})()
        ctx = (f"a task is in progress ({self.t.goal}). If this is an ordinary screen of the app (not a popup, sheet or "
               "dialog), name it and choose action none so the task can go on from it.")
        res = await self.learner.resolve(_LearnerEngine(d, actor, self.t.task_id), frame, flow, ctx)
        actor.cascade = Cascade(actor.store)  # the learned record joins this task's recognition at once
        return res

    def wrote(self) -> bool:
        """Every write the task declared is verified, or was there before it came."""
        done = {w["action"] for w in self.actor.writes if w.get("verified")} | self.actor.already
        return set(self.t.scope) <= done

    def done(self, actor: TaskActor, look: dict, a: Action) -> dict | None:
        """The task's post-condition on this frame: a write the task declared is verified (or was already there); an
        answer is on this screen or was read on an earlier one."""
        if self.t.scope and not self.wrote():
            if not actor.shows_handle(look):
                return None
            actor.already |= {w for w in self.t.scope if write_state(actor.app, w, look, None) is True}  # done before this task came
            if not self.wrote():
                return None
        if a.text and a.text not in self.answers:
            if not any(norm(a.text) in norm(i.get("text", "")) for i in look.get("items") or []):
                if not self.t.scope:
                    return None  # a read task's answer must be on its frame
                a.text = ""  # a write task's post-condition is its write; a summary line in `done` is no answer
        if a.text and a.text not in self.answers:
            sel = read_selector(look, a.text, actor.store, actor.slots)  # the answer's place, for the compiled path's read step
            self.answers.append(a.text)
            self.row(len(self.rows), a.tier, look.get("screen"), Action("read", target=sel or "", tier=a.tier),
                     Outcome(sel is not None, "read", look.get("screen"), sel is not None, target=sel or ""), 0)
        return {"status": "done", "answer": a.text or (self.answers[-1] if self.answers else ""), "screen": look.get("screen"),
                "frame_seq": look.get("frame_seq")}

    def row(self, n: int, tier: int, screen, a: Action, out: Outcome, decide_ms: float) -> None:
        r = {"n": n, "tier": tier, "screen": screen, "op": a.op, "target": out.target or a.target, "item": a.item if a.item >= 0 else None,
             "text": a.text or None, "write": a.write or None, "expect": list(a.expect) or None, "outcome": out.why, "ok": out.ok,
             "reached": out.reached, "sent": out.sent, "screen_after": out.screen, "decide_ms": decide_ms, "path_step": a.step,
             "confidence": a.confidence if a.tier == 1 else None, "reason": a.reason[:200] or None}
        self.rows.append(r)
        self.actor.d.ev("task_step", **r)

    def finish(self, d, res: dict) -> dict:
        t, actor = self.t, self.actor
        tiers = [r["tier"] for r in self.rows if r["op"] != "done"]
        ok = res["status"] == "done"
        res = {"goal": t.goal, "slots": actor.slots, "scope": t.scope, **res, "writes": actor.writes, "refused": actor.refused,
               "answers": self.answers, "tier_trace": tiers, "steps": len(self.rows), "inputs": actor.n,
               "t0_share": round(tiers.count(0) / len(tiers), 3) if tiers else None, "planner_calls": getattr(self.planner, "calls", 0),
               "path": self.used.id if self.used else None, "diverged": self.diverged if self.used else None}
        if self.paths is not None and self.used is not None:
            self.paths.count(self.used, ok and not self.diverged)
        if ok and self.compile and self.paths is not None and (self.used is None or self.diverged):
            steps = []
            for k, r in enumerate(self.rows):
                if r["op"] in ("learn", "done"):
                    continue
                if r["outcome"] == "landed on an unfamiliar screen":  # where it landed is where the task went on after T2
                    after = next((x["screen"] for x in self.rows[k + 1:] if x["op"] != "learn" and x["screen"]), None)
                    steps.append({**r, "reached": after is not None, "expect_seen": [after] if after else []})
                elif r["ok"]:
                    steps.append({**r, "expect_seen": [r["screen_after"]] if r["screen_after"] and r["op"] not in ("read", "dwell") else []})
            p = compile_path(t.goal, actor.slots, d.app, steps, t.scope, t.task_id, {"handle": "handle", "tag": "tag"},
                             verify={"write": t.scope[0]} if t.scope else {"answer": True})
            if p is not None:
                if p.id in self.paths.paths:  # the route as it runs now: its totals kept, its failure streak reset
                    p.counts = {**self.paths.paths[p.id].counts, "streak": 0}
                self.paths.save(p)
                res["compiled"] = p.id
                d.ev("task_compiled", path=p.id, steps=len(p.steps))
        d.ev("task_result", **{k: v for k, v in res.items() if k != "writes"}, n_writes=len(actor.writes))
        return res


def path_expect(p: LearnedPath | None, idx: int) -> tuple[str, ...]:
    return tuple(p.steps[idx].expect) if p is not None and idx < len(p.steps) else ()


def step_action(st, idx: int, slots: dict) -> Action:
    op, _, arg = st.do.partition(":")
    a = Action(op, tier=0, expect=tuple(st.expect), write=st.write or "", step=idx)
    if op == "tap":
        a.target = arg
    elif op == "type":
        a.target, a.text = "field", str(slots.get(arg, ""))
    elif op == "read":
        a.target = "" if arg == "screen" else arg
    return a


LEARNER_STOPS = ("captcha", "2fa", "payment", "warning", "action_block", "login_wall")  # screens/learn.py STOP_KINDS values
BUNDLES = {"instagram": "com.burbn.instagram", "tiktok": "com.zhiliaoapp.musically", "x": "com.atebits.Tweetie2"}


class _LearnerEngine:
    """The screens Learner's engine surface over the director (no second input path): recognise with the task's
    cascade, settle = a fresh look, tap/swipe = the director's gestures, journal = the visit's events."""

    def __init__(self, d, actor: TaskActor, run_id: str):
        self.d, self.actor, self.run_id, self.app_version = d, actor, run_id, None
        self.udid = d.device
        self.phone = self
        self.journal = self

    def step(self, **kw) -> None:
        self.d.ev("task_t2", **{k: v for k, v in kw.items() if k != "answer"}, answer=(kw.get("answer") or {}).get("screen_name"))

    def recognise(self, frame, flow, expect=None):
        return self.actor.cascade.recognise(frame, platform=flow.platform)

    async def settle(self, timeout_s: float = 8, chrome=None):
        from ..screens.cascade import Frame

        reaction, _ = self.d.params.glide["reaction_s"].draw(self.d.rng)
        await self.d.sleep(reaction)
        look = await self.actor.look()
        return Frame(look.get("items") or [], look.get("png"), look["width"], look["height"])

    async def tap(self, x, y, w, h, target=None):
        """The Learner's tap, under the task's rules: only on an OCR line it names (never a model's box or a guessed
        point), re-observed and re-found on a fresh look, never a write; zero input otherwise (the learner then fails)."""
        fresh = await self.actor.look()
        hit = next((i for i in fresh.get("items") or [] if target and norm(i.get("text", "")) == norm(target)
                    and abs(i["x"] - x) < 0.1 * fresh["width"] and abs(i["y"] - y) < 0.05 * fresh["height"]), None)
        if hit is None or write_of(f"text:{hit['text'].strip()}", fresh.get("screen")):
            self.d.ev("task_t2_refused", target=target, why="no such line on a fresh look" if hit is None else "a write")
            return
        try:
            await self.d.tap(hit["x"], hit["y"], f"t2:{hit['text'].strip()}", kind="task", w=hit.get("w", 40), h=hit.get("h", 40), zone_exempt=True)
        except pr_refusals() as e:
            self.d.ev("task_t2_refused", target=target, why=f"{type(e).__name__}: zero input sent")

    async def swipe(self, udid, x0, y0, x1, y1, dur):
        try:
            await self.d.swipe(x0, y0, x1, y1, dur, "t2 scroll", kind="task")
        except pr_refusals() as e:
            self.d.ev("task_t2_refused", target="scroll", why=f"{type(e).__name__}: zero input sent")


# --- the task kind in the device runtime ----------------------------------------------------------------------------------------
def card(app: str) -> str:
    p = CARDS / app / "CARD.md"
    return p.read_text() if p.exists() else ""


def validate(payload: dict) -> Task:
    """The payload as a Task, or ValueError (the runtime's Refused): goal text, slots, a scope of known write classes,
    a budget only for scoped writes."""
    goal = payload.get("goal")
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 400:
        raise ValueError("do.goal must be a non-empty sentence (≤ 400 characters)")
    scope = list(payload.get("scope") or [])
    if bad := [s for s in scope if s not in WRITES]:
        raise ValueError(f"do.scope has unknown write classes {bad} (known: {', '.join(WRITES)})")
    if "post" in scope:
        raise ValueError("do.scope: posting goes through the post kind and its approval (phase 4), not a do task")
    if bad := [s for s in scope if s not in TASK_WRITES]:
        raise ValueError(f"do.scope {bad}: comments, replies and messages are not task writes yet (no composer read-back)")
    budget = {k: int(v) for k, v in (payload.get("budget") or {}).items()}
    if extra := [k for k in budget if k not in scope]:
        raise ValueError(f"do.budget names writes outside the scope: {extra}")
    slots = {k: str(v) for k, v in (payload.get("slots") or {}).items()}
    if payload.get("urgency", "next_session") not in ("now", "next_session"):
        raise ValueError("do.urgency is now | next_session")
    m = payload.get("minutes", DO_MINUTES)
    if isinstance(m, bool) or not isinstance(m, int | float) or not 0 < m <= MAX_TASK_MINUTES:
        raise ValueError(f"do.minutes must be in (0, {MAX_TASK_MINUTES}]: a task's visit is short")
    return Task(goal.strip(), slots, scope, budget, payload.get("path"))



# --- the kind in the device runtime (task protocol 2) ------------------------------------------------------------------------
MAX_TASK_MINUTES = 15  # a task's visit at most (it ends a drawn glance after the task anyway)
PLANNER_TIMEOUT_S = 30.0  # one planner answer at most: a dead tunnel ends the task 'operator' in seconds, the phone released
PLATFORM_STOPS = ("stop_screen", "Stop")  # Stop: taskapi's own (a stop text on screen, Watched._stops)
PLATFORM_REASONS = ("stop_screen", "phone: Stop")  # the visit's own end reason (after the task: the walk home, the closing glance)
PARKS = ("Park", "LeaseError", "LeaseUnavailable", "OperatorActive", "PhoneLocked", "NotFrontmost", "Parked", "TaskEnded")  # the director's Stopped kinds that are the platform pushing back (a stop hand-off, friction); the
# rest (an unknown screen, off the feed after the task, a planner that asks) end the task without holding the account


def _flow():
    """The `do` Flow, defined on taskapi's Browse (open the app, reach the feed on recognised screens, the visit, Home)."""
    import random as _random

    from . import taskapi

    class Do(taskapi.Browse):
        """One visit whose detour is the task (design §5.1: embedded, never teleported): orient, a few posts, the task
        (T0/T1/T2/T3), a glance, leave. A platform stop mid-task is a `stopped` task (the VM's stop hand-off); a task
        that asks the operator or whose write is unverified ends `done` with that status in its result (the VM hands it to
        Ian on Telegram without holding the account)."""
        kind = "do"
        v = 2
        steps = ("open", "task", "home")

        def __init__(self, task):
            pl = task.get("payload") or {}
            plat = task["identity"].split(":")[0]
            if plat in DO_APPS and plat not in taskapi.APPS:  # an app tasks reach before browse visits do (X: no unattended browse)
                taskapi.Flow.__init__(self, task)
                self.bundle, self.plat = DO_APPS[plat]
                self.minutes, self.app, self.knobs = float(pl.get("minutes", DO_MINUTES)), APP[plat], pl.get("settings") or {}
                if self.knobs.get("capture") is False:  # as Browse: the director never acts without seeing
                    raise taskapi.Refused("screen capture is off (operator.toml capture.screen = false): no visit")
            else:
                super().__init__({**task, "payload": {**pl, "minutes": pl.get("minutes", DO_MINUTES)}})
            try:
                self.t = validate(pl)
            except ValueError as e:
                raise taskapi.Refused(str(e)) from None
            self.t.task_id = task["task_id"]
            self.urgent = pl.get("urgency") == "now"

        async def step_task(self, ctx):
            if sent := [e for e in ctx.journal.events() if e.get("write_sent")]:  # resumed after a park: a write already went out
                raise taskapi.Stop(f"this task already sent {sent[-1]['write_sent'].get('action')}: never re-run (a person checks it)")
            self.t.on_write = lambda row: ctx.note(step="task", write_sent=row)
            ref = await self.to_feed(ctx)
            res = await self.visit(taskapi.Watched(ctx), ctx.device, app=self.app, identity=self.task["identity"], minutes=self.minutes,
                                   evidence=ctx.journal.path / "director", lease=ctx.lease, knobs=self.knobs, full_length=True,
                                   task=self.runner(), task_at=_random.randint(1, 2) if self.urgent else _random.randint(2, 6))
            ctx.data["visit"] = {k: res[k] for k in ("status", "reason", "label", "seconds", "writes", "counts", "digest") if k in res}
            t = ctx.data["task"] = res.get("task") or {"status": "not_started", "reason": res.get("reason") or "the visit ended first"}
            ctx.note(result=self.result(ctx))  # the task's writes reach the VM's pacer whatever ends this step
            by = t.get("stopped_by")
            if by in PLATFORM_STOPS or (res.get("status") != "ok" and str(res.get("reason") or "").startswith(PLATFORM_REASONS)):
                raise taskapi.Stop(f"the platform pushed back during the task: {t.get('reason') or res.get('reason')}")
            if by in PARKS:
                if t.get("writes"):
                    raise taskapi.Stop(f"parked after a write was sent ({t.get('reason')}): never re-run, a person checks it")
                raise taskapi.Park(f"the task parked: {t.get('reason')}")
            if t.get("status") in ("not_started", "started"):
                raise RuntimeError(f"the task did not run: {t.get('reason')}")
            return {"screen_id": f"{self.plat}_feed", "frame_ref": ref, "task_status": t.get("status"), "tiers": t.get("tier_trace")}

        def runner(self):
            from ..screens.planner import Planner
            from ..screens.vlm import VLM

            return Runner(self.t, Planner(VLM(timeout=PLANNER_TIMEOUT_S), card=card(self.app)), paths=PathStore())

        def result(self, ctx):
            return {**ctx.data.get("visit", {}), "task": ctx.data.get("task")}

    return Do


Do = _flow()
