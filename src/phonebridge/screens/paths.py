"""Learned paths (docs/design/task-agent.md §2, §7.3): a task's route saved as data, replayed model-free.

A path is a goal template with typed slots, a start Screen, and steps `{at, do, expect, write?}` in the director's
Step grammar plus `type:<slot>` and `read:<target>`; targets are the Screen records' names (or a selector resolved on
the live look: `text:{handle}` = the OCR item showing the slot's value, `tile[first_unpinned]`), never coordinates, and
no timings (drawn at run time). Hand-written paths live in the repo (`apps/feedagent/screens/<platform>/paths/`);
compiled ones are saved at once next to the Learned responses (`<corpus>/paths/<platform>/`), used at once, and
counted: two consecutive failures demote a path back to the planner, as for Learned responses (DECISIONS G3–G17)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .store import CORPUS, REPO_RECORDS

LEARNED = CORPUS / "paths"
DEMOTE_AFTER = 2  # consecutive failures
SLOT = re.compile(r"\{(\w+)\}")
LATEST = re.compile(r"\b(latest|newest|most recent|last)\b", re.I)
SLOT_TYPES = {"handle": r"@?(?P<{n}>[A-Za-z0-9._]{{1,30}})", "tag": r"#?(?P<{n}>[\w]{{2,60}})", "text": r"(?P<{n}>.+?)"}


@dataclass
class PathStep:
    at: str  # the Screen this step starts on (T0 only when the live look names it)
    do: str  # tap:<target> | type:<slot> | back | scroll | read:<target> | dwell:<what>
    expect: tuple[str, ...] = ()  # the Screens the step may land on; a miss is a divergence
    write: str | None = None  # the pacer action this tap spends (save, like, ...)

    def to_dict(self) -> dict:
        return {"at": self.at, "do": self.do, **({"expect": list(self.expect)} if self.expect else {}),
                **({"write": self.write} if self.write else {})}


@dataclass
class LearnedPath:
    id: str
    platform: str
    goal: str  # template: "find @{handle}'s latest post and save it"
    slots: dict[str, dict]
    steps: list[PathStep]
    scope: list[str] = field(default_factory=list)
    start: list[str] = field(default_factory=list)
    verify: dict = field(default_factory=dict)
    origin: dict = field(default_factory=dict)
    counts: dict = field(default_factory=lambda: {"success": 0, "failure": 0, "streak": 0})
    source: Path | None = None

    @property
    def demoted(self) -> bool:
        return self.counts.get("streak", 0) >= DEMOTE_AFTER

    def regex(self) -> re.Pattern:
        """The goal template as a pattern: slots become named groups of their type, the words around them match loosely
        (case, spacing, a trailing period)."""
        def lit(t: str) -> str:
            return r"\s+".join(re.escape(w).replace("@", "@?").replace("'", "['’]") for w in t.split(" "))
        out, pos = [], 0
        for m in SLOT.finditer(self.goal):
            kind = (self.slots.get(m.group(1)) or {}).get("type", "text")
            out += [lit(self.goal[pos:m.start()]), SLOT_TYPES.get(kind, SLOT_TYPES["text"]).format(n=m.group(1))]
            pos = m.end()
        return re.compile(r"^\s*" + "".join(out) + lit(self.goal[pos:]) + r"\s*\.?\s*$", re.I)

    def match(self, goal: str) -> dict | None:
        m = self.regex().match(goal)
        return {k: v for k, v in m.groupdict().items()} if m else None

    def to_dict(self) -> dict:
        return {"id": self.id, "platform": self.platform, "goal": self.goal, "slots": self.slots, "scope": self.scope,
                "start": self.start, "steps": [s.to_dict() for s in self.steps], "verify": self.verify, "origin": self.origin,
                "counts": self.counts}

    @classmethod
    def from_dict(cls, d: dict, source: Path | None = None) -> LearnedPath:
        steps = [PathStep(s["at"], s["do"], tuple(s.get("expect") or ()), s.get("write")) for s in d.get("steps") or []]
        for s in steps:
            if any(k in s.do for k in ("x=", "y=")) or re.search(r"\d+\s*,\s*\d+", s.do):
                raise ValueError(f"{d.get('id')}: a step names a coordinate ({s.do!r}); paths name targets only")
        return cls(d["id"], d["platform"], d["goal"], d.get("slots") or {}, steps, list(d.get("scope") or []),
                   list(d.get("start") or []), d.get("verify") or {}, d.get("origin") or {},
                   {"success": 0, "failure": 0, "streak": 0, **(d.get("counts") or {})}, source)


class PathStore:
    """Repo paths (hand-written) and learned ones (compiled); a learned path with a repo path's id loses to it."""

    def __init__(self, dirs: list[Path] | None = None, learned: Path = LEARNED):
        self.learned = Path(learned)
        self.dirs = [Path(d) for d in (dirs if dirs is not None else [REPO_RECORDS])]
        self.paths: dict[str, LearnedPath] = {}
        self.refresh()

    def refresh(self) -> None:
        self.paths = {}
        for root in [self.learned, *self.dirs]:  # repo last: it wins
            pattern = "*/*.yaml" if root == self.learned else "*/paths/*.yaml"
            for f in sorted(root.glob(pattern)) if root.exists() else ():
                try:
                    p = LearnedPath.from_dict(yaml.safe_load(f.read_text()), f)
                except Exception:  # noqa: BLE001 - a broken file is skipped, never fatal to a task
                    continue
                if root != self.learned and (c := self.learned / p.platform / f"{p.id.split('.', 1)[-1]}.counts.yaml").exists():
                    p.counts.update(yaml.safe_load(c.read_text()) or {})  # a repo path's counters live beside the learned ones
                self.paths[p.id] = p

    def match(self, goal: str, platform: str, screen: str | None = None, hint: str | None = None) -> tuple[LearnedPath, dict] | None:
        """The path for this goal (the hint first), not demoted, whose template matches and whose start is this screen."""
        cands = ([self.paths[hint]] if hint in self.paths else []) + [p for p in self.paths.values() if p.id != hint]
        for p in cands:
            if p.platform != platform or p.demoted:
                continue
            try:
                slots = p.match(goal)
            except re.error:  # a broken template is that path's problem, never every task's
                continue
            if slots is not None and (screen is None or not p.start or screen in p.start):
                return p, slots
        return None

    def save(self, p: LearnedPath) -> Path:
        """Learned paths only: written at once (atomic), used at once."""
        path = p.source if p.source and self.learned in p.source.parents else self.learned / p.platform / f"{p.id.split('.', 1)[-1]}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(p.to_dict(), sort_keys=False, allow_unicode=True))
        tmp.replace(path)
        p.source = path
        self.paths[p.id] = p
        return path

    def count(self, p: LearnedPath, ok: bool) -> None:
        """Success/failure counters (a repo path's counts are kept in the learned dir, never in the repo file)."""
        p.counts["success" if ok else "failure"] = p.counts.get("success" if ok else "failure", 0) + 1
        p.counts["streak"] = 0 if ok else p.counts.get("streak", 0) + 1
        if p.source is None or self.learned in p.source.parents:
            self.save(p)
        else:
            counts = self.learned / p.platform / f"{p.id.split('.', 1)[-1]}.counts.yaml"
            counts.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            counts.write_text(yaml.safe_dump(p.counts))


def generalise(text: str, slots: dict) -> str:
    """A concrete value back to its slot name (`@example` → `{handle}`), longest value first; case-insensitive."""
    out = text
    for k, v in sorted(slots.items(), key=lambda kv: -len(str(kv[1]))):
        if v:  # the whole value only (handle `la` is not in "latest")
            out = re.sub(r"(?<![\w.])" + re.escape(str(v).lstrip("@#")) + r"(?![\w])", "{" + k + "}", out, count=1, flags=re.I)
    return out


def compile_path(goal: str, slots: dict, platform: str, steps: list[dict], scope: list[str], task_id: str,
                 slot_types: dict | None = None, verify: dict | None = None) -> LearnedPath | None:
    """A successful task's journal → a Learned path: the steps whose `expect` held (a step that diverged and was redone
    is dropped), slot values replaced by slot names, numbered-item taps turned into text selectors (an item's number is
    per look; its text is not). `steps` rows: {screen, op, target, text, write, reached, tier}. None when nothing is
    reusable (no step, or a step on an unnamed screen)."""
    out = []
    for s in steps:
        if not s.get("reached") or not s.get("screen") or s["op"] in ("done", "ask_operator"):
            continue
        op, target = s["op"], s.get("target") or ""
        if op == "tap":
            if target.startswith("tile_") and s["screen"].endswith((".profile.other", ".profile")) and LATEST.search(goal):
                target = "tile[first_unpinned]"  # "the latest" on a profile grid: by meaning, never by position (pins vary)
            if target.startswith("text:") and s["screen"] == "x.profile" and LATEST.search(goal):
                target = "post[first_unpinned]"  # X's latest post: a post's text is content, its place in the list is the meaning
            do = f"tap:{generalise(target, slots)}" if target.startswith("text:") else f"tap:{target}"
        elif op == "type":
            slot = next((k for k, v in slots.items() if v and str(v).lstrip("@#").casefold() == (s.get("text") or "").lstrip("@#").casefold()), None)
            if slot is None:
                return None  # typed text that is no slot: not a template, nothing to save
            do = f"type:{slot}"
        elif op == "read":
            do = f"read:{target}" if target else "read:screen"
        elif op in ("back", "scroll", "scroll_up"):
            do = op
        else:
            continue
        out.append(PathStep(s["screen"], do, tuple(s.get("expect_seen") or ()), s.get("write") or None))
    if not out:
        return None
    template = generalise(goal, slots)
    name = re.sub(r"[^a-z0-9]+", "_", SLOT.sub(lambda m: m.group(1), template).casefold()).strip("_")[:60]
    return LearnedPath(f"{platform}.{name}", platform, template,
                       {k: {"type": (slot_types or {}).get(k, "handle" if k == "handle" else "tag" if k == "tag" else "text")} for k in slots},
                       out, list(scope), [out[0].at], verify or {},
                       {"task_id": task_id, "compiled": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "tier_trace": [s.get("tier") for s in steps if s.get("reached")]})
