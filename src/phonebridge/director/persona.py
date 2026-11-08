"""A persona: the pools, rates and seed one account scrolls with (BEH-16). Lives in
`~/.config/agent-reach/personas/<id>/persona.json`; `calibration` is `pending` (research defaults), `partial`
(some pools measured) or `measured` (BEH-03's volumes met). Defaults are never a verbatim replay source: they are
draws from the research distributions that `humanize.py` and `config/pacing.toml` already carry.

Habit vs session (naturalness T8): the seed and pools are drawn once and persisted; a session's own RNG seed never redraws
them. The hand is not per account: `side` comes from the shared motor record (`<root>/_motor/<owner>.json`, one thumb for
every account on the phone), initialised once. `params` holds this account's overrides as written (validated on load by
params.normalize, saved verbatim); keys this version doesn't know are written back unchanged. `prepare` is the start path.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..config import CONFIG_DIR
from . import params as pm

ROOT = Path(os.environ.get("PHONEBRIDGE_PERSONAS", CONFIG_DIR / "personas")).expanduser()
APPS = ("instagram", "tiktok")
CLASSES = ("video", "photo", "ad", "suggested", "live")
# research defaults, seconds, (median, log-sigma, lo, hi): dwell per class. Sources: config/pacing.toml [instagram.dwell]
# / [tiktok.dwell] (photo 4.0/0.5, video 8.0/0.8), the TikTok donation study (24 % skipped before 20 %), and the
# 2026-10-18 live audit (ads and suggestions are skimmed). A6 §4.2 is not in this repo; these stand in for it and
# are flagged `calibration: pending` until the operator's recordings and exports replace them (E5/E5b).
DWELL_DEFAULTS = {
    "instagram": {"video": (8.0, 0.8, 1.0, 45.0), "photo": (4.0, 0.5, 0.8, 25.0), "ad": (1.2, 0.4, 0.4, 4.0),
                  "suggested": (1.5, 0.4, 0.4, 5.0), "live": (1.5, 0.5, 0.4, 8.0)},
    "tiktok": {"video": (7.0, 0.9, 0.8, 60.0), "photo": (4.0, 0.5, 0.8, 25.0), "ad": (1.3, 0.4, 0.4, 4.0),
               "suggested": (1.5, 0.4, 0.4, 5.0), "live": (1.5, 0.5, 0.4, 8.0)},
}
# flick rows (length as a fraction of screen height, duration s, onset s = glance before a skip-flick): medians and
# log-sigmas. Length: a feed flick on a 6.1" phone carries the next post in (the live quarter-screen flicks looked
# small); duration from Touchalytics flicks (~190 ms); onset from inter-stroke gaps (median ~1 s, heavy tail).
FLICK_DEFAULTS = {
    "instagram": {"length": (0.45, 0.35, 0.18, 0.70), "duration": (0.16, 0.35, 0.08, 0.30), "onset": (0.8, 0.6, 0.15, 4.0)},
    "tiktok": {"length": (0.60, 0.25, 0.30, 0.70), "duration": (0.14, 0.30, 0.08, 0.30), "onset": (0.7, 0.6, 0.15, 4.0)},
}
POOL_N = 300
# BEH-05 seed until his likes/follows/saves and data exports are read: his own content queue's example hashtags (gym, fitness)
# plus what the feed already shows the new account. Shown in every trace header so Ian can correct it.
INTEREST_SEED = {"topics": []}  # no seed: interest comes only from words Ian writes (intentions, the niche), never a guessed list (04:1xZ)
MEASURED_DWELLS, MEASURED_SESSIONS = 300, 3  # BEH-03: ≥ 300 dwells per app over ≥ 3 sessions
MOTOR = "owner"  # the one person whose thumb drives every account on this phone (naturalness T8b)


def _lognormal(spec, rng, n):
    med, sig, lo, hi = spec
    return [round(min(hi, max(lo, med * math.exp(rng.gauss(0, sig)))), 4) for _ in range(n)]


@dataclass
class Persona:
    id: str
    seed: int
    side: int  # which side the thumb's swipes bow to; live, `prepare` sets it from the shared motor record (this copy only seeds it)
    calibration: str = "pending"
    dwell: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # app -> class -> seconds
    flicks: dict[str, list[list[float]]] = field(default_factory=dict)  # app -> [length_frac, duration_s, onset_s]
    habits: dict = field(default_factory=dict)  # legacy (habits.py deleted per BEHAVIOUR_MODEL §4); kept so saved persona.json files load; in-session rates live in params.py
    sources: dict[str, list[str]] = field(default_factory=dict)  # app -> where its pools came from
    sessions: dict[str, int] = field(default_factory=dict)  # app -> measured sessions behind the pools
    interest: dict = field(default_factory=lambda: copy.deepcopy(INTEREST_SEED))  # BEH-05's set, secret-free (a copy: never the module's list)
    exploration: dict = field(default_factory=lambda: {"chance": 0.05})  # per session: visit an unmapped part of the app (Ian, 19:3xZ); pending
    params: dict = field(default_factory=dict)  # this account's overrides as written ("engage.like.p": 0.02); params.load applies them
    motor: str = MOTOR  # the shared motor record this account's hand comes from
    traits: dict = field(default_factory=dict)  # this account's habits: {param path: multiplier}, drawn once, drifting slowly (pm.vary's habit layer)
    traits_date: str | None = None  # the day the traits last drifted (a second drift that day is a no-op)
    traits_ref: str = ""  # the drift's version and evidence (naturalness T8b)
    extra: dict = field(default_factory=dict, repr=False)  # keys this version doesn't know: written back unchanged, never dropped

    def save(self, root: Path = ROOT) -> Path:
        d = root / self.id
        d.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        _write(d / "persona.json", json.dumps(data.pop("extra") | data, indent=1, ensure_ascii=False))
        return d / "persona.json"

    @classmethod
    def load(cls, id: str, root: Path = ROOT) -> Persona:
        f = root / id / "persona.json"
        raw = pm.read_json(f)
        known = {x.name for x in fields(cls)} - {"extra"}
        p = cls(**{k: v for k, v in raw.items() if k in known}, extra={k: v for k, v in raw.items() if k not in known})
        pm.normalize(p.params, f"{f} params")  # pm.load's own normalizer: a bad override fails here, before anything runs or is rewritten
        if str(p.interest.get("source", "")).startswith("seed"):
            p.interest = copy.deepcopy(INTEREST_SEED)  # a guessed seed saved by an older version never comes back (Ian 04:1xZ: interest is written by him)
        return p

    def add_dwell(self, app: str, seconds: list[float], source: str, cls: str = "video", sessions: int = 1) -> None:
        """Measured dwells replace the research pool of that class (never mixed with the defaults)."""
        if not seconds:
            return
        pools = self.dwell.setdefault(app, {})
        pools[cls] = [round(float(s), 4) for s in seconds] if source not in self.sources.get(app, []) else pools.get(cls, []) + [round(float(s), 4) for s in seconds]
        self.sources.setdefault(app, []).append(source) if source not in self.sources.get(app, []) else None
        self.sessions[app] = self.sessions.get(app, 0) + sessions
        self._recalibrate()

    def add_flicks(self, app: str, rows: list[tuple[float, float, float]], source: str) -> None:
        if not rows:
            return
        tag = f"{source}:flicks"
        self.flicks[app] = ([] if tag not in self.sources.get(app, []) else self.flicks.get(app, [])) + [[round(float(v), 4) for v in r] for r in rows]
        if tag not in self.sources.setdefault(app, []):
            self.sources[app].append(tag)
        self._recalibrate()

    def _recalibrate(self) -> None:
        measured = {a for a, s in self.sources.items() if s}
        if not measured:
            self.calibration = "pending"
            return
        full = all(sum(len(v) for v in self.dwell.get(a, {}).values()) >= MEASURED_DWELLS and self.sessions.get(a, 0) >= MEASURED_SESSIONS
                   and any(t.endswith(":flicks") for t in self.sources.get(a, [])) for a in APPS)
        self.calibration = "measured" if full else "partial"


def default_persona(id: str, seed: int | None = None) -> Persona:
    """Research defaults, `calibration: pending`: pools drawn from the distributions above with the persona's seed."""
    seed = random.randrange(2**31) if seed is None else seed
    rng = random.Random(seed)
    dwell = {app: {cls: _lognormal(spec, rng, POOL_N) for cls, spec in specs.items()} for app, specs in DWELL_DEFAULTS.items()}
    flicks = {}
    for app, spec in FLICK_DEFAULTS.items():
        rows = []
        for _ in range(POOL_N):
            z = rng.gauss(0, 1)  # longer flicks take a little longer: a shared factor keeps the joint shape
            med, sig, lo, hi = spec["length"]
            length = min(hi, max(lo, med * math.exp(sig * (0.6 * z + 0.8 * rng.gauss(0, 1)))))
            med, sig, lo, hi = spec["duration"]
            duration = min(hi, max(lo, med * math.exp(sig * (0.4 * z + 0.9 * rng.gauss(0, 1)))))
            rows.append([round(length, 4), round(duration, 4), round(_lognormal(spec["onset"], rng, 1)[0], 4)])
        flicks[app] = rows
    return Persona(id=id, seed=seed, side=rng.choice((-1, 1)), dwell=dwell, flicks=flicks,
                   sources={app: [] for app in APPS})


def draw_traits(persona: Persona, paths: list[str], sd: float) -> dict:
    """An account's own habits, drawn once from its seed: exp(N(0, sd)) per parameter, truncated at ±2 sd."""
    out = {}
    for path in paths:
        z = random.Random(f"{persona.id}:{persona.seed}:trait:{path}").gauss(0, sd) if sd > 0 else 0.0
        out[path] = round(math.exp(max(-2 * sd, min(2 * sd, z))), 6)
    return out


def seed_from(base: Persona, id: str, seed: int | None = None) -> Persona:
    """A new account's persona, seeded once from `base`'s pools (public's measured and research pools) with its own draws:
    each pool resampled with this account's seed, so it keeps the base's distributions but not its exact rows."""
    seed = random.randrange(2**31) if seed is None else seed
    rng = random.Random(seed)
    dwell = {app: {cls: [rng.choice(v) for _ in range(len(v))] if v else [] for cls, v in by_cls.items()} for app, by_cls in base.dwell.items()}
    flicks = {app: [list(rng.choice(rows)) for _ in range(len(rows))] if rows else [] for app, rows in base.flicks.items()}
    return Persona(id=id, seed=seed, side=base.side, calibration=base.calibration, dwell=dwell, flicks=flicks,
                   sources={app: [f"seeded from {base.id}"] + list(v) for app, v in base.sources.items()}, sessions=dict(base.sessions),
                   interest=copy.deepcopy(base.interest), motor=base.motor)


DRIFT_REF = ("v1: x' = 0.98·x + N(0, drift_sd) in log space, once per account and day; evidence: within-person day-to-day CV of "
             "daily app minutes 0.4-0.5 (ABCD, docs/research/saves-and-session-variation.md) and the operator, DECISIONS Afternoon 10-19")


def drift(persona: Persona, day, sd: float, innovation: float | None = None) -> bool:
    """The habits' slow drift, once per account and day (naturalness T8b): x' = mu + 0.98·(x − mu) + ε in log space, mu = 0
    (the persona's own values), ε ~ N(0, sd) seeded by account + day (a re-run of the same day draws the same ε). False
    when it already drifted that day, or there is nothing to drift. `innovation`: a fixed ε (T8b's +0.01 test).""" 
    key = day.isoformat()
    if not persona.traits or persona.traits_date == key or sd <= 0:
        return False
    for path, v in list(persona.traits.items()):
        eps = innovation if innovation is not None else random.Random(f"{persona.id}:{persona.seed}:{key}:drift:{path}").gauss(0, sd)
        persona.traits[path] = math.exp(0.98 * math.log(v) + eps)  # full precision: the transition holds to 1e-12
    persona.traits_date, persona.traits_ref = key, DRIFT_REF
    return True


def load_or_default(id: str, root: Path = ROOT, seed: int | None = None) -> Persona:
    try:
        return Persona.load(id, root)
    except FileNotFoundError:
        return default_persona(id, seed)


def _write(path: Path, text: str) -> None:
    """Atomic: a crash mid-write leaves the old file, never half of one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def motor(id: str = MOTOR, root: Path | None = None, side: int | None = None) -> dict:
    """The shared motor record `<root>/_motor/<id>.json`: the owner's hand, the same for every account he scrolls on this
    phone (one thumb, not one per login: naturalness audit, T8b). Initialised once, from the first persona's persisted
    `side` (else one draw), then only read: no account's intention, rates or session seed changes it."""
    f = (root or ROOT) / "_motor" / f"{id}.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        _write(f, json.dumps({"id": id, "side": side if side in (-1, 1) else random.SystemRandom().choice((-1, 1))}, indent=1))
    rec = pm.read_json(f)
    if rec.get("id") != id or rec.get("side") not in (-1, 1):
        raise ValueError(f"{f}: a motor record needs id {id!r} and side -1 or 1, not {rec}")
    return rec


def prepare(id: str, root: Path | None = None, seed: int | None = None, operator_toml: Path | None = None, day=None,
            base: str = "public") -> tuple[Persona, pm.Params]:
    """The one start path (the CLI, the task API's director visit): the account's persona (created and saved once, BEH-16; a
    later seed never redraws it), its hand from the shared motor record, and the effective params (defaults → persona.json
    `params` → operator.toml `[director.params]`). Everything is validated first: a bad file raises ParamsError before any
    write or input."""
    root = root or ROOT
    new = not (root / id / "persona.json").exists()
    if new and id != base and (root / base / "persona.json").exists():  # a habit per account (Ian): seeded once from public's pools
        persona = seed_from(Persona.load(base, root), id, seed)
    else:
        persona = load_or_default(id, root, seed)
    params = pm.load(root / persona.id, operator_toml or pm.OPERATOR_TOML)
    persona.side = motor(persona.motor, root, persona.side)["side"]
    changed = new
    if id != base and not persona.traits:  # its own habits, drawn once (an older persona gets them the first time too)
        persona.traits, changed = draw_traits(persona, pm.trait_paths(params), params.habit_sd), True
        persona.traits_date = day.isoformat() if day is not None else None  # drawn today: today's habits (the drift starts tomorrow)
    if day is not None and drift(persona, day, params.drift_sd):  # the slow drift, once a day
        changed = True
    if changed:
        persona.save(root)
    return persona, params
