"""SCR-01: one YAML record per screen, data not code (PLAN §5.1). Targets are anchor-relative (OCR text),
template-located (an icon file) or region-bounded; never bare coordinates. A `fallback` screen fraction may
exist only for template-located icons, and every use of it is logged by the engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

Region = tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the screen

KINDS = ("screen", "popup", "stop", "alert")
STATUSES = ("candidate", "confirmed")
ID = re.compile(r"^[a-z0-9]+(\.[a-z0-9_]+)+$")
BARE = {"x", "y", "at", "point", "px", "fx", "fy", "xy", "coord", "coords"}
CHROME_DEFAULT: tuple[Region, ...] = ((0.0, 0.0, 1.0, 0.12), (0.0, 0.88, 1.0, 1.0))  # top bar(s) and tab bar


class SchemaError(ValueError):
    pass


def region(v, what="region") -> Region:
    if not (isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(n, (int, float)) for n in v)):
        raise SchemaError(f"{what}: expected [x0, y0, x1, y1] fractions, got {v!r}")
    x0, y0, x1, y1 = (float(n) for n in v)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise SchemaError(f"{what}: fractions must satisfy 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1, got {v!r}")
    return (x0, y0, x1, y1)


def _strs(v, what) -> tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, str) or not all(isinstance(s, str) for s in v):
        raise SchemaError(f"{what}: expected a list of strings, got {v!r}")
    return tuple(v)


@dataclass(frozen=True)
class Target:
    name: str
    ocr: str | None = None
    ocr_any: tuple[str, ...] = ()
    template: str | None = None
    region: Region | None = None
    irreversible: bool = False
    fallback: tuple[float, float] | None = None
    via: str | None = None  # reached through another target (a "More options" sheet)
    min_score: float = 0.92  # template match threshold (mean-abs-diff score; white crops score high on white)

    @classmethod
    def from_dict(cls, name: str, d) -> "Target":
        if not isinstance(d, dict):
            raise SchemaError(f"target {name!r}: bare coordinates are not a target ({d!r}); use ocr/template/region")
        if bare := BARE & set(d):
            raise SchemaError(f"target {name!r}: bare coordinate keys {sorted(bare)}; use ocr/template/region")
        if unknown := set(d) - {f.name for f in fields(cls)} - {"name"}:
            raise SchemaError(f"target {name!r}: unknown keys {sorted(unknown)}")
        t = cls(name=name, ocr=d.get("ocr"), ocr_any=_strs(d.get("ocr_any"), f"target {name!r} ocr_any"),
                template=d.get("template"), region=region(d["region"], f"target {name!r}") if "region" in d else None,
                irreversible=bool(d.get("irreversible", False)),
                fallback=tuple(float(n) for n in d["fallback"]) if d.get("fallback") else None, via=d.get("via"),
                min_score=float(d.get("min_score", 0.92)))
        if not (t.ocr or t.ocr_any or t.template or t.region):
            raise SchemaError(f"target {name!r}: needs one of ocr, ocr_any, template, region")
        if t.fallback and not t.template:
            raise SchemaError(f"target {name!r}: a fallback fraction is only for template-located icons")
        if t.fallback and not (len(t.fallback) == 2 and all(0 <= n <= 1 for n in t.fallback)):
            raise SchemaError(f"target {name!r}: fallback must be [fx, fy] fractions")
        return t

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "name" and v not in (None, (), False) and not (k == "min_score" and v == 0.92)}
        for k in ("ocr_any", "region", "fallback"):
            if k in d:
                d[k] = list(d[k])
        return d


INK_KINDS = ("bright", "cyan", "red")  # cascade.ink_share: white text/icons, the cyan and red halves of TikTok's '+' button


@dataclass(frozen=True)
class MatchSpec:
    ocr_any: tuple[str, ...] = ()
    at_least: int = 1
    ocr_all: tuple[str, ...] = ()
    ocr_in: tuple[tuple[str, Region], ...] = ()
    ocr_none: tuple[str, ...] = ()
    chrome_dhash: tuple[Region, int, int] | None = None  # region, 64-bit value, max_bits
    templates: tuple[tuple[str, str, Region, float], ...] = ()  # name, file, region, min_score
    inks: tuple[tuple[str, str, Region, float, float], ...] = ()  # name, kind (bright|cyan|red), region, min_share, max_share

    @classmethod
    def from_dict(cls, d) -> "MatchSpec":
        if not isinstance(d, dict):
            raise SchemaError(f"match: expected a mapping, got {d!r}")
        if unknown := set(d) - {f.name for f in fields(cls)}:
            raise SchemaError(f"match: unknown keys {sorted(unknown)}")
        ocr_in = tuple((k, region(v, f"match.ocr_in[{k!r}]")) for k, v in (d.get("ocr_in") or {}).items())
        ch = None
        if (c := d.get("chrome_dhash")) is not None:
            try:
                ch = (region(c["region"], "chrome_dhash.region"), int(str(c["value"]), 16), int(c.get("max_bits", 4)))
            except (KeyError, TypeError, ValueError) as e:
                raise SchemaError(f"match.chrome_dhash: {e}") from e
        templates = tuple((t["name"], t["file"], region(t["region"], f"template {t.get('name')!r}"),
                           float(t.get("min_score", 0.8))) for t in (d.get("templates") or []))
        inks = tuple((t["name"], t["kind"], region(t["region"], f"ink {t.get('name')!r}"), float(t.get("min", 0.0)),
                      float(t.get("max", 1.0))) for t in (d.get("inks") or []))
        if bad := [i[0] for i in inks if i[1] not in INK_KINDS]:
            raise SchemaError(f"match.inks {bad}: kind must be one of {INK_KINDS}")
        m = cls(ocr_any=_strs(d.get("ocr_any"), "match.ocr_any"), at_least=int(d.get("at_least", 1)),
                ocr_all=_strs(d.get("ocr_all"), "match.ocr_all"), ocr_in=ocr_in,
                ocr_none=_strs(d.get("ocr_none"), "match.ocr_none"), chrome_dhash=ch, templates=templates, inks=inks)
        if not (m.ocr_any or m.ocr_all or m.ocr_in or m.chrome_dhash or m.templates or m.inks):
            raise SchemaError("match: a record needs at least one anchor (ocr_any/ocr_all/ocr_in/chrome_dhash/templates/inks)")
        if m.ocr_any and m.at_least > len(m.ocr_any):
            raise SchemaError(f"match: at_least {m.at_least} > {len(m.ocr_any)} ocr_any anchors")
        return m

    def to_dict(self) -> dict:
        d: dict = {}
        if self.ocr_any:
            d["ocr_any"], d["at_least"] = list(self.ocr_any), self.at_least
        if self.ocr_all:
            d["ocr_all"] = list(self.ocr_all)
        if self.ocr_in:
            d["ocr_in"] = {k: list(v) for k, v in self.ocr_in}
        if self.ocr_none:
            d["ocr_none"] = list(self.ocr_none)
        if self.chrome_dhash:
            r, v, b = self.chrome_dhash
            d["chrome_dhash"] = {"region": list(r), "value": f"{v:016x}", "max_bits": b}
        if self.templates:
            d["templates"] = [{"name": n, "file": f, "region": list(r), "min_score": s} for n, f, r, s in self.templates]
        if self.inks:
            d["inks"] = [{"name": n, "kind": k, "region": list(r), "min": lo, "max": hi} for n, k, r, lo, hi in self.inks]
        return d


@dataclass
class Record:
    id: str
    platform: str
    bundle: str
    match: MatchSpec
    kind: str = "screen"
    seen_versions: list[str] = field(default_factory=list)
    appearance: str = "light"
    locale: str = "en"
    mask: list[Region] = field(default_factory=list)
    chrome: list[Region] = field(default_factory=lambda: list(CHROME_DEFAULT))
    avoid: list[Region] = field(default_factory=list)  # tap-safe zones: controls a browsing controller must not touch
    header: Region | None = None  # where the visible post's author/handle line sits (the controller reads the handle there)
    targets: dict[str, Target] = field(default_factory=dict)
    checks: dict[str, dict] = field(default_factory=dict)
    next: dict[str, list[str]] = field(default_factory=dict)
    popups: list[str] = field(default_factory=list)
    way_out: list[str] = field(default_factory=list)
    timeout_s: float = 8.0
    evidence: list[str] = field(default_factory=list)
    status: str = "candidate"
    answer: str | None = None  # popup: the target to tap
    stop_reason: str | None = None  # stop: why input halts (captcha, 2fa, login, action_block, ...)
    source: str | None = None  # PT (pacing.toml), tracer, vlm, operator
    matches: list[str] = field(default_factory=list)  # independent successful runs (run id + frame hash)
    successes: int = 0  # Learned response counters (GLOSSARY): success = the expected next Screen appeared
    failures: int = 0  # consecutive failures (a record failing twice is demoted / routed to the operator)
    note: str | None = None

    @classmethod
    def from_dict(cls, d: dict, path: Path | None = None, strict: bool = True) -> "Record":
        """`strict=False` (records read from the shared live corpus) drops keys this schema does not know, so a
        newer lane's harvest never breaks an older tree's suite; repo records stay strict."""
        where = f"{path}: " if path else ""
        if not strict and isinstance(d, dict):
            known = {f.name for f in fields(cls)}
            dropped = sorted(set(d) - known)
            d = {k: v for k, v in d.items() if k in known}
            if dropped:
                d["note"] = ((d.get("note") or "") + f" | schema skipped unknown keys {dropped}").strip(" |")
        try:
            return cls._from(d)
        except SchemaError as e:
            raise SchemaError(f"{where}{e}") from None
        except (KeyError, TypeError, ValueError) as e:
            raise SchemaError(f"{where}{type(e).__name__}: {e}") from None

    @classmethod
    def _from(cls, d: dict) -> "Record":
        if not isinstance(d, dict):
            raise SchemaError("a record is a mapping")
        if unknown := set(d) - {f.name for f in fields(cls)}:
            raise SchemaError(f"unknown keys {sorted(unknown)}")
        for k in ("id", "platform", "bundle", "match"):
            if k not in d:
                raise SchemaError(f"missing {k!r}")
        if not ID.match(d["id"]):
            raise SchemaError(f"id {d['id']!r}: content-free dotted lowercase id expected (platform.area.name)")
        kind, status = d.get("kind", "screen"), d.get("status", "candidate")
        if kind not in KINDS:
            raise SchemaError(f"kind {kind!r} not in {KINDS}")
        if status not in STATUSES:
            raise SchemaError(f"status {status!r} not in {STATUSES}")
        targets = {n: Target.from_dict(n, t) for n, t in (d.get("targets") or {}).items()}
        r = cls(id=d["id"], platform=d["platform"], bundle=d["bundle"], match=MatchSpec.from_dict(d["match"]),
                kind=kind, seen_versions=[str(v) for v in d.get("seen_versions") or []],
                appearance=d.get("appearance", "light" if kind == "screen" else "any"), locale=d.get("locale", "en"),
                mask=[region(m, "mask") for m in d.get("mask") or []],
                avoid=[region(m, "avoid") for m in d.get("avoid") or []],
                header=region(d["header"], "header") if d.get("header") else None,
                chrome=[region(m, "chrome") for m in d.get("chrome") or CHROME_DEFAULT], targets=targets,
                checks=dict(d.get("checks") or {}), next={k: list(v) for k, v in (d.get("next") or {}).items()},
                popups=list(d.get("popups") or []), way_out=list(d.get("way_out") or []),
                timeout_s=float(d.get("timeout_s", 8.0)), evidence=list(d.get("evidence") or []), status=status,
                answer=d.get("answer"), stop_reason=d.get("stop_reason"), source=d.get("source"),
                matches=list(d.get("matches") or []), successes=int(d.get("successes", 0)),
                failures=int(d.get("failures", 0)), note=d.get("note"))
        if r.kind == "stop" and not r.stop_reason:
            raise SchemaError(f"{r.id}: a stop record needs stop_reason")
        if r.kind == "popup" and r.answer not in r.targets:
            raise SchemaError(f"{r.id}: a popup record needs answer = one of its targets {sorted(r.targets)}")
        for k, v in r.next.items():
            if k not in r.targets and k not in ("swipe", "home", "back", "launch", "any", "paste", "type"):
                raise SchemaError(f"{r.id}: next[{k!r}] is not a target or a gesture")
        for k in r.checks:
            if not isinstance(r.checks[k], dict):
                raise SchemaError(f"{r.id}: check {k!r} must be a mapping")
        return r

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "platform": self.platform, "bundle": self.bundle}
        if self.kind != "screen":
            d["kind"] = self.kind
        if self.seen_versions:
            d["seen_versions"] = self.seen_versions
        if self.appearance != ("light" if self.kind == "screen" else "any"):
            d["appearance"] = self.appearance
        if self.locale != "en":
            d["locale"] = self.locale
        if self.mask:
            d["mask"] = [list(m) for m in self.mask]
        if self.avoid:
            d["avoid"] = [list(m) for m in self.avoid]
        if self.header:
            d["header"] = list(self.header)
        if tuple(self.chrome) != CHROME_DEFAULT:
            d["chrome"] = [list(m) for m in self.chrome]
        d["match"] = self.match.to_dict()
        if self.targets:
            d["targets"] = {n: t.to_dict() for n, t in self.targets.items()}
        for k in ("checks", "next", "popups", "way_out"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.timeout_s != 8.0:
            d["timeout_s"] = self.timeout_s
        if self.evidence:
            d["evidence"] = self.evidence
        d["status"] = self.status
        for k in ("answer", "stop_reason", "source", "note"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.matches:
            d["matches"] = self.matches
        if self.successes:
            d["successes"] = self.successes
        if self.failures:
            d["failures"] = self.failures
        return d

    @classmethod
    def load(cls, path: Path, strict: bool = True) -> "Record":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()), path=Path(path), strict=strict)

    def dump(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=110))
        return path

    @property
    def irreversible(self) -> bool:
        return any(t.irreversible for t in self.targets.values())


def load_records(dirs, lenient: tuple[Path, ...] = ()) -> dict[Path, Record]:
    """Every *.yaml under the given directories, schema-checked; errors name the file. Records under a `lenient`
    directory (the shared live corpus) may carry keys newer than this schema: they are dropped, not fatal."""
    out: dict[Path, Record] = {}
    lenient = tuple(Path(x).resolve() for x in lenient)
    for d in dirs:
        for p in sorted(Path(d).rglob("*.yaml")):
            if p.name in ("manifest.yaml", "labels.yaml", "steps.yaml"):
                continue
            strict = not any(str(p.resolve()).startswith(str(l)) for l in lenient)
            out[p] = Record.load(p, strict=strict)
    return out
