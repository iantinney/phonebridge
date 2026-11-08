"""SCR-02, PER-01/03, DEV-05: the recognition cascade R1–R4 on one frame (PLAN §5.2).

R1 settle/change: a ¼-scale diff of the UI chrome between two frames ("tap did nothing", "still animating").
R2 expectation: the flow's `next` records first. R3 chrome fingerprint: dHash of the masked chrome regions,
nearest neighbour within `max_bits`. R4 OCR anchors: content-free words in their regions. A match needs a
margin over the runner-up, else `unknown`. "Unfamiliar" is judged on the chrome, never on the content area
(DECISIONS 18:10Z): a new video filling the screen is not a change and not a new screen.
"""

from __future__ import annotations

import io
import os
import re
import time
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageChops

from .. import ocr as vision
from .schema import CHROME_DEFAULT, Record, Region, Target

MARGIN = 0.25  # R4: score lead over the runner-up
DHASH_MARGIN = 3  # R3: bits between the nearest and the second-nearest record
CHANGE_EPS = 0.05  # R1: fraction of chrome pixels that may differ between "the same" frames (live IG Search idles at 0.02–0.03)
WORDS = re.compile(r"[^a-z0-9 ]+")
PROGRESS = re.compile(r"\s*\(\d+\)$")
ALERT_BUTTONS = {"allow", "don't allow", "dont allow", "ok", "cancel", "not now", "allow once",
                 "allow while using app", "allow full access", "select photos", "don't allow", "continue",
                 "ask app not to track", "allow once only", "keep current selection", "select photos..."}
HOME_APPS = ("settings", "photos", "camera", "clock", "mail", "calendar", "weather", "notes", "maps",
             "app store", "safari", "messages", "facetime", "reminders", "health", "wallet", "threads",
             "instagram", "tiktok", "youtube", "reddit")
SPRINGBOARD = "com.apple.springboard"


def norm(text: str) -> str:
    return WORDS.sub("", PROGRESS.sub("", text).casefold().replace("’", "'")).strip()


@dataclass
class Frame:
    """One look at the phone: cached OCR lines (pixels) and, when available, the pixels: a PNG, or a live image
    from the stream (`from_image`), whose PNG is encoded only when something asks for `png`."""
    ocr: list[dict]
    png: InitVar[bytes | None] = None  # read back through the lazy property below (`Frame.png = property(...)`)
    width: int = 1170
    height: int = 2532
    path: Path | None = None
    ts: float = field(default_factory=time.time)
    source: Image.Image | None = field(default=None, repr=False, compare=False)
    _png: bytes | None = field(default=None, repr=False)

    def __post_init__(self, png: bytes | None) -> None:
        if png is not None:
            self._png = png

    def _get_png(self) -> bytes | None:
        if self._png is None and self.source is not None:
            out = io.BytesIO()
            (self.source if self.source.mode == "RGB" else self.source.convert("RGB")).save(out, "PNG", compress_level=1)
            self._png = out.getvalue()
        return self._png

    def _set_png(self, value: bytes | None) -> None:
        self._png = value

    @classmethod
    def from_files(cls, png_path: Path, ocr_path: Path | None = None) -> "Frame":
        import json
        png_path = Path(png_path)
        ocr_path = ocr_path or png_path.with_suffix(".ocr.json")
        png = png_path.read_bytes()
        w, h = vision.png_size(png)
        items = json.loads(ocr_path.read_text()) if ocr_path.exists() else []
        return cls(ocr=items, png=png, width=w, height=h, path=png_path)

    @classmethod
    def from_png(cls, png: bytes, ocr: list[dict] | None = None) -> "Frame":
        w, h = vision.png_size(png)
        return cls(ocr=ocr or [], png=png, width=w, height=h)

    @classmethod
    def from_image(cls, im: Image.Image, ocr: list[dict] | None = None, ts: float | None = None) -> "Frame":
        """A frame straight from the live stream: no PNG round trip for the pixel layers or for Vision (which
        takes the image as a CGImage); the PNG exists once evidence or the slow model asks for it."""
        f = cls(ocr=ocr or [], width=im.width, height=im.height, ts=ts or time.time(), source=im)
        f.__dict__["image"] = im if im.mode == "L" else im.convert("L")
        return f

    @property
    def pixels(self):
        """What Vision reads: the live image when there is one, else the PNG bytes."""
        return self.source if self.source is not None else self.png

    @cached_property
    def image(self) -> Image.Image:
        if self.png is None:
            raise ValueError("frame has no pixels")
        return Image.open(io.BytesIO(self.png)).convert("L")

    @cached_property
    def rgb(self) -> Image.Image:
        return (self.source if self.source is not None else Image.open(io.BytesIO(self.png))).convert("RGB")

    @cached_property
    def sha(self) -> str:
        return sha256(self.png).hexdigest() if self.png else sha256(repr(self.ocr).encode()).hexdigest()

    @cached_property
    def appearance(self) -> str:
        return appearance(self) if self.pixels is not None else "light"

    def box(self, r: Region) -> tuple[int, int, int, int]:
        return (int(r[0] * self.width), int(r[1] * self.height), int(r[2] * self.width), int(r[3] * self.height))

    def lines(self, min_conf: float = 0.5) -> list[str]:
        return [i["text"] for i in self.ocr if i.get("conf", 1.0) >= min_conf]


Frame.png = property(Frame._get_png, Frame._set_png)  # a dataclass field can't be lazy; this one is


# --- pixels: R1 change, R3 dHash, PER-03 templates ---------------------------------------------

def _gray(png_or_image, scale: int = 4) -> Image.Image:
    im = png_or_image if isinstance(png_or_image, Image.Image) else Image.open(io.BytesIO(png_or_image))
    im = im.convert("L")
    return im.resize((max(1, im.width // scale), max(1, im.height // scale)), Image.BILINEAR)


def _mask(im: Image.Image, regions, w: int, h: int) -> Image.Image:
    """Paint masked regions (fractions of the full frame) mid-gray so they count for nothing."""
    if not regions:
        return im
    im = im.copy()
    sx, sy = im.width / w, im.height / h
    for r in regions:
        im.paste(128, (int(r[0] * w * sx), int(r[1] * h * sy), int(r[2] * w * sx), int(r[3] * h * sy)))
    return im


def dhash(png_or_image, region: Region = (0, 0, 1, 1), mask=(), size: tuple[int, int] = (9, 8)) -> int:
    """64-bit difference hash of a region (fractions) of the frame; `mask` regions are flattened first."""
    im = png_or_image if isinstance(png_or_image, Image.Image) else Image.open(io.BytesIO(png_or_image)).convert("L")
    w, h = im.size
    im = _mask(im, mask, w, h)
    crop = im.crop((int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h)))
    px = crop.resize(size, Image.BILINEAR).tobytes()
    cols, rows = size
    bits = 0
    for r in range(rows):
        for c in range(cols - 1):
            bits = (bits << 1) | (px[r * cols + c] > px[r * cols + c + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def changed(prev: bytes | Image.Image, cur: bytes | Image.Image, chrome=None, mask=(), scale: int = 4,
            threshold: int = 40) -> float:
    """R1: the fraction of ¼-scale pixels that differ by more than `threshold` inside the chrome regions
    (or the whole frame when `chrome` is None), masked regions excluded. Pillow does the arithmetic (~1 ms);
    a Python loop over the pixels took ~200 ms per pair."""
    a, b = _gray(prev, scale), _gray(cur, scale)
    if a.size != b.size:
        b = b.resize(a.size)
    w, h = a.size
    a, b = _mask(a, mask, w, h), _mask(b, mask, w, h)
    boxes = [(int(r[0] * w), int(r[1] * h), int(r[2] * w), int(r[3] * h)) for r in (chrome or [(0, 0, 1, 1)])]
    total = diff = 0
    for box in boxes:
        d = ImageChops.difference(a.crop(box), b.crop(box)).point(lambda v: 255 if v > threshold else 0)
        hist = d.histogram()
        total += sum(hist)
        diff += hist[255]
    return diff / total if total else 0.0


def find_template(png: bytes | Image.Image, template: Image.Image | Path | str, region: Region,
                  min_score: float = 0.8, scale: int = 4) -> tuple[int, int, float] | None:
    """PER-03: single-scale template match inside a region at 1/`scale` resolution, coarse (2x coarser) then
    refined. Score = 1 - mean |frame - template| / 255 (works for flat icons). Returns the centre in frame
    pixels and the score, or None under `min_score`."""
    im = png if isinstance(png, Image.Image) else Image.open(io.BytesIO(png))
    im = im.convert("L")
    tpl = template if isinstance(template, Image.Image) else Image.open(template)
    tpl = tpl.convert("L")
    W, H = im.size
    x0, y0, x1, y1 = (int(region[0] * W), int(region[1] * H), int(region[2] * W), int(region[3] * H))
    crop = im.crop((x0, y0, x1, y1))

    def search(reg: Image.Image, t: Image.Image, window=None):
        rw, rh, tw, th = reg.width, reg.height, t.width, t.height
        if tw > rw or th > rh:
            return None
        rp, tp = reg.tobytes(), t.tobytes()
        best = None
        xs = range(0, rw - tw + 1) if window is None else range(max(0, window[0]), min(rw - tw, window[1]) + 1)
        ys = range(0, rh - th + 1) if window is None else range(max(0, window[2]), min(rh - th, window[3]) + 1)
        for oy in ys:
            for ox in xs:
                s = 0
                for ty in range(th):
                    base = (oy + ty) * rw + ox
                    row = rp[base:base + tw]
                    trow = tp[ty * tw:(ty + 1) * tw]
                    s += sum(abs(p - q) for p, q in zip(row, trow))
                score = 1 - s / (tw * th * 255)
                if best is None or score > best[2]:
                    best = (ox, oy, score)
        return best

    coarse = scale * 2
    c = search(crop.resize((max(1, crop.width // coarse), max(1, crop.height // coarse))),
               tpl.resize((max(1, tpl.width // coarse), max(1, tpl.height // coarse))))
    if c is None:
        return None
    fine_reg = crop.resize((max(1, crop.width // scale), max(1, crop.height // scale)))
    fine_tpl = tpl.resize((max(1, tpl.width // scale), max(1, tpl.height // scale)))
    cx, cy = c[0] * 2, c[1] * 2
    f = search(fine_reg, fine_tpl, window=(cx - 3, cx + 3, cy - 3, cy + 3))
    if f is None or f[2] < min_score:
        return None
    return (x0 + int((f[0] + fine_tpl.width / 2) * scale), y0 + int((f[1] + fine_tpl.height / 2) * scale), round(f[2], 3))


# --- OCR anchors ---------------------------------------------------------------------------------

def ocr_find(items: list[dict], anchor: str, region: Region | None = None, width: int = 1170,
             height: int = 2532, min_conf: float = 0.5) -> dict | None:
    """The OCR line for `anchor` (confidence ≥ min_conf): an exact normalised line first, then a line
    containing it under PT's short-line rule; inside `region` (fractions) when given."""
    a = norm(anchor)
    raw = anchor.strip()
    box = None if region is None else (region[0] * width, region[1] * height, region[2] * width, region[3] * height)

    def inside(i):
        return box is None or (box[0] <= i["x"] <= box[2] and box[1] <= i["y"] <= box[3])

    cands = [i for i in items if inside(i)]
    if not a:  # a symbol like "+" or "←": raw equality only
        return next((i for i in cands if i["text"].strip() == raw), None)
    cands = [i for i in cands if i.get("conf", 1.0) >= min_conf]
    exact = [i for i in cands if norm(i["text"]) == a]
    if exact:
        return max(exact, key=lambda i: i.get("conf", 1.0))
    # PT's one rule (pacing_config.stop_reason): a 1–2 word phrase counts only on a short line (at most 3
    # other words); a phrase of 3+ words counts anywhere in a line
    n = len(a.split())
    slack = 1 if n == 1 else 3  # "Instagram v" still reads as Instagram; "About your account" is not "Account"
    sub = [i for i in cands if a in norm(i["text"]) and (n >= 3 or len(norm(i["text"]).split()) <= n + slack)]
    if sub:
        return min(sub, key=lambda i: (len(i["text"]), -i.get("conf", 1.0)))
    return None


def locate(frame: Frame, target: Target, templates_dir: Path | None = None) -> tuple[int, int, int, int, str] | None:
    """Where a target is on this frame: (x, y, w, h, how) with how in ocr|template|region|fallback."""
    reg = target.region
    for anchor in ([target.ocr] if target.ocr else []) + list(target.ocr_any):
        if i := ocr_find(frame.ocr, anchor, reg, frame.width, frame.height):
            return (int(i["x"]), int(i["y"]), int(i.get("w", 40)), int(i.get("h", 40)), "ocr")
    if target.template and frame.pixels is not None and templates_dir is not None:
        p = Path(templates_dir) / target.template
        if p.exists() and (hit := find_template(frame.image, p, reg or (0, 0, 1, 1), target.min_score)):
            tw, th = Image.open(p).size
            return (hit[0], hit[1], tw, th, "template")
    if target.template and target.fallback:
        return (int(target.fallback[0] * frame.width), int(target.fallback[1] * frame.height), 40, 40, "fallback")
    if reg and not (target.ocr or target.ocr_any or target.template):
        x0, y0, x1, y1 = frame.box(reg)
        return ((x0 + x1) // 2, (y0 + y1) // 2, x1 - x0, y1 - y0, "region")
    return None


def appearance(frame: Frame, chrome=None, by_share: bool = False) -> str:
    """light | dark from the chrome region (status bar excluded): the mean luminance of the top bar by default; with
    `by_share` (a record that declares its own chrome band), whether dark pixels outnumber light ones, so a dark-mode bar
    with a bright photo in it (IG's story tray, 10-19: mean 94, dark share 0.34) stays dark while a light title bar
    (the gates' swap/inverted confusers: dark share ≤ 0.01) does not. A declared chrome judges EVERY band by share, each
    one mostly dark (≥ 0.75; instagram.home: the status bar + the tab bar, the two bands a scrolled feed never covers; a bright photo
    under a scrolled-away top bar is content, while a swapped-in light screen leaves the status band light)."""
    im = _gray(frame.image, 8)
    w, h = im.size
    if by_share:
        for r in chrome or [(0, 0.05, 1, 0.12)]:
            px = im.crop((int(r[0] * w), int(r[1] * h), int(r[2] * w), int(r[3] * h))).tobytes()
            if sum(1 for v in px if v < 60) < 0.75 * len(px):
                return "light"
        return "dark"
    r = (chrome or [(0, 0, 1, 0.12)])[0]  # the top bar, below the status bar
    px = im.crop((int(r[0] * w), int(max(r[1], 0.05) * h), int(r[2] * w), int(r[3] * h))).tobytes()
    total, n = sum(px), len(px)
    return "dark" if n and total / n < 90 else "light"


def ink_share(frame: Frame, kind: str, region: Region) -> float:
    """The share of a region's pixels (at 60x20) that are `kind`: bright (white text/icons), cyan or red (the halves of
    TikTok's '+' button). A pixel-only anchor: it reads no text, so it holds when OCR garbles the chrome."""
    w, h = frame.rgb.size
    b = frame.rgb.crop((int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h))).resize((60, 20)).tobytes()
    px = list(zip(b[0::3], b[1::3], b[2::3]))
    hit = {"bright": lambda r, g, b: r > 225 and g > 225 and b > 225,
           "cyan": lambda r, g, b: r < 110 and g > 180 and b > 180,
           "red": lambda r, g, b: r > 200 and g < 90 and 60 < b < 140}[kind]
    return sum(1 for p in px if hit(*p)) / len(px)


@dataclass
class Score:
    ok: bool
    value: float
    hits: list[str]
    misses: list[str]


def score(record: Record, frame: Frame, templates_dir: Path | None = None, pixels: bool = True) -> Score:
    """R4 (+ templates): how well the record's anchors are on this frame. `ok` means every required anchor
    holds; `value` is the fraction of anchors found, so near-misses rank below full matches."""
    m, hits, misses, need = record.match, [], [], 0
    W, H = frame.width, frame.height
    if pixels and frame.pixels is not None and record.appearance in ("light", "dark") and record.kind == "screen":
        # measured on the record's own chrome, by dark-pixel share, when it declares one (tiktok.feed: its bottom bar, the top
        # band being the video; instagram.home: its top band, where the story tray's photos lift the mean over 90)
        seen = frame.appearance if tuple(record.chrome) == CHROME_DEFAULT else appearance(frame, record.chrome, by_share=True)
        if seen != record.appearance:  # overlays (popup/alert/stop) sit over either appearance
            return Score(False, 0.0, [], ["appearance"])
    for a in m.ocr_none:
        if ocr_find(frame.ocr, a, None, W, H):
            return Score(False, 0.0, [], [f"none:{a}"])
    ok = True
    for a in m.ocr_all:
        need += 1
        (hits if ocr_find(frame.ocr, a, None, W, H) else misses).append(a)
    ok &= len(misses) == 0
    any_hits = 0
    for a in m.ocr_any:
        need += 1
        if ocr_find(frame.ocr, a, None, W, H):
            hits.append(a)
            any_hits += 1
        else:
            misses.append(a)
    if m.ocr_any:
        ok &= any_hits >= m.at_least
    for a, reg in m.ocr_in:
        need += 1
        if ocr_find(frame.ocr, a, reg, W, H):
            hits.append(f"{a}@region")
        else:
            misses.append(f"{a}@region")
            ok = False
    if pixels and frame.pixels is not None:
        for name, file, reg, min_score in m.templates:
            need += 1
            p = Path(templates_dir or ".") / file
            if p.exists() and find_template(frame.image, p, reg, min_score):
                hits.append(f"tpl:{name}")
            else:
                misses.append(f"tpl:{name}")
                ok = False
        for name, kind, reg, lo, hi in m.inks:
            need += 1
            if lo <= ink_share(frame, kind, reg) <= hi:
                hits.append(f"ink:{name}")
            else:
                misses.append(f"ink:{name}")
                ok = False
    if m.chrome_dhash and pixels and frame.pixels is not None and not (m.ocr_any or m.ocr_all or m.ocr_in):
        reg, value, max_bits = m.chrome_dhash
        need += 1
        if hamming(dhash(frame.image, reg, record.mask), value) <= max_bits:
            hits.append("dhash")
        else:
            misses.append("dhash")
            ok = False
    return Score(ok and need > 0, (len(hits) / need) if need else 0.0, hits, misses)


@dataclass
class Match:
    screen_id: str | None
    layer: str  # R2 | R3 | R4 | none
    score: float = 0.0
    runner_up: str | None = None
    runner_up_score: float = 0.0
    ms: float = 0.0
    detail: str = ""

    @property
    def unknown(self) -> bool:
        return self.screen_id is None

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score


UNKNOWN = Match(None, "none")
GENERIC = ("ios",)
GENERIC_ALERT = "ios.alert.generic"  # matched by the system_alert() detector, never by its own anchors  # records that apply to every app (system popups, alerts, lock screen)


TEMPLATES = Path(os.environ.get("SCREENS_TEMPLATES", (Path(__file__).parent / "records" / "templates").resolve()))


class Cascade:
    def __init__(self, store, templates_dir: Path | None = None):
        self.store = store
        self.templates_dir = Path(templates_dir) if templates_dir else TEMPLATES

    def candidates(self, platform: str | None, kinds=None) -> list[Record]:
        recs = [r for r in self.store.records.values()
                if platform is None or r.platform == platform or r.platform in GENERIC]
        return [r for r in recs if kinds is None or r.kind in kinds]

    def recognise(self, frame: Frame, expect: list[str] | None = None, platform: str | None = None,
                  bundle: str | None = None) -> Match:
        t0 = time.perf_counter()
        if hasattr(self.store, "refresh"):
            self.store.refresh()  # a record learned by another process or host since the last look
        m = self._recognise(frame, expect, platform, bundle)
        m.ms = round((time.perf_counter() - t0) * 1000, 1)
        return m

    def _recognise(self, frame: Frame, expect, platform, bundle) -> Match:
        recs = self.candidates(platform)
        if bundle:
            recs = [r for r in recs if r.bundle in (bundle, "*") or r.platform in GENERIC]
        by_id = {r.id: r for r in recs}
        # stops and popups first: an overlay matches both itself and the screen under it, so the screen
        # margin would be small; a stop must win over everything.
        for kind in ("stop", "popup", "alert"):
            scored = sorted(((score(r, frame, self.templates_dir), r) for r in recs if r.kind == kind and r.id != GENERIC_ALERT),
                            key=lambda sr: -sr[0].value)
            ok = [(s, r) for s, r in scored if s.ok]
            if ok:
                s, r = ok[0]
                ru = ok[1] if len(ok) > 1 else None
                if ru is None or s.value - ru[0].value >= MARGIN or kind == "stop":
                    return Match(r.id, "R4", s.value, ru[1].id if ru else None, ru[0].value if ru else 0.0,
                                 detail=f"{kind}: {s.hits}")
        # the generic iOS system-alert class (SCR-04): a centred box with Allow / Don't Allow / OK / Cancel buttons
        # outranks the screen under it even when no record names that alert
        if GENERIC_ALERT in self.store.records and (text := system_alert(frame)) is not None:
            return Match(GENERIC_ALERT, "R4", 1.0, detail=f"alert: {text[:120]}")
        # R2: the expected screens
        if expect:
            exp = [by_id[i] for i in expect if i in by_id and by_id[i].kind == "screen"]
            scored = sorted(((score(r, frame, self.templates_dir), r) for r in exp), key=lambda sr: -sr[0].value)
            ok = [(s, r) for s, r in scored if s.ok]
            if ok and (len(ok) == 1 or ok[0][0].value - ok[1][0].value >= MARGIN):
                s, r = ok[0]
                ru = ok[1] if len(ok) > 1 else None
                return Match(r.id, "R2", s.value, ru[1].id if ru else None, ru[0].value if ru else 0.0,
                             detail=f"expected: {s.hits}")
        screens = [r for r in recs if r.kind == "screen"]
        # R3: chrome fingerprint, nearest neighbour
        if frame.pixels is not None:
            dists = []
            cache: dict[tuple, int] = {}
            for r in screens:
                if not r.match.chrome_dhash:
                    continue
                reg, value, max_bits = r.match.chrome_dhash
                key = (reg, tuple(r.mask))
                if key not in cache:
                    cache[key] = dhash(frame.image, reg, r.mask)
                d = hamming(cache[key], value)
                if not any(ocr_find(frame.ocr, a, None, frame.width, frame.height) for a in r.match.ocr_none):
                    dists.append((d, max_bits, r))
            dists.sort(key=lambda t: t[0])
            if dists and dists[0][0] <= dists[0][1]:
                d, _, r = dists[0]
                ru = dists[1] if len(dists) > 1 else None
                if ru is None or ru[0] - d >= DHASH_MARGIN:
                    return Match(r.id, "R3", 1 - d / 64, ru[2].id if ru else None, (1 - ru[0] / 64) if ru else 0.0,
                                 detail=f"dhash {d} bits")
        # R4: OCR anchors over the whole store
        scored = sorted(((score(r, frame, self.templates_dir), r) for r in screens), key=lambda sr: -sr[0].value)
        ok = [(s, r) for s, r in scored if s.ok]
        if not ok:
            best = scored[0] if scored else None
            return Match(None, "none", detail=f"best partial: {best[1].id} {best[0].hits}" if best else "no records")
        s, r = ok[0]
        ru = ok[1] if len(ok) > 1 else None
        if ru and s.value - ru[0].value < MARGIN:
            return Match(None, "none", s.value, ru[1].id, ru[0].value,
                         detail=f"no margin: {r.id} {s.value:.2f} vs {ru[1].id} {ru[0].value:.2f}")
        return Match(r.id, "R4", s.value, ru[1].id if ru else None, ru[0].value if ru else 0.0, detail=f"{s.hits}")


# --- the generic iOS system-alert class and the device-state classifier (SCR-04, DEV-05) ----------

def system_alert(frame: Frame) -> str | None:
    """A centred alert box with Allow / Don't Allow / OK / Cancel-style buttons: returns its text (the
    message, without the buttons) or None. Used after the records miss, so the task can park with a precise
    message naming the alert."""
    W, H = frame.width, frame.height
    buttons = [i for i in frame.ocr if norm(i["text"]) in ALERT_BUTTONS
               and 0.15 * W <= i["x"] <= 0.85 * W and 0.3 * H <= i["y"] <= 0.8 * H]  # alerts sit mid-screen; a page's bottom Continue is not one
    if not buttons:
        return None
    top_button = min(b["y"] for b in buttons)
    body = [i for i in frame.ocr if abs(i["x"] - W / 2) < 0.2 * W and 0.2 * H <= i["y"] < top_button
            and norm(i["text"]) not in ALERT_BUTTONS and i.get("conf", 1.0) >= 0.5]
    if not body:
        return None
    return " ".join(i["text"].strip() for i in sorted(body, key=lambda i: i["y"]))


def classify(frame: Frame, store, frontmost: str | None = None, cascade: Cascade | None = None) -> str:
    """DEV-05: locked | standby | home | app:<bundle> | system_dialog:<kind> | verification | login_wall |
    unknown. An app screen is never labelled without the frontmost bundle agreeing (when known)."""
    W, H = frame.width, frame.height
    texts = [i["text"].strip().casefold() for i in frame.ocr]
    if frame.pixels is not None:
        try:
            if vision.lock_screen(frame.png, frame.ocr):
                return "locked"
        except Exception:
            pass
    if any(h in t for t in texts for h in vision.LOCK_HINTS):
        return "locked"
    clocks = [i for i in frame.ocr if vision.CLOCK.fullmatch(i["text"].strip()) and i.get("h", 0) > 0.08 * H]
    if clocks and len(frame.ocr) <= 4:
        return "standby"
    if sum(1 for t in texts if t in HOME_APPS) >= 3 or (frontmost == SPRINGBOARD and
                                                          sum(1 for t in texts if t in HOME_APPS) >= 2):
        return "home"
    cascade = cascade or Cascade(store)
    m = cascade.recognise(frame, platform=None)
    if not m.unknown:
        r = store.records[m.screen_id]
        if r.kind == "stop":
            return "login_wall" if (r.stop_reason or "").startswith("login") else "verification"
        if r.kind == "alert":
            return f"system_dialog:{r.id}"
        if r.platform in GENERIC and r.kind == "popup":
            return f"system_dialog:{r.id}"
        if frontmost is None or r.bundle in (frontmost, "*"):
            return f"app:{r.bundle if r.bundle != '*' else frontmost}"
        return "unknown"
    if (text := system_alert(frame)) is not None:
        return "system_dialog:alert"
    return "unknown"
