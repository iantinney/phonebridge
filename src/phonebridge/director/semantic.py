"""The fast semantic cue (CONTROLLER_BAR L2/Q2): how close a post's on-screen text is to words IAN WROTE.

The reference is the account's niche file (niche.py): Ian's own statements plus the account's approved posts. Nothing
about his taste is inferred: saves, likes and follows are not taste data (Ian 2026-10-19 04:1xZ and 04:3xZ). Passive
sessions use no niche. Two on-device spaces: text (Apple NaturalLanguage sentence embeddings, about 5 ms per vector) and
image (Apple Vision feature prints, about 100 ms, off the glance path). Off the Mac the text side falls back to a
bag-of-words cosine and the image side is off, so tests and replays run anywhere. The score (0–1) maps to interest 0–3
through thresholds that are not fitted yet; the nearest reference line is named so the reason can cite it.

Known limits, to be replaced in v3 (DIRECTOR_GOAL_DRAFT.md, research: lanes/director-research-fast-classifier.md): the
score is a percentile against a background of feed posts, so some share of any feed always passes; there is no AGAINST
class; the thresholds are unfitted."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR

THRESHOLDS = (0.50, 0.60, 0.82)  # combined score → interest 1 / 2 / 3 (fit on his exports later, E5b)
WEIGHTS = {"text": 0.5, "tags": 0.3, "image": 0.2}  # the post's sentence vs his docs; its topic tags vs his tags; its media vs his frames
TAG_STOP = {"video", "photo", "post", "clip", "reel", "carousel", "tiktok", "instagram", "short-form-video", "talking-head"}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


BACKGROUND = (  # descriptions of other people's posts, written in the taste notes' own style (one sentence + tags), so the
    # yardstick has the same length and register as the taste docs; the embedding distance to them calibrates a post
    "A home cook plates lemon chicken with roasted vegetables and shares the marinade. tags: cooking, recipe, dinner",
    "A sports page cuts the best dunks from last night's basketball game to a trap beat. tags: basketball, highlights, nba",
    "A plant creator repots an overgrown monstera and explains root rot. tags: plants, gardening, home",
    "A travel vlogger walks a beach town in Portugal and ranks the pastry shops. tags: travel, europe, food",
    "A cat knocks a glass off the kitchen table and stares at the camera. tags: cats, funny, pets",
    "A makeup artist does a soft glam routine for a wedding guest in four minutes. tags: makeup, beauty, tutorial",
    "A car reviewer tests a new electric SUV's real-world range on a highway loop. tags: cars, ev, review",
    "A history teacher explains how the Roman empire paid its legions. tags: history, education, rome",
    "Two friends learn a dance challenge to a pop song in a parking lot. tags: dance, trend, music",
    "A reality TV couple argues about moving in together after the finale. tags: reality-tv, relationships, netflix",
    "A chess grandmaster explains the idea behind the London opening in two minutes. tags: chess, strategy, tutorial",
    "A street interviewer asks strangers what dating apps they use. tags: dating, street-interview, comedy",
    "A woodworker builds a plywood bookshelf with pocket screws and a jig. tags: woodworking, diy, build",
    "A stand-up comedian jokes about airport security lines. tags: comedy, stand-up, travel",
    "A nature channel follows a penguin colony through an Antarctic winter. tags: nature, animals, documentary",
    "A streamer wins a battle royale match with one health point left. tags: gaming, stream, fortnite",
    "A thrift haul shows five vintage jackets found for under twenty dollars. tags: fashion, thrift, style",
    "A baby bursts out laughing every time the dog sneezes. tags: baby, dog, wholesome",
    "A senator's speech argues for cutting payroll taxes for small firms. tags: politics, taxes, news",
    "A tech reviewer unboxes a new phone and tests its telephoto camera at night. tags: tech, phone, unboxing",
    "A gym coach demonstrates pull-up progressions from negatives to full reps. tags: fitness, calisthenics, workout",
    "A real estate agent tours a renovated loft and talks about the neighbourhood. tags: real-estate, home-tour, city",
    "A nutritionist meal-preps high-protein lunches for the week in one pan. tags: nutrition, meal-prep, health",
    "A photographer edits a sunset shot in Lightroom and explains the colour grade. tags: photography, editing, tutorial",
)
CHROME = re.compile(r"^(for you|following|friends|live|explore|shop|stem|your story|reels|home|search|profile|follow|more|sponsored|suggested for you|\d[\d.,:kmbh ]*|•+|\W+)$")


def clean(text_items: list[dict], h: float | None = None) -> str:
    """The post's own words: OCR lines minus the app's chrome (tab names, counters, bullets) and the top nav band."""
    out = []
    for i in text_items:
        t = _norm(i.get("text", ""))
        if not t or CHROME.match(t) or (h and i.get("y", 1e9) < 0.09 * h):
            continue
        out.append(t)
    return " ".join(out)


BACKGROUND_FILE = DATA_DIR / "director" / "taste" / "background.json"


def load_background(path: Path = BACKGROUND_FILE) -> tuple[str, ...]:
    """Real posts the feed served (described by the understanding model) as the yardstick, when the file exists; a random
    suggested post then scores at the median by construction and only posts closer to his taste than most of the feed pass
    the stay line (trial 1: every post 'touched his interests' against the generic sentences)."""
    try:
        items = json.loads(Path(path).read_text()).get("items", [])
        texts = tuple(i["text"] for i in items if i.get("text"))
        return texts if len(texts) >= 10 else BACKGROUND
    except Exception:
        return BACKGROUND


class TextSpace:
    """Similarity of a post's text to a taste document, 0–1: NaturalLanguage sentence embeddings on a Mac, calibrated as a
    percentile against the background (real feed posts when `background.json` exists, else neutral sentences; raw distances
    are not absolute: OCR noise sits at ~1.0 for everything); a stemmed-overlap score elsewhere."""

    def __init__(self, background: tuple[str, ...] | None = None):
        self.background = background or load_background()
        self.nl = None
        try:
            import NaturalLanguage as NL

            self.nl = NL.NLEmbedding.sentenceEmbeddingForLanguage_("en")
        except Exception:
            self.nl = None
        self.backend = "nl-sentence" if self.nl is not None else "bag-of-words"
        self.thresholds = THRESHOLDS if self.nl is not None else (0.25, 0.42, 0.60)  # the overlap score has its own scale
        self._vec: dict = {}

    def vector(self, text: str):
        """The sentence vector (512-d, ~5 ms), cached per text; None when the framework can't embed it."""
        import numpy as np

        v = self._vec.get(text)
        if v is None:
            raw = self.nl.vectorForString_(text[:500])
            if raw is None:
                return None
            v = np.asarray(list(raw), dtype=np.float32)
            n = float(np.linalg.norm(v))
            v = v / n if n else v
            self._vec[text] = v
        return v

    def _dist(self, a: str, b: str) -> float:
        va, vb = self.vector(a), self.vector(b)
        return 2.0 if va is None or vb is None else float(1.0 - va @ vb)  # cosine distance

    def similarity(self, a: str, b: str) -> float:
        if self.nl is not None:
            # the same post against the taste doc and against neutral sentences: the share of neutral sentences farther
            # from the post than this doc is. Controls for the post's own length and OCR noise, which move every distance.
            # One embedding per post (~5 ms); the docs' and background's vectors are cached.
            d = self._dist(a, b)
            bg = [self._dist(a, s) for s in self.background]
            return sum(1 for x in bg if x > d) / len(bg)
        wa, wb = self._bag(a), self._bag(b)
        if not wa or not wb:
            return 0.0
        # share of the query's content words that match a taste word by 5-letter stem (entrepreneur ~ entrepreneurship),
        # on the same 0–1 scale the sentence embeddings are mapped to
        stems_b = {w[:5] for w in wb}
        hit = sum(1 for w in wa if w[:5] in stems_b)
        return min(1.0, hit / max(3, len(wa)) * 1.5)

    @staticmethod
    def _bag(t: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for w in re.findall(r"[a-z][a-z0-9'-]{2,}", _norm(t)):
            if w not in {"the", "and", "for", "with", "that", "this", "you", "your", "are", "from", "more"}:
                out[w] = out.get(w, 0) + 1
        return out


class ImageSpace:
    """Vision feature-print distance between two PNGs (macOS); None elsewhere."""

    def __init__(self):
        self.ok = False
        try:
            import Vision  # noqa: F401

            self.ok = True
        except Exception:
            self.ok = False

    def print(self, png: bytes):
        """The frame's feature print as a unit vector (Vision's 768 float32s; pyobjc can't return computeDistance's
        out-parameter, so the distance is computed here)."""
        import numpy as np
        import Vision
        from Foundation import NSData

        req = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
        h = Vision.VNImageRequestHandler.alloc().initWithData_options_(NSData.dataWithBytes_length_(png, len(png)), {})
        ok, err = h.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(str(err))
        fp = req.results()[0]
        v = np.frombuffer(bytes(fp.data()), dtype=np.float32 if fp.elementType() == 1 else np.float64).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    @staticmethod
    def similarity(a, b) -> float:
        """Cosine of two feature prints, recentred: unrelated phone frames already sit at ~0.5 (same UI), so 0.5 → 0 and 0.9 → 1 [S]."""
        return max(0.0, min(1.0, (float(a @ b) - 0.5) / 0.4))


@dataclass
class Match:
    score: float
    interest: int
    nearest: str  # the taste item it resembles most
    text_sim: float
    image_sim: float | None
    parts: dict = field(default_factory=dict)


class TasteModel:
    def __init__(self, taste: dict, text: TextSpace | None = None, image: ImageSpace | None = None, thresholds=None):
        self.items = taste.get("items", [])
        self.source = taste.get("source", "")
        self.text = text or TextSpace()
        self.image = image or ImageSpace()
        self.thresholds = thresholds or self.text.thresholds
        self.docs = [f"{i.get('title', '')}. {i.get('summary', '')} {' '.join(i.get('tags', []))} {i.get('creator') or ''}" for i in self.items]
        self.prints = []
        self.print_errors: list[str] = []
        self.last_image_error = None
        if self.image.ok:
            for i in self.items:
                try:
                    self.prints.append(self.image.print(Path(i["frame"]).read_bytes()) if i.get("frame") else None)  # taste frames are the media itself
                except Exception as e:
                    self.print_errors.append(f"{i.get('frame')}: {e}"[:160])
                    self.prints.append(None)

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        taste = json.loads(path.read_text())
        for i in taste.get("items", []):  # frame paths are relative to taste.json (the corpus moves between VM and Mac)
            if i.get("frame") and not Path(i["frame"]).is_absolute():
                i["frame"] = str(path.parent / i["frame"])
            elif i.get("frame") and not Path(i["frame"]).exists():
                alt = path.parent / "frames" / Path(i["frame"]).name
                i["frame"] = str(alt) if alt.exists() else None
        return cls(taste)

    @property
    def topics(self) -> list[str]:
        """The taste as a short list for Ian to correct: top tags and creators."""
        c: dict[str, int] = {}
        for i in self.items:
            for t in i.get("tags", []):
                c[t] = c.get(t, 0) + 1
        return [t for t, _ in sorted(c.items(), key=lambda kv: -kv[1])][:12]

    @staticmethod
    def tag_overlap(topics: list[str], tags: list[str]) -> float:
        """Share of the post's topic tags that match one of the item's tags by 5-letter stem (mindset ~ mindset, entrepreneur ~
        entrepreneurship); format words don't count."""
        ts = [_norm(t) for t in topics if _norm(t) and _norm(t) not in TAG_STOP]
        if not ts or not tags:
            return 0.0
        stems = {_norm(t)[:5] for t in tags}
        return sum(1 for t in ts if t[:5] in stems or any(t[:5] in _norm(g) for g in tags)) / len(ts)

    def score(self, text: str, png: bytes | None = None, topics: list[str] | None = None) -> Match:
        if not self.items:
            return Match(0.0, 1, "", 0.0, None, {"note": "no taste data"})
        sims = [self.text.similarity(text, d) if text.strip() else 0.0 for d in self.docs]
        tags = [self.tag_overlap(topics or [], i.get("tags", [])) for i in self.items]
        img = None
        if png is not None and self.image.ok and any(p is not None for p in self.prints):
            try:
                from .cues import media_crop

                fp = self.image.print(media_crop(png))  # the post's media, not the app's chrome
                img = [self.image.similarity(fp, p) if p is not None else 0.0 for p in self.prints]
            except Exception as e:
                self.last_image_error = f"{type(e).__name__}: {e}"[:200]
                img = None
        w = dict(WEIGHTS)
        if not img:  # no image side: its weight goes to the text
            w["text"] += w["image"]
        combined = [w["text"] * s + w["tags"] * tags[k] + (w["image"] * img[k] if img else 0.0) for k, s in enumerate(sims)]
        k = max(range(len(combined)), key=lambda j: combined[j])
        score = combined[k]
        interest = sum(score >= th for th in self.thresholds)
        it = self.items[k]
        return Match(round(score, 3), interest, (it.get("title") or it.get("creator") or "")[:90], round(sims[k], 3), round(img[k], 3) if img else None,
                     {"text_backend": self.text.backend, "image": bool(img), "tag_overlap": round(tags[k], 2), "creator": it.get("creator"), "tags": it.get("tags", [])[:5]})


class InterestModel:
    """Two taste models in one score (Ian 01:0xZ): the account's NICHE (what it posts) and his personal taste. Intentional
    mode: niche first (stays and likes aim at the niche the posts serve); passive mode: personal taste first (realistic
    scrolling), the niche still counts a little."""

    WEIGHTS = {"intentional": (1.0, 0.5), "passive": (0.5, 1.0)}  # (niche, taste)

    def __init__(self, niche: TasteModel | None, taste: TasteModel | None, mode: str = "passive"):
        self.niche, self.taste, self.mode = niche, taste, mode if mode in self.WEIGHTS else "passive"
        self.items = (niche.items if niche else []) + (taste.items if taste else [])
        self.source = " + ".join(x.source for x in (niche, taste) if x)
        self.text = (niche or taste).text if (niche or taste) else TextSpace()
        self.image = (niche or taste).image if (niche or taste) else ImageSpace()
        self.thresholds = (niche or taste).thresholds if (niche or taste) else THRESHOLDS

    @property
    def topics(self) -> list[str]:
        return ((self.niche.topics if self.niche else []) + (self.taste.topics if self.taste else []))[:16]

    def score(self, text: str, png: bytes | None = None, topics: list[str] | None = None) -> Match:
        wn, wt = self.WEIGHTS[self.mode]
        best = None
        for model, w, label in ((self.niche, wn, "niche"), (self.taste, wt, "taste")):
            if model is None or not model.items:
                continue
            m = model.score(text, png, topics)
            scaled = min(1.0, m.score * w) if w < 1.0 else m.score
            if best is None or scaled > best.score:
                best = Match(round(scaled, 3), sum(scaled >= th for th in self.thresholds), m.nearest, m.text_sim, m.image_sim, {**m.parts, "via": label, "mode": self.mode})
        return best or Match(0.0, 1, "", 0.0, None, {"note": "no taste data"})
