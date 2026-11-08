"""Fast local cues (PER-04): what is on screen decides skip vs stay within a latency budget.

Free and instant: OCR markers (Sponsored, Suggested for you, LIVE) from the look the bridge already made, and motion
(two screenshots differ: a video is playing). Optional: a classifier coroutine `classify(png) -> {cls, topics,
interest}` (Apple Vision on the Mac, see `vision_classifier`; MobileCLIP would plug in here) that must answer within
`budget_ms`; a late or failing classifier yields the content-blind state, and both outcomes are counted so the
share of decisions content changed is known.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable

BLIND = "blind"
BUDGET_MS = 200
MARKERS = {
    "ad": ("sponsored", "paid partnership", "promoted"),
    "suggested": ("suggested for you", "suggested reels", "suggested popular accounts", "follow suggestions", "accounts you might like"),
}
SKIP_CLASSES = ("ad", "suggested", "live")
GENERIC = {"document", "screenshot"}
FEED_MARKERS = {  # phrases only a feed shows, matched as whole lines or whole phrases (autopilot 2026-10-18: "follow" is in
    # "followers" on a profile and "Follow and invite friends" in Settings, so single common words are not markers)
    "instagram": ("your story", "suggested for you", "liked by", "suggested reels", "suggested popular accounts"),
    "tiktok": ("for you",),  # TikTok's top nav reads "Following Friends For You" (one or several OCR lines)
}
NOT_FEED = ("followers", "edit profile", "share profile", "settings", "account status", "add a comment", "add comment", "post", "log in", "sign up")


def on_feed(app: str, items: list[dict]) -> bool:
    """Sims and replays only (live gating is the daemon's look): a feed phrase on screen and none of the profile/settings/
    composer phrases."""
    texts = [_norm(i.get("text", "")) for i in items]
    if any(any(t == n or t.startswith(n + " ") or t.endswith(" " + n) for n in NOT_FEED) for t in texts):
        return False
    return any(any(t == m or f" {m} " in f" {t} " for t in texts) for m in FEED_MARKERS.get(app, ()))  # Vision says this of every phone screen (measured 2026-10-18); not a topic
Classifier = Callable[[bytes], Awaitable[dict]]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


NAV_TOP = 0.09  # the top band of the screen: status bar and the app's tab row (TikTok's "LIVE" tab lives there)


def markers(items: list[dict], h: float | None = None) -> set[str]:
    """ad / suggested / live from the OCR lines. A bare "LIVE" counts only below the top nav band (TikTok's For You
    page has a LIVE *tab* top-left on every post; the badge of a livestream sits lower); pass the screen height `h`."""
    out = set()
    for i in items:
        t = _norm(i.get("text", ""))
        for cls, phrases in MARKERS.items():
            if cls == "ad":  # the label line itself ("Sponsored", "Sponsored · Shop now", "Paid partnership with x"), never a caption that ends in it ("not sponsored", "I just got promoted")
                if any(t == p or re.match(rf"^{re.escape(p)}(\s*·|\s+with\s)", t) for p in phrases):
                    out.add(cls)
            elif any(t == p or t.startswith(p + " ") or t.endswith(" " + p) for p in phrases):
                out.add(cls)
        if t == "live" and (h is None or i.get("y", 0) > NAV_TOP * h):  # the badge alone; "I live in..." isn't one
            out.add("live")
    return out


@dataclass
class Cue:
    cls: str
    decision: str  # skip | stay
    source: str  # ocr | motion | classifier | late
    latency_ms: float
    topics: list[str] = field(default_factory=list)
    interest: int | None = None
    markers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class CueEngine:
    def __init__(self, classify: Classifier | None = None, budget_ms: float = BUDGET_MS):
        self.classify, self.budget_ms = classify, budget_ms
        self.counts: Counter = Counter()

    async def judge(self, items: list[dict], moving: bool, png: bytes | None = None, t0: float | None = None, h: float | None = None) -> Cue:
        start = time.perf_counter() if t0 is None else t0
        m = markers(items, h)
        if m & set(SKIP_CLASSES):
            cls = next(c for c in SKIP_CLASSES if c in m)
            self.counts["content"] += 1
            return Cue(cls, "skip", "ocr", (time.perf_counter() - start) * 1000, markers=sorted(m))
        cls, source, topics, interest = ("video", "motion", [], None) if moving else ("photo", "ocr", [], None)
        decision = "stay"
        if self.classify is not None and png is not None:
            try:
                r = await asyncio.wait_for(self.classify(png), timeout=max(0.0, self.budget_ms / 1000 - (time.perf_counter() - start)))
                cls = "video" if moving else str(r.get("cls", cls))
                topics, interest, source = list(r.get("topics", []))[:3], r.get("interest"), "classifier"
                if interest == 0:
                    decision = "skip"
                self.counts["content"] += 1
            except Exception:  # late (TimeoutError) or broken: the content-blind state, counted
                cls, source, topics, interest = BLIND, "late", [], None
                self.counts["late"] += 1
        else:
            self.counts["content" if m or moving else "blind"] += 1
        return Cue(cls, decision, source, (time.perf_counter() - start) * 1000, topics=topics, interest=interest, markers=sorted(m))

    def summary(self) -> dict:
        n = sum(self.counts.values())
        return {"n": n, "content": self.counts["content"], "late": self.counts["late"], "blind": self.counts["blind"],
                "content_share": round(self.counts["content"] / n, 3) if n else None}


def media_crop(png: bytes, band: tuple[float, float] = (0.12, 0.72)) -> bytes:
    """The post's media area (a vertical band of the screen, status bar and caption/tab bar off): a whole screenshot
    classifies as 'document, screenshot' (measured on the Mac, 2026-10-18)."""
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(png))
    w, h = im.size
    out = io.BytesIO()
    im.crop((0, int(h * band[0]), w, int(h * band[1]))).save(out, "PNG")
    return out.getvalue()


def vision_classifier(profile: dict | None = None, min_conf: float = 0.3, band: tuple[float, float] | None = (0.12, 0.72)) -> Classifier:
    """Apple Vision's VNClassifyImageRequest (macOS only, pyobjc already a dependency): scene labels of the media
    band in ~125 ms (measured on the M5 Pro, 2026-10-18). cls is photo (a still) — motion decides video; topics are
    the top labels; interest 0–3 is the overlap with the persona's interest topics (3 when a liked topic leads)."""
    import Vision
    from Foundation import NSData

    want = {t.lower() for t in (profile or {}).get("topics", [])}

    async def classify(png: bytes) -> dict:
        if band:
            png = media_crop(png, band)

        def run():
            req = Vision.VNClassifyImageRequest.alloc().init()
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(NSData.dataWithBytes_length_(png, len(png)), {})
            ok, err = handler.performRequests_error_([req], None)
            if not ok:
                raise RuntimeError(str(err))
            return [(str(o.identifier()), float(o.confidence())) for o in req.results() if o.confidence() >= min_conf][:8]
        labels = await asyncio.to_thread(run)
        topics = [l for l, _ in labels if l not in GENERIC][:3]
        hits = [i for i, (l, _) in enumerate(labels) if l.lower() in want]
        interest = None if not want else (3 if hits and hits[0] == 0 else 2 if hits else 1)
        return {"cls": "photo", "topics": topics, "interest": interest, "labels": labels}
    return classify


# --- the decision, with its reason (director v1, Ian 2026-10-18 21:5xZ: "the way he would", every decision logged) ---------
from dataclasses import dataclass as _dc  # noqa: E402

STOP_WORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "you", "your", "this", "that",
              "it", "my", "me", "we", "be", "more", "ago", "days", "day", "hours", "liked", "by", "others", "follow", "view", "all"}


def words(items: list[dict]) -> set[str]:
    out = set()
    for i in items:
        for w in re.findall(r"[#@]?[a-z][a-z0-9_]{2,}", _norm(i.get("text", ""))):
            out.add(w.lstrip("#@"))
    return out - STOP_WORDS


def interest(items: list[dict], labels: list[str], profile: dict) -> tuple[int, list[str]]:
    """0–3 and the matches: the caption/hashtag words and the scene labels against the persona's interest topics and
    keywords (seeded from his own content's hashtags until his likes and exports replace them, BEH-05)."""
    want = {t.lower() for t in profile.get("topics", [])} | {k.lower() for k in profile.get("keywords", [])}
    if not want:
        return 1, []
    seen = words(items) | {l.lower() for l in labels}
    hits = sorted(w for w in seen if w in want or any((w.startswith(t) and len(t) >= 3) or (t.startswith(w) and len(w) > 3) for t in want))
    return (3 if len(hits) >= 2 else 2 if hits else 1), hits


def asks_a_question(items: list[dict]) -> bool:
    return any("?" in i.get("text", "") for i in items)


@_dc
class Decision:
    action: str  # skip | stay | like_candidate | open_comments
    reason: str
    interest: int
    matches: list

    def as_dict(self) -> dict:
        return asdict(self)


def decide(cue: Cue, items: list[dict], profile: dict, read_only: bool = True, described=None, match=None) -> Decision:
    """What he would do with this post, and why. `described` (a content_vlm.Proposal from the background model) adds
    the post's one-sentence summary and topics to what the interest match sees, and the reason cites it. Writes are
    candidates only (the pacer and WriteGate decide; in a read-only session they are logged, never done)."""
    if described is not None:
        items = items + [{"text": f"{described.summary} {' '.join(described.topics)}"}]
        if described.is_ad and cue.cls not in ("ad",):
            cue = Cue("ad", "skip", cue.source, cue.latency_ms, cue.topics, cue.interest, cue.markers + ["vlm:is_ad"])
    what = (described.summary.rstrip(".") + ": ") if described is not None and described.summary else ""
    if cue.cls == "ad":
        return Decision("skip", "Sponsored/paid label: he flicks past ads", 0, cue.markers)
    if cue.cls == "live":
        return Decision("skip", "a LIVE badge: he doesn't stop for lives", 0, cue.markers)
    if cue.cls == BLIND:
        return Decision("stay", "no cue arrived in time: a short content-blind look, then on", 1, [])
    if match is not None and match.nearest:  # the semantic cue: embeddings against his taste, the nearest item named
        score, hits = match.interest, [f"≈ '{match.nearest[:60]}' ({match.score:.2f})"]
        if described is None:
            score = min(score, 2)  # the glance (OCR fragments) is coarse: stay/skip only; a like needs the understanding's sentence
        # the model's own 0–3 interest is NOT used (supervisor 23:2xZ: it scores against nothing of his; local Qwen is generous):
        # interest comes from the taste match on the model's sentence and topics; the sentence is what the reason cites
    else:
        score, hits = interest(items, cue.topics, profile)
    if cue.cls == "suggested":  # a new account's feed is mostly suggestions: his interests decide, not the label
        if score >= 2:
            return Decision("stay", f"{what}suggested post, but his thing ({', '.join(hits)}): watches it", score, hits)
        return Decision("skip", f"{what}a 'Suggested for you' post, not his topics: skimmed past", 0, cue.markers)
    kind = "video playing" if cue.cls == "video" else "a still"
    if score >= 3:
        if asks_a_question(items):
            return Decision("open_comments", f"{what}{kind}; caption asks a question and it's squarely his thing ({', '.join(hits)}): he reads the comments", score, hits)
        return Decision("like_candidate", f"{what}{kind}; squarely his interests ({', '.join(hits)}): watches through, a like is in character", score, hits)
    if score == 2:
        return Decision("stay", f"{what}{kind}; touches his interests ({', '.join(hits)}): watches a while", score, hits)
    if cue.cls == "video":
        return Decision("stay", f"{what}video playing, not his topics: a few seconds then on", 1, [])
    return Decision("skip", f"{what}a still, not his topics: a glance, then on", 1, [])


HANDLE = re.compile(r"^@?([a-z0-9][a-z0-9_.]{2,29})$")
HANDLE_STOP = set(STOP_WORDS) | {"follow", "following", "followers", "reels", "live", "sponsored", "suggested", "instagram", "tiktok", "explore", "search",
                                 "home", "shop", "friends", "stem", "comments", "share", "save", "more", "original", "audio", "sound", "duet", "stitch",
                                 "likes", "views", "ago", "yesterday", "today", "music", "video", "photo", "story", "reel", "profile"}


TIME_AGO = re.compile(r"^\d+\s*(s|m|h|d|w|mo|y|sec|min|hours?|days?|weeks?|ago)\b|^(yesterday|today)$|^(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
COUNTER = re.compile(r"^[\d.,]+[kmb]?$", re.I)
CARD_BAND = {"instagram": (0.17, 0.80), "tiktok": (0.55, 0.88)}  # where the post's own handle can be: IG below the story tray, TikTok's lower-left text block
HANDLE_IN_SUMMARY = re.compile(r"^(?:(?:a |the )?(?:verified )?(?:user|account|creator)\s+)?@?([A-Za-z0-9_.]{3,40}?)(?:\s*\([^)]*\))?\s+(?:posted|posts|shares|shared|uploads|uploaded|publishes|published)\b", re.I)


HEADER_DEFAULT = {"instagram": [0.08, 0.26, 0.8, 0.33], "tiktok": [0.0, 0.6, 0.75, 0.85]}  # screens-r2 553b2e6: the first card's author line / the lower-left @handle block


def model_handle(summary: str, ocr_tokens: list[str] | None = None) -> str | None:
    """The account the understanding model named: "User X posted…" / "X posts…", else an account the sentence mentions as
    the source ("a video from X", "by X", "account X", "@X"), else any sentence token that fuzzy-matches a card OCR handle."""
    import difflib

    s = summary.strip()
    m = HANDLE_IN_SUMMARY.match(s)
    if m:
        hd = m.group(1).rstrip(".")
        if hd.lower() not in HANDLE_STOP and not COUNTER.match(hd):
            return hd
    m = re.search(r"(?:\bfrom|\bby|\baccount|\bcreator|\buser|@)\s*'?@?([A-Za-z0-9_.]{4,40})'?", s)
    if m:
        hd = m.group(1).rstrip(".")
        if hd.lower() not in HANDLE_STOP and not COUNTER.match(hd) and (any(ch in hd for ch in "._") or any(ch.isdigit() for ch in hd) or len(hd) >= 8):
            return hd
    for tok in re.findall(r"[A-Za-z0-9_.]{4,40}", s):
        tok = tok.rstrip(".")
        shaped = any(ch in tok for ch in "._") or any(ch.isdigit() for ch in tok)  # plain words ("speaker") are never handles here
        if not shaped or tok.lower() in HANDLE_STOP:
            continue
        for o in ocr_tokens or []:
            if difflib.SequenceMatcher(None, tok.lower(), o.lower()).ratio() >= 0.85:
                return tok
    return None


def in_zone(i: dict, zone, w: float | None, h: float | None) -> bool:
    """Is the OCR item's centre inside a store zone [x0, y0, x1, y1] (fractions of the screen)?"""
    if not zone or not w or not h:
        return True
    cx, cy = i.get("x", 0), i.get("y", 0)  # OCR boxes are centres (ocr.py)
    x0, y0, x1, y1 = zone
    return x0 * w <= cx <= x1 * w and y0 * h <= cy <= y1 * h


def ocr_handles(items: list[dict], h: float | None, app: str | None, zone=None, w: float | None = None) -> list[str]:
    """Handle-shaped OCR tokens inside the post card (or inside the store record's `header` zone when the look gave one):
    no counters (62.1K), dates, chrome words, story-tray or tab-bar lines."""
    lo, hi = CARD_BAND.get(app or "", (0.17, 0.85))
    out = []
    for i in items:
        y = i.get("y", 0)
        if zone:
            if not in_zone(i, zone, w, h):
                continue
        elif h and not (lo * h <= y <= hi * h):
            continue
        for tok in _norm(i.get("text", "")).replace("•", " ").replace("·", " ").split():
            raw = tok.rstrip(",:.")
            m = HANDLE.match(raw)
            if not m:
                continue
            hd = m.group(1)
            if hd in HANDLE_STOP or COUNTER.match(hd) or TIME_AGO.match(hd) or sum(ch.isalpha() for ch in hd) < 3:
                continue
            out.append(hd)
    return out


def confirm_handle(summary: str, items: list[dict], h: float | None = None, app: str | None = None, header=None, w: float | None = None) -> tuple[str | None, bool, str]:
    """(handle, confirmed, summary): the model proposes who posted; the OCR confirms it when a card token matches it with
    difflib ratio ≥ 0.8 (then the OCR spelling wins: sam.posternn → sam.postern); otherwise the model's handle stays,
    marked unconfirmed. With the store record's `header` zone (fast-eyes' look, screens-r2), the search is that zone, and a
    post the model left nameless takes the zone's only handle-shaped token. The summary is never prefixed; a corrected
    spelling is substituted in place (controller-verify, Q1)."""
    import difflib

    header = header or HEADER_DEFAULT.get(app or "")
    proposed = model_handle(summary, ocr_handles(items, h, app))
    if not proposed:
        if header:
            toks = sorted(set(ocr_handles(items, h, app, header, w)))
            if len(toks) == 1:
                return toks[0], True, f"{toks[0]}: {summary}"
        return None, False, summary
    best, score = None, 0.0
    for toks in (ocr_handles(items, h, app, header, w), ocr_handles(items, h, app)):  # the header zone first, then the whole card (a scrolled feed)
        for tok in toks:
            r = difflib.SequenceMatcher(None, tok.lower(), proposed.lower()).ratio()
            if r > score:
                best, score = tok, r
        if best is not None and score >= 0.8:
            break
    if best is not None and score >= 0.8:
        if best.rstrip("_").lower() == proposed.rstrip("_").lower() and len(proposed) > len(best):
            best = proposed  # Vision drops trailing underscores (kim_chef_ for kim_chef__): the longer spelling is the handle
        if best.lower() != proposed.lower():
            summary = re.sub(re.escape(proposed), best, summary, count=1)
        return best, True, summary
    return proposed, False, summary


def who(items: list[dict], h: float | None = None) -> str | None:
    """The post's account handle from the OCR lines, only when confident (Q1: handles are read, not described; a wrong
    OCR handle is worse than the model's): an @token anywhere, or a handle-shaped token (lowercase/digits/._, no chrome word,
    not a counter) in the post's header band (0.2–0.8 h: below Instagram's story tray, above the tab bar) that either carries
    a dot/underscore/digit or sits on/next to a header companion (Follow, •, a time-ago, 'Original audio')."""
    lines = sorted((i for i in items if i.get("text")), key=lambda i: i.get("y", 0))
    cands = []
    for k, i in enumerate(lines):
        t = _norm(i["text"]).replace("•", " • ").replace("·", " • ")
        y = i.get("y", 0)
        toks = t.split()
        companions = " ".join(toks + [_norm(lines[j]["text"]) for j in (k + 1, k - 1) if 0 <= j < len(lines) and abs(lines[j].get("y", 0) - y) < 90])
        has_comp = any(w in companions for w in ("follow", "•", "original audio", "verified")) or bool(TIME_AGO.search(companions))
        for tok in toks:
            raw = tok.rstrip(",:.")
            m = HANDLE.match(raw)
            if not m:
                continue
            hd = m.group(1)
            if hd in HANDLE_STOP or COUNTER.match(hd) or sum(ch.isalpha() for ch in hd) < 3 or TIME_AGO.match(hd):
                continue
            at = tok.startswith("@")
            in_band = h is None or 0.27 * h <= y <= 0.6 * h  # below Instagram's story tray, above the next post's header
            shaped = any(ch in hd for ch in "._")  # dots/underscores read as a handle; digits alone are captions ("18pro")
            if at or (in_band and shaped and has_comp and len(toks) <= 4):
                cands.append((0 if at else 1, y, hd))
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def with_handle(summary: str, handle: str | None) -> str:
    """The understanding's sentence with the OCR handle in front of it; the model's own guessed name ("User X posted",
    "X posted", "X posts") is dropped when it differs."""
    if not handle:
        return summary
    s = summary.strip()
    if s.lower().startswith(handle.lower()):
        return s
    def edits(a: str, b: str) -> int:  # Levenshtein, small strings
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]
    for tok in re.findall(r"@?[A-Za-z0-9_.]{3,40}", s):
        if 0 < edits(tok.lstrip("@").lower().rstrip("."), handle.lower()) <= 2:  # the model's near-miss spelling: corrected from the OCR
            return s.replace(tok, handle, 1)
    if re.search(r"(?<![a-z0-9])@?[a-z0-9]*[._][a-z0-9_.]*(?![a-z0-9])|\b[a-z]+\d+\b", s, re.I) and not re.match(r"^(a |the )?(user|account|creator)\b", s, re.I):
        return s  # the model already names a handle-shaped account; only a nameless sentence gets the OCR handle
    m = re.match(r"^(?:(?:a |the )?(?:verified )?(?:user|account|creator)\s+)?@?([A-Za-z0-9_.]{3,40})(?:\s*\([^)]*\))?\s+(?:posted|posts|shares|shared|uploads|uploaded)\s+", s, re.I)
    if m:
        if m.group(1).lower().rstrip(".") == handle.lower():
            return s
        return f"{handle} posted " + s[m.end():]
    return f"{handle}: {s}"
