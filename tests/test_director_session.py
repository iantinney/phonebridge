"""The session director (BEH-02): per post arrive → cues → dwell → habits → (VLM for writes only) → leave; no
fixed beat; read-only sessions never write; evidence written like an agent session."""

import asyncio
import json
import random
import statistics

import pytest

from phonebridge.director import cues as cu
from phonebridge.director.persona import default_persona
from phonebridge.director.session import Director, FeedPost

from test_bridge import tiny_png

W, H = 1170, 2532


def post(kind, text=None, moving=False, markers=()):
    text = text or f"someone{random.randrange(10**6)} a caption... more"  # captions differ post to post
    items = [{"text": text, "x": 40, "y": 2000, "w": 800, "h": 30, "conf": 0.9},
             {"text": "Your story", "x": 60, "y": 300, "w": 120, "h": 24, "conf": 0.95},  # the feed's own chrome (S3 gate)
             {"text": "Following Friends For You", "x": 300, "y": 150, "w": 400, "h": 24, "conf": 0.95}]
    items += [{"text": m, "x": 60, "y": 900, "w": 90, "h": 24, "conf": 0.95} for m in markers]
    return FeedPost(kind=kind, png=tiny_png(12, 26, bytes([random.randrange(256) for _ in range(3)])), items=items, moving=moving)


class FeedSim:
    """A phone whose feed is a list of posts: a flick moves to the next; screens change per post."""
    def __init__(self, posts, size=(W, H), lag_s=0.0):
        self.posts, self.size, self.i, self.lag_s = posts, size, 0, lag_s
        self.calls, self.shots = [], 0
        self.clock = 0.0

    @property
    def cur(self):
        return self.posts[self.i % len(self.posts)]  # an endless feed

    async def screen(self, device):
        self.shots += 1
        p = self.cur
        return p.png if not p.moving else tiny_png(12, 26, bytes([(self.shots * 97) % 256] * 3))  # a playing video differs every look

    async def ocr(self, device, fast=False):
        return self.cur.items

    async def swipe(self, device, x1, y1, x2, y2, duration=0.3, caller="local", lease=None, release=None):
        self.calls.append(("swipe", round(y1 - y2), round(duration, 3)))
        if y2 < y1:
            self.i += 1
        else:
            self.i = max(0, self.i - 1)

    async def tap(self, device, x, y, w=0, h=0, caller="local", lease=None):
        self.calls.append(("tap", round(x), round(y)))


class Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    async def sleep(self, s):
        self.t += s


def run(posts, minutes=2.0, read_only=True, vlm=None, pacer_ok=None, seed=1, tmp_path=None, **kw):
    sim, clock = FeedSim(posts), Clock()
    d = Director(sim, "iphone12", app="instagram", persona=default_persona("public", seed=seed), evidence=tmp_path,
                 read_only=read_only, vlm=vlm, pacer_ok=pacer_ok or (lambda a: True), now=clock.now, sleep=clock.sleep,
                 rng=random.Random(seed), **kw)
    res = asyncio.run(d.run(minutes=minutes))
    return d, sim, res


def test_session_reacts_to_what_is_on_screen(tmp_path):
    posts = [post("ad", markers=("Sponsored",)), post("video", moving=True), post("photo"), post("ad", markers=("Sponsored",)),
             post("video", moving=True), post("suggested", markers=("Suggested for you",))] * 6
    d, sim, res = run(posts, tmp_path=tmp_path)
    views = [e for e in d.events if e["ev"] == "view"]
    assert len(views) >= 12
    by = {}
    for v in views:
        by.setdefault(v["cls"], []).append(v["dwell"])
    assert statistics.median(by["ad"]) < statistics.median(by["video"])
    assert all(v["decision"] == "skip" for v in views if v["cls"] == "ad")  # ads (and lives) are always skipped
    sug = [v for v in views if v["cls"] == "suggested"]
    assert sum(v["decision"] == "skip" for v in sug) >= 0.6 * len(sug)  # passive: off-topic suggestions mostly skipped, the human mixture decides the rest
    assert all("cue_ms" in v and v["source"] in ("ocr", "classifier", "motion", "late") for v in views)


def test_read_only_session_never_writes_but_understands_for_the_journal(tmp_path):
    calls = []
    async def vlm(png):
        calls.append(png)
        return {"kind": "video", "is_ad": False, "topics": ["gym"], "interest": 3, "text_to_read": "", "quick_skip": False, "summary": "a coach lifts"}
    posts = [post("video", moving=True) for _ in range(30)]
    d, sim, res = run(posts, read_only=True, vlm=vlm, tmp_path=tmp_path, mode="intentional")
    assert not [c for c in sim.calls if c[0] == "tap"]  # FeedSim has no look(): tap habits stay unrendered in read-only
    assert not [e for e in d.events if e["ev"] in ("write", "like", "comment")]
    assert calls and [e for e in d.events if e["ev"] == "vlm" and e["summary"]] and res["writes"] == 0 and res["label"] == "done (read-only)"


def test_writes_need_eyes_even_when_allowed(tmp_path):
    """Writes allowed, intentional mode, squarely on-niche posts, but a phone without look(): no like is ever attempted
    (every write is look-gated and verified on the recognised screen); the understanding still journals (Q1)."""
    async def vlm(png):
        return {"kind": "video", "is_ad": False, "topics": ["gym"], "interest": 3, "text_to_read": "", "quick_skip": False}
    posts = [post("video", moving=True, text=f"gymcoach{i} pullups and protein... more") for i in range(40)]
    d, sim, res = run(posts, read_only=False, vlm=vlm, tmp_path=tmp_path, minutes=6, mode="intentional")
    props = [e for e in d.events if e["ev"] == "vlm"]
    assert props and all(e["cached"] in (True, False) for e in props)
    assert not [e for e in d.events if e["ev"] == "write"] and res["writes"] == 0 and res["label"] == "done (no writes)"
    assert not [c for c in sim.calls if c[0] == "tap"]


def test_no_fixed_beat_between_glides(tmp_path):
    posts = [post("video", moving=True), post("photo"), post("photo")] * 20
    d, sim, res = run(posts, minutes=4, tmp_path=tmp_path)
    glides = [e for e in d.events if e["ev"] == "glide"]
    gaps = [b["s"] - a["s"] for a, b in zip(glides, glides[1:])]
    assert len(gaps) > 15
    assert statistics.pstdev(gaps) / statistics.mean(gaps) > 0.5
    nxt = [g for g in glides if g["kind"] == "next"]
    rel = [g["release"] for g in nxt]
    assert len(set(round(x, 2) for x in rel)) > len(nxt) * 0.2 and statistics.pstdev(rel) > 0.03  # (drag_px variety is checked on the full-size FeedPhysics)
    durs = [g["duration_s"] for g in glides]
    assert len(set(round(x, 2) for x in durs)) > len(glides) * 0.2 and statistics.pstdev(durs) > 0.05  # no constant duration (trial 1's 80 ms tell)
    views = [e for e in d.events if e["ev"] == "view"]
    assert len({round(v["q"], 4) for v in views}) > len(views) * 0.85  # one dwell quantile per post, never repeated


def test_evidence_dir_has_events_session_and_frames(tmp_path):
    posts = [post("photo"), post("video", moving=True)] * 5
    d, sim, res = run(posts, minutes=2, tmp_path=tmp_path)
    assert (tmp_path / "events.jsonl").exists() and (tmp_path / "session.json").exists()
    s = json.loads((tmp_path / "session.json").read_text())
    assert s["read_only"] is True and s["label"] == "done (read-only)" and s["counts"]["posts"] >= 5
    assert s["cue"]["content_share"] is not None and "late" in s["cue"]
    assert list(tmp_path.glob("frames/*.png"))


def test_engagement_is_planned_from_the_persona_but_look_gated(tmp_path):
    from phonebridge.director import params as pm
    p = default_persona("public", seed=2)
    params = pm.defaults()
    for k in params.engage:
        params.engage[k] = pm.Rate(0.5)
    sim, clock = FeedSim([post("video", moving=True, text=f"coach{i} pullups and protein... more") for i in range(80)]), Clock()
    d = Director(sim, "iphone12", app="instagram", persona=p, evidence=tmp_path, read_only=True, now=clock.now,
                 sleep=clock.sleep, rng=random.Random(2), params=params)
    asyncio.run(d.run(minutes=15))
    planned = {k for e in d.events if e["ev"] == "view" for k in e["engage"]}
    assert {"expand", "carousel", "tap_media"} <= planned and "like" not in planned  # likes are never an engagement roll
    assert not planned & {"comments_peek", "profile"}  # passive mode: the journal lists only kinds this mode renders
    skipped = [e for e in d.events if e["ev"] == "kind_skip"]
    assert skipped and all("no look" in e["why"] for e in skipped)
    assert not [c for c in sim.calls if c[0] == "tap"]  # FeedSim has no look(): nothing on the post is tapped


def test_exploration_hook_rolls_per_session(tmp_path):
    """EXPLORATION (Ian, 2026-10-18): a small per-account chance per session to visit an unmapped part of the app; the
    director only rolls and calls the hook, which screens' coverage map will supply."""
    visited = []

    async def explore(director):
        visited.append(director.app)
    from phonebridge.director import params as pm
    for chance, n in ((1.0, 1), (0.0, 0)):
        p = default_persona("public", seed=4)
        params = pm.defaults()
        params.detour["exploration"] = pm.Rate(chance)
        sim, clock = FeedSim([post("photo") for _ in range(10)]), Clock()
        d = Director(sim, "iphone12", app="tiktok", persona=p, evidence=None, read_only=True, now=clock.now, sleep=clock.sleep,
                     rng=random.Random(4), explore=explore, params=params)
        asyncio.run(d.run(minutes=0.5))
        assert len([e for e in d.events if e["ev"] == "detour" and e["kind"] == "exploration"]) == n
    assert visited == ["tiktok"]


def test_a_saved_guessed_seed_is_discarded_on_load(tmp_path):
    import json as _json

    from phonebridge.director.persona import Persona, default_persona
    p = default_persona("old", seed=1)
    p.save(tmp_path)
    f = tmp_path / "old" / "persona.json"
    raw = _json.loads(f.read_text())
    raw["interest"] = {"topics": ["gym", "fitness"], "keywords": ["protein"], "source": "seed: content queue hashtags (gym, fitness)"}
    f.write_text(_json.dumps(raw))
    assert Persona.load("old", tmp_path).interest == {"topics": []}

