"""SCR-08: harvest evidence directories into the corpus (read-only on the evidence).

Labelled frames (LABELS) become fixtures `<platform>/<screen_id>/`; labelled confusers go to `confusers/`;
listed runs become replay sequences `sequences/<flow>/<n>/` with the action behind each frame, aligned
from the Mac's phonebridge `calls.jsonl` by timestamp when a copy is given; every other frame is run through
the cascade: recognised ones are left alone (self-labelling would hide regressions), unknown ones are queued
under `unknown/`. Synthetic confusers (dark mode, push transitions, cross-fades, dim, status-bar offset) are
derived from the fixtures so the ≥ 200 confuser gate has teeth before the live harvest. Re-run after each
tracer run; everything is idempotent through the store's phash dedupe.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance, ImageOps

from ..backends.replay import write_sequence
from .cascade import Cascade, Frame
from .store import REPO_RECORDS, Store

T = "tracer-20261018"
A = f"{T}/01-account-check"
LABELS: dict[str, str | None] = {
    f"{A}/01-start": "ios.home",
    f"{A}/02-ig-open": "instagram.home",
    f"{A}/03-profile": "instagram.profile",
    f"{A}/04-menu": "instagram.profile",  # the menu tap had not opened yet
    f"{A}/05-menu2": "instagram.settings",
    f"{A}/06-search-about": "instagram.settings.search",
    f"{A}/07-about-account": "instagram.about_account",
    f"{A}/08-tt-open": "tiktok.feed",
    f"{A}/09-tt-profile": "tiktok.profile",
    f"{A}/10-tt-menu": "tiktok.menu",
    f"{A}/11-tt-settings": "tiktok.settings",
    f"{A}/12-tt-settings-down": "tiktok.settings",
    f"{A}/13-tt-settings-top": "tiktok.settings",
    f"{A}/14-tt-account": "tiktok.settings.account",
    f"{A}/15-tt-settings-back": "tiktok.settings",
    f"{A}/16-tt-scan1": "tiktok.settings",
    f"{A}/17-tt-scan2": "tiktok.settings",
    f"{A}/18-tt-scan3": "tiktok.settings",
    f"{A}/19-tt-scan4": "tiktok.settings",
    f"{A}/20-tt-scan5": "tiktok.settings",
    f"{A}/21-tt-bottom": "tiktok.settings",
    f"{A}/22-tt-help": "tiktok.help",
    f"{A}/23-tt-status": "tiktok.help.article",
    f"{A}/24-tt-status2": "tiktok.help.article",
    f"{A}/25-back": "tiktok.help",
    f"{A}/26-tt-safety": "tiktok.safety_center",
    f"{A}/27-tt-account-status": "tiktok.account_status",
    f"{A}/28-settings": "ios.settings",
    f"{A}/29-settings-search": "confuser",  # Settings search results: no record (three words)
    f"{A}/30-ig-settings": "ios.settings.app",
    f"{A}/31-ig-photos": "ios.settings.photos_access",
    f"{A}/32-ig-photos-after": "ios.alert.photos_full_access",
    f"{A}/33-ig-photos-full": "ios.settings.photos_access",
    f"{T}/02-calibrate/01-now": "instagram.new_post.picker",
    f"{T}/02-calibrate/02-after-R": "instagram.home",
    f"{T}/02-calibrate/03-newpost": "instagram.profile",
    f"{T}/02-calibrate/04-after-swipe": "instagram.profile",
    f"{T}/02-calibrate/05-check": "instagram.profile",
    f"{T}/02-calibrate/06-home": "instagram.home",
    f"{T}/02-rehearse1-unknown-new-post/01-in-home": "instagram.home",
    f"{T}/02-rehearse1-unknown-new-post/02-retry-1": "instagram.new_post.picker",
    f"{T}/02-rehearse1-unknown-new-post/03-stopped-unknown": "instagram.new_post.picker",
    f"{T}/03-calibrate/01-now": "instagram.camera_access",
    f"{T}/03-calibrate/02-watch1": "instagram.reel.editor",
    f"{T}/03-calibrate/03-watch2": "instagram.reel.editor",
    f"{T}/03-calibrate/04-watch3": "instagram.reel.editor",
    f"{T}/05-calibrate/01-now": "ios.home",
    f"{T}/05-calibrate/02-ig": "instagram.home",
    f"{T}/05-calibrate/03-create": "instagram.new_reel.picker",
    f"{T}/05-calibrate/04-picked": "instagram.new_reel.picker",
    f"{T}/05-calibrate/05-after-next": "instagram.reel.editor",
    f"{T}/05-calibrate/06-share": "instagram.reel.details",
    f"{T}/05-calibrate/07-share-scrolled": "instagram.reel.details",
    f"{T}/05-calibrate/08-back1": "instagram.reel.editor",
    f"{T}/05-calibrate/09-back2": "instagram.popup.start_over",
    f"{T}/05-calibrate/10-discarded": "instagram.new_reel.picker",
    f"{T}/05-calibrate/11-closed": "instagram.new_reel.picker",
    # lane tiktok's first live rehearsal (2026-10-19 03:12Z): For You (tabs split by OCR), the camera's access page,
    # iOS's photo-library prompt (TikTok never had Photos access; Ian's call)
    "M2-tiktok/20261019T031149Z-rehearse/01-in-main tab": "tiktok.feed",
    "M2-tiktok/20261019T031149Z-rehearse/02-in-camera": "tiktok.create.camera",
    "M2-tiktok/20261019T031149Z-rehearse/03-retry-1": "ios.alert.generic",
    "M2-tiktok/20261019T031149Z-rehearse/04-stopped-unknown": "ios.alert.generic",
    # backend-mock frames (blank PNG + scripted OCR, 9:41): neither fixtures nor confusers
    "guidebook-20261018/scratch/clip/talk.final.post/instagram/07-final": "skip",
    "guidebook-20261018/scratch/clip/talk.final.post/tiktok/07-final": "skip",
}
# replay sequences: (flow, n) -> frames in order with the screen and, when known without the call log,
# the action that produced the frame (tracer's run.log: "tap create (+, top-left) (84,218)")
SEQUENCES: dict[tuple[str, int], list[tuple[str, dict | None]]] = {
    ("ig_reel", 1): [(f"{T}/02-rehearse1-unknown-new-post/01-in-home", None),
                     (f"{T}/02-rehearse1-unknown-new-post/02-retry-1", {"kind": "tap", "x": 84, "y": 218, "target": "create"})],
    ("ig_reel", 2): [(f"{T}/05-calibrate/02-ig", None), (f"{T}/05-calibrate/03-create", {"kind": "tap", "target": "create"}),
                     (f"{T}/05-calibrate/04-picked", {"kind": "tap", "target": "newest"}),
                     (f"{T}/05-calibrate/05-after-next", {"kind": "tap", "target": "newest"}),
                     (f"{T}/05-calibrate/06-share", {"kind": "tap", "target": "next"}),
                     (f"{T}/05-calibrate/07-share-scrolled", {"kind": "swipe", "direction": "up"}),
                     (f"{T}/05-calibrate/08-back1", {"kind": "tap", "target": "back"}),
                     (f"{T}/05-calibrate/09-back2", {"kind": "tap", "target": "close"}),
                     (f"{T}/05-calibrate/10-discarded", {"kind": "tap", "target": "start_over"}),
                     (f"{T}/05-calibrate/11-closed", {"kind": "tap", "target": "cancel"})],
    ("ig_walk", 1): [(f"{A}/02-ig-open", None), (f"{A}/03-profile", {"kind": "tap", "target": "profile_tab"}),
                     (f"{A}/05-menu2", {"kind": "tap", "target": "menu"}), (f"{A}/06-search-about", {"kind": "tap", "target": "search"}),
                     (f"{A}/07-about-account", {"kind": "tap", "target": "about_account"})],
    ("tt_walk", 1): [(f"{A}/08-tt-open", None), (f"{A}/09-tt-profile", {"kind": "tap", "target": "profile_tab"}),
                     (f"{A}/10-tt-menu", {"kind": "tap", "target": "menu"}), (f"{A}/11-tt-settings", {"kind": "tap", "target": "settings"})],
    ("ios_photos_wall", 1): [(f"{A}/28-settings", None), (f"{A}/30-ig-settings", {"kind": "tap", "target": "search"}),
                             (f"{A}/31-ig-photos", {"kind": "tap", "target": "photos"}),
                             (f"{A}/32-ig-photos-after", {"kind": "tap", "target": "full_access"}),
                             (f"{A}/33-ig-photos-full", {"kind": "tap", "target": "allow"})],
}
INPUT_OPS = ("tap", "swipe", "home", "launch", "paste", "type", "open_url", "unlock")


def load_calls(path: Path | None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("op") in INPUT_OPS and r.get("ok", True):
            out.append(r)
    return out


def action_between(calls: list[dict], t0: float, t1: float) -> dict | None:
    """The last input the phone got between two frames (the action behind the later one)."""
    hits = [c for c in calls if t0 < c["ts"] <= t1]
    if not hits:
        return None
    c = hits[-1]
    if c["op"] == "tap":
        return {"kind": "tap", "x": c.get("x"), "y": c.get("y"), "ts": c["ts"]}
    if c["op"] == "swipe":
        dx, dy = c.get("dx", 0), c.get("dy", 0)
        return {"kind": "swipe", "direction": ("up" if dy < 0 else "down") if abs(dy) >= abs(dx) else ("left" if dx < 0 else "right"), "ts": c["ts"]}
    if c["op"] == "open_url":
        return {"kind": "launch", "bundle": c.get("bundle_id"), "ts": c["ts"]}
    return {"kind": c["op"], "ts": c["ts"]}


def _frame(evidence: Path, rel: str) -> Frame | None:
    png = evidence / f"{rel}.png"
    if not png.exists():
        return None
    return Frame.from_files(png)


def harvest(evidence: Path, store: Store, calls: Path | None = None, synth: bool = True, log=print) -> dict:
    evidence = Path(evidence)
    cascade = Cascade(store)
    counts = {"fixtures": 0, "confusers": 0, "unknown": 0, "known_unlabelled": 0, "sequences": 0, "synthetic": 0, "missing": 0}
    labelled: list[tuple[str, Frame]] = []
    for rel, label in LABELS.items():
        f = _frame(evidence, rel)
        if f is None:
            counts["missing"] += 1
            continue
        meta = {"source": rel, "app_version": None}
        if label == "skip":
            continue
        if label == "confuser":
            if store.save_confuser(f, {**meta, "kind": "real"}, name=f"real-{Path(rel).parent.name}-{Path(rel).name}"):
                counts["confusers"] += 1
        elif label:
            platform = label.split(".")[0]
            if store.save_frame(platform, label, f, meta):
                counts["fixtures"] += 1
            labelled.append((label, f))
    # unlabelled frames: queue the unknown ones
    for png in sorted(evidence.rglob("*.png")):
        rel = png.relative_to(evidence).with_suffix("").as_posix()
        if rel in LABELS or not png.with_suffix(".ocr.json").exists():
            continue
        f = Frame.from_files(png)
        platform = {"ig": "instagram", "tt": "tiktok"}.get(png.name.split("-")[1][:2], "instagram") if "-" in png.name else "instagram"
        m = cascade.recognise(f, platform=None)
        if m.unknown:
            store.queue_unknown(platform, f, {"source": rel, "context": "harvest: unlabelled and unrecognised"})
            counts["unknown"] += 1
        else:
            counts["known_unlabelled"] += 1
    # sequences with the action behind each frame
    call_log = load_calls(calls)
    for (flow, n), frames in SEQUENCES.items():
        items, prev_ts = [], None
        for rel, action in frames:
            png = evidence / f"{rel}.png"
            if not png.exists():
                break
            f = Frame.from_files(png)
            ts = png.stat().st_mtime
            act = dict(action) if action else None
            if prev_ts is not None and call_log:
                logged = action_between(call_log, prev_ts, ts)
                if logged:
                    act = {**(act or {}), **{k: v for k, v in logged.items() if k not in (act or {}) or k in ("x", "y", "ts")}}
            items.append((Path(rel).name, f.png, f.ocr, {"screen": LABELS.get(rel), "action": act, "source": rel, "ts": ts}))
            prev_ts = ts
        if len(items) == len(frames):
            write_sequence(store.corpus / "sequences" / flow / str(n), items)
            counts["sequences"] += 1
    if synth and labelled:
        counts["synthetic"] = synthesize_confusers(store, labelled)
    write_manifest(store)
    log(f"harvest {evidence.name}: {counts}")
    return counts


def _png(im: Image.Image) -> bytes:
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=False, compress_level=1)
    return out.getvalue()


def synthesize_confusers(store: Store, labelled: list[tuple[str, Frame]], seed: int = 7) -> int:
    """Derived confusers (SCR-03): dark mode (inverted pixels, same OCR), an iOS push transition caught
    halfway (two screens side by side, OCR shifted), a cross-fade (OCR at low confidence), a dim/off display
    (no OCR), and a frame offset by a fifth of its height (anchors out of their regions)."""
    rnd = random.Random(seed)
    n = 0
    frames = [(l, f) for l, f in labelled if f.png is not None]
    for i, (label, f) in enumerate(frames):
        im = Image.open(io.BytesIO(f.png)).convert("RGB")
        W, H = im.size
        stem = f"synth-{label}-{i:02d}"
        others = [(l, o) for l, o in frames if l.split(".")[0] != label.split(".")[0]] or [(l, o) for l, o in frames if l != label]
        other_label, other = others[(i * 7 + 3) % len(others)]  # another platform's screen, never the same label
        oim = Image.open(io.BytesIO(other.png)).convert("RGB")
        variants = []
        rec = store.records.get(label)
        from .cascade import ocr_find
        hits = {a: ocr_find(f.ocr, a, reg, W, H) for a, reg in (rec.match.ocr_in if rec else ())}
        in_band = [a for a, it in hits.items() if it and H * 0.03 <= it["y"] < H * 0.115]
        top_anchored = [a for a, it in hits.items() if it and it["y"] < H * 0.3]
        overlay = rec is not None and rec.kind in ("popup", "alert")  # the screen beneath an overlay stays recognisable
        if rec is None or (rec.appearance in ("light", "dark") and not overlay):  # the other appearance is a confuser
            variants.append(("dark", _png(ImageOps.invert(im)), f.ocr))
        shift = int(W * 0.4)
        for name, sign in (("push", 1), ("pushr", -1)):  # a push transition caught halfway, either direction
            push = Image.new("RGB", (W, H), "black")
            push.paste(im, (-sign * shift, 0))
            push.paste(oim, (sign * (W - shift) if sign > 0 else -(W - shift), 0))
            if sign > 0:
                ocr = [{**it, "x": it["x"] - shift} for it in f.ocr if it["x"] - shift > 0] + \
                      [{**it, "x": it["x"] + W - shift} for it in other.ocr if it["x"] + W - shift < W]
            else:
                ocr = [{**it, "x": it["x"] + shift} for it in f.ocr if it["x"] + shift < W] + \
                      [{**it, "x": it["x"] - (W - shift)} for it in other.ocr if it["x"] - (W - shift) > 0]
            variants.append((name, _png(push), ocr))
        other2_label, other2 = others[(i * 5 + 1) % len(others)]
        oim2 = Image.open(io.BytesIO(other2.png)).convert("RGB")
        variants.append(("fade", _png(Image.blend(im, oim, 0.5)), [{**it, "conf": 0.4} for it in f.ocr + other.ocr]))
        variants.append(("fade2", _png(Image.blend(im, oim2, 0.5)), [{**it, "conf": 0.4} for it in f.ocr + other2.ocr]))
        variants.append(("dim", _png(ImageEnhance.Brightness(im).enhance(0.06)), []))
        if rec is None or top_anchored:  # a shifted layout only confuses a record anchored near the top
            off = Image.new("RGB", (W, H), "black")
            off.paste(im, (0, int(H * 0.2)))
            variants.append(("offset", _png(off), [{**it, "y": it["y"] + int(H * 0.2)} for it in f.ocr if it["y"] + int(H * 0.2) < H]))
        if rec is not None and not overlay:  # a near-miss: the same layout with its identifying anchors gone (Drafts vs Post)
            keys = [a for a, _ in rec.match.ocr_in] + list(rec.match.ocr_all) + list(rec.match.ocr_any)
            gone, near, draw = [], im.copy(), None
            from PIL import ImageDraw
            draw = ImageDraw.Draw(near)
            for a in keys:
                hit = ocr_find(f.ocr, a, None, W, H)
                if hit and hit not in gone:
                    gone.append(hit)
                    draw.rectangle([hit["x"] - hit["w"] // 2 - 4, hit["y"] - hit["h"] // 2 - 4, hit["x"] + hit["w"] // 2 + 4, hit["y"] + hit["h"] // 2 + 4],
                                   fill=near.getpixel((max(0, hit["x"] - hit["w"] // 2 - 8), hit["y"])))
                if len(gone) >= max(1, len(rec.match.ocr_any) - rec.match.at_least + 1 if rec.match.ocr_any else 1):
                    break
            for name, file, reg, min_score in rec.match.templates:  # the identifying icon goes too (a scrolled home keeps it)
                from .cascade import find_template
                tpl = REPO_RECORDS / "templates" / file
                if tpl.exists() and (hit := find_template(f.png, tpl, reg, 0.0)):
                    from PIL import Image as _I
                    tw, th = _I.open(tpl).size
                    draw.rectangle([hit[0] - tw // 2, hit[1] - th // 2, hit[0] + tw // 2, hit[1] + th // 2], fill=near.getpixel((hit[0] + tw, hit[1])))
                    gone = gone or [None]
            if gone:
                variants.append(("nearmiss", _png(near), [it for it in f.ocr if it not in gone]))
        if rec is None or in_band:  # another screen's title bar over this body: a near-miss for title-identified records
            swap = im.copy()
            swap.paste(oim.crop((0, int(H * 0.03), W, int(H * 0.115))), (0, int(H * 0.03)))
            socr = [it for it in other.ocr if H * 0.03 <= it["y"] < H * 0.115] + [it for it in f.ocr if not (H * 0.03 <= it["y"] < H * 0.115)]
            variants.append(("swap", _png(swap), socr))
        for kind, png, items in variants:
            fr = Frame.from_png(png, items)
            if store.save_confuser(fr, {"kind": f"synthetic:{kind}", "source": label, "other": other_label}, name=f"{stem}-{kind}"):
                n += 1
    return n


def write_manifest(store: Store) -> Path:
    """TST-01: tie every stop text, system alert, irreversible screen and fault to fixtures or an xfail."""
    screens: dict[str, list[str]] = {}
    for platform, sid, png in store.fixtures():
        screens.setdefault(sid, []).append(png.relative_to(store.corpus).as_posix())
    irreversible = {}
    for r in store.records.values():
        if r.irreversible:
            n = len(screens.get(r.id, []))
            irreversible[r.id] = screens[r.id] if n >= 5 else f"xfail: {n} < 5 live frames (harvest in rehearse; tracer never reached it)"
    stops = {r.id: screens.get(r.id) or "xfail: no live stop frame yet (PT texts cited; harvested when one happens)"
             for r in store.records.values() if r.kind == "stop"}
    alerts = {r.id: screens.get(r.id) or "xfail: no frame yet" for r in store.records.values() if r.kind in ("alert", "popup")}
    seqs = sorted(p.relative_to(store.corpus).as_posix() for p in (store.corpus / "sequences").glob("*/*")) \
        if (store.corpus / "sequences").exists() else []
    faults = {"popup": "replay: tests/test_screens_engine.py (ReplayPhone Fault popup)", "system_alert": "replay: Fault popup/alert",
              "crash": "replay: Fault crash", "background": "replay: Fault background", "banner_tap": "replay: Fault banner_tap",
              "lost_link": "replay: Fault lost_link", "lock": "replay: Fault lock", "wrong_account": "replay: Fault wrong_account",
              "stale_layout": "replay: Fault stale_layout", "photos_wall": "sequences/ios_photos_wall/1" if seqs else "xfail"}
    p = store.corpus / "manifest.yaml"
    # every lane's worktree shares this corpus: keep the notes of records this worktree doesn't have (yet)
    old = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
    irreversible, stops, alerts = ({**{k: v for k, v in (old.get(name) or {}).items() if k not in store.records}, **new}
                                   for name, new in (("irreversible", irreversible), ("stops", stops), ("alerts", alerts)))
    m = {"screens": screens, "confusers": len(store.confusers()), "sequences": seqs, "irreversible": irreversible,
         "stops": stops, "alerts": alerts, "faults": faults}
    p.write_text(yaml.safe_dump(m, sort_keys=False, allow_unicode=True, width=120))
    return p
