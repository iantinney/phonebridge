"""Lane tiktok: the read-only TikTok walk (feed → search → profile → settings → back) on replay over the real
tracer frames (iPhone 12, 2026-10-18), the Search screen learned by the slow model (scripted: mock), and a
finished walk counted as an OBSERVING session on the account."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from phonebridge.backends.replay import ReplayPhone, write_sequence
from phonebridge.screens import live
from phonebridge.screens.cascade import Cascade, Frame
from phonebridge.screens.engine import Engine, ListJournal
from phonebridge.screens.learn import Learner
from phonebridge.screens.store import CORPUS, RECORD_DIRS, Store
from phonebridge.screens.vlm import VLM
from screens_fixtures.synth import frame as synth_frame
from phonebridge.config import DATA_DIR

TT = CORPUS / "sequences" / "tt_walk" / "1"
pytestmark = pytest.mark.skipif(not TT.exists(), reason=f"no harvested TikTok walk under {TT}")


@pytest.fixture(autouse=True)
def records(tmp_path, monkeypatch):
    """A copy of the repo records: a run that demotes or counts writes to the record files it loaded."""
    import shutil
    dirs = []
    for i, d in enumerate(RECORD_DIRS):
        shutil.copytree(d, tmp_path / f"records{i}")
        dirs.append(tmp_path / f"records{i}")
    monkeypatch.setattr(sys.modules[__name__], "RECORD_DIRS", dirs)
    return dirs


def _tt(name: str) -> Frame:
    return Frame.from_files(TT / f"{name}.png")


def _item(text, x, y, w=200, h=44):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": 1.0}


# TikTok's Search page as the app shows it after the magnifier (field + Search button up top, suggestions below)
SEARCH_OCR = [_item("1:52", 156, 74, 80), _item("Search", 1080, 210, 120), _item("You may like", 200, 380, 260),
              _item("Clear", 1060, 380, 100), _item("love is blind season 9", 330, 520, 420),
              _item("gym motivation", 260, 640, 300), _item("Search", 1080, 1880, 160)]


def _seq(tmp_path, names_actions):
    frames = {"feed": _tt("08-tt-open"), "profile": _tt("09-tt-profile"), "menu": _tt("10-tt-menu"),
              "settings": _tt("11-tt-settings"),
              "search": Frame.from_png(synth_frame(top="black", tab="black", body="black", seed=7), SEARCH_OCR)}
    rows = []
    for i, (name, target) in enumerate(names_actions, 1):
        f = frames[name]
        rows.append((f"{i:02d}-{name}", f.png, f.ocr, {"action": {"kind": "tap", "target": target} if target else None}))
    return write_sequence(tmp_path / "seq", rows)


WALK = [("feed", None), ("search", "search"), ("feed", "back"), ("profile", "profile_tab"), ("menu", "menu"),
        ("settings", "settings"), ("profile", "back"), ("feed", "home_tab"), ("search", "search"), ("feed", "back")]
KNOWN = [("feed", None), ("profile", "profile_tab"), ("menu", "menu"), ("settings", "settings"), ("profile", "back"),
         ("feed", "home_tab")]


def _vlm(tmp_path, calls):
    def transport(payload):
        calls.append(payload["model"])
        a = {"screen_name": "tiktok_search", "kind": "screen", "action": "none", "element_id": -1,
             "target_box": [0, 0, 0, 0], "label_text": "", "confidence": 0.8, "reason": "TikTok's search page"}
        return {"content": [{"type": "text", "text": json.dumps(a)}], "usage": {"input_tokens": 900, "output_tokens": 60}}
    return VLM(transport=transport, state_dir=tmp_path / "vlm")


def _engine(store, phone, learner=None):
    return Engine(store=store, cascade=Cascade(store), phone=phone, eyes=phone.ocr, journal=ListJournal(), learner=learner,
                  sleep=lambda s: asyncio.sleep(0), templates_dir=RECORD_DIRS[0] / "templates")


def test_tiktok_known_walk_on_real_frames(tmp_path):
    """No slow model: feed → profile → ≡ → Settings and privacy → back → home, every tap on its target."""
    phone = ReplayPhone(_seq(tmp_path, KNOWN))
    store = Store(records_dirs=RECORD_DIRS, corpus=tmp_path / "corpus")
    out = asyncio.run(_engine(store, phone).run(live.KNOWN_WALKS["tiktok"], rehearse=True))
    assert out.status == "done", (out.reason, [e["event"] for e in out.journal.events])
    taps = [t for t in phone.log if t["op"] == "tap"]
    assert len(taps) == len(KNOWN) - 1 and all(t["matched"] for t in taps), taps  # the drawer-close step is skipped


def test_back_from_settings_onto_the_drawer_closes_it(tmp_path):
    """≡ opens a drawer from the right over a dimmed strip of the profile; if Back from Settings lands on the drawer
    again, the walk closes it by that strip (never by the drawer's own rows: QR code, TikTok Studio)."""
    phone = ReplayPhone(_seq(tmp_path, KNOWN[:4] + [("menu", "back"), ("profile", "close"), ("feed", "home_tab")]))
    store = Store(records_dirs=RECORD_DIRS, corpus=tmp_path / "corpus")
    out = asyncio.run(_engine(store, phone).run(live.KNOWN_WALKS["tiktok"], rehearse=True))
    assert out.status == "done", (out.reason, [e["event"] for e in out.journal.events])
    taps = [t for t in phone.log if t["op"] == "tap"]
    assert all(t["matched"] for t in taps) and not any(e["event"] == "expectation_failed" for e in out.journal.events)
    close = next(t for t in taps if t["target"] == "close")
    assert close["x"] < 0.12 * 1170  # the dimmed strip, left of the drawer


def test_tiktok_search_learned_once_then_fast(tmp_path):
    """The live walk: Search is unknown, the slow model names it, it is saved at once as `tiktok.search`, and the
    second visit in the same walk (and a second walk) needs no slow-model call."""
    store = Store(records_dirs=RECORD_DIRS, corpus=tmp_path / "corpus")
    calls = []
    learner = Learner(_vlm(tmp_path, calls), store)
    for run in (1, 2):
        phone = ReplayPhone(_seq(tmp_path / str(run), WALK))
        out = asyncio.run(_engine(store, phone, learner).run(live.WALKS["tiktok"], rehearse=True))
        assert out.status == "done", (run, out.reason, [e["event"] for e in out.journal.events][-8:])
        assert all(t["matched"] for t in phone.log if t["op"] == "tap"), phone.log
    assert calls == ["claude-haiku-5-5"]
    rec = store.records["tiktok.search"]
    assert rec.source == "vlm" and rec.successes >= 2 and any("Search" in a or "You may like" in a for a in rec.match.ocr_any)


class _FakeMCP(ReplayPhone):
    """The walk's MCP phone, replayed."""
    seq = None

    def __init__(self, url, device, lease=None):
        super().__init__(self.seq)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def lock_state(self, udid=None):
        return {"locked": False}

    async def launch(self, udid, bundle_id, url=None, **_):
        pass


def test_finished_walk_counts_as_an_observing_session(tmp_path, monkeypatch):
    """A walk that finishes is a read-only session on the account (lifecycle OBSERVING); one that stops is not."""
    sessions, real_sleep = [], asyncio.sleep

    async def no_wait(s):
        await real_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", no_wait)  # the walk polls at human speed; replay needs no waiting
    monkeypatch.setattr(live, "record_session", sessions.append)
    monkeypatch.setattr(live, "calls_slice", lambda t0, t1: [])
    _FakeMCP.seq = _seq(tmp_path, KNOWN)
    monkeypatch.setattr(live, "MCPPhone", _FakeMCP)
    store = Store(records_dirs=RECORD_DIRS, corpus=tmp_path / "corpus")
    out = asyncio.run(live.walk(store, "iphone12", "tiktok", "http://x", None, tmp_path / "ev", use_vlm=False,
                                identity="tiktok:public"))
    assert out["status"] == "done" and out["session"] == "tiktok:public" and sessions == ["tiktok:public"]
    own = next(i["text"] for i in _tt("09-tt-profile").ocr if i["text"].startswith("@"))  # the recorded profile's own handle
    assert out["handle"] == own  # the signed-in account, read off its own profile
    assert "tiktok:public" in (result := (tmp_path / "ev" / "RESULT.md").read_text()) and own in result

    _FakeMCP.seq = _seq(tmp_path / "stuck", [("feed", None), ("search", "profile_tab")])  # unknown, no slow model
    out = asyncio.run(live.walk(store, "iphone12", "tiktok", "http://x", None, tmp_path / "ev2", use_vlm=False,
                                identity="tiktok:public"))
    assert out["status"] != "done" and out["session"] is None and sessions == ["tiktok:public"]


# --- the upload flow (records in the store; frames synthesized from the 2025-26 app's labels until a live
# rehearsal harvests real ones: mock) ------------------------------------------------------------------------------
def _screen(items, seed):
    # a mark per screen in the content area: identical PNGs would share one OCR on replay (looked up by hash)
    png = synth_frame(top="black", tab="black", body="black", icon=("gray", 100 + 60 * seed, 1200))
    return Frame.from_png(png, [_item(*i) for i in items])


# TikTok 47 on iOS as 2026-09 screen recordings show it (research 2026-10-18): the mode bar is one OCR line, the gallery
# tile carries no text. Live on the iPhone 12 (2026-10-19 04:06Z), TikTok auto-attached a sound, so its title
# replaced "Add sound" and the editor read "Aa", "AutoCut", "Story 24h" and "Next".
CAMERA = [("Add sound", 585, 253, 260), ("10m 60s 15s PHOTO TEXT", 470, 1848, 800), ("POST", 585, 2342, 120)]
PICKER = [("Recents", 585, 203, 160), ("Text", 1076, 203, 100), ("All Videos Photos Live Photos", 585, 342, 900),
          ("00:13", 320, 890, 90), ("00:12", 710, 890, 90), ("Select multiple", 269, 2355, 380), ("Next", 866, 2355, 120)]
EDITOR = [("Gitty Up", 585, 253, 220), ("Aa", 1088, 700, 60), ("AutoCut", 585, 2160, 220),
          ("Story 24h", 293, 2355, 220), ("Next", 878, 2355, 120)]
POST = [("Add description...", 234, 324, 420), ("Preview", 1023, 300, 140), ("Edit cover", 1023, 684, 180),
        ("# Hashtags", 175, 836, 200), ("@ Mention", 420, 836, 200), ("Location", 210, 1038, 180),
        ("+ Add link", 140, 1291, 200), ("Everyone can view this post", 421, 1436, 600), ("More options", 199, 1583, 260),
        ("Manage upload quality", 300, 1640, 380), ("Share to", 140, 1793, 160),
        ("I accept the Music Usage Confirmation", 351, 2228, 640), ("Drafts", 339, 2355, 220), ("Post", 901, 2355, 120)]
UPLOAD = [("feed", None), ("camera", "create"), ("picker", "upload"), ("editor", "next"), ("post", "next")]


def _upload_seq(tmp_path):
    frames = {"feed": _tt("08-tt-open"), "camera": _screen(CAMERA, 11), "picker": _screen(PICKER, 12),
              "editor": _screen(EDITOR, 13), "post": _screen(POST, 14)}
    return write_sequence(tmp_path / "upload", [
        (f"{i:02d}-{n}", frames[n].png, frames[n].ocr, {"action": {"kind": "tap", "target": t} if t else None})
        for i, (n, t) in enumerate(UPLOAD, 1)])


def test_tiktok_upload_rehearsal_stops_before_post(tmp_path):
    """feed → + → camera → Upload → picker → Next → editor → Next → the post screen: rehearse stops there, before
    the irreversible Post; Your Story (it posts a Story) and Drafts are never tapped."""
    from phonebridge.screens.flows import FLOWS
    phone = ReplayPhone(_upload_seq(tmp_path))
    store = Store(records_dirs=RECORD_DIRS, corpus=tmp_path / "corpus")
    out = asyncio.run(_engine(store, phone).run(FLOWS["tt_post"], rehearse=True))
    assert out.status == "rehearsed" and out.screen == "tiktok.post.final", (out.reason, out.journal.events[-4:])
    taps = [t for t in phone.log if t["op"] == "tap"]
    assert len(taps) == 4 and all(t["matched"] for t in taps), taps
    upload = next(e for e in out.journal.events if e.get("target") == "upload")
    assert upload["x"] < 0.15 * 1170 and upload["y"] > 0.88 * 2532  # the unlabelled gallery tile, bottom-left
    assert {t for t, x in store.records["tiktok.post.final"].targets.items() if x.irreversible} == {"post"}
    assert store.records["tiktok.upload.editor"].targets["your_story"].irreversible


def test_tiktok_post_screen_is_not_mistaken_for_others(tmp_path):
    """Each upload screen is recognised as itself only (and the real feed/profile frames as theirs)."""
    store = Store(records_dirs=RECORD_DIRS, corpus=tmp_path / "corpus")
    c = Cascade(store)
    for items, want in ((CAMERA, "tiktok.create.camera"), (PICKER, "tiktok.upload.picker"),
                        (EDITOR, "tiktok.upload.editor"), (POST, "tiktok.post.final")):
        assert c.recognise(_screen(items, 3), platform="tiktok", bundle="com.zhiliaoapp.musically").screen_id == want
    for name, want in (("08-tt-open", "tiktok.feed"), ("09-tt-profile", "tiktok.profile")):
        assert c.recognise(_tt(name), platform="tiktok", bundle="com.zhiliaoapp.musically").screen_id == want


@pytest.mark.parametrize("cmd", ["walk", "bench"])
def test_cli_walk_and_bench_print_and_exit_0(cmd, monkeypatch, capsys):
    """`phonebridge screens walk|bench` print the summary and exit 0 (after the bench merge, walk fell through to 2)."""
    import argparse
    from phonebridge.screens import cli

    async def fake(*a, **kw):
        return {"status": "done", "kw": sorted(kw)}
    monkeypatch.setattr(live, "walk", fake)
    monkeypatch.setattr(live, "bench", fake)
    p = argparse.ArgumentParser()
    cli.add_parser(p.add_subparsers(dest="cmd"))
    args = ["iphone12", "--identity", "tiktok:public"] if cmd == "walk" else ["--device", "iphone12"]
    a = p.parse_args(["screens", cmd, "--platform", "tiktok", *args])
    assert cli.run(a) == 0 and "'status': 'done'" in capsys.readouterr().out


LIVE = DATA_DIR / "evidence/M2-tiktok/20261019T031149Z-rehearse"


@pytest.mark.skipif(not LIVE.exists(), reason="needs lane tiktok's first live rehearsal (03:12Z 10-19)")
@pytest.mark.parametrize("frame,want", [("01-in-main tab", "tiktok.feed"), ("02-in-camera", "tiktok.create.camera"),
                                        ("04-stopped-unknown", "ios.alert.generic")])
def test_first_live_rehearsal_frames(frame, want, tmp_path):
    """The real frames: For You with the top tabs split by OCR ("Following" | "Friends For You"), TikTok's camera
    access page, and iOS's photo-library prompt (an alert: parks for the operator)."""
    store = Store(records_dirs=RECORD_DIRS, corpus=tmp_path / "corpus")
    m = Cascade(store).recognise(Frame.from_files(LIVE / f"{frame}.png"), platform="tiktok", bundle="com.zhiliaoapp.musically")
    assert m.screen_id == want, m.detail
