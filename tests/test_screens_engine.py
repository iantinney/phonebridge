"""SCR-07/09 + the Q3 triggers: the flow engine on the replay phone, no fixed sleeps, demotion, halted progress."""
import asyncio

import pytest

from phonebridge.backends.replay import Fault, ReplayPhone, write_sequence
from phonebridge.screens.cascade import Cascade
from phonebridge.screens.engine import Engine, Flow, Step, ListJournal, Escalate
from phonebridge.screens.schema import Record
from phonebridge.screens.store import Store
from screens_fixtures.synth import frame as synth_frame, PALETTE


def item(text, x, y, w=60, h=20):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": 1.0}


HOME = {"id": "instagram.home", "platform": "instagram", "bundle": "com.burbn.instagram", "status": "confirmed",
        "match": {"ocr_any": ["Instagram", "Your story"], "at_least": 2},
        "targets": {"create": {"ocr": "+", "region": [0, 0, 0.3, 0.1]}},
        "next": {"create": ["instagram.new_post.picker"]}, "popups": ["ios.allow_paste"]}
PICKER = {"id": "instagram.new_post.picker", "platform": "instagram", "bundle": "com.burbn.instagram",
          "status": "confirmed", "match": {"ocr_any": ["New post", "Recents", "Next"], "at_least": 2},
          "targets": {"next": {"ocr": "Next", "region": [0.6, 0, 1, 0.15]}}, "next": {"next": ["instagram.reel.final"]}}
FINAL = {"id": "instagram.reel.final", "platform": "instagram", "bundle": "com.burbn.instagram", "status": "confirmed",
         "match": {"ocr_any": ["New reel", "Write a caption", "Share"], "at_least": 2},
         "targets": {"share": {"ocr": "Share", "region": [0, 0.8, 1, 1], "irreversible": True}},
         "next": {"share": ["instagram.home"]}}
PASTE = {"id": "ios.allow_paste", "platform": "ios", "bundle": "*", "kind": "popup", "status": "confirmed",
         "match": {"ocr_any": ["Allow Paste", "Don't Allow"], "at_least": 2}, "answer": "allow",
         "targets": {"allow": {"ocr": "Allow Paste"}}}
FRAMES = [
    ("01-home", "instagram.home", None, [item("Instagram", 300, 90), item("Your story", 70, 300), item("+", 40, 90)]),
    ("02-picker", "instagram.new_post.picker", {"kind": "tap", "x": 40, "y": 90, "target": "create"},
     [item("New post", 585, 212), item("Next", 1058, 210), item("Recents", 167, 1560)]),
    ("03-final", "instagram.reel.final", {"kind": "tap", "x": 1058, "y": 210, "target": "next"},
     [item("New reel", 585, 212), item("Write a caption", 300, 600), item("Share", 585, 2300)]),
    ("04-home", "instagram.home", {"kind": "tap", "x": 585, "y": 2300, "target": "share"},
     [item("Instagram", 300, 90), item("Your story", 70, 300), item("+", 40, 90)]),
]
IG_REEL = Flow(name="ig_reel", platform="instagram", bundle="com.burbn.instagram", final="instagram.home", steps=[
    Step(at="instagram.home", do="tap", target="create"),
    Step(at="instagram.new_post.picker", do="tap", target="next"),
    Step(at="instagram.reel.final", do="tap", target="share"),
])


@pytest.fixture
def world(tmp_path):
    d = tmp_path / "sequences" / "ig_reel" / "1"
    write_sequence(d, [(n, synth_frame(top=PALETTE[i], seed=i), ocr, {"screen": s, "action": a})
                       for i, (n, s, a, ocr) in enumerate(FRAMES)])
    extra = tmp_path / "extra"
    write_sequence(extra, [("popup", synth_frame(top="gray", seed=9),
                            [item("Allow Paste", 585, 1500), item("Don't Allow", 585, 1600)], {"screen": "ios.allow_paste"}),
                           ("stale", synth_frame(top="gray", seed=8),
                            [item("New reel", 585, 212), item("Next", 1058, 210), item("Add topics", 300, 1000)],
                            {"screen": None})])
    store = Store(records=[Record.from_dict(r) for r in (HOME, PICKER, FINAL, PASTE)], corpus=tmp_path / "corpus")
    return d, extra, store


def engine(store, phone, **kw):
    return Engine(store=store, cascade=Cascade(store), phone=phone, eyes=phone.ocr, journal=ListJournal(),
                  sleep=lambda s: asyncio.sleep(0), **kw)


@pytest.mark.parametrize("flow", [IG_REEL], ids=["ig_reel"])
def test_flow_reaches_final_screen_on_replay(world, flow):
    d, _, store = world
    phone = ReplayPhone(d)
    approved = flow.__class__(**{**flow.__dict__, "approved": True})  # the publish task carries Ian's approval
    out = asyncio.run(engine(store, phone).run(approved))
    assert out.status == "done" and out.screen == "instagram.home"
    taps = [e for e in phone.log if e["op"] == "tap"]
    assert len(taps) == 3 and sum(1 for t in taps if t.get("target") == "share") == 1  # exactly one tap near Share
    steps = [e for e in store_journal(out) if e["event"] == "step"]
    assert [s["screen"] for s in steps] == ["instagram.home", "instagram.new_post.picker", "instagram.reel.final"]


def store_journal(out):
    return out.journal.events


def test_rehearse_stops_before_first_irreversible_tap(world):
    d, _, store = world
    phone = ReplayPhone(d)
    out = asyncio.run(engine(store, phone).run(IG_REEL, rehearse=True))
    assert out.status == "rehearsed" and out.screen == "instagram.reel.final"
    assert not any(t.get("target") == "share" for t in phone.log if t["op"] == "tap")


def test_popup_dismissed_with_recorded_answer(world):
    d, extra, store = world
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "popup.png", answer="Allow Paste")])
    out = asyncio.run(engine(store, phone).run(IG_REEL, rehearse=True))
    assert out.status == "rehearsed"
    assert any(e["event"] == "popup" and e["screen"] == "ios.allow_paste" for e in out.journal.events)


def test_stale_layout_demotes_after_two_failed_expectations(world):
    d, extra, store = world
    phone = ReplayPhone(d, faults=[Fault(at=2, kind="stale_layout", frame=extra / "stale.png")])
    e = engine(store, phone)
    with pytest.raises(Escalate):
        asyncio.run(e.run(IG_REEL))
    assert store.records["instagram.new_post.picker"].failures >= 2  # its counters say so; the slow model takes over
    assert not any(t.get("target") == "share" for t in phone.log if t["op"] == "tap")
    assert list((store.corpus / "unknown").rglob("*.png"))


def test_no_fixed_sleeps_settle_polls_frames(world):
    d, _, store = world
    phone = ReplayPhone(d)
    slept = []
    e = Engine(store=store, cascade=Cascade(store), phone=phone, eyes=phone.ocr, journal=ListJournal(),
               sleep=lambda s: (slept.append(s), asyncio.sleep(0))[1])
    asyncio.run(e.run(IG_REEL, rehearse=True))
    assert slept and max(slept) <= 0.5  # only short polls between looks, never a fixed step sleep
