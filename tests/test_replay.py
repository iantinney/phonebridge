"""TST-01 / PLAN §9.3: ReplayPhone serves recorded frames, advances by the transition table, branches into faults."""
import asyncio
import json

import pytest

from phonebridge.backends.replay import Fault, ReplayPhone, write_sequence
from screens_fixtures.synth import frame as synth_frame, W, H, PALETTE


def item(text, x, y, w=60, h=20):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": 1.0}


@pytest.fixture
def seq(tmp_path):
    """home -(tap create)-> picker -(tap next)-> final; plus a popup frame and a lock frame for faults."""
    frames = [
        ("01-home", "instagram.home", None,
         [item("Instagram", 300, 90), item("Your story", 70, 300), item("+", 40, 90)]),
        ("02-picker", "instagram.new_post.picker", {"kind": "tap", "x": 40, "y": 90, "target": "create"},
         [item("New post", 585, 212), item("Next", 1058, 210), item("Recents", 167, 1560)]),
        ("03-final", "instagram.reel.final", {"kind": "tap", "x": 1058, "y": 210, "target": "next"},
         [item("New reel", 585, 212), item("Write a caption", 300, 600), item("Share", 585, 2300)]),
    ]
    d = tmp_path / "sequences" / "ig_reel" / "1"
    write_sequence(d, [(name, synth_frame(top=PALETTE[i], seed=i), ocr, {"screen": screen, "action": action})
                       for i, (name, screen, action, ocr) in enumerate(frames)])
    extra = tmp_path / "extra"
    write_sequence(extra, [("popup", synth_frame(top="gray", seed=9),
                            [item("Allow Paste", 585, 1500), item("Don't Allow", 585, 1600)], {"screen": "ios.allow_paste"}),
                           ("lock", synth_frame(top="black", body="black", seed=9),
                            [item("Swipe up to open", 585, 2400)], {"screen": "ios.lock"})])
    return d, extra


def run(coro):
    return asyncio.run(coro)


def test_replayphone_serves_sequence_and_branches_on_fault(seq):
    d, extra = seq
    phone = ReplayPhone(d)
    assert phone.ocr(run(phone.screen("replay")))[0]["text"] == "Instagram"
    run(phone.tap("replay", 500, 1200, 0.08))  # a tap nowhere near the recorded target does nothing
    assert phone.index == 0 and phone.log[-1]["matched"] is False
    run(phone.tap("replay", 42, 88, 0.08))
    assert phone.index == 1 and phone.current.meta["screen"] == "instagram.new_post.picker"
    run(phone.tap("replay", 1060, 212, 0.08))
    assert phone.index == 2 and phone.at_end

    # a popup fault at step 1: the popup frame is served instead of the picker until its answer is tapped
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="popup", frame=extra / "popup.png", answer="Allow Paste")])
    run(phone.tap("replay", 42, 88, 0.08))
    assert phone.ocr(run(phone.screen("replay")))[0]["text"] == "Allow Paste"
    run(phone.tap("replay", 585, 1500, 0.08))
    assert phone.current.meta["screen"] == "instagram.new_post.picker"

    # lost link raises on the next look; lock serves the lock frame; crash goes home and stays
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="lost_link")])
    run(phone.tap("replay", 42, 88, 0.08))
    with pytest.raises(ConnectionError):
        run(phone.screen("replay"))
    phone = ReplayPhone(d, faults=[Fault(at=1, kind="lock", frame=extra / "lock.png")])
    run(phone.tap("replay", 42, 88, 0.08))
    assert "Swipe up" in phone.ocr(run(phone.screen("replay")))[0]["text"]
    assert run(phone.lock_state("replay"))["locked"] is True


def test_replayphone_implements_backend_contract(seq):
    from phonebridge.bridge import Backend
    d, _ = seq
    phone = ReplayPhone(d)
    assert {n for n in dir(Backend) if not n.startswith("_")} <= {n for n in dir(phone)}
    assert run(phone.devices())[0]["udid"] == "replay"
    run(phone.home("replay")); run(phone.launch("replay", "com.burbn.instagram")); run(phone.close())
