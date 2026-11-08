"""SCR-01: records are data; targets are anchor-relative, template-located or region-bounded, never bare coordinates."""
import pytest

from phonebridge.screens.schema import Record, SchemaError

BASE = {"id": "instagram.home", "platform": "instagram", "bundle": "com.burbn.instagram",
        "match": {"ocr_any": ["Instagram", "Your story"], "at_least": 2}}


def test_record_schema_rejects_bare_coordinates():
    for bad in ([0.5, 0.9], {"x": 100, "y": 200}, {"at": [0.5, 0.9]}, {"point": [84, 218]}):
        with pytest.raises(SchemaError):
            Record.from_dict({**BASE, "targets": {"create": bad}})
    # a fallback fraction exists only for icons (template-located targets)
    with pytest.raises(SchemaError):
        Record.from_dict({**BASE, "targets": {"create": {"ocr": "+", "fallback": [0.06, 0.065]}}})


def test_record_accepts_anchor_template_and_region_targets():
    r = Record.from_dict({**BASE, "targets": {
        "create": {"template": "instagram/create.png", "region": [0, 0.03, 0.2, 0.1], "fallback": [0.07, 0.086]},
        "post": {"ocr": "Share", "region": [0, 0.75, 1, 1], "irreversible": True},
        "caption": {"ocr_any": ["Write a caption", "Add a caption"]},
    }})
    assert r.targets["post"].irreversible and r.targets["create"].fallback == (0.07, 0.086)
    assert r.status == "candidate" and r.kind == "screen"


def test_record_round_trips_through_yaml(tmp_path):
    r = Record.from_dict({**BASE, "targets": {"create": {"ocr": "+", "region": [0, 0, 0.3, 0.1]}},
                          "next": {"create": ["instagram.new_post.picker"]}, "status": "confirmed",
                          "popups": ["ios.allow_paste"], "way_out": ["back", "home"], "timeout_s": 8})
    p = tmp_path / "home.yaml"
    r.dump(p)
    assert Record.load(p) == r


def test_record_rejects_unknown_kind_status_and_regions():
    with pytest.raises(SchemaError):
        Record.from_dict({**BASE, "kind": "thing"})
    with pytest.raises(SchemaError):
        Record.from_dict({**BASE, "status": "maybe"})
    with pytest.raises(SchemaError):
        Record.from_dict({**BASE, "mask": [[0, 0, 2, 0.06]]})


def test_stop_record_needs_a_reason_and_popup_an_answer():
    with pytest.raises(SchemaError):
        Record.from_dict({**BASE, "id": "instagram.stop.captcha", "kind": "stop"})
    with pytest.raises(SchemaError):
        Record.from_dict({**BASE, "id": "ios.allow_paste", "kind": "popup", "targets": {"allow": {"ocr": "Allow Paste"}}})
    r = Record.from_dict({**BASE, "id": "ios.allow_paste", "kind": "popup", "answer": "allow",
                          "targets": {"allow": {"ocr": "Allow Paste"}}})
    assert r.answer == "allow"
