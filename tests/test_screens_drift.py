"""SCR-12: the unknown rate per day and the doubling alert, from RESULT.md summaries."""
import json

from phonebridge.screens import drift


def run(tmp_path, name, platform, steps, unknown, vlm=0, status="done"):
    d = tmp_path / name
    d.mkdir()
    s = {"status": status, "frames": 0 if status == "escalate" else steps + 1, "steps": steps, "unknown_events": unknown, "vlm_calls": vlm}
    (d / "RESULT.md").write_text(f"# walk\n- device: x, platform: {platform}, flow: f\n- summary: {json.dumps(s)}\n")
    (d / "events.jsonl").write_text("\n".join(json.dumps({"event": "unknown", "context": f"screen {i}"}) for i in range(unknown)))


def test_rate_per_day_and_alert_on_doubling(tmp_path):
    run(tmp_path, "20261018T100000Z", "instagram", steps=9, unknown=1)
    run(tmp_path, "20261018T110000Z", "instagram", steps=0, unknown=0, status="escalate")  # a USB-loss park: ignored
    run(tmp_path, "20261019T100000Z", "instagram", steps=6, unknown=4, vlm=4)
    rep = drift.report(drift.read_runs(tmp_path))
    assert rep["instagram"]["20261018"]["rate"] == 0.1 and rep["instagram"]["20261018"]["runs"] == 1
    assert rep["instagram"]["20261019"]["rate"] == 0.4 and rep["instagram"]["20261019"]["vlm_calls"] == 4
    assert drift.alerts(rep) and "screen 0" in drift.alerts(rep)[0]
    assert "Canary" in drift.render(rep)
