import json
import stat

from phonebridge.run.journal import Journal

TASK = {"task_id": "t1", "identity": "ig:public", "kind": "snapshot_status", "payload": {}}


def test_append_numbers_events_from_one_without_gaps(tmp_path):
    j = Journal.create(tmp_path, {**TASK, "leases": {"device": "secret"}})
    for step in ("open", "profile", "home"):
        j.append(step=step, outcome="ok")
    seqs = [e["seq"] for e in j.events()]
    assert seqs == list(range(1, len(seqs) + 1))
    assert [e["seq"] for e in j.events(after_seq=2)] == seqs[2:]
    assert all({"ts", "mono"} <= set(e) for e in j.events())


def test_task_file_never_holds_lease_tokens_and_is_private(tmp_path):
    j = Journal.create(tmp_path, {**TASK, "leases": {"device": "secret", "account": "secret2"}})
    text = (tmp_path / "t1" / "task.json").read_text()
    assert "secret" not in text and json.loads(text)["kind"] == "snapshot_status"
    assert j.task["identity"] == "ig:public"
    for f in (tmp_path / "t1" / "task.json", tmp_path / "t1" / "journal.jsonl"):
        assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "t1").stat().st_mode) == 0o700


def test_a_reopened_journal_continues_the_sequence(tmp_path):
    Journal.create(tmp_path, TASK).append(step="open", outcome="ok")
    j = Journal.open(tmp_path, "t1")
    j.append(step="profile", outcome="ok")
    assert [e["seq"] for e in j.events()] == [1, 2, 3]  # submitted, open, profile


def test_a_torn_last_line_is_dropped_and_repaired(tmp_path):
    j = Journal.create(tmp_path, TASK)
    j.append(step="open", outcome="ok")
    with open(tmp_path / "t1" / "journal.jsonl", "a") as f:
        f.write('{"seq": 3, "step": "prof')  # power lost mid-write
    j = Journal.open(tmp_path, "t1")
    assert [e["seq"] for e in j.events()] == [1, 2]
    j.append(step="profile", outcome="ok")
    assert [e["seq"] for e in Journal.open(tmp_path, "t1").events()] == [1, 2, 3]


def test_state_progress_and_completed_steps(tmp_path):
    j = Journal.create(tmp_path, TASK)
    assert j.status == "submitted" and not j.final
    j.append(state="running")
    j.append(step="pick", outcome="started")
    j.append(step="pick", outcome="ok", state="PICKED")
    j.append(step="stage", outcome="started")
    assert j.done_steps == ["pick"] and j.post_state == "PICKED" and j.status == "running"
    j.append(state="parked", reason="test")
    assert j.status == "parked" and not j.final
    j.append(state="done")
    assert j.final and Journal.all(tmp_path)[0].task["task_id"] == "t1"
