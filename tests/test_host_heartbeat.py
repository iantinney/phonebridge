"""REC-05 heartbeat (Mac -> VM every 60 s while awake) and the Mac-side log caps."""

import asyncio
import json

from phonebridge.host.heartbeat import Heartbeat, cap_logs


class Sender:
    def __init__(self, fail=0):
        self.sent, self.fail = [], fail

    def __call__(self, beat):
        if self.fail:
            self.fail -= 1
            raise OSError("ssh: connect to host oracle: Network is unreachable")
        self.sent.append(beat)


def test_a_beat_carries_the_state_the_task_and_the_boot(tmp_path):
    send, state = Sender(), {"s": "available", "task": {"iphone12": {"app": "com.burbn.instagram"}}}
    hb = Heartbeat(send, state=lambda: state["s"], tasks=lambda: state["task"], boot=1234.5,
                   log=tmp_path / "beats.jsonl", mac=lambda: {"battery": 88, "ac": True, "disk_free_gb": 677})
    hb.beat()
    state["s"], state["task"] = "no_usb", {}
    hb.beat()
    hb.beat("sleeping")  # from the will-sleep handler, before the Mac goes down
    assert [(b["state"], bool(b["tasks"])) for b in send.sent] == [("available", True), ("no_usb", False),
                                                                   ("sleeping", False)]
    assert send.sent[0]["boot"] == 1234.5 and send.sent[0]["mac"]["battery"] == 88 and "ts" in send.sent[0]
    assert all(set(b) >= {"ts", "state", "tasks", "boot", "every", "proto"} for b in send.sent)
    Heartbeat(send, state=lambda: "available", tasks=dict, boot=1.0, log=tmp_path / "b.jsonl", extra={"planned": True}).beat()
    assert send.sent[-1]["planned"] is True  # this boot was a deploy's restart, not a crash


def test_failed_sends_back_off_and_are_logged_once_per_change(tmp_path):
    send = Sender(fail=3)
    log = tmp_path / "beats.jsonl"
    hb = Heartbeat(send, state=lambda: "available", tasks=dict, boot=1.0, log=log, every=60)
    delays = [hb.beat() for _ in range(5)]
    assert delays == [120, 240, 480, 60, 60]  # 60 s normally; doubling after each failure (cap 600)
    lines = [json.loads(line) for line in log.open()]
    assert [line["event"] for line in lines] == ["send_failed", "send_ok"]  # not one line per failure
    assert lines[0]["state"] == "tunnel_down" and len(send.sent) == 2


def test_run_beats_until_stopped(tmp_path):
    send = Sender()
    hb = Heartbeat(send, state=lambda: "available", tasks=dict, boot=1.0, log=tmp_path / "b.jsonl", every=0.02)

    async def go():
        task = asyncio.create_task(hb.run())
        await asyncio.sleep(0.07)
        task.cancel()

    asyncio.run(go())
    assert 3 <= len(send.sent) <= 5


def test_logs_are_capped_in_place(tmp_path):
    """A daemon log held open by launchd (O_APPEND) is truncated in place, its tail kept beside it (bthid.log hit
    13 GB on 2026-10-18 from a reconnect loop)."""
    big, small = tmp_path / "bthid.log", tmp_path / "phonebridge.log"
    big.write_bytes(b"x" * 5000 + b"\nlast line\n")
    small.write_bytes(b"ok\n")
    with big.open("ab") as writer:
        capped = cap_logs([big, small, tmp_path / "missing.log"], max_bytes=4096, keep=64)
        writer.write(b"after\n")
        writer.flush()
    assert capped == [big]
    assert big.read_bytes() == b"after\n"  # the writer's next line lands at the start, no sparse hole
    assert (tmp_path / "bthid.log.1").read_bytes().endswith(b"last line\n")
    assert len((tmp_path / "bthid.log.1").read_bytes()) == 64 and small.read_bytes() == b"ok\n"


def test_screen_p95_over_the_last_5_minutes(tmp_path):
    """REC-08: Given screen p95 2.5 s over 5 min, Then alert (the daemon sends one an hour)."""
    from phonebridge.host.heartbeat import screen_p95

    calls, now = tmp_path / "calls.jsonl", 10_000.0
    rows = [{"ts": now - 400, "op": "screen", "ok": True, "ms": 9000}]  # outside the window
    rows += [{"ts": now - 250 + i, "op": "screen", "ok": True, "ms": 2500} for i in range(19)]
    rows += [{"ts": now - 10, "op": "screen", "ok": True, "ms": 400}, {"ts": now - 5, "op": "tap", "ok": True, "ms": 99}]
    calls.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p95, n = screen_p95(calls, now)
    assert n == 20 and p95 == 2.5
    calls.write_text("".join(json.dumps({"ts": now - i, "op": "screen", "ok": True, "ms": 500}) + "\n" for i in range(20)))
    assert screen_p95(calls, now) == (0.5, 20)
    assert screen_p95(tmp_path / "missing.jsonl", now) == (0.0, 0)
