"""The exported workspace must acquire the daemon's real local lease, not private infrastructure."""

import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
from phonebridge import local_lease  # required: missing standalone support must fail collection, never skip

from teachable_worker import missions, standalone, workspace


@pytest.fixture
def local(monkeypatch, tmp_path):
    command = [sys.executable, "-m", "phonebridge.local_lease", "lease"]
    monkeypatch.setenv("PHONEBRIDGE_STATE", str(tmp_path / "phonebridge"))
    monkeypatch.setenv("TW_PHONE_ADAPTER", "local")
    monkeypatch.setenv("TW_LEASE_CMD", shlex.join(command))
    def private_lease(*args, **kwargs):
        pytest.fail("standalone tasks must never call authhub")

    if missions._site is not None:  # the private deployment's session must stay unreachable here
        monkeypatch.setattr(missions._site, "AUTHHUB_PY", tmp_path / "absent-private-python")
        monkeypatch.setattr(missions._site, "FOCUSGATE", str(tmp_path / "absent-private-focusgate"))
        monkeypatch.setattr(missions._site, "_lease", private_lease)

    class Hands:
        def __init__(self, device, token):
            self.device, self.token, self.released = device, token, False

        def check(self):
            return subprocess.run([*command, "check", self.token, f"device:{self.device}"],
                                  capture_output=True, text=True).returncode

        def release(self):
            assert self.check() == 0  # controller release happens before device release
            self.released = True

    monkeypatch.setattr(missions, "Hands", Hands)
    return lambda device="phone": SimpleNamespace(device={"id": device}, task={"id": "local-test"})


def test_local_tasks_share_the_real_daemon_lease(local):
    ctx = local()
    assert workspace.session_hook() is standalone.session
    assert missions.AccountCheck().available() == (True, "")
    assert missions.FeedScout().available() == (True, "")
    with workspace.session_hook()(ctx):
        assert ctx.hands.check() == 0
        with pytest.raises(workspace.Busy):
            with standalone.session(local()):
                pytest.fail("second controller entered an occupied phone")
        with standalone.session(local("second-phone")) as other:
            assert other.hands.check() == 0
        assert ctx.hands.check() == 0
    assert ctx.hands.released
    assert ctx.hands.check() != 0


def test_local_exception_releases_controller_then_lease(local):
    ctx = local()
    with pytest.raises(RuntimeError, match="executor failed"):
        with standalone.session(ctx):
            raise RuntimeError("executor failed")
    assert ctx.hands.released
    with standalone.session(local()):
        pass  # another worker can immediately acquire it


def test_local_configuration_has_no_private_account_defaults(local, monkeypatch):
    monkeypatch.delenv("TW_PHONE_ACCOUNTS", raising=False)
    monkeypatch.delenv("TW_ACCOUNT_HANDLES", raising=False)
    r = subprocess.run([sys.executable, "-c", "from teachable_worker import missions; import json; "
                        "print(json.dumps([missions.ACCOUNTS, missions.ACCOUNT_HANDLES]))"],
                       capture_output=True, text=True, check=True)
    assert json.loads(r.stdout) == [{}, {}]


def test_missing_local_lease_is_a_prerequisite_not_a_busy_queue(local, monkeypatch, tmp_path):
    monkeypatch.setenv("TW_LEASE_CMD", shlex.quote(str(tmp_path / "missing-lease")) + " lease")
    ok, reason = missions.AccountCheck().available()
    assert not ok and "Install Phonebridge" in reason


def test_private_session_selection_is_preserved(monkeypatch):
    if missions.session is None:
        pytest.skip("no private deployment in this install")
    monkeypatch.delenv("TW_PHONE_ADAPTER", raising=False)
    assert workspace.session_hook() is missions.session


def test_lease_renewal_timeout_stops_the_task_without_logging_its_token(local, monkeypatch):
    ctx = local()
    session = workspace.LeaseSession(ctx)
    session.token = "x" * 20  # a stand-in lease token
    session._stop = SimpleNamespace(wait=lambda _: False)
    lost = []
    monkeypatch.setattr(workspace, "lose_session", lambda task, why: lost.append((task, why)))

    def timed_out(*args):
        raise subprocess.TimeoutExpired(["lease", "renew", session.token], 30)

    monkeypatch.setattr(session, "_run", timed_out)
    session._renew()
    assert session.token is None
    assert lost == [(ctx.task["id"], "lease renewal failed (TimeoutExpired)")]


def test_the_launcher_serves_the_workspace_and_teach_locally(monkeypatch, tmp_path, capsys):
    import os

    from teachable_worker import httpd, server

    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != "TW_PHONE_ADAPTER"})  # main sets it
    monkeypatch.setenv("TW_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("TW_LEASE_CMD", "absent-lease-command lease")
    served = []
    monkeypatch.setattr(httpd, "serve", lambda routes, host, port, *a, **k: served.append((routes, host, port)))
    assert standalone.main(["--port", "8799"]) == 0
    out = capsys.readouterr()
    (routes, host, port), = served
    assert (host, port) == ("127.0.0.1", 8799) and standalone.enabled()
    token = workspace.ensure_operator()["token"]
    assert f"http://127.0.0.1:8799/t/{token}" in out.out and token != httpd.admin_token()
    assert httpd.is_operator(httpd.person_by_token(token))  # drives the phones, and is a person for Teach
    paths = {r[1] for r in routes}
    assert {"/api/devices", "/api/tasks", "/api/teach/sessions", "/api/routines/{id}", "/api/media"} <= paths
    assert not paths & {"/api/jobs", "/api/metrics", "/api/ops/overview"}  # no agency product, no scorecard
    assert "Phone tasks are off" in out.err  # the lease command isn't installed: said, not hidden


def test_release_timeout_preserves_outcome_and_never_logs_token(local, monkeypatch, caplog):
    session = workspace.LeaseSession(local())
    with pytest.raises(ValueError, match="original executor outcome"):
        with session:
            token = session.token

            def timed_out(*args):
                raise subprocess.TimeoutExpired(["lease", "release", token], 30)

            monkeypatch.setattr(session, "_run", timed_out)
            raise ValueError("original executor outcome")
    assert session.token is None
    assert "TimeoutExpired" in caplog.text
    assert token not in caplog.text
