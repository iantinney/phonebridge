"""Standalone lease admission: races, expiry, and actual CLI compatibility."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

import pytest

from phonebridge.lease import LeaseError, Leases
from phonebridge.local_lease import LeaseDenied, operate


def test_concurrent_controllers_cannot_both_acquire(tmp_path):
    db = tmp_path / "leases.sqlite"

    def contender(_):
        try:
            return operate(db, "acquire", "device:phone", now=100)
        except LeaseDenied:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(contender, range(16)))
    assert len([r for r in results if r]) == 1


def test_expiry_foreign_device_and_stale_release(tmp_path):
    db = tmp_path / "leases.sqlite"
    first = operate(db, "acquire", "device:a", ttl=10, now=100)
    with pytest.raises(LeaseDenied):
        operate(db, "check", first, identity="device:b", now=101)
    with pytest.raises(LeaseDenied):
        operate(db, "renew", first, now=111)
    second = operate(db, "acquire", "device:a", now=111)
    operate(db, "release", first, now=112)
    assert float(operate(db, "check", second, identity="device:a", now=112)) == 119
    assert first.encode() not in db.read_bytes()
    assert second.encode() not in db.read_bytes()
    assert db.stat().st_mode & 0o777 == 0o600


def test_independent_devices_and_renewal(tmp_path):
    db = tmp_path / "leases.sqlite"
    a = operate(db, "acquire", "device:a", now=100)
    b = operate(db, "acquire", "device:b", now=100)
    assert a != b
    operate(db, "renew", a, ttl=200, now=110)
    assert float(operate(db, "check", a, identity="device:a", now=130)) == 180
    operate(db, "release", a, now=131)
    with pytest.raises(LeaseDenied):
        operate(db, "check", a, identity="device:a", now=132)
    assert float(operate(db, "check", b, identity="device:b", now=132)) == 88


def test_bridge_lease_protocol_via_separate_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("PHONEBRIDGE_STATE", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    cmd = [sys.executable, "-m", "phonebridge.local_lease"]
    first = Leases(cmd, holder="first")
    second = Leases(cmd, holder="second")
    first.ensure("device:phone")
    token = first._held["device:phone"][0]
    assert second.verify(token, "device:phone", cache_s=0) > 0
    with pytest.raises(LeaseError):
        second.ensure("device:phone")
    first.release_all()
    second.ensure("device:phone")
    assert second.holds("device:phone")
    second.release_all()


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan"), 3601])
def test_invalid_ttl(tmp_path, ttl):
    with pytest.raises(LeaseDenied):
        operate(tmp_path / "leases.sqlite", "acquire", "device:a", ttl=ttl)


def test_does_not_follow_database_symlink(tmp_path):
    target = tmp_path / "other"
    target.write_text("unrelated")
    link = tmp_path / "leases.sqlite"
    link.symlink_to(target)
    with pytest.raises(OSError):
        operate(link, "acquire", "device:a")
    assert target.read_text() == "unrelated"
