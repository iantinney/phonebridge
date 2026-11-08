"""DEV-01/02/08, NET-01 preflight: a probe table, operator attestations with expiry; `None` is unknown, never pass."""

import json

import pytest

from phonebridge.host.preflight import CHECKS, Attestations, ok, preflight

DAY = 86400.0
T0 = 1_790_000_000.0


def passing(att):
    """A reading where every probed check passes, and every attested check is attested (True) at T0."""
    reading = {name: c.good for name, c in CHECKS.items() if c.source != "attestation"}
    for name, c in CHECKS.items():
        if c.source == "attestation":
            att.attest(name, now=T0)
    return reading


@pytest.fixture
def att(tmp_path):
    return Attestations(tmp_path / "attestations.json")


def by_check(results):
    return {r.check: r for r in results}


def test_all_pass_starts_and_the_reading_is_logged(att, tmp_path):
    log = tmp_path / "preflight.jsonl"
    results = preflight(passing(att), att, now=T0 + DAY, log=log)
    assert ok(results) and {r.status for r in results} == {"pass"}
    logged = json.loads(log.read_text())
    assert logged["ok"] and set(logged["results"]) == set(CHECKS)


def test_preflight_expired_attestation_is_unknown(att):
    reading = passing(att)
    results = by_check(preflight(reading, att, now=T0 + 31 * DAY))  # attestations last 30 days by default
    r = results["auto_lock_never"]
    assert r.status == "unknown" and "re-attest" in r.hint
    assert not ok(results.values())


def test_a_missing_reading_is_unknown_never_pass(att):
    reading = passing(att)
    del reading["battery_temp_c"]
    reading["unlocked"] = None
    results = by_check(preflight(reading, att, now=T0))
    assert results["battery_temp_c"].status == results["unlocked"].status == "unknown"
    assert not ok(results.values())


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_each_failing_check_stops_the_task_and_names_itself(att, name):
    reading = passing(att)
    check = CHECKS[name]
    if check.source == "attestation":
        att.attest(name, value=False, now=T0)
    else:
        reading[name] = check.bad
    results = preflight(reading, att, now=T0)
    failed = [r for r in results if r.status != "pass"]
    assert [r.check for r in failed] == [name] and failed[0].status == "fail" and failed[0].hint
    assert not ok(results)


def test_thresholds(att):
    reading = passing(att)
    reading.update(battery_temp_c=41.0, mac_disk_free_gb=4.0)  # DEV-08 pause above ~40 °C; the Mac's free disk
    results = by_check(preflight(reading, att, now=T0))
    assert results["battery_temp_c"].status == "fail" and results["mac_disk_free_gb"].status == "fail"
    reading.update(battery_temp_c=35.0, mac_disk_free_gb=600.0)
    assert ok(preflight(reading, att, now=T0))


def test_attestations_persist_with_who_and_when(att, tmp_path):
    att.attest("agent_focus", days=7, by="ian", now=T0)
    again = Attestations(tmp_path / "attestations.json")
    entry = again.entries["agent_focus"]
    assert entry["by"] == "ian" and entry["at"] == T0 and entry["expires"] == T0 + 7 * DAY and entry["value"] is True
    assert (tmp_path / "attestations.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(KeyError):
        att.attest("no_such_check")


def test_read_probes_storage_app_version_and_egress():
    import asyncio

    from phonebridge.host.preflight import read

    class Phone:
        async def lock_state(self, udid):
            return {"locked": False}

        async def storage(self, udid):
            return {"free_gb": 42.5, "total_gb": 64.0}

        async def app_version(self, udid, bundle_id):
            return "412.0" if bundle_id == "com.burbn.instagram" else None

    r = asyncio.run(read(Phone(), "U12", app="com.burbn.instagram", egress_ip="46.203.114.136", seen_ip="46.203.114.136"))
    assert r["free_space_gb"] == 42.5 and r["app_installed"] == "412.0" and r["egress_confirmed"] is True
    r = asyncio.run(read(Phone(), "U12", app="com.zhiliaoapp.musically", egress_ip="46.203.114.136", seen_ip="1.2.3.4"))
    assert "app_installed" not in r and r["egress_confirmed"] is False  # not installed: unknown; a home IP: fail


def test_egress_reads_the_ip_off_the_echo_page():
    import asyncio

    from phonebridge.host.preflight import egress

    class Bridge:
        opened = []

        async def open_url(self, device, url, lease=None):
            self.opened.append(url)

        async def ocr(self, device):
            return [{"text": "{"}, {"text": "“ip”: “46.203.114.136”,"}, {"text": "\"city\": \"Bellerose Terrace\","}]

    b = Bridge()
    assert asyncio.run(egress(b, "iphone12")) == "46.203.114.136" and b.opened == ["https://ipinfo.io/json"]
