from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from phonebridge import cli
from phonebridge.config import Config, ConfigError
from phonebridge.lease import LeaseError, Leases
from phonebridge.setup import check, configure


DEVICES = [{"udid": "00008101-AAAAAAAAAAAAAAAA"}, {"udid": "00008140-BBBBBBBBBBBBBBBB"}]
SEVEN = [{"udid": f"00008101-{n:016X}", "name": f"Test phone {n}", "model": "iPhone 12"} for n in range(1, 8)]
ALIASES = ["phone", *(f"phone-{n}" for n in range(2, 8))]
LISTING = {d["udid"]: [{"tunnel-address": "fd00::1"}] for d in SEVEN}
ROOT = Path(__file__).resolve().parents[1]
CUSTOM = f'''# my notes: don't lose this
tunnel = "tunneld"
eyes = "stream"   # chosen on purpose
pointer_surface = "remove"

[devices.office]
udid = "{SEVEN[0]["udid"]}"
lease = "phone:office-rig"
supervision = "hold-outward"

[lease]
cmd = ["my-lease-client", "--site", "mac"]
enforce = true

[host]
alert_cmd = ["my-notifier"]

[policy]
urls = ["https://example.com/*"]

[humanize]
seed = 7

[humanize.think]
tap = [300, 0.1, 100, 500]

[some_future_table]
kept = "yes"
'''


@pytest.fixture(autouse=True)
def no_eyes_override(monkeypatch):
    monkeypatch.delenv("PHONEBRIDGE_EYES", raising=False)


@pytest.fixture
def lease_on_path(monkeypatch):
    monkeypatch.setattr("phonebridge.setup.shutil.which", lambda c: f"/bin/{c}")


def test_setup_never_replaces_existing_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("production config")
    with pytest.raises(ConfigError, match="not valid TOML"):
        configure(DEVICES[:1], path)
    assert path.read_text() == "production config"


def test_multiple_phones_require_explicit_choice(tmp_path):
    path = tmp_path / "config.toml"
    with pytest.raises(ValueError, match="Several phones"):
        configure(DEVICES, path)
    assert not path.exists()
    configure(DEVICES, path, device=DEVICES[1]["udid"], alias="second")
    config = Config.load(path)
    assert config.devices["second"].udid == DEVICES[1]["udid"]
    assert config.lease_enforce is True
    assert config.lease_cmd == ["phonebridge-local-lease"]
    assert path.stat().st_mode & 0o777 == 0o600


def test_single_device_has_no_cloud_dependency(tmp_path):
    path = tmp_path / "config.toml"
    result = configure(DEVICES[:1], path)
    assert result["next"] == "phonebridge check phone"
    config = Config.load(path)
    assert config.devices["phone"].supervision == "read-only"
    assert config.devices["phone"].lease == "device:phone"
    assert (config.tunnel, config.eyes) == ("auto", "auto")  # one phone: nothing about its transport is touched


def test_invalid_alias_cannot_inject_config(tmp_path):
    with pytest.raises(ValueError):
        configure(DEVICES[:1], tmp_path / "config.toml", alias='phone]\nenforce=false\n[')


# --- seven phones ---------------------------------------------------------------------------------------------------
def test_all_gives_seven_phones_stable_unique_aliases_and_leases(tmp_path):
    a, b = tmp_path / "a.toml", tmp_path / "b.toml"
    result = configure(SEVEN, a, add_all=True, multi_phone="stream")
    configure(SEVEN[::-1], b, add_all=True, multi_phone="stream")
    assert a.read_bytes() == b.read_bytes()  # the order phones enumerate in changes nothing
    config = Config.load(a)
    assert list(config.devices) == ALIASES
    assert [config.devices[x].udid for x in ALIASES] == [d["udid"] for d in SEVEN]
    assert sorted(d.lease for d in config.devices.values()) == sorted(f"device:{x}" for x in ALIASES)
    assert {d.supervision for d in config.devices.values()} == {"read-only"}
    assert (config.tunnel, config.eyes, config.lease_enforce) == ("tunneld", "stream", True)
    assert [config.resolve(d["udid"]).alias for d in SEVEN] == ALIASES
    assert result["added"][3] == {"alias": "phone-4", "udid": SEVEN[3]["udid"], "lease": "device:phone-4",
                                  "name": "Test phone 4", "model": "iPhone 12"}
    assert result["settings_added"] == {"tunnel": "tunneld", "eyes": "stream"} and "Restart" in result["restart"]
    assert "tunneld" in result["next"] and "setup --check" in result["next"]


def test_rerunning_changes_nothing(tmp_path):
    path = tmp_path / "c.toml"
    configure(SEVEN, path, add_all=True, multi_phone="stream")
    before = path.read_bytes(), path.stat().st_mtime_ns
    result = configure(SEVEN[::-1], path, add_all=True, multi_phone="stream")
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert result["added"] == [] and sorted(result["already"]) == sorted(ALIASES) and "restart" not in result


def test_plugging_in_one_at_a_time_never_renames_an_earlier_phone(tmp_path):
    path = tmp_path / "c.toml"
    for n in range(1, 8):
        result = configure(SEVEN[:n], path, multi_phone="stream")  # no UDID to copy: the one new phone is unambiguous
        assert [x["alias"] for x in result["added"]] == [ALIASES[n - 1]] and "Restart" in result["restart"]
    config = Config.load(path)
    assert [config.devices[x].udid for x in ALIASES] == [d["udid"] for d in SEVEN]


def test_several_new_phones_still_need_an_explicit_choice_on_an_existing_config(tmp_path):
    path = tmp_path / "c.toml"
    configure(SEVEN[:1], path, multi_phone="stream")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="--all"):
        configure(SEVEN[:3], path)
    assert path.read_bytes() == before


def test_multi_phone_works_from_the_first_phone(tmp_path):
    path = tmp_path / "c.toml"
    first = configure(SEVEN[:1], path, multi_phone="stream")
    assert first["settings_added"] == {"tunnel": "tunneld", "eyes": "stream"} and "tunneld" in first["next"]
    second = configure(SEVEN[:2], path)  # already safe: no flag and no settings needed for the next phone
    assert "settings_added" not in second and [x["alias"] for x in second["added"]] == ["phone-2"]
    assert (Config.load(path).tunnel, len(Config.load(path).devices)) == ("tunneld", 2)


def test_a_second_phone_is_refused_until_the_operator_chooses_the_transport(tmp_path):
    path = tmp_path / "c.toml"
    configure(SEVEN[:1], path)
    before = path.read_bytes()
    with pytest.raises(ValueError) as e:
        configure(SEVEN[:2], path)
    assert 'tunnel = "tunneld"' in str(e.value) and 'eyes = "stream"' in str(e.value) and "--multi-phone" in str(e.value)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="--multi-phone"):
        configure(SEVEN[:2], tmp_path / "fresh.toml", add_all=True)
    assert not (tmp_path / "fresh.toml").exists()


def test_multi_phone_adds_only_what_the_file_lacks(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(f'eyes = "dvt"\n[devices.a]\nudid = "{SEVEN[0]["udid"]}"\n[lease]\ncmd = ["x"]\n')
    old = path.read_text()
    result = configure(SEVEN[:2], path, multi_phone="stream")
    assert result["settings_added"] == {"tunnel": "tunneld"} and old in path.read_text()
    assert (Config.load(path).eyes, Config.load(path).tunnel) == ("dvt", "tunneld")


def test_multi_phone_repairs_an_unsafe_config_without_adding_a_phone(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("".join(f'[devices.p{n}]\nudid = "{SEVEN[n]["udid"]}"\n' for n in range(2)) + '[lease]\ncmd = ["x"]\n')
    with pytest.raises(ConfigError):
        Config.load(path)
    result = configure(SEVEN[:2], path, multi_phone="stream")
    assert result["added"] == [] and len(Config.load(path).devices) == 2 and "Restart" in result["restart"]


@pytest.mark.parametrize("line", ['tunnel = "auto"', 'tunnel = "userspace"', 'eyes = "auto"', 'eyes = "avf"'])
@pytest.mark.parametrize("flag", [None, "stream"])
def test_a_conflicting_value_is_named_and_never_rewritten(tmp_path, line, flag):
    path = tmp_path / "c.toml"
    path.write_text(f'{line}\n[devices.a]\nudid = "{SEVEN[0]["udid"]}"\n[lease]\ncmd = ["x"]\n')
    before = path.read_bytes()
    with pytest.raises(ValueError) as e:
        configure(SEVEN[:2], path, multi_phone=flag)
    assert line in str(e.value) and "won't rewrite" in str(e.value)
    assert path.read_bytes() == before


# --- nothing already in the file is lost ------------------------------------------------------------------------------
def test_custom_settings_survive_byte_for_byte(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(CUSTOM)
    path.chmod(0o640)
    result = configure(SEVEN[:2], path)
    text = path.read_text()
    assert text.startswith(CUSTOM)  # nothing above the new table moved, comments included
    assert [x["alias"] for x in result["added"]] == ["phone"] and "settings_added" not in result
    new, old = tomllib.loads(text), tomllib.loads(CUSTOM)
    assert new["devices"].pop("phone") == {"udid": SEVEN[1]["udid"], "supervision": "read-only"}
    assert new == old
    config = Config.load(path)
    assert config.devices["office"].lease == "phone:office-rig" and config.devices["office"].supervision == "hold-outward"
    assert config.lease_cmd == ["my-lease-client", "--site", "mac"] and config.humanize.seed == 7
    assert config.pointer_surface == "remove" and config.urls == ["https://example.com/*"]
    assert path.stat().st_mode & 0o777 == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == [".c.toml.lock", "c.toml"]  # no temp file left behind


def test_transport_lines_go_above_the_file_and_leave_the_rest_alone(tmp_path):
    path = tmp_path / "c.toml"
    old = f'# notes\n[devices.a]\nudid = "{SEVEN[0]["udid"]}"\n\n[lease]\ncmd = ["x"]\n\n[host]\nalert_cmd = ["y"]\n'
    path.write_text(old)
    configure(SEVEN[:2], path, multi_phone="stream")
    text = path.read_text()
    assert old in text
    assert text.index('tunnel = "tunneld"') < text.index(old) < text.index("[devices.phone]")
    assert tomllib.loads(text)["host"] == {"alert_cmd": ["y"]}


@pytest.mark.parametrize("old", [
    f'tunnel = "tunneld"\neyes = "stream"\n[lease]\ncmd = ["x"]\n[devices.a]\nudid = "{SEVEN[0]["udid"]}"',  # no final newline
    f'tunnel = "tunneld"\r\neyes = "stream"\r\n[lease]\r\ncmd = ["x"]\r\n[devices.a]\r\nudid = "{SEVEN[0]["udid"]}"\r\n',
])
def test_line_endings_are_kept(tmp_path, old):
    path = tmp_path / "c.toml"
    path.write_bytes(old.encode())
    configure(SEVEN[:2], path)
    assert path.read_bytes().startswith(old.encode()) and len(Config.load(path).devices) == 2


def test_a_symlinked_config_stays_a_symlink(tmp_path):
    real, link = tmp_path / "dotfiles.toml", tmp_path / "c.toml"
    real.write_text(CUSTOM)
    link.symlink_to(real)
    configure(SEVEN[:2], link)
    assert link.is_symlink() and "[devices.phone]" in real.read_text()


def test_concurrent_adds_lose_no_phone(tmp_path):
    path = tmp_path / "c.toml"
    with ThreadPoolExecutor(7) as pool:
        results = list(pool.map(lambda d: configure(SEVEN, path, device=d["udid"], multi_phone="stream"), SEVEN))
    config = Config.load(path)
    assert sorted(d.udid for d in config.devices.values()) == sorted(d["udid"] for d in SEVEN)
    assert sorted(config.devices) == sorted(ALIASES) and all(len(r["added"]) == 1 for r in results)


@pytest.mark.parametrize("text", ["", "# only notes so far\n"])
def test_a_file_with_no_settings_is_treated_as_new(tmp_path, text):
    path = tmp_path / "c.toml"
    path.write_text(text)
    configure(SEVEN[:1], path)
    assert path.read_text().startswith(text) and Config.load(path).lease_cmd == ["phonebridge-local-lease"]


def test_a_layout_setup_cannot_extend_is_refused_intact(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(f'tunnel = "tunneld"\neyes = "stream"\ndevices = {{ a = {{ udid = "{SEVEN[0]["udid"]}" }} }}\n'
                    '[lease]\ncmd = ["x"]\n')
    before = path.read_bytes()
    with pytest.raises(ConfigError, match="by hand"):
        configure(SEVEN[:2], path)
    assert path.read_bytes() == before


# --- ambiguity ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("existing", ["office", "Office"])
def test_an_alias_in_use_is_never_repointed(tmp_path, existing):
    path = tmp_path / "c.toml"
    path.write_text(f'tunnel = "tunneld"\neyes = "stream"\n[devices.{existing}]\nudid = "{SEVEN[0]["udid"]}"\n'
                    '[lease]\ncmd = ["x"]\n')
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already names another phone"):
        configure(SEVEN, path, device=SEVEN[1]["udid"], alias="office")
    assert path.read_bytes() == before


def test_a_phone_is_never_given_a_second_alias(tmp_path):
    path = tmp_path / "c.toml"
    configure(SEVEN, path, device=SEVEN[0]["udid"], alias="office", multi_phone="stream")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already configured as 'office'"):
        configure(SEVEN, path, device=SEVEN[0]["udid"], alias="desk")
    with pytest.raises(ValueError, match="already configured"):
        configure(SEVEN, path, device=SEVEN[0]["udid"].lower(), alias="desk")  # the same phone, however it is typed
    with pytest.raises(ValueError, match="already configured"):
        configure(SEVEN, path, device=SEVEN[0]["udid"].replace("-", ""), alias="desk")
    assert path.read_bytes() == before
    assert configure(SEVEN, path, device=SEVEN[0]["udid"], alias="office")["added"] == []


def test_a_phone_written_without_its_hyphen_is_not_added_again(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(f'tunnel = "tunneld"\neyes = "stream"\n[devices.a]\nudid = "{SEVEN[0]["udid"].replace("-", "")}"\n'
                    '[lease]\ncmd = ["x"]\n')
    before = path.read_bytes()
    assert configure(SEVEN[:1], path, add_all=True)["already"] == ["a"] and path.read_bytes() == before
    assert [x["alias"] for x in configure(SEVEN[:2], path, add_all=True)["added"]] == ["phone"]


def test_generated_aliases_skip_names_already_taken(tmp_path):
    path = tmp_path / "c.toml"
    configure(SEVEN, path, device=SEVEN[0]["udid"], alias="phone-2", multi_phone="stream")
    configure(SEVEN[:3], path, add_all=True)
    assert sorted(Config.load(path).devices) == ["phone", "phone-2", "phone-3"]


def test_an_existing_duplicate_stops_setup_with_the_reason(tmp_path):
    path = tmp_path / "c.toml"
    u = SEVEN[0]["udid"]
    path.write_text(f'[devices.a]\nudid = "{u}"\n[devices.b]\nudid = "{u}"\n')
    with pytest.raises(ConfigError, match="two leases"):
        configure(SEVEN, path, add_all=True, multi_phone="stream")


def test_a_bad_udid_from_usb_writes_nothing(tmp_path):
    path = tmp_path / "c.toml"
    with pytest.raises(ValueError, match="invalid UDID"):
        configure([*SEVEN[:1], {"udid": "../../evil"}], path, add_all=True, multi_phone="stream")
    assert not path.exists()


def test_all_and_alias_contradict(tmp_path):
    with pytest.raises(ValueError, match="--all"):
        configure(SEVEN, tmp_path / "c.toml", add_all=True, alias="x", multi_phone="stream")


@pytest.mark.parametrize("settings", ["", "[lease]\nenforce = true\n", '[lease]\ncmd = ["x"]\nenforce = false\n'])
def test_setup_will_not_add_a_phone_without_an_enforced_lease(tmp_path, settings):
    path = tmp_path / "c.toml"
    path.write_text(f'[policy]\nurls = ["https://*"]\n{settings}')
    before = path.read_bytes()
    with pytest.raises(ConfigError, match="enforce = true"):
        configure(SEVEN[:1], path)
    assert path.read_bytes() == before


def test_no_phone_connected(tmp_path):
    with pytest.raises(ValueError, match="No matching phone"):
        configure([], tmp_path / "c.toml", add_all=True)
    with pytest.raises(ValueError, match="No matching phone"):
        configure(SEVEN, tmp_path / "c.toml", device="00008101-00000000000000FF")


# --- leases ---------------------------------------------------------------------------------------------------------
def test_seven_phones_hold_seven_independent_local_leases(tmp_path, monkeypatch):
    monkeypatch.setenv("PHONEBRIDGE_STATE", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    path = tmp_path / "c.toml"
    configure(SEVEN, path, add_all=True, multi_phone="stream")
    config = Config.load(path)
    cmd = [sys.executable, "-m", "phonebridge.local_lease"]
    first, second = Leases(cmd, holder="first"), Leases(cmd, holder="second")
    for alias in ALIASES:
        first.ensure(config.resolve(alias).lease)
    for d in SEVEN:  # asking by UDID lands on the same lease: there is no second name to go around it
        with pytest.raises(LeaseError):
            second.ensure(config.resolve(d["udid"]).lease)
    first.release("device:phone-4")
    second.ensure(config.resolve(SEVEN[3]["udid"]).lease)  # one phone freed, the other six still held
    assert second.holds("device:phone-4") and all(first.holds(f"device:{x}") for x in ALIASES if x != "phone-4")
    first.release_all()
    second.release_all()


# --- setup --check --------------------------------------------------------------------------------------------------
def seven(tmp_path):
    path = tmp_path / "c.toml"
    configure(SEVEN, path, add_all=True, multi_phone="stream")
    return path


def test_check_passes_when_all_seven_are_ready(tmp_path, lease_on_path):
    r = check(seven(tmp_path), SEVEN, LISTING)
    assert r["ok"] and r["problems"] == [] and [p["alias"] for p in r["phones"]] == ALIASES
    assert all(p["connected"] and p["tunneld"] for p in r["phones"]) and r["next"].startswith("phonebridge check phone ")


def test_check_names_each_missing_or_unready_phone(tmp_path, lease_on_path):
    devices = [SEVEN[0], {**SEVEN[1], "error": "PasswordProtected"}, {**SEVEN[2], "hint": "unlock the phone and tap Trust"},
               {**SEVEN[3], "developer_mode": False}, *SEVEN[4:6]]
    listing = {u: v for u, v in LISTING.items() if u != SEVEN[5]["udid"]}
    r = check(seven(tmp_path), devices, listing)
    assert not r["ok"]
    text = "\n".join(r["problems"])
    assert all(s in text for s in ("phone-2: PasswordProtected", "phone-3: unlock", "phone-4: Developer Mode",
                                   "phone-6: tunneld doesn't list", "phone-7: not connected"))
    assert "phone-5" not in text and "next" not in r


def test_check_says_how_to_start_tunneld(tmp_path, lease_on_path):
    r = check(seven(tmp_path), SEVEN, None)
    assert not r["ok"] and any("sudo uv run pymobiledevice3 remote tunneld" in p for p in r["problems"])


def test_check_reports_a_new_phone_without_failing(tmp_path, lease_on_path):
    path = tmp_path / "c.toml"
    configure(SEVEN[:1], path)
    r = check(path, SEVEN[:2], None)
    assert r["ok"] and r["unconfigured"] == [SEVEN[1]["udid"]]


def test_check_reports_config_problems_without_a_phone(tmp_path):
    path = tmp_path / "c.toml"
    assert check(path, [], None)["problems"] == ["no config here: run phonebridge setup"]
    u = SEVEN[0]["udid"]
    path.write_text(f'[devices.a]\nudid = "{u}"\n[devices.b]\nudid = "{u}"\n')
    r = check(path, [], None, "Could not list USB phones (Oops: x)")
    assert not r["ok"] and "two leases" in r["problems"][0]


def test_check_finds_an_unenforced_or_absent_lease_command(tmp_path, monkeypatch):
    path = tmp_path / "c.toml"
    configure(SEVEN[:1], path)
    monkeypatch.setattr("phonebridge.setup.shutil.which", lambda c: None)
    assert any("isn't on PATH" in p for p in check(path, SEVEN[:1], None)["problems"])
    path.write_text(f'[devices.a]\nudid = "{SEVEN[0]["udid"]}"\n[lease]\nenforce = false\n')
    assert any("no enforced device lease" in p for p in check(path, SEVEN[:1], None)["problems"])


def test_the_seven_phone_example_matches_what_setup_writes(tmp_path, lease_on_path):
    example = tomllib.loads((ROOT / "phonebridge.seven-phones.example.toml").read_text())
    written = tomllib.loads(seven(tmp_path).read_text())
    assert list(example["devices"]) == ALIASES
    for table in (example, written):
        for d in table["devices"].values():
            d.pop("udid")
    assert example == written


def test_check_flags_the_example_placeholders(lease_on_path):
    r = check(ROOT / "phonebridge.seven-phones.example.toml", [], LISTING)
    assert not r["ok"] and sum("isn't a real UDID" in p for p in r["problems"]) == 7


# --- the command line -----------------------------------------------------------------------------------------------
@pytest.fixture
def usb(monkeypatch):
    from phonebridge.backends.pmd3 import Pmd3Backend

    connected = list(SEVEN)

    async def devices(self):
        return connected

    async def listing(url, timeout):
        return LISTING

    monkeypatch.setattr(Pmd3Backend, "devices", devices)
    monkeypatch.setattr("phonebridge.presence.tunneld_listing", listing)
    monkeypatch.setattr("phonebridge.setup.shutil.which", lambda c: f"/bin/{c}")
    return connected


def run_cli(capsys, *args):
    with pytest.raises(SystemExit) as e:
        cli.main(["setup", *args])
    return e.value.code, json.loads(capsys.readouterr().out or "null")


def test_cli_adds_seven_then_checks_them(tmp_path, usb, capsys):
    path = str(tmp_path / "c.toml")
    code, out = run_cli(capsys, "--all", "--multi-phone", "stream", "--out", path)
    assert code == 0 and [x["alias"] for x in out["added"]] == ALIASES and "Restart" in out["restart"]
    code, out = run_cli(capsys, "--check", "--config", path)
    assert code == 0 and out["ok"] and len(out["phones"]) == 7


def test_cli_default_mode_for_a_bare_multi_phone_flag(tmp_path, usb, capsys):
    path = tmp_path / "c.toml"
    assert run_cli(capsys, "--all", "--multi-phone", "--out", str(path))[0] == 0
    assert Config.load(path).eyes == "stream"


def test_cli_check_exits_nonzero_with_the_reasons(tmp_path, usb, capsys):
    path = str(tmp_path / "c.toml")
    run_cli(capsys, "--all", "--multi-phone", "--out", path)
    del usb[2]
    code, out = run_cli(capsys, "--check", "--out", path)
    assert code == 1 and any("phone-3: not connected" in p for p in out["problems"])


def test_cli_refusal_is_a_plain_message(tmp_path, usb):
    with pytest.raises(SystemExit) as e:
        cli.main(["setup", "--all", "--out", str(tmp_path / "c.toml")])
    assert "--multi-phone" in str(e.value.code) and not (tmp_path / "c.toml").exists()


def test_cli_device_and_all_are_exclusive(tmp_path, usb, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["setup", "--all", "--device", SEVEN[0]["udid"], "--out", str(tmp_path / "c.toml")])
    assert e.value.code == 2 and "not allowed with" in capsys.readouterr().err


def test_setup_runs_with_no_usb_stack_only_to_say_so(tmp_path, monkeypatch):
    from phonebridge.backends.pmd3 import Pmd3Backend

    async def devices(self):
        raise ConnectionError("no usbmuxd")

    monkeypatch.setattr(Pmd3Backend, "devices", devices)
    with pytest.raises(SystemExit, match="Could not list USB phones"):
        cli.main(["setup", "--all", "--out", str(tmp_path / "c.toml")])


def test_an_unsafe_config_stops_the_bridge_with_its_reasons(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("".join(f'[devices.p{n}]\nudid = "{SEVEN[n]["udid"]}"\n' for n in range(2)))
    env = {**os.environ, "PHONEBRIDGE_CONFIG": str(path), "PHONEBRIDGE_STATE": str(tmp_path / "state"),
           "PYTHONPATH": str(ROOT / "src")}
    env.pop("PHONEBRIDGE_EYES", None)
    r = subprocess.run([sys.executable, "-c", "from phonebridge.cli import main; main()", "devices"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "Traceback" not in r.stderr
    assert all(s in r.stderr for s in (str(path), 'tunnel = "auto"', 'eyes = "auto"', "enforced lease"))
