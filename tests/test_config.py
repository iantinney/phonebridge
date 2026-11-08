"""What Config.load refuses: anything that would break one controller per phone or reach the lease command unchecked."""

import tomllib
from pathlib import Path

import pytest

from phonebridge.config import Config, ConfigError

A, B = "00008101-00000000000000AA", "00008101-00000000000000BB"
LEASE = '[lease]\ncmd = ["phonebridge-local-lease"]\nenforce = true\n'
SAFE = f'tunnel = "tunneld"\neyes = "stream"\n{LEASE}'


@pytest.fixture(autouse=True)
def no_eyes_override(monkeypatch):
    monkeypatch.delenv("PHONEBRIDGE_EYES", raising=False)


def load(tmp_path: Path, text: str) -> Config:
    path = tmp_path / "phonebridge.toml"
    path.write_text(text)
    return Config.load(path)


def refused(tmp_path: Path, text: str) -> str:
    with pytest.raises(ConfigError) as e:
        load(tmp_path, text)
    return "\n".join(e.value.problems)


def phones(*pairs, head=SAFE) -> str:
    return head + "".join(f'[devices."{a}"]\nudid = "{u}"\n' for a, u in pairs)


@pytest.mark.parametrize("second", [A, A.lower(), A.replace("-", ""), A.replace("-", "").lower()])
def test_one_phone_under_two_aliases_would_take_two_leases(tmp_path, second):
    assert "two leases" in refused(tmp_path, phones(("a", A), ("b", second)))


@pytest.mark.parametrize("extra", ['lease = "shared"', 'lease = "device:a"'])
def test_two_phones_cannot_share_a_lease(tmp_path, extra):
    text = SAFE + f'[devices.a]\nudid = "{A}"\nlease = "{"shared" if "shared" in extra else "device:a"}"\n' \
                  f'[devices.b]\nudid = "{B}"\n{extra}\n'
    assert "share lease" in refused(tmp_path, text)


def test_an_alias_shaped_like_another_phones_udid_is_refused(tmp_path):
    for alias in (A, A.replace("-", ""), A.lower(), "a" * 40, "A" * 40):  # however the driver would read it
        assert "wrong phone" in refused(tmp_path, phones((alias, B)))
    for alias in (B, B.replace("-", "")):  # naming a phone by its own UDID misdirects nothing
        assert list(load(tmp_path, phones((alias, B), head="")).devices) == [alias]


def test_aliases_differing_only_in_case_are_ambiguous(tmp_path):
    assert "differ only in case" in refused(tmp_path, phones(("phone", A), ("Phone", B)))


@pytest.mark.parametrize("alias", ["../x", "a/b", "a b", "$(reboot)", "-rf", "a;b", ""])
def test_unsafe_aliases_are_refused(tmp_path, alias):
    assert "alias" in refused(tmp_path, f'[devices."{alias}"]\nudid = "{A}"\n')


@pytest.mark.parametrize("line, says", [
    ('lease = "a b"', "lease"), ('lease = ""', "lease"), ("lease = 3", "lease"),
    ('supervision = "yolo"', "supervision"), ('owner = "me"', "unknown setting owner"),
])
def test_bad_device_settings_are_named(tmp_path, line, says):
    assert says in refused(tmp_path, f'[devices.a]\nudid = "{A}"\n{line}\n')


@pytest.mark.parametrize("text, says", [
    ('[devices.a]\nsupervision = "auto"\n', "needs udid"),
    ('[devices.a]\nudid = 7\n', "needs udid"),
    ('devices = "phone"\n', "devices"),
    ('[devices]\na = "x"\n', "table"),
    ('eyes = "x-ray"\n', "eyes"), ('tunnel = "wifi"\n', "tunnel"), ('input = "voice"\n', "input"),
    ('pointer_surface = "hide"\n', "pointer_surface"),
    ('[lease]\ncmd = "phonebridge-local-lease"\n', "cmd"),  # a string would run as a list of letters
    ('[lease]\nenforce = "false"\n', "enforce"),  # a non-empty string is truthy
    ('[policy]\nurls = "https://*"\n', "urls"),  # each letter would become a pattern, "*" among them
    ('[humanize]\npress = 3\n', "humanize"),
])
def test_malformed_settings_are_problems_not_tracebacks(tmp_path, text, says):
    assert says in refused(tmp_path, text)


def test_every_problem_is_reported_at_once(tmp_path):
    text = 'eyes = "x-ray"\n' + phones(("a", A), ("b", A), ("c/d", B), head="")
    problems = refused(tmp_path, text)
    assert all(s in problems for s in ("eyes", "two leases", "c/d"))


def test_invalid_toml_says_where(tmp_path):
    assert "line 1" in refused(tmp_path, "production config")


TWO = (("a", A), ("b", B))


@pytest.mark.parametrize("head, says", [
    (LEASE, "tunnel"), (LEASE, "eyes"),
    (f'tunnel = "auto"\neyes = "stream"\n{LEASE}', 'tunnel = "auto"'),
    (f'tunnel = "userspace"\neyes = "stream"\n{LEASE}', 'tunnel = "userspace"'),
    (f'tunnel = "tunneld"\neyes = "auto"\n{LEASE}', 'eyes = "auto"'),
    (f'tunnel = "tunneld"\neyes = "avf"\n{LEASE}', 'eyes = "avf"'),
    ('tunnel = "tunneld"\neyes = "stream"\n[lease]\nenforce = true\n', "enforced lease"),
    ('tunnel = "tunneld"\neyes = "stream"\n[lease]\ncmd = ["x"]\nenforce = false\n', "enforced lease"),
])
def test_several_phones_need_the_per_phone_transport_and_a_lease(tmp_path, head, says):
    assert says in refused(tmp_path, phones(*TWO, head=head))


@pytest.mark.parametrize("eyes", ["stream", "dvt"])
def test_per_phone_transport_loads(tmp_path, eyes):
    c = load(tmp_path, phones(*TWO, head=f'tunnel = "tunneld"\neyes = "{eyes}"\n{LEASE}'))
    assert (c.tunnel, c.eyes) == ("tunneld", eyes)


def test_the_eyes_override_counts_because_the_daemon_uses_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PHONEBRIDGE_EYES", "avf")
    assert "PHONEBRIDGE_EYES" in refused(tmp_path, phones(*TWO))
    monkeypatch.setenv("PHONEBRIDGE_EYES", "dvt")  # the kill switch stays usable
    assert len(load(tmp_path, phones(*TWO)).devices) == 2


@pytest.mark.parametrize("head", ['eyes = "avf"\ntunnel = "userspace"\n', "", '[lease]\nenforce = false\n'])
def test_one_phone_keeps_every_mode(tmp_path, head):
    assert list(load(tmp_path, phones(("a", A), head=head)).devices) == ["a"]


def test_seven_is_not_a_limit(tmp_path):
    many = [(f"p{n}", f"00008101-{n:016X}") for n in range(1, 11)]
    assert len(load(tmp_path, phones(*many)).devices) == 10


def test_a_udid_however_spelled_finds_the_configured_phone_and_its_lease(tmp_path):
    c = load(tmp_path, phones(*TWO))
    assert c.resolve(A.lower()).alias == "a" and c.resolve(B).lease == "device:b" and c.alias_of(B.lower()) == "b"
    assert c.resolve(A.replace("-", "")).lease == "device:a" and c.alias_of(B.replace("-", "").lower()) == "b"


def test_an_unconfigured_udid_still_has_no_lease(tmp_path):
    assert load(tmp_path, phones(*TWO)).resolve("00008101-00000000000000CC").lease is None
    with pytest.raises(ValueError, match="unknown device"):
        load(tmp_path, phones(*TWO)).resolve("nope")


def test_the_shipped_examples_load():
    root = Path(__file__).resolve().parents[1]
    for name in ("phonebridge.example.toml", "phonebridge.standalone.toml", "phonebridge.seven-phones.example.toml"):
        tomllib.loads((root / name).read_text())
        assert Config.load(root / name).devices
