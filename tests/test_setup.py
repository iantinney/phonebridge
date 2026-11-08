from pathlib import Path

import pytest

from phonebridge.config import Config
from phonebridge.setup import configure


DEVICES = [{"udid": "00008101-AAAAAAAAAAAAAAAA"}, {"udid": "00008140-BBBBBBBBBBBBBBBB"}]


def test_setup_never_replaces_existing_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("production config")
    with pytest.raises(FileExistsError):
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


def test_invalid_alias_cannot_inject_config(tmp_path):
    with pytest.raises(ValueError):
        configure(DEVICES[:1], tmp_path / "config.toml", alias='phone]\nenforce=false\n[')
