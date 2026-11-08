"""Device aliases and settings from ~/.config/phonebridge/phonebridge.toml (see phonebridge.example.toml)."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .humanize import HumanConfig

try:  # a private deployment's own defaults (its directories, alert and model routes); the public release has none
    import phonebridge._deployment as _site
except ModuleNotFoundError as e:
    if e.name != "phonebridge._deployment":
        raise
    _site = None


def site(name: str, default):
    """The deployment's default for `name` when phonebridge/_deployment.py sets one, else the public `default`."""
    return getattr(_site, name, default)


CONFIG_DIR = Path(site("CONFIG_DIR", "~/.config/phonebridge")).expanduser()
CONFIG_PATH = Path(os.environ.get("PHONEBRIDGE_CONFIG", CONFIG_DIR / "phonebridge.toml")).expanduser()
STATE_DIR = Path(os.environ.get("PHONEBRIDGE_STATE", "~/.local/state/phonebridge")).expanduser()
DATA_DIR = Path(site("DATA_DIR", STATE_DIR)).expanduser()  # evidence, the screens corpus, model caches
UDID = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{16}|[0-9a-f]{40}")


@dataclass(frozen=True)
class Device:
    alias: str
    udid: str
    lease: str | None = None  # lease identity held while sending input; from the config: device:<alias>
    supervision: str = "read-only"  # auto | hold-outward | read-only (advisory; enforced by the agent/lease service)


@dataclass
class Config:
    devices: dict[str, Device] = field(default_factory=dict)
    lease_cmd: list[str] = field(default_factory=list)  # e.g. ["phonebridge-local-lease"]
    lease_enforce: bool = True  # no lease_cmd while enforcing => input is refused (fail closed)
    urls: list[str] = field(default_factory=lambda: ["https://*"])  # open_url allowlist (fnmatch patterns)
    input: str = "pmd3"  # pmd3 (CoreDevice HID, Developer Mode) | bthid (the Mac as a Bluetooth keyboard + mouse)
    eyes: str = "auto"  # auto (avf on macOS) | avf (QuickTime-style capture) | stream (CoreDevice display stream) | dvt (stills only)
    tunnel: str = "auto"  # auto (tunneld if it lists the phone, else in-process) | tunneld (only) | userspace (in-process only)
    pointer_surface: str = "keep"  # keep | remove: drop CoreDevice's Mouse-usage surface (pages/apps see a mouse otherwise)
    humanize: HumanConfig = field(default_factory=HumanConfig)  # [humanize]: how gestures and pauses are drawn

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> Config:
        if not path.exists():
            return cls()
        raw = tomllib.loads(path.read_text())
        devices = {a: Device(alias=a, **{"lease": f"device:{a}", "supervision": "auto", **d})
                   for a, d in raw.get("devices", {}).items()}
        lease = raw.get("lease", {})
        return cls(devices=devices, lease_cmd=lease.get("cmd", []), lease_enforce=lease.get("enforce", True),
                   urls=raw.get("policy", {}).get("urls", ["https://*"]), input=raw.get("input", "pmd3"),
                   eyes=raw.get("eyes", "auto"), tunnel=raw.get("tunnel", "auto"),
                   pointer_surface=raw.get("pointer_surface", "keep"),
                   humanize=HumanConfig.from_toml(raw.get("humanize", {})))

    def resolve(self, device: str) -> Device:
        """Alias or UDID -> Device. A UDID that isn't in the config is read-only: no lease, so no input."""
        if device in self.devices:
            return self.devices[device]
        for d in self.devices.values():
            if d.udid == device:
                return d
        if UDID.fullmatch(device):
            return Device(alias=device, udid=device)
        raise ValueError(f"unknown device {device!r}; aliases: {', '.join(self.devices) or 'none (see phonebridge.example.toml)'}")

    def alias_of(self, udid: str) -> str | None:
        return next((a for a, d in self.devices.items() if d.udid == udid), None)
