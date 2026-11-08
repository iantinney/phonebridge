"""Device aliases and settings from ~/.config/phonebridge/phonebridge.toml (see phonebridge.example.toml)."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .humanize import HumanConfig
from .lease import SAFE

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
UDID_SHAPED = re.compile(r"[0-9a-f]{24}|[0-9a-f]{40}")  # matched against udid_key()
ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")  # one word in a lease identity, one component in a path
SUPERVISION = ("auto", "hold-outward", "read-only")
CHOICES = {"input": ("pmd3", "bthid"), "eyes": ("auto", "avf", "stream", "dvt"),
           "tunnel": ("auto", "tunneld", "userspace"), "pointer_surface": ("keep", "remove")}
# Two or more phones on one server. The in-process tunnel carries one phone per process, and AVF stays single-phone: one
# identity-bound capture per Mac; fleets use device-specific streams or stills instead.
MULTI_PHONE_TUNNEL = "tunneld"
MULTI_PHONE_EYES = ("stream", "dvt")


def udid_key(udid: str) -> str:
    """One phone's identity however its UDID is spelled; the driver compares UDIDs the same way."""
    return str(udid).replace("-", "").casefold()


class ConfigError(ValueError):
    """Everything wrong with one config file, each line saying how to fix it."""

    def __init__(self, where, problems: list[str]):
        self.problems = problems
        super().__init__(f"{where}:\n" + "\n".join(f"  - {p}" for p in problems))


def read_config(path: Path) -> str | None:
    """The file's text with its line endings intact; None when there is no file."""
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as e:
        raise ConfigError(path, [f"not UTF-8 text: {e}"]) from None


def integrity_problems(raw: dict) -> list[str]:
    """What would break one controller per phone, or reach the lease command unchecked; [] when `raw` can load."""
    devices, out = raw.get("devices", {}), []
    if not isinstance(devices, dict):
        return ["devices: write each phone as [devices.<alias>] with a udid"]
    first: dict[tuple[str, str], str] = {}

    def once(kind: str, value: str, alias: str, why: str) -> None:
        owner = first.setdefault((kind, udid_key(value) if kind == "udid" else value.casefold()), alias)
        if owner != alias:
            out.append(f"devices.{owner} and devices.{alias} share {kind} {value}: {why}")

    for alias, d in devices.items():
        if not ALIAS.fullmatch(alias):
            out.append(f"alias {alias!r}: use letters, digits, '.', '_' or '-', starting with a letter or digit")
            continue
        if not isinstance(d, dict):
            out.append(f"devices.{alias}: write it as a [devices.{alias}] table with a udid")
            continue
        once("alias", alias, alias, "aliases that differ only in case are ambiguous; rename one")
        if extra := sorted(d.keys() - {"udid", "lease", "supervision"}):
            out.append(f"devices.{alias}: unknown setting {', '.join(extra)} (allowed: udid, lease, supervision)")
        udid = d.get("udid")
        if not isinstance(udid, str) or not udid:
            out.append(f'devices.{alias}: needs udid = "..." (`phonebridge devices` prints it)')
        else:
            once("udid", udid, alias, "one phone under two aliases takes two leases, so two controllers could drive it; "
                 "keep one alias")
            if UDID_SHAPED.fullmatch(udid_key(alias)) and udid_key(alias) != udid_key(udid):
                out.append(f"alias {alias}: it looks like a UDID but names {udid}, so that UDID could reach the wrong "
                           "phone; rename it")
        lease = d.get("lease", f"device:{alias}")
        if not isinstance(lease, str) or not SAFE.fullmatch(lease):
            out.append(f"devices.{alias}: lease {lease!r}: use 1-200 of letters, digits and @:._/=+-")
        else:
            once("lease", lease, alias, "phones on one lease block each other, and a token for one passes for the other; "
                 "give each its own")
        if d.get("supervision", "auto") not in SUPERVISION:
            out.append(f"devices.{alias}: supervision {d['supervision']!r}: use one of {', '.join(SUPERVISION)}")
    for key, allowed in CHOICES.items():
        if raw.get(key, allowed[0]) not in allowed:
            out.append(f"{key} = {raw[key]!r}: use one of {', '.join(allowed)}")
    lease = raw.get("lease", {})
    if not isinstance(lease, dict):
        return [*out, "lease: write it as a [lease] table"]
    if not _strings(lease.get("cmd", [])):
        out.append('[lease] cmd: use a list of strings, like cmd = ["phonebridge-local-lease"]')
    if not isinstance(lease.get("enforce", True), bool):
        out.append("[lease] enforce: use true or false, without quotes")
    policy = raw.get("policy", {})
    if not isinstance(policy, dict) or not _strings(policy.get("urls", [])):  # a bare "https://*" would match char by char
        out.append('[policy] urls: use a list of patterns, like urls = ["https://*"]')
    return out


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v for v in value)


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
        text = read_config(path)
        return cls() if text is None else cls.from_text(text, path)

    @classmethod
    def from_text(cls, text: str, where: object = "config", multi_phone: bool = True) -> Config:
        """Parse and validate; a config that can't run safely raises ConfigError listing every problem.
        `multi_phone=False` skips the transport rule, for setup, which is how such a file gets repaired."""
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(where, [f"not valid TOML: {e}"]) from None
        if found := integrity_problems(raw):
            raise ConfigError(where, found)
        try:
            humanize = HumanConfig.from_toml(raw.get("humanize", {}))
        except (AttributeError, TypeError, ValueError) as e:
            raise ConfigError(where, [f"[humanize]: {e}"]) from None
        devices = {a: Device(alias=a, **{"lease": f"device:{a}", "supervision": "auto", **d})
                   for a, d in raw.get("devices", {}).items()}
        lease = raw.get("lease", {})
        config = cls(devices=devices, lease_cmd=lease.get("cmd", []), lease_enforce=lease.get("enforce", True),
                     urls=raw.get("policy", {}).get("urls", ["https://*"]), input=raw.get("input", "pmd3"),
                     eyes=raw.get("eyes", "auto"), tunnel=raw.get("tunnel", "auto"),
                     pointer_surface=raw.get("pointer_surface", "keep"), humanize=humanize)
        if multi_phone and (found := config.multi_phone_problems()):
            raise ConfigError(where, found)
        return config

    def multi_phone_problems(self) -> list[str]:
        """Why this config can't safely drive its phones from one server; [] for one phone, in any mode.
        PHONEBRIDGE_EYES counts: it is what the daemon would use."""
        n, out = len(self.devices), []
        if n < 2:
            return out
        adds = "`phonebridge setup --multi-phone` adds it if the file doesn't set one"
        if self.tunnel != MULTI_PHONE_TUNNEL:
            out.append(f'tunnel = "{self.tunnel}" with {n} phones: set tunnel = "{MULTI_PHONE_TUNNEL}" (the in-process '
                       f"tunnel carries one phone per server, and auto can fall back to it); {adds}")
        env = os.environ.get("PHONEBRIDGE_EYES")
        if (env or self.eyes) not in MULTI_PHONE_EYES:
            out.append(f'{"PHONEBRIDGE_EYES" if env else "eyes"} = "{env or self.eyes}" with {n} phones: use '
                       f'{" or ".join(MULTI_PHONE_EYES)} (AVFoundation is restricted to one attached phone); {adds}')
        if not (self.lease_cmd and self.lease_enforce):
            out.append(f'{n} phones without an enforced lease: under [lease] set cmd = ["phonebridge-local-lease"] and '
                       "enforce = true")
        return out

    def resolve(self, device: str) -> Device:
        """Alias or UDID -> Device. A UDID that isn't in the config is read-only: no lease, so no input."""
        if device in self.devices:
            return self.devices[device]
        if alias := self.alias_of(device):
            return self.devices[alias]
        if UDID.fullmatch(device):
            return Device(alias=device, udid=device)
        raise ValueError(f"unknown device {device!r}; aliases: {', '.join(self.devices) or 'none (see phonebridge.example.toml)'}")

    def alias_of(self, udid: str) -> str | None:
        return next((a for a, d in self.devices.items() if udid_key(d.udid) == udid_key(udid)), None)
