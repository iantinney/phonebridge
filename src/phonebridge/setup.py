"""Create or extend a standalone config. Whatever is already in the file stays, byte for byte."""

import asyncio
from contextlib import contextmanager
import fcntl
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import tomllib

from .config import MULTI_PHONE_EYES, MULTI_PHONE_TUNNEL, UDID, Config, ConfigError, read_config, udid_key

NEW_ALIAS = re.compile(r"[a-z][a-z0-9_-]{0,31}")
NEW_FILE_TAIL = '[lease]\ncmd = ["phonebridge-local-lease"]\nenforce = true\n\n[policy]\nurls = ["https://*"]\n'
TUNNELD = "sudo uv run pymobiledevice3 remote tunneld"
RESTART = "Restart any running phonebridge server: it reads this file only when it starts."
NO_PHONE = "No matching phone found. Connect it by USB, unlock it, and accept Trust, then retry."


@contextmanager
def _locked(path: Path):
    """One setup at a time per config, so concurrent adds can't lose each other's phones. The lock is a sidecar:
    the config's own inode changes on every replace."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path.parent / f".{path.name}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _write(path: Path, text: str, existed: bool) -> None:
    data = text.encode("utf-8")
    if not existed:  # exclusive creation: a file that appeared meanwhile is never replaced
        with path.open("xb") as f:
            path.chmod(0o600)
            f.write(data)
        return
    target = path.resolve()  # a symlinked config stays a symlink
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), stat.S_IMODE(target.stat().st_mode))
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        os.unlink(tmp)
        raise


def _multi_phone_keys(raw: dict, mode: str | None, total: int) -> dict:
    """The transport lines this file lacks. A value that is there is the operator's: never rewritten, only named."""
    wrong = []
    if raw.get("tunnel", MULTI_PHONE_TUNNEL) != MULTI_PHONE_TUNNEL:
        wrong.append(f'tunnel = "{raw["tunnel"]}" must become tunnel = "{MULTI_PHONE_TUNNEL}"')
    if raw.get("eyes", MULTI_PHONE_EYES[0]) not in MULTI_PHONE_EYES:
        wrong.append(f'eyes = "{raw["eyes"]}" must become eyes = "{MULTI_PHONE_EYES[0]}" (or "{MULTI_PHONE_EYES[1]}")')
    if wrong:
        raise ValueError(f"Several phones on one server need a per-phone tunnel and capture, and setup won't rewrite "
                         f"what this file sets. Nothing was changed. Edit it: {'; '.join(wrong)}. Then retry.")
    missing = {k: v for k, v in (("tunnel", MULTI_PHONE_TUNNEL), ("eyes", mode or MULTI_PHONE_EYES[0])) if k not in raw}
    if missing and mode is None:
        lines = ", ".join(f'{k} = "{v}"' for k, v in missing.items())
        raise ValueError(f"{total} phones on one server need a per-phone tunnel and capture, which also changes how any "
                         f"existing phone runs. Nothing was changed. Re-run with --multi-phone to add {lines} "
                         f"(--multi-phone dvt for stills only). It needs `{TUNNELD}` running.")
    return missing


def configure(devices: list[dict], path: Path, *, device: str | None = None, alias: str | None = None,
              add_all: bool = False, multi_phone: str | None = None) -> dict:
    """Add connected phones to the config at `path`, creating it if needed. `devices` is the USB listing."""
    if alias is not None and not NEW_ALIAS.fullmatch(alias):
        raise ValueError("alias must start with a lowercase letter and contain letters, digits, - or _")
    if add_all and alias:
        raise ValueError("--all names each new phone itself; use --device and --alias to name one")
    with _locked(path):
        old = read_config(path)
        config = Config.from_text(old or "", path, multi_phone=False)
        raw = tomllib.loads(old or "")
        known = {udid_key(d.udid): a for a, d in config.devices.items()}
        taken = {a.casefold() for a in config.devices}
        connected = [d for d in devices if d.get("udid")]
        new = sorted((d for d in connected if udid_key(d["udid"]) not in known), key=lambda d: d["udid"])
        chosen = [d for d in connected if udid_key(d["udid"]) == udid_key(device)] if device else new
        if not chosen and (device or not connected):
            raise ValueError(NO_PHONE)
        if len(chosen) > 1 and not add_all:
            raise ValueError("Several phones are connected. Run phonebridge devices, then setup --device UDID, "
                             "or setup --all --multi-phone to add them all.")
        if alias and chosen:
            udid, owner = chosen[0]["udid"], known.get(udid_key(chosen[0]["udid"]))
            if owner and owner != alias:
                raise ValueError(f"{udid} is already configured as {owner!r}. Nothing was changed.")
            if not owner and alias.casefold() in taken:
                raise ValueError(f"Alias {alias!r} already names another phone. Nothing was changed. Choose another --alias.")

        add: dict[str, dict] = {}
        for d in chosen:
            if udid_key(d["udid"]) in known:
                continue
            if not UDID.fullmatch(d["udid"]):
                raise ValueError("The connected device returned an invalid UDID")
            name = alias or next(a for a in itertools.chain(["phone"], (f"phone-{n}" for n in itertools.count(2)))
                                 if a not in taken)
            taken.add(name)
            add[name] = d
        if add and raw and not (config.lease_cmd and config.lease_enforce):
            raise ConfigError(path, ['this file doesn\'t enforce a device lease, so setup won\'t add a phone to it: '
                                     'under [lease] set cmd = ["phonebridge-local-lease"] and enforce = true'])
        total = len(config.devices) + len(add)
        keys = _multi_phone_keys(raw, multi_phone, total) if multi_phone or (add and total > 1) else {}

        changed = bool(add or keys)
        if changed:
            head = "" if not keys else ("# Added by phonebridge setup --multi-phone: one server driving several phones "
                                        "needs a per-phone tunnel and capture.\n"
                                        + "".join(f'{k} = "{v}"\n' for k, v in keys.items()) + "\n")
            tables = "".join(f'[devices.{a}]\nudid = {json.dumps(d["udid"])}\nsupervision = "read-only"\n\n'
                             for a, d in add.items())
            body, new = old or "", (tables + ("" if raw else NEW_FILE_TAIL)).rstrip("\n")  # a file with no settings is new
            if new:
                body += ("\n" if body and not body.endswith("\n") else "") + ("\n" if body else "") + new + "\n"
            text = head + body
            expected = {**raw, **keys, **({} if raw else tomllib.loads(NEW_FILE_TAIL))}
            if add:
                expected["devices"] = {**raw.get("devices", {}),
                                       **{a: {"udid": d["udid"], "supervision": "read-only"} for a, d in add.items()}}
            try:
                unchanged = tomllib.loads(text) == expected
            except tomllib.TOMLDecodeError:
                unchanged = False
            if not unchanged:
                raise ConfigError(path, ["setup can't add to this file's layout without changing what it says (is "
                                         "`devices` an inline table?). Nothing was changed; add the phone by hand"])
            Config.from_text(text, path)  # the file as it will run, multi-phone rule included
            _write(path, text, old is not None)

    added = [{"alias": a, "udid": d["udid"], "lease": f"device:{a}",
              **{k: d[k] for k in ("name", "model", "hint", "error") if d.get(k)}} for a, d in add.items()]
    result = {"config": str(path), "added": added,
              "already": [known[udid_key(d["udid"])] for d in connected if udid_key(d["udid"]) in known]}
    if keys:
        result["settings_added"] = keys
    if changed:
        result["restart"] = RESTART
    if keys.get("tunnel", raw.get("tunnel")) == MULTI_PHONE_TUNNEL:
        result["next"] = f"{TUNNELD} (leave it running), then phonebridge setup --check"
    else:
        result["next"] = f"phonebridge check {added[0]['alias']}" if added else "phonebridge setup --check"
    return result


def check(path: Path, devices: list[dict], tunneld: dict | None, usb_error: str | None = None) -> dict:
    """Is this config ready to drive what's plugged in? `tunneld` is its listing, None when it can't be reached."""
    result = {"ok": False, "config": str(path), "problems": [], "phones": [], "unconfigured": []}
    if not path.exists():
        return {**result, "problems": ["no config here: run phonebridge setup"]}
    try:
        config = Config.load(path)
    except ConfigError as e:
        return {**result, "problems": e.problems}
    problems: list[str] = result["problems"]
    if usb_error:
        problems.append(usb_error)
    if not config.devices:
        problems.append("no phones configured: run phonebridge setup")
    if not config.lease_cmd or not config.lease_enforce:
        problems.append('no enforced device lease: under [lease] set cmd = ["phonebridge-local-lease"] and enforce = true')
    elif not shutil.which(config.lease_cmd[0]):
        problems.append(f"lease command {config.lease_cmd[0]!r} isn't on PATH")
    connected = {udid_key(d["udid"]): d for d in devices if d.get("udid")}
    if config.tunnel == MULTI_PHONE_TUNNEL and tunneld is None:
        problems.append(f"tunneld isn't answering on 127.0.0.1:49151: run `{TUNNELD}` and leave it running")
    for d in config.devices.values():
        here = connected.get(udid_key(d.udid))
        row = {"alias": d.alias, "udid": d.udid, "lease": d.lease, "supervision": d.supervision, "connected": here is not None}
        result["phones"].append(row)
        if not UDID.fullmatch(d.udid):
            problems.append(f"{d.alias}: {d.udid} isn't a real UDID (a placeholder?); `phonebridge devices` prints the real one")
            continue
        if here is None:
            if not usb_error:
                problems.append(f"{d.alias}: not connected over USB")
            continue
        if reason := here.get("error") or here.get("hint"):
            problems.append(f"{d.alias}: {reason}")
        if here.get("developer_mode") is False:
            problems.append(f"{d.alias}: Developer Mode is off (Settings › Privacy & Security › Developer Mode)")
        if config.tunnel == MULTI_PHONE_TUNNEL and tunneld is not None:
            row["tunneld"] = bool(tunneld.get(d.udid))
            if not row["tunneld"]:
                problems.append(f"{d.alias}: tunneld doesn't list it (unlock it, check Trust, give tunneld a few seconds)")
    known = {udid_key(d.udid) for d in config.devices.values()}
    result["unconfigured"] = [d["udid"] for k, d in connected.items() if k not in known]
    result["ok"] = not problems
    if result["ok"]:
        result["next"] = f"phonebridge check {next(iter(config.devices))} (presses Home, swipes, taps and types on that phone)"
    return result


def run(a) -> dict:
    """`phonebridge setup`: it needs only the USB listing, so a broken or unsafe config can still be repaired."""
    from .backends.pmd3 import Pmd3Backend
    from .presence import TUNNELD_URL, tunneld_listing

    try:
        devices, usb_error = asyncio.run(Pmd3Backend().devices()), None
    except Exception as e:  # e.g. no usbmuxd
        devices, usb_error = [], f"Could not list USB phones ({type(e).__name__}: {e})"
    try:
        if a.check:
            try:
                tunneld = asyncio.run(tunneld_listing(TUNNELD_URL, 1.0))
            except Exception:  # not running, or not answering: the same advice
                tunneld = None
            return check(a.out, devices, tunneld, usb_error)
        if usb_error:
            raise SystemExit(usb_error)
        return configure(devices, a.out, device=a.device, alias=a.alias, add_all=a.all, multi_phone=a.multi_phone)
    except FileExistsError:
        raise SystemExit(f"{a.out} appeared while setup was writing; keeping it intact. Run setup again.")
    except ValueError as e:
        raise SystemExit(str(e))
