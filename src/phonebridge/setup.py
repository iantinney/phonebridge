"""Create a standalone config without replacing an existing deployment."""

import json
from pathlib import Path
import re

from .config import UDID


def configure(devices: list[dict], path: Path, *, device: str | None = None, alias: str = "phone") -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", alias):
        raise ValueError("alias must start with a lowercase letter and contain letters, digits, - or _")
    matches = [d for d in devices if d.get("udid") == device] if device else devices
    if not matches:
        raise ValueError("No matching phone found. Connect it by USB, unlock it, and accept Trust, then retry.")
    if len(matches) != 1:
        raise ValueError("Several phones are connected. Run phonebridge devices, then setup --device UDID.")
    udid = matches[0]["udid"]
    if not UDID.fullmatch(udid):
        raise ValueError("The connected device returned an invalid UDID")
    content = (f"[devices.{alias}]\nudid = {json.dumps(udid)}\nsupervision = \"read-only\"\n\n"
               "[lease]\ncmd = [\"phonebridge-local-lease\"]\nenforce = true\n\n"
               "[policy]\nurls = [\"https://*\"]\n")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Exclusive creation: an existing production/authhub config is never overwritten.
    with path.open("x") as f:
        path.chmod(0o600)
        f.write(content)
    return {"config": str(path), "device": alias, "next": f"phonebridge check {alias}"}
