"""phonebridge CLI: one-shot calls, the MCP server, and the live acceptance check."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from .bridge import Bridge
from .config import CONFIG_DIR, CONFIG_PATH, STATE_DIR, Config, ConfigError

TOKEN_FILE = Path(os.environ.get("PHONEBRIDGE_TOKEN_FILE", CONFIG_DIR / "phonebridge.token")).expanduser()


def make_bridge() -> Bridge:
    from .backends.pmd3 import Pmd3Backend

    config = Config.load()
    # eyes / tunnel from phonebridge.toml; PHONEBRIDGE_EYES overrides the eyes (dvt = no live capture: the kill switch)
    backend = Pmd3Backend(eyes=os.environ.get("PHONEBRIDGE_EYES") or config.eyes, tunnel=config.tunnel,
                          pointer_surface=config.pointer_surface)
    if config.input == "bthid":
        from .backends.bthid import BtHidBackend

        backend = BtHidBackend(eyes=backend, keystrokes=lambda text: bridge.human.keystrokes(text))
    bridge = Bridge(backend, config)
    return bridge


async def check(b: Bridge, device: str, out: Path, lease: str | None = None) -> dict:
    """Live test: list, screenshot, find 'Settings' (home pages, else App Library search), tap it,
    prove it opened, go home. It presses Home, swipes, taps and types, but takes no account actions."""
    from . import ocr as vision

    out.mkdir(parents=True, exist_ok=True)
    ms: dict[str, list[int]] = {}

    async def timed(name, coro):
        t = time.perf_counter()
        r = await coro
        ms.setdefault(name, []).append(round((time.perf_counter() - t) * 1000))
        return r

    async def look(name):
        png = await timed("screen", b.screen(device))
        (out / f"{name}.png").write_bytes(png)
        items = await timed("ocr", asyncio.to_thread(vision.recognize, png))
        return png, items

    devs = await timed("devices", b.devices())
    await timed("unlock", b.unlock(device, lease=lease))  # taps/swipes are refused on the lock screen
    await timed("home", b.home(device, lease=lease))
    await asyncio.sleep(1.0)
    for _ in range(2):
        await timed("screen", b.screen(device))
    png, items = await look("1-home")
    w, h = vision.png_size(png)
    hit = None
    for page in range(6):  # swipe through home pages; the last one is the App Library
        hit = next((i for i in items if i["text"].casefold() == "settings"), None)
        library = vision.find(items, "App Library")
        if hit:  # a home-screen label isn't tappable (iOS 27): tap the icon above it
            hit = {**hit, "y": round(hit["y"] - 2.2 * hit["h"])}
            break
        if library:
            break
        await timed("swipe", b.swipe(device, w * 0.85, h * 0.5, w * 0.15, h * 0.5, 0.3, lease=lease))
        await asyncio.sleep(0.8)
        png, items = await look(f"1-page{page + 1}")
    if not hit and library:  # App Library search: tap the field, type, take the first result below it
        await timed("tap", b.tap(device, library["x"], library["y"], lease=lease))
        await asyncio.sleep(1.0)
        await timed("type", b.type(device, "Settings", lease=lease))
        await asyncio.sleep(1.0)
        _, items = await look("1-search")
        rows = [i for i in items if i["text"].casefold() == "settings" and i["y"] > library["y"] + 80]
        hit = min(rows, key=lambda i: i["y"], default=None)
    if not hit:
        raise SystemExit(f"'Settings' not found; screenshots in {out}")
    await timed("tap", b.tap(device, hit["x"], hit["y"], lease=lease))
    await asyncio.sleep(1.5)
    _, items = await look("2-settings")
    texts = [i["text"] for i in items]
    opened = any(t in texts for t in ("General", "Privacy & Security", "Apple Account", "Wi-Fi", "Bluetooth"))
    await timed("home", b.home(device, lease=lease))
    return {"devices": devs, "hit": hit, "settings_opened": opened, "texts_after": texts[:12], "ms": ms}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="phonebridge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    s = sub.add_parser("setup", help="add connected phones to the config, each with a local exclusive lease; "
                                     "what is already in the file is never changed")
    which = s.add_mutually_exclusive_group()
    which.add_argument("--device", help="UDID of the phone to add, when several new phones are connected")
    which.add_argument("--all", action="store_true", help="add every connected phone that isn't configured yet")
    s.add_argument("--alias", help="the phone's name (default: phone, phone-2, phone-3, ...)")
    s.add_argument("--multi-phone", nargs="?", const="stream", choices=["stream", "dvt"],
                   help='needed for a second phone: add tunnel = "tunneld" and eyes = "stream" (or dvt: stills only) '
                        "where the file sets neither; never rewrites a value that is there")
    s.add_argument("--check", action="store_true",
                   help="validate the config against the connected phones and tunneld; exit 1 if anything is wrong")
    s.add_argument("--out", "--config", dest="out", type=Path, default=CONFIG_PATH, help="the config file")
    for name in ("screen", "ocr", "lock", "pair"):
        s = sub.add_parser(name)
        s.add_argument("device")
        if name == "screen":
            s.add_argument("-o", "--out", default="screen.png")
    s = sub.add_parser("find"); s.add_argument("device"); s.add_argument("text")

    def input_cmd(name):  # input commands can use the caller's own device lease
        s = sub.add_parser(name)
        s.add_argument("device")
        s.add_argument("--lease", help="your lease token for device:<alias> (else phonebridge takes its own)")
        return s

    input_cmd("home")
    input_cmd("unlock")
    s = input_cmd("tap"); s.add_argument("x", type=float); s.add_argument("y", type=float)
    s = input_cmd("swipe")
    for c in ("x1", "y1", "x2", "y2"):
        s.add_argument(c, type=float)
    s.add_argument("--duration", type=float, default=0.3)
    s = input_cmd("scroll"); s.add_argument("direction", nargs="?", default="up", choices=["up", "down", "left", "right"])
    s = input_cmd("type"); s.add_argument("text"); s.add_argument("--submit", action="store_true", help="press Return after")
    s = input_cmd("paste"); s.add_argument("text")
    s = input_cmd("launch"); s.add_argument("bundle_id")
    s = input_cmd("open"); s.add_argument("url"); s.add_argument("--bundle-id")
    s = sub.add_parser("check", help="live acceptance test (presses Home, swipes, taps, types; no account actions)")
    s.add_argument("device")
    s.add_argument("--out", type=Path, default=STATE_DIR / "check")
    s.add_argument("--lease", help="your lease token for device:<alias> (else phonebridge takes its own)")
    s = sub.add_parser("measure", help="live timing: humanized taps/swipes/flicks/typing as produced vs as the phone "
                                       "observed them (serves a logger page to Safari over the LAN)")
    s.add_argument("device")
    s.add_argument("--out", type=Path, default=STATE_DIR / "measure")
    s.add_argument("--lease", help="your lease token for device:<alias> (else phonebridge takes its own)")
    s.add_argument("--taps", type=int, default=12); s.add_argument("--swipes", type=int, default=8)
    s.add_argument("--scrolls", type=int, default=8); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--text", default=None)
    s.add_argument("--url", help="what the phone opens instead of http://<this Mac's LAN ip>:<port>/ (a tunnel to that port)")
    s = sub.add_parser("attest", help="the operator confirms a preflight check the host can't probe (DEV-01)")
    s.add_argument("check"); s.add_argument("--days", type=float, default=30); s.add_argument("--by", default="ian")
    s.add_argument("--false", dest="value", action="store_false", help="record that it does NOT hold")
    s = sub.add_parser("preflight", help="probe table + attestations for a device: no task starts unless all pass")
    s.add_argument("device"); s.add_argument("--app", help="the task's app (bundle id): installed + version")
    s.add_argument("--egress", action="store_true", help="open an IP-echo page in Safari (input, under the lease) and "
                   "compare with [host] egress_ip")
    s.add_argument("--lease", help="your lease token for device:<alias> (for --egress)")
    s = sub.add_parser("serve", help="MCP server (stdio, or --http on 127.0.0.1)")
    s.add_argument("--http", action="store_true"); s.add_argument("--port", type=int, default=8934)
    from .screens.cli import add_parser as screens_parser, run as screens_run  # screens: the known-screens store
    screens_parser(sub)
    u = sub.add_parser("understand", help="the local understanding model as a service (Mac; extra `understand`)")
    us = u.add_subparsers(dest="understand_cmd", required=True)
    usv = us.add_parser("serve", help="serve Qwen3.5-4B via mlx-vlm on 127.0.0.1 (macOS; keep it running with a LaunchAgent)")
    usv.add_argument("--port", type=int, default=8935)
    usv.add_argument("--idle", type=float, default=600.0, help="s without a request before the model is unloaded")
    usv.add_argument("--model", default=None)
    usv.add_argument("--preload", action="store_true", help="load the model at start instead of on the first request")
    a = p.parse_args(argv)
    if a.cmd == "screens":
        raise SystemExit(screens_run(a))
    if a.cmd == "understand":
        from . import understand

        understand.serve(port=a.port, idle_s=a.idle, model_id=a.model or understand.MODEL, preload=a.preload)
        return

    if a.cmd == "setup":  # no bridge: setup is how a broken or unsafe config gets repaired
        from .setup import run as setup_run

        r = setup_run(a)
        json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
        print()
        raise SystemExit(0 if r.get("ok", True) else 1)

    try:
        b = make_bridge()
    except ConfigError as e:
        raise SystemExit(str(e))
    if a.cmd == "serve":
        from .mcp_server import build

        mcp = build(b)
        try:
            if a.http:
                import uvicorn

                from .mcp_server import BearerAuth

                from .mcp_server import with_presence_route

                app = mcp.streamable_http_app(host="127.0.0.1")  # loopback + DNS-rebinding protection
                app = with_presence_route(app, b)  # GET /presence for monitors
                from .host.watch import with_host

                app = with_host(app, b)  # macOS: per-task power assertion, sleep/wake, USB watchdog, heartbeat
                if TOKEN_FILE.exists():  # opt-in: clients send Authorization: Bearer <token>
                    app = BearerAuth(app, TOKEN_FILE.read_text().strip())
                uvicorn.run(app, host="127.0.0.1", port=a.port)
            else:
                mcp.run()
        finally:
            b.leases.release_all()  # don't leave device locks to expire after a restart
        return

    async def run():
        try:
            match a.cmd:
                case "devices": return await b.devices()
                case "screen":
                    Path(a.out).write_bytes(await b.screen(a.device))
                    return a.out
                case "ocr": return await b.ocr(a.device)
                case "find": return await b.find_text(a.device, a.text)
                case "lock": return await b.lock_state(a.device)
                case "pair": return await b.backend.pair(b.config.resolve(a.device).udid)
                case "home": return await b.home(a.device, lease=a.lease)
                case "unlock": return await b.unlock(a.device, lease=a.lease)
                case "tap": return await b.tap(a.device, a.x, a.y, lease=a.lease)
                case "swipe": return await b.swipe(a.device, a.x1, a.y1, a.x2, a.y2, a.duration, lease=a.lease)
                case "scroll": return await b.scroll(a.device, a.direction, lease=a.lease)
                case "type": return await b.type(a.device, a.text, submit=a.submit, lease=a.lease)
                case "paste": return await b.paste(a.device, a.text, lease=a.lease)
                case "launch": return await b.launch(a.device, a.bundle_id, lease=a.lease)
                case "open": return await b.open_url(a.device, a.url, a.bundle_id, lease=a.lease)
                case "check": return await check(b, a.device, a.out, a.lease)
                case "attest":
                    from .host.preflight import Attestations

                    Attestations().attest(a.check, value=a.value, days=a.days, by=a.by)
                    return {"attested": a.check, "value": a.value, "days": a.days}
                case "preflight":
                    from .host.preflight import Attestations, egress, ok, preflight, read
                    from .host.watch import _settings

                    want = _settings().get("egress_ip")
                    seen = await egress(b, a.device, a.lease) if a.egress and want else None
                    reading = await read(b.backend, b.config.resolve(a.device).udid, a.app, want, seen)
                    results = preflight(reading, Attestations(), log=STATE_DIR / "host" / "preflight.jsonl")
                    return {"ok": ok(results), "results": [r.__dict__ for r in results]}
                case "measure":
                    from .measure import TEXT, measure

                    return await measure(b, a.device, a.out, a.lease, a.taps, a.swipes, a.scrolls, a.text or TEXT, a.port, a.url)
        finally:
            await b.close()

    r = asyncio.run(run())
    if r is not None:
        json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
        print()
