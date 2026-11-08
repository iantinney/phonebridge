"""Live timing on a phone: what the humanizer produced against what the phone itself observed.

`phonebridge measure <device>` serves measure.html from this Mac on the LAN, opens it in Safari on the
phone, then taps a target, swipes, flicks and types through the Bridge like any agent would. The page
logs the touch and key events iOS delivered (coordinates, timestamps) back to us; the produced plans are
captured on their way into the backend. The pairs and their summary go to <out>/measure-<time>.json and
stdout. It needs the config's `[policy] urls` to allow http://<this Mac's LAN address>:<port>/*.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .bridge import Bridge

PAGE = Path(__file__).with_name("measure.html").read_bytes()
TEXT = "Loved this, the lighting in the second clip is unreal. Where was it shot?"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    finally:
        s.close()


class Log:
    """What the page reports: its geometry (`hello`) and the event stream, plus one-shot commands back."""

    def __init__(self):
        self.hello: dict | None = None
        self.events: list[dict] = []
        self.commands: dict = {}
        self.lock = threading.Lock()

    def handler(self):
        log = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("cache-control", "no-store")
                self.end_headers()
                self.wfile.write(PAGE)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0)) or 0) or b"{}")
                with log.lock:
                    log.hello = body.get("hello") or log.hello
                    log.events.extend(body.get("events", []))
                    reply, log.commands = log.commands, {}
                out = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(out)
        return H

    def since(self, n: int, kinds: tuple[str, ...] = ()) -> list[dict]:
        with self.lock:
            evs = self.events[n:]
        return [e for e in evs if not kinds or e["e"] in kinds]

    def __len__(self):
        with self.lock:
            return len(self.events)


def _stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": round(statistics.median(xs), 1), "mean": round(statistics.mean(xs), 1),
            "sd": round(statistics.pstdev(xs), 1) if len(xs) > 1 else 0.0, "min": round(min(xs), 1), "max": round(max(xs), 1)}


def _strokes(events: list[dict]) -> list[list[dict]]:
    """Group touch events into strokes (touchstart .. touchend/cancel) by touch id."""
    open_: dict[int, list[dict]] = {}
    done = []
    for e in events:
        if e["e"] not in ("touchstart", "touchmove", "touchend", "touchcancel"):
            continue
        for p in e["p"]:
            pt = {"t": e["t"], "x": p["x"], "y": p["y"], "e": e["e"]}
            if e["e"] == "touchstart":
                open_[p["id"]] = [pt]
            elif p["id"] in open_:
                open_[p["id"]].append(pt)
                if e["e"] in ("touchend", "touchcancel"):
                    done.append(open_.pop(p["id"]))
    return done


def _stroke_summary(s: list[dict], dpr: float) -> dict:
    pts = [(p["x"] * dpr, p["y"] * dpr, p["t"]) for p in s]
    (x0, y0, t0), (x1, y1, t1) = pts[0], pts[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    # speed over a 3-sample window: iOS coalesces touch samples, so adjacent pairs spike
    v = [math.hypot(b[0] - a[0], b[1] - a[1]) / max(1e-3, (b[2] - a[2]) / 1000) for a, b in zip(pts, pts[2:]) if b[2] > a[2]]
    # bow: max distance of a sample from the chord, as a fraction of the length
    bow = 0.0
    if length > 1:
        for x, y, _ in pts[1:-1]:
            bow = max(bow, abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / length)
    peak_at = (v.index(max(v)) + 0.5) / len(v) if v else None
    return {"samples": len(pts), "ms": round(t1 - t0, 1), "len_px": round(length, 1), "start": [round(x0), round(y0)],
            "end": [round(x1), round(y1)], "peak_px_s": round(max(v)) if v else None, "mean_px_s": round(length / max(1e-3, (t1 - t0) / 1000)),
            "peak_at": round(peak_at, 2) if peak_at is not None else None, "bow_px": round(bow, 1)}


def _plan_summary(path: list[tuple[float, float, float]]) -> dict:
    (x0, y0, _), (x1, y1, t1) = path[0], path[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    v = [math.hypot(b[0] - a[0], b[1] - a[1]) / (b[2] - a[2]) for a, b in zip(path, path[1:])]
    bow = max((abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / length for x, y, _ in path[1:-1]), default=0.0) if length > 1 else 0.0
    return {"samples": len(path), "ms": round(t1 * 1000, 1), "len_px": round(length, 1), "start": [round(x0), round(y0)],
            "end": [round(x1), round(y1)], "peak_px_s": round(max(v)), "mean_px_s": round(length / t1),
            "peak_at": round((v.index(max(v)) + 0.5) / len(v), 2), "bow_px": round(bow, 1)}


async def _gesture(log: Log, coro, settle: float):
    """Run one input call and return the touch strokes the page logged for it (usually one; none = lost)."""
    m0 = len(log)
    await coro
    await asyncio.sleep(settle)
    return _strokes(log.since(m0))


async def measure(b: Bridge, device: str, out: Path, lease: str | None = None, taps: int = 12, swipes: int = 8,
                  scrolls: int = 8, text: str = TEXT, port: int = 8765, url: str | None = None) -> dict:
    """`url`: what the phone opens when it can't reach this Mac's LAN address (a phone behind a proxy):
    a tunnel to this port, e.g. `cloudflared tunnel --url http://<this mac>:<port>`."""
    out.mkdir(parents=True, exist_ok=True)
    log = Log()
    server = ThreadingHTTPServer(("0.0.0.0", port), log.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    produced: dict[str, list] = {"tap": [], "swipe": [], "type": []}
    wall: dict[str, list[float]] = {"tap": [], "swipe": [], "type": []}
    backend = b.backend
    orig = {k: getattr(backend, k) for k in produced}

    def capture(name):
        async def call(udid, *args):
            produced[name].append(args)
            t = time.perf_counter()
            try:
                return await orig[name](udid, *args)
            finally:
                wall[name].append(round((time.perf_counter() - t) * 1000, 1))
        return call
    for k in produced:
        setattr(backend, k, capture(k))
    report: dict = {"device": device, "backend": type(backend).__name__, "input": b.config.input, "url": None}
    try:
        url = url or f"http://{lan_ip()}:{port}/"
        report["url"] = url
        await b.unlock(device, lease=lease)
        await b.open_url(device, url, lease=lease)
        for _ in range(60):
            if log.hello:
                break
            await asyncio.sleep(0.5)
        else:
            raise SystemExit(f"the phone never loaded {url}: is it on this LAN (else pass --url with a tunnel to port "
                             f"{port}), and does [policy] urls allow it?")
        hello = log.hello
        dpr = hello["dpr"]
        report["phone"] = {"dpr": dpr, "viewport_css": [hello["w"], hello["h"]], "ua": hello["ua"]}
        await asyncio.sleep(1.0)

        # --- calibration: where the viewport sits on the screen (Safari's bars) ----------------------
        png = await b.screen(device)
        from .ocr import png_size
        W, H = png_size(png)
        n0 = len(log)
        cal_x, cal_y = W * 0.5, H * 0.62  # free area under the text box
        await b.tap(device, cal_x, cal_y, lease=lease)
        await asyncio.sleep(1.0)
        strokes = _strokes(log.since(n0))
        if not strokes:
            raise SystemExit("the calibration tap produced no touch event on the page")
        at = produced["tap"][-1]
        off_x, off_y = at[0] - strokes[0][0]["x"] * dpr, at[1] - strokes[0][0]["y"] * dpr
        report["viewport_offset_px"] = [round(off_x, 1), round(off_y, 1)]
        to_screen = lambda cx, cy: (cx * dpr + off_x, cy * dpr + off_y)

        def observed_tap(stroke):
            x, y = to_screen(stroke[0]["x"], stroke[0]["y"])
            return {"at": [round(x, 1), round(y, 1)], "press_ms": round(stroke[-1]["t"] - stroke[0]["t"], 1),
                    "t": stroke[0]["t"], "moves": len(stroke) - 2}

        # --- taps on the target ---------------------------------------------------------------------
        tr = hello["target"]
        cx, cy = to_screen(tr["x"] + tr["w"] / 2, tr["y"] + tr["h"] / 2)
        tw, th = tr["w"] * dpr, tr["h"] * dpr
        p0, seen, lost = len(produced["tap"]), [], 0
        for _ in range(taps):
            strokes = await _gesture(log, b.tap(device, cx, cy, tw, th, lease=lease), 0.6)
            seen.append(observed_tap(strokes[0]) if strokes else None)
            lost += not strokes
        prod = [{"at": [round(x, 1), round(y, 1)], "press_ms": round(press * 1000, 1)} for x, y, press in produced["tap"][p0:]]
        pairs = [(p, o) for p, o in zip(prod, seen) if o]
        obs = [o for _, o in pairs]
        report["tap"] = {
            "target": {"center": [round(cx), round(cy)], "size": [round(tw), round(th)]},
            "sent": taps, "lost": lost, "pairs": pairs,
            "produced": {"press_ms": _stats([p["press_ms"] for p in prod]),
                         "dx": _stats([p["at"][0] - cx for p in prod]), "dy": _stats([p["at"][1] - cy for p in prod])},
            "observed": {"press_ms": _stats([o["press_ms"] for o in obs]),
                         "dx": _stats([o["at"][0] - cx for o in obs]), "dy": _stats([o["at"][1] - cy for o in obs]),
                         "inside_target": sum(abs(o["at"][0] - cx) <= tw / 2 and abs(o["at"][1] - cy) <= th / 2 for o in obs),
                         "gap_ms": _stats([b_["t"] - a_["t"] for a_, b_ in zip(obs, obs[1:])])},
            "point_error_px": _stats([math.hypot(p["at"][0] - o["at"][0], p["at"][1] - o["at"][1]) for p, o in pairs]),
            "press_error_ms": _stats([o["press_ms"] - p["press_ms"] for p, o in pairs]),
            "call_ms": _stats(wall["tap"][p0:]),
        }

        # --- swipes in the free area ----------------------------------------------------------------
        p0, seen, lost = len(produced["swipe"]), [], 0
        sx, y_lo, y_hi = W * 0.5, H * 0.78, H * 0.5
        for i in range(swipes):
            a, z = (y_lo, y_hi) if i % 2 == 0 else (y_hi, y_lo)
            strokes = await _gesture(log, b.swipe(device, sx, a, sx, z, 0.3, lease=lease), 1.0)
            seen.append(_stroke_summary(strokes[0], dpr) if strokes else None)
            lost += not strokes
        prod = [_plan_summary(path) for (path,) in produced["swipe"][p0:]]
        pairs = [(p, o) for p, o in zip(prod, seen) if o]
        obs = [o for _, o in pairs]
        report["swipe"] = {
            "requested": {"len_px": round(y_lo - y_hi), "duration_ms": 300},
            "sent": swipes, "lost": lost, "pairs": pairs,
            **{side: {k: _stats([d[k] for d in rows]) for k in ("ms", "len_px", "samples", "peak_px_s", "mean_px_s", "peak_at", "bow_px")}
               for side, rows in (("produced", prod), ("observed", obs))},
            "duration_error_ms": _stats([o["ms"] - p["ms"] for p, o in pairs]),
            "length_error_px": _stats([o["len_px"] - p["len_px"] for p, o in pairs]),
            "call_ms": _stats(wall["swipe"][p0:]),
        }

        # --- scroll flicks --------------------------------------------------------------------------
        p0, rows, lost = len(produced["swipe"]), [], 0
        for i in range(scrolls):
            before = log.hello.get("scroll", 0)
            m0 = len(log)
            strokes = await _gesture(log, b.scroll(device, "up" if i % 2 == 0 else "down", lease=lease), 1.6)  # momentum runs out
            ys = [e["y"] for e in log.since(m0, ("scroll",))]
            rows.append({"direction": "up" if i % 2 == 0 else "down", "produced": _plan_summary(produced["swipe"][-1][0]),
                         "observed": _stroke_summary(strokes[0], dpr) if strokes else None,
                         "scrolled_css_px": round((ys[-1] - before) if ys else 0), "scroll_events": len(ys)})
            lost += not strokes
        prod, obs = [r["produced"] for r in rows], [r["observed"] for r in rows if r["observed"]]
        report["scroll"] = {"sent": scrolls, "lost": lost, "flicks": rows,
                            "produced": {k: _stats([d[k] for d in prod]) for k in ("ms", "len_px", "peak_px_s")},
                            "observed": {k: _stats([d[k] for d in obs]) for k in ("ms", "len_px", "peak_px_s")},
                            "scrolled_css_px": _stats([abs(r["scrolled_css_px"]) for r in rows]),
                            "call_ms": _stats(wall["swipe"][p0:])}
        log.commands["reset"] = True

        # --- typing ----------------------------------------------------------------------------------
        tx = hello["text"]
        fx, fy = to_screen(tx["x"] + tx["w"] / 2, tx["y"] + tx["h"] / 2)
        await b.tap(device, fx, fy, tx["w"] * dpr, tx["h"] * dpr, lease=lease)
        await asyncio.sleep(1.5)
        focused_by_tap = log.hello.get("active") == "text"
        if not focused_by_tap:  # fall back to focusing from the page, so the keys still get measured
            log.commands["focus"] = True
            await asyncio.sleep(1.0)
        log.commands["clear"] = True
        await asyncio.sleep(1.0)
        n0, p0 = len(log), len(produced["type"])
        await b.type(device, text, lease=lease)
        await asyncio.sleep(1.5)
        keys = produced["type"][p0][0]
        downs = [e for e in log.since(n0, ("keydown",)) if e["k"] != "Shift"]  # Shift is a modifier, not a key
        ups = [e for e in log.since(n0, ("keyup",)) if e["k"] != "Shift"]
        inputs = log.since(n0, ("input",))
        holds = []
        for d in downs:  # pair each keydown with the next keyup of the same key
            u = next((u for u in ups if u["t"] >= d["t"] and u["k"] == d["k"]), None)
            if u:
                holds.append(u["t"] - d["t"])
                ups.remove(u)
        prod_iki = [(h_ + g) * 1000 for (_, h_, _), (_, _, g) in zip(keys, keys[1:])]
        obs_iki = [b_["t"] - a_["t"] for a_, b_ in zip(downs, downs[1:])]
        report["type"] = {
            "text": text, "keys_planned": len(keys), "slips": sum(c == "\b" for c, _, _ in keys),
            "keydowns": len(downs), "typed_ok": bool(inputs) and inputs[-1]["v"] == text,
            "focused_by_tap": focused_by_tap, "focused_after": log.hello.get("active"),
            "keys_outside_text_box": sum(e.get("in") != "text" for e in downs),
            "typed": inputs[-1]["v"] if inputs else None,
            "key_sequence": "".join(e["k"] if len(e["k"]) == 1 else f"<{e['k']}>" for e in log.since(n0, ("keydown",))),
            "produced": {"iki_ms": _stats(prod_iki), "hold_ms": _stats([h_ * 1000 for _, h_, _ in keys]),
                         "total_ms": round(sum(h_ + g for _, h_, g in keys) * 1000)},
            "observed": {"iki_ms": _stats(obs_iki), "hold_ms": _stats(holds),
                         "total_ms": round(downs[-1]["t"] - downs[0]["t"]) if len(downs) > 1 else None},
            "iki_error_ms": _stats([o - p for p, o in zip(prod_iki, obs_iki)]) if len(downs) == len(keys) else "key count differs",
            "call_ms": wall["type"][p0:],
        }

        # --- think time, from the Bridge's own log ----------------------------------------------------
        recs = [json.loads(line) for line in b.log_path.read_text().splitlines()]
        thinks = [r["think_ms"] for r in recs[-(taps + swipes + scrolls + 4):] if "think_ms" in r]
        report["think_ms_waited"] = _stats(thinks)
        report["events_logged"] = len(log)
        await b.home(device, lease=lease)  # leave the phone on its home screen for the next lane
    finally:
        for k, fn in orig.items():
            setattr(backend, k, fn)
        server.shutdown()
        path = out / f"measure-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        path.write_text(json.dumps({**report, "raw_events": log.events}, indent=1))
        report["saved"] = str(path)
    return report
