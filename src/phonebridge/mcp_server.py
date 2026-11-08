"""MCP server: one tool per interface call. stdio, or streamable HTTP on Mac loopback."""

from __future__ import annotations

import asyncio
import functools
import hmac
import io

from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .bridge import Bridge


def to_jpeg(png: bytes, quality: int = 80) -> bytes:
    """Same pixels, ~10x fewer bytes than the device's PNG (it crosses ssh tunnels as base64)."""
    from PIL import Image as PILImage

    out = io.BytesIO()
    PILImage.open(io.BytesIO(png)).convert("RGB").save(out, "JPEG", quality=quality)
    return out.getvalue()


INSTRUCTIONS = """Drive real iPhones over USB. `device` is an alias from phonebridge.toml (devices() lists them) or a UDID.
Use a specific device on every call: a screen, OCR box or frame from one phone must never guide input on another.
Separate agents may work concurrently on separate phones; within one phone, observe then act sequentially.
Coordinates are screenshot pixels: read with ocr/find_text, then tap the returned x, y.
Home-screen app labels are not tappable: tap the icon, about 2.2 label-heights (h) above the label.
Input is humanized (tap spread inside the target, press and swipe timing, typing cadence, a think
time before each action), so calls take a little longer than the raw gesture; use scroll() for feeds.
These are interaction mechanics, not a promise of undetectability or permission to automate a service.
Input needs the phone's device lock (the [lease] cmd's lease on device:<alias>). For exclusive use, hold that lease
yourself and pass `lease=<token>` on every input call. Without it, this server takes the lease itself
until 60 s after the last input. Stateless input requires a unique controller ID on every call.
All inputs reserve the physical device; another controller, even with the same lease, is refused.
devices() reports each phone's `supervision`: `hold-outward` means outward account actions
(post, DM, comment, like, follow) need the operator's approval first.
Guards (zero input is sent when one refuses): the app you last launched must stay frontmost, else only
launch(that app) is allowed (NotFrontmost); after a flick lands on an unfamiliar screen, no more flicks
until it is recognised or the app relaunched (UnfamiliarScreen); while the Mac is going to sleep, the phone
is off USB or too hot, input waits (Parked: retry later); TaskEnded means back out and start a new task."""


def _caller(ctx: Context, controller: str | None = None) -> str:
    """Who owns phonebridge's own device lock: the MCP session where the transport has one (handshake-era
    clients). Stateless HTTP clients must supply a controller ID."""
    if controller:
        return "remote:" + controller
    try:
        session = ctx.request_context.session_id
    except (AttributeError, ValueError):
        session = None
    if not session:
        raise ValueError("stateless input needs a unique controller ID")
    return f"mcp:{session}"


def build(bridge: Bridge) -> MCPServer:
    mcp = MCPServer("phonebridge", instructions=INSTRUCTIONS)

    def tool(fn):
        """Register fn; device failures (unknown alias, lease busy, unplugged, iOS too old) reach the agent verbatim."""
        @functools.wraps(fn)
        async def call(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                raise ToolError(f"{e.__class__.__name__}: {e}") from e  # `type` is a tool name here
        return mcp.tool()(call)

    @tool
    async def release_controller(device: str, controller: str) -> str:
        """Release this controller after its final input; waits for in-flight inputs to end."""
        await bridge.release_reservation(device, "remote:" + controller)
        return "ok"

    @tool
    async def devices() -> list[dict]:
        """Connected iPhones: id (alias), udid, name, ios, product, supervision (unconfigured phones: read-only)."""
        return await bridge.devices()

    @tool
    async def screen(device: str, format: str = "jpeg") -> Image:
        """Screenshot at device pixels (the space tap() uses). format: jpeg (default, small) or png (lossless)."""
        png = await bridge.screen(device)
        if format == "png":
            return Image(data=png, format="png")
        return Image(data=await asyncio.to_thread(to_jpeg, png), format="jpeg")

    @tool
    async def ocr(device: str, fast: bool = False, with_image: bool = False) -> list[dict] | dict:
        """Screenshot + Apple Vision OCR: [{text, x, y, w, h, conf}], x/y = tap-ready center. It takes its OWN
        screenshot: a separate `screen` call is another moment, so pair lines with pixels via `with_image=True`
        ({items, image_jpeg: base64 of the very image read, width, height}) or `look(with_frame=True)`."""
        return await bridge.ocr(device, fast=fast, with_image=with_image)

    @tool
    async def look(device: str, platform: str | None = None, expect: list[str] | None = None,
                   settle: bool | str = True, with_frame: bool = False) -> dict:
        """One look through the live screen stream, judged on the Mac: {screen, layer, score, settled, frames_seen,
        frame_seq, frame_wall, frame_age_ms, ocr: [levels], ms: {settle, ocr, pixels, total}, items: OCR lines of
        that very frame}. `platform` narrows the store (instagram, tiktok, ...), `expect` lists the screens a flow
        expects next (resolved first). settle: true/"content" (default) waits until the chrome holds still AND the
        content stops moving (a coasting feed; `content` in the result says how it ended: frames agree, still window,
        timeout → settled false), "chrome" checks the chrome bands only, false judges the latest frame at once.
        with_frame=True adds `frame_jpeg` (base64): the very frame judged; frame(seq=frame_seq) fetches it later."""
        return await bridge.look(device, platform=platform, expect=expect, settle=settle, with_frame=with_frame)

    @tool
    async def record_start(device: str, path: str, ctx: Context, fps: float = 30.0, lease: str | None = None, controller: str | None = None) -> dict:
        """Record the daemon's live frames (the eyes: no second capture) to a video-only mp4 at `path` (under the
        evidence root) at `fps` (1–60), until record_stop, the lease's end or 15 min. Needs the session's lease."""
        return await bridge.record_start(device, path, fps=fps, lease=lease, caller=_caller(ctx, controller))

    @tool
    async def record_stop(device: str) -> dict:
        """Stop the recording: {path, t0 (wall time of frame 0; frame n is at t0 + n/fps), fps, frames, dropped,
        duration_s, gaps: [[from_s, to_s], ...] (capture resets/parks: the last frame repeats), ended, size}."""
        return await bridge.record_stop(device)

    @tool
    async def frame(device: str, format: str = "jpeg", seq: int | None = None) -> Image:
        """The latest frame of the live screen stream (tens of ms old; opens the stream like input does), jpeg or png.
        `seq` = a recent look's frame_seq: that judged frame instead (error once it is no longer kept)."""
        data, _ = await bridge.frame(device, format=format, seq=seq)
        return Image(data=data, format=format)

    @tool
    async def understand(device: str, prompt: str | None = None, max_tokens: int | None = None) -> dict:
        """What is on screen now, from the warm local model on this Mac (Qwen3.5-4B via mlx-vlm, ~1.5 s warm):
        {summary: one sentence (who, what, hook, topic), interest: 0-3 or null, parsed, raw, ms, model, frame: {seq,
        wall, age_ms, source}}. `prompt` replaces the default short prompt (JSON with summary + interest)."""
        return await bridge.understand(device, prompt=prompt, max_tokens=max_tokens)

    @tool
    async def presence(device: str | None = None, fresh: bool = False) -> dict:
        """Is the phone really there (default: the first configured phone)? A real round trip (usbmux + tunneld + one lockdown GetValue, ≤ ~2 s, cached
        10 s; fresh=True skips the cache): {attached, transport usb|wifi|none, tunnel, lockdown, name, ios, foreground,
        locked (None: iOS 27 doesn't report it), awake_seen_s_ago, errors, ms, ts, cached, age_s, device, daemon}."""
        if not (device := device or next(iter(bridge.config.devices), None)):
            raise ToolError("no phone configured: run phonebridge setup")
        return await bridge.presence(device, fresh=fresh)

    @tool
    async def find_text(device: str, substring: str) -> dict | None:
        """Best on-screen match for `substring` (case-insensitive), or null."""
        return await bridge.find_text(device, substring)

    @tool
    async def foreground(device: str) -> str | None:
        """Bundle id of the frontmost app (com.apple.springboard on the home and lock screens), or null."""
        return await bridge.foreground(device)

    @tool
    async def ax_probe(device: str, ctx: Context, ops: list[str] | None = None, max_nodes: int = 120,
                       lease: str | None = None, controller: str | None = None) -> dict:
        """EXPERIMENT (accessibility-tree spike): the accessibility audit daemon's view of the foreground app. ops:
        caps (selector list), focus_first (moves the inspector focus: the app scrolls; probe page/home screen only),
        tree (labels and roles from the root, no focus move), hit (element at a point), audit (the only rects).
        Default: caps, tree, hit. Each op and DTX call is timed."""
        return await bridge.ax_probe(device, ops or ["caps", "tree", "hit"], max_nodes, caller=_caller(ctx, controller),
                                     lease=lease)

    @tool
    async def tap(device: str, x: float, y: float, ctx: Context, w: float = 0, h: float = 0,
                  lease: str | None = None, controller: str | None = None, path: str = "report", think: bool = True,
                  contact_mm: float = 9.0) -> str:
        """Tap the target centred on screenshot pixel (x, y). Pass its size w, h (an OCR box) so the finger
        lands naturally inside it; else a 44 pt target is assumed. `path`: leave it ("report"); "indigo" and
        "vdigitizer" (a virtual digitizer whose reports carry a `contact_mm` contact) are diagnostic experiments for
        comparing touch attributes. think=False: the caller paces its own inputs, so no drawn pause first (every
        guard and the frontmost check still run)."""
        await bridge.tap(device, x, y, w, h, caller=_caller(ctx, controller), lease=lease, path=path, think=think,
                         contact_mm=contact_mm)
        return "ok"

    @tool
    async def double_tap(device: str, x: float, y: float, ctx: Context, w: float = 0, h: float = 0,
                         lease: str | None = None, controller: str | None = None, think: bool = True) -> dict:
        """Two quick taps at screenshot pixel (x, y) as one input (a like on IG/TikTok), ~100–300 ms apart.
        think=False: the caller paces its own inputs (no drawn pause; every guard and the frontmost check still run)."""
        return await bridge.double_tap(device, x, y, w, h, caller=_caller(ctx, controller), lease=lease, think=think)

    @tool
    async def swipe(device: str, x1: float, y1: float, x2: float, y2: float, ctx: Context,
                    duration: float = 0.3, lease: str | None = None, controller: str | None = None, release: float | None = None,
                    path: str = "report", think: bool = True) -> str:
        """Swipe between screenshot pixels over about `duration` seconds (≤ 5), along a human-like path. `release` (0–1,
        lift speed over mean speed) makes it a planned glide: length, duration and release kept exactly, texture only drawn.
        think=False: the caller paces its own inputs (no drawn pause; every guard and the frontmost check still run).
        `path`: leave it ("report"); "indigo" is a diagnostic experiment (CoreDevice IndigoDigitizerEvents)."""
        await bridge.swipe(device, x1, y1, x2, y2, duration, caller=_caller(ctx, controller), lease=lease, release=release, path=path,
                           think=think)
        return "ok"

    @tool
    async def scroll(device: str, ctx: Context, direction: str = "up", lease: str | None = None, controller: str | None = None) -> str:
        """One natural scroll flick of varying length and speed. direction = where the content moves:
        up (read on down a feed), down, left, right."""
        await bridge.scroll(device, direction, caller=_caller(ctx, controller), lease=lease)
        return "ok"

    @tool
    async def type(device: str, text: str, ctx: Context, submit: bool = False, lease: str | None = None, controller: str | None = None) -> str:
        """Type printable text (≤ 300 chars, no newlines) into the focused field; submit=True presses Return after."""
        await bridge.type(device, text, submit=submit, caller=_caller(ctx, controller), lease=lease)
        return "ok"

    @tool
    async def paste(device: str, text: str, ctx: Context, lease: str | None = None, controller: str | None = None) -> str:
        """Put text in the focused field fast (long text, emoji, accents)."""
        await bridge.paste(device, text, caller=_caller(ctx, controller), lease=lease)
        return "ok"

    @tool
    async def home(device: str, ctx: Context, lease: str | None = None, controller: str | None = None) -> str:
        """Press the Home button (go to the home screen; wakes a sleeping phone)."""
        await bridge.home(device, caller=_caller(ctx, controller), lease=lease)
        return "ok"

    @tool
    async def unlock(device: str, ctx: Context, lease: str | None = None, controller: str | None = None) -> dict:
        """Wake the phone and swipe up from the bottom edge (phones without a passcode). Taps and swipes on
        the lock screen are refused until it is unlocked."""
        return await bridge.unlock(device, caller=_caller(ctx, controller), lease=lease)

    @tool
    async def launch(device: str, bundle_id: str, ctx: Context, lease: str | None = None, controller: str | None = None) -> str:
        """Launch an app by bundle id, e.g. com.apple.Preferences."""
        await bridge.launch(device, bundle_id, caller=_caller(ctx, controller), lease=lease)
        return "ok"

    @tool
    async def open_url(device: str, url: str, ctx: Context, bundle_id: str | None = None,
                       lease: str | None = None, controller: str | None = None) -> str:
        """Open a URL in its app (http/https -> Safari, shortcuts:// -> Shortcuts; else pass bundle_id) if the
        phonebridge config's [policy] urls allows it, e.g. shortcuts://run-shortcut?name=ReadyFeed"""
        await bridge.open_url(device, url, bundle_id, caller=_caller(ctx, controller), lease=lease)
        return "ok"

    @tool
    async def lock_state(device: str) -> dict:
        """Whether the phone is locked (needs the passcode) and the screen state."""
        return await bridge.lock_state(device)

    from .run import taskapi
    taskapi.register(mcp, tool, bridge)  # run_submit / run_events / run_status / run_park / run_resume / run_ack
    return mcp


def with_presence_route(app, bridge):
    """GET /presence?device=<alias>[&fresh=1] (default: the first configured phone) → the same JSON as the MCP tool, for
    monitors that don't speak MCP (a status page, a health alert). Same server, same loopback port."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def presence(request: Request):
        device = request.query_params.get("device") or next(iter(bridge.config.devices), None)
        if not device:
            return JSONResponse({"error": "no phone configured: run phonebridge setup"}, status_code=404)
        fresh = request.query_params.get("fresh", "") in ("1", "true", "yes")
        try:
            return JSONResponse(await bridge.presence(device, fresh=fresh))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
    app.router.routes.insert(0, Route("/presence", presence, methods=["GET"]))
    return app


class BearerAuth:
    """ASGI guard: every HTTP request needs `Authorization: Bearer <token>` (constant-time compare)."""

    def __init__(self, app, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not hmac.compare_digest(dict(scope["headers"]).get(b"authorization", b""),
                                                                self.expected):
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self.app(scope, receive, send)
