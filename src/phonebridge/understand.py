"""The local understanding model as a warm Mac service (CONTROLLER_BAR L3; supervisor 2026-10-18 23:5xZ).

Qwen3.5-4B (4-bit, `mlx-community/Qwen3.5-4B-4bit`, Apache-2.0) through mlx-vlm, measured on the M5 Pro
(`docs/reports/local-vlm-bench.md`): warm p95 1.55–1.74 s per post with the short prompt at 672 px, cold first call
1.2–4.5 s, ~4.3 GB resident. So it runs as its own process (mlx-vlm is ~800 MB of packages: optional group
`understand`), loads the model on the first request of a session and drops it after `idle_s` without requests,
and answers on 127.0.0.1 only. The daemon's `understand` tool sends it the latest frame; the director calls that
tool, or `Client` from the Mac.

    uv run --extra understand phonebridge understand serve --port 8935     # keep it running with a LaunchAgent

    POST /understand  {"image": base64 JPEG/PNG, "prompt"?: str, "max_tokens"?: int}
                      -> {"summary", "interest", "parsed", "raw", "ms", "model", "loaded_s", "tokens": {...}}
    GET  /health      -> {"loaded", "model", "idle_s", "last_ms", "loads", "unloads", "requests"}
    POST /load, POST /unload
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable

from PIL import Image

log = logging.getLogger(__name__)

MODEL = "mlx-community/Qwen3.5-4B-4bit"
WIDTH = 672  # the bench's width: 1050 prompt tokens, the same summaries as full size
MAX_TOKENS = 120
IDLE_S = 600.0  # unload after this long without a request (his daily laptop; 4.3 GB back)
PORT = 8935
SEED = ["gym", "fitness", "calisthenics", "nutrition", "workout", "mindset", "business"]  # until Ian's taste list lands
SYSTEM = "You describe one social-media post from a phone screenshot."
SHORT = ("This is a phone screenshot of one social-media feed post. Reply with JSON only: "
         '{"summary": one sentence: who posted (the @handle shown), what it shows, its hook, its topic; '
         f'"interest": an integer from 0 to 3, how much a viewer who likes {", ".join(SEED)} would care}}')


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    generation_tokens: int = 0


class MlxModel:
    """Qwen through mlx-vlm, loaded in __init__ (the slow part); generate() is the warm path."""

    def __init__(self, model_id: str = MODEL):
        from mlx_vlm import apply_chat_template, load

        self.model, self.processor = load(model_id)
        self._template = lambda text: apply_chat_template(
            self.processor, self.model.config, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            num_images=1, enable_thinking=False)

    def generate(self, image: Image.Image, prompt: str, max_tokens: int) -> Generation:
        from mlx_vlm import generate

        r = generate(self.model, self.processor, self._template(prompt), image=[image], max_tokens=max_tokens,
                     temperature=0.0, verbose=False)
        return Generation(r.text, getattr(r, "prompt_tokens", 0), getattr(r, "generation_tokens", 0))


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


class Service:
    """One model, loaded on demand, unloaded after idle; one request at a time (the GPU is one)."""

    def __init__(self, loader: Callable[[], object] | None = None, model_id: str = MODEL, idle_s: float = IDLE_S,
                 width: int = WIDTH, max_tokens: int = MAX_TOKENS):
        self.model_id, self.idle_s, self.width, self.max_tokens = model_id, idle_s, width, max_tokens
        self._loader = loader or (lambda: MlxModel(model_id))
        self.model = None
        self._lock = asyncio.Lock()
        self._timer: asyncio.TimerHandle | None = None
        self.loaded_at = self.last_at = None
        self.last_ms = None
        self.loads = self.unloads = self.requests = 0

    async def load(self) -> float:
        if self.model is None:
            t = time.perf_counter()
            self.model = await asyncio.to_thread(self._loader)
            self.loaded_at, self.loads = time.monotonic(), self.loads + 1
            log.warning("understand: %s loaded in %.1f s", self.model_id, time.perf_counter() - t)
        self._arm()
        return 0.0 if self.loaded_at is None else time.monotonic() - self.loaded_at

    def _arm(self) -> None:
        if self._timer:
            self._timer.cancel()
        if self.idle_s:
            self._timer = asyncio.get_running_loop().call_later(self.idle_s, self.unload)

    def unload(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self.model is not None:
            self.model, self.loaded_at, self.unloads = None, None, self.unloads + 1
            with contextlib.suppress(Exception):
                import gc

                gc.collect()
                import mlx.core as mx  # type: ignore

                mx.clear_cache()
            log.warning("understand: model unloaded after %.0f s idle", self.idle_s)

    def health(self) -> dict:
        now = time.monotonic()
        return {"loaded": self.model is not None, "model": self.model_id,
                "idle_s": None if self.last_at is None else round(now - self.last_at, 1),
                "last_ms": self.last_ms, "loads": self.loads, "unloads": self.unloads, "requests": self.requests}

    async def understand(self, image: bytes | Image.Image, prompt: str | None = None, max_tokens: int | None = None) -> dict:
        t0 = time.perf_counter()
        async with self._lock:
            loaded_s = await self.load()
            im = image if isinstance(image, Image.Image) else Image.open(io.BytesIO(image))
            im = im.convert("RGB")
            if im.width > self.width:
                im = im.resize((self.width, round(im.height * self.width / im.width)), Image.BILINEAR)
            gen = await asyncio.to_thread(self.model.generate, im, prompt or SHORT, max_tokens or self.max_tokens)
            self._arm()
        self.requests += 1
        self.last_at = time.monotonic()
        self.last_ms = round((time.perf_counter() - t0) * 1000, 1)
        data = parse(gen.text)
        out = {"summary": (data or {}).get("summary", gen.text.strip()), "parsed": data is not None, "raw": gen.text,
               "ms": self.last_ms, "model": self.model_id, "loaded_s": round(loaded_s, 1),
               "tokens": {"prompt": gen.prompt_tokens, "generation": gen.generation_tokens}}
        interest = (data or {}).get("interest")
        out["interest"] = int(interest) if isinstance(interest, (int, float)) and 0 <= interest <= 3 else None
        for k, v in (data or {}).items():
            out.setdefault(k, v)
        return out


def app(service: Service):
    """A Starlette app on the service (uvicorn serves it on 127.0.0.1 only)."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(_: Request):
        return JSONResponse(service.health())

    async def understand(req: Request):
        body = await req.json()
        try:
            image = base64.b64decode(body["image"])
        except Exception:
            return JSONResponse({"error": "image: base64 JPEG or PNG required"}, status_code=400)
        try:
            return JSONResponse(await service.understand(image, body.get("prompt"), body.get("max_tokens")))
        except Exception as e:
            log.exception("understand failed")
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    async def load(_: Request):
        await service.load()
        return JSONResponse(service.health())

    async def unload(_: Request):
        service.unload()
        return JSONResponse(service.health())

    return Starlette(routes=[Route("/health", health), Route("/understand", understand, methods=["POST"]),
                             Route("/load", load, methods=["POST"]), Route("/unload", unload, methods=["POST"])])


class Client:
    """The director's handle on the service (from the Mac, or through the daemon's `understand` tool)."""

    def __init__(self, url: str = f"http://127.0.0.1:{PORT}", http=None, timeout: float = 30.0):
        import httpx2 as httpx  # mcp's HTTP client (httpx API), no new dependency

        self.url = url.rstrip("/")
        self.http = http or httpx.AsyncClient(timeout=timeout)

    async def understand(self, image: bytes, prompt: str | None = None, max_tokens: int | None = None) -> dict:
        body = {"image": base64.b64encode(image).decode()}
        if prompt:
            body["prompt"] = prompt
        if max_tokens:
            body["max_tokens"] = max_tokens
        r = await self.http.post(f"{self.url}/understand", json=body)
        r.raise_for_status()
        return r.json()

    async def health(self) -> dict:
        r = await self.http.get(f"{self.url}/health")
        r.raise_for_status()
        return r.json()


def serve(port: int = PORT, idle_s: float = IDLE_S, model_id: str = MODEL, preload: bool = False) -> None:
    import uvicorn

    service = Service(model_id=model_id, idle_s=idle_s)
    application = app(service)
    if preload:
        @application.on_event("startup")
        async def _preload():
            await service.load()
    uvicorn.run(application, host="127.0.0.1", port=port, log_level="warning")
