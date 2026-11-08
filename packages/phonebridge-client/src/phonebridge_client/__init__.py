"""Small transport shared by the VM poster, Mac director and screen engine."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import uuid
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass

from PIL import Image

INPUTS = frozenset({"tap", "double_tap", "swipe", "scroll", "type", "paste", "home", "unlock",
                    "launch", "open_url", "record_start", "ax_probe"})


def as_png(data: bytes) -> bytes:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data
    out = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(out, "PNG", compress_level=1)
    return out.getvalue()


@dataclass(frozen=True)
class Observation:
    frame_id: str
    width: int
    height: int
    png: bytes
    items: list[dict]
    age_ms: float
    received: float
    frame_seq: int | None = None  # the stream frame both the text and the pixels came from (None: a still)

    @classmethod
    def from_reply(cls, reply: dict) -> Observation:
        encoded = reply.get("frame_jpeg") or reply.get("image_jpeg")
        if not encoded:
            raise ValueError("observation has no pixels from its OCR frame")
        png = as_png(base64.b64decode(encoded, validate=True))
        width, height = Image.open(io.BytesIO(png)).size
        if reply.get("width", width) != width or reply.get("height", height) != height:
            raise ValueError("observation dimensions disagree with its pixels")
        # The hash disambiguates sequence resets after a daemon/capture restart.
        frame_id = f"{reply.get('frame_seq', 'still')}:{hashlib.sha256(png).hexdigest()}"
        return cls(frame_id, width, height, png, reply["items"], float(reply.get("frame_age_ms", 0)),
                   time.monotonic(), reply.get("frame_seq"))

    @property
    def current_age_ms(self) -> float:
        return self.age_ms + (time.monotonic() - self.received) * 1000


class MCPClient:
    error_type = RuntimeError

    def __init__(self, server):
        self.server = server
        self._stack = AsyncExitStack()
        self._client = None
        self._tools = None
        self._schemas = {}
        self.controller = uuid.uuid4().hex
        self._controlled = set()
        self.stream_fallbacks, self.last_fallback = 0, None

    async def __aenter__(self):
        from mcp import Client
        self._client = await self._stack.enter_async_context(Client(self.server, read_timeout_seconds=90))
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        try:
            for device in self._controlled:
                with suppress(Exception):
                    await self._call("release_controller", device=device, controller=self.controller)
            self._controlled.clear()
        finally:
            await self._stack.aclose()

    async def _list_tools(self):
        if self._tools is None:
            listed = (await self._client.list_tools()).tools
            self._tools = {t.name for t in listed}
            self._schemas = {t.name: (getattr(t, "input_schema", None) or
                                     getattr(t, "inputSchema", None) or {}).get("properties", {}) for t in listed}

    async def _offered(self, tool):
        await self._list_tools()
        return tool in self._tools

    async def _accepts(self, tool, argument):
        await self._list_tools()
        return argument in self._schemas.get(tool, {})

    async def _call(self, tool, **args):
        args = {k: v for k, v in args.items() if v is not None}
        if tool in INPUTS:
            if not await self._accepts(tool, "controller"):
                raise self.error_type(f"phonebridge {tool}: daemon lacks controller ownership; upgrade it first")
            args["controller"] = self.controller
            self._controlled.add(args["device"])
        result = await self._client.call_tool(tool, args)
        if result.is_error:
            raise self.error_type(f"phonebridge {tool}: " + " ".join(getattr(c, "text", "") for c in result.content))
        return result

    async def _data(self, tool, **args):
        result = await self._call(tool, **args)
        structured = result.structured_content
        if structured is not None:
            return structured["result"] if set(structured) == {"result"} else structured
        text = "".join(getattr(c, "text", "") for c in result.content)
        return json.loads(text) if text else None

    async def observe(self, device, platform=None, *, use_look=True):
        if use_look and await self._accepts("look", "with_frame"):
            try:
                return Observation.from_reply(await self._data("look", device=device, platform=platform,
                                                               with_frame=True))
            except Exception as error:
                self.stream_fallbacks += 1
                self.last_fallback = str(error)[:160]
        if not await self._accepts("ocr", "with_image"):
            raise self.error_type("daemon cannot return paired pixels/OCR; upgrade it first")
        return Observation.from_reply(await self._data("ocr", device=device, with_image=True))
