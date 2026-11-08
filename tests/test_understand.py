"""The warm local understanding model as a Mac service (supervisor 2026-10-18 23:5xZ): loaded on demand, unloaded
after idle, health for the heartbeat, a client for the director, and the daemon's `understand` tool that sends it the
latest frame. Offline: the model is a fake that returns JSON."""
import asyncio
import base64
import io
import json
import time

import httpx2 as httpx  # the HTTP client mcp already ships (httpx API)
import pytest
from PIL import Image

from phonebridge import understand
from phonebridge.host import heartbeat


class FakeModel:
    loads = 0

    def __init__(self, reply='{"summary": "@samplewatches shows a Rolex on a wrist: a watch flex, topic luxury watches", "interest": 1}'):
        FakeModel.loads += 1
        self.reply, self.calls = reply, []

    def generate(self, image: Image.Image, prompt: str, max_tokens: int) -> understand.Generation:
        self.calls.append((image.size, prompt, max_tokens))
        return understand.Generation(self.reply, prompt_tokens=1000, generation_tokens=40)


def png_bytes(w=1170, h=2532, color="navy") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), color).save(out, "PNG")
    return out.getvalue()


@pytest.fixture(autouse=True)
def reset():
    FakeModel.loads = 0


@pytest.fixture
def service():
    return understand.Service(loader=FakeModel, idle_s=5.0, width=672)


def test_first_request_loads_the_model_and_returns_the_parsed_answer(service):
    async def go():
        r = await service.understand(png_bytes())
        assert FakeModel.loads == 1 and r["summary"].startswith("@samplewatches") and r["interest"] == 1
        assert r["ms"] >= 0 and r["model"] == service.model_id and r["loaded_s"] >= 0
        assert service.model.calls[0][0] == (672, 1454)  # the frame is downscaled to the bench's width
        assert "JSON" in service.model.calls[0][1]
        await service.understand(png_bytes())
        assert FakeModel.loads == 1  # warm
    asyncio.run(go())


def test_model_unloads_after_idle_and_health_says_so():
    service = understand.Service(loader=FakeModel, idle_s=0.3, width=672)

    async def go():
        assert service.health()["loaded"] is False
        await service.understand(png_bytes(w=200, h=400))
        assert service.health()["loaded"] is True and service.health()["last_ms"] >= 0
        await asyncio.sleep(0.8)
        assert service.health()["loaded"] is False and service.health()["unloads"] == 1
        await service.understand(png_bytes())
        assert FakeModel.loads == 2
    asyncio.run(go())


def test_an_answer_that_is_not_json_is_returned_raw_with_interest_none():
    s = understand.Service(loader=lambda: FakeModel("a wrist watch, nice"), idle_s=10)
    r = asyncio.run(s.understand(png_bytes()))
    assert r["summary"] == "a wrist watch, nice" and r["interest"] is None and r["parsed"] is False


def test_http_app_and_client_round_trip(service):
    app = understand.app(service)

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://understand") as http:
            h = (await http.get("/health")).json()
            assert h["loaded"] is False and h["model"] == service.model_id
            c = understand.Client("http://understand", http=http)
            r = await c.understand(png_bytes(), prompt="what is it")
            assert r["interest"] == 1 and service.model.calls[-1][1] == "what is it"
            assert (await c.health())["loaded"] is True
            assert (await http.post("/unload")).json()["loaded"] is False
    asyncio.run(go())


def test_daemon_understand_tool_sends_the_latest_frame(tmp_path, monkeypatch):
    from phonebridge import mcp_server, stream
    from phonebridge.bridge import Bridge
    from test_bridge import FakeBackend, config
    from test_screens_fast import bgra

    class Streaming(FakeBackend):
        def __init__(self):
            super().__init__()
            self.stream = stream.Frames()

        async def frames(self, udid):
            return self.stream

    be = Streaming()
    be.stream.push(*bgra(png_bytes(color="red")))
    b = Bridge(be, config([]), log_path=tmp_path / "calls.jsonl")
    sent = []

    class FakeClient:
        async def understand(self, image: bytes, prompt=None, max_tokens=None):
            sent.append((image[:3], prompt))
            return {"summary": "a red screen", "interest": 0, "ms": 1200.0, "model": "fake", "parsed": True}
    b.understanding = FakeClient()
    out = asyncio.run(b.understand("iphone12", prompt="short"))
    assert out["summary"] == "a red screen" and out["frame"]["seq"] == 1 and out["frame"]["source"] == "stream"
    assert sent[0][0] == b"\xff\xd8\xff" and sent[0][1] == "short"  # a JPEG of the frame, not a PNG
    assert "understand" in {t.name for t in asyncio.run(mcp_server.build(b).list_tools())}


def test_heartbeat_mac_status_carries_the_service_health(monkeypatch):
    monkeypatch.setattr(heartbeat, "understand_health", lambda: {"loaded": True, "idle_s": 3.0})
    monkeypatch.setattr(heartbeat.sys, "platform", "darwin")
    monkeypatch.setattr(heartbeat.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "98%; AC Power"})())
    assert heartbeat.mac_status()["understand"] == {"loaded": True, "idle_s": 3.0}
