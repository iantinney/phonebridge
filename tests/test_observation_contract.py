"""Live adapter evidence must come from exactly one immutable frame."""
import asyncio
import base64
import hashlib

from mcp.server.mcpserver import Image, MCPServer

from phonebridge.screens.live import MCPPhone
from test_bridge import LIT, OTHER


def test_screen_and_ocr_are_one_observation():
    server = MCPServer('moving-phone')
    calls = []
    @server.tool()
    async def screen(device: str, format: str = 'png') -> Image:
        calls.append('screen')
        return Image(data=LIT, format='png')
    @server.tool()
    async def ocr(device: str, with_image: bool = False) -> dict | list:
        calls.append('ocr')
        items = [{'text': hashlib.sha256(OTHER).hexdigest()}]
        return {'items': items, 'image_jpeg': base64.b64encode(OTHER).decode(),
                'width': 120, 'height': 260} if with_image else items
    async def go():
        async with MCPPhone(server, 'iphone12') as phone:
            png = await phone.screen()
            items = await phone.ocr(png)
            assert items[0]['text'] == hashlib.sha256(png).hexdigest()
            assert calls == ['ocr']
    asyncio.run(go())


def test_adapter_refuses_ocr_for_unobserved_pixels():
    async def go():
        phone = MCPPhone('unused', 'iphone12')
        import pytest
        with pytest.raises(ValueError, match='other than this observation'):
            await phone.ocr(LIT)
    asyncio.run(go())
