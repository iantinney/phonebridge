import base64
import io

import pytest
from PIL import Image

from phonebridge_client import Observation


def reply():
    out = io.BytesIO()
    Image.new('RGB', (3, 5)).save(out, 'PNG')
    return {'image_jpeg': base64.b64encode(out.getvalue()).decode(), 'items': [{'text': 'one frame'}],
            'width': 3, 'height': 5, 'frame_age_ms': 10}


def test_observation_carries_frame_identity_dimensions_and_age():
    r = reply()
    observation = Observation.from_reply(r)
    assert observation.width == 3 and observation.height == 5
    assert observation.items == r['items']
    assert observation.current_age_ms >= 10
    assert observation.frame_id == Observation.from_reply(r).frame_id


def test_observation_rejects_missing_or_mis_sized_pixels():
    with pytest.raises(ValueError, match='no pixels'):
        Observation.from_reply({'items': []})
    with pytest.raises(ValueError, match='dimensions'):
        Observation.from_reply({**reply(), 'width': 4})


def test_observation_keeps_its_frame_seq():
    assert Observation.from_reply({**reply(), 'frame_seq': 41}).frame_seq == 41
    assert Observation.from_reply(reply()).frame_seq is None


def test_optional_arguments_are_read_from_either_schema_spelling():
    """R1 attempt 3: a daemon exposing only `inputSchema` must still show its optional args (swipe release, think)."""
    import asyncio
    from types import SimpleNamespace

    from phonebridge_client import MCPClient

    tool = SimpleNamespace(name='swipe', inputSchema={'properties': {'release': {}, 'think': {}}})  # no input_schema attribute
    c = MCPClient.__new__(MCPClient)
    c._tools, c._schemas = None, {}
    c._client = SimpleNamespace(list_tools=lambda: asyncio.sleep(0, SimpleNamespace(tools=[tool])))
    assert asyncio.run(c._accepts('swipe', 'release')) and asyncio.run(c._accepts('swipe', 'think'))
    assert not asyncio.run(c._accepts('swipe', 'teleport'))
