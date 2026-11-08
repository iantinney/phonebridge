"""Independent review (azure/tw-phone-review) of the device display size in tap mapping (root GO 11-08). The stream
decoder can deliver 1184×2576 frames (1170×2532 coded up to 16-px macroblocks, screen at the origin, padding right
and bottom); an OCR box on such a frame is in true device pixels, so a tap must normalise against the device's own
display size (one DVT screenshot or device metadata), never the padded frame's. Fakes only: no phone, no Mac."""

import asyncio
import struct
import types

import pytest

from phonebridge.backends import pmd3

DISPLAY = (1170, 2532)  # what the device reports (a DVT screenshot's PNG size)
PADDED = (1184, 2576)  # what the stream decoder delivers


def _png(w: int, h: int) -> bytes:  # a PNG header is all png_size reads
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h) + bytes(8)


class Shot:
    def __init__(self, size=DISPLAY, fails=False):
        self.size, self.fails, self.calls = size, fails, 0

    async def get_screenshot(self):
        self.calls += 1
        if self.fails:
            raise ConnectionError("DVT screenshot unavailable")
        return _png(*self.size)


def _frame(w, h):
    return types.SimpleNamespace(width=w, height=h, age_ms=lambda: 5.0, png=lambda: _png(w, h), cropped=None,
                                 bgra=bytes(w * h * 4))  # (tw-phone) a real padded frame's margin is black: checked


def _backend(monkeypatch, live, shot):
    """A backend whose phone has a fresh live stream frame `live` and a DVT screenshot instrument `shot`; taps are
    recorded at the HID layer (nothing is sent anywhere)."""
    b, p, sent = pmd3.Pmd3Backend(), pmd3._Phone("udid"), []
    p.dvt_shot = shot
    monkeypatch.setattr(pmd3.Pmd3Backend, "_live", lambda self, ph: live)

    class Hid:
        async def send_report(self, service_id, report):
            sent.append(report)

    async def hid(self, ph):
        return Hid()

    async def run(self, udid, fn, retry, narrow=None):
        return await fn(p)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_hid", hid)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_run", run)
    monkeypatch.setattr(pmd3.Pmd3Backend, "_rsd", lambda self, ph: None)
    return b, p, sent


def _hid_point(x, y):
    return round(x * pmd3.HID_MAX / DISPLAY[0]), round(y * pmd3.HID_MAX / DISPLAY[1])


def test_a_known_ocr_point_on_a_padded_frame_taps_exactly_that_device_point(monkeypatch):
    """A tab-bar line at (585, 2357) on a padded stream frame: the touch is (585, 2357) on the 1170×2532 display, not
    ~43 px higher (2357 × 2532/2576 ≈ 2317) as dividing by the padded frame put it."""
    b, p, sent = _backend(monkeypatch, _frame(*PADDED), Shot())
    got = asyncio.run(b._norm(p, 585, 2357))
    assert got == _hid_point(585, 2357), (got, _hid_point(585, 2357))
    assert asyncio.run(b._norm(p, 1169, 2531)) == _hid_point(1169, 2531)  # the last content pixel stays on screen


def test_the_display_size_is_not_replaced_by_a_newer_frame_size(monkeypatch):
    """A look (fresh stream frame) after the display was learned must not change how taps map."""
    b, p, sent = _backend(monkeypatch, _frame(*PADDED), Shot())
    first = asyncio.run(b._norm(p, 585, 2357))
    asyncio.run(b._capture(p))  # a look from the padded stream
    assert asyncio.run(b._norm(p, 585, 2357)) == first == _hid_point(585, 2357)


def test_an_unknown_display_size_sends_zero_input(monkeypatch):
    """The DVT screenshot can't be taken and no device metadata says the display size: the tap is refused before any
    report reaches the touch surface (never mapped by the padded frame)."""
    b, p, sent = _backend(monkeypatch, _frame(*PADDED), Shot(fails=True))
    with pytest.raises(Exception):
        asyncio.run(b.tap("udid", 585, 2357, 0.01))
    assert sent == []


def test_unsupported_geometry_is_refused_not_guessed(monkeypatch):
    """A live frame smaller than the display (or in the other orientation) can't be cropped to it: input is refused
    or the display is re-learned for the new geometry; never a tap mapped by a guess."""
    for live, shot in ((_frame(1000, 2000), Shot()), (_frame(2576, 1184), Shot())):
        b, p, sent = _backend(monkeypatch, live, shot)
        try:
            got = asyncio.run(b._norm(p, 585, 1000))
        except Exception:
            continue  # refused: fine
        refreshed = shot.calls >= 2 or (shot.calls == 1 and live.width < live.height)
        assert refreshed and got == _hid_point(585, 1000), (live.width, live.height, got, shot.calls)



def test_a_padded_frame_whose_margin_is_not_padding_is_refused(monkeypatch):
    """(tw-phone) The margin beyond the display is checked dark before a padded frame's pixels count as the display's."""
    import asyncio
    b, p = pmd3.Pmd3Backend(), pmd3._Phone("udid")
    p.display = DISPLAY
    w, h = 1184, 2576
    px = bytearray(w * h * 4)
    for y in range(h):  # a bright right margin: content, not padding
        px[(y * w + 1177) * 4:(y * w + 1177) * 4 + 3] = b"\xff\xff\xff"
    frame = types.SimpleNamespace(width=w, height=h, age_ms=lambda: 5.0, png=lambda: _png(w, h), cropped=None, bgra=bytes(px))
    monkeypatch.setattr(b, "_live", lambda p: frame)
    with pytest.raises(RuntimeError, match="unsupported geometry"):
        asyncio.run(b._norm(p, 585, 2357))
