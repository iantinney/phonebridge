"""A synthetic CoreDevice-style screen stream for the tests: PIL frames -> HEVC access units (libx265 via PyAV)
-> RTP/HEVC packets (RFC 7798: single NAL units and fragmentation units, marker on the last packet of a frame)."""
from __future__ import annotations

import av
from PIL import Image

START = b"\x00\x00\x00\x01"


def hevc_aus(frames: list[Image.Image], keyint: int = 3) -> list[bytes]:
    """One Annex-B access unit per frame; VPS/SPS/PPS repeat before every IDR (as Apple's stream does)."""
    ctx = av.CodecContext.create("libx265", "w")
    w, h = frames[0].size
    ctx.width, ctx.height, ctx.pix_fmt = w, h, "yuv420p"
    ctx.framerate = 30
    ctx.options = {"x265-params": f"keyint={keyint}:min-keyint={keyint}:scenecut=0:repeat-headers=1:annexb=1:"
                                  "bframes=0:log-level=none", "preset": "ultrafast", "tune": "zerolatency"}
    aus = []
    for im in frames:
        fr = av.VideoFrame.from_image(im.convert("RGB")).reformat(format="yuv420p")
        aus += [bytes(p) for p in ctx.encode(fr)]
    aus += [bytes(p) for p in ctx.encode(None)]
    return aus


def nals(annexb: bytes) -> list[bytes]:
    """Split an Annex-B stream on its 4- and 3-byte start codes."""
    parts = annexb.replace(START, b"\x00\x00\x01").split(b"\x00\x00\x01")
    return [p for p in parts if p]


def rtp_packets(aus: list[bytes], mtu: int = 1200, ssrc: int = 0x1234, seq0: int = 0, ts0: int = 0,
                pt: int = 100) -> list[bytes]:
    """RTP packets for the access units: AP-free, FU for NALs over `mtu`, marker on each AU's last packet."""
    pk, seq, ts = [], seq0, ts0

    def header(marker: bool) -> bytes:
        nonlocal seq
        b = bytes([0x80, (0x80 if marker else 0) | pt]) + (seq & 0xFFFF).to_bytes(2, "big") + \
            (ts & 0xFFFFFFFF).to_bytes(4, "big") + ssrc.to_bytes(4, "big")
        seq += 1
        return b

    for au in aus:
        units = nals(au)
        for k, nal in enumerate(units):
            last_nal = k == len(units) - 1
            if len(nal) <= mtu:
                pk.append(header(last_nal) + nal)
                continue
            nal_type = (nal[0] >> 1) & 0x3F
            fu_indicator = bytes([(nal[0] & 0x81) | (49 << 1), nal[1]])
            body, pos, first = nal[2:], 0, True
            while pos < len(body):
                chunk = body[pos:pos + mtu]
                pos += len(chunk)
                end = pos >= len(body)
                fu_header = bytes([(0x80 if first else 0) | (0x40 if end else 0) | nal_type])
                pk.append(header(last_nal and end) + fu_indicator + fu_header + chunk)
                first = False
        ts += 800  # 24 kHz media clock, 30 fps
    return pk


def solid(w: int, h: int, color) -> Image.Image:
    return Image.new("RGB", (w, h), color)
