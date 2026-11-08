"""On-device OCR with Apple Vision (macOS only), straight from PNG bytes — no decode/re-encode."""

from __future__ import annotations

import io
import re
import struct


def png_size(png: bytes) -> tuple[int, int]:
    """(width, height) from the PNG IHDR chunk."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", png[16:24])


def _cgimage(im):
    """A CGImage over a PIL image's pixels (no encode): Vision reads a live stream frame directly."""
    import Quartz
    from Foundation import NSData

    im = im if im.mode == "RGBA" else im.convert("RGBA")
    data = im.tobytes()
    provider = Quartz.CGDataProviderCreateWithCFData(NSData.dataWithBytes_length_(data, len(data)))
    return Quartz.CGImageCreate(im.width, im.height, 8, 32, im.width * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
                                Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrderDefault, provider, None,
                                False, Quartz.kCGRenderingIntentDefault)


def recognize(png, fast: bool = False, min_conf: float = 0.3, correct: bool | None = None) -> list[dict]:
    """Text lines with tap-ready centers, in screenshot pixels: [{text, x, y, w, h, conf}]. `png` is PNG bytes or
    a PIL image (a live stream frame: handed to Vision as a CGImage, no encode). `correct` = Vision's language
    correction (None: its default, on); off makes the fast level ~40 % quicker on UI text."""
    import objc
    import Vision
    from Foundation import NSData

    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(1 if fast else 0)  # 0 = accurate, 1 = fast
        if correct is not None:
            req.setUsesLanguageCorrection_(bool(correct))
        if isinstance(png, (bytes, bytearray)):
            width, height = png_size(png)
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                NSData.dataWithBytes_length_(png, len(png)), None
            )
        else:
            width, height = png.size
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(_cgimage(png), None)
        ok = handler.performRequests_error_([req], None)
        if not (ok[0] if isinstance(ok, tuple) else ok):
            raise RuntimeError(f"Vision OCR failed: {ok}")
        out = []
        for r in req.results() or []:
            cand = r.topCandidates_(1)[0]
            if cand.confidence() < min_conf:
                continue
            b = r.boundingBox()  # normalized, origin bottom-left
            w, h = b.size.width * width, b.size.height * height
            out.append({
                "text": str(cand.string()),
                "x": round(b.origin.x * width + w / 2),
                "y": round((1 - b.origin.y) * height - h / 2),
                "w": round(w),
                "h": round(h),
                "conf": round(float(cand.confidence()), 2),
            })
        return out


def find(items: list[dict], substring: str) -> dict | None:
    """Best match for `substring` (case-insensitive): exact line first, then shortest containing line."""
    s = substring.casefold()
    hits = [i for i in items if s in i["text"].casefold()]
    if not hits:
        return None
    return min(hits, key=lambda i: (i["text"].casefold() != s, len(i["text"]), -i["conf"]))


CLOCK = re.compile(r"\d{1,2}:\d{2}")
DATE = re.compile(r"(mon|tue|wed|thu|fri|sat|sun)[a-z]*,? (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}")
LOCK_HINTS = ("swipe up to open", "swipe up to unlock", "press home to open", "press home to unlock")


def lock_screen(png: bytes, items: list[dict]) -> str | None:
    """Why this screenshot looks like the lock screen (or a dark display), else None. Signals: a dark
    display; "Swipe up to open"-style text; the centred date line ("Sun Oct 18") at the top;
    the large clock centred in the upper half; or nothing readable but the status bar (the iOS 27 lock
    screen can show only a lock glyph). Use accurate OCR: fast mode misses the lock-screen clock."""
    from PIL import Image

    width, height = png_size(png)
    if Image.open(io.BytesIO(png)).convert("L").getextrema()[1] < 8:
        return "the display is off"
    for i in items:
        text = i["text"].strip()
        centred = abs(i["x"] - width / 2) < 0.15 * width
        if any(hint in text.casefold() for hint in LOCK_HINTS) or (
                DATE.fullmatch(text.casefold()) and centred and i["y"] < 0.3 * height):
            return f"{text!r} is on screen"
        if CLOCK.fullmatch(text) and i["h"] > 0.05 * height and centred and i["y"] < 0.45 * height:
            return "the lock-screen clock is showing"
    if all(i["y"] < 0.06 * height for i in items):
        return "nothing but the status bar is readable"
    return None


CHANGED = 0.08  # share of cells: other screens scored ≥ 0.23 (dark mode included); the same one ≤ 0.02 when
# dimmed, re-compressed or nudged 6 px


def change(a: bytes, b: bytes) -> float:
    """How different two screenshots are: the share of 16×28 grey-thumbnail cells that differ by over 24 levels,
    ignoring the top 18% (status bar, notification banners) and the bottom 4% (home indicator). Counting cells,
    not averaging them, still sees two dark-mode screens apart."""
    from PIL import Image, ImageChops, ImageStat

    def thumb(png: bytes):
        im = Image.open(io.BytesIO(png)).convert("L")
        w, h = im.size
        return im.crop((0, int(h * 0.18), w, int(h * 0.96))).resize((16, 28), Image.Resampling.BOX)

    return ImageStat.Stat(ImageChops.difference(thumb(a), thumb(b)).point(lambda v: 255 if v > 24 else 0)).mean[0] / 255
