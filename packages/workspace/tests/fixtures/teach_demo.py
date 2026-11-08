"""FIXTURE — a synthetic screen recording, not a human demonstration and not a real app.

`render(dir)` draws a fictional app ("Studio") with ffmpeg (static screens: drawbox/drawtext) and composes the motion a
person makes in Python: push transitions, a pressed button's highlight for a few frames before it navigates, three
flicks with a decaying coast in the clip picker, a selection badge, a keyboard sliding up, a caption typed character by
character with a pause and a corrected typo, think-time holds drawn from a log-normal. One transition (Edit → Details)
has no visible highlight, as when a quick tap falls between frames, so the analysis must ask about it. The result is an
H.264 mp4 like a phone's screen recording; `truth` (returned, never given to the analyser) is what the "person" did.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

W, H, FPS = 590, 1276, 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
HEADER = 200  # the picker's fixed header (px); its grid scrolls below it
NARRATION = ("This is how I prepare a client post in Studio. I'm on the client's account @acme_studio. Tap Create, pick "
             "the newest video, the one I send you, then Next. Write the caption I give you and always end it with #acme. "
             "Save it as a draft. Never press Share without my approval.")
CAPTION = "New drop is live! Link in bio #acme"
DURATIONS = ["0:15", "0:42", "1:03", "0:08", "0:27", "0:51", "0:12", "2:10", "0:33", "0:19", "0:46", "1:20",
             "0:09", "0:37", "0:58", "0:24", "1:41", "0:06"]


def available() -> bool:
    return bool(shutil.which("ffmpeg")) and Path(FONT).exists() and Path(BOLD).exists()


class Canvas:
    """One static image as ffmpeg filters (boxes, text through text files: no escaping)."""

    def __init__(self, tmp: Path, h: int = H, bg: str = "white"):
        self.tmp, self.h, self.bg, self.f = tmp, h, bg, []

    def box(self, x, y, w, h, color):
        self.f.append(f"drawbox=x={x}:y={y}:w={w}:h={h}:color={color}:t=fill")
        return self

    def text(self, s, x, y, size=30, color="black", bold=False, centre=False):
        p = tempfile.NamedTemporaryFile("w", dir=self.tmp, suffix=".txt", delete=False)
        p.write(s)
        p.close()
        xs = f"({x})-text_w/2" if centre else str(x)
        self.f.append(f"drawtext=fontfile={BOLD if bold else FONT}:textfile={p.name}:x={xs}:y={y}:fontsize={size}:fontcolor={color}")
        return self

    def render(self) -> bytes:
        vf = ",".join(["format=gray", *self.f]) if self.f else "format=gray"
        return subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c={self.bg}:s={W}x{self.h}:d=1", "-vf", vf,
                               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout


def _status(c: Canvas) -> Canvas:
    return c.text("9:41", 28, 18, 24, bold=True)


def _nav(c: Canvas, left: str | None, title: str, right: str | None, press: str | None = None) -> Canvas:
    if press == "left":
        c.box(14, 84, 130, 56, "0xCCCCCC")
    if press == "right":
        c.box(W - 150, 84, 136, 56, "0xCCCCCC")
    if left:
        c.text(left, 30, 96, 30, "0x1E5BD8")
    c.text(title, W // 2, 96, 32, bold=True, centre=True)
    if right:
        c.text(right, W - 120, 96, 30, "0x1E5BD8", bold=True)
    return c.box(0, 150, W, 2, "0xDDDDDD")


def home(tmp, press=False) -> bytes:
    c = _status(Canvas(tmp))
    c.text("Studio", W // 2, 96, 34, bold=True, centre=True).text("@acme_studio", W // 2, 150, 26, "0x555555", centre=True)
    for k, y in enumerate((220, 560)):
        c.box(30, y, W - 60, 300, "0xE6E6E6").text(f"Client post {k + 1}", 60, y + 30, 28, "0x333333")
    c.box(0, H - 120, W, 2, "0xDDDDDD")
    if press:
        c.box(150, H - 105, 140, 70, "0xCCCCCC")
    for x, t in ((30, "Home"), (170, "Create"), (330, "Inbox"), (460, "Profile")):
        c.text(t, x, H - 85, 28, "0x222222")
    return c.render()


def picker_header(tmp, press=False) -> bytes:
    c = _nav(_status(Canvas(tmp, HEADER)), "Cancel", "New post", "Next", "right" if press else None)
    return c.text("Recents", 30, 160, 26, "0x555555").render()


def picker_grid(tmp, selected: bool) -> bytes:
    """The scrolling grid (taller than the screen): 3 columns of clips, newest first, each with its duration."""
    rows = math.ceil(len(DURATIONS) / 3)
    c = Canvas(tmp, rows * 262 + 40)
    for n, d in enumerate(DURATIONS):
        x, y = 8 + (n % 3) * 194, 10 + (n // 3) * 262
        c.box(x, y, 186, 252, f"0x{0x50 + (n * 23) % 0x70:02X}{0x50 + (n * 23) % 0x70:02X}{0x50 + (n * 23) % 0x70:02X}")
        c.box(x + 100, y + 200, 78, 42, "white").text(d, x + 108, y + 206, 26)
        if selected and n == 0:
            c.box(x, y, 186, 8, "0x1E5BD8").box(x, y + 244, 186, 8, "0x1E5BD8").box(x, y, 8, 252, "0x1E5BD8").box(x + 178, y, 8, 252, "0x1E5BD8")
            c.box(x + 136, y + 14, 38, 38, "0x1E5BD8").text("1", x + 147, y + 18, 26, "white", bold=True)
    return c.render()


def edit(tmp) -> bytes:
    c = _nav(_status(Canvas(tmp)), "Back", "Edit", "Next")
    c.box(60, 200, W - 120, 700, "0x3A3A3A").box(W - 170, 840, 90, 44, "white").text("0:15", W - 160, 846, 26)
    for x, t in ((40, "Trim"), (170, "Text"), (300, "Music"), (440, "Filters")):
        c.text(t, x, 960, 28, "0x222222")
    return c.render()


KEYS = ("Q W E R T Y U I O P", "A S D F G H J K L", "Z X C V B N M")


def details(tmp, caption: str | None, keyboard: bool, press: str | None = None) -> bytes:
    c = _nav(_status(Canvas(tmp)), "Back", "Details", None)
    c.box(30, 190, 150, 200, "0x4A4A4A")
    if caption:
        words, line, lines = caption.split(" "), "", []
        for w in words:  # wrap at ~18 characters, as a narrow caption field does
            if len(line) + len(w) + 1 > 18 and line:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        for k, ln in enumerate(lines):
            c.text(ln, 200, 200 + k * 42, 28)
    else:
        c.text("Write a caption...", 200, 200, 28, "0x888888")
    c.box(0, 420, W, 2, "0xDDDDDD").text("Posting as @acme_studio", 30, 450, 26, "0x444444")
    c.text("Tag people", 30, 520, 28, "0x222222").text("Add location", 30, 590, 28, "0x222222")
    if keyboard:
        top = H - 520
        c.box(0, top, W, 520, "0xD1D3D9").box(0, top, W, 64, "0xF2F2F2")
        if press == "done":
            c.box(W - 130, top + 6, 116, 52, "0xBBBBBB")
        c.text("Done", W - 110, top + 16, 28, "0x1E5BD8", bold=True)
        for r, row in enumerate(KEYS):
            keys = row.split(" ")
            x0 = (W - len(keys) * 57) // 2
            for k, ch in enumerate(keys):
                c.box(x0 + k * 57 + 3, top + 90 + r * 100, 51, 80, "white").text(ch, x0 + k * 57 + 17, top + 108 + r * 100, 32)
        c.box(150, top + 395, 290, 80, "white").text("space", 255, top + 418, 26, "0x444444")
    else:
        if press == "save":
            c.box(30, H - 150, 255, 90, "0xBBBBBB")
        else:
            c.box(30, H - 150, 255, 90, "0x222222").box(33, H - 147, 249, 84, "white")
        c.text("Save draft", 85, H - 122, 30, "0x222222")
        c.box(305, H - 150, 255, 90, "0x1E5BD8").text("Share", 390, H - 122, 30, "white", bold=True)
    return c.render()


def drafts(tmp) -> bytes:
    c = _nav(_status(Canvas(tmp)), "Back", "Drafts", None)
    c.box(30, 190, 110, 150, "0x4A4A4A").text(CAPTION[:18] + "...", 160, 220, 26).text("Just now", 160, 270, 24, "0x777777")
    return c.box(130, H - 260, 330, 70, "0x333333").text("Draft saved", 215, H - 245, 28, "white").render()


# ---- motion ----------------------------------------------------------------------------------------------------------------

def _rows(img: bytes) -> list[bytes]:
    return [img[r * W:(r + 1) * W] for r in range(len(img) // W)]


def push(a: bytes, b: bytes, p: float) -> bytes:
    """A push transition at progress p: the old screen slides left, the new one comes in from the right (status bar fixed)."""
    off = int(W * (1 - (1 - p) ** 3))
    ra, rb = _rows(a), _rows(b)
    return b"".join(ra[r] if r < 60 else (ra[r][off:] + rb[r][:off]) for r in range(H))


def picker(header: bytes, grid: bytes, offset: int) -> bytes:
    rows = _rows(grid)
    body = [rows[min(len(rows) - 1, r + offset)] for r in range(H - HEADER)]
    return header + b"".join(body)


def slide_up(base: bytes, over: bytes, p: float, top: int) -> bytes:
    """The keyboard part of `over` (rows ≥ top) slides up over `base` by progress p."""
    rb, ro = _rows(base), _rows(over)
    shift = int((H - top) * (1 - p))
    return b"".join(ro[r - shift] if r >= top + shift else (ro[r] if r < top else rb[r]) for r in range(H))


def render(out_dir: Path, seed: int = 7) -> dict:
    """Write out_dir/demo.mp4; return {video, narration, truth}. The truth is the scripted person, for the tests only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    think = lambda med: max(0.35, med * math.exp(rng.gauss(0, 0.35)))  # noqa: E731
    tmp = Path(tempfile.mkdtemp(dir=out_dir))
    frames: list[bytes] = []
    truth: list[dict] = []

    def hold(img: bytes, s: float):
        frames.extend([img] * max(1, round(s * FPS)))

    def anim(fn, s: float):
        n = max(2, round(s * FPS))
        frames.extend(fn((k + 1) / n) for k in range(n))

    def mark(**d):
        truth.append({"t": round(len(frames) / FPS, 2), **d})

    s_home, s_home_p = home(tmp), home(tmp, press=True)
    hdr, hdr_p = picker_header(tmp), picker_header(tmp, press=True)
    grid, grid_sel = picker_grid(tmp, False), picker_grid(tmp, True)
    s_edit = edit(tmp)
    s_det, s_det_kb = details(tmp, None, False), details(tmp, None, True)
    s_drafts = drafts(tmp)
    hold(s_home, 2.2)
    mark(op="tap", label="Create", screen="Studio", visible_press=True)
    hold(s_home_p, 0.13)
    anim(lambda p: push(s_home, picker(hdr, grid, 0), p), 0.35)
    offset = 0
    for length, sign in ((380, 1), (260, 1), (640, -1)):  # three flicks: drag 1:1, then a coast decaying at ~0.998/ms
        hold(picker(hdr, grid, offset), think(1.0))
        mark(op="scroll", length_px=length, direction="up" if sign > 0 else "down")
        start, n = offset, round(0.55 * FPS)
        for k in range(1, n + 1):
            frac = 1 - math.exp(-6.5 * k / n)
            frames.append(picker(hdr, grid, max(0, start + sign * int(length * frac / (1 - math.exp(-6.5))))))
        offset = max(0, start + sign * length)
    hold(picker(hdr, grid, offset), think(1.1))
    mark(op="select_media", rank=0, duration="0:15")
    hold(picker(hdr, grid_sel, 0), think(0.9))
    mark(op="tap", label="Next", screen="New post", visible_press=True)
    hold(picker(hdr_p, grid_sel, 0), 0.13)
    anim(lambda p: push(picker(hdr, grid_sel, 0), s_edit, p), 0.35)
    hold(s_edit, 3.1)  # a long look at the preview
    mark(op="tap", label="Next", screen="Edit", visible_press=False)  # too quick to show a highlight
    anim(lambda p: push(s_edit, s_det, p), 0.35)
    hold(s_det, think(1.2))
    mark(op="focus", label="Write a caption...")
    anim(lambda p: slide_up(s_det, s_det_kb, p, H - 520), 0.25)
    hold(s_det_kb, think(0.8))
    mark(op="type", text=CAPTION)
    typed, plan = "", list(CAPTION)
    plan = plan[:plan.index("L") + 1] + ["n", "<"] + plan[plan.index("L") + 1:]  # a slip: "Ln", noticed, deleted
    for ch in plan:
        typed = typed[:-1] if ch == "<" else typed + ch
        hold(details(tmp, typed, True), 1.1 if typed.endswith("live!") else 0.6 if ch == "<" else max(0.07, 0.17 * math.exp(rng.gauss(0, 0.4))))
    hold(details(tmp, typed, True), think(0.9))
    mark(op="dismiss_keyboard", label="Done", visible_press=True)
    hold(details(tmp, typed, True, press="done"), 0.1)
    s_filled = details(tmp, typed, False)
    anim(lambda p: slide_up(s_filled, details(tmp, typed, True), 1 - p, H - 520) if p < 1 else s_filled, 0.25)
    hold(s_filled, think(1.6))
    mark(op="tap", label="Save draft", screen="Details", visible_press=True)
    hold(details(tmp, typed, False, press="save"), 0.13)
    anim(lambda p: push(s_filled, s_drafts, p), 0.35)
    hold(s_drafts, 2.0)
    video = out_dir / "demo.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(video)],
                   input=b"".join(frames), check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"video": video, "narration": NARRATION, "caption": CAPTION, "truth": truth, "duration_s": len(frames) / FPS}


def clip(path: Path, seconds: int) -> Path:
    """A finished customer video stand-in (FIXTURE): `seconds` long, tiny, for a new job's media."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x284:rate=10:duration={seconds}",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


if __name__ == "__main__":
    import json
    import sys

    r = render(Path(sys.argv[1] if len(sys.argv) > 1 else "teach_demo"))
    print(json.dumps({**r, "video": str(r["video"])}, indent=2))
