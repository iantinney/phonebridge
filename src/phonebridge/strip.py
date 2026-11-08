"""A strip: the daemon's own live frames written to a video-only mp4 while a session runs, so an evidence recording
never opens a second capture of the phone. Two captures of one device fight: each start re-enumerates the phone on
USB and the other session loses it (R1-IG attempt 5: the director's own recorder held 20 s of a 42.5 s session).

Constant frame rate on wall time: frame n is at t0 + n/fps and shows the newest frame that had arrived by then, so the
strip lines up with call logs. A slot handled late (a slow source or lease check, a busy encoder) never shows pixels
that arrived after its time: it takes the frame of its time from the source's recent window, else repeats the previous
slot's frame and counts as late; slots a stop overtook are written as repeats too. When no new frame came for GAP_S (a capture reset, the phone parked) the last frame repeats
and the gap is journaled: a gap in the strip, not its end. ffmpeg encodes (VideoToolbox on the Mac, libx264
elsewhere) from a writer thread, so the event loop never blocks on the pipe."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable

from .stream import Frames

log = logging.getLogger(__name__)

GAP_S = 0.5  # no new frame for this long: the strip is in a gap (the last frame repeats meanwhile)
MAX_S = 900.0  # a strip stops by itself after this long
SOURCE_REFRESH_S = 0.5  # how often the frame source is re-asked (it keeps the capture alive; a restart brings new Frames)
LATE_UNRELIABLE = 0.05  # over this share of late slots the strip's timeline is labelled unreliable (summary "timeline")
# Live 11-08 (tsk_91d67ff507cb): full-size BGRA (1170x2532, ~11.9 MB/frame, ~178 MB/s at 15 fps) outran the pipe to ffmpeg;
# 817 of 915 slots were written late as repeats of a NEWER frame, so the video ran ahead of the journal. Frames go into the
# pipe at half size now (4x less), and a strip that still falls behind says so instead of passing for a timeline.
HALF = True


def _half(bgra: bytes, w: int, h: int) -> tuple[bytes, int, int]:
    """Every other row and column of a BGRA frame (numpy), else the frame as it is."""
    if not HALF:
        return bgra, w, h
    try:
        import numpy as np
    except ImportError:
        return bgra, w, h
    a = np.frombuffer(bgra, dtype=np.uint8).reshape(h, w, 4)[::2, ::2]
    return a.tobytes(), a.shape[1], a.shape[0]


def by(source, t: float):
    """The newest frame that had arrived by monotonic time `t` (the latest, or one in the source's recent window), or
    None: never a frame that arrived after `t`."""
    if source is None or (f := source.latest()) is None or f.t <= t:
        return None if source is None else f
    return next((x for x in reversed(source.recent()) if x.t <= t), None) if hasattr(source, "recent") else None


PAD_MAX_S = 10.0  # at the end, slots the loop hadn't reached are written as repeats up to this long (frame n at t0 + n/fps)


FFMPEG_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")  # launchd's PATH has none of Homebrew's


def tool(name: str) -> str | None:
    """An ffmpeg-family binary (ffmpeg, ffprobe): on PATH, else in Homebrew's or the system's bin (launchd's PATH misses
    /opt/homebrew/bin: record_start failed live on 11a8159 with ffmpeg installed)."""
    return shutil.which(name) or next((str(Path(d) / name) for d in FFMPEG_DIRS if os.access(Path(d) / name, os.X_OK)), None)


def ffmpeg() -> str | None:
    """ffmpeg's path: $PHONEBRIDGE_FFMPEG, else `tool`."""
    if (p := os.environ.get("PHONEBRIDGE_FFMPEG")) and os.access(p, os.X_OK):
        return p
    return tool("ffmpeg")


def encoder_args() -> list[str]:
    if sys.platform == "darwin":
        return ["-c:v", "h264_videotoolbox", "-b:v", "8M"]
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]


class Strip:
    def __init__(self, path: Path, fps: float, frames_of: Callable[[], Awaitable[Frames | None]], max_s: float = MAX_S,
                 alive: Callable[[], Awaitable[bool]] | None = None):
        if not 1 <= fps <= 60:
            raise ValueError("fps must be in [1, 60]")
        self.path, self.fps, self.frames_of, self.max_s, self.alive = Path(path), float(fps), frames_of, max_s, alive
        self.t0: float | None = None  # wall time of the strip's first frame
        self.frames = self.dropped = self.late = 0  # late: slots written as a repeat once the encoder had room
        self.late_spans: list[list[float]] = []  # [from_s, to_s] where slots fell behind (for the next diagnosis)
        self._owed, self._owed_bgra = 0, None  # slots the encoder couldn't take yet, and the (written-size) frame they repeat
        self._owed_of: tuple[int, float] | None = None  # that frame's (stream seq, arrival wall time)
        self.shows: list[tuple[int, float]] = []  # per written slot n: the (seq, arrival wall time) of the frame it shows
        self.gaps: list[list[float]] = []
        self.ended: str | None = None
        self._stop = asyncio.Event()
        self._q: queue.Queue = queue.Queue(maxsize=16)
        self._proc: subprocess.Popen | None = None
        self._writer: threading.Thread | None = None
        self._task: asyncio.Task | None = None
        self._size: tuple[int, int] | None = None  # as written (half)
        self._src: tuple[int, int] | None = None  # as the phone sends it
        self._last_bgra: bytes | None = None

    async def start(self) -> None:
        self._ffmpeg = ffmpeg()
        if not self._ffmpeg:
            raise RuntimeError(f"ffmpeg not found (PATH, {', '.join(FFMPEG_DIRS)}; or set PHONEBRIDGE_FFMPEG): no strip")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self._task = asyncio.ensure_future(self._run())

    def _open(self, w: int, h: int) -> None:
        self._size = (w, h)
        cmd = [self._ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}", "-r", f"{self.fps:g}",
               "-i", "-", "-an", *encoder_args(), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._writer = threading.Thread(target=self._write, name="strip-writer", daemon=True)
        self._writer.start()

    def _write(self) -> None:
        while (b := self._q.get()) is not None:
            try:
                self._proc.stdin.write(b)
            except (BrokenPipeError, ValueError):
                break
        with contextlib.suppress(Exception):
            self._proc.stdin.close()

    async def _run(self) -> None:
        source, asked, last, gap_from, last_seq, last_new = None, -1e9, None, None, None, time.monotonic()
        start = time.monotonic()
        n = 0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now - start >= self.max_s:
                    self.ended = "max_s"
                    break
                if now - asked >= SOURCE_REFRESH_S:
                    asked = now
                    source = await self.frames_of()  # None: no live source (parked, the eyes down)
                    if self.alive is not None and not await self.alive():
                        self.ended = "lease ended"
                        break
                # this slot's own time: its frame is the newest that had arrived by then, however late the loop handles it
                # (a slow source or lease await); before the first frame, now
                slot = start + n / self.fps if self._proc is not None else time.monotonic()
                f = by(source, slot)
                behind = f is None and source is not None and source.latest() is not None  # only later frames: a repeat
                if f is not None and f.seq != last_seq:
                    last_seq, last_new = f.seq, f.t  # the frame's own arrival (monotonic), not when we saw it
                if f is not None and slot - last_new < GAP_S and (f.width, f.height) == (self._src or (f.width, f.height)):
                    if gap_from is not None:
                        self.gaps.append([round(gap_from, 2), round(slot - start, 2)])
                        gap_from = None
                    if f is not last:
                        last = f
                        self._last_bgra = None  # the next write sizes it (half) once
                elif gap_from is None and last is not None and not behind:
                    gap_from = round(max(0.0, last_new - start), 2)
                if last is not None:
                    if self._last_bgra is None:
                        self._last_bgra, lw, lh = _half(last.bgra, last.width, last.height)
                    if self._proc is None:
                        self._src = (last.width, last.height)
                        self._open(lw, lh)
                        self.t0, start, n = time.time(), time.monotonic(), 0  # frame n of the strip is at t0 + n/fps
                    elif behind:
                        self._behind(round(slot - start, 2))
                    # the slots the encoder couldn't take earlier (each repeats the first one's frame, never a newer one),
                    # then this slot: a slot is never lost, it is written late, so frame n stays at t0 + n/fps (R1 probe
                    # 13:05Z: 59 slots dropped shortened the strip 2 s)
                    todo = [(self._owed_bgra, True, self._owed_of)] * self._owed + [(self._last_bgra, behind, (last.seq, last.wall))]
                    self._owed = 0
                    for i, (bgra, late, of) in enumerate(todo):
                        try:
                            self._q.put_nowait(bgra)
                        except queue.Full:
                            self._owed, self._owed_bgra, self._owed_of = len(todo) - i, bgra, of
                            self._behind(round(time.monotonic() - start, 2))
                            break
                        self.frames += 1
                        self.late += late
                        self.shows.append(of)
                n += 1
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), max(0.0, start + n / self.fps - time.monotonic()))
        finally:
            end = time.monotonic()
            if gap_from is not None:
                self.gaps.append([gap_from, round(end - start, 2)])
            pad = 0
            if self._proc is not None and self._last_bgra is not None:  # slots up to the end the loop hadn't reached (a
                pad = min(max(0, int((end - start) * self.fps) - n), int(PAD_MAX_S * self.fps))  # stop during an await)
                if pad:
                    if not self._owed:  # repeats of the last frame written; stop() writes them
                        self._owed_bgra, self._owed_of = self._last_bgra, (last.seq, last.wall)
                    self._owed += pad
                    self._behind(round(n / self.fps, 2))
                    self._behind(round((n + pad) / self.fps, 2))
            self.duration_s = round(min(end - start, (n + pad) / self.fps), 2) if self._proc is not None else round(end - start, 2)
            self.ended = self.ended or "stopped"

    def _behind(self, at: float) -> None:
        if self.late_spans and at - self.late_spans[-1][1] <= 0.5:
            self.late_spans[-1][1] = at
        else:
            self.late_spans.append([at, at])

    async def stop(self) -> dict:
        self._stop.set()
        if self._task:
            await self._task
        if self._owed and self._writer is not None:  # what the encoder still owes is written before the file closes
            for _ in range(self._owed):  # as the frame of the first slot it owes (never a newer one)
                await asyncio.to_thread(self._q.put, self._owed_bgra)
                self.frames += 1
                self.late += 1
                self.shows.append(self._owed_of)
            self._owed = 0
        self._q.put(None)
        if self._writer:
            await asyncio.to_thread(self._writer.join, 30)
        err = ""
        if self._proc:
            with contextlib.suppress(subprocess.TimeoutExpired):
                await asyncio.to_thread(self._proc.wait, 60)
            err = (self._proc.stderr.read() or b"").decode(errors="replace")[-300:] if self._proc.stderr else ""
        if err:
            log.warning("strip %s: ffmpeg: %s", self.path, err)
        if self.shows:  # the sidecar: which stream frame each slot shows and when it arrived (the strip's own evidence)
            self.path.with_suffix(".frames.json").write_text(json.dumps(
                {"t0": self.t0, "fps": self.fps, "slots": [[seq, round(wall, 3)] for seq, wall in self.shows]}))
        return self.summary()

    def summary(self) -> dict:
        late_share = self.late / self.frames if self.frames else 0.0
        return {"timeline": "reliable" if late_share <= LATE_UNRELIABLE else
                f"unreliable: {self.late} of {self.frames} slots written late as repeats (the encoder or the capture loop fell behind)",
                "path": str(self.path), "t0": self.t0, "started": getattr(self, "started", None), "fps": self.fps, "frames": self.frames, "dropped": self.dropped,
                "late": self.late, "late_spans": self.late_spans, "duration_s": getattr(self, "duration_s", None), "gaps": self.gaps, "ended": self.ended,
                "size": list(self._size) if self._size else None, "source_size": list(self._src) if self._src else None,
                "new_content_frames": sum(1 for a, b in zip([None] + self.shows, self.shows) if a is None or a[0] != b[0]),
                "sidecar": str(self.path.with_suffix(".frames.json")) if self.shows else None,
                # measured against an external clock only (phonebridge measure); arrival is the Mac's decode time
                "latency_s": None, "time_base": "arrival on the Mac (decode), not the phone's display"}
