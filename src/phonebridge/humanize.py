"""Human-like input: every gesture phonebridge sends is drawn here, from one seedable config.

Backends are deterministic players. The Bridge turns tap(x, y) into a point inside the target and a
press duration, swipe() into a curved path with an accelerate/decelerate velocity profile and a natural
end point, type() into keystrokes with a variable cadence (and the odd corrected slip), scroll() into a
flick of varying length, and waits a think time between actions. Units are screenshot pixels and seconds.
Defaults come from published touch/typing studies and timing recorded on the iPhone 12 (@3x, 1170x2532);
see docs/research/human-input-timing.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, fields

Point = tuple[float, float]
PathPoint = tuple[float, float, float]  # x, y, seconds since touch-down
Key = tuple[str, float, float]  # character ("\b" = Backspace, "\n" = Return), hold s, pause before it s

QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]  # each row sits half a key right of the one above
NEIGHBOURS: dict[str, str] = {}
for _r, _row in enumerate(QWERTY):
    for _c, _ch in enumerate(_row):
        _near = set(_row[max(0, _c - 1):_c + 2]) - {_ch}
        if _r:
            _near |= set(QWERTY[_r - 1][_c:_c + 2])
        if _r + 1 < len(QWERTY):
            _near |= set(QWERTY[_r + 1][max(0, _c - 1):_c + 1])
        NEIGHBOURS[_ch] = "".join(sorted(_near))


@dataclass(frozen=True)
class Dist:
    """A clipped log-normal: `median`, log-space `sigma`, and the [lo, hi] it is clipped to. In TOML:
    `press = [85, 0.25, 45, 200]` (ms) — all times in the config are ms, draws are seconds."""
    median: float
    sigma: float
    lo: float
    hi: float

    def draw(self, rng: random.Random) -> float:
        return min(self.hi, max(self.lo, self.median * math.exp(rng.gauss(0, self.sigma))))


def _dist(*v) -> Dist:
    return field(default=Dist(*v))


@dataclass
class HumanConfig:
    seed: int | None = None  # fixed seed: reproducible gestures (tests); None: a fresh one per process
    # --- tap: where inside the target, how long the finger stays down ----------------------------
    # Finger precision is about 1.5 mm per axis on targets of 5 mm and up (FFitts law, Bi/Li/Zhai 2013;
    # Henze 2011/2012): 27 px at the iPhone 12's 18 px/mm. Taps land a little low (Henze 2012: 0.7–1.5 mm).
    tap_target: float = 132.0  # assumed target size in px when the caller gives none (44 pt at @3x)
    tap_sigma: float = 27.0  # px, per axis, before truncation to the target
    tap_clip: float = 0.4  # never land further from the centre than this fraction of the target size
    tap_offset: tuple[float, float] = (0.0, 0.12)  # systematic bias as a fraction of target size, capped at tap_sigma
    press: Dist = _dist(85, 0.30, 40, 250)  # ms down (BB-MAS 81–103 ms key holds; SapiPin 77 ms)
    # --- swipe: a bowed path, an asymmetric velocity profile, released while still moving ---------
    # Touchalytics (Frank 2013, 21k strokes): max lateral deviation ~5% of the length (log-sd 0.55), peak
    # speed at 0.23–0.35 of the duration, release speed ~0.3x the mean; a dead-straight swipe is a bot tell.
    swipe_bow: Dist = _dist(0.067, 0.55, 0.005, 0.3)  # control-point offset / length (max deviation ~0.75x)
    swipe_jitter: float = 8.0  # sigma of the start point, px
    swipe_end_scale: float = 0.06  # sigma of the length multiplier (under/overshoot)
    swipe_drift: float = 0.12  # sigma of the lateral end drift, fraction of length (~7° of tilt)
    swipe_time: Dist = _dist(1.0, 0.15, 0.6, 1.6)  # multiplier on the requested duration
    swipe_profile: tuple[float, float, float] = (1.6, 3.0, 0.8)  # speed ~ Beta(a, b) in time, cut at `c`
    swipe_hz: float = 60.0  # samples per second along the path
    swipe_tremor: float = 0.7  # px of noise per sample
    # --- scroll flick (scroll()): length as a fraction of screen height, duration in ms -----------
    # Flicks: ~190 ms (log-sd 0.45) in Touchalytics; lengths 18 mm (Mayer 2019) on small phones, longer
    # on a tall screen — the length default is a judgement call (low confidence), see the report.
    flick_len: Dist = _dist(0.22, 0.5, 0.08, 0.55)
    flick_time: Dist = _dist(190, 0.45, 80, 600)
    flick_x: tuple[float, float] = (0.3, 0.7)  # where across the screen a flick starts
    flick_y: tuple[float, float] = (0.55, 0.8)  # where down the screen an upward flick starts
    # --- typing ---------------------------------------------------------------------------------
    # Phone typing ~36 wpm (Palin 2019); letter-to-letter 200–270 ms, space-to-letter ~2.2x (BB-MAS);
    # heavy-tailed (Buschek 2018: mean 369, sd 425 ms); backspace 7–20% of keys (Buschek, Komninos 2018).
    key_gap: Dist = _dist(150, 0.45, 40, 1500)  # ms from one key's release to the next press (+ hold = IKI ~230)
    key_hold: Dist = _dist(85, 0.25, 40, 200)  # ms a key stays down
    backspace_hold: Dist = _dist(128, 0.25, 60, 250)
    word_pause: float = 2.2  # gap multiplier for the first key of a word (after space/punctuation)
    pause_rate: float = 0.03  # chance per key of a long hesitation (the heavy tail)
    pause: Dist = _dist(1200, 0.5, 500, 4000)
    typo_rate: float = 0.03  # chance per letter of hitting a neighbouring key, then noticing and fixing it
    typo_notice: Dist = _dist(600, 0.5, 200, 2000)  # ms before the Backspace (Komninos: 0.85 s, sd 0.6)
    # --- think time: the minimum gap before an input since the device's last input ---------------
    # Tap-to-tap gaps median ~1 s with a long tail (Pfister & Ghosh; Touchalytics inter-stroke gaps). An
    # agent's own latency counts toward it, so these mostly bite on back-to-back calls.
    think: dict[str, Dist] = field(default_factory=lambda: {
        "tap": Dist(1000, 0.8, 150, 8000), "swipe": Dist(700, 0.8, 100, 8000), "type": Dist(1200, 0.7, 200, 8000),
        "paste": Dist(1200, 0.7, 200, 8000), "home": Dist(900, 0.7, 200, 8000), "launch": Dist(900, 0.7, 200, 8000),
        "open_url": Dist(900, 0.7, 200, 8000)})

    @classmethod
    def from_toml(cls, raw: dict) -> HumanConfig:
        """`[humanize]` table: scalars as-is, distributions as [median, sigma, lo, hi], think as a table."""
        kw = {}
        for f in fields(cls):
            if f.name not in raw:
                continue
            v = raw[f.name]
            if f.name == "think":
                kw[f.name] = {**cls().think, **{k: Dist(*d) for k, d in v.items()}}
            elif isinstance(f.default, Dist):
                kw[f.name] = Dist(*v)
            elif isinstance(v, list):
                kw[f.name] = tuple(v)
            else:
                kw[f.name] = v
        return cls(**kw)


def profile(n: int, a: float, b: float, cut: float) -> list[float]:
    """Positions 0..1 at n+1 evenly spaced times for a speed that follows Beta(a, b) over the first `cut`
    of a stroke: peak early (a < b), still moving when the finger lifts (cut < 1)."""
    v = [((i + 0.5) / n * cut) ** (a - 1) * (1 - (i + 0.5) / n * cut) ** (b - 1) for i in range(n)]
    total, acc, out = sum(v), 0.0, [0.0]
    for x in v:
        acc += x
        out.append(acc / total)
    return out


def _cut_for_release(a: float, b: float, release: float) -> float:
    """Where to cut a Beta(a, b) speed profile so the speed at the lift is `release` × the mean speed up to the cut.
    Monotone past the peak: a quick bisection on [peak, 0.999]."""
    peak = (a - 1) / (a + b - 2) if a + b > 2 else 0.5

    def ratio(cut: float) -> float:
        n = 200
        v = [((i + 0.5) / n * cut) ** (a - 1) * (1 - (i + 0.5) / n * cut) ** (b - 1) for i in range(n)]
        mean = sum(v) / n
        return v[-1] / mean if mean else 0.0
    lo, hi = max(peak, 0.05), 0.999
    for _ in range(30):
        mid = (lo + hi) / 2
        if ratio(mid) > release:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _bezier(p0: Point, p1: Point, p2: Point, p3: Point, s: float) -> Point:
    u = 1 - s
    return (u ** 3 * p0[0] + 3 * u * u * s * p1[0] + 3 * u * s * s * p2[0] + s ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * s * p1[1] + 3 * u * s * s * p2[1] + s ** 3 * p3[1])


class Humanizer:
    def __init__(self, config: HumanConfig | None = None, seed: int | None = None):
        self.c = config or HumanConfig()
        self.rng = random.Random(self.c.seed if seed is None else seed)
        self.side = self.rng.choice((-1, 1))  # a person's swipes bow to the same side (their thumb's arc)

    def _ms(self, d: Dist) -> float:
        return d.draw(self.rng) / 1000

    # --- tap ------------------------------------------------------------------------------------
    def tap(self, x: float, y: float, w: float = 0, h: float = 0) -> tuple[float, float, float]:
        """(x, y, press seconds): a point inside the target of size w x h (default tap_target) around
        (x, y), from a truncated normal with a slight systematic offset."""
        c = self.c
        w, h = w or c.tap_target, h or c.tap_target

        def off(size, bias):
            mean = max(-c.tap_sigma, min(c.tap_sigma, bias * size))
            while True:
                d = self.rng.gauss(mean, c.tap_sigma)
                if abs(d) <= c.tap_clip * size:
                    return d
        return x + off(w, c.tap_offset[0]), y + off(h, c.tap_offset[1]), self._ms(c.press)

    # --- swipe ----------------------------------------------------------------------------------
    def glide_path(self, x1: float, y1: float, x2: float, y2: float, duration: float, release: float | None = None,
                   exact: bool = True) -> list[PathPoint]:
        """THE path generator (BEHAVIOUR_MODEL: one generator, two callers). From near (x1, y1) to (x2, y2) in `duration`
        seconds: a bow to this person's side, start jitter, tremor, and a velocity profile that peaks early and lifts while
        still moving at `release` × the mean speed. With `exact`, the length, the duration and the release are kept as
        given (the director plans them and measures the travel); `swipe()` draws its own variations before calling this."""
        c, g = self.c, self.rng
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        p0 = (x1 + g.gauss(0, c.swipe_jitter), y1 + g.gauss(0, c.swipe_jitter))
        if length < 1:
            p3 = (p0[0] + g.gauss(0, 1), p0[1] + g.gauss(0, 1))
            ux, uy = 0.0, 0.0
        else:
            ux, uy = dx / length, dy / length
            drift = g.gauss(0, c.swipe_drift * length) if not exact else g.gauss(0, 0.25 * c.swipe_drift * length)
            p3 = (p0[0] + dx - uy * drift, p0[1] + dy + ux * drift)  # exact length along the axis; drift is texture across it
        bow = self.side * c.swipe_bow.draw(g) * length
        bow2 = bow * g.uniform(0.4, 1.0) + g.gauss(0, 0.1 * abs(bow))
        p1 = (p0[0] + (p3[0] - p0[0]) / 3 - uy * bow, p0[1] + (p3[1] - p0[1]) / 3 + ux * bow)
        p2 = (p0[0] + 2 * (p3[0] - p0[0]) / 3 - uy * bow2, p0[1] + 2 * (p3[1] - p0[1]) / 3 + ux * bow2)
        n = max(4, round(duration * c.swipe_hz))
        a, b, cut = c.swipe_profile
        if release is not None:  # Beta(a, b) cut at `cut`: the speed at the cut over the mean speed sets the release
            cut = _cut_for_release(a, b, release)
        path: list[PathPoint] = []
        for i, s in enumerate(profile(n, a, b, cut)):
            t = i / n
            x, y = _bezier(p0, p1, p2, p3, s)
            if 0 < i < n:
                x, y = x + g.gauss(0, c.swipe_tremor), y + g.gauss(0, c.swipe_tremor)
            path.append((x, y, t * duration))
        return path

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration: float) -> list[PathPoint]:
        """A requested swipe as a person would do it: the length and the duration vary a little, the lateral drift is
        free, the release is the profile's; then the one path generator."""
        c, g = self.c, self.rng
        dx, dy = x2 - x1, y2 - y1
        scale = g.gauss(1, c.swipe_end_scale)
        return self.glide_path(x1, y1, x1 + dx * scale, y1 + dy * scale, duration * c.swipe_time.draw(g), release=None, exact=False)

    def flick(self, w: float, h: float, direction: str = "up") -> tuple[float, float, float, float, float]:
        """A scroll flick (x1, y1, x2, y2, duration) on a w x h screen: varying length and speed, started
        from the thumb's comfortable band. direction: up (content moves up = scroll down), down, left, right."""
        c, g = self.c, self.rng
        length = c.flick_len.draw(g) * (h if direction in ("up", "down") else w)
        duration = c.flick_time.draw(g) / 1000
        x = g.uniform(*c.flick_x) * w
        y = g.uniform(*c.flick_y) * h
        if direction == "up":
            return x, y, x, max(0.02 * h, y - length), duration
        if direction == "down":
            y = h - y
            return x, y, x, min(0.98 * h, y + length), duration
        x = g.uniform(*c.flick_y) * w if direction == "left" else (1 - g.uniform(*c.flick_y)) * w
        y = g.uniform(*c.flick_x) * h + 0.1 * h
        if direction == "left":
            return x, y, max(0.02 * w, x - length), y, duration
        return x, y, min(0.98 * w, x + length), y, duration

    # --- named gestures (INP-05) -----------------------------------------------------------------
    def double_tap(self, x: float, y: float, w: float = 0, h: float = 0) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        """Two quick taps (each (x, y, press), then the gap between them): the second lands near the first, both
        presses short (double-taps are ~70 ms holds, 100–300 ms apart)."""
        g = self.rng
        x1, y1, _ = self.tap(x, y, w, h)
        x2, y2 = x1 + g.gauss(0, 7), y1 + g.gauss(0, 7)
        quick = Dist(70, 0.25, 40, 180)
        return (x1, y1, self._ms(quick)), (x2, y2, self._ms(quick)), self._ms(Dist(160, 0.3, 80, 350))

    def long_press(self, x: float, y: float, w: float = 0, h: float = 0) -> tuple[float, float, float]:
        """A press-and-hold: the tap's point with a 0.5–1.5 s hold."""
        x1, y1, _ = self.tap(x, y, w, h)
        return x1, y1, self._ms(Dist(800, 0.25, 500, 1500))

    def edge_back(self, w: float, h: float) -> tuple[float, float, float, float, float]:
        """The iOS back gesture: from the left edge, a third to two thirds of the way across, roughly level."""
        g = self.rng
        x1, y1 = g.uniform(0.002, 0.02) * w, g.uniform(0.3, 0.7) * h
        return x1, y1, x1 + g.uniform(0.3, 0.6) * w, y1 + g.gauss(0, 0.02 * h), self._ms(Dist(300, 0.3, 150, 600))

    def drag_to_value(self, x1: float, y1: float, x2: float, y2: float, duration: float = 0.8) -> list[PathPoint]:
        """A slider/scrubber drag: eases in and out, lands within a few px, then holds still before the lift."""
        g, c = self.rng, self.c
        n = max(6, round(duration * c.swipe_hz))
        p0 = (x1 + g.gauss(0, 3), y1 + g.gauss(0, 3))
        p3 = (x2 + g.gauss(0, 4), y2 + g.gauss(0, 4))
        bow = self.side * c.swipe_bow.draw(g) * 0.3 * math.hypot(x2 - x1, y2 - y1)
        dx, dy = p3[0] - p0[0], p3[1] - p0[1]
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        p1 = (p0[0] + dx / 3 - uy * bow, p0[1] + dy / 3 + ux * bow)
        p2 = (p0[0] + 2 * dx / 3 - uy * bow, p0[1] + 2 * dy / 3 + ux * bow)
        path: list[PathPoint] = []
        for i, s in enumerate(profile(n, 2.0, 2.0, 1.0)):  # symmetric: slow, faster, slow
            x, y = _bezier(p0, p1, p2, p3, s)
            if 0 < i < n:
                x, y = x + g.gauss(0, 0.5), y + g.gauss(0, 0.5)
            path.append((x, y, i / n * duration))
        hold = self._ms(Dist(180, 0.3, 100, 400))
        for k in range(1, max(2, round(hold * c.swipe_hz)) + 1):
            path.append((p3[0] + g.gauss(0, 0.6), p3[1] + g.gauss(0, 0.6), duration + k / c.swipe_hz))
        return path

    SAFE_Y = (0.15, 0.88)  # a flick's touch-down and lift stay off the top tabs and the tab bar (CONTROLLER_BAR S2)

    def flick_from(self, w: float, h: float, row, direction: str = "up") -> tuple[float, float, float, float, float]:
        """A feed flick (x1, y1, x2, y2, duration) rendered from a persona row (length as a fraction of the screen
        height, duration s, onset s): the row's duration exactly and its length unless the stroke would touch the top
        tabs or the tab bar, in which case the flick is shortened, never moved onto a control (BEH-03: the pools, not
        flick_len/flick_time, set the sizes; S2: x stays in the thumb band, off the right action rail)."""
        g, c = self.rng, self.c
        length_frac, duration = float(row[0]), float(row[1])
        lo, hi = self.SAFE_Y[0] * h, self.SAFE_Y[1] * h
        length = min(length_frac * h, hi - lo - 0.02 * h)
        x = g.uniform(*c.flick_x) * w
        if direction == "up":
            y = min(hi, max(g.uniform(*c.flick_y) * h, lo + length + 0.01 * h))
            return x, y, x, y - length, duration
        y = max(lo, min(h - g.uniform(*c.flick_y) * h, hi - length - 0.01 * h))
        return x, y, x, y + length, duration

    # --- typing ---------------------------------------------------------------------------------
    def keystrokes(self, text: str) -> list[Key]:
        """Keys to press for `text`: (char, hold s, pause before s). The pause is log-normal, longer
        at the start of a word; a letter is sometimes mistyped as a neighbour, noticed, backspaced
        and retyped. Applying the keys (with "\\b" deleting) reproduces `text` exactly."""
        c, g = self.c, self.rng
        keys: list[Key] = []
        prev = " "
        for ch in text:
            gap = self._ms(c.key_gap)
            if prev in " .,!?\n":
                gap *= c.word_pause
            if g.random() < c.pause_rate:
                gap += self._ms(c.pause)
            if keys == []:
                gap = 0.0  # think time already covered the pause before the first key
            if (near := NEIGHBOURS.get(ch.lower())) and g.random() < c.typo_rate:
                wrong = g.choice(near)
                keys.append((wrong.upper() if ch.isupper() else wrong, self._ms(c.key_hold), gap))
                keys.append(("\b", self._ms(c.backspace_hold), self._ms(c.typo_notice)))
                gap = self._ms(c.key_gap)
            keys.append((ch, self._ms(c.key_hold), gap))
            prev = ch
        return keys

    # --- think time -----------------------------------------------------------------------------
    def think(self, op: str) -> float:
        """Seconds a person would pause before this kind of action (the Bridge subtracts time already spent)."""
        d = self.c.think.get(op)
        return self._ms(d) if d else 0.0


def typed(keys: list[Key]) -> str:
    """The text a key plan produces (Backspace deletes)."""
    out: list[str] = []
    for ch, _, _ in keys:
        if ch == "\b":
            if out:
                out.pop()
        else:
            out.append(ch)
    return "".join(out)
