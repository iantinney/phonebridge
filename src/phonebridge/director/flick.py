"""Leaving flicks by joint smoothed bootstrap (BEH-03): a whole row (length, duration, onset) from the persona's
measured flicks, jittered per component in log space, clipped to a physical envelope. Joint rows keep his
length↔speed correlation; the jitter means no draw equals a measured stroke (nearest-neighbour distance > 0)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .dwell import bandwidth

ENVELOPE = ((0.05, 0.95), (0.05, 1.0), (0.0, 6.0))  # length frac, duration s, onset s
Row = tuple[float, float, float]


@dataclass(frozen=True)
class Flick:
    length_frac: float
    duration_s: float
    onset_s: float

    @property
    def row(self) -> Row:
        return (self.length_frac, self.duration_s, self.onset_s)


def distance(a: Row, b: Row) -> float:
    return math.sqrt(sum((math.log(max(1e-3, x)) - math.log(max(1e-3, y))) ** 2 for x, y in zip(a, b)))


class FlickModel:
    def __init__(self, rows: list, rng: random.Random | None = None):
        if not rows:
            raise ValueError("a flick model needs rows")
        self.rows = [tuple(float(v) for v in r) for r in rows]
        self.rng = rng or random.Random()
        cols = list(zip(*self.rows))
        self.h = [bandwidth([math.log(max(1e-3, v)) for v in c]) for c in cols]

    def draw(self) -> Flick:
        row = self.rng.choice(self.rows)
        out = []
        for v, h, (lo, hi) in zip(row, self.h, ENVELOPE):
            x = math.exp(math.log(max(1e-3, v)) + self.rng.gauss(0, h or 0.02))  # a constant pool still varies a little
            out.append(min(hi, max(lo, x)))
        return Flick(*out)
