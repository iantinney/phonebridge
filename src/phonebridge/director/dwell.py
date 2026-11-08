"""Dwell per post by smoothed bootstrap (BEH-03): a value from the persona's pool for the content class, jittered in
log space by a Silverman bandwidth, so the quantiles follow his and no draw is a verbatim replay. "Latency is the
glance" (PER-04): the dwell is drawn when the class arrives and the time already spent is subtracted. A late or
missing cue uses the content-blind pool: every class together."""

from __future__ import annotations

import math
import random
import statistics

BLIND = "blind"
FLOOR_S = 0.3


def bandwidth(logs: list[float]) -> float:
    """Silverman's rule in log space: 0.9 · min(sd, IQR/1.34) · n^-1/5 (0 for a constant pool)."""
    n = len(logs)
    if n < 2:
        return 0.0
    sd = statistics.pstdev(logs)
    q = statistics.quantiles(logs, n=4) if n >= 4 else None
    iqr = (q[2] - q[0]) / 1.34 if q else 0.0
    a = min(sd, iqr) if iqr > 0 else sd
    return 0.9 * a * n ** -0.2


class DwellModel:
    def __init__(self, pools: dict[str, list[float]], rng: random.Random | None = None, floor: float = FLOOR_S):
        self.rng, self.floor = rng or random.Random(), floor
        self.logs = {k: [math.log(max(1e-3, v)) for v in vs] for k, vs in pools.items() if vs}
        if not self.logs:
            raise ValueError("a dwell model needs at least one non-empty pool")
        self.logs[BLIND] = [x for k, xs in self.logs.items() if k != BLIND for x in xs]
        self.h = {k: bandwidth(v) for k, v in self.logs.items()}

    def pool(self, cls: str) -> str:
        return cls if cls in self.logs and cls != BLIND else BLIND

    def draw(self, cls: str) -> float:
        k = self.pool(cls)
        x = self.rng.choice(self.logs[k]) + self.rng.gauss(0, self.h[k])
        return max(self.floor, math.exp(x))

    BANDS = {0: (0.0, 0.25), 1: (0.0, 0.5), 2: (0.3, 0.85), 3: (0.6, 1.0)}  # quantile band of his pool per interest (policy (c))

    def draw_for(self, cls: str, interest: int) -> float:
        """A dwell from his pool for this class, conditioned on the interest score: low interest draws from the short
        end of his own distribution, high interest from the long end (still his values, still smoothed)."""
        k = self.pool(cls)
        xs = sorted(self.logs[k])
        lo, hi = self.BANDS.get(max(0, min(3, interest)), (0.0, 1.0))
        a, b = int(lo * (len(xs) - 1)), max(int(lo * (len(xs) - 1)) + 1, int(hi * (len(xs) - 1)) + 1)
        x = self.rng.choice(xs[a:b]) + self.rng.gauss(0, self.h[k])
        return max(self.floor, math.exp(x))

    def remaining(self, cls: str, elapsed: float, draw: float | None = None) -> float:
        """Seconds still to spend on this post once the class is known `elapsed` seconds after arrival."""
        d = self.draw(cls) if draw is None else draw
        return max(0.0, d - elapsed)
