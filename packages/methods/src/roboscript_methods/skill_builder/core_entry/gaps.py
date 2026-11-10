"""The rule the core entry check's per-call criteria are measured by: a quantity's
values from the runs, sorted; the threshold sits in the widest gap between consecutive values that
leaves at least `SHARE` of them on each side, at its middle; the gap is clear when it is at least
`DOMINANCE` times the next widest such gap. Distances and errors are compared on a log scale
(their spread grows with their size), values below a noise floor taken as the floor; signed or
bounded quantities (a rise, a gripper reading or command) on a linear one. A quantity with no
clear gap gets no measured threshold. numpy only: it runs in any interpreter."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

SHARE: Final = 0.05
DOMINANCE: Final = 1.5


@dataclass(frozen=True)
class Gap:
    """The widest gap: the values on its two sides, the threshold at its middle, how many values
    lie below and above it, and how many times wider it is than the next widest (inf: no other)."""

    low: float
    high: float
    threshold: float
    below: int
    above: int
    ratio: float

    @property
    def clear(self) -> bool:
        return self.ratio >= DOMINANCE

    def record(self) -> dict[str, object]:
        return {
            "gap": [round(self.low, 6), round(self.high, 6)],
            "threshold": round(self.threshold, 6),
            "below": self.below,
            "above": self.above,
            "ratio": None if math.isinf(self.ratio) else round(self.ratio, 3),
            "clear": self.clear,
        }


def widest_gap(values: Sequence[float], log: bool = False, floor: float = 0.0) -> Gap | None:
    """The widest gap of `values` that leaves `SHARE` of them on each side; None when no gap does
    (too few values, or all equal)."""
    raw = np.sort(np.asarray(values, dtype=float))
    n = len(raw)
    scaled = np.log10(np.maximum(raw, floor)) if log else raw
    least = math.ceil(SHARE * n)
    widths = np.diff(scaled)
    candidates = [
        (float(widths[i]), i)
        for i in range(n - 1)
        if i + 1 >= least and n - i - 1 >= least and widths[i] > 0
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    width, i = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else 0.0
    middle = (scaled[i] + scaled[i + 1]) / 2
    threshold = float(10**middle) if log else float(middle)
    ratio = width / second if second > 0 else math.inf
    return Gap(float(raw[i]), float(raw[i + 1]), threshold, i + 1, n - i - 1, ratio)
