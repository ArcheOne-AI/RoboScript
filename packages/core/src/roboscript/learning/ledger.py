"""What any protocol records of a try and how it scores a rate: `TryRecord` (one try: the Memory
it read, how it ended, its cost and its identification; `evaluate`'s and S2's rows hold them)
and `wilson`, the Wilson score interval of a success rate."""

import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from roboscript.learning.episode import EndClass, EpisodeOutcome


class TryRecord(BaseModel):
    """One try: the Memory it read (a version like `v003`, or the cache `c1` / `c2`), how it
    ended, its cost (tokens: the brain's and perception's) and how its objects were identified
    (`identified`, `reused:<try directory>`, or None: not at all); `directory` is relative to the
    task's directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int
    memory: str
    memory_hash: str | None
    success: bool
    end_class: EndClass
    end: str
    steps: NonNegativeInt
    turns: NonNegativeInt
    model_calls: NonNegativeInt
    tokens: NonNegativeInt
    wall_clock_s: float
    infra_reruns: NonNegativeInt
    directory: str
    identification: str | None = None

    @classmethod
    def of(
        cls, attempt: int, memory: str, outcome: EpisodeOutcome, reruns: int, root: Path
    ) -> "TryRecord":
        found = outcome.identification
        identification = None
        if found is not None and found.status == "reused":
            identification = f"reused:{found.source}"
        elif found is not None and found.status == "identified":
            identification = "identified"
        return cls(
            attempt=attempt,
            memory=memory,
            memory_hash=outcome.memory,
            success=outcome.success,
            end_class=outcome.end_class,
            end=outcome.end.reason,
            steps=outcome.steps,
            turns=outcome.turns,
            model_calls=outcome.model_calls,
            tokens=outcome.tokens,
            wall_clock_s=outcome.wall_clock_s,
            infra_reruns=reruns,
            directory=outcome.directory.relative_to(root).as_posix(),
            identification=identification,
        )


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The Wilson score interval of a success rate (95 % by default); (0, 1) without trials."""
    if n == 0:
        return 0.0, 1.0
    p, z2 = successes / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, center - half), min(1.0, center + half)
