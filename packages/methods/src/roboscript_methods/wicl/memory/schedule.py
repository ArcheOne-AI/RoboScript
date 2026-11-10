"""The learning-rate schedule of S2 (docs/concepts/learning.md):
`schedule(k, stats) -> (alpha_k, B_k)`, the merge step for numbers and the cache's edit budget at
the k-th counted seed.

Any callable of that shape is a schedule. The recipe's `schedule` section (`memory.yaml`, held
as written in the core's `MemoryRecipe.schedule`) picks a built-in one by `name`
(`recipe_schedule`; the discriminated union `ScheduleConfig` is their registry; an empty
section is `DecaySchedule()`): `decay`, the recipe's default,
`alpha_k = max(alpha_min, 1 / (1 + (k - 1) / tau))` and `B_k = max(budget_min,
round(budget_scale * alpha_k))` (half up), so both shrink with the seed index, or `constant`.
The recipe's values are in its `memory.yaml`.
"""

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    TypeAdapter,
    ValidationError,
)

from roboscript.memory.config import RECIPE_FILE


class ScheduleStats(BaseModel):
    """The counted seeds so far by outcome: first-try passes, merges, jitters (a try-2 success
    with an unchanged cache) and discarded caches (three failed tries)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: NonNegativeInt = 0
    merged: NonNegativeInt = 0
    jitter: NonNegativeInt = 0
    discarded: NonNegativeInt = 0


class Schedule(Protocol):
    def __call__(self, k: int, stats: ScheduleStats) -> tuple[float, int]:
        """`(alpha_k, B_k)` for the k-th counted seed (k >= 1)."""
        ...


AlphaFloat = Annotated[float, Field(gt=0, le=1)]


class DecaySchedule(BaseModel):
    """Fast at first, slower later, so that one or two odd seeds cannot drag the Memory away."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["decay"] = "decay"
    alpha_min: AlphaFloat = 0.2
    tau: PositiveFloat = 10.0
    budget_scale: PositiveFloat = 4.0
    budget_min: NonNegativeInt = 1

    def __call__(self, k: int, stats: ScheduleStats) -> tuple[float, int]:
        _check(k)
        # tau / (tau + k - 1) is 1 / (1 + (k - 1) / tau) with fewer roundings: at tau 3, k = 6,
        # exactly 0.375, so that B_6 = round(1.5) is 2, as in exact arithmetic.
        alpha = max(self.alpha_min, self.tau / (self.tau + k - 1))
        return alpha, max(self.budget_min, math.floor(self.budget_scale * alpha + 0.5))


class ConstantSchedule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["constant"] = "constant"
    alpha: AlphaFloat
    budget: NonNegativeInt

    def __call__(self, k: int, stats: ScheduleStats) -> tuple[float, int]:
        _check(k)
        return self.alpha, self.budget


ScheduleConfig = Annotated[DecaySchedule | ConstantSchedule, Field(discriminator="name")]
_CONFIG: TypeAdapter[DecaySchedule | ConstantSchedule] = TypeAdapter(ScheduleConfig)


def recipe_schedule(section: Mapping[str, Any]) -> DecaySchedule | ConstantSchedule:
    """The schedule a recipe's `schedule` section names (`MemoryRecipe.schedule`; empty: the
    default `DecaySchedule()`)."""
    return DecaySchedule() if not section else _CONFIG.validate_python(dict(section))


def checked_schedule(section: Mapping[str, Any]) -> DecaySchedule | ConstantSchedule:
    """`recipe_schedule`, refusing a section that names no schedule with the recipe file it
    comes from: S2 reads it as it starts, before any episode (P3 never does)."""
    try:
        return recipe_schedule(section)
    except ValidationError as error:
        raise ValueError(f"the recipe's {RECIPE_FILE}, its schedule: {error}") from error


def _check(k: int) -> None:
    if k < 1:
        raise ValueError(f"k counts seeds from 1, got {k}")
