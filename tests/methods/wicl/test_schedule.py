"""The schedule: the recipe's decay schedule shrinks with the seed index, and a bad schedule
stops practice before any episode."""

import itertools
import shutil
from pathlib import Path
from typing import Any

import pytest

from roboscript.memory import MemoryRecipe, load_recipe
from roboscript.testing.protocols import LIFT_RED, acting, lift_memory
from roboscript_methods.wicl.memory.schedule import (
    DecaySchedule,
    ScheduleStats,
    recipe_schedule,
)
from roboscript_methods.wicl.method import Wicl
from roboscript_methods.wicl.practice import Practice

REPO = Path(__file__).resolve().parents[3]

STATS = ScheduleStats(passed=2, merged=1, jitter=1, discarded=0)


def test_the_decay_shrinks_with_the_seed_index() -> None:
    """The default spreads over practice's 60 seeds; tau 3 is a short one."""
    alphas, budgets = zip(*(DecaySchedule()(k, STATS) for k in range(1, 61)), strict=True)
    assert alphas[:3] == pytest.approx((1.0, 10 / 11, 10 / 12)) and alphas[10] == 0.5
    assert alphas[39] > 0.2 and alphas[40:] == (0.2,) * 20
    assert all(a >= b for a, b in itertools.pairwise(alphas))
    assert budgets == (4,) * 2 + (3,) * 5 + (2,) * 10 + (1,) * 43
    short = DecaySchedule(tau=3)
    alphas, budgets = zip(*(short(k, STATS) for k in range(1, 16)), strict=True)
    assert alphas[:4] == pytest.approx((1.0, 0.75, 0.6, 0.5)) and alphas[5] == 0.375
    assert alphas[12] == pytest.approx(0.2) and alphas[13:] == (0.2, 0.2)
    assert budgets == (4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1)


def test_the_recipe_uses_the_decay_schedule(recipe: MemoryRecipe) -> None:
    assert recipe_schedule(recipe.schedule) == DecaySchedule()
    assert recipe_schedule({}) == DecaySchedule()


async def test_a_bad_schedule_stops_practice_before_any_episode(setup: Any, tmp_path: Path) -> None:
    """The core holds the schedule section as written; S2 refuses a bad one as it starts, naming
    the file, while building wicl's runtime (as the P3 host does) reads no Memory recipe."""
    recipe = tmp_path / "recipe"
    shutil.copytree(REPO / "recipes" / "wicl", recipe)
    main = recipe / "memory.yaml"
    main.write_text(
        main.read_text(encoding="utf-8").replace("tau: 10", "tau: 10\n  speed: 2"), encoding="utf-8"
    )
    assert load_recipe(recipe).schedule["speed"] == 2
    prompts_only = tmp_path / "p3"
    shutil.copytree(recipe / "prompts", prompts_only / "prompts")
    Wicl.runtime(prompts_only)  # no memory.yaml, no memory_lint.yaml
    await setup.start(lift_memory(0.06))
    async with setup.runner({"actor": acting(lambda e: "pass")}, recipe=recipe) as runner:
        with pytest.raises(ValueError, match=r"memory\.yaml, its schedule"):
            await runner.run(Practice(), LIFT_RED, {"start": 0})
    assert not setup.env.opened
