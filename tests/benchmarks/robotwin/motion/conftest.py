"""Fixtures of the GPU tests here: the RoboTwin checkout and a warmed-up cuRobo planner."""

import asyncio
import time
from pathlib import Path

import pytest

from roboscript.motion import CuroboPlanner
from roboscript.motion.config import CuroboConfig
from roboscript.settings import load_config_ref, path_variable

EXAMPLE = (
    Path(__file__).parents[4] / "examples/simBenchmarks/RoboTwin/configs" / "motion.example.yaml"
)


@pytest.fixture(scope="module")
def robotwin_root() -> Path:
    root = path_variable("ROBOTWIN_ROOT")  # the RoboTwin package's
    if root is None:
        pytest.skip("ROBOTWIN_ROOT is not set")
    return root


@pytest.fixture(scope="module")
def planner(robotwin_root: Path) -> CuroboPlanner:
    """A fresh cuRobo planner per module (GPU), in the seed state its warm-up leaves. The example's
    profile finds its assets under `robotwin_root`, which skips without it."""
    planner = CuroboPlanner(load_config_ref(str(EXAMPLE), CuroboConfig))
    started = time.perf_counter()
    asyncio.run(planner.start())
    print(f"cuRobo planner built and warmed up in {time.perf_counter() - started:.1f} s")
    return planner
