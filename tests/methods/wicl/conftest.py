"""Fixtures of the `wicl` tests: the core tests' own (`recipe`, `memory` from
`tests/core/conftest.py`; `setup`, MockEnv runners with scripted brains, from
`tests/core/learning/conftest.py`; `events`, `rig`, `model`, the toolset tests', from
`tests/core/toolsets/conftest.py`), loaded from those files so that both suites share one
definition."""

import dataclasses
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from roboscript.learning.runtime import MethodRuntime
from roboscript.testing.phases import TestMethod
from roboscript_methods.wicl.finish import CHECK_DONE, DoneGate
from roboscript_methods.wicl.goal import demo_goal
from roboscript_methods.wicl.method import Wicl

CORE = Path(__file__).resolve().parents[2] / "core"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load("core_conftest", CORE / "conftest.py")
_learning = _load("core_learning_conftest", CORE / "learning" / "conftest.py")
recipe, memory = _core.recipe, _core.memory
_toolsets = _load("core_toolsets_conftest", CORE / "toolsets" / "conftest.py")
events, rig, model = _toolsets.events, _toolsets.rig, _toolsets.model


class UngatedWithoutDone(DoneGate):
    """wicl's finish gate, but an episode on a stack without `check_done` is told nothing of
    the demonstration's end, so it runs ungated (most of these tests' stacks mount no done
    toolset; wicl's own gate refuses to open such an episode)."""

    def episode(self, episode: Any, *args: Any, **kwargs: Any) -> Any:
        if CHECK_DONE not in kwargs["mounted"]:
            known = dataclasses.replace(demo_goal(episode.goal), views=None, end=())
            episode = dataclasses.replace(episode, goal=known)
        return super().episode(episode, *args, **kwargs)


class WiclTest(TestMethod):
    """The core tests' phases (MockEnv's tools) with wicl's runtime: its open-time
    identification, and its finish gate where `check_done` is mounted (`UngatedWithoutDone`)."""

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        return dataclasses.replace(Wicl.runtime(recipe), finish=UngatedWithoutDone())


@pytest.fixture
def setup(tmp_path: Path, artifacts_dir: Path) -> Any:
    """The core tests' `setup`, its runners on `WiclTest`."""
    made = _learning.make_setup(tmp_path, artifacts_dir)
    made.method = WiclTest
    return made
