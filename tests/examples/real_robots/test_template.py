"""The real-robot template: its embodiment is the profile its configuration generates and passes
the embodiment suite, its skillset skeleton fits, and wicl is refused on a real robot with a
line for each gap."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from robot_template import (
    CONFIG,
    EMBODIMENT,
    ROBOT_SKILLS,
)

from roboscript.contracts.capabilities import missing
from roboscript.contracts.skills import Bindings
from roboscript.embodiments import load_embodiment
from roboscript.learning.method import load_method
from roboscript.testing.embodiments import EmbodimentConformance, EmbodimentTarget
from roboscript.testing.skills import SkillSetConformance, SkillSetTarget

TEMPLATE = Path(__file__).resolve().parents[3] / "examples" / "realRobots" / "_template"


@pytest.fixture
def embodiment_target() -> EmbodimentTarget:
    return EmbodimentTarget(EMBODIMENT)


@pytest.fixture
def skillset_target() -> SkillSetTarget:
    return SkillSetTarget(ROBOT_SKILLS.manifest, load_embodiment(EMBODIMENT), Bindings())


class TestTemplateEmbodiment(EmbodimentConformance):
    pass


class TestTemplateSkillSet(SkillSetConformance):
    pass


def _script(name: str) -> ModuleType:
    path = TEMPLATE / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"template_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wicl_is_refused_on_a_real_robot_with_a_line_for_each_gap() -> None:
    gaps = missing(load_method("wicl").requires, CONFIG.description().capabilities)
    assert gaps == ["success judged by latch", "reset by seed", "truth", "expert"]
