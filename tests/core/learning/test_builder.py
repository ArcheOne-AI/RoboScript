"""P1 skill builders (docs/how-to/add-a-skillset.md): a method whose artifact is a skillset and
whose phases all learn loads as a builder; one freezing another kind, or with an eval phase, does
not."""

from typing import ClassVar

import pytest

from roboscript.contracts.artifact import Artifact
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.skills import SkillSet
from roboscript.kernel import KernelError
from roboscript.learning.builder import DERIVE, VERIFY, check_builder, load_builder
from roboscript.testing.phases import TestMethod

pytestmark = pytest.mark.unit


class Builder(TestMethod):
    __test__ = False
    name: ClassVar[str] = "builder"
    phases: ClassVar[list[PhaseSpec]] = [
        PhaseSpec(
            name=DERIVE, kind="learn", visibility={"actor": ["hub.finish"]}, tiers={"actor": "P"}
        ),
        PhaseSpec(name=VERIFY, kind="learn", visibility={"actor": ["hub.finish"]}),
    ]
    artifact: ClassVar[type[Artifact]] = SkillSet


class Freezing(Builder):
    __test__ = False
    artifact: ClassVar[type[Artifact]] = Artifact


class Evaluating(Builder):
    __test__ = False
    phases: ClassVar[list[PhaseSpec]] = [
        *Builder.phases,
        PhaseSpec(name="p3", kind="eval", visibility={"actor": ["hub.finish"]}),
    ]


def test_a_skillset_learning_method_is_a_builder() -> None:
    assert check_builder(Builder) == []
    assert load_builder(f"{__name__}:Builder") is Builder


def test_a_builder_freezes_a_skillset_and_only_learns() -> None:
    assert check_builder(Freezing) == [
        f"artifact {Artifact!r} is not a SkillSet: a builder freezes a skillset"
    ]
    assert check_builder(Evaluating) == [
        "phases ['p3'] are not learn phases: a builder only learns"
    ]
    with pytest.raises(KernelError, match=r"builder .* does not load: phases"):
        load_builder(f"{__name__}:Evaluating")
