"""The method declaration `docs/how-to/add-a-method.md` walks through, with the conformance suite
run on it. If you change one, change the other."""

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, PositiveInt

from roboscript.contracts.artifact import Artifact, ArtifactRef
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.phase import PhaseSpec, RoleSpec
from roboscript.learning.method import check_method, load_method
from roboscript.learning.runtime import MethodRuntime
from roboscript.testing.methods import MethodConformance

pytestmark = pytest.mark.unit


class Notes(Artifact):
    """What the method freezes and the evaluation then reads."""

    kind: ClassVar[str] = "notes"
    text: str = ""


class PracticeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tries: PositiveInt = 3


class Practice:
    name: ClassVar[str] = "practice_only"
    Config: ClassVar[type[BaseModel]] = PracticeConfig
    phases: ClassVar[list[PhaseSpec]] = [
        PhaseSpec(
            name="practice",
            kind="learn",
            visibility={
                "actor": ["sensing.*", "manipulation.+", "hub.finish"],
                "drafter": ["memory.*", "hub.finish"],
            },
            tiers={"actor": "D", "drafter": "P"},
        ),
        PhaseSpec(
            name="eval",
            kind="eval",  # tier D only, whatever the roles say
            visibility={"actor": ["sensing.*", "manipulation.+", "hub.finish"]},
        ),
    ]
    roles: ClassVar[list[RoleSpec]] = [
        RoleSpec(name="actor", prompt="prompts/actor.md"),
        RoleSpec(name="drafter", prompt="prompts/drafter.md"),
    ]
    requires: ClassVar[EnvCapabilities] = EnvCapabilities(success={"latch"}, reset={"seed"})
    artifact: ClassVar[type[Artifact]] = Notes

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        return MethodRuntime(evaluate_phase="practice", host_phase="eval")

    async def learn(self, ctx: object, config: object) -> ArtifactRef:
        raise NotImplementedError("the protocols run the phases; see the how-to's section 5")


class TestPractice(MethodConformance):
    @pytest.fixture
    def method_target(self) -> type:
        return Practice


def test_the_declaration_loads_by_reference() -> None:
    assert check_method(Practice) == []
    assert load_method(f"{__name__}:Practice") is Practice
