"""The method framework (docs/how-to/add-a-method.md, docs/concepts/operations.md): method
declarations, the evaluation record view and seed splits."""

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ValidationError

from roboscript.contracts.artifact import Artifact, ArtifactRef
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.ids import TaskName
from roboscript.contracts.phase import PhaseSpec, RoleSpec
from roboscript.learning.eval_view import EvalEpisode, EvalRecordView, RecordPart
from roboscript.learning.runtime import MethodRuntime
from roboscript.learning.seeds import SeedRange, SeedSplit
from roboscript.testing.methods import MethodConformance
from roboscript.testing.phases import TestMethod


class _Notes(Artifact):
    kind: ClassVar[str] = "notes"
    text: str


class _Method:
    """A two-phase method in the shape of wicl's s2 and p3."""

    name: ClassVar[str] = "notes"
    Config: ClassVar[type[BaseModel]] = BaseModel
    phases: ClassVar[list[PhaseSpec]] = [
        PhaseSpec(
            name="practice",
            kind="learn",
            visibility={
                "actor": ["sensing.*", "manipulation.*", "hub.finish"],
                "reviser": ["memory.*", "hub.finish"],
            },
            tiers={"actor": "D", "reviser": "P"},
        ),
        PhaseSpec(
            name="eval",
            kind="eval",
            visibility={"actor": ["sensing.*", "manipulation.*", "hub.finish"]},
        ),
    ]
    roles: ClassVar[list[RoleSpec]] = [
        RoleSpec(name="actor", prompt="prompts/actor.md"),
        RoleSpec(name="reviser", prompt="prompts/reviser.md"),
    ]
    requires: ClassVar[EnvCapabilities] = EnvCapabilities(success={"latch"}, reset={"seed"})
    artifact: ClassVar[type[Artifact]] = _Notes

    @classmethod
    def runtime(cls, recipe: Any) -> MethodRuntime:
        return MethodRuntime()

    async def learn(self, ctx: Any, config: Any) -> ArtifactRef:
        raise NotImplementedError


class TestMethodConformance(MethodConformance):
    @pytest.fixture
    def method_target(self) -> type:
        return _Method


def test_an_eval_phase_with_privilege_fails_to_load() -> None:
    with pytest.raises(ValidationError, match="uses D only"):
        PhaseSpec(name="p3", kind="eval", visibility={"actor": []}, tiers={"actor": "P"})
    PhaseSpec(name="s1", kind="learn", visibility={"actor": []}, tiers={"actor": "P"})


class _Practice:
    stage = "s2"

    async def run(self, host: Any, request: Any, made: Any) -> Any:
        raise NotImplementedError


def _episode() -> EvalEpisode:
    return EvalEpisode(
        task=TaskName("lift_red"),
        seed=100000,
        success=False,
        parts=(
            RecordPart(kind="instruction", content="Lift the red block."),
            RecordPart(kind="tool_call", content={"name": "grasp_object"}, step=3),
            RecordPart(kind="diagnosis", content={"lift_red": "slipped"}),
        ),
    )


def test_the_eval_view_shows_d_parts_and_success_by_default() -> None:
    view = EvalRecordView.of([_episode()])
    assert view.tiers == {"D"}
    (episode,) = view.episodes
    assert [part.kind for part in episode.parts] == ["instruction", "tool_call"]
    assert episode.success is False
    privileged = EvalRecordView.of([_episode()], privileged=True)
    assert [part.kind for part in privileged.episodes[0].parts][-1] == "diagnosis"


def _official(seed: int) -> SeedRange:
    return SeedRange(start=100000 * (1 + seed), count=100)


def test_seed_blocks_must_not_overlap() -> None:
    SeedSplit(
        blocks={
            "demo": SeedRange(start=0, count=68),
            "s2": SeedRange(start=10000, count=60),
            "internal": SeedRange(start=20000, count=100),
            "p3": _official(0),
            "p4_eval": _official(1),
        },
        reflects={"p4_eval": "p3"},
    )
    with pytest.raises(ValidationError, match="'demo' and 's2' overlap"):
        SeedSplit(blocks={"demo": SeedRange(start=0, count=68), "s2": SeedRange(start=60, count=5)})


class TestTestMethodConformance(MethodConformance):
    @pytest.fixture
    def method_target(self) -> type:
        return TestMethod
