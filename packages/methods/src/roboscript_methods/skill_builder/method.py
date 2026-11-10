"""`skill_builder`'s declaration (`roboscript.learning.builder`): a P1 builder whose frozen
artifact is a skillset, with the learn phases `derive` and `verify` and one role, `builder`.

The builder's work runs outside the Hub: a benchmark's glue (`examples/simBenchmarks/<Name>/p1/`)
drives `core_entry`, `demo_coverage` and `builder`, with the coding-agent sessions of
`skill_builder.session` and the freeze of `skill_builder.freeze`; the coding agent reaches no Hub
tool, so no phase shows the role a tool. It is a builder-type method: its benchmark's scripts
drive it, not the scheduler, so `learn`, `jobs` and `runtime` raise `NotImplementedError` naming
the script.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict

from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.jobs import Evaluation, JobKind, TaskJobs
from roboscript.contracts.phase import PhaseSpec, RoleSpec
from roboscript.contracts.skills import SkillSet
from roboscript.learning.builder import DERIVE, VERIFY
from roboscript.learning.context import TaskContext
from roboscript.learning.runtime import MethodRuntime

SCRIPT_DRIVEN: Final = (
    "skill_builder is a builder-type method driven by its benchmark's scripts, not by the "
    "scheduler: run the benchmark's build script (examples/simBenchmarks/<Name>/p1/build.py)"
)


class SkillBuilderConfig(BaseModel):
    """`skillset`: the name the frozen skillset takes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skillset: str


class SkillBuilderMethod:
    """P1: derive the skills the demonstrations need, verify them, freeze a `SkillSet`."""

    name: ClassVar[str] = "skill_builder"
    phases: ClassVar[list[PhaseSpec]] = [
        PhaseSpec(name=DERIVE, kind="learn", visibility={"builder": []}),
        PhaseSpec(name=VERIFY, kind="learn", visibility={"builder": []}),
    ]
    roles: ClassVar[list[RoleSpec]] = [
        RoleSpec(name="builder", prompt="builder/prompts/builder.md")
    ]
    requires: ClassVar[EnvCapabilities] = EnvCapabilities()
    artifact: ClassVar[type[SkillSet]] = SkillSet
    Config: ClassVar[type[BaseModel]] = SkillBuilderConfig
    kinds: ClassVar[Mapping[str, JobKind]] = {}

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        """The benchmark's build script drives the builder, not the scheduler."""
        raise NotImplementedError(SCRIPT_DRIVEN)

    def jobs(self, units: Sequence[str], evaluation: Evaluation, **options: Any) -> TaskJobs:
        """The benchmark's build script drives the builder, not the scheduler."""
        raise NotImplementedError(SCRIPT_DRIVEN)

    async def learn(self, ctx: TaskContext, config: Any) -> ArtifactRef:
        """The benchmark's build script drives the builder, not the scheduler, and freezes the
        skillset with `skill_builder.freeze.freeze_run`."""
        raise NotImplementedError(SCRIPT_DRIVEN)
