"""The method interface: a method declares its phases, roles, needed env capabilities and the
artifact it freezes, then learns it (docs/how-to/add-a-method.md).

The declarations (`contracts.phase`: `PhaseSpec`, `RoleSpec`) configure the Hub: which roles of
which phase see which tools at which tier, under which default rules. The phase names are the
method's own (the core names none): a table names the tools skillsets add by category
(`toolset.+`), never a benchmark's skills; a learn phase may be held to tier D as an eval phase is
(`PhaseSpec.d_only`); and the method's runtime (`learning.runtime.MethodRuntime`) names the phase
the core's `evaluate` protocol runs under (`evaluate_phase`) and the one the Agent host opens its
episodes in (`host_phase`). Methods register in the
entry-point group `roboscript.methods`; a config names one by that name or as
`package.module:Class` (`load_method`). The runner does not call `learn`: the protocols in
`roboscript.protocols` run each phase. It does not read `RoleSpec.brain` either:
`learning.config.BrainSpec` picks brains. `jobs` gives a launch's job graph (`contracts.jobs`, run
by `roboscript.orchestrate`): the method's jobs of each task, with the benchmark's evaluation of
each unit it learns.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol

from pydantic import BaseModel

from roboscript.contracts.artifact import Artifact, ArtifactRef
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.jobs import Evaluation, JobKind, TaskJobs
from roboscript.contracts.phase import PhaseSpec, RoleSpec
from roboscript.kernel import KernelError, entrypoints
from roboscript.learning.context import TaskContext
from roboscript.learning.runtime import MethodRuntime

GROUP: Final = "roboscript.methods"


class Declaration(Protocol):
    """What a method declares: what the core needs to run its episodes (the Hub's phases, the
    roles' bounds and brains) before anything of the method runs."""

    name: ClassVar[str]
    phases: ClassVar[list[PhaseSpec]]
    roles: ClassVar[list[RoleSpec]]
    requires: ClassVar[EnvCapabilities]
    artifact: ClassVar[type[Artifact]]

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        """What the core runs the method's episodes with (`learning.runtime`), for the recipe in
        `recipe`."""
        ...


class Method(Declaration, Protocol):
    """A learning method. `learn` runs one task to a frozen artifact of type `artifact`; `jobs`
    are a launch's jobs of its `units` (each learns the task apart, e.g. a variant of the
    method's parameters), each unit's artifact evaluated by `evaluation`, under the method's own
    launch `options`; `kinds` are their bodies by phase (`TaskJobs.stages` but the
    evaluation's)."""

    Config: ClassVar[type[BaseModel]]
    kinds: ClassVar[Mapping[str, JobKind]]

    def jobs(self, units: Sequence[str], evaluation: Evaluation, **options: Any) -> TaskJobs: ...

    async def learn(self, ctx: TaskContext, config: Any) -> ArtifactRef: ...


def check_method(method: type[Declaration] | Declaration) -> list[str]:
    """What is wrong with a method's declarations, one line each (empty: it loads): phase and
    role names unique, every role a phase names declared, the artifact an `Artifact`. Eval tiers
    are checked when each `PhaseSpec` is built."""
    lines = []
    roles = [role.name for role in method.roles]
    phases = [phase.name for phase in method.phases]
    for kind, names in (("role", roles), ("phase", phases)):
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            lines.append(f"{kind}s declared twice: {twice}")
    for phase in method.phases:
        unknown = sorted((set(phase.visibility) | set(phase.tiers)) - set(roles))
        if unknown:
            lines.append(f"phase {phase.name!r} names undeclared roles {unknown}")
    if not (isinstance(method.artifact, type) and issubclass(method.artifact, Artifact)):
        lines.append(f"artifact {method.artifact!r} is not an Artifact subclass")
    return lines


def load_method(ref: str) -> type[Declaration]:
    """The method `ref` names: registered in `roboscript.methods`, or `package.module:Class`.
    Its declarations are checked (`check_method`): a method that does not load is an error."""
    method = entrypoints.load_ref(ref) if ":" in ref else entrypoints.load(GROUP, ref)
    if problems := check_method(method):
        raise KernelError(f"method {ref!r} does not load: {'; '.join(problems)}")
    return method  # type: ignore[no-any-return]
