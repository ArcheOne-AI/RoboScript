"""P1 skill builders: methods whose frozen artifact is a skillset (docs/how-to/add-a-skillset.md).

A builder is a `Method` (`learning.method`) with `artifact = SkillSet`, so it reuses the method
interface, the scheduler, the ledger and the information boundary. Its phases are all `learn`
phases, typically `derive` (from the task list, an optional demonstration and the env's low-level
interfaces: write bindings for the core's entries, new skills and a guide; may use tier P,
offline) and `verify` (run the skillset validation, `roboscript.testing.skills.validate`). What
it builds runs at tier D only: the Hub mounts a skillset's tools at the tier each declares, and
the validation's boundary step refuses a built skill that is not D or that reaches the control
plane. The manifest it freezes records the skillset itself.

Builders register in `roboscript.methods` like any method; `load_builder` loads one and checks
it (`check_builder`). The built-in builder is
`skill_builder` (`roboscript_methods.skill_builder`).
"""

from typing import ClassVar, Final, Protocol

from roboscript.contracts.ids import PhaseName
from roboscript.contracts.skills import SkillSet
from roboscript.kernel import KernelError
from roboscript.learning.method import Declaration, Method, check_method, load_method

DERIVE: Final[PhaseName] = "derive"
"""The typical first phase: work out the operations the tasks need and write them."""
VERIFY: Final[PhaseName] = "verify"
"""The typical last phase: validate the skillset before it is frozen."""


class SkillBuilder(Method, Protocol):
    """A P1 builder: a method whose `learn` freezes a `SkillSet` (`artifact`) for one env and
    embodiment and returns its ref (`skills.store.SkillSetStore`)."""

    artifact: ClassVar[type[SkillSet]]


def check_builder(builder: type[Declaration] | Declaration) -> list[str]:
    """What is wrong with a builder's declarations, one line each (empty: it loads): those of any
    method (`check_method`), an artifact that is not a `SkillSet`, an eval phase (a builder only
    learns: what it built is evaluated by the methods that mount it)."""
    lines = check_method(builder)
    artifact = builder.artifact
    if not (isinstance(artifact, type) and issubclass(artifact, SkillSet)):
        lines.append(f"artifact {artifact!r} is not a SkillSet: a builder freezes a skillset")
    if evals := [phase.name for phase in builder.phases if phase.kind != "learn"]:
        lines.append(f"phases {evals} are not learn phases: a builder only learns")
    return lines


def load_builder(ref: str) -> type[Declaration]:
    """The builder `ref` names (`load_method`), its declarations checked as a builder's."""
    builder = load_method(ref)
    if problems := check_builder(builder):
        raise KernelError(f"builder {ref!r} does not load: {'; '.join(problems)}")
    return builder
