"""P4's declaration (docs/concepts/learning.md): its two phases, the reflection roles (reflector,
curator, reviewer), its config and the stack an evaluated Actor reads.

- `p4_reflect` (learn, held to tier D): three roles read the read-only view of an evaluation's
  records (`learning.eval_view`; D parts and the official success, the truth diagnosis only when
  the config opens it) through the `rsi` toolset: one reflector per task stages proposals, the
  curator stages GLOBAL's, the reviewer gives each a verdict (`reflect`);
- `p4_eval` (eval, D only): the base method's evaluation phase under P4's name, its Actor reading
  the RSI document's GLOBAL part and its task's part beside the Task Memory (`context_stack`); an
  evaluation block of P4 is run by the Agent host in the base's evaluation phase itself, so that
  only the document differs between the arms (`attach`).

P4 attaches after any method's evaluation (`attach`): the base method's phases, roles and runtime
stay as they are, P4's phases and role are added, and the Agent host runs in the base's evaluation
phase (so its Actor's rules are the base's; the RSI document is the only difference). `Rsi` alone
is P4's own declaration (its `p4_eval` lists only the Hub's own tools), which `attach` completes
with the base method's evaluation phase.
"""

import dataclasses
from pathlib import Path
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict

from roboscript.contracts.artifact import Artifact, ArtifactLayer, ArtifactRef, ArtifactStack
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.phase import PhaseSpec, RoleSpec
from roboscript.learning.method import Declaration
from roboscript.learning.runtime import MethodRuntime
from roboscript_methods.rsi.doc import RsiDoc

REFLECTOR: Final = "reflector"
CURATOR: Final = "curator"
REVIEWER: Final = "reviewer"
ROLES: Final = (REFLECTOR, CURATOR, REVIEWER)
REFLECT: Final = "p4_reflect"
EVAL: Final = "p4_eval"

_READ: Final = ["rsi.rsi_episodes", "rsi.rsi_episode", "rsi.rsi_frames", "rsi.rsi_doc"]
P4_REFLECT: Final = PhaseSpec(
    name=REFLECT,
    kind="learn",
    visibility={
        REFLECTOR: [*_READ, "rsi.rsi_memory", "rsi.rsi_edit", "hub.finish"],
        CURATOR: [*_READ, "rsi.rsi_memory", "rsi.rsi_edit", "rsi.rsi_drop", "hub.finish"],
        REVIEWER: [*_READ, "rsi.rsi_memory", "rsi.rsi_verdict", "hub.finish"],
    },
    # the truth diagnosis is the reflection's input when the config opens it, never a tool
    d_only=True,
)
P4_EVAL: Final = PhaseSpec(
    name=EVAL, kind="eval", visibility={"actor": ["hub.finish", "hub.scratchpad"]}
)
"""P4's evaluation phase on its own; `attach` takes the base method's evaluation phase."""


class RsiConfig(BaseModel):
    """`diagnosis`: the reflection also reads the truth diagnosis (tier P, an upper bound,
    reported apart); off, it stays at tier D.
    `evaluation`: the seed block the RSI document is evaluated on (the split's `reflects` names the
    block it reflects on). The edit budgets, caps, lint vocabulary and the reflection's rules are
    the recipe's (`recipe.load_rsi_recipe`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    diagnosis: bool = False
    evaluation: str = EVAL


class Rsi:
    """P4's declaration: `p4_reflect` and `p4_eval`, the three reflection roles and the evaluated
    Actor, the RSI document as artifact. Registered as `rsi` in `roboscript.methods`; run it
    attached to a method (`attach`)."""

    name: ClassVar[str] = "rsi"
    phases: ClassVar[list[PhaseSpec]] = [P4_REFLECT, P4_EVAL]
    roles: ClassVar[list[RoleSpec]] = [
        *(RoleSpec(name=role, prompt=f"prompts/{role}.md") for role in ROLES),
        RoleSpec(name="actor", prompt=""),  # the base method's Actor, once attached
    ]
    requires: ClassVar[EnvCapabilities] = EnvCapabilities()
    artifact: ClassVar[type[Artifact]] = RsiDoc

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        """No prompts of its own: the Agent host runs `p4_eval`, with the base method's prompts
        once attached."""
        return MethodRuntime(host_phase=EVAL)


def attach(base: type[Declaration], evaluation: str | None = None) -> type[Declaration]:
    """`base` with P4 after its evaluation: its phases, then `p4_reflect` and `p4_eval` (the
    base's evaluation phase `evaluation` under P4's name: its rules, its roles' tables and tiers;
    None: the base's one eval phase), its roles and the reflection roles, and its runtime with the
    tools it calls in its evaluation phase called in `p4_eval` too. The Agent host runs in the
    base's evaluation phase (`MethodRuntime.host_phase` is the base's): an evaluated Actor with an
    RSI document reads the base's prompt under the base's rules, the document around it, and with
    no readable entry the base's prompt byte for byte."""
    taken = {phase.name for phase in base.phases} & {REFLECT, EVAL}
    if taken or set(ROLES) & {role.name for role in base.roles}:
        raise ValueError(f"method {base.name!r} declares P4's names already")
    evals = [phase for phase in base.phases if phase.kind == "eval"]
    if evaluation is not None:
        evals = [phase for phase in evals if phase.name == evaluation]
    if len(evals) != 1:
        named = "" if evaluation is None else f" named {evaluation!r}"
        raise ValueError(f"method {base.name!r} has {len(evals)} eval phases{named}, not one")
    (source,) = evals
    phase = PhaseSpec.model_validate({**source.model_dump(), "name": EVAL})
    reflection = [role for role in Rsi.roles if role.name in ROLES]

    def runtime(cls: type, recipe: Path) -> MethodRuntime:
        own = base.runtime(recipe)
        mounts = {**own.mounts}
        if source.name in mounts:
            mounts[EVAL] = mounts[source.name]
        # the Agent host stays in the base's evaluation phase: its Actor's stage rules, the env
        # episode's stage and every rule keyed on that phase are the base's, so the two arms of a
        # P4 evaluation differ in the RSI document alone (`agent.overlay`)
        return dataclasses.replace(own, host_phase=source.name, mounts=mounts)

    attached: type[Declaration] = type(
        f"{base.__name__}WithRsi",
        (base,),
        {
            "name": f"{base.name}+rsi",
            "phases": [*base.phases, P4_REFLECT, phase],
            "roles": [*base.roles, *reflection],
            "runtime": classmethod(runtime),
        },
    )
    return attached


def context_stack(memories: ArtifactStack, rsi: ArtifactRef) -> ArtifactStack:
    """What an evaluated Actor reads, in render order: each task's Task Memory (`memories`, the
    evaluation manifest's stack), then the RSI document's GLOBAL part, then its task's part
    (`agent.overlay`)."""
    tasks = [layer.task for layer in memories.layers if layer.task is not None]
    rsi_layers = [ArtifactLayer(ref=rsi), *(ArtifactLayer(ref=rsi, task=task) for task in tasks)]
    return ArtifactStack(layers=(*memories.layers, *rsi_layers))
