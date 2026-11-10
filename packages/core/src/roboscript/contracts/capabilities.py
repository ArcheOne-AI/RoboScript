"""The env contract's optional faces as declared capabilities, and the generic reset spec
(docs/concepts/embodied-core.md).

One `EnvCapabilities` serves both sides: an env declares what it offers (its `Benchmark`
descriptor, and `discover()` on its public plane, `EnvDiscovery.capabilities`), a method what it
`requires`, and the startup check (`roboscript.envs.capabilities`) reports
`missing(required, offered)` before a run starts. What the sensors give (scene identity, measured
proprioception, depth, cameras, step unit) is the `sensing` sub-block; those five fields sent
flat also parse (lifted into `sensing`). `reruns` is left out of a dump while at its default, so
a declaration that does not use it keeps its hash.
"""

from typing import Annotated, Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.ids import TaskName

SuccessKind = Literal["latch", "terminal_score", "external"]
"""How an env judges success: a latch checked after every action, a score at the end, or an
outside judge (a person at a real robot)."""
ResetKind = Literal["seed", "initial_state", "manual"]
"""How an episode's scene is set: by seed, by an initial-state index, or by a person."""
EpisodeReset = Literal["seed", "initial_state"]
"""The resets an `EpisodeSpec` opens by (`reset_by`); a person's is a real robot's `OpenSpec`."""


_SENSING: Final = ("scene_identity", "measured_proprio", "depth", "cameras", "step_unit")


class SensingCapabilities(BaseModel):
    """What an env's sensors give. `scene_identity`: a seed reproduces its scene;
    `measured_proprio`: the joints and gripper openings are measured, not commanded; `step_unit`:
    what the step budget counts."""

    model_config = ConfigDict(frozen=True)

    scene_identity: bool
    measured_proprio: bool
    depth: bool
    cameras: list[str]
    step_unit: str


class EnvCapabilities(BaseModel):
    """What an env offers beyond the required faces (describe, open/close, the agent face), or,
    as a method's `requires`, what it needs. `truth`: ground-truth snapshots and diagnosis
    (simulator privilege); `expert`: an expert pre-run and its trace; `demos`: a demonstration
    source; `real_robot`: the real-robot extensions (estop, home, control rate, limits, manual
    reset prompt; `contracts.robot_server`). A real robot never offers truth or an expert, judges
    success from outside and is reset by hand. `reruns`: one reset may be opened again
    (False: each only once); as a requirement, the method opens one reset more than once, so the
    default asks for it and a method that never does says False. `sensing`: what the sensors give
    (an env always says; a requirement
    only when it has one: its cameras must be offered, its flags set, its step unit the same)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: frozenset[SuccessKind] = frozenset()
    reset: frozenset[ResetKind] = frozenset()
    truth: bool = False
    expert: bool = False
    demos: bool = False
    real_robot: bool = False
    sensing: SensingCapabilities | None = None
    reruns: bool = True

    @model_validator(mode="before")
    @classmethod
    def _flat_sensing(cls, values: Any) -> Any:
        # the sensing fields may come flat, as the whole capabilities object
        if isinstance(values, dict) and any(key in values for key in _SENSING):
            flat = {key: values[key] for key in _SENSING if key in values}
            values = {k: v for k, v in values.items() if k not in _SENSING}
            values["sensing"] = flat
        return values

    @model_serializer(mode="wrap")
    def _leave_defaults_out(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if self.reruns:
            data.pop("reruns", None)
        return data

    @model_validator(mode="after")
    def _real_robot(self) -> Self:
        if not self.real_robot:
            return self
        wrong = []
        if self.truth:
            wrong.append("truth")
        if self.expert:
            wrong.append("expert")
        if self.success - {"external"}:
            wrong.append(f"success {sorted(self.success - {'external'})}")
        if self.reset - {"manual"}:
            wrong.append(f"reset {sorted(self.reset - {'manual'})}")
        if wrong:
            raise ValueError(f"a real robot cannot declare {', '.join(wrong)}")
        return self


def missing(required: EnvCapabilities, offered: EnvCapabilities) -> list[str]:
    """What `required` asks for that `offered` lacks, one line each (empty: satisfied). Kinds
    are sets: every required kind must be offered."""
    lines = [f"success judged by {kind}" for kind in sorted(required.success - offered.success)]
    lines += [f"reset by {kind}" for kind in sorted(required.reset - offered.reset)]
    for face in ("truth", "expert", "demos", "real_robot"):
        if getattr(required, face) and not getattr(offered, face):
            lines.append(face)
    if required.reruns and not offered.reruns:
        lines.append("reruns of one reset (the env opens each only once)")
    need = required.sensing
    if need is None:
        return lines
    have = offered.sensing
    if have is None:
        return [*lines, "sensing (the env declares none)"]
    for flag in ("scene_identity", "measured_proprio", "depth"):
        if getattr(need, flag) and not getattr(have, flag):
            lines.append(flag)
    lines += [f"camera {name}" for name in need.cameras if name not in have.cameras]
    if need.step_unit != have.step_unit:
        lines.append(f"steps counted in {need.step_unit} (the env counts {have.step_unit})")
    return lines


class SeedReset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["seed"] = "seed"
    seed: NonNegativeInt


class StateReset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["initial_state"] = "initial_state"
    index: NonNegativeInt


class ManualReset(BaseModel):
    """A person sets the scene up; `prompt` tells them how, and the env waits for them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["manual"] = "manual"
    prompt: str


ResetSpec = Annotated[SeedReset | StateReset | ManualReset, Field(discriminator="kind")]


def unoffered_reset(kind: ResetKind, offered: EnvCapabilities) -> str | None:
    """Why an env offering `offered` cannot open an episode reset by `kind` (None: it can), for
    the env to refuse the open with (`invalid_request`)."""
    if kind in offered.reset:
        return None
    return f"this env resets by {sorted(offered.reset)}, not by {kind}"


def episode_reset(offered: EnvCapabilities) -> EpisodeReset:
    """How the core resets the episodes it opens on an env offering `offered`
    (`EpisodeSpec.reset_by`): by seed where the env offers it, else by initial-state index.
    ValueError for an env that offers neither (a real robot is reset by hand, through
    `OpenSpec`)."""
    if "seed" in offered.reset:
        return "seed"
    if "initial_state" in offered.reset:
        return "initial_state"
    raise ValueError(
        f"an env reset by {sorted(offered.reset)} cannot open an EpisodeSpec (by seed or "
        "initial state)"
    )


class OpenSpec(BaseModel):
    """Open one episode of `task` (the minimal control face). `instruction` reuses an earlier
    try's text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    reset: ResetSpec
    instruction: str | None = None
