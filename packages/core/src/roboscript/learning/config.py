"""Settings of the learning protocols, in two files:

- the recipe's `protocols.yaml` (method content, e.g. `recipes/wicl/protocols.yaml`): each
  role's Hub budgets and loop limits, and each protocol's stage parameters (`ProtocolRecipe`,
  validated against the protocol's `Config` when it runs);
- `configs/learning.yaml` (`LearningConfig`, see `configs/learning.example.yaml`): where runs go,
  which recipe, the brain of each role, infrastructure re-runs and env timeouts.

Relative paths are relative to the repository root.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from roboscript.agent import Limits
from roboscript.contracts.hub import Budgets, RenderLimits
from roboscript.contracts.ids import RoleName
from roboscript.contracts.provenance import Provenance
from roboscript.settings import load_config_file, repo_root

PROTOCOLS_FILE: Final = "protocols.yaml"


class PerStep(BaseModel):
    """How a role's bounds grow with its task's official step limit, per step of it: the Hub's
    turns (`turns`, rounded, at most `max_turns`), the loop's model calls (`model_calls`, rounded
    up) and both wall clocks (`wall_clock_s`). The role's configured values are the floors, so a
    short task keeps them; unset fields do not scale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turns: PositiveFloat | None = None
    max_turns: PositiveInt | None = None
    model_calls: PositiveFloat | None = None
    wall_clock_s: PositiveFloat | None = None


class RoleConfig(BaseModel):
    """What bounds a role's episodes: the Hub's budgets, the loop's limits and, if set, render
    limits replacing the Hub's for the role. With `per_step`, an episode of a task with a step
    limit gets bounds in proportion to it, the configured ones being the floor (`for_steps`):
    a longer task gets a longer episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budgets: Budgets = Budgets()
    limits: Limits = Limits()
    render: RenderLimits | None = None
    per_step: PerStep | None = None

    @model_validator(mode="after")
    def _ceiling_above_floor(self) -> Self:
        scale = self.per_step
        if scale is None or scale.max_turns is None:
            return self
        if scale.turns is None:
            raise ValueError("per_step.max_turns caps per_step.turns: set both")
        if self.budgets.max_turns is not None and scale.max_turns < self.budgets.max_turns:
            raise ValueError(
                f"per_step.max_turns ({scale.max_turns}) is below budgets.max_turns "
                f"({self.budgets.max_turns}), the floor"
            )
        return self

    def for_steps(self, step_limit: int | None) -> "RoleConfig":
        """The bounds of an episode whose task allows `step_limit` steps (None: not known, the
        configured bounds)."""
        scale = self.per_step
        if scale is None or step_limit is None:
            return self

        def longer(floor: float | None, per_step: float | None) -> float | None:
            if per_step is None:
                return floor
            scaled = per_step * step_limit
            return scaled if floor is None else max(floor, scaled)

        turns = self.budgets.max_turns
        if scale.turns is not None:
            grown = max(1, round(scale.turns * step_limit))
            turns = grown if turns is None else max(turns, grown)
            turns = turns if scale.max_turns is None else min(turns, scale.max_turns)
        calls = longer(self.limits.max_model_calls, scale.model_calls)
        budgets = self.budgets.model_copy(
            update={
                "max_turns": turns,
                "wall_clock_s": longer(self.budgets.wall_clock_s, scale.wall_clock_s),
            }
        )
        limits = self.limits.model_copy(
            update={
                "max_model_calls": None if calls is None else math.ceil(calls),
                "wall_clock_s": longer(self.limits.wall_clock_s, scale.wall_clock_s),
            }
        )
        return self.model_copy(update={"budgets": budgets, "limits": limits})


class ProtocolRecipe(BaseModel):
    """`protocols.yaml`: the roles' bounds (`for_roles` gives every role of the method one), and
    each protocol's parameters by protocol name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roles: dict[RoleName, RoleConfig] = {}
    protocols: dict[str, dict[str, Any]] = {}

    def for_roles(self, roles: Sequence[RoleName]) -> Self:
        """This recipe with a bound for each of the method's `roles`, in their order (the
        default for a role it leaves out); a role the method does not declare is an error."""
        return self.model_copy(update={"roles": _every(self.roles, roles, RoleConfig, "recipe")})


def load_protocol_recipe(directory: Path) -> ProtocolRecipe:
    """The recipe's `protocols.yaml`."""
    directory = directory if directory.is_absolute() else repo_root() / directory
    return load_config_file(directory / PROTOCOLS_FILE, ProtocolRecipe)


class BrainSpec(BaseModel):
    """A brain registered in `roboscript.brains` and its config (the brain's `Config`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "responses"
    config: dict[str, Any] = {}


class LearningConfig(BaseModel):
    """`configs/learning.yaml`. `method`: the method whose phases and roles the run's episodes have
    (registered in `roboscript.methods`, or `package.module:Class`); `brains`: by role, the
    default brain for a role left out. `max_infra_reruns`: how often S1 and S2 re-run an episode
    that ended for infrastructure reasons before the protocol stops, `infra_backoff_s` how long
    to wait before each of those re-runs (the last value for any beyond the list; empty: none),
    so that a gateway or env server that falters gets time to recover; `env_timeout_s` bounds
    each control-plane call, `open_timeout_s` an `open` (the expert pre-runs). `provenance`: what
    every episode summary records as its code and behaviour config (a launcher's; None: the
    code's revision alone, `settings.source_sha`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_dir: Path
    method: str
    recipe: Path
    lint_overlay: Path | None = None
    brains: dict[RoleName, BrainSpec] = {}
    max_infra_reruns: NonNegativeInt = 3
    infra_backoff_s: list[NonNegativeFloat] = [10.0, 30.0, 90.0]
    env_timeout_s: PositiveFloat = 120.0
    open_timeout_s: PositiveFloat = 900.0
    provenance: Provenance | None = None

    @field_validator("run_dir", "recipe", "lint_overlay")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        return path if path is None or path.is_absolute() else repo_root() / path

    def for_roles(self, roles: Sequence[RoleName]) -> Self:
        """This config with a brain for each of the method's `roles`, in their order (the
        default brain for a role it leaves out); a role the method does not declare is an
        error."""
        return self.model_copy(update={"brains": _every(self.brains, roles, BrainSpec, "brains")})


def _every(given: dict[RoleName, Any], roles: Sequence[RoleName], default: Any, what: str) -> Any:
    if unknown := sorted(set(given) - set(roles)):
        raise ValueError(f"{what} names roles the method does not declare: {unknown}")
    return {role: given.get(role, default()) for role in roles}
