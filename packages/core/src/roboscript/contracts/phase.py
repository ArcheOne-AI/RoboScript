"""Phases and roles as a method declares them (docs/how-to/add-a-method.md,
docs/architecture.md).

A method names its phases and roles (`PhaseName`, `RoleName`: opaque ids, the core has no fixed
list) and declares, per phase, its kind, rules, what each role sees and each role's tier. The
Hub runs episodes from these declarations (`roboscript.hub.visibility`). The tier
(`contracts.tools.Tier`) and `PhaseKind` stay core concepts: a method names its phases, the core
decides what an eval phase may use.
"""

import re
from collections.abc import Mapping
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Self

from roboscript.contracts.ids import PhaseName, RoleName, ToolName
from roboscript.contracts.tools import Tier

PhaseKind = Literal["learn", "eval"]
"""`eval` phases use D-tier tools only, whatever the method or a config says (as does a learn
phase the method marks `PhaseSpec.d_only`)."""

ENTRY: Final = re.compile(r"^[a-z][a-z0-9_]{0,39}\.([a-z][a-z0-9_]{0,39}|\*|\+)$")
"""A visibility entry: one tool, `toolset.tool`; a whole toolset, `toolset.*` (its tools in the
order the toolset declares them); or `toolset.+`, the tools mounted skillsets add to that toolset
(in the order they are mounted; none where no mounted skillset adds any), so that a method names
a category of tools without naming a benchmark's skills."""


class PhaseRules(BaseModel):
    """What a phase enforces beyond visibility and budgets.

    `fail_fast`: a hard-failure receipt ends the episode. `recover`: it does not, and its result
    tells the model that the attempt goes on from a safe pose (no count, no cap; the benchmark's
    step limit ends the episode). With both, `fail_fast` wins. `continue_unmoved`: under
    `fail_fast`, a rejection that moved nothing (`unmoved`: unreachable or plan_failed with no
    step used, the scene as it was) does not end the episode; its result tells the model to change
    the call, and each of the first `unmoved_events` such rejections adds `unmoved_turns` turns.
    A hard failure that moved anything still ends it. `max_calls`: caps on model calls per tool.
    `done_check`: a tool whose result is a `DoneReport`; the model's `finish` is refused until a
    call of it passed after the last action; `done_refusals`: at most this many such refusals,
    after which `finish` stops waiting for it (perception that never closes a gap must not hold an
    attempt to its turn limit); None: refused every time. `call_arguments`: by workflow stage (the
    call's `rationale.stage`), the tools its Memory call names and the arguments it names for
    each: a model call of such a tool at that stage passing another optional argument at other
    than its default (an axis the Memory never wrote) is refused, but for `freed_arguments` once
    a call of that tool at that stage was rejected as unreachable or plan_failed before anything
    moved (`continue_unmoved`): from then on the attempt may change its approach, as the stage's
    text tells it to. `retry_guard` (with `recover`): a physical call that repeats one which
    failed hard, unchanged (its arguments but the rationale), is refused with a short message
    while nothing has been looked at or done since (no information call, no action without a hard
    failure), and once `retry_after` hard failures happened in its workflow stage, while nothing
    was measured since the stage's last one; another call (candidate, grasp, placement, arm) always
    goes through. At most `retry_refusals` refusals per stage, after which calls go through: the
    guard redirects a loop, it never holds an episode. `failure_turns` (without `fail_fast`): the
    first hard failure that moved anything (what `fail_fast` would end the episode on) leaves at
    most this many more turns, and its result says so; None: no such cap. `give_up_failures` (with
    `recover`; 0: off): once this many of the model's physical calls in a row failed hard for the
    same reason in the same workflow stage, with no gripper taking or letting go of anything since
    (`steps.give_up`), `finish` stops waiting for the done check: the attempt may end, never
    credited, instead of repeating what does not get there until the step limit. In `OpenEpisode`,
    only the fields given
    replace the phase's defaults.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fail_fast: bool = False
    recover: bool = False
    continue_unmoved: bool = False
    unmoved_turns: NonNegativeInt = 0
    unmoved_events: NonNegativeInt = 0
    max_calls: dict[ToolName, PositiveInt] = Field(default_factory=dict)
    done_check: ToolName | None = None
    done_refusals: PositiveInt | None = None
    retry_guard: bool = False
    retry_after: PositiveInt = 3
    retry_refusals: PositiveInt = 3
    give_up_failures: NonNegativeInt = 0
    call_arguments: dict[str, dict[ToolName, list[str]]] = Field(default_factory=dict)
    freed_arguments: list[str] = Field(default_factory=list)
    failure_turns: PositiveInt | None = None


def eval_tier_violations(
    kind: PhaseKind, tiers: Mapping[str, Tier], *, d_only: bool = False
) -> list[str]:
    """The roles of an `eval` phase, or of a learn phase marked `d_only`, given a tier other than
    D (empty for any other `learn` phase). One check for the three places that hold the boundary:
    method load, the Hub at episode start, tool-to-tool calls."""
    if kind != "eval" and not d_only:
        return []
    return [f"role {role!r} has tier {tier}" for role, tier in tiers.items() if tier != "D"]


class BrainRef(BaseModel):
    """A brain registered in `roboscript.brains` and its config (the brain's `Config`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    config: dict[str, Any] = {}


class PhaseSpec(BaseModel):
    """One phase: its kind, its default rules (an episode may replace fields, `OpenEpisode.rules`),
    what each role sees and each role's highest tier. `visibility` lists, per role and in the
    order the role's model is shown them, the tools (`toolset.tool`), whole toolsets
    (`toolset.*`) and the tools skillsets add to a toolset (`toolset.+`) it sees, system-only ones
    included; an entry whose toolset is not mounted is
    left out, one naming a tool its mounted toolset lacks is an error. A role without a tier is
    D: its episodes open at D only. An `eval` phase gives every role D, checked here so that a
    method asking otherwise fails to load. `optional`: entries (of `visibility`) naming a tool a
    skillset adds that may be left out where no mounted skillset adds it (an env without that
    skillset); any other entry naming such a tool fails at start instead of dropping out
    (`hub.visibility`). `d_only`: this learn phase is held to tier D as an eval phase is (every
    role D, checked here; a table resolving to a tier-P tool fails at start; a tier-P episode or
    tool-to-tool call is refused), e.g. a phase whose results must transfer to a real robot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: PhaseName
    kind: PhaseKind
    rules: PhaseRules = PhaseRules()
    visibility: dict[RoleName, list[str]]
    tiers: dict[RoleName, Tier] = {}
    optional: list[str] = []
    d_only: bool = False

    @model_validator(mode="after")
    def _valid(self) -> Self:
        wrong = eval_tier_violations(self.kind, self.tiers, d_only=self.d_only)
        if wrong:
            raise ValueError(f"{self.held} {self.name!r} uses D only: {'; '.join(wrong)}")
        for role, entries in self.visibility.items():
            if bad := [entry for entry in entries if not ENTRY.fullmatch(entry)]:
                raise ValueError(f"phase {self.name!r}, role {role!r}: not entries {bad}")
            if twice := sorted({entry for entry in entries if entries.count(entry) > 1}):
                raise ValueError(f"phase {self.name!r}, role {role!r}: listed twice {twice}")
        listed = {entry for entries in self.visibility.values() for entry in entries}
        if stray := sorted(set(self.optional) - listed):
            raise ValueError(f"phase {self.name!r}: optional entries no role lists {stray}")
        return self

    @property
    def holds_d(self) -> bool:
        """Whether the phase runs at tier D only: an eval phase, or a learn phase `d_only`."""
        return self.kind == "eval" or self.d_only

    @property
    def held(self) -> str:
        """How messages name a phase held to tier D: `eval phase` or `D-only phase`."""
        return "eval phase" if self.kind == "eval" else "D-only phase"

    def tier(self, role: RoleName) -> Tier:
        """The highest tier `role`'s episodes of this phase may open at."""
        return self.tiers.get(role, "D")


class RoleSpec(BaseModel):
    """A role: its default prompt (a path inside the method's package, which a recipe may
    override) and its brain (None: the run's default)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: RoleName
    prompt: str
    brain: BrainRef | None = None
