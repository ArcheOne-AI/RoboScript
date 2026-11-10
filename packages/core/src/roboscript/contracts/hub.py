"""The Tool Hub's API (docs/concepts/agent-runtime.md): episodes, tool calls, outcomes, budgets,
rules.

Brains and runtimes type against `ToolHub`, the in-process API. A served Hub adds the
`hub.control` plane (`HubControl`, the same API over RPC for the runner) and, per episode, what a
brain needs: the key of its gateway session (with permits on) and, for a brain in another
process, its access to the MCP endpoint (`EpisodeAccess`).
"""

from typing import Any, ClassVar, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt

from roboscript.contracts.gateway import PermitMode
from roboscript.contracts.ids import EpisodeId, FrameId, PhaseName, RoleName, ToolName, WireSecret
from roboscript.contracts.observation import FrameEntry
from roboscript.contracts.phase import PhaseRules
from roboscript.contracts.tools import (
    EnvBinding,
    Tier,
    ToolCall,
    ToolDescriptor,
    ToolResult,
)
from roboscript.contracts.versions import VERSIONS

CallCode = Literal[
    "unknown_episode",  # no such episode
    "episode_ended",  # the one fixed error for every call after the end
    "unknown_tool",  # not visible in this episode, whether or not it exists
    "invalid_arguments",  # arguments or aliases the params model refuses
    "action_in_progress",  # a turn-ending call while an action runs
    "budget_exhausted",
    "stage_rule",
    "finish_refused",  # the runner's finish guard refused the first `finish`
    "retry_refused",  # a hard-failed call repeated with nothing changed (`PhaseRules.retry_guard`)
    "timeout",
    "tool_error",  # the tool refused (ToolError)
    "infra_error",  # retryable infrastructure failure (InfraError)
    "internal_error",  # anything unexpected; details stay in the Hub's log
    "cancelled",  # the caller went away (only in the event log)
]
"""Error codes of the built-in pipeline steps; guards added later may use their own."""

ENDED_MESSAGE: Final = "The episode has ended; no more tool calls are accepted."


class Budgets(BaseModel):
    """Per-episode limits, None meaning unlimited. Only model calls count: turns are execution
    calls and `finish`, information calls are pure and perception calls. The episode ends when
    the last turn is used or the wall clock runs out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_turns: PositiveInt | None = None
    max_info_calls: PositiveInt | None = None
    wall_clock_s: PositiveFloat | None = None


class DoneReport(BaseModel):
    """What a `PhaseRules.done_check` tool reports: whether every done criterion holds, and a
    line for each one that does not (the refused `finish` repeats them). `gave_up`: the check
    has made no progress for so many calls in a row that it no longer holds `finish`, which
    then ends the episode (a finish is never a success: the official check alone decides)."""

    passed: bool
    failing: list[str]
    gave_up: bool = False


class RenderLimits(BaseModel):
    """How much of a result a model is shown; None keeps the Hub's value for the role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_max_side: PositiveInt | None = None
    max_images: PositiveInt | None = None
    max_text_chars: PositiveInt | None = None


class SeedAlias(BaseModel):
    """An alias an earlier episode showed a model: of `kind` (`obj`, `region`, ...) for `value`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    value: str
    alias: str


class EpisodeSeed(BaseModel):
    """What an episode starts with from earlier ones: their frames, under the same aliases (so a
    Reviser can look at the Actor's `frame_3`), their scene facts by name, and other aliases they
    showed (objects, regions), so that text written there names the same things here. `live`
    names the newest frame the episode itself took, when the seed is a running episode's snapshot
    (a remote toolset's): seeded frames of other episodes are never live."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames: list[FrameEntry] = Field(default_factory=list)
    facts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    aliases: list[SeedAlias] = Field(default_factory=list)
    live: FrameId | None = None


class OpenEpisode(BaseModel):
    """Open a Hub episode of phase `stage` (a phase the Hub's method declares; the field keeps
    its wire name) for `role` at `tier`. `toolsets` is the profile (None: every mounted
    toolset); `rules` changes the phase's default rules field by field; `structured` also sends
    results as JSON; `render` changes the role's render limits; `seed` fills the frame store;
    `permits` is how a served Hub with permits on matches this episode's calls (None: the Hub's
    default), for a client that must be relaxed to `tool_only`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: PhaseName
    tier: Tier = "D"
    role: RoleName = "actor"
    toolsets: list[str] | None = None
    budgets: Budgets = Budgets()
    rules: PhaseRules | None = None
    env: EnvBinding | None = None
    embodiment: str | None = None
    structured: bool = False
    render: RenderLimits | None = None
    seed: EpisodeSeed | None = None
    permits: PermitMode | None = None


class EpisodeAccess(BaseModel):
    """How a brain outside the Hub's process reaches its episode: the Hub's MCP endpoint with
    `token` as bearer, until `expires_at` (Unix time) or until the episode closes. The runtime's
    system calls go through the control plane (`HubControl.system_call`), never through MCP."""

    model_config = ConfigDict(frozen=True)

    mcp_url: str
    token: WireSecret
    expires_at: float


class EpisodeSession(BaseModel):
    """An open episode: its id (also the gateway session id), its fixed tool list and the
    system-only tools only the runtime calls (`identify`); from a served Hub with permits on,
    `session_key`, which every model request of the episode's gateway session must carry
    (`openai_client(episode_id, key=...)`), and with an MCP endpoint, `access` for a brain in
    another process."""

    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    stage: PhaseName
    tier: Tier
    role: RoleName
    tools: list[ToolDescriptor]
    system_tools: list[ToolName] = Field(default_factory=list)
    budgets: Budgets
    rules: PhaseRules
    session_key: WireSecret | None = None
    access: EpisodeAccess | None = None


class CallOutcome(ToolResult):
    """A call's rendered result; `episode_ended` says whether the episode is over after it."""

    call_id: str
    episode_ended: bool = False


class Counters(BaseModel):
    """Calls so far: model calls by kind, system calls, and how many were rejected before
    reaching their tool or ended with an error there; `steps`: the env steps the actions'
    receipts report used (what a stall guard watches, `agent.session.Limits.max_stalled_calls`)."""

    model_config = ConfigDict(frozen=True)

    turns: NonNegativeInt = 0
    info_calls: NonNegativeInt = 0
    other_calls: NonNegativeInt = 0
    system_calls: NonNegativeInt = 0
    rejected: NonNegativeInt = 0
    errors: NonNegativeInt = 0
    steps: NonNegativeInt = 0


class EpisodeEnd(BaseModel):
    """Why an episode ended: `finish`, `fail_fast:<reason>@<stage>`, `max_turns`, `wall_clock`,
    `action_interrupted` (an action timed out or was cancelled midway: infrastructure),
    `env_ended` (a toolset found the env episode over), `closed`, or the reason a runner gave;
    `at` is Unix time."""

    model_config = ConfigDict(frozen=True)

    reason: str
    note: str = ""
    at: float


class EpisodeStatus(BaseModel):
    """An episode now."""

    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    stage: PhaseName
    tier: Tier
    role: RoleName
    tools: list[ToolName]
    budgets: Budgets
    rules: PhaseRules
    counters: Counters
    elapsed_s: float
    ended: EpisodeEnd | None
    acting: bool


class EpisodeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    end: EpisodeEnd
    counters: Counters
    elapsed_s: float


class ToolHub(Protocol):
    """The Hub's API. `call` takes model calls and never raises for a bad call: every problem is
    an error outcome; `system_call` is the runtime's channel."""

    async def open_episode(self, spec: OpenEpisode) -> EpisodeSession: ...

    async def call(self, episode_id: EpisodeId, call: ToolCall) -> CallOutcome: ...

    async def system_call(
        self, episode_id: EpisodeId, name: ToolName, arguments: dict[str, Any]
    ) -> CallOutcome: ...

    async def tools(self, episode_id: EpisodeId) -> list[ToolDescriptor]: ...

    async def end_episode(
        self, episode_id: EpisodeId, reason: str, note: str = ""
    ) -> EpisodeStatus: ...

    async def credit_clock(self, episode_id: EpisodeId, seconds: float, reason: str) -> None: ...

    async def close_episode(self, episode_id: EpisodeId) -> EpisodeSummary: ...

    async def state(self, episode_id: EpisodeId) -> EpisodeStatus: ...


class HubControl(ToolHub, Protocol):
    """The Hub's control plane: the `ToolHub` API of a Hub in another process, for the runner
    only (bearer: the control token derived from `RS_HUB_SECRET`). `call` stays a model's call:
    with permits on it needs one, like any other."""

    PLANE: ClassVar[str] = "hub.control"
    VERSION: ClassVar[str] = VERSIONS[PLANE]
