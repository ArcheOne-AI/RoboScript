"""The Agent host plane `agent.host` (docs/concepts/learning.md): a benchmark's own evaluation
client drives the agent's episodes through it, one agent turn per call of its policy.

A client keeps one key, a slot for its sequence of episodes: `reset` begins the slot's next
episode (ending the one before if it still runs); each turn it `poll`s (the agent may be done
already), `submit`s an observation and `poll`s until the turn's actions come as an `ActionBatch`
(or `done`, once the agent's episode is over), runs them and `report`s the states after every
step; `end` says that the benchmark ended the episode (success latch or step limit), with what
outcome and how many of the client's calls failed on the way (infrastructure, counted apart from
the agent's failures).
"""

from typing import Annotated, ClassVar, Final, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.ids import ManifestId, TaskName
from roboscript.contracts.observation import Observation
from roboscript.contracts.versions import VERSIONS

EpisodeKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
"""A client's episode slot, e.g. `my_task.mode_a.seed0`; it names the
slot's directory of episode records, so it is a plain file name."""

INFRA_ENDS: Final = frozenset(
    {"gateway_error", "infra_error", "action_interrupted", "host_error", "not_started"}
)
"""Episode ends that are infrastructure (the gateway or model, the client and relay, the simulator),
not the agent's doing; an episode whose end the benchmark reported (`env_ended`) or whose brain
the next reset cancelled ended as usual."""


class Poll(BaseModel):
    """`pending`: the agent is still thinking; `batch`: the turn's actions; `done`: the agent's
    episode is over (`reason`, e.g. `finish`), so the client holds still until the benchmark
    ends the episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["pending", "batch", "done"]
    batch: ActionBatch | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _batch_with_status(self) -> Self:
        if (self.batch is not None) != (self.status == "batch"):
            raise ValueError("a poll answer carries a batch exactly when its status is batch")
        return self


class OfficialOutcome(BaseModel):
    """How the benchmark ended an episode: its success latch, the steps taken of its limit, and
    how often the client's calls failed in the episode (the host unreachable, a timeout; each
    failure once; the agent acted no more after the first)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    steps: NonNegativeInt
    step_limit: PositiveInt | None
    failed_calls: NonNegativeInt = 0


class AgentHost(Protocol):
    """The plane a benchmark client's policy calls; `manifest` in `reset` is the id of the Memory
    manifest the client evaluates, which must be the host's."""

    PLANE: ClassVar[str] = "agent.host"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def reset(
        self, key: EpisodeKey, task: TaskName, manifest: ManifestId, instruction: str | None = None
    ) -> None: ...

    async def submit(self, key: EpisodeKey, observation: Observation) -> None: ...

    async def poll(self, key: EpisodeKey) -> Poll: ...

    async def report(self, key: EpisodeKey, result: ExecuteResult) -> None: ...

    async def end(self, key: EpisodeKey, outcome: OfficialOutcome) -> None: ...
