"""What a brain is given and what it returns (docs/concepts/learning.md,
docs/concepts/agent-runtime.md).

The runner opens the Hub episode, renders the role's prompt and hands a brain an `AgentSession`;
the brain drives its model through the Hub until the episode ends and returns a `BrainResult`.
Brains reach tools only through the Hub's API and models only through the client factory, which
the runner binds to the gateway (`roboscript.gateway.openai_client`): the episode id is the
gateway session id.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt

from roboscript.agent.recording import Recorder
from roboscript.blobs import BlobStore
from roboscript.contracts.hub import EpisodeEnd, EpisodeSession, ToolHub
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import ContentPart

if TYPE_CHECKING:
    import openai

ModelClients = Callable[[str], "openai.AsyncOpenAI"]
"""Session id -> an OpenAI client of the gateway for that session."""

CONFIG_ENDS: Final = ("model_error:config:", "config_error:")
"""Prefixes of the ends a broken configuration causes (a model request refused as malformed, the
open-time identification failing other than as infrastructure): every episode would end alike,
so S1 and S2 stop the run and the P3 host fails it."""


def unanswered_end(end: EpisodeEnd, unanswered: str) -> EpisodeEnd:
    """`end` as the infrastructure end it is after a model call failed for infrastructure
    reasons with no model call answered after it (`Hub.unanswered_infra`; S1, S2 and the P3 host
    end such an episode so unless the task's success latched): `infra_error`, the end two such
    calls in a row give (`agent.loop`), with the end it had, and its note, in the note. Records
    keep the end it had as `voided_end` and `unanswered` as `voided_by`."""
    note = f"it ended ({end.reason}) after a call failed for infrastructure reasons: {unanswered}"
    if end.note:
        note += f"; its note: {end.note}"
    return end.model_copy(update={"reason": "infra_error", "note": note})


class Limits(BaseModel):
    """The loop's own limits on top of the Hub's budgets: model calls, wall-clock seconds (model
    calls included; the Hub's wall clock applies too, whichever ends first) and consecutive
    format errors (a reply without a tool call, arguments that are not a JSON object, a call the
    Hub refuses as unknown or invalid or over a budget or stage rule). `credit_model_time`: the
    model calls' own time (their latency) is given back to the loop's wall clock as the gateway
    queue is, so that it bounds only the time spent between calls (tools, the simulator) and does
    not trip on a slow model while the agent still acts; `wall_clock_cap_s` is then an absolute
    cap that nothing is credited back to (the runaway's last stop). `max_stalled_calls`: the
    episode ends (`stalled`) once this many model calls in a row passed with no env step used
    (`Counters.steps`): an agent that only looks, checks and finishes without acting is a runaway
    whatever its clock; None: no such guard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_model_calls: PositiveInt | None = None
    wall_clock_s: PositiveFloat | None = None
    max_format_errors: PositiveInt = 3
    credit_model_time: bool = False
    wall_clock_cap_s: PositiveFloat | None = None
    max_stalled_calls: PositiveInt | None = None


@dataclass(frozen=True)
class AgentSession:
    """One episode for a brain: the Hub episode (`episode`: id, stage, role and visible tools),
    the Hub, the system prompt (`instructions`), the blob store that images come from, the
    recorder, the limits, an optional first user message (`brief`: text and images, e.g. the
    Reviser's material with key frames) and the model-client factory, which brains without a
    model do without."""

    episode: EpisodeSession
    hub: ToolHub
    instructions: str
    blobs: BlobStore
    recorder: Recorder
    models: ModelClients | None = None
    brief: Sequence[ContentPart] | None = None
    limits: Limits = field(default_factory=Limits)

    @property
    def episode_id(self) -> EpisodeId:
        return self.episode.episode_id


class Usage(BaseModel):
    """Tokens over an episode's model calls, as the API reports them."""

    model_config = ConfigDict(frozen=True)

    input_tokens: NonNegativeInt = 0
    cached_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    reasoning_tokens: NonNegativeInt = 0


class BrainResult(BaseModel):
    """How a run ended: the Hub episode's end, the turns begun, the model calls made, their token
    usage and the seconds taken. The end is the Hub's (`finish`, a stage rule, a Hub budget) or
    the loop's: `format_errors`, `max_model_calls`, `wall_clock`, `context_overflow` (the
    upstream refused the prompt as too long twice, after the loop had shrunk it; policy failures);
    `gateway_error`, `infra_error` (infrastructure: S1 and S2 re-run the episode);
    `model_error:config:<code>` (the request was refused: configuration) or
    `model_error:policy:<code>` (refused because of what the episode sent, e.g. unsafe);
    `cancelled` (the run was cancelled or crashed; the note says which)."""

    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    end: EpisodeEnd
    turns: NonNegativeInt
    model_calls: NonNegativeInt
    usage: Usage
    elapsed_s: float


class Brain(Protocol):
    """Emits the tool calls of one episode (entry-point group `roboscript.brains`)."""

    async def run(self, session: AgentSession) -> BrainResult: ...
