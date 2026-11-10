"""The event envelope `rs.event/1`: one JSON line per event in each service's log.

The event log is the source of truth: anything a model saw must be rebuildable from it.
`data` holds plain JSON values and BlobRef dumps, never secrets.
"""

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.trace import TraceContext

SCHEMA: Final = "rs.event/1"


class EventKind:
    """Known kinds; `Event.kind` stays a string so services can add their own."""

    EPISODE_OPEN: Final = "episode.open"
    EPISODE_END: Final = "episode.end"
    EPISODE_CLOSE: Final = "episode.close"
    EPISODE_CLOCK: Final = "episode.clock"  # the wall clock restarted or credited (infra waits)
    EPISODE_REBIND: Final = "episode.rebind"  # rules and facts rebound mid-episode
    HUB_ALIAS: Final = "hub.alias"  # a new alias a model may be shown: kind, value, alias
    TOOL_CALL: Final = "tool.call"
    TOOL_RESULT: Final = "tool.result"
    TOOL_REJECT: Final = "tool.reject"
    HUB_UNMOVED: Final = "hub.unmoved"  # a rejection that moved nothing went on, or its fix ran
    MODEL_REQUEST: Final = "model.request"
    MODEL_RESPONSE: Final = "model.response"
    MODEL_RETRY: Final = "model.retry"  # a model call infrastructure failed is sent again
    MODEL_CONTEXT: Final = "model.context"  # the context budget shrank a request's input
    GATEWAY_REQUEST: Final = "gateway.request"
    GATEWAY_CAP: Final = "gateway.cap"  # a deployment's (or pool's) adaptive cap changed
    GATEWAY_HEALTH: Final = "gateway.health"  # a deployment left rotation, probes or is back
    ENV_OBSERVE: Final = "env.observe"
    SCENE_FACT: Final = "scene.fact"
    SCENE_STALE: Final = "scene.stale"  # a remote call's registry change lost a race: not kept
    ENV_EXECUTE: Final = "env.execute"
    PERMIT_ISSUE: Final = "permit.issue"
    PERMIT_CONSUME: Final = "permit.consume"
    MEMORY_EDIT: Final = "memory.edit"
    SCRATCHPAD: Final = "hub.scratchpad"  # the episode's scratchpad after a write
    MEMORY_MERGE: Final = "memory.merge"
    PROTOCOL_STEP: Final = "protocol.step"
    PERCEPTION_STEP: Final = "perception.step"
    PERCEPTION_REUSE: Final = "perception.reuse"  # an episode starts from an earlier one's
    PERCEPTION_REIDENTIFY: Final = "perception.reidentify"  # roles left open, identified again
    REVISER_LEDGER: Final = "reviser.ledger"  # a Reviser's ledger of earlier tries, as shown
    TASK_CARD: Final = "task.card"  # the success condition card a role's prompt shows (envs.cards)
    # identification when an episode opens


class Event(BaseModel):
    """One event; `ts` is unix seconds, `seq` counts the writing service's events."""

    # extra="allow": a reader keeps fields added by newer minors when it rewrites a log.
    model_config = ConfigDict(frozen=True, extra="allow", serialize_by_alias=True)

    schema_: Literal["rs.event/1"] = Field(default=SCHEMA, alias="schema")
    ts: float
    seq: int
    service: str
    kind: str
    episode_id: EpisodeId | None = None
    trace: TraceContext | None = None
    data: dict[str, Any] = Field(default_factory=dict)
