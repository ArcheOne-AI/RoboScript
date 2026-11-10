"""What a toolset gets: `ToolsetRuntime` once when it starts, a `ToolContext` with every call."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from roboscript.blobs import BlobStore
from roboscript.contracts.env import EnvAgent
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import CallContext, EnvBinding
from roboscript.records import EventWriter
from roboscript.tools.frames import FrameStore

ServiceCall = Callable[["ToolContext", str, Mapping[str, Any]], Awaitable[BaseModel]]
"""Runs another toolset's tool for the episode of the given context, outside the Hub pipeline."""

EventSink = Callable[["ToolContext", str, dict[str, Any]], None]


@dataclass(frozen=True)
class ToolsetRuntime:
    """The toolset's validated config and the handles it may keep: the blob store, a client
    factory for env bindings, a model-client factory (an OpenAI client through the gateway,
    e.g. `models(session_id=...)`), the service-call function and the event log."""

    config: BaseModel
    blobs: BlobStore
    env_client: Callable[[EnvBinding], EnvAgent]
    models: Callable[..., Any]
    call: ServiceCall
    events: EventWriter | None


@dataclass(frozen=True)
class ToolContext:
    """One call's episode context (`context`, what travels as MCP `_meta`) plus handles.

    `state` is this toolset's dict for the episode, dropped when the episode closes; `frames` is
    the episode's frame store, shared by all toolsets.
    """

    context: CallContext
    toolset: str
    call_id: str
    env_client: EnvAgent | None
    blobs: BlobStore
    state: dict[str, Any]
    frames: FrameStore = field(repr=False)
    service_call: ServiceCall = field(repr=False)
    event_sink: EventSink = field(repr=False)

    @property
    def episode_id(self) -> EpisodeId:
        return self.context.episode_ref

    @property
    def env(self) -> EnvAgent:
        """The episode's env (agent plane); call it with `env_episode`."""
        if self.env_client is None:
            raise LookupError(f"episode {self.episode_id} has no env binding")
        return self.env_client

    @property
    def env_episode(self) -> EpisodeId:
        """The env's own id for this episode."""
        if self.context.env is None:
            raise LookupError(f"episode {self.episode_id} has no env binding")
        return self.context.env.episode_id

    async def call(self, tool: str, **args: Any) -> BaseModel:
        """Run another toolset's tool for this episode, bypassing the Hub pipeline (no budget,
        lock or aliases: pass ids); returns its result model."""
        return await self.service_call(self, tool, args)

    def emit(self, kind: str, data: dict[str, Any]) -> None:
        """Write an event `<toolset>.<kind>` to the episode's log: plain JSON, no secrets."""
        self.event_sink(self, kind, data)
