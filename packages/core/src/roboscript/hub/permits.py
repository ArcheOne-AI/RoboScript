"""Only model-proposed calls run (docs/concepts/agent-runtime.md): the Hub's side.

Once a model answer in a registered session is complete, the gateway posts a single-use permit
for each of its tool calls, before the client sees them (`roboscript.gateway.permits`); the
permit guard (`PermitLedger.guard`) lets a model call through only by consuming one of its
episode. `strict` matches the tool and the canonical hash of the arguments, `tool_only` the tool
alone. A permit not used within `ttl_s` of its arrival lapses. System calls (the runtime's
channel) and service calls (toolset to toolset) need none. The guard runs right after
`visibility`, so a permit is spent by the first call that matches it whatever later steps
decide: a call replayed after an error finds none left.
"""

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.events import EventKind
from roboscript.contracts.gateway import UNHASHABLE, Permit, PermitMode, args_hash
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import ToolCall
from roboscript.hub.pipeline import Call, Pipeline, Reject
from roboscript.hub.steps import visibility
from roboscript.records import EventWriter

NO_PERMIT: Final = (
    "Refused: only tool calls the model proposed run, each once and with the arguments the model "
    "gave; this call matches none."
)


@dataclass
class _Ledger:
    mode: PermitMode
    permits: list[tuple[Permit, float]] = field(default_factory=list)  # with the time it came


class PermitLedger:
    """The unconsumed permits of each episode that takes permits, each valid for `ttl_s` from
    its arrival on `clock` (monotonic seconds); `events` gets `permit.issue` and
    `permit.consume` events (the latter saying whether the arguments matched, which measures
    what `tool_only` lets through). `guard` is the pipeline step (`with_permits`)."""

    def __init__(
        self,
        events: EventWriter | None = None,
        *,
        ttl_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events = events
        self.ttl_s = ttl_s
        self.clock = clock
        self._episodes: dict[EpisodeId, _Ledger] = {}

        async def permits(call: Call) -> None:
            if call.request.origin == "model" and self.take(call.episode_id, call.request) is None:
                raise Reject("no_permit", NO_PERMIT)

        self.guard = permits

    def open(self, episode_id: EpisodeId, mode: PermitMode) -> None:
        self._episodes[episode_id] = _Ledger(mode)

    def close(self, episode_id: EpisodeId) -> None:
        self._episodes.pop(episode_id, None)

    def add(self, permits: Sequence[Permit]) -> None:
        """Hold permits (all of one episode) until calls consume them or they lapse."""
        episodes = {permit.episode_id for permit in permits}
        if len(episodes) != 1:
            raise ProtocolError("invalid_request", "send the permits of one episode at a time")
        episode_id = episodes.pop()
        ledger = self._episodes.get(episode_id)
        if ledger is None:
            raise ProtocolError("not_found", f"episode {episode_id} takes no permits")
        now = self.clock()
        ledger.permits += [(permit, now) for permit in permits]
        for permit in permits:
            self._emit(EventKind.PERMIT_ISSUE, permit.model_dump(mode="json"), episode_id)

    def take(self, episode_id: EpisodeId, call: ToolCall) -> Permit | None:
        """Consume the oldest live permit matching `call` in the episode's mode, if any. Arguments
        that are not plain JSON match no permit in `strict` mode."""
        ledger = self._episodes.get(episode_id)
        if ledger is None:
            return None
        oldest = self.clock() - self.ttl_s
        ledger.permits = [(permit, came) for permit, came in ledger.permits if came > oldest]
        try:
            digest: str | None = args_hash(call.arguments)
        except UNHASHABLE:
            digest = None
        for index, (permit, _) in enumerate(ledger.permits):
            exact = digest is not None and permit.args_hash == digest
            if permit.tool == call.name and (exact or ledger.mode == "tool_only"):
                del ledger.permits[index]
                data = {
                    "call_id": call.call_id,
                    "permit_call_id": permit.call_id,
                    "tool": permit.tool,
                    "mode": ledger.mode,
                    "args_match": exact,
                }
                self._emit(EventKind.PERMIT_CONSUME, data, episode_id)
                return permit
        return None

    def _emit(self, kind: str, data: dict[str, Any], episode_id: EpisodeId) -> None:
        if self.events is not None:
            self.events.emit(kind, data, episode_id=episode_id)


def with_permits(pipeline: Pipeline, ledger: PermitLedger) -> Pipeline:
    """`pipeline` with the ledger's guard right after its `visibility` step."""
    if visibility not in pipeline.pre:
        raise ValueError("the pipeline has no `visibility` step to put the permit guard after")
    at = pipeline.pre.index(visibility) + 1
    pre = (*pipeline.pre[:at], ledger.guard, *pipeline.pre[at:])
    return dataclasses.replace(pipeline, pre=pre)
