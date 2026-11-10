"""The own loop (docs/concepts/learning.md, docs/concepts/agent-runtime.md): the turn state
machine every brain on it shares. A brain supplies the model call; the loop builds the requests,
runs the calls through the Hub and records everything.

A turn opens, once the loop's limits allow it, with the observation: `observe` on the Hub's
system channel when the episode has it, shown as one user message, followed by the episode's
`scratchpad` (also a system call) once the model has written in it. The model then makes
information calls, each answered with a `function_call_output`, and closes the turn with a
turn-ending call (an action or `finish`) that the Hub admits; later calls of the same reply are
answered "one physical action per turn". A reply without a tool call gets a short nudge.

The episode ends when the Hub says so, or the loop ends it:

- `format_errors`: `Limits.max_format_errors` replies in a row that are format errors: without a
  tool call, or whose calls all have arguments that are not a JSON object or are refused by the
  Hub as unknown, invalid, over a budget or against a stage rule (a reply counts once however
  many such calls it makes; one call that goes through resets the count);
- `stalled`: `Limits.max_stalled_calls` model calls in a row with no env step used
  (`Counters.steps`), whatever the clock;
- `max_model_calls`; `wall_clock`: the earlier of `Limits.wall_clock_s` and the Hub's wall
  clock, which also cuts off a model call; with `Limits.credit_model_time` each model call's
  latency (less its gateway queue, credited already) is given back to the loop's wall clock, which
  then no longer cuts off a call, and `Limits.wall_clock_cap_s`, never credited, bounds the
  episode and its calls instead;
- `infra_error`: the observation, or two tool calls in a row, failed for infrastructure reasons
  (S1 and S2 re-run such episodes, like those a `gateway_error` ends);
- the reason a failed model call gives: `gateway_error`, `model_error:<config|policy>:<code>`;
  a call that failed on infrastructure (`ModelError.retry`) is first sent again after each of
  `LoopConfig.retry_waits_s` (a `model.retry` event each; every attempt is in `llm_calls.jsonl`,
  as the same call), within the wall clock. Waiting on infrastructure is not the role's time:
  the retry waits and the time a reply says it queued at the gateway (`meta["queued_s"]`) are
  given back to both wall clocks (`ToolHub.credit_clock`);
- `context_overflow`: the upstream refused a prompt as too long twice. Every request is first
  held to the context budget (`agent.context`; a `model.context` event whenever it shrinks the
  input); after the first refusal the rest of the episode runs on the tighter overflow budget,
  and the call is sent again, shrunk, as its next attempt;
- `cancelled`: the run was cancelled or crashed; the unfinished turn is recorded first.
"""

import contextlib
import dataclasses
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import anyio
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat

from roboscript.agent.context import ContextBudget, estimate, tighter
from roboscript.agent.history import (
    CallRecord,
    Detail,
    Exchange,
    HistoryPolicy,
    ImageDelivery,
    Shrink,
    Turn,
    build_input,
    text_of,
    tool_images,
)
from roboscript.agent.recording import materialize
from roboscript.agent.session import AgentSession, BrainResult, Usage
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.hub import CallOutcome, EpisodeEnd, EpisodeStatus
from roboscript.contracts.ids import ToolName
from roboscript.contracts.tools import TURNS, ContentPart, ImagePart, TextPart, ToolCall
from roboscript.contracts.trace import get_trace, new_trace, use_trace
from roboscript.tools import to_responses_tool

FORMAT_ERRORS: Final = frozenset(
    {"invalid_arguments", "unknown_tool", "budget_exhausted", "stage_rule"}
)
"""Hub error codes that count as format errors."""
INFRA: Final = "infra_error"
ONE_ACTION: Final = "one_action_per_turn"
SCRATCHPAD: Final = ToolName("scratchpad")
OVERFLOW: Final = "context_overflow"
"""The reason of a model call the upstream refused because its prompt was too long."""
_BAD_OBSERVE: Final = frozenset({"invalid_arguments", "unknown_tool"})


@dataclass(frozen=True)
class Reply:
    """A model call's answer: output items (Responses JSON) and usage, and what the log keeps
    about how it was served (`meta`: response id, status, deployment, ...)."""

    output: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ModelError(Exception):
    """A failed model call; the loop ends the episode with `reason` and `detail` as its note.
    `retry`: infrastructure failed it (a gateway 502/503/504 of the upstream's making, a broken
    connection, a timeout), so that the same request may succeed when sent again."""

    def __init__(self, reason: str, detail: str, *, retry: bool = False) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.retry = retry


Model = Callable[[dict[str, Any]], Awaitable[Reply]]
"""One model call: the request as sent (images as data URLs) -> the reply, or ModelError."""


class LoopConfig(BaseModel):
    """What brains on the own loop share: the history policy; where images from tool results go
    (`output`, the default: inside the `function_call_output`; `message`: in one user message
    after the reply's outputs, which drops the model's reasoning carry-over within the turn);
    the images' `detail`; whether each turn opens with `observe` (when the episode has it) and its
    arguments; the nudge for a reply without a tool call; the waits before a model call that
    infrastructure failed is sent again (one retry each); the context budget every request is
    held to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    history: HistoryPolicy = HistoryPolicy()
    image_delivery: ImageDelivery = "output"
    image_detail: Detail = "auto"
    observe: bool = True
    observe_arguments: dict[str, Any] = Field(default_factory=dict)
    nudge: str = (
        "Answer with a tool call: information calls as needed, then one physical action or finish."
    )
    retry_waits_s: list[NonNegativeFloat] = [5.0, 20.0, 60.0, 120.0]
    context: ContextBudget = ContextBudget()


async def run_loop(
    session: AgentSession, model: Model, options: dict[str, Any], config: LoopConfig
) -> BrainResult:
    """Play `session`'s episode until it ends; `options` are the request's fixed fields
    (`model`, `store`, ...), to which each call adds `instructions`, `input` and `tools`."""
    return await _Loop(session, model, options, config).run()


class _Loop:
    def __init__(
        self, session: AgentSession, model: Model, options: dict[str, Any], config: LoopConfig
    ) -> None:
        self.session = session
        self.hub = session.hub
        self.episode_id = session.episode_id
        self.model = model
        self.options = options
        self.config = config
        tools = session.episode.tools
        self.exported = [to_responses_tool(tool) for tool in tools]
        self.turn_ending = {tool.name for tool in tools if tool.meta.kind in TURNS}
        self.observing = config.observe and any(tool.name == "observe" for tool in tools)
        self.noting = False  # the model has written in its scratchpad
        self.turns: list[Turn] = []
        self.calls = 0
        self.format_errors = 0
        self.infra_errors = 0
        self.usage: Counter[str] = Counter()
        self.shrink = Shrink()  # the context budget's, tighter only (but per turn for folds)
        self.shrunk_turn = 0  # the turn `shrink.fold_exchanges` counts in
        self.overflowed = False  # the upstream refused a prompt as too long
        self.trace = get_trace() or new_trace()  # model calls are its child spans
        self.started = time.monotonic()
        self.infra_s = 0.0  # waited on infrastructure (or the model): off the loop's wall clock
        self.latency = 0.0  # the last model call's
        self.steps = 0  # the env steps used so far (`Counters.steps`)
        self.stepped_at = 0  # the model calls made when the step count last changed

    async def run(self) -> BrainResult:
        end: EpisodeEnd | None = None
        turn: Turn | None = None  # not yet recorded
        try:
            while end is None:
                turn = Turn(index=len(self.turns) + 1)
                self.turns.append(turn)
                if await self._turn(turn):
                    end = (await self.hub.state(self.episode_id)).ended
                self.session.recorder.turn(_record(turn, end))
                turn = None
        except BaseException as error:
            with anyio.CancelScope(shield=True):
                await self._abort(turn, error)
            raise
        return BrainResult(
            episode_id=self.episode_id,
            end=end,
            turns=len(self.turns),
            model_calls=self.calls,
            usage=Usage(**self.usage),
            elapsed_s=round(time.monotonic() - self.started, 3),
        )

    async def _turn(self, turn: Turn) -> bool:
        """Play one turn; whether the episode is over."""
        if await self._stop():
            return True
        if self.observing and await self._observe(turn):
            return True
        if self.noting and await self._notes(turn):
            return True
        while not await self._stop():
            reply = await self._ask(turn)
            if reply is None:
                return True
            exchange = Exchange(call=self.calls, output=reply.output)
            turn.exchanges.append(exchange)
            calls = [item for item in reply.output if item.get("type") == "function_call"]
            if not calls:
                exchange.nudge = self.config.nudge
                if await self._format_error():
                    return True
                continue
            ending: CallRecord | None = None  # the reply's admitted turn-ending call
            malformed = valid = False  # the reply's format errors count once, as one reply
            for item in calls:
                if ending is not None:
                    exchange.results.append(_refused(item, ending))
                    continue
                record, over, admitted = await self._call(item)
                exchange.results.append(record)
                if admitted:
                    ending = turn.action = record
                if over or await self._infra(record):
                    return True
                if _malformed(record):
                    malformed = True
                else:
                    valid = True
            if valid:
                self.format_errors = 0
            elif malformed and await self._format_error():
                return True
            if turn.action is not None:
                return False
        return True

    async def _observe(self, turn: Turn) -> bool:
        """The turn's observation; whether the episode is over. Refused arguments are the
        runner's configuration error."""
        arguments = dict(self.config.observe_arguments)
        outcome = await self.hub.system_call(self.episode_id, ToolName("observe"), arguments)
        if outcome.error_code in _BAD_OBSERVE:
            raise ValueError(
                f"the per-turn observe was refused ({outcome.error_code}: "
                f"{text_of(outcome.content)}); fix LoopConfig.observe_arguments"
            )
        turn.observation = _shown(outcome)
        if outcome.episode_ended:
            return True
        if outcome.error_code == INFRA:
            await self._end(INFRA, f"the observation failed: {text_of(outcome.content)}")
            return True
        return False

    async def _notes(self, turn: Turn) -> bool:
        """The scratchpad, after the turn's observation; whether the episode is over."""
        outcome = await self.hub.system_call(self.episode_id, SCRATCHPAD, {})
        turn.observation = [*(turn.observation or []), *_shown(outcome)]
        return outcome.episode_ended

    async def _ask(self, turn: Turn) -> Reply | None:
        """One model call, cut off at the wall clock; None when it failed, which ended the
        episode. The request is logged before it is sent."""
        self.calls += 1
        recorder, call = self.session.recorder, self.calls
        waits = self.config.retry_waits_s
        attempt, retries = 1, 0
        request, refs = self._request(turn, call, attempt, "budget")
        sent = await materialize(request, refs, self.session.blobs)
        while True:
            reply, error = await self._send(turn, call, attempt, request, refs, sent)
            if error is not None and error.reason == OVERFLOW and not self.overflowed:
                self.overflowed = True
                attempt += 1
                request, refs = self._request(turn, call, attempt, "overflow")
                sent = await materialize(request, refs, self.session.blobs)
                continue
            if error is None or not error.retry or retries == len(waits):
                break
            wait = waits[retries]
            left = self._left(await self.hub.state(self.episode_id))
            if left is not None and left <= wait:
                break  # the wall clock would end it first
            attempt, retries = attempt + 1, retries + 1
            recorder.retry(self.episode_id, call, turn.index, attempt, wait, str(error))
            await anyio.sleep(wait)
            await self._credit(wait, "model_retry")
        if error is not None:
            await self._end(error.reason, error.detail)
            return None
        assert reply is not None
        self._count(reply.usage)
        queued = float(reply.meta.get("queued_s") or 0.0)
        await self._credit(queued, "gateway_queue")
        if self.session.limits.credit_model_time:  # the loop's own clock only, not the Hub's
            self.infra_s += max(0.0, self.latency - queued)
        return reply

    def _request(
        self, turn: Turn, call: int, attempt: int, reason: str
    ) -> tuple[dict[str, Any], dict[str, BlobRef]]:
        """The request for model call `call`, held to the context budget: over it (or after an
        overflow), the input shrinks step by step down to `shrink_to` of it, and a
        `model.context` event says how."""
        config, budgeting = self.config, self.config.context
        if self.shrunk_turn != turn.index:
            self.shrink = dataclasses.replace(self.shrink, fold_exchanges=0)
            self.shrunk_turn = turn.index
        if self.overflowed and budgeting.overflow_delivery is not None:
            self.shrink = dataclasses.replace(self.shrink, delivery=budgeting.overflow_delivery)
        budget = budgeting.budget(str(self.options.get("model", "")), self.overflowed)
        request, refs = self._build()
        before = after = estimate(request, refs, budgeting)
        if after <= budget and reason == "budget":
            return request, refs
        start = self.shrink
        while after > budget * budgeting.shrink_to:
            shown = tool_images(self.turns, config.history, self.shrink)
            step = tighter(self.shrink, self.turns, config.history, shown)
            if step is None:
                break
            self.shrink = step
            request, refs = self._build()
            after = estimate(request, refs, budgeting)
        data = {
            "call": call,
            "turn": turn.index,
            "reason": reason,
            "budget": budget,
            "before": before,
            "after": after,
            "from": dataclasses.asdict(start),
            "to": dataclasses.asdict(self.shrink),
            "over": after > budget,
            **({} if attempt == 1 else {"attempt": attempt}),
        }
        self.session.recorder.context(self.episode_id, data)
        return request, refs

    def _build(self) -> tuple[dict[str, Any], dict[str, BlobRef]]:
        config = self.config
        items, refs = build_input(
            self.turns,
            self.session.brief,
            config.history,
            config.image_delivery,
            config.image_detail,
            self.shrink,
        )
        request = {
            **self.options,
            "instructions": self.session.instructions,
            "input": items,
            "tools": self.exported,
        }
        return request, refs

    async def _send(
        self,
        turn: Turn,
        call: int,
        attempt: int,
        request: dict[str, Any],
        refs: Mapping[str, BlobRef],
        sent: dict[str, Any],
    ) -> tuple[Reply | None, ModelError | None]:
        """One attempt of model call `call`, cut off at the wall clock, logged before it is sent
        and once it is over."""
        recorder = self.session.recorder
        with use_trace(self.trace.child()):
            recorder.request(self.episode_id, call, turn.index, request, refs, attempt=attempt)
            left = self._left(await self.hub.state(self.episode_id), calling=True)
            started, reply, error = time.monotonic(), None, None
            try:
                with anyio.move_on_after(left) as deadline:
                    try:
                        reply = await self.model(sent)
                    except ModelError as failed:
                        error = failed
                    except TimeoutError as failed:  # the model's own, not the wall clock
                        error = ModelError("gateway_error", f"TimeoutError: {failed}", retry=True)
            except BaseException as stopped:  # cancelled or crashed: logged, then passed on
                recorder.response(
                    self.episode_id, call, turn.index, None, _stopped(stopped), attempt=attempt
                )
                raise
            if deadline.cancelled_caught:
                error = ModelError("wall_clock", "a model call ran past the wall clock")
            response = None
            if reply is not None:
                latency = round(time.monotonic() - started, 3)
                self.latency = latency
                response = {"output": reply.output, "usage": reply.usage, **reply.meta}
                response["latency_s"] = latency
            failure = None if error is None else str(error)
            recorder.response(self.episode_id, call, turn.index, response, failure, attempt=attempt)
        return reply, error

    async def _call(self, item: dict[str, Any]) -> tuple[CallRecord, bool, bool]:
        """Send a function call to the Hub: its record, whether the episode is over and whether
        the Hub admitted it as the turn's action."""
        arguments = _arguments(item.get("arguments"))
        if arguments is None:
            text = (
                f"Error (invalid_arguments): the arguments of {item.get('name')} are not a JSON "
                "object; call it again with a JSON object."
            )
            record = CallRecord(item, None, [TextPart(text=text)], "invalid_arguments", "invalid")
            return record, False, False
        name = ToolName(str(item.get("name", "")))
        ending = name in self.turn_ending
        before = (await self.hub.state(self.episode_id)).counters.turns if ending else 0
        request = ToolCall(
            call_id=str(item.get("call_id", "")), name=name, arguments=arguments, origin="model"
        )
        outcome = await self.hub.call(self.episode_id, request)
        if name == SCRATCHPAD and not outcome.is_error and arguments.get("note") is not None:
            self.noting = True
        admitted = ending and (await self.hub.state(self.episode_id)).counters.turns > before
        error = outcome.error_code if outcome.is_error else None
        return CallRecord(item, arguments, _shown(outcome), error), outcome.episode_ended, admitted

    async def _infra(self, record: CallRecord) -> bool:
        """Count a call's infrastructure errors; whether they ended the episode."""
        self.infra_errors = self.infra_errors + 1 if record.error == INFRA else 0
        if self.infra_errors == 2:
            await self._end(INFRA, "two tool calls in a row failed for infrastructure reasons")
            return True
        return False

    async def _format_error(self) -> bool:
        """Count a reply's format error(s), once per reply: parallel calls the Hub refuses
        together are one mistake, not several in a row; whether it ended the episode."""
        self.format_errors += 1
        limit = self.session.limits.max_format_errors
        if self.format_errors < limit:
            return False
        await self._end("format_errors", f"{limit} in a row")
        return True

    async def _stop(self) -> bool:
        """Whether the episode is over, ending it if a loop limit is reached."""
        status = await self.hub.state(self.episode_id)
        if status.ended is not None:
            return True
        limit = self.session.limits.max_model_calls
        if limit is not None and self.calls >= limit:
            await self._end("max_model_calls", f"all {limit} model calls are used")
            return True
        if status.counters.steps != self.steps:
            self.steps, self.stepped_at = status.counters.steps, self.calls
        stall = self.session.limits.max_stalled_calls
        if stall is not None and self.calls - self.stepped_at >= stall:
            await self._end("stalled", f"{stall} model calls in a row used no env step")
            return True
        left = self._left(status)
        if left is not None and left <= 0:
            await self._end("wall_clock", "the wall clock ran out")
            return True
        return False

    def _left(self, status: EpisodeStatus, calling: bool = False) -> float | None:
        """Seconds until the earlier of the loop's and the Hub's wall clocks and the loop's cap;
        None if none. `calling`: for a model call in flight, which a wall clock that credits model
        time back cannot run out during."""
        left = []
        limits = self.session.limits
        now = time.monotonic()
        credited = calling and limits.credit_model_time
        if (limit := limits.wall_clock_s) is not None and not credited:
            left.append(limit - (now - self.started - self.infra_s))
        if (cap := limits.wall_clock_cap_s) is not None:
            left.append(cap - (now - self.started))
        if (budget := status.budgets.wall_clock_s) is not None:
            left.append(budget - status.elapsed_s)
        return min(left, default=None)

    async def _credit(self, seconds: float, reason: str) -> None:
        if seconds > 0:
            self.infra_s += seconds
            await self.hub.credit_clock(self.episode_id, seconds, reason)

    async def _end(self, reason: str, note: str) -> None:
        await self.hub.end_episode(self.episode_id, reason, note)

    async def _abort(self, turn: Turn | None, error: BaseException) -> None:
        """After a cancellation or crash: end the episode as `cancelled`, record the turn."""
        end = None
        with contextlib.suppress(ProtocolError):  # closed already: nothing to end
            end = (await self.hub.end_episode(self.episode_id, "cancelled", _stopped(error))).ended
        if turn is not None:
            self.session.recorder.turn(_record(turn, end))

    def _count(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        self.usage["input_tokens"] += usage.get("input_tokens") or 0
        self.usage["output_tokens"] += usage.get("output_tokens") or 0
        self.usage["cached_tokens"] += (usage.get("input_tokens_details") or {}).get(
            "cached_tokens"
        ) or 0
        self.usage["reasoning_tokens"] += (usage.get("output_tokens_details") or {}).get(
            "reasoning_tokens"
        ) or 0


def _stopped(error: BaseException) -> str:
    """What stopped a run: a cancellation, or a crash with its exception."""
    if isinstance(error, anyio.get_cancelled_exc_class()):
        return "cancelled"
    return f"crashed: {type(error).__name__}: {error}"


def _shown(outcome: CallOutcome) -> list[ContentPart]:
    """What the model is shown of an outcome: an error's text says it is one."""
    if not outcome.is_error:
        return list(outcome.content)
    text = f"Error ({outcome.error_code}): {text_of(outcome.content)}"
    return [TextPart(text=text), *(p for p in outcome.content if isinstance(p, ImagePart))]


def _malformed(record: CallRecord) -> bool:
    return record.status == "invalid" or record.error in FORMAT_ERRORS


def _refused(item: dict[str, Any], ending: CallRecord) -> CallRecord:
    text = (
        f"Error ({ONE_ACTION}): not executed. One physical action per turn, and this turn's is "
        f"{ending.name} ({ending.call_id}); call it again next turn if it is still needed."
    )
    arguments = _arguments(item.get("arguments"))
    return CallRecord(item, arguments, [TextPart(text=text)], ONE_ACTION, "refused")


def _arguments(text: Any) -> dict[str, Any] | None:
    """Function-call arguments as a JSON object, or None."""
    try:
        value = json.loads(text) if isinstance(text, str) else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _record(turn: Turn, end: EpisodeEnd | None) -> dict[str, Any]:
    """The turn's line in `turns.jsonl`."""
    action = turn.action
    rationale = (
        None if action is None or action.arguments is None else action.arguments.get("rationale")
    )
    return {
        "turn": turn.index,
        "observation": None
        if turn.observation is None
        else [part.model_dump(mode="json") for part in turn.observation],
        "exchanges": [
            {
                "call": exchange.call,
                "text": "".join(
                    part.get("text", "")
                    for item in exchange.output
                    if item.get("type") == "message"
                    for part in item.get("content", [])
                ),
                "calls": [
                    {
                        "call_id": record.call_id,
                        "name": record.name,
                        "arguments": record.item.get("arguments")
                        if record.arguments is None
                        else record.arguments,
                        "status": record.status,
                        "error": record.error,
                        "content": [part.model_dump(mode="json") for part in record.content],
                    }
                    for record in exchange.results
                ],
                "nudge": exchange.nudge,
            }
            for exchange in turn.exchanges
        ],
        "action": None if action is None else action.name,
        "rationale": rationale,
        "end": None if end is None else end.model_dump(mode="json"),
    }
