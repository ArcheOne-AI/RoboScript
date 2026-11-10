"""The guard pipeline every tool call passes (docs/concepts/agent-runtime.md):

    record the call
     -> pre:    episode open, visibility, strict validation, alias resolution
     -> guards: action lock, budget, stage rules
     -> around: timeout, metering, then the tool
     -> post:   output validation; the episode transitions (S2 fail-fast, P3 recovery,
                finish handling); sanitizing, rendering
     -> normalization: any exception becomes an error outcome with a stable code
    record the result

The transitions run before sanitizing and rendering, so that neither can keep an episode from
ending. Steps are plain async functions over a `Call`, grouped by phase in a `Pipeline`; a step
stops a call by raising `Reject`, and a guard (a permit check, say) is added by listing it.
Cleanups registered with `Call.defer` run after the post steps, whatever happened. The built-in
steps are in `roboscript.hub.steps`.

Every step is a `Guard` of the `roboscript.guards` entry-point group (the built-in ones too,
`roboscript.hub.steps.GUARDS`): its phase, the step it makes from its config, and where an added
one goes (`after` a named step, else the end of its phase). `assemble` builds a `Pipeline` from
guards in order; `DEFAULT_PIPELINE` is the built-in guards assembled.
"""

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

from pydantic import BaseModel, ConfigDict, ValidationError

from roboscript.contracts.errors import EpisodeEnded, InfraError, ToolError
from roboscript.contracts.hub import ENDED_MESSAGE, CallOutcome
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import ContentPart, TextPart, ToolCall
from roboscript.hub.render import cut
from roboscript.hub.store import EpisodeState
from roboscript.hub.visibility import Tool
from roboscript.kernel import entrypoints
from roboscript.tools import ToolModel

if TYPE_CHECKING:
    from roboscript.hub.hub import Hub

log = logging.getLogger(__name__)


class Reject(Exception):
    """Stops a call with an error outcome: a stable `code` and a message for the model."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(eq=False)
class Call:
    """One call on its way through the pipeline; the steps read and fill it in order."""

    hub: "Hub"
    request: ToolCall
    episode_id: EpisodeId
    # Filled in by the step named, so a step may read what an earlier one set (see `reached`).
    episode: EpisodeState = field(init=False, repr=False)  # episode_open
    tool: Tool = field(init=False, repr=False)  # visibility
    params: ToolModel = field(init=False, repr=False)  # validate; ids after resolve_aliases
    result: BaseModel = field(init=False, repr=False)  # check_output; the tool's ids
    shown: BaseModel = field(init=False, repr=False)  # sanitize; aliases for ids
    raw: object = None  # what the tool returned
    returned: bool = False  # the tool returned or raised, rather than being cut off
    interrupted: bool = False  # an action was cut off midway (see `steps.timeout`)
    unmoved: bool = False  # a rejection that moved nothing, going on (see `steps.unmoved`)
    actions: int = 0  # the episode's actions admitted before this call (`steps.count_actions`)
    view: dict[str, Any] | None = None  # `shown` as JSON, for opted-in sessions and system calls
    content: list[ContentPart] = field(default_factory=list)
    forwarded: bool = False  # passed every guard
    step: str | None = None  # the step running, left set when one stops the call
    duration_s: float | None = None
    seq: int = 0  # a model call's place among its episode's model calls, as dispatched
    cleanups: list[Callable[[], object]] = field(default_factory=list)

    def reached(self, name: str) -> bool:
        """Whether the field `name` is filled in yet."""
        return name in vars(self)

    def defer(self, cleanup: Callable[[], object]) -> None:
        """Run `cleanup` once the post steps are done, or the call stopped (latest first)."""
        self.cleanups.append(cleanup)

    @property
    def ended(self) -> bool:
        """Whether the episode is over (ended or closed)."""
        if self.reached("episode"):
            return self.episode.ended is not None
        return self.hub.store.is_closed(self.episode_id)


Step = Callable[[Call], Awaitable[None]]
Proceed = Callable[[], Awaitable[None]]
Around = Callable[[Call, Proceed], Awaitable[None]]


@dataclass(frozen=True)
class Pipeline:
    pre: tuple[Step, ...]
    guards: tuple[Step, ...]
    around: tuple[Around, ...]
    post: tuple[Step, ...]


Phase = Literal["pre", "guards", "around", "post"]
PHASES: Final[tuple[Phase, ...]] = get_args(Phase)
GROUP: Final = "roboscript.guards"


class NoConfig(BaseModel):
    """The config of a guard without settings: an empty mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class Guard:
    """One step of the pipeline as a plugin: `make(config)` returns the step (an `Around` in
    phase `around`, else a `Step`), `config` an instance of `Config`. An added guard goes right
    after the step named `after` in its phase, or at the phase's end when `after` is None."""

    name: str
    phase: Phase
    make: Callable[[Any], Step | Around]
    Config: type[BaseModel] = NoConfig
    after: str | None = None


def assemble(
    guards: Sequence[Guard], configs: Mapping[str, Mapping[str, Any]] | None = None
) -> Pipeline:
    """The pipeline of `guards`, each made from its config in `configs` (by name; none: its
    defaults) and placed in order: at its phase's end, or after the step its `after` names."""
    configs = dict(configs or {})
    names = [guard.name for guard in guards]
    if twice := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"guards listed twice: {twice}")
    if unknown := sorted(set(configs) - set(names)):
        raise ValueError(f"config for guards that are not listed: {unknown}")
    phases: dict[Phase, list[tuple[str, Any]]] = {phase: [] for phase in PHASES}
    for guard in guards:
        try:
            config = guard.Config.model_validate(configs.get(guard.name, {}))
        except ValidationError as error:
            raise ValueError(f"invalid config for guard {guard.name!r}: {error}") from error
        steps = phases[guard.phase]
        at = len(steps)
        if guard.after is not None:
            found = [i for i, (name, _) in enumerate(steps) if name == guard.after]
            if not found:
                detail = f"guard {guard.name!r} goes after {guard.after!r}, no {guard.phase} step"
                raise ValueError(detail)
            at = found[0] + 1
        steps.insert(at, (guard.name, guard.make(config)))
    return Pipeline(*(tuple(step for _, step in phases[phase]) for phase in PHASES))


def load_guard(ref: str) -> Guard:
    """The guard registered as `ref` in `roboscript.guards`, or by import path
    `package.module:attribute`."""
    found = entrypoints.load_ref(ref) if ":" in ref else entrypoints.load(GROUP, ref)
    if not isinstance(found, Guard):
        where = ref if ":" in ref else f"{ref} in {GROUP}"
        raise TypeError(f"{where} is a {type(found).__name__}, not a Guard")
    return found


def fixed(step: Step | Around) -> Callable[[Any], Step | Around]:
    """`make` of a guard without config: always `step` itself."""
    return lambda _config: step


async def run(pipeline: Pipeline, call: Call) -> CallOutcome:
    """Record `call`, pass it through `pipeline` to its tool, and record the outcome."""
    hub = call.hub
    hub.record_call(call)
    try:
        outcome = await _steps(pipeline, call)
    except BaseException:  # cancelled: the caller is gone, but the log shows the call ended
        hub.record_cancelled(call)
        raise
    hub.record_outcome(call, outcome)
    return outcome


async def _steps(pipeline: Pipeline, call: Call) -> CallOutcome:
    try:
        try:
            for step in (*pipeline.pre, *pipeline.guards):
                call.step = _name(step)
                await step(call)
            call.step, call.forwarded = None, True
            await _around(pipeline.around, call)
            for step in pipeline.post:
                call.step = _name(step)
                await step(call)
            call.step = None
        finally:
            for cleanup in reversed(call.cleanups):
                cleanup()
    except Exception as error:
        code, message = _error(call, error)
        render = call.episode.render if call.reached("episode") else call.hub.config.render
        text = cut(message, render.max_text_chars)
        return CallOutcome(
            call_id=call.request.call_id,
            content=[TextPart(text=text)],
            is_error=True,
            error_code=code,
            episode_ended=call.ended,
        )
    return CallOutcome(
        call_id=call.request.call_id,
        content=call.content,
        structured=call.view,
        episode_ended=call.ended,
    )


async def _around(steps: tuple[Around, ...], call: Call) -> None:
    async def proceed(index: int) -> None:
        if index == len(steps):
            await call.hub.forward(call)
        else:
            await steps[index](call, lambda: proceed(index + 1))

    await proceed(0)


def _error(call: Call, error: Exception) -> tuple[str, str]:
    """A stable code and a sanitized message; unexpected errors go to the log in full."""
    if isinstance(error, Reject):
        return error.code, error.message
    if isinstance(error, ToolError):
        return "tool_error", error.detail or "The tool failed."
    if isinstance(error, EpisodeEnded):
        return "episode_ended", ENDED_MESSAGE
    infra = _infra(error)
    if infra is not None:
        return "infra_error", f"The tool is unavailable right now ({infra.code}); try again."
    if isinstance(error, TimeoutError) and call.reached("tool"):
        return "timeout", f"The tool did not finish within {call.tool.meta.timeout_s:g} s."
    where = call.step or "the tool"
    log.error(
        "%s failed in %s (episode %s)", call.request.name, where, call.episode_id, exc_info=error
    )
    return "internal_error", "The tool failed unexpectedly."


def _infra(error: BaseException) -> InfraError | None:
    """The infrastructure failure `error` is: itself, or, raised out of a task group (an
    exception group), the first of its failures when every one of them is one."""
    if isinstance(error, InfraError):
        return error
    inner = getattr(error, "exceptions", None)  # an exception group (a backport on 3.10)
    if not inner:
        return None
    found = [_infra(e) for e in inner]
    return found[0] if all(found) else None


def _name(step: Callable[..., Any]) -> str:
    return getattr(step, "__name__", type(step).__name__)
