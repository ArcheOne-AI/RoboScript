"""The Tool Hub (docs/concepts/agent-runtime.md): the in-process dispatcher between brains and
toolsets, and the keeper of the method's phase rules: visibility (phase x tier x role, from the
method's declaration), the information boundary of eval phases, budgets, one action at a time,
fail-fast and recovery, `finish`, aliases and the event log."""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import asdict
from types import MappingProxyType
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from typing_extensions import Self

from roboscript import rpc
from roboscript.blobs import BlobStore
from roboscript.contracts.env import EnvAgent
from roboscript.contracts.errors import EpisodeEnded, ProtocolError
from roboscript.contracts.events import EventKind
from roboscript.contracts.hub import (
    CallOutcome,
    Counters,
    DoneReport,
    EpisodeEnd,
    EpisodeSeed,
    EpisodeSession,
    EpisodeStatus,
    EpisodeSummary,
    OpenEpisode,
)
from roboscript.contracts.ids import EpisodeId, ToolName, new_id
from roboscript.contracts.phase import PhaseRules, PhaseSpec
from roboscript.contracts.tools import (
    CallContext,
    EnvBinding,
    TextPart,
    ToolCall,
    ToolDescriptor,
)
from roboscript.contracts.trace import TraceContext, get_trace, new_trace, use_trace
from roboscript.hub.builtins import HubTools
from roboscript.hub.config import HubConfig
from roboscript.hub.pipeline import Call, Pipeline, run
from roboscript.hub.steps import DEFAULT_PIPELINE, plain_json
from roboscript.hub.store import ActionNotes, Aliases, EpisodeState, EpisodeStore, FinishGuard
from roboscript.hub.visibility import Phases, Tool, absent, mount
from roboscript.records import EventWriter, replay
from roboscript.tools import FrameStore, LocalProvider, ToolContext, ToolProvider, ToolsetRuntime

log = logging.getLogger(__name__)


class Hub:
    """Mounts toolsets and runs episodes over them; `async with` starts and stops the toolsets.

    `providers` are mounted in order, then the Hub's own `finish` and `scratchpad`; `phases`
    are the method's (`contracts.phase.PhaseSpec`): which roles of which phase see which tools,
    in what order, at which tier, under which default rules. `pipeline` defaults to the
    built-in steps. `env_client` makes an episode's env client
    from its binding (default: a typed RPC client); `models` is the model-client factory
    toolsets get (e.g. `roboscript.gateway.client.openai_client`); `clock` times budgets.
    """

    def __init__(
        self,
        providers: Sequence[ToolProvider],
        *,
        blobs: BlobStore,
        phases: Sequence[PhaseSpec],
        config: HubConfig | None = None,
        events: EventWriter | None = None,
        pipeline: Pipeline = DEFAULT_PIPELINE,
        env_client: Callable[[EnvBinding], EnvAgent] | None = None,
        models: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = HubConfig() if config is None else config
        self.blobs = blobs
        self.events = events
        self.pipeline = pipeline
        self.clock = clock
        self.store = EpisodeStore()
        self._providers = [*providers, LocalProvider(HubTools())]
        self._tools = mount(self._providers)
        self.phases = Phases(phases, self._tools, absent(self._providers))
        self._env_client = self._rpc_env_client if env_client is None else env_client
        self._models = _no_models if models is None else models
        self._http: httpx.AsyncClient | None = None
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> Self:
        async with AsyncExitStack() as stack:
            self._http = await stack.enter_async_context(httpx.AsyncClient())
            for provider in self._providers:
                await provider.start(self._runtime(provider))
                stack.push_async_callback(provider.stop)
            self._stack = stack.pop_all()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()

    # The API (contracts.hub.ToolHub)

    async def open_episode(self, spec: OpenEpisode) -> EpisodeSession:
        """Fix the episode's tools (its phase's list for its role, in that order, tier-P
        ones only at tier P, then the toolset profile; the system-only ones apart), its
        rules (the phase's, with the fields `spec.rules` gives replaced) and render limits
        (the role's, then `spec.render`), and start its frame store and alias table (with
        `spec.seed`'s frames, facts and aliases). Every new alias is logged (`hub.alias`)."""
        if self._stack is None:
            raise RuntimeError("the Hub is not started: use `async with hub:`")
        phase = self.phases.phase(spec.stage)
        self.phases.check_open(phase, spec.role, spec.tier)
        mounted = [provider.name for provider in self._providers]
        if unknown := sorted(set(spec.toolsets or ()) - set(mounted)):
            raise ProtocolError(
                "invalid_request", f"unknown toolsets {unknown}; mounted: {mounted}"
            )
        rules = _rules(phase.rules, spec)
        if unknown := sorted(set(rules.max_calls) - set(self._tools)):
            raise ProtocolError("invalid_request", f"max_calls names unknown tools {unknown}")
        check = rules.done_check
        if check is not None and (
            check not in self._tools or not issubclass(self._tools[check].spec.result, DoneReport)
        ):
            raise ProtocolError("invalid_request", f"done_check {check!r} is no done-check tool")
        profile = None if spec.toolsets is None else {*spec.toolsets, HubTools.name}
        visible = [
            (name, tool)
            for name in self.phases.tools(phase.name, spec.role)
            if ((tool := self._tools[name]).meta.tier == "D" or spec.tier == "P")
            and (profile is None or tool.meta.toolset in profile)
        ]
        if phase.holds_d and (shown := [n for n, t in visible if t.meta.tier != "D"]):
            raise ProtocolError(  # cannot happen at tier D; the invariant says it never may
                "invalid_request", f"{phase.held} {phase.name!r} would show tier-P tools {shown}"
            )
        tools = tuple(name for name, tool in visible if not tool.meta.system_only)
        system_tools = tuple(name for name, tool in visible if tool.meta.system_only)
        if check is not None and check not in tools:  # else no finish could ever pass
            raise ProtocolError(
                "invalid_request", f"done_check {check!r} is not among the episode's tools"
            )
        parent = get_trace()
        episode_id = EpisodeId(new_id("ep"))

        def made(kind: str, value: str, alias: str) -> None:
            data = {"kind": kind, "value": value, "alias": alias}
            self._emit(EventKind.HUB_ALIAS, data, episode_id)

        aliases = Aliases(made)
        seeded = [] if spec.seed is None else spec.seed.frames
        try:
            for entry in seeded:
                aliases.seed("frame", entry.frame_id, entry.alias)
            for given in [] if spec.seed is None else spec.seed.aliases:
                aliases.seed(given.kind, given.value, given.alias)
        except ValueError as error:
            raise ProtocolError("invalid_request", f"seed: {error}") from None
        frames = FrameStore(
            self.blobs,
            alias=lambda frame_id: aliases.alias("frame", frame_id),
            emit=lambda kind, data: self._emit(kind, data, episode_id),
            seed=spec.seed,
            turn=lambda: self.store.get(episode_id).counters.turns + 1,
        )
        episode = EpisodeState(
            id=episode_id,
            stage=spec.stage,
            tier=spec.tier,
            role=spec.role,
            tools=tools,
            system_tools=system_tools,
            budgets=spec.budgets,
            rules=rules,
            started=self.clock(),
            trace=new_trace() if parent is None else parent.child(),
            aliases=aliases,
            frames=frames,
            render=self.config.render.merged(self.config.roles.get(spec.role), spec.render),
            env=spec.env,
            env_client=None if spec.env is None else self._taped_env(spec.env),
            embodiment=spec.embodiment,
            structured=spec.structured,
        )
        self.store.add(episode)
        opened = spec.model_dump(mode="json")  # the binding's token comes out masked
        opened["rules"] = rules.model_dump(mode="json")
        opened["tools"] = {name: self._tools[name].descriptor.hash for name in tools}
        opened["system_tools"] = {name: self._tools[name].descriptor.hash for name in system_tools}
        self._emit(EventKind.EPISODE_OPEN, opened, episode.id, episode.trace)
        if (tape := replay.active()) is not None:
            keys = ("stage", "tier", "role", "tools", "system_tools", "rules")
            tape.observe("episode", {key: opened[key] for key in keys})
        return EpisodeSession(
            episode_id=episode.id,
            stage=episode.stage,
            tier=episode.tier,
            role=episode.role,
            tools=self._descriptors(episode),
            system_tools=list(system_tools),
            budgets=episode.budgets,
            rules=episode.rules,
        )

    async def call(self, episode_id: EpisodeId, call: ToolCall) -> CallOutcome:
        """Run one model call through the pipeline; problems come back as error outcomes. The
        call is a model's whatever its `origin` says: only `system_call` makes system calls."""
        if call.origin != "model":
            call = call.model_copy(update={"origin": "model"})
        return await self._dispatch(episode_id, call)

    async def system_call(
        self, episode_id: EpisodeId, name: ToolName, arguments: dict[str, Any]
    ) -> CallOutcome:
        """A call by the runtime rather than the model (the per-turn observation, say). The
        episode's tools, the action lock, the wall clock and the rules on receipts apply; the
        model's turn, information-call and per-tool limits do not."""
        request = ToolCall(call_id=new_id("sys"), name=name, arguments=arguments, origin="system")
        return await self._dispatch(episode_id, request)

    async def tools(self, episode_id: EpisodeId) -> list[ToolDescriptor]:
        return self._descriptors(self.store.get(episode_id))

    async def end_episode(
        self, episode_id: EpisodeId, reason: str, note: str = ""
    ) -> EpisodeStatus:
        """End an episode (a no-op if it has ended); later calls get the ended error."""
        episode = self.store.get(episode_id)
        self.end(episode, reason, note)
        return self._status(episode)

    async def credit_clock(self, episode_id: EpisodeId, seconds: float, reason: str) -> None:
        """Give `seconds` back to the episode's wall clock: time the brain spent waiting on
        infrastructure (a model call's retry waits, its queue at the gateway), which is not the
        role's to spend. An `episode.clock` event says how much and why."""
        episode = self.store.get(episode_id)
        if seconds <= 0 or episode.ended is not None:
            return
        episode.started += seconds
        data = {"reason": reason, "credited_s": round(seconds, 3)}
        self._emit(EventKind.EPISODE_CLOCK, data, episode.id)

    async def close_episode(self, episode_id: EpisodeId) -> EpisodeSummary:
        """End (reason `closed` unless it has ended) and forget an episode, its aliases and its
        toolset state, and tell every provider (`ToolProvider.close_episode`); a provider that
        fails to hear it does not keep the episode open."""
        episode = self._expire(self.store.get(episode_id))
        end = self.end(episode, "closed")
        self.store.close(episode_id)
        for provider in self._providers:
            try:
                await provider.close_episode(episode_id)
            except Exception:
                log.exception("toolset %s failed to close episode %s", provider.name, episode_id)
        summary = EpisodeSummary(
            episode_id=episode.id,
            end=end,
            counters=Counters(**asdict(episode.counters)),
            elapsed_s=self.elapsed(episode),
        )
        close = summary.model_dump(mode="json")
        self._emit(EventKind.EPISODE_CLOSE, close, episode.id, episode.trace)
        return summary

    async def state(self, episode_id: EpisodeId) -> EpisodeStatus:
        return self._status(self._expire(self.store.get(episode_id)))

    # For the runner

    def set_finish_guard(
        self, episode_id: EpisodeId, guard: FinishGuard | None, times: int = 1
    ) -> None:
        """Have `guard` look at the episode's `finish` calls: each message it returns refuses that
        `finish` (code `finish_refused`), up to `times` refusals; the `finish` after those goes
        through."""
        if times < 1:
            raise ValueError(f"a finish guard refuses at least once, not {times} times")
        state = self.store.get(episode_id)
        state.finish_guard, state.finish_guard_times = guard, times

    def set_action_notes(self, episode_id: EpisodeId, notes: ActionNotes | None) -> None:
        """Have `notes` add content after each physical action's result (hinted S1: the keypoint
        progress); it is part of the result, so the event log keeps what the model saw."""
        self.store.get(episode_id).action_notes = notes

    def restart_clock(self, episode_id: EpisodeId, reason: str) -> None:
        """Start the episode's wall clock again, after the runtime's calls at its open (the
        one-time identification), so that they are not charged to the role's budget."""
        episode = self.store.get(episode_id)
        spent = self.elapsed(episode)
        episode.started = self.clock()
        self._emit(EventKind.EPISODE_CLOCK, {"reason": reason, "spent_s": spent}, episode.id)

    def note(self, episode_id: EpisodeId, kind: str, data: dict[str, Any]) -> None:
        """Log an event of the episode on the runtime's behalf (an identification reused)."""
        self._emit(kind, data, self.store.get(episode_id).id)

    def unanswered_infra(self, episode_id: EpisodeId) -> str | None:
        """What the episode's last model call that failed for infrastructure reasons failed with,
        if no model call after it but `finish` was answered without an error: its runner then
        voids the episode, unless the task's success latched (docs/concepts/learning.md)."""
        return self.store.get(episode_id).unanswered_infra

    def snapshot(self, episode_id: EpisodeId) -> EpisodeSeed:
        """The episode's frames, facts and aliases now: an episode opened with them as its seed
        starts where this one stands (a later try of an S2 seed, from its first try's
        identification), with no live frame: those frames are another episode's."""
        episode = self.store.get(episode_id)
        seed = episode.frames.snapshot()
        aliases = episode.aliases.items(exclude=("frame",))
        return seed.model_copy(update={"aliases": aliases, "live": None})

    def rebind(
        self,
        episode_id: EpisodeId,
        rules: PhaseRules | None,
        facts: Mapping[str, Mapping[str, Any]],
        drop: Sequence[str] = (),
    ) -> None:
        """Bind an open episode to other rules and facts, as an episode would have opened with
        them (`OpenEpisode.rules` over the phase's, `EpisodeSeed.facts`): what a P3 Actor's
        choice of Memories in use changes (`memory.library`). `drop` names facts to remove. A
        done check must be one of the episode's tools, as at open. Logged as `episode.rebind`."""
        episode = self.store.get(episode_id)
        phase = self.phases.phase(episode.stage)
        given = {} if rules is None else {n: getattr(rules, n) for n in rules.model_fields_set}
        bound = phase.rules.model_copy(update=given)
        check = bound.done_check
        if check is not None and check not in episode.tools:
            raise ProtocolError(
                "invalid_request", f"done_check {check!r} is not among the episode's tools"
            )
        episode.rules = bound
        for name in drop:
            episode.frames.drop_fact(name)
        for name, value in facts.items():
            episode.frames.set_fact(name, value)
        data = {"rules": bound.model_dump(mode="json"), "facts": sorted(facts), "drop": list(drop)}
        self._emit(EventKind.EPISODE_REBIND, data, episode.id)

    def progress(self, episode_id: EpisodeId) -> tuple[int, int, float]:
        """Where the episode stands: env steps used, hard failures of its physical calls so far
        (`retry_guard`'s count) and seconds on its clock."""
        episode = self.store.get(episode_id)
        failures = sum(episode.retry.failures.values())
        return episode.counters.steps, failures, self.elapsed(episode)

    def fact(self, episode_id: EpisodeId, name: str) -> dict[str, Any] | None:
        """One of the episode's scene facts now (None: not set)."""
        return self.store.get(episode_id).frames.fact(name)

    def put_fact(
        self, episode_id: EpisodeId, name: str, value: Mapping[str, Any] | None, reason: str
    ) -> None:
        """Set one of the episode's scene facts (None: remove it), logged as `episode.rebind`
        with `reason`: a P3 Actor's failed `library_use` puts `IDENTIFIED` back as it was."""
        episode = self.store.get(episode_id)
        if value is None:
            episode.frames.drop_fact(name)
        else:
            episode.frames.set_fact(name, value)
        given = value is not None
        data = {"facts": [name] if given else [], "drop": [] if given else [name]}
        self._emit(EventKind.EPISODE_REBIND, {**data, "reason": reason}, episode.id)

    # For the pipeline steps

    def tool(self, name: ToolName) -> Tool:
        return self._tools[name]

    @property
    def mounted(self) -> Mapping[ToolName, Tool]:
        """Every mounted tool by name, in mount order (the tool lock hashes them)."""
        return MappingProxyType(self._tools)

    def elapsed(self, episode: EpisodeState) -> float:
        return self.clock() - episode.started

    def remaining(self, episode: EpisodeState) -> float | None:
        """Seconds left on the episode's wall clock (negative once over), None without one or
        once the episode has ended (the runtime's readings after the end are not the role's)."""
        limit = episode.budgets.wall_clock_s
        if limit is None or episode.ended is not None:
            return None
        return limit - self.elapsed(episode)

    def end(self, episode: EpisodeState, reason: str, note: str = "") -> EpisodeEnd:
        """End `episode` unless it has ended already (the first reason stands); its end."""
        if episode.ended is None:
            episode.ended = EpisodeEnd(reason=reason, note=note, at=time.time())
            data = {"reason": reason, "note": note}
            self._emit(EventKind.EPISODE_END, data, episode.id, get_trace() or episode.trace)
        return episode.ended

    async def forward(self, call: Call) -> None:
        """Hand the call to its toolset (the end of the around steps)."""
        tool, episode = call.tool, call.episode
        ctx = self._context(episode, tool, call.request.call_id)
        try:
            call.raw = await tool.provider.call(tool.spec.name, call.params, ctx)
        except Exception as error:  # a cancelled call raises no Exception: it did not return
            call.returned = True
            if isinstance(error, EpisodeEnded):  # the env's episode is over, so this one is too
                self.end(episode, "env_ended", note=error.detail or "")
            raise
        call.returned = True

    def record_call(self, call: Call) -> None:
        """`tool.call`; arguments that are not plain JSON (`validate` refuses them) are logged
        as null, since the log cannot hold them."""
        request = call.request
        data = {"call_id": request.call_id, "name": request.name, "origin": request.origin}
        arguments = request.arguments if plain_json(request.arguments) else None
        self._emit(EventKind.TOOL_CALL, {**data, "arguments": arguments}, call.episode_id)

    def record_outcome(self, call: Call, outcome: CallOutcome) -> None:
        """`tool.result` for calls that reached their tool, `tool.reject` for the others."""
        if call.reached("episode") and outcome.is_error:
            if call.forwarded:
                call.episode.counters.errors += 1
            else:
                call.episode.counters.rejected += 1
        later = call.reached("episode") and call.seq > call.episode.unanswered_seq
        if later and call.request.origin == "model":  # by dispatch: parallel calls end in any order
            episode = call.episode
            if outcome.error_code == "infra_error":
                said = " ".join(part.text for part in outcome.content if isinstance(part, TextPart))
                episode.unanswered_infra = f"{call.request.name}: {said or 'infra_error'}"
                episode.unanswered_seq = call.seq
            elif call.forwarded and not outcome.is_error and call.tool.meta.kind != "control":
                episode.unanswered_infra = None  # answered: the infrastructure is back
                episode.unanswered_seq = call.seq
        data: dict[str, Any] = {
            "call_id": call.request.call_id,
            "name": call.request.name,
            "origin": call.request.origin,
            "is_error": outcome.is_error,
            "code": outcome.error_code,
            "content": [part.model_dump(mode="json") for part in outcome.content],
        }
        if call.step is not None:
            data["step"] = call.step
        if outcome.structured is not None:
            data["structured"] = outcome.structured
        if call.reached("result"):
            data["result"] = call.result.model_dump(mode="json")
        if call.duration_s is not None:
            data["duration_s"] = call.duration_s
        kind = EventKind.TOOL_RESULT if call.forwarded else EventKind.TOOL_REJECT
        self._emit(kind, data, call.episode_id)
        if (tape := replay.active()) is not None:  # what the model got back
            shown = ("name", "origin", "is_error", "code", "content")
            tape.observe("tool", {key: data[key] for key in shown if key in data})

    def record_cancelled(self, call: Call) -> None:
        request = call.request
        data: dict[str, Any] = {"call_id": request.call_id, "name": request.name}
        data |= {"origin": request.origin, "is_error": True, "code": "cancelled", "step": call.step}
        kind = EventKind.TOOL_RESULT if call.forwarded else EventKind.TOOL_REJECT
        self._emit(kind, data, call.episode_id)

    # Internals

    async def _dispatch(self, episode_id: EpisodeId, request: ToolCall) -> CallOutcome:
        episode = self.store.find(episode_id)
        call = Call(self, request, episode_id)
        if episode is not None and request.origin == "model":
            episode.dispatched += 1
            call.seq = episode.dispatched
        with use_trace(new_trace() if episode is None else episode.trace.child()):
            return await run(self.pipeline, call)

    def _expire(self, episode: EpisodeState) -> EpisodeState:
        """End the episode if its wall clock has run out, even without a call noticing."""
        remaining = self.remaining(episode)
        if remaining is not None and remaining < 0:
            self.end(episode, "wall_clock")
        return episode

    def _descriptors(self, episode: EpisodeState) -> list[ToolDescriptor]:
        found = [self._tools[name].descriptor for name in episode.tools]
        if episode.structured:
            return found
        return [d.model_copy(update={"output_schema": None}) for d in found]

    def _status(self, episode: EpisodeState) -> EpisodeStatus:
        return EpisodeStatus(
            episode_id=episode.id,
            stage=episode.stage,
            tier=episode.tier,
            role=episode.role,
            tools=list(episode.tools),
            budgets=episode.budgets,
            rules=episode.rules,
            counters=Counters(**asdict(episode.counters)),
            elapsed_s=self.elapsed(episode),
            ended=episode.ended,
            acting=episode.action is not None,
        )

    def _runtime(self, provider: ToolProvider) -> ToolsetRuntime:
        return ToolsetRuntime(
            config=provider.config,
            blobs=self.blobs,
            env_client=self._env_client,
            models=self._models,
            call=self._service_call,
            events=self.events,
        )

    def _context(self, episode: EpisodeState, tool: Tool, call_id: str) -> ToolContext:
        context = CallContext(
            episode_ref=episode.id,
            stage=episode.stage,
            tier=episode.tier,
            role=episode.role,
            env=episode.env,
            embodiment=episode.embodiment,
            trace=get_trace(),
        )
        return ToolContext(
            context=context,
            toolset=tool.meta.toolset,
            call_id=call_id,
            env_client=episode.env_client,
            blobs=self.blobs,
            state=episode.toolset_state.setdefault(tool.meta.toolset, {}),
            frames=episode.frames,
            service_call=self._service_call,
            event_sink=self._toolset_event,
        )

    async def service_call(
        self, episode_id: EpisodeId, name: str, args: Mapping[str, Any], parent: str
    ) -> BaseModel:
        """A toolset calling another toolset's tool for an episode, on behalf of its call
        `parent`: no pipeline (no budgets, lock or aliases: pass ids), but the episode's tier
        holds and both ends are in the event log. `ToolContext.call` comes here; remote toolset
        servers reach it through the MCP endpoint with a service token."""
        tool = self._tools.get(ToolName(name))
        if tool is None:
            raise LookupError(f"no tool {name!r} is mounted")
        episode = self.store.get(episode_id)
        if tool.meta.tier == "P" and (phase := self.phases.phase(episode.stage)).holds_d:
            raise PermissionError(
                f"{tool.qualified} is a tier-P tool; phase {episode.stage!r} is "
                f"{'an eval' if phase.kind == 'eval' else 'a D-only'} phase"
            )
        if tool.meta.tier == "P" and episode.tier == "D":
            raise PermissionError(f"{tool.qualified} is a tier-P tool; the episode runs at tier D")
        call_id = new_id("svc")
        request = {"call_id": call_id, "name": name, "origin": "service", "parent": parent}
        with use_trace((get_trace() or episode.trace).child()):
            arguments = to_jsonable_python(args)
            self._emit(EventKind.TOOL_CALL, {**request, "arguments": arguments}, episode.id)
            try:
                ctx = self._context(episode, tool, call_id)
                result = await tool.provider.call(tool.spec.name, args, ctx)
                if not isinstance(result, tool.spec.result):
                    result = tool.spec.result.model_validate(result)
            except Exception as error:
                failed = {**request, "is_error": True, "error": type(error).__name__}
                self._emit(EventKind.TOOL_RESULT, failed, episode.id)
                raise
            done = {**request, "is_error": False, "result": result.model_dump(mode="json")}
            self._emit(EventKind.TOOL_RESULT, done, episode.id)
        return result

    async def _service_call(
        self, parent: ToolContext, name: str, args: Mapping[str, Any]
    ) -> BaseModel:
        return await self.service_call(parent.episode_id, name, args, parent.call_id)

    def _toolset_event(self, ctx: ToolContext, kind: str, data: dict[str, Any]) -> None:
        self._emit(f"{ctx.toolset}.{kind}", data, ctx.episode_id)

    def _taped_env(self, binding: EnvBinding) -> EnvAgent:
        """The episode's env client, its calls in the replay tape's comparison stream when one
        is on (the joint waypoints sent, the states and latch the env answered)."""
        return rpc.observed(EnvAgent, self._env_client(binding))  # type: ignore[type-abstract]

    def _rpc_env_client(self, binding: EnvBinding) -> EnvAgent:
        return rpc.client(
            EnvAgent,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
            binding.url,
            token=binding.token,
            http=self._http,
            timeout=self.config.env_timeout_s,
        )

    def _emit(
        self,
        kind: str,
        data: dict[str, Any],
        episode_id: EpisodeId,
        trace: TraceContext | None = None,
    ) -> None:
        """Log an event; `trace` defaults to the current one (a call's span)."""
        if self.events is not None:
            self.events.emit(kind, data, episode_id=episode_id, trace=trace)


def _rules(defaults: PhaseRules, spec: OpenEpisode) -> PhaseRules:
    """The phase's rules with the fields the runner gave replaced."""
    if spec.rules is None:
        return defaults
    given = {name: getattr(spec.rules, name) for name in spec.rules.model_fields_set}
    return defaults.model_copy(update=given)


def _no_models(*args: Any, **kwargs: Any) -> Any:
    raise LookupError("this Hub has no model-client factory: pass `models=` when creating it")
