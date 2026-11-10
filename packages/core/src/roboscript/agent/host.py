"""The Agent host (docs/concepts/learning.md, docs/concepts/operations.md): the agent's side of
the episodes that a benchmark's own evaluation client drives (the evaluation phase), served as the
`agent.host` plane that the benchmark's policy module calls (`roboscript.contracts.host.AgentHost`).

`start()` warms it up (`ready` then turns true): it loads the frozen Memory version the manifest
names for each task, by reference (`memory.artifact.manifest_stack`), through the loader it was
given (`MemoryLoader`, the method's), which checks it against the store. Given an artifact stack
(`stack`, as a method composed it, e.g. with documents of its own), it loads each task's other
layers through its `OverlayLoader` too, and the Actor's prompt is the method's with that overlay
around it (`agent.overlay`; `config.json` and the `p3.reset` event name its layers); without one,
the prompt and records are the method's alone. A `reset` prepares a
slot's next episode: its Memory and a `BridgeEnv`. The first `submit` opens the Hub episode
(the method's phase for the host, tier D, the Actor) bound to the bridge and starts a brain on the
Actor's prompt. The host's episodes recover from errors: a hard failure
does not end the episode (the Hub tells the agent to move to a safe pose and try again), and the
benchmark's step limit is its only limit; the wall clock and the model calls are bounded only as
runaway guards, derived from that step limit (`Guards`). An opening hook, when the host has one
(the one-time identification, `roboscript.learning.opening`), runs in the brain's task before it:
its outcome goes into the Actor's brief, and a failure ends the episode without a brain. With a
task's `DemoView` (the demonstration at a `demo_context` level), its content leads the Actor's
brief, its frames seed the Hub episode, and its label goes into `config.json`. A `finish`
while the benchmark has not ended the episode (its success latch has not fired: the client ends
an episode only there or at the step limit) is refused up to `HostConfig.finish_refusals` times,
with the demonstration's latch hint (`OpaqueGoal.hint`, of `DemoView.goal`); the
`finish` after those goes through. A finish check, when the host has one, then refuses a first
`finish` while the final workflow stage's predicates fail. `poll` hands the client the batch
the brain's action queued, `report` completes it, `end` records the benchmark's outcome.
Records go to `<records>/<key>/ep<NNN>/`: `llm_calls.jsonl` and `turns.jsonl` (the recorder),
`events.jsonl` (the episode's events), `config.json`, `memory_before.md`, `summary.json` (with
`infra`: whether infrastructure failed the episode, apart from the agent's own failures: such an
end, a failed call of the client, or an end after a model call that failed for infrastructure
reasons with none answered after it, which ends as `infra_error` (`Hub.unanswered_infra`; the end
it had kept as `voided_end`, the failed call as `voided_by`) unless the success latched, even
while the client held still after a `finish`: the client's outcome restores the end then; and
`guard_tripped`: whether a runaway guard ended it, which is the agent's outcome; `provenance`: the
code and behaviour config that ran it, `HostConfig.provenance`).

With a `HostLibrary` (the host's library mode, the default of `learning.agent_host`), the Actor's
prompt holds no Memory of its task id: the `library` toolset gives it the library's entries (less
the excluded ones) for the whole episode, and `library_use` changes the set of Memories in use,
which the host binds as an episode bound to them would have opened (`_bind`: their roles
identified by the opening hook, their specs, arm sides and done criteria applied with
`Hub.rebind`); the summary's `library` lists what the Actor read and used, when, and whether
before or after a hard failure, and `library.json` replaces `memory_before.md`. Without one,
the episode stays bound to its task's Memory.

An episode that ends on a configuration error (`CONFIG_ENDS`: say, `config_error:identify`)
fails the run: every episode would end alike. The host writes `<records>/config_error.json`
(the episode and its end), which a launcher watches for to stop the benchmark's client, and ends
each later episode at once with that end, without a Hub episode or a model call.

`roboscript.learning.agent_host` composes the host process (Hub, toolsets, gateway), with the
brains and the Actor's prompt made as the learning runner makes them (`roboscript.learning`).
"""

import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol

import anyio
import anyio.abc
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, field_serializer
from typing_extensions import Self

from roboscript.agent.overlay import NO_OVERLAY, Overlay, OverlayLoader
from roboscript.agent.recording import Recorder
from roboscript.agent.session import (
    CONFIG_ENDS,
    AgentSession,
    Brain,
    BrainResult,
    Limits,
    ModelClients,
    unanswered_end,
)
from roboscript.blobs import BlobStore
from roboscript.contracts.action import ExecuteResult
from roboscript.contracts.artifact import ArtifactLayer, ArtifactStack
from roboscript.contracts.errors import InfraError, ProtocolError, ToolError
from roboscript.contracts.events import EventKind
from roboscript.contracts.host import INFRA_ENDS, EpisodeKey, OfficialOutcome, Poll
from roboscript.contracts.hub import (
    Budgets,
    Counters,
    EpisodeEnd,
    EpisodeSeed,
    EpisodeSession,
    OpenEpisode,
    ToolHub,
)
from roboscript.contracts.ids import EpisodeId, ManifestId, PhaseName, TaskName
from roboscript.contracts.observation import Observation
from roboscript.contracts.phase import PhaseRules
from roboscript.contracts.provenance import Provenance
from roboscript.contracts.tools import ContentPart, EnvBinding, TextPart
from roboscript.envs.bridge import BridgeConfig, BridgeEnv, Bridges
from roboscript.envs.cards import TaskCard, shown, task_text
from roboscript.memory import (
    MemoryDoc,
    MemoryManifest,
    Version,
    WorkflowStage,
    view,
)
from roboscript.memory.artifact import KIND, manifest_stack
from roboscript.memory.library import (
    EpisodeLibrary,
    LibraryEntry,
    Progress,
    combine,
    role_clashes,
)
from roboscript.records import EventWriter, read_events, replay, write_atomic
from roboscript.settings import source_sha
from roboscript.tools.frames import ARM_SIDES, DONE, IDENTIFIED, MEMORY_SPECS
from roboscript.tools.memory_facts import spec_seed

log = logging.getLogger(__name__)

CANCEL_WAIT_S: Final = 10.0
"""How long a `reset` waits for the slot's previous episode to record itself after cancelling it."""

MEMORY_USED: Final = "memory_used.md"
"""Library mode: the Memories in use as the episode ended, in the place of `memory_before.md`."""

CONFIG_ERROR: Final = "config_error.json"
"""In the records directory once an episode ended on a configuration error: the run has failed."""

GUARD_ENDS: Final = frozenset({"wall_clock", "max_model_calls", "stalled"})
"""The runaway guards' ends: the agent's own outcome (an agent that loops), flagged in the summary
(`guard_tripped`), never infrastructure; the official result stands."""

ActorPrompt = Callable[[str, MemoryDoc], str]
"""The Actor's system prompt for an instruction and the Memory it reads."""

MemoryLoader = Callable[[TaskName, Sequence[ArtifactLayer]], Awaitable[Version]]
"""The method's loader: the Memory version a task's Actor reads, from the artifact layers the
manifest names for it (loaded by reference, checked against their store)."""

FinishCheck = Callable[[EpisodeId, WorkflowStage], Awaitable[str | None]]
"""Checks the final workflow stage's predicates in the episode's scene: a message with the
measured values refuses a first `finish`, None accepts it."""


NO_DEMO: Final = "none"
"""The demonstration level (`demo_context`) at which an Actor reads nothing of it (the clean
one-shot), which every method accepts and which needs no demo package; any other level is a name
the method declares (`learning.runtime.DemoBriefs.levels`) and shows its own way."""


class OpaqueGoal(Protocol):
    """The demonstration's end as a method reads it (`learning.runtime.GoalReader`; wicl's
    `goal.DemoGoal`), which the core passes to the method's finish gate without reading it
    (`DemoView.goal`, `learning.episode.Episode.goal`), but for `hint`: which way to keep moving
    at a finish short of the success latch, what the host's latch refusals add ("": nothing)."""

    @property
    def hint(self) -> str: ...


@dataclass(frozen=True)
class DemoView:
    """What a task's Actor reads of its demonstration (as the method shows it): content
    for its brief, frames for its episode, the label `config.json` records, and its end as the
    method reads it (`goal`; None: none), which the method's finish gate compares the scene with
    and whose `hint` the latch refusals add."""

    brief: list[ContentPart]
    seed: EpisodeSeed | None
    label: dict[str, str]
    goal: OpaqueGoal | None = None


@dataclass(frozen=True)
class Opened:
    """What the runtime's calls as an episode opens left it: content for the Actor's brief, or
    the end they call for (`end`, a reason, and `note`: a failure; the brain does not run)."""

    brief: list[ContentPart] = field(default_factory=list)
    end: str | None = None
    note: str = ""


OpeningHook = Callable[[EpisodeSession, str, MemoryDoc], Awaitable[Opened]]
"""Runs as an episode opens (its Hub episode, instruction and Memory), before the brain: the
one-time identification (`roboscript.learning.opening`)."""

IdentifyHook = Callable[[EpisodeSession, str, MemoryDoc], Awaitable[Opened]]
"""Identifies a Memory's roles in an episode that runs (library mode: a Memory the Actor adds,
`learning.opening.Opener.identify`): a failure is only reported (`Opened.end`: how it would have
ended the episode at its open, an infrastructure or a configuration end); it never ends the
episode, and the host credits its time back to the episode's clock."""


FinishRules = Callable[
    [MemoryDoc, PhaseRules | None, "DemoView | None", str, bool],
    tuple[PhaseRules | None, EpisodeSeed | None],
]
"""An episode's rules and seed as its finish is gated, from the frozen Memory, the host's rules,
the task's demonstration, the live instruction and whether base targets stay positional: the
method's finish gate (`learning.runtime.FinishGate.host`)."""


def ungated(
    doc: MemoryDoc,
    rules: PhaseRules | None,
    demo: "DemoView | None",
    instruction: str,
    positional: bool,
) -> tuple[PhaseRules | None, EpisodeSeed | None]:
    """No finish gate: the rules as given, the demonstration's seed."""
    return rules, None if demo is None else demo.seed


class RunnerHub(ToolHub, Protocol):
    """The Hub as a runner uses it (`roboscript.hub.Hub`): the API plus finish guards, and what
    binding a host Actor's Memories in use mid-episode needs (`rebind`, `fact`, `progress`)."""

    def set_finish_guard(
        self,
        episode_id: EpisodeId,
        guard: Callable[[str], Awaitable[str | None]] | None,
        times: int = 1,
    ) -> None: ...

    def rebind(
        self,
        episode_id: EpisodeId,
        rules: PhaseRules | None,
        facts: Mapping[str, Mapping[str, Any]],
        drop: Sequence[str] = (),
    ) -> None: ...

    def fact(self, episode_id: EpisodeId, name: str) -> dict[str, Any] | None: ...

    def progress(self, episode_id: EpisodeId) -> tuple[int, int, float]: ...

    def put_fact(
        self, episode_id: EpisodeId, name: str, value: Mapping[str, Any] | None, reason: str
    ) -> None: ...


class LibraryMount(Protocol):
    """The `library` toolset as the host attaches each episode's library to it
    (`toolsets.library.LibraryTools`)."""

    def attach(self, episode_id: EpisodeId, library: EpisodeLibrary) -> None: ...

    def detach(self, episode_id: EpisodeId) -> EpisodeLibrary | None: ...


BOUND_FACTS: Final = (DONE, MEMORY_SPECS, ARM_SIDES)
"""The facts the Memories in use set; a set without them drops them."""


@dataclass(frozen=True)
class HostLibrary:
    """The host's library mode (`memory.library`): the Actor gets no Task Memory of its task id
    but the library's `entries` (less the tasks in `exclude`, and with `exclude_own` its own
    task's), through the `library` toolset (`mount`), and chooses the Memories it uses; `prompt` is
    the Actor's prompt for an instruction without a Memory (the method's,
    `PromptSet.library_actor`), `digest` its template's version."""

    entries: Sequence[LibraryEntry]
    mount: LibraryMount
    prompt: Callable[[str], str]
    exclude: frozenset[str] = frozenset()
    exclude_own: bool = False
    digest: str = ""

    def seen(self, task: str) -> tuple[list[LibraryEntry], list[str]]:
        """The entries an episode of `task` sees and the tasks left out, in library order."""
        out = set(self.exclude) | ({task} if self.exclude_own else set())
        seen = [e for e in self.entries if e.task not in out]
        return seen, sorted(e.task for e in self.entries if e.task in out)

    def withholds(self, task: str) -> bool:
        """Whether an episode of `task` is a generalization test of its own Memory (it is left
        out: `exclude_own`, or `task` in `exclude`): then the host also withholds what reaches the
        Actor by task id, its demonstration (`AgentHostService._demo`)."""
        return self.exclude_own or task in self.exclude


class Guards(BaseModel):
    """Runaway guards of a host episode, not protocol limits: its wall clock and its model calls,
    per action of the task's step limit, so that they bind only when the agent stops spending
    steps (say, calls that fail before anything moves); the step limit binds first otherwise.
    Tripping one is the agent's outcome, not infrastructure (`GUARD_ENDS`).

    `credit_model_time` (on): the wall clock leaves out the model calls' own latency, as it
    leaves out the gateway queue, so that a slow model (a loaded gateway) cannot end an episode
    whose agent still acts: the clock bounds the time spent between calls, and the model calls
    are bounded by their count. `cap_s_per_step` is then the absolute cap (nothing credited back),
    the last stop of a runaway. Off: one clock with the model's time on it.
    `stalled_calls`: the episode ends after this many model calls in a row with no env step used
    (`Limits.max_stalled_calls`; None: off): a done check that refuses every finish, or any loop
    of looks and checks, otherwise runs until the cap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    wall_clock_s_per_step: PositiveFloat = 10.0
    model_calls_per_step: PositiveFloat = 4.0
    credit_model_time: bool = True
    cap_s_per_step: PositiveFloat = 60.0
    stalled_calls: PositiveInt | None = 30

    def limits(self, step_limit: int, max_format_errors: int) -> Limits:
        credit = self.credit_model_time
        return Limits(
            max_model_calls=math.ceil(self.model_calls_per_step * step_limit),
            wall_clock_s=self.wall_clock_s_per_step * step_limit,
            max_format_errors=max_format_errors,
            credit_model_time=credit,
            wall_clock_cap_s=self.cap_s_per_step * step_limit if credit else None,
            max_stalled_calls=self.stalled_calls,
        )


class HostConfig(BaseModel):
    """The host's episodes: where records go, the embodiment toolsets see, the runaway guards,
    how many format errors in a row end an episode, changes to the host phase's rules, the toolset
    profile (None: every mounted toolset) and the bridge's waits. There is no turn budget: the
    benchmark's step limit is the only limit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: Path
    embodiment: str
    guards: Guards = Guards()
    max_format_errors: PositiveInt = 3
    rules: PhaseRules | None = None
    toolsets: list[str] | None = None
    bridge: BridgeConfig = BridgeConfig()
    base_targets_positional: bool = False
    """As `PracticeConfig.base_targets_positional`: set it as the practice did."""
    latch_finish: bool = True
    """Whether a `finish` while the episode runs (its success latch has not fired) is refused up to
    `finish_refusals` times, with the latch hint."""
    finish_refusals: PositiveInt = 3
    provenance: Provenance | None = None
    """What every episode's summary records as its code and behaviour config (the launcher's;
    None: the code's revision alone, `settings.source_sha`)."""

    @field_serializer("rules")
    def _given_rules(self, rules: PhaseRules | None) -> dict[str, Any] | None:
        # Only the fields given: the host phase's rules keep their defaults for the others
        # (`recover` above all) when the dumped config is read back, as a launcher does.
        return None if rules is None else rules.model_dump(mode="json", exclude_unset=True)


@dataclass(eq=False)
class _Episode:
    key: str
    index: int
    task: TaskName
    memory: Version | None  # library mode: None for a task without a Memory of its own
    bridge: BridgeEnv
    binding: EnvBinding
    instruction: str | None
    directory: Path
    overlay: Overlay = NO_OVERLAY
    library: EpisodeLibrary | None = None  # library mode: what the Actor sees and uses
    guard: "_FinishGuard | None" = None
    hub_id: EpisodeId | None = None
    log_start: int = 0  # the host log's size when the Hub episode opened
    running: anyio.CancelScope | None = None  # the brain's run, once started
    done: anyio.Event = field(default_factory=anyio.Event)  # over and recorded
    end: EpisodeEnd | None = None
    result: BrainResult | None = None
    counters: Counters | None = None
    outcome: OfficialOutcome | None = None
    voided: EpisodeEnd | None = None  # the end it had, lost to an unanswered infra error
    voided_by: str | None = None  # that error (`Hub.unanswered_infra`)
    summarized: bool = False  # its run's close has written its summary (or begun to)
    writing: anyio.Lock = field(default_factory=anyio.Lock)  # its summary's writes, one at a time
    handed: set[str] = field(default_factory=set)  # batch ids the client polled
    recorder: Recorder | None = None  # the brain's model-call records, once started


class AgentHostService:
    """The `agent.host` plane over `hub`; use it as `async with service:` around serving it.

    `load` gives each task's Memory (`MemoryLoader`); `brain()` makes each episode's brain and
    `prompt` the Actor's system prompt (`prompt_digest`,
    the templates' version, goes into `config.json`); `models` is the gateway client factory
    brains get; `finish_check`, when given, guards `finish` in episodes whose final workflow
    stage has predicates; `opening`, when given, runs before each brain (`OpeningHook`); `finish`
    gates each episode's finish (`FinishRules`; default: none); `demos`
    holds the tasks' views of their demonstrations (none: a clean one-shot); `stack`, the artifacts
    each task's Actor reads in render order (None: the manifest's Task Memories alone), whose
    layers other than the Task Memory `overlay` loads and renders around the Actor's prompt;
    `cards`, the tasks' success condition cards, which the Actor reads under the instruction
    (`envs.cards.task_text`; a task without one reads the instruction alone). Episodes open in
    the method's `phase` (its `MethodRuntime.host_phase`: wicl's `p3`), at tier D, as the
    Actor."""

    def __init__(
        self,
        config: HostConfig,
        *,
        hub: RunnerHub,
        bridges: Bridges,
        blobs: BlobStore,
        events: EventWriter,
        load: MemoryLoader,
        manifest: MemoryManifest,
        brain: Callable[[], Brain],
        prompt: ActorPrompt,
        phase: PhaseName,
        prompt_digest: str = "",
        models: ModelClients | None = None,
        finish_check: FinishCheck | None = None,
        opening: OpeningHook | None = None,
        finish: FinishRules = ungated,
        demos: Mapping[TaskName, DemoView] | None = None,
        stack: ArtifactStack | None = None,
        overlay: OverlayLoader | None = None,
        library: HostLibrary | None = None,
        identify: IdentifyHook | None = None,
        cards: Mapping[TaskName, TaskCard] | None = None,
    ) -> None:
        self.config = config
        self.provenance = config.provenance or Provenance(source=source_sha())
        self.hub = hub
        self.bridges = bridges
        self.blobs = blobs
        self.events = events
        self.load = load
        self.manifest = manifest
        self.brain = brain
        self.prompt = prompt
        self.phase = phase
        self.prompt_digest = prompt_digest
        self.models = models
        self.finish_check = finish_check
        self.opening = opening
        self.finish = finish
        self.demos = dict(demos or {})
        self.stack = stack
        self.overlay = overlay
        self.library = library
        self.identify = identify
        self.cards = dict(cards or {})
        self._overlays: dict[str, Overlay] = {}
        self._slots: dict[str, _Episode] = {}
        self._counts: Counter[str] = Counter()
        self._memories: dict[str, Version] = {}
        self._group: anyio.abc.TaskGroup | None = None
        self._failed: EpisodeEnd | None = None  # the configuration error that failed the run

    @property
    def ready(self) -> bool:
        return len(self._memories) == len(self.manifest.tasks)

    async def start(self) -> None:
        """Load the Memory of every task of the manifest by reference, through `load` (a
        mismatch with the store fails here), and with a `stack`, each task's overlay."""
        memories = manifest_stack(self.manifest)
        stack = memories if self.stack is None else self.stack
        own = tuple(layer for layer in stack.layers if layer.ref.kind == KIND)
        if own != memories.layers:
            raise ValueError("the stack's Task Memory layers are not the manifest's")
        for task in self.manifest.tasks:
            name = TaskName(task)
            layers = stack.for_task(name)
            if self.library is not None and self.library.withholds(name):
                # leave-one-out: an overlay's part for the task (its TASK part) is withheld,
                # its parts for every task (GLOBAL) stay
                layers = [layer for layer in layers if layer.task is None or layer.ref.kind == KIND]
            memory = [layer for layer in layers if layer.ref.kind == KIND]
            self._memories[task] = await self.load(name, memory)
            if len(memory) == len(layers):
                continue
            if self.overlay is None:
                raise ValueError(f"{task}: the stack has layers beside its Memory: give an overlay")
            self._overlays[task] = await self.overlay(name, layers)

    async def __aenter__(self) -> Self:
        self._group = anyio.create_task_group()
        await self._group.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        group, self._group = self._group, None
        assert group is not None
        group.cancel_scope.cancel()  # running episodes are cancelled and recorded
        await group.__aexit__(exc_type, exc, traceback)

    # The agent.host plane

    async def reset(
        self, key: EpisodeKey, task: TaskName, manifest: ManifestId, instruction: str | None = None
    ) -> None:
        if manifest != self.manifest.id:
            detail = f"this host evaluates Memory manifest {self.manifest.id}, not {manifest}"
            raise ProtocolError("invalid_request", detail)
        if task not in self.manifest.tasks and self.library is None:
            # library mode: a task without a Memory of its own (an open task) draws on
            # the library alone, with nothing in use as it opens
            raise ProtocolError("invalid_request", f"manifest {manifest} has no Memory for {task}")
        if not self.ready:
            raise InfraError("not_ready", "the Agent host is still loading its Memory versions")
        if (previous := self._slots.pop(key, None)) is not None:
            await self._stop(previous)
        if (tape := replay.active()) is not None and instruction is not None:
            instruction = tape.pin("p3.instruction", instruction)  # as in `submit`
        memory = self._memories.get(task)
        overlay = self._overlays.get(task, NO_OVERLAY)
        index = self._counts[key]
        self._counts[key] += 1
        bridge, binding = self.bridges.open(self.config.bridge)
        directory = self.config.records / key / f"ep{index:03d}"
        episode = _Episode(
            key, index, task, memory, bridge, binding, instruction, directory, overlay
        )
        self._slots[key] = episode
        data: dict[str, Any] = {"task": task, "memory": memory and memory.meta.version}
        self._emit("p3.reset", episode, data | _overlaid(overlay))

    async def submit(self, key: EpisodeKey, observation: Observation) -> None:
        episode = self._current(key)
        if (tape := replay.active()) is not None and observation.instruction is not None:
            # a benchmark's client may draw it without a seed: a replay pins it
            pinned = episode.instruction or tape.pin("p3.instruction", observation.instruction)
            observation = observation.model_copy(update={"instruction": pinned})
        episode.bridge.submit(observation)
        data = {"frame_id": observation.frame_id, "step": observation.step}
        self._emit("p3.submit", episode, data)
        if episode.running is None and not episode.done.is_set():
            await self._start(episode, observation)

    async def poll(self, key: EpisodeKey) -> Poll:
        episode = self._current(key)
        if episode.done.is_set():
            assert episode.end is not None
            return Poll(status="done", reason=episode.end.reason)
        batch = episode.bridge.poll() if episode.running is not None else None
        if batch is None:
            return Poll(status="pending")
        if batch.batch_id not in episode.handed:
            episode.handed.add(batch.batch_id)
            self._emit("p3.poll", episode, {"batch_id": batch.batch_id, "steps": len(batch.steps)})
        return Poll(status="batch", batch=batch)

    async def report(self, key: EpisodeKey, result: ExecuteResult) -> None:
        episode = self._current(key)
        episode.bridge.report(result)
        last = result.states[-1].robot.step if result.states else None
        data = {"batch_id": result.batch_id, "executed": result.executed, "step": last}
        self._emit("p3.report", episode, {**data, "stopped": result.stopped, "error": result.error})

    async def end(self, key: EpisodeKey, outcome: OfficialOutcome) -> None:
        episode = self._current(key)
        episode.outcome = outcome
        if outcome.success and episode.voided is not None:  # it latched after all: not lost
            episode.end, episode.voided, episode.voided_by = episode.voided, None, None
        episode.bridge.end("success" if outcome.success else "step_limit")
        self._emit("p3.end", episode, outcome.model_dump(mode="json"))
        if episode.summarized or episode.done.is_set():  # written already: again, as it ended
            with anyio.CancelScope(shield=True):  # a client gone mid-call leaves it whole
                await self._summary(episode)  # else the run writes it when it ends

    # Episodes

    async def _start(self, episode: _Episode, observation: Observation) -> None:
        """Open the Hub episode and start the brain; a failure ends the episode here, and after
        a configuration error nothing starts."""
        if self._failed is not None:
            note = f"the run failed on a configuration error earlier: {self._failed.note}"
            episode.end = EpisodeEnd(reason=self._failed.reason, note=note, at=time.time())
            await self._close(episode)
            return
        instruction = episode.instruction or observation.instruction
        if instruction is None:
            raise ProtocolError(
                "invalid_request", "the episode's first observation has no instruction"
            )
        step_limit = observation.robot.step_limit
        if step_limit is None:
            detail = "the episode's first observation has no step limit, the episode's only limit"
            raise ProtocolError("invalid_request", detail)
        episode.instruction = instruction
        limits = self.config.guards.limits(step_limit, self.config.max_format_errors)
        # library mode: no Memory until the Actor adds some (`_bind`)
        if self.library is not None or episode.memory is None:
            doc = MemoryDoc(task=episode.task)
        else:
            doc = episode.memory.doc
        demo = self._demo(episode.task)
        rules, seed = self.finish(
            doc, self.config.rules, demo, instruction, self.config.base_targets_positional
        )
        seed = spec_seed(doc, seed)  # a call may name a Memory spec: all its stored fields
        if self.library is None and (rules is None or rules.done_check is None):
            log.warning(
                "%s: no done criteria and no demonstration: finish is not gated", episode.task
            )
        spec = OpenEpisode(
            stage=self.phase,
            tier="D",
            role="actor",
            toolsets=self.config.toolsets,
            budgets=Budgets(),
            rules=rules,
            env=episode.binding,
            embodiment=self.config.embodiment,
            seed=seed,
        )
        size = await anyio.to_thread.run_sync(lambda: self.events.path.stat().st_size)
        episode.log_start = size  # its events come after this (the log is filtered by id too)
        try:
            session = await self.hub.open_episode(spec)
        except Exception as error:
            episode.end = EpisodeEnd(reason="host_error", note=str(error), at=time.time())
            await self._close(episode)
            raise
        episode.hub_id = session.episode_id
        card = self.cards.get(episode.task)
        text = task_text(instruction, card)
        if self.library is None:
            prompt = episode.overlay.wrap(self.prompt(text, doc))
        else:
            prompt = episode.overlay.wrap(self.library.prompt(text))
            episode.library = self._episode_library(episode, session)
            self.library.mount.attach(session.episode_id, episode.library)
        card = shown(card, prompt)
        if card is not None:
            self.events.emit(EventKind.TASK_CARD, card.record(), episode_id=session.episode_id)
        recorder = await anyio.to_thread.run_sync(
            functools.partial(Recorder, episode.directory, self.events, background=True)
        )
        episode.recorder = recorder
        if episode.library is None:
            await self._write(episode, "memory_before.md", view(doc, detail=False))
        else:
            seen = {
                "entries": [e.record() for e in episode.library.entries],
                "excluded": episode.library.excluded,
                "own": episode.library.own,
            }
            await self._write(episode, "library.json", json.dumps(seen, indent=2) + "\n")
        config = {
            "key": episode.key,
            "index": episode.index,
            "task": episode.task,
            "instruction": instruction,
            "manifest": self.manifest.id,
            "memory": _memory(episode.memory),
            "episode_id": session.episode_id,
            "tools": {tool.name: tool.hash for tool in session.tools},
            "prompt": {
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "templates": self.prompt_digest,
            },
            "host": self.config.model_dump(mode="json"),
            "demo_context": {"demo_context": NO_DEMO} if demo is None else demo.label,
            "limits": limits.model_dump(mode="json"),  # the guards for this step limit
            **_overlaid(episode.overlay),
            **({} if card is None else {"card": card.record()}),
        }
        if self.library is not None and episode.library is not None:
            config["memory_access"] = "library"
            config["prompt"]["library_template"] = self.library.digest
            config["library"] = {
                "seen": [e.id for e in episode.library.entries],
                "excluded": episode.library.excluded,
                "own": episode.library.own,
            }
        await self._write(episode, "config.json", json.dumps(config, indent=2) + "\n")
        final = doc.workflow[-1] if doc.workflow else None
        check = self.finish_check if final is not None and final.done else None
        if self.library is not None:  # the Memories in use set the final stage (`_bind`)
            check = self.finish_check
        refusals = self.config.finish_refusals if self.config.latch_finish else 0
        if refusals or check is not None:
            hint = "" if demo is None or demo.goal is None else demo.goal.hint
            guard = _FinishGuard(episode.bridge, session.episode_id, refusals, hint, check, final)
            self.hub.set_finish_guard(session.episode_id, guard, times=refusals + 1)
            episode.guard = guard
        agent = AgentSession(
            episode=session,
            hub=self.hub,
            instructions=prompt,
            blobs=self.blobs,
            recorder=recorder,
            models=self.models,
            limits=limits,
            brief=None if demo is None else demo.brief or None,
        )
        episode.running = anyio.CancelScope()
        assert self._group is not None, "use the service as `async with service:`"
        self._group.start_soon(self._run, episode, agent, self.brain(), session, doc)

    async def _run(
        self,
        episode: _Episode,
        agent: AgentSession,
        brain: Brain,
        session: EpisodeSession,
        doc: MemoryDoc,
    ) -> None:
        """The opening hook (off the client's calls: identifying takes a while), then the
        brain."""
        assert episode.running is not None and episode.instruction is not None
        try:
            with episode.running:
                if self.opening is not None:
                    opened = await self.opening(session, episode.instruction, doc)
                    if opened.end is not None:
                        await self.hub.end_episode(session.episode_id, opened.end, opened.note)
                        return
                    brief = [*(agent.brief or []), *opened.brief]
                    agent = dataclasses.replace(agent, brief=brief or None)
                episode.result = await brain.run(agent)
        except Exception:  # the loop ended the Hub episode as crashed; record it all the same
            log.exception("the brain of %s episode %d crashed", episode.key, episode.index)
        finally:
            with anyio.CancelScope(shield=True):
                await self._close(episode)

    def _episode_library(self, episode: _Episode, session: EpisodeSession) -> EpisodeLibrary:
        """The episode's library (library mode): the entries its task sees, bound by `_bind` as
        the Actor changes the Memories in use."""
        assert self.library is not None
        seen, excluded = self.library.seen(episode.task)
        own = next((e.id for e in self.library.entries if e.task == episode.task), None)

        def progress() -> Progress:
            steps, failures, elapsed = self.hub.progress(session.episode_id)
            return Progress(steps=steps, failures=failures, elapsed_s=elapsed)

        async def bind(using: list[LibraryEntry], added: list[LibraryEntry]) -> list[ContentPart]:
            return await self._bind(episode, session, using, added)

        return EpisodeLibrary(seen, excluded, own, bind, progress)

    async def _bind(
        self,
        episode: _Episode,
        session: EpisodeSession,
        using: list[LibraryEntry],
        added: list[LibraryEntry],
    ) -> list[ContentPart]:
        """Bind the Memories in use (`using`, in the order added) to the episode as an episode
        bound to them would have opened: each added one's roles identified in the scene (the
        `identify` hook, its time credited back to the episode's clock; its roles merged into the
        `IDENTIFIED` fact beside those known), then their specs and numbers (`spec_seed`), roles'
        arm sides and done criteria (the method's finish gate) over the combined Memory
        (`memory.library.combine`). What it tells the Actor.

        Never ends the episode: roles two Memories name for other objects refuse the use
        (`ToolError`, naming them), a failed identification is the Actor's tool error
        (`InfraError` when the infrastructure failed it), and on any failure, the tool's timeout
        included, the `IDENTIFIED` fact is put back as it was (the objects an identification
        registered stay registered, as its events record) and nothing is rebound."""
        assert episode.instruction is not None
        try:
            clashes = role_clashes(using)
        except ValueError as error:
            raise ToolError(f"a Memory's OBJECTS section does not parse: {error}") from error
        if clashes:
            raise ToolError(
                f"these Memories name one role for different objects: {'; '.join(clashes)}. "
                "Binding both would put one role on the wrong object: drop one of them first, or "
                "use the other alone. The set in use is unchanged."
            )
        eid = session.episode_id
        known = self.hub.fact(eid, IDENTIFIED)
        roles = {} if known is None else dict(known.get("roles", {}))
        content: list[ContentPart] = []
        facts: dict[str, Mapping[str, Any]] = {}
        try:
            for entry in added:
                if self.identify is None:
                    break
                began = time.monotonic()
                try:
                    opened = await self.identify(session, episode.instruction, entry.version.doc)
                finally:
                    with anyio.CancelScope(shield=True):  # identifying is not the Actor's time
                        await self.hub.credit_clock(eid, time.monotonic() - began, "identification")
                if opened.end is not None:
                    detail = f"identifying the roles of {entry.id} failed ({opened.note})"
                    if opened.end in INFRA_ENDS:
                        raise InfraError("not_ready", f"{detail}; the set in use is unchanged")
                    raise ToolError(f"{detail}; the set in use is unchanged")
                after = self.hub.fact(eid, IDENTIFIED)
                if after is not None:  # an identification names only its own roles: keep others
                    roles |= after.get("roles", {})
                    facts[IDENTIFIED] = {**after, "roles": dict(roles)}
                content += [TextPart(text=f"{entry.id}: its roles in this scene."), *opened.brief]
            doc = combine(episode.task, [entry.version.doc for entry in using])
            demo = self._demo(episode.task)
            positional = self.config.base_targets_positional
            rules, seed = self.finish(doc, self.config.rules, demo, episode.instruction, positional)
            seed = spec_seed(doc, seed)
            bound = (
                {} if seed is None else {k: v for k, v in seed.facts.items() if k in BOUND_FACTS}
            )
            facts |= bound
            drop = [name for name in BOUND_FACTS if name not in bound]
            self.hub.rebind(eid, rules, facts, drop)
        except BaseException:
            if self.hub.fact(eid, IDENTIFIED) != known:
                self.hub.put_fact(eid, IDENTIFIED, known, "library_use failed: as it was")
            raise
        if episode.guard is not None:
            final = doc.workflow[-1] if doc.workflow else None
            episode.guard.final = final if final is not None and final.done else None
        gated = rules is not None and rules.done_check is not None
        criteria = sum(1 for p in doc.params.values() if p.kind == "done_spec")
        gate = f"finish is gated on {rules.done_check}" if gated and rules else "finish is ungated"
        content.append(TextPart(text=f"Done criteria in use: {criteria}; {gate}."))
        return content

    def _demo(self, task: TaskName) -> DemoView | None:
        """The task's demonstration as its Actor reads it; none in a generalization test of its
        own Memory (`HostLibrary.withholds`): its brief, frames, end (the finish gate's final-state
        anchor and end targets) and latch hint all reach the Actor by task id."""
        if self.library is not None and self.library.withholds(task):
            return None
        return self.demos.get(task)

    async def _close(self, episode: _Episode) -> None:
        """Close the bridge and the Hub episode, write the episode's events and summary, then
        mark it done, whatever failed."""
        try:
            self.bridges.close(episode.bridge)
            if self.library is not None and episode.hub_id is not None:
                self.library.mount.detach(episode.hub_id)
            if episode.library is not None:
                with self._recording(episode, MEMORY_USED):
                    await self._write(episode, MEMORY_USED, self._used(episode))
            if episode.hub_id is not None:
                unanswered = self.hub.unanswered_infra(episode.hub_id)
                closed = await self.hub.close_episode(episode.hub_id)
                episode.end, episode.counters = closed.end, closed.counters
                if unanswered is not None and _voidable(episode):
                    episode.voided, episode.voided_by = closed.end, unanswered
                    episode.end = unanswered_end(closed.end, unanswered)
                with self._recording(episode, "events.jsonl"):
                    path, start, hub_id = self.events.path, episode.log_start, episode.hub_id
                    await anyio.to_thread.run_sync(self.events.flush)  # its lines all in
                    lines = await anyio.to_thread.run_sync(_episode_log, path, start, hub_id)
                    await self._write(episode, "events.jsonl", lines)
            if episode.recorder is not None:
                await anyio.to_thread.run_sync(episode.recorder.close)
            with self._recording(episode, "summary.json"):
                await self._summary(episode)
                end = None if episode.end is None else episode.end.reason
                self._emit("p3.episode", episode, {"end": end})
            if self._failed is None and episode.end and episode.end.reason.startswith(CONFIG_ENDS):
                with self._recording(episode, CONFIG_ERROR):
                    await self._fail(episode, episode.end)
        finally:
            episode.done.set()

    async def _fail(self, episode: _Episode, end: EpisodeEnd) -> None:
        """The run fails on `end`, a configuration error (`CONFIG_ERROR`)."""
        self._failed = end
        data = {"key": episode.key, "index": episode.index, "task": episode.task}
        data["end"] = end.model_dump(mode="json")
        log.error(
            "the run failed: %s episode %d: %s: %s",
            episode.key,
            episode.index,
            end.reason,
            end.note,
        )
        self._emit("p3.config_error", episode, data)
        text = (json.dumps(data, indent=2) + "\n").encode()
        await anyio.to_thread.run_sync(write_atomic, self.config.records / CONFIG_ERROR, text)

    async def _stop(self, episode: _Episode) -> None:
        """End a slot's episode that a new one replaces."""
        if episode.running is not None and not episode.done.is_set():
            episode.running.cancel()
            with anyio.move_on_after(CANCEL_WAIT_S):
                await episode.done.wait()
        elif not episode.done.is_set():  # never started: nothing ran
            episode.end = EpisodeEnd(reason="not_started", at=time.time())
            await self._close(episode)

    def _current(self, key: str) -> _Episode:
        episode = self._slots.get(key)
        if episode is None:
            raise ProtocolError("not_found", f"no episode in slot {key}: reset it first")
        return episode

    async def _summary(self, episode: _Episode) -> None:
        """Write the summary as the episode stands, one write at a time, so the last one written
        has the latest of it (the client's `end` may come while the run's close writes it)."""
        episode.summarized = True
        async with episode.writing:
            await self._write_summary(episode)

    async def _write_summary(self, episode: _Episode) -> None:
        result, ended = episode.result, episode.voided or episode.end
        summary: dict[str, Any] = {
            "key": episode.key,
            "index": episode.index,
            "task": episode.task,
            "instruction": episode.instruction,
            "manifest": self.manifest.id,
            "memory": _memory(episode.memory),
            "episode_id": episode.hub_id,
            "end": None if episode.end is None else episode.end.model_dump(mode="json"),
            "official": None if episode.outcome is None else episode.outcome.model_dump(),
            "infra": _infra(episode),
            "guard_tripped": ended is not None and ended.reason in GUARD_ENDS,
            "counters": None if episode.counters is None else episode.counters.model_dump(),
            "turns": None if result is None else result.turns,
            "model_calls": None if result is None else result.model_calls,
            "usage": None if result is None else result.usage.model_dump(),
            "elapsed_s": None if result is None else result.elapsed_s,
            "provenance": self.provenance.model_dump(mode="json"),
        }
        if episode.library is not None:
            summary["memory_access"] = "library"
            summary["library"] = episode.library.summary()
        if episode.voided is not None:  # lost to an unanswered infra error: what it would have been
            summary |= {"voided_end": episode.voided.reason, "voided_by": episode.voided_by}
        await self._write(episode, "summary.json", json.dumps(summary, indent=2) + "\n")

    def _used(self, episode: _Episode) -> str:
        """The Memories in use at the end (library mode, `MEMORY_USED`): each one's view, in the
        order added, as `memory_before.md` holds a bound episode's."""
        assert episode.library is not None
        using = [e for i in episode.library.in_use if (e := episode.library.get(i)) is not None]
        if not using:
            return "(no Memory in use at the end)\n"
        return "\n\n".join(
            f"<!-- {entry.id} (from task {entry.task}) -->\n{view(entry.version.doc, detail=False)}"
            for entry in using
        )

    @contextlib.contextmanager
    def _recording(self, episode: _Episode, what: str) -> Iterator[None]:
        """A record that cannot be written is logged, and the host serves on."""
        try:
            yield
        except (OSError, ValueError):  # a full disk; a log that is not what it should be
            log.exception("could not write %s of %s episode %d", what, episode.key, episode.index)

    async def _write(self, episode: _Episode, name: str, text: str) -> None:
        """Off the event loop: a stalled disk holds only this episode's record, not the host."""
        await anyio.to_thread.run_sync(write_atomic, episode.directory / name, text.encode())

    def _emit(self, kind: str, episode: _Episode, data: dict[str, Any]) -> None:
        data = {"key": episode.key, "index": episode.index, **data}
        self.events.emit(kind, data, episode_id=episode.hub_id)


LATCH_REFUSAL: Final = (
    "the episode has not ended, so the task's success has not latched (the benchmark ends an "
    "episode the moment it does): the task is not complete yet, look at the scene again and go on"
)


@dataclass
class _FinishGuard:
    """A host episode's finish guard: up to `latch` refusals while the bridge's episode runs (its
    success latch has not fired), with `hint`; then the finish check, once."""

    bridge: BridgeEnv
    episode_id: EpisodeId
    latch: int
    hint: str
    check: FinishCheck | None = None
    final: WorkflowStage | None = None

    async def __call__(self, note: str) -> str | None:
        if self.latch > 0 and self.bridge.ended is None:
            self.latch -= 1
            return LATCH_REFUSAL + self.hint
        if self.check is not None and self.final is not None:
            check, self.check = self.check, None
            return await check(self.episode_id, self.final)
        return None


def _episode_log(path: Path, start: int, episode_id: EpisodeId) -> str:
    """The host log's lines of one episode, from where it opened (not the whole log again)."""
    events = read_events(path, start)
    return "".join(
        event.model_dump_json() + "\n" for event in events if event.episode_id == episode_id
    )


def _voidable(episode: _Episode) -> bool:
    """Whether the episode is lost after an unanswered infrastructure error
    (`agent.session.unanswered_end`), whatever ended it: not if the benchmark's success latched,
    nor if it ended on infrastructure already, was cancelled or ended on a configuration
    error."""
    if episode.bridge.ended == "success" or (episode.outcome and episode.outcome.success):
        return False
    reason = None if episode.end is None else episode.end.reason
    return (
        reason is not None
        and reason not in INFRA_ENDS
        and reason != "cancelled"
        and not reason.startswith(CONFIG_ENDS)
    )


def _infra(episode: _Episode) -> bool:
    """Whether infrastructure failed the episode: a failed call of the client, or such an end;
    never one whose success latched, since a success stands (the client and the launcher count
    it)."""
    if episode.bridge.ended == "success" or (episode.outcome and episode.outcome.success):
        return False
    failed = episode.outcome is not None and episode.outcome.failed_calls > 0
    return failed or (episode.end is not None and episode.end.reason in INFRA_ENDS)


def infra_lost(summary: Mapping[str, Any]) -> bool:
    """Whether a host's episode summary (`summary.json`) says infrastructure lost the episode:
    its `infra`, unless its success latched, since a success stands. A launcher's count and the
    reports read it so."""
    return bool(summary.get("infra")) and not (summary.get("official") or {}).get("success")


def _overlaid(overlay: Overlay) -> dict[str, Any]:
    """The overlay's layers as records name them; nothing without any."""
    if not overlay.layers:
        return {}
    return {"overlay": [layer.model_dump(mode="json") for layer in overlay.layers]}


def _memory(version: Version | None) -> dict[str, Any] | None:
    if version is None:  # library mode, a task without a Memory of its own
        return None
    return {"task": version.meta.task, "version": version.meta.version, "hash": version.meta.hash}
