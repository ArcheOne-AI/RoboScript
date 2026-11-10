"""One episode end to end (docs/concepts/learning.md, docs/concepts/operations.md).

`EpisodeDriver.run(Episode)`: the env's control plane opens the scene (`EpisodeSpec`: seed,
stage, instruction to reuse, expert pre-runs), the Hub opens the episode (stage, tier, role,
toolset profile, rules, the role's budgets and render limits, the env binding, the embodiment,
seeded frames), perception identifies the objects once when the episode has `identify`
(`roboscript.learning.opening`: the Memory's OBJECTS roles and the live instruction; or the
episode starts from an earlier identification of the same scene, `Episode.reuse`, if its first
frame still looks like the one that identification saw), the Hub's wall clock restarts, the
role's brain plays it with the identification in its brief, the env judges it and diagnoses a
failure, both episodes close, and the episode's directory holds

    config.json        what ran: env spec, Hub episode, brain, limits, Memory, prompt, tools
    identify.json      the identification, for a later try to start from (`opening.Reuse`)
    objects.json       the registered objects at the end, as the model would be shown them
    events.jsonl       the episode's events: Hub, toolsets, model calls
    llm_calls.jsonl    turns.jsonl  (roboscript.agent.Recorder)
    tool_calls.jsonl   each tool call with its result (a view of the events)
    physics.npz        the robot after every executed step: TCP, gripper command and opening
    memory_before.md   the Memory the role read
    summary.json       the `EpisodeOutcome`
    referee/           ground truth only: success.json (the judgement), diagnostics.json and
                       explained.json, the diagnosis in the try's own terms, which a reviewing
                       role reads (the method's explainer, `MethodRuntime.explain`; none without
                       one)

The driver watches every `execute` of its episodes (`Trajectories`): the robot states become
`physics.npz`, and the success latch or the step limit ends the Hub episode as `success` or
`step_limit`, as the benchmark's own loop ends its episodes.

End classes: `success`; `policy` (`finish` without success, turn, step, call or time limits,
format errors, a request the model refused for what the episode sent); `fail_fast` (a hard
failure under fail-fast); `infra` (a faulted env, one that did not open or answer, a gateway
failure, an interrupted action, an identification the infrastructure failed, an Actor's episode
whose last failed model call failed for infrastructure reasons with no model call answered after
it, `Hub.unanswered_infra`, unless the task's success latched: it ends as `infra_error` whatever
ended it, the end it had kept as `voided_end` and the failed call as `voided_by`;
`open_failed:<code>` when the env did not open: `run_counted` re-runs these);
`unsolvable` (the
expert pre-runs failed, or an env without an expert skips the reset by the benchmark's rule);
`cancelled` (the brain was cancelled or crashed: the driver re-raises);
`config` (a request refused as malformed, an identification that failed otherwise:
`run_counted` stops). Only an Actor's episode is judged; for the other roles `finish` is success:
the role did its job. A failed diagnosis leaves the judgement as it is. The perception model
calls of an episode count in its cost (`perception_usage`).
"""

import hashlib
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal

import anyio
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from roboscript.agent import AgentSession, Brain, BrainResult, Recorder, Usage
from roboscript.agent.host import OpaqueGoal
from roboscript.agent.session import CONFIG_ENDS, ModelClients, unanswered_end
from roboscript.blobs import BlobStore
from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.env import (
    Diagnosis,
    EnvAgent,
    EnvControl,
    EpisodeOpened,
    EpisodeSpec,
    JudgeResult,
    ObserveRequest,
)
from roboscript.contracts.errors import InfraError, RSError
from roboscript.contracts.events import EventKind
from roboscript.contracts.geometry import Pose
from roboscript.contracts.hub import (
    EpisodeEnd,
    EpisodeSeed,
    EpisodeSession,
    OpenEpisode,
    ToolHub,
)
from roboscript.contracts.ids import BatchId, EpisodeId, PhaseName, RoleName, TaskName
from roboscript.contracts.observation import Observation
from roboscript.contracts.phase import PhaseRules
from roboscript.contracts.provenance import Provenance
from roboscript.contracts.robot import RobotState
from roboscript.contracts.tools import ContentPart, EnvBinding, Tier
from roboscript.contracts.trace import new_trace, use_trace
from roboscript.demos import ObjectShape, object_shapes
from roboscript.envs.cards import TaskCard, shown
from roboscript.hub import ActionNotes, Hub
from roboscript.learning.config import RoleConfig
from roboscript.learning.events import EventRouter
from roboscript.learning.opening import (
    Drift,
    Opening,
    Reuse,
    save,
)
from roboscript.learning.runtime import MethodRuntime
from roboscript.memory import Editor, MemoryDoc, content_hash, view
from roboscript.records import read_events, write_atomic
from roboscript.settings import source_sha
from roboscript.tools import REFS, seed_from_logs
from roboscript.tools.memory_facts import spec_seed
from roboscript.toolsets.memory import MemoryTools
from roboscript.toolsets.scene import ObjectReport, describe, object_report

log = logging.getLogger(__name__)

EndClass = Literal["success", "policy", "fail_fast", "infra", "unsolvable", "cancelled", "config"]
INFRA_ENDS: Final = frozenset({"gateway_error", "infra_error", "env_ended", "action_interrupted"})
"""Hub and loop end reasons that are infrastructure, not policy."""
OBJECTS: Final = "objects.json"

EVENTS: Final = "events.jsonl"
PHYSICS: Final = "physics.npz"
SUMMARY: Final = "summary.json"
REFEREE: Final = "referee"
SHAPES: Final = "shapes.json"
"""The objects' shapes under `REFEREE` (`demos.referee.object_shapes`)."""
EXPLAINED: Final = "explained.json"

Prompt = Callable[[EpisodeOpened | None], tuple[str, Sequence[ContentPart] | None]]
"""Builds the system prompt and the brief from what the env opened (None without an env)."""


@dataclass(frozen=True)
class Episode:
    """What to run. `spec` opens the env (None: an episode without one, like a reviewing role's);
    `rules` changes the stage's rules, `toolsets` is the Hub profile, `frames` seeds the frame
    store; `editor` is attached to the memory toolset; `diagnose` asks for ground truth after a
    failure; `notes` makes the notes shown after each physical action from the episode's
    trajectory; `memory` is what the role reads (for `memory_before.md` and the hash) and
    `memory_label` names it (`v003`, the cache `c1`); `reuse` is an earlier identification of
    the same scene to start from (taken when its roles and instruction are this episode's);
    `extra` is the protocol's context, in `config.json`; `finish_guard` may refuse a
    `finish` with a message, up to `finish_refusals` times (`Hub.set_finish_guard`); without
    one, the method's finish gate may set its own (`FinishGate.guard`), up to as many times;
    `shapes` has the objects' shapes (symmetry, center) read from the ground truth as the
    episode opens, for the keypoint referee (`referee/shapes.json`). An Actor's `finish` the
    env's success check does not confirm ends the episode as the method's finish gate says
    (`FinishGate.unconfirmed`: a failure, never an accepted finish). `goal`: the
    demonstration's end as the method reads it (`agent.host.OpaqueGoal`; wicl's
    `goal.DemoGoal`: its final-state views, press, final frame, end targets, latched lifts and
    latch motion), which the driver passes to the method's finish gate (`MethodRuntime.finish`)
    without reading it; None: none. `method`: the episode's own settings for the method's
    hooks (`FinishGate`, `MethodRuntime.judged`, `MethodRuntime.explain`; wicl's
    `extension.WiclEpisode`), carried without reading; None: none. `step_limit` is the task's,
    for an episode without an env (a reviewing role's), to scale the role's bounds
    (`RoleConfig.for_steps`) as an open env episode's own step limit does. `max_turns` caps the
    role's turn budget (as scaled) for this episode alone: it ends at the smaller of the two.
    `stalled_calls` sets the loop's stall guard for this episode (`Limits.max_stalled_calls`)."""

    task: TaskName
    stage: PhaseName
    role: RoleName
    directory: Path
    prompt: Prompt
    spec: EpisodeSpec | None = None
    tier: Tier = "D"
    rules: PhaseRules | None = None
    toolsets: list[str] | None = None
    frames: EpisodeSeed | None = None
    editor: Editor | None = None
    diagnose: bool = False
    notes: Callable[["Trajectory"], ActionNotes] | None = None
    memory: MemoryDoc | None = None
    memory_label: str | None = None
    memory_hash: str | None = None  # a stored version's `meta.hash`; else its content hash
    reuse: Reuse | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    finish_guard: Callable[[str], Awaitable[str | None]] | None = None
    finish_refusals: int = 1
    shapes: bool = False
    goal: OpaqueGoal | None = None
    method: object | None = None
    step_limit: int | None = None
    max_turns: int | None = None
    stalled_calls: int | None = None


BrainFactory = Callable[[Episode], Brain]


class Identification(BaseModel):
    """How an episode's objects were identified: `identified`, `reused` (from `source`), or not
    (`skipped`, or a failure: `infra`, `config`)."""

    model_config = ConfigDict(frozen=True)

    status: str
    source: str | None = None
    note: str = ""


class EpisodeOutcome(BaseModel):
    """How an episode went. `episode_id` is the Hub's (the gateway session; None when the env did
    not open); `memory` is the content hash of the Memory the role read; `usage` is the brain's
    and `perception_usage` that of the perception model calls (both count in its cost);
    `provenance` the code and behaviour config that ran it (None: not recorded)."""

    model_config = ConfigDict(frozen=True)

    task: TaskName
    stage: PhaseName
    role: RoleName
    seed: int | None
    directory: Path
    episode_id: EpisodeId | None
    env_episode: EpisodeId | None
    instruction: str | None
    success: bool
    end_class: EndClass
    end: EpisodeEnd
    success_step: NonNegativeInt | None = None
    steps: NonNegativeInt = 0
    step_limit: NonNegativeInt | None = None
    faulted: bool = False
    prerun_attempts: NonNegativeInt = 0
    turns: NonNegativeInt = 0
    model_calls: NonNegativeInt = 0
    usage: Usage = Usage()
    perception_calls: NonNegativeInt = 0
    perception_usage: Usage = Usage()
    identification: Identification | None = None
    wall_clock_s: float = 0.0
    memory: str | None = None
    provenance: Provenance | None = None
    voided_end: str | None = Field(default=None, exclude_if=lambda value: value is None)
    """The end an Actor's episode had before an unanswered infrastructure error voided it
    (`agent.session.unanswered_end`); None (and left out of the record) otherwise."""
    voided_by: str | None = Field(default=None, exclude_if=lambda value: value is None)
    """The call that failed for infrastructure reasons then, and its message."""

    @property
    def tokens(self) -> int:
        """Input and output tokens of every model call, the brain's and perception's."""
        brain, perception = self.usage, self.perception_usage
        return sum(u.input_tokens + u.output_tokens for u in (brain, perception))

    @property
    def solvable(self) -> bool:
        return self.end_class != "unsolvable"


class InfraExhausted(RuntimeError):
    """An episode kept ending for infrastructure reasons; `outcomes` are its voided runs."""

    def __init__(self, outcomes: list[EpisodeOutcome]) -> None:
        last = outcomes[-1]
        super().__init__(
            f"{last.directory.name}: {len(outcomes)} runs in a row ended for infrastructure "
            f"reasons, the last as {last.end.reason} ({last.end.note})"
        )
        self.outcomes = outcomes


class Trajectory:
    """The robot at the start of an episode and after every step its actions executed, as the
    env reported it; `turns` holds the Hub turn each sample belongs to (0: the start)."""

    def __init__(self, start: RobotState) -> None:
        self.samples: list[RobotState] = [start]
        self.turns: list[int] = [0]
        self.batches: list[int] = [-1]
        self._seen: list[BatchId] = []
        self._hub: ToolHub | None = None
        self._episode: EpisodeId | None = None
        self._truth: Callable[[], Awaitable[dict[str, Pose]]] | None = None
        self.truth: list[tuple[int, dict[str, Pose]]] = []  # (sample, object poses)

    def bind(self, hub: ToolHub, episode_id: EpisodeId) -> None:
        """The Hub episode whose turns the samples count, and which the latch or limit ends."""
        self._hub, self._episode = hub, episode_id

    async def watch(self, truth: Callable[[], Awaitable[dict[str, Pose]]]) -> None:
        """Also record the objects' ground-truth poses now and after every executed batch (the
        keypoint referee: placements judged by the held object)."""
        self._truth = truth
        self.truth.append((len(self.samples) - 1, await truth()))

    async def add(self, batch: ActionBatch, result: ExecuteResult) -> None:
        if batch.batch_id in self._seen:  # a batch retried gets its first result again
            return
        self._seen.append(batch.batch_id)
        turn = 0
        if self._hub is not None and self._episode is not None:
            turn = (await self._hub.state(self._episode)).counters.turns
        for state in result.states:
            self.samples.append(state.robot)
            self.turns.append(turn)
            self.batches.append(len(self._seen) - 1)
        if self._truth is not None:
            self.truth.append((len(self.samples) - 1, await self._truth()))
        if self._hub is None or self._episode is None:
            return
        if result.stopped == "success":
            await self._hub.end_episode(self._episode, "success", "the task's success latched")
        elif result.stopped == "step_limit":
            await self._hub.end_episode(self._episode, "step_limit", "the step limit was reached")
        elif result.stopped == "ended":
            note = "the benchmark ended the episode before its step limit"
            await self._hub.end_episode(self._episode, "ended", note)

    def arrays(self) -> dict[str, np.ndarray]:
        """Per sample: `step`, `turn`, `batch` (-1: the start), `qpos_cmd` and per arm
        `<arm>.tcp` (N, 7), `<arm>.gripper_cmd`, `<arm>.gripper` (measured; NaN: unknown); when
        watched, `truth.sample` (K,) and per object `truth.<name>` (K, 7): the ground-truth poses
        at the start and after each batch."""
        samples = self.samples
        data: dict[str, np.ndarray] = {
            "step": np.array([s.step for s in samples]),
            "turn": np.array(self.turns),
            "batch": np.array(self.batches),
            "qpos_cmd": np.array([s.qpos_cmd for s in samples]),
        }
        for arm in samples[0].arms:
            states = [s.arms[arm] for s in samples]
            data[f"{arm}.tcp"] = np.array([a.tcp.as_list() for a in states])
            data[f"{arm}.gripper_cmd"] = np.array([a.gripper_cmd for a in states])
            data[f"{arm}.gripper"] = np.array(
                [np.nan if a.gripper is None else a.gripper for a in states]
            )
        known = [(sample, poses) for sample, poses in self.truth if poses]
        if known:
            data["truth.sample"] = np.array([sample for sample, _ in known])
            for name in set.intersection(*(set(poses) for _, poses in known)):
                data[f"truth.{name}"] = np.array([poses[name].as_list() for _, poses in known])
        return data


class _Tracked:
    """An env agent plane whose `execute` results also go to a trajectory."""

    def __init__(self, env: EnvAgent, trajectory: Trajectory) -> None:
        self._env = env
        self._trajectory = trajectory

    async def robot(self, episode_id: EpisodeId) -> RobotState:
        return await self._env.robot(episode_id)

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation:
        return await self._env.observe(episode_id, request)

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        result = await self._env.execute(episode_id, batch)
        await self._trajectory.add(batch, result)
        return result


class Trajectories:
    """The Hub's env clients (`client`, its `env_client` factory): the agent plane that `agent`
    makes, recording `execute` for the env episodes the driver tracks."""

    def __init__(self, agent: Callable[[EnvBinding], EnvAgent]) -> None:
        self.agent = agent
        self._tracked: dict[EpisodeId, Trajectory] = {}

    def client(self, binding: EnvBinding) -> EnvAgent:
        env = self.agent(binding)
        trajectory = self._tracked.get(binding.episode_id)
        return env if trajectory is None else _Tracked(env, trajectory)  # type: ignore[return-value]

    def track(self, env_episode: EpisodeId, start: RobotState) -> Trajectory:
        trajectory = self._tracked[env_episode] = Trajectory(start)
        return trajectory

    def untrack(self, env_episode: EpisodeId) -> None:
        del self._tracked[env_episode]


class EpisodeDriver:
    """Runs episodes on a started Hub for the tasks whose env control planes it holds."""

    def __init__(
        self,
        *,
        hub: Hub,
        memory: MemoryTools,
        events: EventRouter,
        blobs: BlobStore,
        trajectories: Trajectories,
        controls: Mapping[TaskName, EnvControl],
        brains: Mapping[RoleName, BrainFactory],
        roles: Mapping[RoleName, RoleConfig],
        embodiment: str,
        models: ModelClients | None = None,
        max_infra_reruns: int = 3,
        infra_backoff_s: Sequence[float] = (),
        prompt_digest: str = "",
        provenance: Provenance | None = None,
        runtime: MethodRuntime | None = None,
        cards: Mapping[TaskName, TaskCard] | None = None,
    ) -> None:
        self.hub = hub
        # the tasks' success condition cards, which their roles read under the instruction
        self.cards: Mapping[TaskName, TaskCard] = dict(cards or {})
        self.runtime = runtime or MethodRuntime()  # the method's hooks (`learning.runtime`)
        self.memory = memory
        self.events = events
        self.blobs = blobs
        self.trajectories = trajectories
        self.controls = controls
        self.brains = brains
        self.roles = roles
        self.embodiment = embodiment
        self.models = models
        self.max_infra_reruns = max_infra_reruns
        self.infra_backoff_s = tuple(infra_backoff_s)
        self.prompt_digest = prompt_digest  # of the recipe's templates, for config.json
        # every outcome's: the launcher's, else the code's revision alone
        self.provenance = provenance or Provenance(source=source_sha())

    async def run_counted(self, episode: Episode) -> tuple[EpisodeOutcome, int]:
        """Run `episode`, again after each infrastructure outcome, whose directory stays beside
        as `<name>-infra<n>` (its summary rewritten there): the outcome and how many runs were
        voided. A voided run's Memory edits are undone (its editor goes back to the snapshot
        taken before it), and a voided run that opened passed the expert pre-run, so the next
        reopens the seed without it, with the same instruction. Each re-run waits its
        `infra_backoff_s` first. `InfraExhausted` after `max_infra_reruns` re-runs; ValueError
        for a configuration error."""
        voided: list[EpisodeOutcome] = []
        while True:
            state = None if episode.editor is None else episode.editor.snapshot()
            outcome = await self.run(episode)
            if outcome.end_class == "config":
                raise ValueError(
                    f"{outcome.directory}: a configuration error ({outcome.end.reason}: "
                    f"{outcome.end.note}); fix the configuration"
                )
            if outcome.end_class != "infra":
                return outcome, len(voided)
            if episode.editor is not None and state is not None:
                episode.editor.restore(state)
            n, name = len(voided) + 1, episode.directory.name
            # a run voided before a resume keeps its name: the next free one
            while (moved := episode.directory.with_name(f"{name}-infra{n}")).exists():
                n += 1
            episode.directory.rename(moved)
            outcome = outcome.model_copy(update={"directory": moved})
            _write_json(moved / SUMMARY, outcome)
            by = outcome.voided_by
            if by is not None and any(earlier.voided_by == by for earlier in voided):
                log.warning(  # the same failure again: one that a re-run may not mend
                    "%s: voided again by the same unanswered call on its seed: %s",
                    episode.directory,
                    by,
                )
            voided.append(outcome)
            if len(voided) > self.max_infra_reruns:
                raise InfraExhausted(voided)
            why = "the env faulted" if outcome.faulted else outcome.end.reason
            backoff = self.infra_backoff_s
            wait = backoff[min(len(voided), len(backoff)) - 1] if backoff else 0.0
            log.warning("re-running %s in %g s: %s", episode.directory, wait, why)
            await anyio.sleep(wait)
            if episode.spec is not None and outcome.env_episode is not None:
                again = {"repeat": True, "instruction": outcome.instruction}
                episode = replace(episode, spec=episode.spec.model_copy(update=again))

    async def run(self, episode: Episode) -> EpisodeOutcome:
        """Run `episode` into its directory, which must not exist yet."""
        episode.directory.mkdir(parents=True)
        trace = new_trace()
        try:
            with use_trace(trace), self.events.route(trace, episode.directory / EVENTS):
                return await self._run(episode)
        finally:
            _write_tool_calls(episode.directory)

    async def _run(self, episode: Episode) -> EpisodeOutcome:
        """Open the env episode, play the Hub episode, and close the env episode whatever
        happened in between."""
        started = time.monotonic()
        if episode.spec is None:
            return await self._play(episode, started, None, None)
        control = self.controls[episode.task]
        try:
            opened = await control.open(episode.spec)
        except InfraError as error:  # `open_failed:sim_fault`: this scene could not be set up
            end = _end(f"open_failed:{error.code}", f"the env did not open: {error.detail}")
            return self._summary(episode, started, None, end, "infra")
        if not opened.solvable:
            note = f"the expert pre-run failed {opened.prerun_attempts} times"
            if not opened.prerun_attempts:  # no expert: the benchmark's own rule skips the reset
                note = "the env does not count this reset (the benchmark's own rule skips it)"
            return self._summary(episode, started, opened, _end("unsolvable", note), "unsolvable")
        try:
            return await self._play(episode, started, control, opened)
        finally:
            with anyio.CancelScope(shield=True):
                await _close(control, opened.episode_id)

    async def _play(
        self,
        episode: Episode,
        started: float,
        control: EnvControl | None,
        opened: EpisodeOpened | None,
    ) -> EpisodeOutcome:
        """The Hub episode and its brain, on the open env episode if there is one; only an
        Actor's episode is judged."""
        actor = episode.role == "actor"
        binding = None
        if opened is not None:
            binding = EnvBinding(
                url=opened.agent_url, episode_id=opened.episode_id, token=opened.agent_token
            )
        trajectory: Trajectory | None = None
        session: EpisodeSession | None = None
        result: BrainResult | None = None
        judged: JudgeResult | None = None
        diagnosis: Diagnosis | None = None
        closed: EpisodeEnd | None = None
        unanswered: str | None = None  # `Hub.unanswered_infra` as the episode closed
        failed: InfraError | None = None  # the env, before the Hub episode opened
        crash: BaseException | None = None
        opening: Opening | None = None
        instruction = None if opened is None else opened.instruction
        opener = self.runtime.opening
        reusable = opener.reusable(episode.reuse, episode.memory, instruction)
        reuse = episode.reuse if reusable else None
        drift: Drift | None = None
        gated = False  # under the method's finish gate (`MethodRuntime.finish`)
        try:
            if binding is not None:
                env = self.trajectories.agent(binding)
                start = await env.robot(binding.episode_id)
                trajectory = self.trajectories.track(binding.episode_id, start)
                if reuse is not None:  # only if the scene still looks as it did
                    first = await env.observe(binding.episode_id, ObserveRequest(depth=False))
                    drift = await opener.drift(reuse, first, self.blobs)
                    reuse = None if drift.too_far else reuse
            instructions, brief = episode.prompt(opened)
            card = shown(self.cards.get(episode.task), instructions)
            step_limit = episode.step_limit if opened is None else opened.step_limit
            role = self.roles[episode.role].for_steps(step_limit)
            if episode.max_turns is not None:  # this episode's cap on the role's turns
                own = role.budgets.max_turns
                capped = episode.max_turns if own is None else min(own, episode.max_turns)
                budgets = role.budgets.model_copy(update={"max_turns": capped})
                role = role.model_copy(update={"budgets": budgets})
            if episode.stalled_calls is not None:  # this episode's stall guard
                stall = {"max_stalled_calls": episode.stalled_calls}
                role = role.model_copy(update={"limits": role.limits.model_copy(update=stall)})
            rules, seed = episode.rules, _seed(episode.frames, reuse)
            rules, seed, gated = self.runtime.finish.episode(
                episode,
                rules,
                seed,
                instruction=instruction,
                actor=actor,
                mounted=self.hub.mounted,
            )
            if actor:  # a call may name a Memory spec: it runs with all the stored fields
                seed = spec_seed(episode.memory, seed)
            session = await self.hub.open_episode(
                OpenEpisode(
                    stage=episode.stage,
                    tier=episode.tier,
                    role=episode.role,
                    toolsets=episode.toolsets,
                    budgets=role.budgets,
                    rules=rules,
                    env=binding,
                    embodiment=self.embodiment,
                    render=role.render,
                    seed=seed,
                )
            )
            if trajectory is not None:
                trajectory.bind(self.hub, session.episode_id)
                if episode.notes is not None:
                    self.hub.set_action_notes(session.episode_id, episode.notes(trajectory))
            if episode.editor is not None:
                self.memory.attach(session.episode_id, episode.editor)
            guard = episode.finish_guard
            if guard is None and control is not None and binding is not None:
                guard = self.runtime.finish.guard(episode, control, binding.episode_id)
            if guard is not None:
                self.hub.set_finish_guard(session.episode_id, guard, times=episode.finish_refusals)
            if episode.shapes and control is not None and binding is not None:
                await _shapes(control, binding.episode_id, episode.directory / REFEREE)
                if trajectory is not None:  # the referee judges placements by the held object
                    await trajectory.watch(partial(_true_poses, control, binding.episode_id))
            brain = self.brains[episode.role](episode)
            self._write_start(episode, session.tools, brain, role, instructions, card)
            if card is not None:
                shown_card = card.record()
                self.events.emit(EventKind.TASK_CARD, shown_card, episode_id=session.episode_id)
            opening = await opener.open(
                self.hub,
                session,
                instruction=instruction,
                memory=episode.memory,
                source=episode.directory.name,
                reuse=reuse,
                drift=drift,
            )
            if opening.reuse is not None:
                save(episode.directory, opening.reuse)
            if not opening.failed:  # a failure ended the episode
                agent = AgentSession(
                    episode=session,
                    hub=self.hub,
                    instructions=instructions,
                    blobs=self.blobs,
                    recorder=Recorder(episode.directory, self.events),
                    models=self.models,
                    brief=[*(brief or []), *opening.brief()] or None,
                    limits=role.limits,
                )
                result = await brain.run(agent)
                if actor and control is not None and binding is not None:
                    judged = await _judge(control, binding.episode_id)
                    if judged is not None and episode.diagnose and not judged.success:
                        diagnosis = await _diagnose(control, binding.episode_id)
                    if judged is not None and self.runtime.judged is not None:
                        hook = self.runtime.judged
                        await hook(self.hub, episode, session.episode_id, judged.success)
        except InfraError as error:
            if session is None:
                failed = error
            else:
                crash = error
        except BaseException as error:  # cancelled or crashed: recorded below, then raised
            crash = error
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    if session is not None:
                        if episode.editor is not None:
                            self.memory.detach(session.episode_id)
                        last = None if trajectory is None else trajectory.samples[-1]
                        await self._keep_objects(episode, session.episode_id, last)
                        if diagnosis is not None:
                            await self._explain(episode, session.episode_id, last, diagnosis)
                        unanswered = self.hub.unanswered_infra(session.episode_id)
                        closed = (await self.hub.close_episode(session.episode_id)).end
                finally:
                    if trajectory is not None and binding is not None:
                        self.trajectories.untrack(binding.episode_id)
                        arrays = trajectory.arrays()
                        np.savez_compressed(episode.directory / PHYSICS, **arrays)  # type: ignore[arg-type]
        if failed is not None:
            end = _end("infra_error", f"the env failed at the start: {failed}")
            return self._summary(episode, started, opened, end, "infra")
        if closed is None:  # the Hub episode never opened
            assert crash is not None
            raise crash
        voided = None  # the end it had and the call, if an unanswered infrastructure error voids it
        if actor and unanswered is not None and voids(closed, judged):
            voided, closed = (closed, unanswered), unanswered_end(closed, unanswered)
        # the method says whether a finish the env does not confirm fails the episode
        refused = self.runtime.finish.unconfirmed(episode, gated)
        if refused is not None and closed.reason == "finish" and actor and not _passed(judged):
            closed = closed.model_copy(update={"reason": refused})
        else:
            refused = None
        has_env = opened is not None
        kind = _classify(closed.reason, judged, actor=actor, has_env=has_env, refused=refused)
        outcome = self._summary(
            episode,
            started,
            opened,
            closed,
            kind,
            judged=judged,
            result=result,
            episode_id=None if session is None else session.episode_id,
            opening=opening,
            voided=voided,
        )
        if judged is not None:
            _write_json(episode.directory / REFEREE / "success.json", judged)
        if diagnosis is not None:
            _write_json(episode.directory / REFEREE / "diagnostics.json", diagnosis)
        if crash is not None:
            raise crash
        return outcome

    async def _keep_objects(
        self, episode: Episode, episode_id: EpisodeId, robot: RobotState | None
    ) -> None:
        """`objects.json`: the registered objects as the episode ends, as a model is shown them
        (for a reviewing role's log of the try); nothing without a registry."""
        try:
            state = self.hub.store.get(episode_id)
            registry = await state.frames.registry()
            lines = []
            for record in registry.objects.values():
                if (record.grasps or record.part_grasps) and robot is None:
                    continue  # held (whole or by a part): where it is needs the robot
                report = object_report(record, robot).model_dump()
                shown = ObjectReport.model_validate(report, context={REFS: state.aliases.alias})
                lines.append(describe(shown))
        except (OSError, LookupError, ValueError):  # the registry's blob, a record, a report
            # A report of the end must not change how the episode ended.
            log.exception("%s: the registered objects could not be listed", episode.directory)
            return
        if lines:
            data = json.dumps(lines, indent=2, ensure_ascii=False) + "\n"
            write_atomic(episode.directory / OBJECTS, data.encode())

    async def _explain(
        self,
        episode: Episode,
        episode_id: EpisodeId,
        robot: RobotState | None,
        diagnosis: Diagnosis,
    ) -> None:
        """`referee/explained.json`: the diagnosis in the terms of the episode's registry and
        aliases, a reviewing role's reading of it (the method's explainer, `MethodRuntime.explain`;
        nothing without one), with the shapes taken at the start (`SHAPES`, where the env gave
        them)."""
        explain = self.runtime.explain
        if explain is None:
            return
        state = self.hub.store.get(episode_id)
        found = episode.directory / REFEREE / SHAPES
        try:
            shapes = None
            if found.is_file():
                saved = json.loads(found.read_bytes())
                shapes = {name: ObjectShape.model_validate(s) for name, s in saved.items()}
            registry = await state.frames.registry()
            explained = explain(
                diagnosis,
                registry,
                lambda obj: state.aliases.alias("obj", obj),
                robot,
                shapes,
                episode.method,
            )
        except (OSError, LookupError, ValueError):  # the registry's blob, a record, the shapes
            # A report of the end must not change how the episode ended: no diagnosis, then.
            log.exception("%s: the diagnosis could not be explained", episode.directory)
            return
        _write_json(episode.directory / REFEREE / EXPLAINED, explained)

    def _summary(
        self,
        episode: Episode,
        started: float,
        opened: EpisodeOpened | None,
        end: EpisodeEnd,
        kind: EndClass,
        *,
        judged: JudgeResult | None = None,
        result: BrainResult | None = None,
        episode_id: EpisodeId | None = None,
        opening: Opening | None = None,
        voided: tuple[EpisodeEnd, str] | None = None,
    ) -> EpisodeOutcome:
        calls, perception = _perception_cost(episode.directory / EVENTS)
        identification = None
        if opening is not None:
            source = None if opening.reuse is None else opening.reuse.source
            identification = Identification(status=opening.status, source=source, note=opening.note)
        outcome = EpisodeOutcome(
            task=episode.task,
            stage=episode.stage,
            role=episode.role,
            seed=None if episode.spec is None else episode.spec.seed,
            directory=episode.directory,
            episode_id=episode_id,
            env_episode=None if opened is None or not opened.solvable else opened.episode_id,
            instruction=None if opened is None else opened.instruction,
            success=kind == "success",
            end_class=kind,
            end=end,
            success_step=None if judged is None else judged.success_step,
            steps=0 if judged is None else judged.steps_used,
            step_limit=None if judged is None else judged.step_limit,
            faulted=judged is not None and judged.faulted,
            prerun_attempts=0 if opened is None else opened.prerun_attempts,
            turns=0 if result is None else result.turns,
            model_calls=0 if result is None else result.model_calls,
            usage=Usage() if result is None else result.usage,
            perception_calls=calls,
            perception_usage=perception,
            identification=identification,
            wall_clock_s=round(time.monotonic() - started, 3),
            memory=_memory_hash(episode),
            provenance=self.provenance,
            voided_end=None if voided is None else voided[0].reason,
            voided_by=None if voided is None else voided[1],
        )
        _write_json(episode.directory / SUMMARY, outcome)
        return outcome

    def _write_start(
        self,
        episode: Episode,
        tools: Sequence[Any],
        brain: Brain,
        role: RoleConfig,
        instructions: str,
        card: TaskCard | None = None,
    ) -> None:
        """`config.json` and `memory_before.md`; `card`, the success condition card the prompt
        shows (left out of `config.json` without one)."""
        brain_config = getattr(brain, "config", None)
        data = {
            "task": episode.task,
            "stage": episode.stage,
            "role": episode.role,
            "tier": episode.tier,
            "env": None if episode.spec is None else episode.spec.model_dump(mode="json"),
            "rules": None if episode.rules is None else episode.rules.model_dump(mode="json"),
            "toolsets": episode.toolsets,
            "seeded_frames": 0 if episode.frames is None else len(episode.frames.frames),
            "seeded_facts": [] if episode.frames is None else sorted(episode.frames.facts),
            "brain": {
                "type": type(brain).__name__,
                "config": brain_config.model_dump(mode="json")
                if isinstance(brain_config, BaseModel)
                else None,
            },
            "role_config": role.model_dump(mode="json"),
            "memory": None
            if episode.memory is None
            else {
                "label": episode.memory_label or episode.memory.label,
                "hash": _memory_hash(episode),
            },
            "prompt": {
                "sha256": hashlib.sha256(instructions.encode()).hexdigest(),
                "templates": self.prompt_digest,
            },
            "tools": {tool.name: tool.hash for tool in tools},
            "source": os.environ.get("RS_SOURCE_SHA"),
            "protocol": episode.extra,
        }
        if card is not None:
            data["card"] = card.record()
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        write_atomic(episode.directory / "config.json", text.encode())
        if episode.memory is not None:
            shown = view(episode.memory, detail=episode.role != "actor")
            write_atomic(episode.directory / "memory_before.md", shown.encode())


async def _judge(control: EnvControl, env_episode: EpisodeId) -> JudgeResult | None:
    """The judgement; None if the env did not answer (an infrastructure outcome)."""
    try:
        return await control.judge(env_episode)
    except InfraError as error:
        log.warning("the env could not judge %s: %s", env_episode, error)
        return None


async def _true_poses(control: EnvControl, env_episode: EpisodeId) -> dict[str, Pose]:
    """The objects' ground-truth poses now (none if the env cannot give them)."""
    try:
        return dict((await control.gt_snapshot(env_episode)).objects)
    except RSError as error:
        log.warning("no ground truth for %s: %s", env_episode, error)
        return {}


async def _shapes(control: EnvControl, env_episode: EpisodeId, directory: Path) -> None:
    """The objects' shapes for the keypoint referee (`referee/shapes.json`); nothing if the env
    cannot give them (the referee then compares strictly)."""
    try:
        snapshot = await control.gt_snapshot(env_episode, geometry=True)
    except RSError as error:
        log.warning("no object shapes for %s: %s", env_episode, error)
        return
    shapes = {
        name: shape.model_dump(mode="json") for name, shape in object_shapes(snapshot).items()
    }

    def write() -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SHAPES).write_text(json.dumps(shapes, indent=2), encoding="utf-8")

    await anyio.to_thread.run_sync(write)


async def _diagnose(control: EnvControl, env_episode: EpisodeId) -> Diagnosis | None:
    """The ground truth after a failure; None if the env could not give it, and the judgement
    stands (a policy outcome stays one)."""
    try:
        return await control.diagnose(env_episode)
    except RSError as error:
        log.warning("the env could not diagnose %s; judged without: %s", env_episode, error)
        return None


async def _close(control: EnvControl, env_episode: EpisodeId) -> None:
    try:
        await control.close(env_episode)
    except InfraError as error:  # the episode's outcome says what went wrong
        log.warning("the env could not close %s: %s", env_episode, error)


def tries_frames(
    tries: Sequence[tuple[int, EpisodeOutcome]],
) -> tuple[EpisodeSeed, dict[int, dict[str, str]]]:
    """The frames the tries (`(try, outcome)`) kept, aliased `log:<try>:<turn>`, and the last
    one's scene facts, to start a reviewing role's episode with; with each try's own aliases mapped
    to these (for the logs it reads)."""
    logs = []
    for attempt, outcome in tries:
        assert outcome.episode_id is not None
        logs.append((attempt, list(read_events(outcome.directory / EVENTS)), outcome.episode_id))
    return seed_from_logs(logs)


def _write_tool_calls(directory: Path) -> None:
    """`tool_calls.jsonl`, a view of the events: each tool call (composites' inner calls, origin
    `service`, too) with its result or why it was refused."""
    calls: dict[str, dict[str, Any]] = {}
    for event in read_events(directory / EVENTS):
        call_id = str(event.data.get("call_id"))
        if event.kind == EventKind.TOOL_CALL:
            calls[call_id] = {"ts": event.ts, **event.data}
        elif event.kind in (EventKind.TOOL_RESULT, EventKind.TOOL_REJECT) and call_id in calls:
            done = {k: v for k, v in event.data.items() if k not in ("call_id", "name", "origin")}
            calls[call_id] |= {"refused": event.kind == EventKind.TOOL_REJECT, **done}
    if calls:
        lines = "".join(json.dumps(call, ensure_ascii=False) + "\n" for call in calls.values())
        write_atomic(directory / "tool_calls.jsonl", lines.encode())


def _seed(frames: EpisodeSeed | None, reuse: Reuse | None) -> EpisodeSeed | None:
    """The episode's seed: its frames, and an identification it starts from."""
    if reuse is None:
        return frames
    if frames is None:
        return reuse.seed
    return EpisodeSeed(
        frames=[*frames.frames, *reuse.seed.frames],
        facts={**frames.facts, **reuse.seed.facts},
        aliases=[*frames.aliases, *reuse.seed.aliases],
    )


def _perception_cost(path: Path) -> tuple[int, Usage]:
    """The perception model calls an episode's log shows, and their tokens."""
    if not path.is_file():
        return 0, Usage()
    calls, total = 0, Counter[str]()
    for event in read_events(path):
        asked = event.data.get("asked") if event.kind == EventKind.PERCEPTION_STEP else None
        if not isinstance(asked, dict):
            continue
        calls += 1
        usage = asked.get("usage") or {}
        total.update({key: int(usage.get(key) or 0) for key in Usage.model_fields})
    return calls, Usage(**total)


def _passed(judged: JudgeResult | None) -> bool:
    return judged is not None and judged.success


def _classify(
    reason: str,
    judged: JudgeResult | None,
    *,
    actor: bool,
    has_env: bool,
    refused: str | None = None,
) -> EndClass:
    if judged is not None and judged.success:
        return "success"
    if reason == "cancelled":
        return "cancelled"
    if reason.startswith(CONFIG_ENDS):
        return "config"
    if reason in INFRA_ENDS or (judged is not None and judged.faulted):
        return "infra"
    if actor and has_env and judged is None:  # the env did not answer the judgement
        return "infra"
    if reason.startswith("fail_fast:") or (refused is not None and reason == refused):
        return "fail_fast"
    if not actor and reason == "finish":  # a role that does not act did its job
        return "success"
    return "policy"


def voids(end: EpisodeEnd, judged: JudgeResult | None) -> bool:
    """Whether an unanswered infrastructure error voids an Actor's episode that closed on `end`
    and was judged `judged` (`agent.session.unanswered_end`): not if the task's success latched
    (the episode ended there, or the benchmark's success check passed whatever ended it), nor if it
    ended on infrastructure already, was cancelled or ended on a configuration error."""
    if end.reason == "success" or _passed(judged):
        return False
    reason = end.reason
    return reason not in INFRA_ENDS and reason != "cancelled" and not reason.startswith(CONFIG_ENDS)


def _end(reason: str, note: str) -> EpisodeEnd:
    return EpisodeEnd(reason=reason, note=note, at=time.time())


def _memory_hash(episode: Episode) -> str | None:
    if episode.memory is None:
        return None
    return episode.memory_hash or content_hash(episode.memory)


def set_aside(directory: Path) -> Path | None:
    """Move what an interrupted run left (a seed's directory, a stage's) out of the way, as
    `<name>-aborted<n>`, which it returns."""
    if not directory.exists():
        return None
    n = 1
    while (aside := directory.with_name(f"{directory.name}-aborted{n}")).exists():
        n += 1
    directory.rename(aside)
    return aside


def _write_json(path: Path, model: BaseModel) -> None:
    write_atomic(path, (model.model_dump_json(indent=2) + "\n").encode())
