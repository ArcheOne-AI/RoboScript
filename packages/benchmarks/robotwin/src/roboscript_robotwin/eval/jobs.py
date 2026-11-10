"""RoboTwin's evaluation jobs of a unit's frozen Memory (`roboscript.orchestrate.graph.Evaluation`,
docs/how-to/add-a-method.md): its official P3 blocks (`blocks.P3Settings`), as one launch runs them.

    after the unit's Memory froze: evaluate (every mode), else settle (done: evaluated before)
    under `randomized: spare`: clean -> randomized (a spare unit) -> evaluate -> settle
    under `spare_parallel`: clean beside the unit's lane (randomized, a job of its own), then
        lane_waits (the unit waits on its lane) and ended (done once both ended); a unit that
        had ended runs its lane alone (lane_note)
    a launch of P3 modes only (`modes`): reopen (those modes' blocks of a unit that froze)

A unit is not evaluated while `gated` holds it: the caller decides from its learning's outcome
(`roboscript.contracts.jobs.EvaluationFactory`).
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Final

from roboscript.contracts.ids import TaskName
from roboscript.contracts.jobs import ANY, SKIPPED, Edge, JobKind, JobSpec
from roboscript.memory import FileMemoryStore
from roboscript.orchestrate.graph import ComposedGraph, Unit
from roboscript.orchestrate.jobs import RunKey, RunState
from roboscript_robotwin.eval.blocks import P3_PLACES, SPARE, P3Settings, p3_pending
from roboscript_robotwin.eval.kinds import EvaluateJob
from roboscript_robotwin.eval.launch import Mode

PHASE: Final = "evaluate"
SPARE_NOTE: Final = "waits for spare GPU room (P3 randomized)"
LANE_NOTE: Final = f"P3 randomized runs as its {SPARE} lane"
"""A unit's note while its lane (`p3.spare_parallel`) has blocks to run."""
RANDOMIZED: Final[list[Mode]] = ["randomized"]
CLEAN: Final[list[Mode]] = ["clean"]
QUEUED: Final = frozenset({"queued"})
EVERY: Final = frozenset({ANY})

Gated = Callable[[Path, TaskName], bool]
"""Whether the caller holds a unit (its directory, its task) back from P3."""


def reopens(
    p3: P3Settings,
    run_dir: Path,
    state: RunState,
    retry_failed: bool,
    modes: Sequence[Mode],
    gated: Gated,
) -> bool:
    """Whether a unit that froze (it reached P3) has P3 blocks of `modes` to run, unless it is
    `gated`."""
    retry = state.status != "failed" or state.failure == "infra" or retry_failed
    return (
        state.stage == "p3"
        and retry
        and not gated(run_dir, state.task)
        and p3_pending(p3, run_dir, modes)
    )


def lane_due(
    p3: P3Settings,
    run_dir: Path,
    main: RunState,
    lane: RunState,
    retry_failed: bool,
    frozen: bool,
    gated: Gated,
) -> bool:
    """Whether a unit's spare lane (`p3.spare_parallel`) has randomized blocks to run: its
    Memory `frozen`, neither the unit's job nor the lane failed for good (unless
    `retry_failed`), not `gated`, a block of randomized not done."""
    fine = all(s.status != "failed" or s.failure == "infra" or retry_failed for s in (main, lane))
    return frozen and fine and not gated(run_dir, main.task) and p3_pending(p3, run_dir, RANDOMIZED)


class P3Jobs:
    """RoboTwin's evaluation part of a launch's job graph, under `p3` (a launch of P3 `modes`
    only reopens those)."""

    phases = (PHASE,)
    settings: ClassVar[type[P3Settings]] = P3Settings
    """The model of its settings (`EvaluationFactory`)."""
    kinds: ClassVar[Mapping[str, JobKind]] = {PHASE: EvaluateJob()}
    """Its jobs' bodies (`EvaluationFactory`, `kinds.EvaluateJob`)."""

    def __init__(self, p3: P3Settings, gated: Gated, modes: Sequence[Mode] | None = None) -> None:
        self.p3, self.gated, self.modes = p3, gated, modes

    def add(self, graph: ComposedGraph, key: RunKey, learned: Edge, begun: Edge) -> None:
        if self.modes is not None:
            graph.add(self, key, "reopen", [learned], job=self._job(graph, key, list(self.modes)))
        elif self.p3.parallel:
            self._parallel(graph, key, learned, begun)
        else:
            last = learned
            if self.p3.spare:
                clean = graph.add(self, key, "clean", [last], job=self._job(graph, key, CLEAN))
                last = Edge(node=clean, accept=QUEUED, skip="pass")
                spare = self._job(graph, key, RANDOMIZED, spare=True)
                last = Edge(node=graph.add(self, key, "randomized", [last], job=spare), skip="pass")
            every = graph.add(self, key, "evaluate", [last], job=self._job(graph, key))
            graph.add(self, key, "settle", [Edge(node=every, skip="pass")], action="settle")

    def _parallel(self, graph: ComposedGraph, key: RunKey, learned: Edge, begun: Edge) -> None:
        lane = RunKey(key.task, key.variant, SPARE)
        noted = graph.add(self, key, "lane_note", [begun], action="lane_note")
        clean = graph.add(self, key, "clean", [learned], job=self._job(graph, key, CLEAN))
        after_note = Edge(node=noted, accept=QUEUED, skip="pass")
        spare = self._job(graph, lane, RANDOMIZED, spare=True)
        ran = graph.add(self, lane, "randomized", [learned, after_note], job=spare)
        cleaned = Edge(node=clean, accept=QUEUED, skip="pass")
        waits = graph.add(self, key, "lane_waits", [cleaned], action="lane_waits")
        ended = [cleaned, Edge(node=ran, accept=EVERY, skip="pass")]
        graph.add(
            self,
            key,
            "ended",
            [*ended, Edge(node=waits, accept=EVERY, skip="pass")],
            action="ended",
        )

    def _job(
        self,
        graph: ComposedGraph,
        key: RunKey,
        modes: list[Mode] | None = None,
        *,
        spare: bool = False,
    ) -> JobSpec:
        return JobSpec(
            task=key.task,
            unit=key.variant,
            lane=key.lane,
            phase=PHASE,
            stage=graph.stages.index(PHASE),
            weight=P3_PLACES * self.p3.shards,
            shards=self.p3.shards,
            spare=spare,
            note=SPARE_NOTE if spare else None,
            modes=modes,
        )

    def _lane_due(self, unit: Unit, frozen: bool) -> bool:
        main = RunKey(unit.key.task, unit.key.variant)
        lane = RunKey(unit.key.task, unit.key.variant, SPARE)
        retry = unit.records.retry_failed
        return lane_due(
            self.p3,
            unit.directory(main),
            unit.state(main),
            unit.state(lane),
            retry,
            frozen,
            self.gated,
        )

    async def due(self, unit: Unit) -> bool:
        main = RunKey(unit.key.task, unit.key.variant)
        state, run_dir, began = unit.state(main), unit.directory(main), unit.began
        retry, step = unit.records.retry_failed, unit.step
        if step == "reopen":
            assert self.modes is not None
            return reopens(self.p3, run_dir, state, retry, self.modes, self.gated)
        if step == "lane_note":  # a unit that ended, its lane due after it froze
            frozen = await FileMemoryStore(run_dir).frozen(unit.key.task) is not None
            due = self._lane_due(unit, frozen)
            return not began and due
        if step == "clean":
            return began and p3_pending(self.p3, run_dir, CLEAN)
        if step == "randomized" and unit.key.lane is not None:
            if not began:
                return unit.outcomes[f"{main.name}:lane_note"] != SKIPPED
            return self._lane_due(unit, True)
        if step == "randomized":
            if began:
                return p3_pending(self.p3, run_dir, RANDOMIZED)
            return reopens(self.p3, run_dir, state, retry, RANDOMIZED, self.gated)
        if step == "evaluate":
            return began and p3_pending(self.p3, run_dir)
        if step == "settle":
            return began and state.status != "done"
        if step == "lane_waits":
            return began and self._lane_due(unit, True)
        if step == "ended":
            lane = RunKey(unit.key.task, unit.key.variant, SPARE).name
            return began or unit.outcomes[f"{lane}:randomized"] != SKIPPED
        raise ValueError(f"P3 has no step {step!r}")

    async def act(self, unit: Unit) -> str:
        state, run_dir = unit.state(), unit.directory()
        if unit.step in ("lane_note", "lane_waits"):
            state.status, state.error = "queued", LANE_NOTE
            state.save(run_dir)
        elif unit.step == "settle":
            state.end(run_dir, "done")
        elif unit.step == "ended":
            self._ended(unit)
        else:
            raise ValueError(f"P3 has no action {unit.step!r}")
        return unit.state().status

    def _ended(self, unit: Unit) -> None:
        """Once a unit's clean blocks and its lane both ended: done if no block is left, or
        failed as its lane failed (a launch again, `retry_failed` for that, takes it up)."""
        state, run_dir = unit.state(), unit.directory()
        if state.status not in ("queued", "done"):
            return
        lane = unit.state(RunKey(unit.key.task, unit.key.variant, SPARE))
        if lane.status == "failed" and p3_pending(self.p3, run_dir, RANDOMIZED):
            state.failure = lane.failure
            state.end(run_dir, "failed", f"its P3 randomized ({SPARE}): {lane.error}")
        elif not p3_pending(self.p3, run_dir):
            state.end(run_dir, "done")
        elif state.status == "done":  # a block left (cut shorter than asked): a launch again
            state.end(run_dir, "queued")
