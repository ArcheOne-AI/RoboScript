"""wicl's learning jobs (`Wicl.jobs`): each task's S1 `imitate`, then each unit's (a
variant's) S2 `practice`, then the benchmark's evaluation of the Memory it froze
(`roboscript.orchestrate.graph.Evaluation`), as one launch runs them. A launch of S1s only
(`s1_only`) has no unit's nodes; one that only evaluates (`evaluate_only`) no S2. A unit whose
Memory froze runs its practice phase again when `practice_again` says so (for example, when
the end targets of the version its S2 pick froze are still to be read).

    <task>/S1: imitate -> block (the units wait when it did not pass)
    <task>/<unit>, after imitate done: practice -> the evaluation's nodes
"""

from collections.abc import Callable, Sequence
from typing import Final

from roboscript.contracts.ids import TaskName
from roboscript.contracts.jobs import Edge, Evaluation, JobSpec
from roboscript.memory import FileMemoryStore
from roboscript.orchestrate.graph import ComposedGraph, Unit
from roboscript.orchestrate.jobs import RunKey, pending
from roboscript_methods.wicl.ops import S1

PHASES: Final = ("imitate", "practice")
"""The phases of wicl's jobs, in the order `by_stage` places them."""
DONE: Final = frozenset({"done"})
QUEUED: Final = frozenset({"queued"})
NOT_PASSED: Final = frozenset({"not_passed", "failed"})


class LearnJobs:
    """wicl's part of a launch's job graph: its `units` per task, after the S1 each shares,
    and the benchmark's `evaluation` of each unit."""

    def __init__(
        self,
        units: Sequence[str],
        evaluation: Evaluation,
        *,
        s1_only: bool = False,
        evaluate_only: bool = False,
        practice_again: Callable[[RunKey], bool] | None = None,
    ) -> None:
        self.units, self.evaluation = list(units), evaluation
        self.s1_only, self.evaluate_only = s1_only, evaluate_only
        self.practice_again = practice_again
        self.stages = (*PHASES, *evaluation.phases)

    def add(self, graph: ComposedGraph, task: TaskName) -> None:
        """The task's nodes."""
        s1 = RunKey(task, S1)
        first = graph.add(self, s1, "imitate", job=self._job(s1, "imitate"))
        graph.add(
            self, s1, "block", [Edge(node=first, accept=NOT_PASSED, skip="status")], action="block"
        )
        if self.s1_only:
            return
        begun = Edge(node=first, accept=DONE, skip="status")
        for unit in self.units:
            key = RunKey(task, unit)
            if self.evaluate_only:
                self.evaluation.add(graph, key, begun, begun)
                continue
            practice = graph.add(self, key, "practice", [begun], job=self._job(key, "practice"))
            learned = Edge(node=practice, accept=QUEUED, skip="pass")
            self.evaluation.add(graph, key, learned, begun)

    def _job(self, key: RunKey, phase: str) -> JobSpec:
        return JobSpec(
            task=key.task,
            unit=key.variant,
            phase=phase,
            stage=self.stages.index(phase),
            first=phase == PHASES[0],
        )

    async def due(self, unit: Unit) -> bool:
        state = unit.state()
        if unit.step == "imitate":
            return not self.evaluate_only and pending(state, unit.records.retry_failed)
        if unit.step == "block":
            return not self.evaluate_only
        if unit.step == "practice":
            frozen = await FileMemoryStore(unit.directory()).frozen(unit.key.task) is not None
            again = frozen and self.practice_again is not None and self.practice_again(unit.key)
            return unit.began and (not frozen or again)
        raise ValueError(f"wicl has no step {unit.step!r}")

    async def act(self, unit: Unit) -> str:
        if unit.step != "block":
            raise ValueError(f"wicl has no action {unit.step!r}")
        s1 = unit.state()
        why = f"S1 {s1.status}" + (f": {s1.error}" if s1.error else "")
        for name in self.units:  # a task's S1 did not pass: its units wait for it
            key = RunKey(unit.key.task, name)
            state = unit.state(key)
            if pending(state, unit.records.retry_failed):
                state.status, state.error = "blocked", why
                state.save(unit.directory(key))
        return s1.status
