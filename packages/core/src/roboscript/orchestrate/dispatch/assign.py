"""Which worker runs which task: an extension point (entry-point group
`roboscript.dispatch_policies`), whose built-in `balanced` assigns once, statically.

Why static and not a work-stealing queue: a worker runs its tasks as one pipeline, whose own
scheduler keeps that machine's cards busy as long as it has more jobs than places; taking a queued
task away from a running pipeline would mean stopping and relaunching it, and every
task started on its own pipeline would start its own GPU services. Balancing the expected cost per
card up front leaves the tail imbalance at about one task, which the longest task bounds anyway.
A lost worker's tasks are the one reassignment: each goes to the least loaded live worker."""

from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol

from roboscript.contracts.ids import TaskName
from roboscript.kernel import entrypoints
from roboscript.orchestrate.dispatch.config import WorkerSpec

GROUP: Final = "roboscript.dispatch_policies"


class AssignmentPolicy(Protocol):
    """`assign` splits the tasks over the workers (every task exactly once; a worker may get
    none); `reassign` picks a live worker for a lost worker's task, given each live worker's
    load (the cost of its unfinished tasks)."""

    def assign(
        self,
        tasks: Sequence[TaskName],
        cost: Callable[[TaskName], float],
        workers: Sequence[WorkerSpec],
    ) -> dict[str, list[TaskName]]: ...

    def reassign(
        self,
        task: TaskName,
        cost: Callable[[TaskName], float],
        workers: Sequence[WorkerSpec],
        loads: Mapping[str, float],
    ) -> str: ...


class Balanced:
    """Longest expected cost first, each to the worker whose cost per GPU stays lowest (ties: the
    worker listed first)."""

    def assign(
        self,
        tasks: Sequence[TaskName],
        cost: Callable[[TaskName], float],
        workers: Sequence[WorkerSpec],
    ) -> dict[str, list[TaskName]]:
        loads = {w.name: 0.0 for w in workers}
        found: dict[str, list[TaskName]] = {w.name: [] for w in workers}
        order = sorted(range(len(tasks)), key=lambda i: (-cost(tasks[i]), i))
        for i in order:
            name = self.reassign(tasks[i], cost, workers, loads)
            loads[name] += cost(tasks[i])
            found[name].append(tasks[i])
        return {name: sorted(assigned, key=tasks.index) for name, assigned in found.items()}

    def reassign(
        self,
        task: TaskName,
        cost: Callable[[TaskName], float],
        workers: Sequence[WorkerSpec],
        loads: Mapping[str, float],
    ) -> str:
        if not workers:
            raise ValueError(f"no live worker for {task}")
        best = min(workers, key=lambda w: (loads.get(w.name, 0.0) + cost(task)) / len(w.gpus))
        return best.name


def load_policy(name: str) -> AssignmentPolicy:
    policy: AssignmentPolicy = entrypoints.load(GROUP, name)()
    return policy
