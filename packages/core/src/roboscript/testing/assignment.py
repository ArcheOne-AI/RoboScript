"""Conformance checks for dispatch assignment policies (`roboscript.orchestrate.dispatch.assign`).

A policy conforms when it assigns every task to exactly one of the given workers, gives the same
answer for the same input, and reassigns a task to one of the live workers it is given. Use them
from pytest by subclassing `AssignmentConformance` as a `Test*` class with a `policy_target`
fixture (the policy), or without pytest via `roboscript.testing.precheck`.
"""

from roboscript.contracts.ids import TaskName
from roboscript.orchestrate.dispatch.assign import AssignmentPolicy
from roboscript.orchestrate.dispatch.config import WorkerSpec
from roboscript.testing import unit

TASKS = [TaskName(f"task{i}") for i in range(7)]
COSTS = {task: float(1 + (i * 5) % 7) for i, task in enumerate(TASKS)}
WORKERS = [
    WorkerSpec(name=f"w{i}", host=f"h{i}", code_dir="/c", gpus=list(range(n)), run_root="/r")
    for i, n in enumerate((1, 2, 4))
]


class AssignmentConformance:
    """The checks, as `test_*` methods taking an `AssignmentPolicy`."""

    @unit
    async def test_every_task_once(self, policy_target: AssignmentPolicy) -> None:
        found = policy_target.assign(TASKS, COSTS.__getitem__, WORKERS)
        assert set(found) <= {w.name for w in WORKERS}
        assert sorted(t for tasks in found.values() for t in tasks) == sorted(TASKS)
        assert found == policy_target.assign(TASKS, COSTS.__getitem__, WORKERS)

    @unit
    async def test_reassign_to_a_live_worker(self, policy_target: AssignmentPolicy) -> None:
        live = WORKERS[1:]
        name = policy_target.reassign(TASKS[0], COSTS.__getitem__, live, {"w1": 5.0, "w2": 0.0})
        assert name in {w.name for w in live}
