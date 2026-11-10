"""The runner: tasks in parallel on one Hub, one protocol at a time per task (its env serves one
episode at a time)."""

from typing import Any

import anyio

from roboscript.contracts.ids import TaskName
from roboscript.envs.mock import Lifted, MockConfig, MockTask
from roboscript.learning import Evaluate, EvaluateResult
from roboscript.testing.protocols import LIFT_RED, acting, lift_memory
from roboscript_methods.wicl.practice import Practice, PracticeResult

LIFT_BLUE = TaskName("lift_blue")


async def test_two_tasks_in_parallel_on_one_runner(setup: Any) -> None:
    task = MockTask(
        primary="lift the blue box", paraphrase="raise the blue block", success=Lifted(box="blue")
    )
    setup.add(MockConfig(task=LIFT_BLUE, blob_dir=setup.tmp / "blobs", tasks={LIFT_BLUE: task}))
    for name in (LIFT_RED, LIFT_BLUE):
        await setup.start(lift_memory(0.06, task=name))
    async with setup.runner({"actor": acting(lambda _: "pass")}) as runner:
        done = await runner.run_tasks(
            {name: [(Practice(), {"start": 0})] for name in (LIFT_RED, LIFT_BLUE)}
        )
    for name in (LIFT_RED, LIFT_BLUE):
        [result] = done[name]  # type: ignore[misc]
        assert isinstance(result, PracticeResult) and (result.reason, result.k) == ("converged", 3)
        assert (setup.run_dir / name / "S2" / "stop.json").is_file()
    assert {spec.task for spec in setup.envs[LIFT_BLUE].opened} == {LIFT_BLUE}


async def test_one_protocol_at_a_time_per_task(setup: Any) -> None:
    await setup.start(lift_memory(0.06))
    results: dict[str, EvaluateResult] = {}
    async with setup.runner({"actor": acting(lambda _: "pass")}) as runner:

        async def evaluate(name: str, seeds: list[int]) -> None:
            config = {"version": 0, "seeds": seeds, "name": name}
            result = await runner.run(Evaluate(), LIFT_RED, config)
            assert isinstance(result, EvaluateResult)
            results[name] = result

        async with anyio.create_task_group() as group:
            group.start_soon(evaluate, "a", [20000, 20001])
            group.start_soon(evaluate, "b", [30000, 30001])
    seeds = [spec.seed for spec in setup.env.opened]
    assert seeds in ([20000, 20001, 30000, 30001], [30000, 30001, 20000, 20001])
    assert {name: r.successes for name, r in results.items()} == {"a": 2, "b": 2}
