"""Dispatch the RoboTwin pipeline over several machines (roboscript.orchestrate.dispatch,
docs/how-to/dispatch.md): each worker runs scripts/run_pipeline.py over its share of the tasks,
and each finished task moves back here.

    PYTHONPATH=examples/simBenchmarks/RoboTwin setsid nohup .venv/bin/python \\
        examples/simBenchmarks/RoboTwin/scripts/dispatch.py run \\
        examples/simBenchmarks/RoboTwin/configs/dispatch.yaml > dispatch.log 2>&1 &

    ... run <config> [--set k=v] [--costs-from <pipeline dir>/summary.json]
    ... check <config>     validate the config and print the assignment; start nothing
    ... resume|stop|status <root>/<name>   as python -m roboscript.orchestrate.dispatch

Before anything starts, the pipeline config every worker would run is validated as the pipeline's
own `PipelineConfig`, and tasks without a cost get their official step limit (with
`--costs-from`, times the S1 attempts and S2 seeds an earlier pipeline spent on them).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anyio
from robotwin_pipeline import PipelineConfig

from roboscript.contracts.ids import TaskName
from roboscript.orchestrate.dispatch import Coordinator, DispatchConfig, load_dispatch
from roboscript.orchestrate.dispatch.__main__ import main as dispatch_main
from roboscript.orchestrate.dispatch.assign import load_policy
from roboscript.orchestrate.dispatch.config import load_pipeline, worker_config
from roboscript.orchestrate.dispatch.coordinator import status_table
from roboscript_robotwin.benchmark import ROBOTWIN


def costs(config: DispatchConfig, history: Path | None) -> dict[TaskName, float]:
    """The config's costs, else a task's official step limit, times the attempts and seeds an
    earlier pipeline's summary shows for it (its S1 attempts plus each variant's S2 seeds)."""
    spent: dict[str, int] = {}
    if history is not None:
        for row in json.loads(history.read_bytes())["rows"]:
            task = row["run"].split("/", 1)[0]
            n = (row.get("s1") or {}).get("attempts", 0) + (row.get("s2") or {}).get("seeds", 0)
            spent[task] = spent.get(task, 0) + n
    found = {}
    for task in config.tasks:
        if task in config.costs:
            found[task] = config.costs[task]
        else:
            limit = ROBOTWIN.tasks[task].step_limit
            found[task] = float(limit or 1) * max(1, spent.get(task, 1))
    return found


def checked(path: Path, overrides: list[str], history: Path | None) -> DispatchConfig:
    config = load_dispatch(path, overrides)
    config = config.model_copy(update={"costs": costs(config, history)})
    pipeline = load_pipeline(config)
    try:
        PipelineConfig.model_validate(pipeline)
        for worker in config.workers:
            PipelineConfig.model_validate(
                worker_config(pipeline, config.unit, worker, config.tasks)
            )
    except ValueError as error:
        raise SystemExit(
            f"the pipeline config the workers would run is invalid: {error}"
        ) from error
    return config


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    if given[:1] in (["resume"], ["stop"], ["status"]):
        return dispatch_main(given)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["run", "check"])
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", action="append", default=[], help="a config override key=value")
    ap.add_argument("--costs-from", type=Path, help="an earlier pipeline's summary.json")
    args = ap.parse_args(given)
    config = checked(args.config, args.set, args.costs_from)
    if args.command == "check":
        assigned = load_policy(config.policy).assign(config.tasks, config.cost, config.workers)
        for worker in config.workers:
            share: list[Any] = assigned[worker.name]
            total = sum(config.cost(t) for t in share)
            print(f"{worker.name} ({len(worker.gpus)} GPUs, cost {total:g}): {', '.join(share)}")
        return 0
    found = anyio.run(Coordinator(config).run)
    print(status_table(found))
    return 0 if set(found["tasks"]) <= {"moved"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
