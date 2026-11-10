"""The pose-math ports against RoboTwin's own functions on every task (sim.posemath_check).

Run from the repository root in the simulator env, ROBOTWIN_ROOT in the environment or .env (or
the file RS_ENV_FILE names), on the GPU CUDA_VISIBLE_DEVICES names:

    python examples/simBenchmarks/RoboTwin/scripts/posemath_equivalence.py [--seeds 2] [--parallel
    5] [tasks...]

One process per task (a process switching tasks leaks GPU memory), `--parallel` at a time; prints
the run directory under runs/posemath/, which holds `tasks/<task>.jsonl` (one row per traced call),
`<task>.log`, and `summary.json`: per function the calls, the largest difference of the port's
result from the official one as run (float32 SAPIEN poses) and run again in float64, and the calls
beyond 1e-6.
"""

import argparse
import json
import subprocess
import sys
import time
from typing import Any

from roboscript.settings import repo_root
from roboscript_robotwin.benchmark import ROBOTWIN

LIMIT = 1e-6


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "calls": len(rows),
        "tasks": len({row["task"] for row in rows}),
        "limit": LIMIT,
    }
    for fn in sorted({row["fn"] for row in rows}):
        mine = [row for row in rows if row["fn"] == fn]
        for name in ("diff", "diff64"):
            values = [row[name] for row in mine if row[name] is not None]
            summary[f"{fn}.{name}.max"] = max(values, default=0.0)
            summary[f"{fn}.{name}.beyond"] = sum(value > LIMIT for value in values)
            summary[f"{fn}.{name}.none_mismatch"] = sum(
                row[name] is None and not row[f"{name}_same_none"] for row in mine
            )
        summary[f"{fn}.calls"] = len(mine)
        summary[f"{fn}.targets_diff.max"] = max(row["targets_diff"] for row in mine)
        summary[f"{fn}.unused_plans"] = sum(row["unused_plans"] for row in mine)
        summary[f"{fn}.tilt_mismatch"] = sum(not row["tilt"] for row in mine)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*", help="default: all of RoboTwin's tasks")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--start", type=int, default=10000)
    options = parser.parse_args()
    out = repo_root() / "runs" / "posemath" / time.strftime("%Y%m%d-%H%M%S")
    (out / "tasks").mkdir(parents=True)
    todo = list(options.tasks or sorted(ROBOTWIN.tasks))
    running: list[subprocess.Popen[bytes]] = []
    while todo or running:
        running = [p for p in running if p.poll() is None]
        while todo and len(running) < options.parallel:
            task = todo.pop(0)
            command = [
                sys.executable,
                "-m",
                "roboscript_robotwin.sim.posemath_check",
                "--task",
                task,
                "--seeds",
                str(options.seeds),
                "--start",
                str(options.start),
                "--out",
                str(out / "tasks" / f"{task}.jsonl"),
            ]
            with (out / f"{task}.log").open("wb") as log:
                running.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(2)
    rows = [
        json.loads(line)
        for path in sorted((out / "tasks").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    (out / "summary.json").write_text(
        json.dumps(summarize(rows), indent=1) + "\n", encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()
