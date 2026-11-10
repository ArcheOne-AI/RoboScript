"""`ROBOTWIN`, the RoboTwin 2.0 benchmark descriptor. Agent side: imports no simulator code."""

from pathlib import Path
from typing import Final

import yaml

from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.ids import TaskName
from roboscript.envs import Benchmark, EvalSeeds, TaskInfo

HERE: Final = Path(__file__).resolve().parent
TASK_CONFIGS: Final = {
    "clean": HERE / "task_configs" / "rs_agentic_clean.yml",
    "randomized": HERE / "task_configs" / "rs_agentic_randomized.yml",
}
# The same with the official video on, for RoboTwin's own evaluation client (P3).
P3_TASK_CONFIGS: Final = {
    "clean": HERE / "task_configs" / "rs_agentic_clean_p3.yml",
    "randomized": HERE / "task_configs" / "rs_agentic_randomized_p3.yml",
}


def _tasks() -> dict[TaskName, TaskInfo]:
    tasks = yaml.safe_load((HERE / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]
    return {TaskName(name): TaskInfo.model_validate(info) for name, info in tasks.items()}


ROBOTWIN: Final = Benchmark(
    name="robotwin",
    tasks=_tasks(),
    env_server="roboscript_robotwin.sim.host",
    interpreter="sim",
    pythonpath=[HERE.parent],
    embodiment=HERE / "embodiments" / "aloha_agilex.yaml",
    env_config={"mode": "clean"},
    # The success latch after every action, seeds, the ground truth and diagnosis, the official
    # expert pre-run and trace, and recorded demonstrations (scripts/record_demos.py).
    capabilities=EnvCapabilities(
        success={"latch"}, reset={"seed"}, truth=True, expert=True, demos=True
    ),
    env_config_model="roboscript_robotwin.config:RoboTwinConfig",
    eval_seeds=EvalSeeds(first=100000, stride=100000),  # the official blocks
    skillsets=["robotwin"],
)
