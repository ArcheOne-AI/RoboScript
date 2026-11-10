"""`BEHAVIOR`, the BEHAVIOR challenge descriptor. Agent side: imports no OmniGibson code."""

from pathlib import Path
from typing import Final

import yaml

from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.ids import TaskName
from roboscript.envs import Benchmark, TaskInfo

HERE: Final = Path(__file__).resolve().parent
_TABLE: Final = yaml.safe_load((HERE / "tasks.yaml").read_text(encoding="utf-8"))
SCENES: Final[dict[str, str]] = _TABLE["scenes"]
"""The scene each activity is sampled in (placeholders, see tasks.yaml)."""
INSTANCES: Final[int] = _TABLE["instances"]


BEHAVIOR: Final = Benchmark(
    name="behavior",
    tasks={TaskName(n): TaskInfo.model_validate(info) for n, info in _TABLE["tasks"].items()},
    env_server="roboscript_behavior.sim.host",
    interpreter="sim",
    pythonpath=[HERE.parent],
    embodiment=HERE / "embodiments" / "r1_pro_behavior.yaml",
    # An instance of an activity is reset by its index. The verdict is a terminal score (the
    # q-score, `JudgeResult.score`). The challenge runs each instance once, so a method that opens
    # one reset again is refused before the run starts.
    capabilities=EnvCapabilities(
        success={"terminal_score"}, reset={"initial_state"}, truth=True, reruns=False
    ),
    env_config_model="roboscript_behavior.config:BehaviorConfig",
    skillsets=["roboscript_behavior.skills:BEHAVIOR_SKILLS"],
)
