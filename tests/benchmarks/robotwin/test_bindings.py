"""RoboTwin's bindings: the planner profile has two sources, the binding
(`roboscript_robotwin.bindings`), which the toolsets plan with, and the motion service's example
config, which the service plans with; and the perception binding (`roboscript_robotwin.config`)
resolves RoboTwin's anchor camera, table area and prompt words, value for value, while a config's
own values stay."""

from typing import Any

from roboscript.motion.config import CuroboConfig
from roboscript.perception.config import PerceptionConfig, TableConfig
from roboscript.settings import load_config_ref, repo_root
from roboscript_robotwin.bindings import PLANNER_PROFILE
from roboscript_robotwin.config import with_perception_binding, with_perception_place
from roboscript_robotwin.eval.launch import P3Config


def test_the_motion_example_plans_with_the_bindings_profile() -> None:
    example = repo_root() / "examples/simBenchmarks/RoboTwin/configs" / "motion.example.yaml"
    assert load_config_ref(str(example), CuroboConfig).planner_profile == PLANNER_PROFILE


ROBOTWIN: dict[str, Any] = {
    "anchor": "head_camera",
    "scene_words": "scene",
    "anchor_words": "a robot's camera",
    "table": {"xy_box": (-0.62, 0.62, -0.37, 0.37), "z_range": (0.60, 0.80)},
}


"""RoboTwin's anchor camera and table area."""


WORDS = {
    "scene_words": "tabletop",
    "anchor_words": "a robot's head camera",
    "camera_words": {
        "left_camera": "a camera on the robot's left wrist",
        "right_camera": "a camera on the robot's right wrist",
    },
}


SCENE = PerceptionConfig.model_validate({**ROBOTWIN, **WORDS})


"""What RoboTwin's scene toolsets perceive with: its camera and table with its prompt words."""


TABLE = PerceptionConfig.model_validate(ROBOTWIN)


"""What RoboTwin's manipulation toolset finds the demonstration's table with."""


def test_the_binding_resolves_robotwins_values() -> None:
    assert PerceptionConfig.model_validate(with_perception_binding({})) == SCENE
    assert PerceptionConfig.model_validate(with_perception_place({})) == TABLE
    assert SCENE.table == TableConfig(**ROBOTWIN["table"])
    own = with_perception_binding({"anchor": "front", "table": {"z_range": (0.1, 0.2)}})
    assert own["anchor"] == "front"
    assert own["table"] == {"xy_box": ROBOTWIN["table"]["xy_box"], "z_range": (0.1, 0.2)}


def test_p3_resolves_robotwins_values() -> None:
    data = {
        "host": {
            "toolsets": [
                {"name": "manipulation"},
                {"name": "perception"},
                {"name": "geometry", "config": {"pipeline": {"cell": 0.005}}},
            ]
        }
    }
    manipulation, perception, geometry = P3Config._perception_binding(data)["host"]["toolsets"]
    assert PerceptionConfig.model_validate(manipulation["config"]["table_perception"]) == TABLE
    assert PerceptionConfig.model_validate(perception["config"]["pipeline"]) == SCENE
    pipeline = PerceptionConfig.model_validate(geometry["config"]["pipeline"])
    assert pipeline == SCENE.model_copy(update={"cell": 0.005})
