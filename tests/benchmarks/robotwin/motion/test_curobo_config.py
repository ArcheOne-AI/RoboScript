"""The cuRobo planner's configuration and pure helpers; nothing here touches torch or a GPU."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import roboscript
from roboscript.contracts.geometry import Pose
from roboscript.motion import CuroboConfig

EXAMPLE = (
    Path(roboscript.__file__).parents[4]
    / "examples/simBenchmarks/RoboTwin/configs"
    / "motion.example.yaml"
)
ALOHA_BASE = Pose(p=(0.0, -0.65, 0.0), q=(0.707, 0.0, 0.0, 0.707))


def test_profile_values_resolve_and_overrides_win(tmp_path: Path) -> None:
    right = {
        "yaw_rad": -0.01,
        "curobo_yml": "custom/right.yml",
        "joints": ["a", "b"],
        "ee_to_tcp": [0.1, 0.0, 0.0],
        "base": {"p": [1.0, 2.0, 3.0], "q": [1.0, 0.0, 0.0, 0.0]},
    }
    data = {**yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")), "assets_root": str(tmp_path)}
    data["arms"] = {**data["arms"], "right": right}
    directory, setups = CuroboConfig.model_validate(data).resolve()
    left, custom = setups["left"], setups["right"]
    assert directory == tmp_path and left.curobo_yml == tmp_path / "curobo_left.yml"
    assert left.joints[0] == "fl_joint1" and left.ee_to_tcp == (0.12, 0.0, 0.0)
    assert left.base == ALOHA_BASE and left.yaw_rad == -0.02
    assert custom.curobo_yml == tmp_path / "custom" / "right.yml" and custom.joints == ["a", "b"]
    assert custom.ee_to_tcp == (0.1, 0.0, 0.0) and custom.base.p == (1.0, 2.0, 3.0)
    assert left.stops is not None and left.stops[4] == (-1.69, 1.76)  # the profile's wrist stops
    assert custom.stops is None  # stops belong to the profile's joints, not overridden ones


@pytest.mark.parametrize(
    "change",
    [
        {"embodiment": "../elsewhere/robot.yaml"},
        {"assets_root": "assets/aloha-agilex"},  # relative: ambiguous across working dirs
        {"arms": {"left": {"curobo_yml": "../curobo_left.yml"}}},
        {"arms": {"left": {"curobo_yml": "/tmp/curobo_left.yml"}}},
        {"device": "cuda:1"},  # the GPU is chosen with CUDA_VISIBLE_DEVICES
    ],
)
def test_paths_stay_inside_their_roots_and_the_device_is_the_first_visible(
    change: dict[str, object],
) -> None:
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        CuroboConfig.model_validate({**data, **change})
