"""roboscript.motion.urdf_fk on the aloha-agilex arms: the CPU forward kinematics a launch rechecks
passed S1s with must give the referee the paths the cuRobo planner gave it. The fixture holds one
recorded attempt and the referee's paths cuRobo made from it (`demos.paths.ArmPaths` through the
motion service, 1237 path points); the URDF chains need RoboTwin's assets (ROBOTWIN_ROOT)."""

from pathlib import Path

import anyio
import numpy as np
import pytest

from roboscript.demos.paths import ArmPaths
from roboscript.motion.config import CuroboConfig
from roboscript.motion.urdf_fk import UrdfKinematics
from roboscript.settings import load_config_ref

FIXTURE = Path(__file__).parent / "data" / "curobo_fk_attempt.npz"


def _kinematics() -> UrdfKinematics:
    try:
        return UrdfKinematics(
            load_config_ref(
                "examples/simBenchmarks/RoboTwin/configs/motion.example.yaml", CuroboConfig
            )
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        pytest.skip(f"RoboTwin's aloha-agilex assets are not here ({error})")


def test_the_cpu_paths_are_curobo_s_on_a_stored_attempt() -> None:
    kinematics = _kinematics()
    stored = dict(np.load(FIXTURE))
    physics = {k.removeprefix("physics."): v for k, v in stored.items() if k.startswith("phys")}
    config = load_config_ref(
        "examples/simBenchmarks/RoboTwin/configs/motion.example.yaml", CuroboConfig
    )
    paths = ArmPaths(kinematics, config.profile(), 0.01, 0.005)
    anyio.run(paths.extend, physics)
    made, sources = paths.paths(), paths.sources()
    for arm in ("left", "right"):
        assert np.array_equal(sources[arm], stored[f"curobo.{arm}.source"])
        assert np.array_equal(made[arm].gripper, stored[f"curobo.{arm}.gripper"])
        # cuRobo computes in float32: well under a millimeter apart, orientation alike
        assert np.abs(made[arm].tcp[:, :3] - stored[f"curobo.{arm}.tcp"][:, :3]).max() < 1e-5
        dots = np.abs(np.sum(made[arm].tcp[:, 3:] * stored[f"curobo.{arm}.tcp"][:, 3:], axis=1))
        assert dots.min() > 1 - 1e-8
