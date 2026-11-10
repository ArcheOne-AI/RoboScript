"""Embodiment profiles (docs/concepts/embodied-core.md): what toolsets know about a robot, kept
as data.

A profile is a YAML file, loaded by path or by a name registered in the `roboscript.embodiments`
entry-point group. The core ships `mock_bimanual`, the robot of `roboscript.envs.mock`; a
benchmark's profile ships with that benchmark's package. Toolsets depend on
profiles only, never on a benchmark.
"""

from pathlib import Path
from typing import Final

from roboscript.embodiments.profile import (
    ArmProfile,
    CameraSpec,
    EmbodimentProfile,
    GraspPreference,
    Gripper,
    JointGroup,
    PlannerAssets,
    load_embodiment,
)

MOCK_BIMANUAL: Final = Path(__file__).with_name("mock_bimanual.yaml")

__all__ = [
    "MOCK_BIMANUAL",
    "ArmProfile",
    "CameraSpec",
    "EmbodimentProfile",
    "GraspPreference",
    "Gripper",
    "JointGroup",
    "PlannerAssets",
    "load_embodiment",
]
