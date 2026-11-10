"""The fake backend: a robot that exists only in software (docs/how-to/add-a-real-robot.md), for
tests and tutorials.

Each arm's six "joints" are its ee pose in the world, x, y, z in meters then roll, pitch, yaw
in radians (the `mock_bimanual` convention), so forward kinematics is trivial and the TCP follows
from the profile. A joint command is reached at once; a gripper closes the way a real one does,
a fraction `gripper_rate` of the remaining way per command, so that `until_gripper_stable`
has something to wait for. Cameras show a fixed scene: a gradient that shifts with every
command, over a flat depth of 0.8 m. A torso's joints are plain numbers; there is no base.

It records what the server did to it (`commands`, `halts`), which tests assert on, and it refuses
ee-delta and base commands as every backend does (`NOT_CARRIED`).
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from roboscript.contracts.camera import CameraModel
from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import ArmName
from roboscript.spatial.rotation import quat_from_axis_angle, quat_multiply
from roboscript_robots.base.backend import NOT_CARRIED, Capture, Measured
from roboscript_robots.base.config import RobotConfig

_FOVY_DEG = 37.0
_HEAD = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -0.8, 0.6, -0.45], [0.0, -0.6, -0.8, 1.35], [0.0, 0.0, 0.0, 1.0]]
)
_FLAT_DEPTH_M = 0.8


class FakeBackend:
    """A software robot over a `RobotConfig` whose arms have six joints each."""

    def __init__(self, config: RobotConfig, *, gripper_rate: float = 0.9) -> None:
        for name, arm in config.arms.items():
            if len(arm.joints) != 6:
                raise ValueError(
                    f"the fake backend takes an arm's six joints as its ee pose (x, y, z, roll, "
                    f"pitch, yaw); arm {name!r} has {len(arm.joints)}. Give the robot a fake "
                    "backend of its own (RobotSpec.fake)."
                )
        if not 0.0 < gripper_rate <= 1.0:
            raise ValueError("gripper_rate must be in (0, 1]")
        self.config = config
        self.profile = config.profile()
        self.gripper_rate = gripper_rate
        self.connected = False
        self.commands: list[list[float]] = []
        """Every joint command, in order: one per control tick."""
        self.halts = 0
        """How many times the server halted it."""
        self._qpos = config.home_vector()  # joints as commanded; gripper slots as measured

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def measure(self) -> Measured:
        return Measured(list(self._qpos), self.forward_kinematics(self._qpos))

    async def command_joints(self, qpos: Sequence[float]) -> None:
        self.commands.append(list(qpos))
        previous = self._qpos
        self._qpos = list(qpos)
        for arm in self.profile.arms.values():  # grippers lag behind their commands
            i = arm.gripper_index
            self._qpos[i] = previous[i] + self.gripper_rate * (qpos[i] - previous[i])

    async def command_ee_delta(self, deltas: Mapping[ArmName, Sequence[float]]) -> None:
        raise NotImplementedError(f"FakeBackend.command_ee_delta(): {NOT_CARRIED}")

    async def command_base(self, velocity: Sequence[float]) -> None:
        raise NotImplementedError(f"FakeBackend.command_base(): {NOT_CARRIED}")

    async def halt(self) -> None:
        self.halts += 1

    def forward_kinematics(self, qpos: Sequence[float]) -> dict[ArmName, Pose]:
        return {
            name: _ee(qpos[arm.action[0] : arm.action[1]])
            for name, arm in self.profile.arms.items()
        }

    async def capture(self, cameras: Sequence[str], depth: bool) -> dict[str, Capture]:
        shift = len(self.commands) % 256
        out = {}
        for name in cameras:
            spec = self.profile.cameras[name]
            fy = spec.height / 2 / math.tan(math.radians(_FOVY_DEG) / 2)
            K = ((fy, 0.0, spec.width / 2), (0.0, fy, spec.height / 2), (0.0, 0.0, 1.0))
            if spec.arm is None:
                cam2world = _HEAD
            else:
                cam2world = self.forward_kinematics(self._qpos)[spec.arm].matrix()
            model = CameraModel(
                name=name, width=spec.width, height=spec.height, K=K, cam2world=cam2world.tolist()
            )
            x = np.arange(spec.width, dtype=np.int64)[None, :]
            y = np.arange(spec.height, dtype=np.int64)[:, None]
            rgb = np.stack(
                [(x + shift) % 256 + 0 * y, (y + shift) % 256 + 0 * x, np.full_like(x + y, 128)],
                axis=-1,
            ).astype(np.uint8)
            flat = (
                np.full((spec.height, spec.width), _FLAT_DEPTH_M, dtype=np.float32)
                if depth and self.config.depth
                else None
            )
            out[name] = Capture(model, rgb, flat)
        return out


def _ee(joints: Sequence[float]) -> Pose:
    x, y, z, roll, pitch, yaw = joints
    about = [
        quat_from_axis_angle(axis, angle)
        for axis, angle in zip(np.eye(3), (roll, pitch, yaw), strict=True)
    ]
    return Pose.from_list([x, y, z, *quat_multiply(about[2], quat_multiply(about[1], about[0]))])
