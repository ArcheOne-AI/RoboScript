"""The hardware seam of a robot server (docs/how-to/add-a-real-robot.md).

`RobotBackend` is everything the server needs from a robot's hardware: a connection, the
measured state, camera captures, the robot's forward kinematics, one joint-position target per
control tick, and a halt. The server owns everything else (episodes, limits, the control loop,
the e-stop, the heartbeat), so a backend is thin and a new robot is mostly its vendor SDK.

`HardwareBackend` is the skeleton a robot's model module subclasses. It composes `measure`,
`command_joints` and `capture` from a few hardware points (`read_arm`, `write_arms`,
`read_torso`, `write_torso`, `read_camera`, `forward_kinematics`, `halt`, `connect`,
`disconnect`) and every one of those raises `NotImplementedError` naming itself, so a half-wired
robot fails at the first call that needs the missing point, with a message that says what to
implement. `command_ee_delta` and `command_base` raise `NotImplementedError` with `NOT_CARRIED`:
`execute` carries joint positions only, so no server path calls them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NoReturn, Protocol

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.camera import CameraModel
from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import ArmName
from roboscript_robots.base.config import RobotConfig

NOT_CARRIED: Final = (
    "`env.agent.execute` carries joint positions only, so no server path sends end-effector "
    "delta or base steps to a backend."
)
"""What a backend says where it would need ee_delta or base execution."""


@dataclass(frozen=True)
class ArmReading:
    """One arm as measured: its joints (rad or m, as the arm's joints are), the gripper's
    opening (normalized, 1 open and 0 closed: `Gripper.opening` of its measured joint) and the
    ee's pose in the world frame."""

    joints: list[float]
    gripper: float
    ee: Pose


@dataclass(frozen=True)
class ArmTarget:
    """What to command one arm to: joint positions and the gripper command (1 open, 0 closed)."""

    joints: list[float]
    gripper: float


@dataclass(frozen=True)
class Measured:
    """The robot as measured: the full action vector (joints measured, each gripper's slot its
    measured normalized opening) and each arm's ee pose in the world frame. Never the commands:
    the server keeps those."""

    qpos: list[float]
    ee: dict[ArmName, Pose]


@dataclass(frozen=True)
class Capture:
    """One camera capture: its calibration, RGB (H, W, 3) uint8 and depth (H, W) float32 in
    meters (0: no return), or None where the camera gives none or none was asked for."""

    camera: CameraModel
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32] | None


class RobotBackend(Protocol):
    """A robot's hardware, as the robot server drives it. All methods may be called from the
    server's event loop; none may block it for long (run blocking vendor calls in a worker
    thread)."""

    async def connect(self) -> None:
        """Open the connection and bring the robot to a state it can be commanded from."""
        ...

    async def disconnect(self) -> None:
        """Release the robot (hold or relax it as the vendor advises) and close the connection."""
        ...

    async def measure(self) -> Measured: ...

    async def capture(self, cameras: Sequence[str], depth: bool) -> dict[str, Capture]: ...

    def forward_kinematics(self, qpos: Sequence[float]) -> dict[ArmName, Pose]:
        """Each arm's ee pose in the world frame for a full action vector (grippers ignored).
        Pure and quick: the server calls it for every waypoint of a batch before any motion."""
        ...

    async def command_joints(self, qpos: Sequence[float]) -> None:
        """Set the target of one control tick: the full action vector, grippers as normalized
        commands. The server calls it at the robot's control rate and has already kept the
        target inside the limits."""
        ...

    async def command_ee_delta(self, deltas: Mapping[ArmName, Sequence[float]]) -> None:
        """No server path calls this: raises `NotImplementedError` with `NOT_CARRIED`."""
        ...

    async def command_base(self, velocity: Sequence[float]) -> None:
        """No server path calls this: raises `NotImplementedError` with `NOT_CARRIED`."""
        ...

    async def halt(self) -> None:
        """Stop all motion at once and hold. Called by the e-stop and by the safety watchdog,
        from outside the control loop: it must not wait for a command in flight, and it must be
        safe to call twice."""
        ...


class HardwareBackend:
    """The skeleton a model module subclasses: implement the hardware points below; the
    composition (`measure`, `command_joints`, `capture`) is done."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.profile = config.profile()

    def _unimplemented(self, point: str) -> NoReturn:
        raise NotImplementedError(
            f"{type(self).__name__}.{point}() is a hardware point and is not implemented: "
            f"implement it for {self.config.name!r} with its vendor SDK "
            "(docs/how-to/add-a-real-robot.md)"
        )

    # Hardware points ------------------------------------------------------------------------

    async def connect(self) -> None:
        self._unimplemented("connect")

    async def disconnect(self) -> None:
        self._unimplemented("disconnect")

    async def read_arm(self, arm: ArmName) -> ArmReading:
        self._unimplemented("read_arm")

    async def write_arms(self, targets: Mapping[ArmName, ArmTarget]) -> None:
        """Command every arm for one tick in one call, so that arms move on the same tick."""
        self._unimplemented("write_arms")

    async def read_torso(self) -> list[float]:
        self._unimplemented("read_torso")

    async def write_torso(self, joints: Sequence[float]) -> None:
        self._unimplemented("write_torso")

    async def read_camera(self, name: str, depth: bool) -> Capture:
        self._unimplemented("read_camera")

    def forward_kinematics(self, qpos: Sequence[float]) -> dict[ArmName, Pose]:
        self._unimplemented("forward_kinematics")

    async def halt(self) -> None:
        self._unimplemented("halt")

    async def command_ee_delta(self, deltas: Mapping[ArmName, Sequence[float]]) -> None:
        raise NotImplementedError(f"{type(self).__name__}.command_ee_delta(): {NOT_CARRIED}")

    async def command_base(self, velocity: Sequence[float]) -> None:
        raise NotImplementedError(f"{type(self).__name__}.command_base(): {NOT_CARRIED}")

    # Composed from the points ---------------------------------------------------------------

    async def measure(self) -> Measured:
        qpos = [0.0] * self.config.action_dim
        ee: dict[ArmName, Pose] = {}
        for name, arm in self.profile.arms.items():
            reading = await self.read_arm(name)
            start, stop = arm.action
            qpos[start:stop] = reading.joints
            qpos[arm.gripper_index] = reading.gripper
            ee[name] = reading.ee
        if (span := self.config.torso_range()) is not None:
            qpos[span[0] : span[1]] = await self.read_torso()
        return Measured(qpos, ee)

    async def command_joints(self, qpos: Sequence[float]) -> None:
        targets = {}
        for name, arm in self.profile.arms.items():
            start, stop = arm.action
            targets[name] = ArmTarget(list(qpos[start:stop]), qpos[arm.gripper_index])
        await self.write_arms(targets)
        if (span := self.config.torso_range()) is not None:
            await self.write_torso(qpos[span[0] : span[1]])

    async def capture(self, cameras: Sequence[str], depth: bool) -> dict[str, Capture]:
        return {name: await self.read_camera(name, depth) for name in cameras}
