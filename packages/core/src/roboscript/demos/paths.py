"""The paths the keypoint referee reads (docs/concepts/learning.md): each arm's TCP between the
recorded steps of an episode, along the joint line. `take_action('qpos')` moves an arm in a
straight line in joint space from its last drive target to the new one, so the TCP between two
steps is the forward kinematics of the interpolated drive targets (through the `Planner`
contract: the cuRobo service, or `MockPlanner` for MockEnv), and at each step's end the measured
TCP. The gripper command changes where the step that changed it ends. A step's joint line is cut
into pieces of at most `step_rad` of joint motion (the default 0.01 rad moves a TCP half a meter
from the joint by 5 mm), and into at least as many as `step_m` pieces of the chord between its
measured ends.
"""

import math
from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import MAX_JOINT_VECTORS, FKRequest, FKResult
from roboscript.contracts.robot import Arm
from roboscript.demos import ArmPath
from roboscript.embodiments import EmbodimentProfile

Array = NDArray[np.float64]
_WORLD = Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))


class Kinematics(Protocol):
    """What the paths need of a `Planner`: its forward kinematics (a planner service, or
    `motion.urdf_fk.UrdfKinematics` on the CPU)."""

    async def fk(self, request: FKRequest) -> FKResult: ...


async def fk(
    planner: Kinematics, profile: EmbodimentProfile, arm: Arm, joints: Array
) -> tuple[Array, Array]:
    """The ee and TCP poses (N, 7) of an arm's joint vectors (N, J), through `planner`."""
    if not len(joints):
        return np.zeros((0, 7)), np.zeros((0, 7))
    spec = profile.arms[arm]
    # Profiles without planner assets (mock_bimanual) plan in the world frame.
    base = _WORLD if spec.planner is None else spec.planner.base
    ee: list[list[float]] = []
    tcp: list[list[float]] = []
    for at in range(0, len(joints), MAX_JOINT_VECTORS):
        chunk = joints[at : at + MAX_JOINT_VECTORS].tolist()
        result = await planner.fk(FKRequest(arm=arm, base=base, joints=chunk))
        ee += [pose.as_list() for pose in result.ee]
        tcp += [pose.as_list() for pose in result.tcp]
    return np.asarray(ee, dtype=float), np.asarray(tcp, dtype=float)


class ArmPaths:
    """Each arm's path through an episode's samples (`physics.npz`, or `Trajectory.arrays()`),
    extended as samples come (`extend` takes only the new ones), with the turn of every point."""

    def __init__(
        self, planner: Kinematics, profile: EmbodimentProfile, step_rad: float, step_m: float
    ) -> None:
        self.planner = planner
        self.profile = profile
        self.step_rad = step_rad
        self.step_m = step_m
        self._taken = 0
        self._tcp: dict[Arm, list[Array]] = {arm: [] for arm in profile.arms}
        self._grip: dict[Arm, list[Array]] = {arm: [] for arm in profile.arms}
        self._turn: dict[Arm, list[Array]] = {arm: [] for arm in profile.arms}
        self._source: dict[Arm, list[Array]] = {arm: [] for arm in profile.arms}

    async def extend(self, samples: Mapping[str, np.ndarray]) -> None:
        count = len(samples["step"])
        first = self._taken
        if count <= first:
            return
        for arm, spec in self.profile.arms.items():
            q = np.asarray(samples["qpos_cmd"], dtype=float)[:, spec.action[0] : spec.action[1]]
            tcp = np.asarray(samples[f"{arm}.tcp"], dtype=float)
            grip = np.asarray(samples[f"{arm}.gripper_cmd"], dtype=float)
            turn = np.asarray(samples["turn"])
            if first == 0:
                self._add(arm, tcp[:1], grip[:1], turn[:1])
                self._source[arm].append(np.zeros(1, dtype=int))
            steps = range(max(first, 1), count)
            lines = [self._line(q[i - 1], q[i], tcp[i - 1], tcp[i]) for i in steps]
            inside = await self._fk(arm, np.concatenate(lines) if lines else np.zeros((0, 1)))
            at = 0
            for i, line in zip(steps, lines, strict=True):
                ends = len(line)
                self._add(
                    arm,
                    np.concatenate([inside[at : at + ends], tcp[i : i + 1]]),
                    np.r_[np.full(ends, grip[i - 1]), grip[i]],
                    np.full(ends + 1, turn[i]),
                )
                self._source[arm].append(np.full(ends + 1, i))
                at += ends
        self._taken = count

    def paths(self) -> dict[Arm, ArmPath]:
        return {
            arm: ArmPath(
                tcp=np.concatenate(self._tcp[arm]), gripper=np.concatenate(self._grip[arm])
            )
            for arm in self.profile.arms
        }

    def sources(self) -> dict[Arm, np.ndarray]:
        """The recorded sample each path point leads to (the start: 0)."""
        return {arm: np.concatenate(self._source[arm]) for arm in self.profile.arms}

    def turns(self) -> dict[Arm, np.ndarray]:
        """The turn each path point belongs to (0: the start)."""
        return {arm: np.concatenate(self._turn[arm]) for arm in self.profile.arms}

    def _line(self, q0: Array, q1: Array, tcp0: Array, tcp1: Array) -> Array:
        """The joint vectors strictly inside the joint line from `q0` to `q1`."""
        joints = float(np.max(np.abs(q1 - q0), initial=0.0)) / self.step_rad
        travel = float(np.linalg.norm(tcp1[:3] - tcp0[:3])) / self.step_m
        n = max(1, math.ceil(max(joints, travel)))
        t = np.arange(1, n)[:, None] / n
        return q0 + t * (q1 - q0)

    async def _fk(self, arm: Arm, joints: Array) -> Array:
        return (await fk(self.planner, self.profile, arm, joints))[1]

    def _add(self, arm: Arm, tcp: Array, grip: Array, turn: Array) -> None:
        self._tcp[arm].append(tcp)
        self._grip[arm].append(grip)
        self._turn[arm].append(turn)
