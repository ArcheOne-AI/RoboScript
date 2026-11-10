"""A pure-numpy `Planner` for unit tests: the `mock_bimanual` robot of `roboscript.envs.mock`.

That robot's six arm "joints" are its ee pose, x, y, z (m) then roll, pitch, yaw (rad) with
R = Rz(yaw) Ry(pitch) Rx(roll), in the frame of `base`: MockEnv's world, so its base is the
identity. FK is exact and IK closed-form; at pitch ±90 deg, where only yaw ∓ roll is determined,
IK keeps the seed's roll. The joint limits bound a workspace over MockEnv's table. A path
interpolates the joints linearly at `speed` units per second; a goal outside the limits is
unreachable. Nothing collides but a single plan's `obstacles`, and those only with the TCP: a start
inside one is `invalid_start`, a goal inside one `unreachable`, a path through one `no_path` (the
mock plans no way around). An arm's collision spheres are its hand: one at the ee, one at the TCP
(`HAND_RADIUS_M`).
"""

import math
import time
from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.geometry import OrientedBox, Pose
from roboscript.contracts.motion import (
    ArmInfo,
    BatchPlanRequest,
    Constraint,
    FKRequest,
    FKResult,
    IKRequest,
    IKSolution,
    PlannerInfo,
    PlanReason,
    PlanRequest,
    PlanResult,
)
from roboscript.contracts.robot import Arm
from roboscript.embodiments import MOCK_BIMANUAL, ArmProfile, EmbodimentProfile, load_embodiment
from roboscript.motion.validate import arm_of, check_joints, check_links
from roboscript.spatial.pose import pose_compose, pose_inverse
from roboscript.spatial.rotation import (
    quat_angle_deg,
    quat_from_axis_angle,
    quat_multiply,
    quat_to_matrix,
)

LOWER: Final = (-0.6, -0.5, 0.70, -math.pi, -math.pi / 2, -math.pi)
UPPER: Final = (0.6, 0.5, 1.4, math.pi, math.pi / 2, math.pi)
"""The workspace: 0.70 m is MockEnv's table top; pitch spans ±90 deg, so Euler angles are unique
but at its ends."""
HAND_RADIUS_M: Final = 0.04
LOCK_TOLERANCE_DEG: Final = math.degrees(0.1)
"""The largest start-goal angle cuRobo accepts for an orientation lock (its metric's 0.05)."""


class MockPlanner:
    def __init__(
        self,
        profile: EmbodimentProfile | None = None,
        *,
        dt: float = 1 / 250,
        speed: float = 1.0,
    ) -> None:
        self.profile = profile or load_embodiment(MOCK_BIMANUAL)
        self.dt = dt
        self.speed = speed

    async def info(self) -> PlannerInfo:
        arms = {
            name: ArmInfo(
                joints=spec.joints,
                lower=list(LOWER),
                upper=list(UPPER),
                links=[],
                ee_to_tcp=spec.ee_to_tcp,
            )
            for name, spec in self.profile.arms.items()
        }
        return PlannerInfo(provider="mock", arms=arms, dt=self.dt)

    async def plan(self, request: PlanRequest) -> PlanResult:
        spec = self._arm(request.arm, [request.start])
        result = self._plan(request.base, request.start, request.goal, request.constraint, True)
        if request.obstacles and result.path is not None:
            result = _blocked(result, spec, request.obstacles) or result
        return result if request.ee_path else result.model_copy(update={"ee_path": None})

    async def plan_batch(self, request: BatchPlanRequest) -> list[PlanResult]:
        self._arm(request.arm, [request.start])
        return [
            self._plan(request.base, request.start, goal, request.constraint, request.ee_path)
            for goal in request.goals
        ]

    async def ik(self, request: IKRequest) -> list[IKSolution]:
        self._arm(request.arm, [request.seed])
        solutions = [_ik(request.base, goal, request.seed) for goal in request.goals]
        return [
            IKSolution(joints=None)
            if q is None
            else IKSolution(joints=q.tolist(), position_error_m=0.0, rotation_error_deg=0.0)
            for q in solutions
        ]

    async def fk(self, request: FKRequest) -> FKResult:
        spec = self._arm(request.arm, request.joints)
        check_links(request.arm, [], request.links)
        ee = _fk(request.base, np.asarray(request.joints, dtype=float))
        tcp = pose_compose(ee, [*spec.ee_to_tcp, 1.0, 0.0, 0.0, 0.0])
        spheres = [
            [(*map(float, e[:3]), HAND_RADIUS_M), (*map(float, t[:3]), HAND_RADIUS_M)]
            for e, t in zip(ee, tcp, strict=True)
        ]
        return FKResult(
            ee=[Pose.from_list(p) for p in ee],
            tcp=[Pose.from_list(p) for p in tcp],
            spheres=spheres if request.spheres else [],
        )

    def _arm(self, name: Arm, vectors: Sequence[Sequence[float]]) -> ArmProfile:
        spec = arm_of(self.profile.arms, name)
        check_joints(name, len(spec.joints), vectors)
        return spec

    def _plan(
        self,
        base: Pose,
        start: Sequence[float],
        goal: Pose,
        constraint: Constraint,
        ee_path: bool,
    ) -> PlanResult:
        started = time.perf_counter()

        def failed(reason: PlanReason, detail: str) -> PlanResult:
            return PlanResult(
                reason=reason, detail=detail, dt=self.dt, time_s=time.perf_counter() - started
            )

        # The order of cuRobo's checks: start state, orientation lock, IK.
        q0 = np.asarray(start, dtype=float)
        if not _within(q0):
            return failed("invalid_start", "the start is outside the joint limits")
        if constraint == "orientation":
            angle = float(quat_angle_deg(_fk(base, q0[None])[0, 3:], goal.q))
            if angle > LOCK_TOLERANCE_DEG:
                return failed("orientation_mismatch", f"start and goal differ by {angle:.1f} deg")
        q1 = _ik(base, goal, q0)
        if q1 is None:
            return failed("unreachable", "the goal is outside the joint limits")
        steps = max(1, math.ceil(float(np.abs(q1 - q0).max()) / self.speed / self.dt))
        path = q0 + (q1 - q0) * np.linspace(0.0, 1.0, steps + 1)[:, None]
        poses = [Pose.from_list(p) for p in _fk(base, path)] if ee_path else None
        return PlanResult(
            reason="ok",
            path=path.tolist(),
            dt=self.dt,
            ee_path=poses,
            time_s=time.perf_counter() - started,
        )


def _blocked(
    result: PlanResult, spec: ArmProfile, obstacles: Sequence[OrientedBox]
) -> PlanResult | None:
    """The failure of a planned path whose TCP starts in, ends in or crosses an obstacle."""
    assert result.ee_path is not None
    ee = np.array([pose.as_list() for pose in result.ee_path])
    tcp = pose_compose(ee, [*spec.ee_to_tcp, 1.0, 0.0, 0.0, 0.0])[:, :3]
    inside = np.zeros(len(tcp), dtype=bool)
    for box in obstacles:
        local = (tcp - np.asarray(box.center)) @ quat_to_matrix(box.q)
        inside |= np.all(np.abs(local) <= np.asarray(box.size) / 2, axis=1)
    if not inside.any():
        return None
    reason: PlanReason = (
        "invalid_start" if inside[0] else "unreachable" if inside[-1] else "no_path"
    )
    return PlanResult(reason=reason, detail="an obstacle is in the way", dt=result.dt)


def _fk(base: Pose, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """(N, 6) joint vectors -> (N, 7) world ee poses, composed as MockEnv composes them."""
    about = [quat_from_axis_angle(axis, q[:, 3 + i]) for i, axis in enumerate(np.eye(3))]
    turn = quat_multiply(about[2], quat_multiply(about[1], about[0]))
    return pose_compose(base.as_list(), np.concatenate([q[:, :3], turn], axis=1))


def _ik(base: Pose, goal: Pose, seed: Sequence[float]) -> NDArray[np.float64] | None:
    local = pose_compose(pose_inverse(base.as_list()), goal.as_list())
    m = quat_to_matrix(local[3:])
    if abs(m[2, 0]) < 1 - 1e-9:
        roll, pitch = math.atan2(m[2, 1], m[2, 2]), math.asin(-m[2, 0])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:  # R depends on yaw - roll at pitch 90 deg, on yaw + roll at -90 deg
        roll, pitch = seed[3], math.copysign(math.pi / 2, -m[2, 0])
        if pitch > 0:
            yaw = roll - math.atan2(m[0, 1], m[1, 1])
        else:
            yaw = math.atan2(-m[0, 1], m[1, 1]) - roll
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    q = np.array([*local[:3], roll, pitch, yaw])
    return q if _within(q) else None


def _within(q: NDArray[np.float64]) -> bool:
    # A goal FK made at a limit may come back a rounding error beyond it.
    return bool(np.all(q >= np.subtract(LOWER, 1e-9)) and np.all(q <= np.add(UPPER, 1e-9)))
