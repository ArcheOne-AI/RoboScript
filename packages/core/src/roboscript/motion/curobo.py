"""A cuRobo `MotionGen` wrapper as a `Planner`, driven by the planner profile.

Per arm it builds, from the embodiment's curobo yml (asset paths rebased into the embodiment's
asset directory), a single-goal and a batch `MotionGen` with their warm-ups, over a collision world
holding only the planner profile's table cuboid (`CuroboConfig.planner_profile`), and a single
plan's `obstacles` besides, as cuboids for that request (the single planner's collision cache holds
the table and `MAX_OBSTACLES`; its world is reloaded only when a request's differs from the
last, and IK, which shares its collision checker, always runs on the table alone). Plans run with
`max_attempts=10`, orientation locks as `PoseCostMetric(hold_partial_pose=True,
hold_vec_weight=[1, 1, 1, 0, 0, 0])`, and world targets map into the planner frame: into the root
frame, plus `frame_bias`, turned by the arm's yaw. The ee is the yml's
ee link: this planner adds no fixed transform between the gripper and the end link, so an
embodiment that has one must state its yml's ee link accordingly.
Batches are padded to `batch_size` goals, the one count the batch planner is warmed up for
(cuRobo's batch planner fails every later batch after a call with another count). FK runs on a
separate kinematics model of the arm's links, fingers at the opening the yml locks them at; IK
calls the single planner's solver exactly as its planning does. Where the profile gives an arm's
joints physical stops narrower than the URDF's limits, the planners are built with the stops as
their joint limits (`within_stops`).

One daemon thread owns everything on the GPU, and only this module imports torch and curobo, on
that thread. Every plan and IK call advances cuRobo's seed state, and its GPU
arithmetic is not bit-reproducible: an identical request can come back a few steps longer or
shorter, with the same end. An error on that thread (a CUDA fault, say) may leave cuRobo's buffers
inconsistent, so the planner then stops being ready, refuses calls (those already queued too)
and names the error as its `failure`, on which its service exits to be restarted; the call that
failed fails as an infrastructure error (`not_ready`), chained to the error. Two exceptions: the
batch planner's own give-up after a failed graph search (a shape mismatch it raises before
touching any buffer):
its goals are planned one at a time by the
single planner instead; and scipy's out-of-range `ValueError` the graph planner raises as it
interpolates a path it found (`graph_interpolation_failed`; a point past the range by a float
rounding of the path's length):
that request fails as an infrastructure error (`not_ready`) and the planner stays ready. That step
is CPU post-processing of a search that has ended and returned (cuRobo catches its search's own
errors and resets the graph's buffer then; not this one, which comes after, `GRAPH_INTERPOLATION`):
it copies the found path to numpy and writes only the graph planner's output buffer, whose positions
every interpolation rewrites before they are read; the graph it searched persists across plans
anyway (cleared when the world changes or it fills).
What the plan would have done after it (trajectory optimization, its trailing synchronize and
lock release) leaves nothing a later plan reads before setting it, but the orientation lock,
which `_plan_single` releases. The plan is not retried: the same path would fail the same way.
"""

import logging
import math
import time
import traceback
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, TypeVar

import numpy as np
import yaml
from numpy.typing import NDArray

from roboscript.contracts.errors import InfraError
from roboscript.contracts.geometry import OrientedBox, Pose
from roboscript.contracts.motion import (
    MAX_OBSTACLES,
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
from roboscript.contracts.skills import PlannerTable
from roboscript.motion.config import ArmSetup, CuroboConfig
from roboscript.motion.validate import arm_of, check_joints, check_links
from roboscript.rpc import Worker
from roboscript.spatial.pose import pose_compose, pose_inverse
from roboscript.spatial.rotation import quat_angle_deg, quat_to_matrix

T = TypeVar("T")
log = logging.getLogger(__name__)

DT: Final = 1 / 250
"""`interpolation_dt`: one path point per physics step."""
MAX_ATTEMPTS: Final = 10
HOLD_ORIENTATION: Final = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
"""`hold_vec_weight` in cuRobo's order rx, ry, rz, px, py, pz: 1 holds, 0 frees."""
GRAPH_INTERPOLATION: Final = (
    ("curobo.graph.graph_base", "find_paths"),
    ("curobo.graph.graph_base", "get_interpolated_trajectory"),
    ("curobo.util.trajectory", "get_interpolated_trajectory"),
    ("curobo.util.trajectory", "linear_smooth"),
)
"""The frames, outer to inner, of the graph planner's interpolation of a path it found
(decorators' frames between them left out).

`find_paths` interpolates after the `try` around its search (`_find_paths`), whose `ValueError`
and `RuntimeError` it catches itself (resetting the graph's buffer and returning no path): an
error in the interpolation is not caught in cuRobo, and its search had ended and returned."""
INTERPOLATION_RANGE: Final = "interpolation range"
"""In the message of scipy's `interp1d` for a point above or below the range it interpolates
(`_check_bounds`), the error of `GRAPH_INTERPOLATION`."""
_REASONS: Final[dict[str, PlanReason]] = {
    "IK_FAIL": "unreachable",
    "INVALID_START_STATE_JOINT_LIMITS": "invalid_start",
    "INVALID_START_STATE_WORLD_COLLISION": "invalid_start",
    "INVALID_START_STATE_SELF_COLLISION": "invalid_start",
    "INVALID_START_STATE_UNKNOWN_ISSUE": "invalid_start",
    "INVALID_PARTIAL_POSE_COST_METRIC": "orientation_mismatch",
}


@dataclass
class _Arm:
    joints: list[str]
    links: list[str]
    lower: list[float]
    upper: list[float]
    planner_T_root: NDArray[np.float64]
    single: Any  # MotionGen
    batch: Any  # MotionGen
    kinematics: Any  # CudaRobotModel reporting `links`
    ee_to_tcp: tuple[float, float, float]
    world_base: list[float]  # the base the batch planner's collision world was built for
    world: tuple[Any, ...]  # the single planner's: its base and obstacles


class CuroboPlanner:
    """The `Planner` of an embodiment's arms; `await start()` builds and warms it up. Served by
    the planner service (`roboscript.planners` name `curobo`) from its `Config`."""

    Config: ClassVar[type[CuroboConfig]] = CuroboConfig

    def __init__(self, config: CuroboConfig) -> None:
        self.config = config
        self._directory, self._setups = config.resolve()
        self._arms: dict[Arm, _Arm] = {}
        self._worker = Worker("curobo-planner")
        self._failure: str | None = None

    @property
    def ready(self) -> bool:
        return bool(self._arms) and self._failure is None

    @property
    def failure(self) -> str | None:
        """What failed on the planner thread, leaving its state suspect; None while it is sound."""
        return self._failure

    async def start(self) -> None:
        """Build and warm up every arm's planners, tens of seconds; then `ready` is true."""
        self._arms = await self._worker.run(self._build)

    async def info(self) -> PlannerInfo:
        arms = {
            name: ArmInfo(
                joints=a.joints, lower=a.lower, upper=a.upper, links=a.links, ee_to_tcp=a.ee_to_tcp
            )
            for name, a in self._ready_arms().items()
        }
        return PlannerInfo(provider="curobo", arms=arms, dt=DT)

    async def plan(self, request: PlanRequest) -> PlanResult:
        arm = self._arm(request.arm, [request.start])
        return await self._run(
            self._plan_single,
            arm,
            request.base,
            request.start,
            request.goal,
            request.constraint,
            request.ee_path,
            request.obstacles,
        )

    async def plan_batch(self, request: BatchPlanRequest) -> list[PlanResult]:
        arm = self._arm(request.arm, [request.start])
        return await self._run(self._plan_batch, arm, request)

    async def ik(self, request: IKRequest) -> list[IKSolution]:
        arm = self._arm(request.arm, [request.seed])
        return await self._run(self._ik, arm, request)

    async def fk(self, request: FKRequest) -> FKResult:
        arm = self._arm(request.arm, request.joints)
        check_links(request.arm, arm.links, request.links)
        return await self._run(self._fk, arm, request)

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        """`fn(*args)` on the planner thread. An error there fails the call as an infrastructure
        error, chained to it: one that leaves the planner sound (`graph_interpolation_failed`)
        fails this call alone; any other takes the planner out of service (`_latched`)."""
        try:
            return await self._worker.run(self._latched, fn, *args)
        except InfraError:  # refused on the planner thread: it failed before
            raise
        except Exception as error:
            if graph_interpolation_failed(error):  # the request's alone (the module docstring)
                detail = f"ValueError in cuRobo's graph interpolation in {fn.__name__}: {error}"
                log.warning("a plan failed, the planner stays ready: %s", detail)
                failed = f"the planner failed this request ({detail})"
                raise InfraError("not_ready", failed) from error
            log.error("the planner failed and must be restarted: %s", self._failure, exc_info=error)
            raise self._refusal() from error

    def _refusal(self) -> InfraError:
        return InfraError("not_ready", f"the planner failed ({self._failure}); restart it")

    def _ready_arms(self) -> dict[Arm, _Arm]:
        if self._failure is not None:
            raise self._refusal()
        if not self._arms:
            raise InfraError("not_ready", "the planner is still warming up")
        return self._arms

    def _arm(self, name: Arm, vectors: Sequence[Sequence[float]]) -> _Arm:
        arm = arm_of(self._ready_arms(), name)
        check_joints(name, len(arm.joints), vectors)
        return arm

    # Everything below runs on the worker thread.

    def _latched(self, fn: Callable[..., T], *args: Any) -> T:
        """`fn(*args)`, unless the planner failed: the failure is recorded here, on the planner
        thread, so that the calls queued behind the one that failed are refused, not run on a
        planner whose state is suspect. The first failure is the one it names."""
        if self._failure is not None:
            raise self._refusal()
        try:
            return fn(*args)
        except Exception as error:
            if self._failure is None and not graph_interpolation_failed(error):
                self._failure = f"{type(error).__name__} in {fn.__name__}"
            raise

    def _build(self) -> dict[Arm, _Arm]:
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.util.logger import setup_logger

        setup_logger(level="error", logger_name="curobo")
        tensors = TensorDeviceType(device=torch.device(self.config.device))
        return {name: self._build_arm(setup, tensors) for name, setup in self._setups.items()}

    def _build_arm(self, setup: ArmSetup, tensors: Any) -> _Arm:
        from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        robot = load_robot(setup.curobo_yml, self._directory, setup.joints)
        kinematics = robot["robot_cfg"]["kinematics"]
        locked = kinematics.get("lock_joints") or {}
        joints = [j for j in kinematics["cspace"]["joint_names"] if j not in locked]
        if joints != setup.joints:
            raise ValueError(f"{setup.curobo_yml} plans the joints {joints}, not {setup.joints}")
        world = table_world(setup.base, self.config.planner_profile.table)

        def motion_gen(**extra: Any) -> Any:
            config = MotionGenConfig.load_from_robot_config(
                within_stops(robot, joints, setup.stops, tensors),
                world,
                tensors,
                interpolation_dt=DT,
                num_trajopt_seeds=1,
                **extra,
            )
            return MotionGen(config)

        # The order and warm-ups here fix the seed state.
        single = motion_gen(collision_cache={"obb": 1 + MAX_OBSTACLES})
        if self.config.warmup:
            single.warmup()
        batch = motion_gen(num_graph_seeds=1)
        if self.config.warmup:
            batch.warmup(batch=self.config.batch_size)

        ee = kinematics["ee_link"]
        links = [ee, *(link for link in kinematics["collision_link_names"] if link != ee)]
        generator = CudaRobotGeneratorConfig(
            **{**kinematics, "link_names": links}, tensor_args=tensors
        )
        limits = single.kinematics.get_joint_limits().position.cpu().numpy().astype(float)
        return _Arm(
            joints=joints,
            links=links,
            lower=limits[0].tolist(),
            upper=limits[1].tolist(),
            planner_T_root=planner_T_root(robot["planner"]["frame_bias"], setup.yaw_rad),
            single=single,
            batch=batch,
            kinematics=CudaRobotModel(CudaRobotModelConfig.from_config(generator)),
            ee_to_tcp=setup.ee_to_tcp,
            world_base=setup.base.as_list(),
            world=_world_key(setup.base, []),
        )

    def _plan_single(
        self,
        arm: _Arm,
        base: Pose,
        start: Sequence[float],
        goal: Pose,
        constraint: Constraint,
        ee_path: bool,
        obstacles: Sequence[OrientedBox] = (),
    ) -> PlanResult:
        from curobo.types.math import Pose as CuroboPose
        from curobo.wrap.reacher.motion_gen import PoseCostMetric

        started = time.perf_counter()
        self._ensure_world(arm, base, obstacles)
        target = self._to_planner(arm, base, [goal])[0]
        try:
            result = arm.single.plan_single(
                self._start(arm, start, 1),
                CuroboPose.from_list(target.tolist(), tensor_args=arm.single.tensor_args),
                self._plan_config(arm, constraint),
            )
        except BaseException:
            # cuRobo releases an orientation lock when a plan returns; this one raised.
            if constraint == "orientation":
                arm.single.update_pose_cost_metric(PoseCostMetric.reset_metric())
            raise
        if not result.success.item():
            reason = _REASONS.get(getattr(result.status, "name", ""), "no_path")
            return PlanResult(
                reason=reason, detail=_detail(result.status), dt=DT, time_s=_since(started)
            )
        path = result.interpolated_plan.position.cpu().numpy().astype(float)
        return self._success(arm, base, path, ee_path).model_copy(
            update={"time_s": _since(started)}
        )

    def _plan_batch(self, arm: _Arm, request: BatchPlanRequest) -> list[PlanResult]:
        started = time.perf_counter()
        base, goals = request.base, request.goals
        if request.constraint == "orientation":
            # cuRobo's batch mode refuses a whole batch when one goal's orientation differs from
            # the start's (and then raises), so each goal is planned on its own.
            results = [
                self._plan_single(arm, base, request.start, g, "orientation", request.ee_path)
                for g in goals
            ]
        else:
            size = self.config.batch_size
            results = [
                result
                for i in range(0, len(goals), size)
                for result in self._plan_chunk(arm, request, goals[i : i + size])
            ]
        elapsed = _since(started)
        return [result.model_copy(update={"time_s": elapsed}) for result in results]

    def _plan_chunk(
        self, arm: _Arm, request: BatchPlanRequest, goals: list[Pose]
    ) -> list[PlanResult]:
        import torch
        from curobo.types.math import Pose as CuroboPose

        base = request.base
        self._ensure_world(arm, base)
        padded = [*goals, *[goals[-1]] * (self.config.batch_size - len(goals))]
        targets = torch.as_tensor(
            self._to_planner(arm, base, padded),
            dtype=torch.float32,
            device=arm.batch.tensor_args.device,
        )
        try:
            result = arm.batch.plan_batch(
                self._start(arm, request.start, len(padded)),
                CuroboPose(targets[:, :3], targets[:, 3:]),
                self._plan_config(arm, "free"),
            )
        except RuntimeError as error:
            # Once a batch falls back to graph search (from the fourth attempt) and that finds
            # no path, cuRobo zeroes the result's success mask, which aliases the IK one it
            # then indexes the trajopt goals with, and raises before touching any buffer: trajopt
            # never runs, though it may still reach the goals. The single planner has no such
            # path, so the chunk's goals are planned one at a time instead.
            if not str(error).startswith("shape mismatch"):
                raise
            log.warning("a batch plan gave up (%s); planning its goals one at a time", error)
            return [
                self._plan_single(arm, base, request.start, g, "free", request.ee_path)
                for g in goals
            ]
        success = result.success.view(-1).tolist()
        status = getattr(result.status, "name", "")
        reason: PlanReason = "unreachable" if status == "IK_FAIL" else "no_path"
        if not result.valid_query:  # batch mode checks the start state this way
            reason = "invalid_start"
        results = []
        for i in range(len(goals)):
            if not success[i]:
                results.append(PlanResult(reason=reason, detail=_detail(result.status), dt=DT))
                continue
            steps = result.path_buffer_last_tstep[i]
            path = result.interpolated_plan.position[i, :steps].cpu().numpy().astype(float)
            results.append(self._success(arm, base, path, request.ee_path))
        return results

    def _ik(self, arm: _Arm, request: IKRequest) -> list[IKSolution]:
        import torch
        from curobo.types.math import Pose as CuroboPose
        from curobo.wrap.reacher.types import ReacherSolveType

        self._ensure_world(arm, request.base)
        solver = arm.single.ik_solver
        seed = torch.as_tensor(request.seed, dtype=torch.float32, device=solver.tensor_args.device)
        found: list[NDArray[np.float64] | None] = []
        for target in self._to_planner(arm, request.base, request.goals):
            # The arguments motion_gen._solve_ik_from_solve_state passes when planning.
            result = solver.solve_any(
                ReacherSolveType.SINGLE,
                CuroboPose.from_list(target.tolist(), tensor_args=solver.tensor_args),
                seed.view(1, -1),
                seed.view(1, 1, -1),
                1,
                arm.single.ik_seeds,
                False,
            )
            ok = bool(result.success.view(-1)[0].item())
            found.append(result.solution.view(-1, len(arm.joints))[0].cpu().numpy() if ok else None)
        solved = [q for q in found if q is not None]
        reached = iter(self._fk_world(arm, request.base, np.array(solved))[0] if solved else [])
        solutions = []
        for goal, q in zip(request.goals, found, strict=True):
            if q is None:
                solutions.append(IKSolution(joints=None))
                continue
            pose = next(reached)
            solutions.append(
                IKSolution(
                    joints=q.astype(float).tolist(),
                    position_error_m=float(np.linalg.norm(pose[:3] - goal.p)),
                    rotation_error_deg=float(quat_angle_deg(pose[3:], goal.q)),
                )
            )
        return solutions

    def _fk(self, arm: _Arm, request: FKRequest) -> FKResult:
        joints = np.asarray(request.joints)
        ee, links = self._fk_world(arm, request.base, joints, request.links)
        tcp = pose_compose(ee, [*arm.ee_to_tcp, 1.0, 0.0, 0.0, 0.0])
        return FKResult(
            ee=_poses(ee),
            tcp=_poses(tcp),
            links={name: _poses(poses) for name, poses in links.items()},
            spheres=self._spheres(arm, request.base, joints) if request.spheres else [],
        )

    def _spheres(
        self, arm: _Arm, base: Pose, joints: NDArray[Any]
    ) -> list[list[tuple[float, float, float, float]]]:
        """The arm's collision spheres in the world for (N, dof) joint vectors; cuRobo marks
        unused spheres with a negative radius."""
        import torch

        q = torch.as_tensor(joints, dtype=torch.float32, device=arm.kinematics.tensor_args.device)
        state = arm.kinematics.get_state(q.view(-1, len(arm.joints)))
        spheres = state.link_spheres_tensor.cpu().numpy().astype(float)  # (N, n, 4), planner frame
        world_T_planner = pose_compose(base.as_list(), pose_inverse(arm.planner_T_root))
        rotation, offset = quat_to_matrix(world_T_planner[3:]), world_T_planner[:3]
        return [
            [(*map(float, rotation @ s[:3] + offset), float(s[3])) for s in vector if s[3] > 0]
            for vector in spheres
        ]

    def _success(
        self, arm: _Arm, base: Pose, path: NDArray[np.float64], ee_path: bool
    ) -> PlanResult:
        poses = _poses(self._fk_world(arm, base, path)[0]) if ee_path else None
        return PlanResult(reason="ok", path=path.tolist(), dt=DT, ee_path=poses)

    def _fk_world(
        self, arm: _Arm, base: Pose, joints: NDArray[Any], links: Sequence[str] = ()
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        """World poses (N, 7) of the ee and of `links` for (N, dof) joint vectors."""
        import torch

        q = torch.as_tensor(joints, dtype=torch.float32, device=arm.kinematics.tensor_args.device)
        state = arm.kinematics.get_state(q.view(-1, len(arm.joints)))
        world_T_planner = pose_compose(base.as_list(), pose_inverse(arm.planner_T_root))

        def world(position: Any, quaternion: Any) -> NDArray[np.float64]:
            local = torch.cat([position, quaternion], dim=-1).cpu().numpy().astype(float)
            return pose_compose(world_T_planner, local)

        index = {name: i for i, name in enumerate(state.link_names)}
        poses = {
            name: world(
                state.links_position[:, index[name]], state.links_quaternion[:, index[name]]
            )
            for name in links
        }
        return world(state.ee_position, state.ee_quaternion), poses

    def _to_planner(self, arm: _Arm, base: Pose, poses: Sequence[Pose]) -> NDArray[np.float64]:
        """World ee poses as planner-frame targets (N, 7), mapped as `plan_path` maps them."""
        world = np.array([pose.as_list() for pose in poses])
        local = pose_compose(arm.planner_T_root, pose_compose(pose_inverse(base.as_list()), world))
        local[:, 3:] *= np.where(local[:, 3:4] < 0, -1.0, 1.0)  # w >= 0, as transforms3d gives
        return local

    def _ensure_world(self, arm: _Arm, base: Pose, obstacles: Sequence[OrientedBox] = ()) -> None:
        """The single planner's world: the table and `obstacles`; the batch planner's: the table
        (IK uses the single planner's checker, so it calls this with no obstacles)."""
        from curobo.geom.types import WorldConfig

        table = self.config.planner_profile.table
        if base.as_list() != arm.world_base:
            arm.batch.update_world(WorldConfig.from_dict(table_world(base, table)))
            arm.world_base = base.as_list()
        key = _world_key(base, obstacles)
        if key != arm.world:
            world = table_world(base, table)
            to_planner = pose_compose(arm.planner_T_root, pose_inverse(base.as_list()))
            for i, box in enumerate(obstacles):
                pose = pose_compose(to_planner, [*box.center, *box.q])
                world["cuboid"][f"obstacle_{i}"] = {"dims": list(box.size), "pose": pose.tolist()}
            arm.single.update_world(WorldConfig.from_dict(world))
            arm.world = key

    def _start(self, arm: _Arm, start: Sequence[float], n: int) -> Any:
        import torch
        from curobo.types.robot import JointState

        # rounded to 1e-5 rad to avoid float precision problems
        angles = torch.tensor(
            [round(v, 5) for v in start], dtype=torch.float32, device=arm.single.tensor_args.device
        )
        return JointState.from_position(angles.view(1, -1).repeat(n, 1), joint_names=arm.joints)

    def _plan_config(self, arm: _Arm, constraint: Constraint) -> Any:
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig, PoseCostMetric

        config = MotionGenPlanConfig(max_attempts=MAX_ATTEMPTS)
        if constraint == "orientation":
            config.pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=arm.single.tensor_args.to_device(HOLD_ORIENTATION),
            )
        return config


def graph_interpolation_failed(error: BaseException) -> bool:
    """Whether `error` is scipy's `ValueError` for a point outside the interpolation range
    (`INTERPOLATION_RANGE` in its message), raised in cuRobo's graph planner as it interpolates a
    path it found (`GRAPH_INTERPOLATION`, in that order on its traceback), which leaves the planner
    sound (the module docstring). Any other error there takes the planner out of service."""
    if type(error) is not ValueError or INTERPOLATION_RANGE not in str(error):
        return False
    frames = iter(
        (frame.f_globals.get("__name__"), frame.f_code.co_name)
        for frame, _ in traceback.walk_tb(error.__traceback__)
    )
    return all(step in frames for step in GRAPH_INTERPOLATION)  # each after the one before


def within_stops(
    robot: dict[str, Any],
    joints: list[str],
    stops: Sequence[tuple[float, float]] | None,
    tensors: Any,
) -> Any:
    """The yml's robot as `MotionGenConfig` would load it (`RobotConfig.from_dict`), each joint's
    position limits narrowed to its physical stops where they are narrower: IK, trajectory
    optimization and graph search all take their bounds from these limits, so plans stay inside
    the stops instead of the executor refusing them afterwards."""
    import torch
    from curobo.types.robot import RobotConfig

    config = RobotConfig.from_dict(deepcopy(robot), tensors)
    if stops is None:
        return config
    limits = config.kinematics.kinematics_config.joint_limits
    if list(limits.joint_names) != joints:
        raise ValueError(f"cuRobo limits the joints {limits.joint_names}, not {joints}")
    low, high = torch.as_tensor(stops, device=limits.position.device, dtype=limits.position.dtype).T
    limits.position[0] = torch.maximum(limits.position[0], low)
    limits.position[1] = torch.minimum(limits.position[1], high)
    return config


def load_robot(path: Path, directory: Path, planned: Sequence[str] | None = None) -> dict[str, Any]:
    """A curobo yml with its asset paths rebased into the embodiment `directory`: the shipped
    files point into another checkout (or at a `${ASSETS_PATH}` placeholder), so only what
    follows the embodiment directory's name is kept; collision spheres written in the yml (not a
    file) stay as they are. A yml of cuRobo's format 2 is read as format 1, the one
    this planner runs (`from_format_2`). With `planned`, the arm's joints, a joint of the yml's
    cspace that the arm does not plan and the yml does not lock (fingers on no link the model
    builds, which cuRobo cannot lock) is left out of the cspace, with its entries in
    the cspace's per-joint lists."""
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    kinematics = data["robot_cfg"]["kinematics"]
    if float(kinematics.get("format_version", 1)) >= 2:
        kinematics = data["robot_cfg"]["kinematics"] = from_format_2(kinematics)
    for key in ("urdf_path", "collision_spheres"):
        if isinstance(kinematics[key], str):
            kinematics[key] = str(rebase(kinematics[key], directory))
    if planned is not None:
        cspace = kinematics["cspace"]
        locked = kinematics.get("lock_joints") or {}
        names = cspace["joint_names"]
        keep = [i for i, name in enumerate(names) if name in planned or name in locked]
        if len(keep) < len(names):
            for key, value in list(cspace.items()):
                if isinstance(value, list) and len(value) == len(names):
                    cspace[key] = [value[i] for i in keep]
    return data


FORMAT_2_ONLY: Final = ("format_version", "grasp_contact_link_names", "load_tool_frames_with_mesh")
"""Kinematics keys of cuRobo's format 2 that format 1 does not take (besides `tool_frames`)."""


def from_format_2(kinematics: dict[str, Any]) -> dict[str, Any]:
    """Format-2 kinematics as format 1: its one tool frame is the `ee_link`, the cspace's
    `default_joint_position` its `retract_config`, and what format 1 has no field for is left
    out (`null_space_maximum_distance` of the cspace too)."""
    frames = kinematics.get("tool_frames") or []
    if len(frames) != 1:
        raise ValueError(f"a format-2 yml needs exactly one tool frame, got {frames}")
    converted = {k: v for k, v in kinematics.items() if k not in (*FORMAT_2_ONLY, "tool_frames")}
    converted["ee_link"] = frames[0]
    cspace = dict(kinematics["cspace"])
    cspace["retract_config"] = cspace.pop("default_joint_position")
    cspace.pop("null_space_maximum_distance", None)
    converted["cspace"] = cspace
    return converted


def rebase(asset: str, directory: Path) -> Path:
    marker = f"/{directory.name}/"
    if marker not in asset:
        raise ValueError(
            f"{asset!r} does not lie in an embodiment directory named {directory.name}"
        )
    path = directory / asset.rsplit(marker, 1)[1]
    if not path.is_file():
        raise FileNotFoundError(f"{asset!r} rebased to {path}, which does not exist")
    return path


def table_world(base: Pose, table: PlannerTable) -> dict[str, Any]:
    """The planner's collision world: one cuboid, `table`, placed in the planner frame from the
    root pose."""
    _, y, z = base.p
    cuboid = {"dims": list(table.size), "pose": [y, 0.0, table.top_z - z, 1, 0, 0, 0.0]}
    return {"cuboid": {"table": cuboid}}


def planner_T_root(frame_bias: Sequence[float], yaw: float) -> NDArray[np.float64]:
    """The map from the root frame to the planner frame: add `frame_bias`, turn by `yaw`
    about z (the inverse of the arm's mount in the root frame)."""
    turn = [0.0, 0.0, 0.0, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return pose_compose(turn, [*frame_bias, 1.0, 0.0, 0.0, 0.0])


def _world_key(base: Pose, obstacles: Sequence[OrientedBox]) -> tuple[Any, ...]:
    return (tuple(base.as_list()), *(box.model_dump_json() for box in obstacles))


def _poses(values: NDArray[np.float64]) -> list[Pose]:
    return [Pose.from_list(v) for v in values]


def _detail(status: Any) -> str:
    return str(getattr(status, "value", status or ""))


def _since(started: float) -> float:
    return time.perf_counter() - started
