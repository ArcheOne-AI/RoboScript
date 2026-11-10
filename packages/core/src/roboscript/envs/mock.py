"""MockEnv: a deterministic, CPU-only env with all three planes
(docs/concepts/embodied-core.md).

Colored boxes on a table whose top is at z = 0.70, the `mock_bimanual` robot whose
joints are its ee poses, and three pinhole cameras rendered by numpy ray casting into the blob
store. It backs the tests of every layer above the env contract, in-process or as a server
(`python -m roboscript.envs.mock --config <file> --ready-file <path>`).

Each step moves every ee straight to its commanded pose. A gripper closing with its TCP within
3 cm of a box center takes that box, which then follows the TCP; opening drops it onto whatever
is below (the table or another box). The measured finger gap reflects a held box. The (pretend)
expert pre-run fails on `unsolvable_seeds`; a scripted expert's API calls (grasp the
box from above, lift it, put it on the base) serve `expert_trace`.
"""

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import anyio
import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from roboscript.blobs import BlobStore, LocalBlobStore, put_frame
from roboscript.contracts import versions
from roboscript.contracts.action import ActionBatch, ExecuteResult, StepState
from roboscript.contracts.camera import CameraModel
from roboscript.contracts.capabilities import (
    EnvCapabilities,
    SensingCapabilities,
    unoffered_reset,
)
from roboscript.contracts.env import (
    Diagnosis,
    EnvDiscovery,
    EpisodeOpened,
    EpisodeSpec,
    ExpertAction,
    ExpertCall,
    ExpertTrace,
    GtSnapshot,
    JudgeResult,
    ObjectTruth,
    ObserveRequest,
)
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import EpisodeId, FrameId, TaskName, new_id
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import Arm, ArmState, RobotState
from roboscript.embodiments import MOCK_BIMANUAL, EmbodimentProfile, load_embodiment
from roboscript.envs.benchmark import Benchmark, TaskInfo
from roboscript.envs.episodes import Batches, Episodes, reset_options, unsolvable
from roboscript.envs.server import ServerConfig, control_token, serve_env
from roboscript.settings import config_hash, load_config_file
from roboscript.spatial.pose import pose_compose, pose_inverse, pose_to_matrix
from roboscript.spatial.rotation import quat_from_axis_angle, quat_multiply, quat_to_matrix

TABLE_Z: Final = 0.70
_TABLE_HALF: Final = (0.5, 0.4)  # half extents of the table top in x and y
_BOX_HALF: Final = 0.025
COLORS: Final = {
    "red": (200, 40, 40),
    "green": (40, 160, 70),
    "blue": (50, 80, 210),
    "yellow": (220, 190, 40),
}
_TABLE_RGB: Final = np.array([205, 190, 165])
_SKY_RGB: Final = np.array([235, 235, 240], dtype=np.uint8)
_LIGHT: Final = np.array([0.3, -0.5, 1.0]) / np.linalg.norm([0.3, -0.5, 1.0])
_GRASP_RADIUS: Final = 0.03
_FOVY_DEG: Final = 45.0
_NEAR: Final = 0.1  # meters: near clipping plane
# Arms start 25 cm above the table's near half, grippers pointing down (pitch 90 degrees), open.
_HOME: Final = (
    -0.25,
    -0.2,
    0.95,
    0.0,
    math.pi / 2,
    0.0,
    1.0,
    0.25,
    -0.2,
    0.95,
    0.0,
    math.pi / 2,
    0.0,
    1.0,
)
# The overhead camera, behind the robot looking down at the table: x right, y down,
# z forward, as columns, then the position.
_HEAD: Final = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -0.8, 0.6, -0.45], [0.0, -0.6, -0.8, 1.35], [0.0, 0.0, 0.0, 1.0]]
)
# The home orientation (pitch 90 degrees): the gripper points down, the TCP 0.1 m below the ee.
_DOWN: Final = tuple(quat_from_axis_angle([0.0, 1.0, 0.0], math.pi / 2).tolist())
# Wrist cameras sit at the ee looking along its x axis (the gripper's pointing direction).
_WRIST: Final = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


class Lifted(BaseModel):
    """`box` is at least `height` above where it started."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["lifted"] = "lifted"
    box: str
    height: PositiveFloat = 0.05


class On(BaseModel):
    """`box` rests on `base`, released."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["on"] = "on"
    box: str
    base: str


class MockTask(BaseModel):
    """Instructions (`primary`, `paraphrase`: `MockOptions.instructions`), step limit, boxes (by
    color) and success predicate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary: str
    paraphrase: str
    step_limit: PositiveInt = 40
    boxes: list[Literal["red", "green", "blue", "yellow"]] = ["red", "green", "blue"]
    success: Lifted | On = Field(discriminator="kind")


class MockOptions(BaseModel):
    """The mock's reset options (`EpisodeSpec.options`): which instruction set an episode's text
    comes from, and whether it has the task's step limit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instructions: Literal["primary", "paraphrase"] = "primary"
    step_limit: bool = True


TASKS: Final = {
    TaskName("lift_red"): MockTask(
        primary="lift the red box", paraphrase="raise the red block", success=Lifted(box="red")
    ),
    TaskName("stack_red_on_blue"): MockTask(
        primary="put the red box on the blue box",
        paraphrase="stack the red block onto the blue one",
        success=On(box="red", base="blue"),
    ),
}


class MockConfig(BaseModel):
    """One env process serves one `task` of `tasks`. `step_delay_s` makes every step take that
    long (to test concurrent calls); the expert pre-run fails on `unsolvable_seeds`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    blob_dir: Path
    server: ServerConfig = ServerConfig()
    tasks: dict[TaskName, MockTask] = TASKS
    step_delay_s: NonNegativeFloat = 0.0
    unsolvable_seeds: list[NonNegativeInt] = []

    @model_validator(mode="after")
    def _task_is_defined(self) -> Self:
        if self.task not in self.tasks:
            raise ValueError(f"task {self.task!r} is not one of {sorted(self.tasks)}")
        return self


MOCK: Final = Benchmark(
    name="mock",
    tasks={
        name: TaskInfo(step_limit=t.step_limit, category=t.success.kind)
        for name, t in TASKS.items()
    },
    env_server="roboscript.envs.mock",
    interpreter="agent",
    embodiment=MOCK_BIMANUAL,
    # Not a demonstration source: the "missing capability" path is tested against it.
    capabilities=EnvCapabilities(success={"latch"}, reset={"seed"}, truth=True, expert=True),
    env_config_model="roboscript.envs.mock:MockConfig",
    skillsets=["mock"],  # roboscript.testing.skills: its bindings for the core's entries
)


@dataclass
class _Box:
    rgb: tuple[int, int, int]
    half: NDArray[np.float64]  # (3,) half extents
    pose: NDArray[np.float64]  # (7,) world [x, y, z, qw, qx, qy, qz]
    start: NDArray[np.float64]


@dataclass
class _Scene:
    task: MockTask
    boxes: dict[str, _Box]
    step_limit: int | None
    instruction: str
    trace: list[ExpertCall] | None = None
    qpos: NDArray[np.float64] = field(default_factory=lambda: np.array(_HOME))
    held: dict[Arm, tuple[str, NDArray[np.float64]]] = field(default_factory=dict)
    step: int = 0
    success_step: int | None = None


class MockEnv:
    """The env planes (`EnvPublic`, `EnvControl`, `EnvAgent`) over one scene at a time."""

    def __init__(self, config: MockConfig, blobs: BlobStore | None = None) -> None:
        self.config = config
        self.profile: EmbodimentProfile = load_embodiment(MOCK_BIMANUAL)
        self.blobs = blobs if blobs is not None else LocalBlobStore(config.blob_dir)
        self.episodes = Episodes()
        self.agent_url = ""  # set by `serve_env`
        self._batches = Batches()
        self._scene: _Scene | None = None
        self._lock = anyio.Lock()  # one scene user at a time

    async def discover(self) -> EnvDiscovery:
        return EnvDiscovery(
            contracts={
                plane: versions.VERSIONS[plane]
                for plane in ("env.public", "env.control", "env.agent")
            },
            benchmark=MOCK.name,
            embodiment=self.profile.name,
            upstream_commit=None,
            config_hash=config_hash(self.config),
            capabilities=MOCK.discovered(
                SensingCapabilities(
                    scene_identity=True,
                    measured_proprio=True,
                    depth=True,
                    cameras=list(self.profile.cameras),
                    step_unit="action",
                )
            ),
        )

    async def open(self, spec: EpisodeSpec) -> EpisodeOpened:
        if spec.task != self.config.task:
            detail = f"this env serves task {self.config.task!r}, not {spec.task!r}"
            raise ProtocolError("invalid_request", detail)
        if (refused := unoffered_reset(spec.reset_by, MOCK.capabilities)) is not None:
            raise ProtocolError("invalid_request", refused)
        if spec.record:
            raise ProtocolError("invalid_request", "recording episodes is not supported yet")
        options = reset_options(MockOptions, spec)
        task = self.config.tasks[spec.task]
        async with self._lock:
            self.episodes.end()
            if not spec.repeat and spec.seed in self.config.unsolvable_seeds:
                return unsolvable(self.profile.name, spec.prerun_attempts)
            instruction = spec.instruction or getattr(task, options.instructions)
            step_limit = task.step_limit if options.step_limit else None
            boxes = _boxes(task, spec.seed)
            trace = _expert(task, boxes, self.profile) if spec.trace_expert else None
            self._scene = _Scene(task, boxes, step_limit, instruction, trace)
            self._batches = Batches()
            episode_id, token = self.episodes.open()
        return EpisodeOpened(
            episode_id=episode_id,
            agent_url=self.agent_url,
            agent_token=token,
            instruction=instruction,
            step_limit=step_limit,
            embodiment=self.profile.name,
            cameras=list(self.profile.cameras),
            solvable=True,
            prerun_attempts=0 if spec.repeat else 1,
        )

    async def judge(self, episode_id: EpisodeId) -> JudgeResult:
        scene = self._current(episode_id)
        return JudgeResult(
            episode_id=episode_id,
            success=scene.success_step is not None,
            success_step=scene.success_step,
            steps_used=scene.step,
            step_limit=scene.step_limit,
        )

    async def diagnose(self, episode_id: EpisodeId) -> Diagnosis:
        """The boxes at the start and now; a box's pose is at its center."""
        scene = self._current(episode_id)
        objects = [
            ObjectTruth(
                name=name,
                initial=Pose.from_list(box.start),
                final=Pose.from_list(box.pose),
                displacement_m=float(np.linalg.norm(box.pose[:3] - box.start[:3])),
                initial_center=Pose.from_list(box.start).p,
                final_center=Pose.from_list(box.pose).p,
            )
            for name, box in scene.boxes.items()
        ]
        return Diagnosis(
            episode_id=episode_id,
            success=scene.success_step is not None,
            objects=objects,
            notes=[],
        )

    async def gt_snapshot(self, episode_id: EpisodeId, geometry: bool = False) -> GtSnapshot:
        scene = self._current(episode_id)
        objects = {name: Pose.from_list(box.pose) for name, box in scene.boxes.items()}
        return GtSnapshot(
            episode_id=episode_id, step=scene.step, objects=objects, table_top_z=TABLE_Z
        )

    async def expert_trace(self, episode_id: EpisodeId) -> ExpertTrace:
        scene = self._current(episode_id)
        if scene.trace is None:
            detail = f"episode {episode_id} was opened without trace_expert"
            raise ProtocolError("invalid_request", detail)
        return ExpertTrace(episode_id=episode_id, calls=scene.trace)

    async def close(self, episode_id: EpisodeId) -> None:
        async with self._lock:
            self.episodes.close(episode_id)
            self._scene = None

    async def robot(self, episode_id: EpisodeId) -> RobotState:
        return _robot(self._current(episode_id), self.profile)

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation:
        names = request.cameras if request.cameras is not None else list(self.profile.cameras)
        if unknown := sorted(set(names) - set(self.profile.cameras)):
            raise ProtocolError(
                "invalid_request", f"no cameras {unknown}; cameras: {sorted(self.profile.cameras)}"
            )
        async with self._lock:
            scene = self._current(episode_id)
            robot = _robot(scene, self.profile)
            cameras = [_camera(name, self.profile, scene) for name in names]
            boxes = [(box.pose.copy(), box.half, box.rgb) for box in scene.boxes.values()]
        frames = {}
        for camera in cameras:
            rgb, depth = await anyio.to_thread.run_sync(render, camera, boxes)
            frames[camera.name] = await put_frame(
                self.blobs, camera, rgb, depth if request.depth else None
            )
        return Observation(
            frame_id=FrameId(new_id("frame")),
            step=robot.step,
            cameras=frames,
            robot=robot,
            instruction=scene.instruction,
        )

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        self._current(episode_id)
        for step in batch.steps:
            if len(step.qpos) != self.profile.action_dim:
                raise ProtocolError(
                    "invalid_request",
                    f"a qpos has {self.profile.action_dim} values, got {len(step.qpos)}",
                )
        return await self._batches.run(batch, lambda: self._execute(episode_id, batch))

    async def _execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        states: list[StepState] = []
        async with self._lock:
            scene = self._current(episode_id)
            for index, step in enumerate(batch.steps):
                closing = [
                    arm
                    for arm, spec in self.profile.arms.items()
                    if step.qpos[spec.gripper_index] < spec.gripper.open
                ]
                for _ in range(1 + (step.max_repeats if step.until_gripper_stable else 0)):
                    if scene.success_step is not None and batch.stop_on_success:
                        return _result(batch, states, "success")
                    if scene.step_limit is not None and scene.step >= scene.step_limit:
                        return _result(batch, states, "step_limit")
                    before = _openings(scene, self.profile, closing)
                    await anyio.sleep(self.config.step_delay_s)
                    _step(scene, np.asarray(step.qpos, dtype=float), self.profile)
                    states.append(StepState(index=index, robot=_robot(scene, self.profile)))
                    moved = np.abs(_openings(scene, self.profile, closing) - before)
                    if np.max(moved, initial=0.0) < 1e-4:
                        break  # the closing grippers settled
        succeeded = scene.success_step is not None and batch.stop_on_success
        final: Literal["completed", "success"] = "success" if succeeded else "completed"
        return _result(batch, states, final)

    def _current(self, episode_id: EpisodeId) -> _Scene:
        self.episodes.check(episode_id)
        assert self._scene is not None  # an open episode has its scene
        return self._scene


def _result(
    batch: ActionBatch,
    states: list[StepState],
    stopped: Literal["completed", "success", "step_limit"],
) -> ExecuteResult:
    return ExecuteResult(
        batch_id=batch.batch_id, states=states, executed=len(states), stopped=stopped
    )


def _boxes(task: MockTask, seed: int) -> dict[str, _Box]:
    """Boxes at random table positions at least 10 cm apart, turned up to 45 degrees."""
    rng = np.random.default_rng(seed)
    boxes: dict[str, _Box] = {}
    for name in task.boxes:
        xy = rng.uniform((-0.25, -0.1), (0.25, 0.15))
        while any(np.hypot(*(xy - box.pose[:2])) < 0.1 for box in boxes.values()):
            xy = rng.uniform((-0.25, -0.1), (0.25, 0.15))
        quat = quat_from_axis_angle([0.0, 0.0, 1.0], rng.uniform(-math.pi / 4, math.pi / 4))
        pose = np.array([*xy, TABLE_Z + _BOX_HALF, *quat])
        boxes[name] = _Box(COLORS[name], np.full(3, _BOX_HALF), pose, pose.copy())
    return boxes


def _expert(task: MockTask, boxes: dict[str, _Box], profile: EmbodimentProfile) -> list[ExpertCall]:
    """The scripted expert's calls: grasp the task's box from above with the arm on its side,
    lift it 10 cm, and for `on` put it onto the base and let go."""
    goal = task.success
    held = boxes[goal.box]
    arm: Arm = "left" if held.pose[0] < 0 else "right"
    gripper = profile.arms[arm].gripper
    holding = gripper.opening(2 * float(np.min(held.half[:2])))  # the fingers on the box

    def at(center: NDArray[np.float64], up: float = 0.0) -> ExpertAction:
        tcp = Pose(p=(center[0], center[1], center[2] + up), q=_DOWN)
        return ExpertAction(target=profile.ee_pose(arm, tcp))

    def call(api: str, opening: float, *actions: ExpertAction) -> ExpertCall:
        return ExpertCall(
            kind="move",
            actions={arm: list(actions)},
            api={arm: api},
            sim_steps=1,
            openings={arm: opening},
        )

    approach = at(held.pose[:3]).model_copy(update={"constraint": "orientation"})
    closed = ExpertAction(gripper=gripper.closed)
    calls = [
        call("grasp_object", holding, at(held.pose[:3], 0.1), approach, closed),
        call("move_by", holding, at(held.pose[:3], 0.1)),
    ]
    if isinstance(goal, On):
        base = boxes[goal.base]
        top = np.add(base.pose[:3], (0.0, 0.0, base.half[2] + held.half[2] + 0.005))
        opened = ExpertAction(gripper=gripper.open)
        calls += [call("place_object", gripper.open, at(top, 0.1), at(top), opened)]
        calls += [call("move_by", gripper.open, at(top, 0.1))]
    return calls


def _ee(qpos: NDArray[np.float64], profile: EmbodimentProfile, arm: Arm) -> NDArray[np.float64]:
    start, stop = profile.arms[arm].action
    x, y, z, roll, pitch, yaw = qpos[start:stop]
    about = [
        quat_from_axis_angle(axis, angle)
        for axis, angle in zip(np.eye(3), (roll, pitch, yaw), strict=True)
    ]
    return np.array([x, y, z, *quat_multiply(about[2], quat_multiply(about[1], about[0]))])


def _tcp(qpos: NDArray[np.float64], profile: EmbodimentProfile, arm: Arm) -> NDArray[np.float64]:
    return pose_compose(_ee(qpos, profile, arm), [*profile.arms[arm].ee_to_tcp, 1.0, 0.0, 0.0, 0.0])


def _gap(scene: _Scene, profile: EmbodimentProfile, arm: Arm) -> float:
    """Meters between the fingers: the command's, or the held box's width if that is wider."""
    spec = profile.arms[arm].gripper
    gap = spec.scale[0] + scene.qpos[profile.arms[arm].gripper_index] * (
        spec.scale[1] - spec.scale[0]
    )
    if arm in scene.held:
        gap = max(gap, 2 * float(np.min(scene.boxes[scene.held[arm][0]].half[:2])))
    return float(gap)


def _openings(scene: _Scene, profile: EmbodimentProfile, arms: list[Arm]) -> NDArray[np.float64]:
    return np.array([profile.arms[arm].gripper.opening(_gap(scene, profile, arm)) for arm in arms])


def _step(scene: _Scene, qpos: NDArray[np.float64], profile: EmbodimentProfile) -> None:
    previous, scene.qpos = scene.qpos, qpos
    scene.step += 1
    for arm, spec in profile.arms.items():
        tcp = _tcp(qpos, profile, arm)
        command, before = qpos[spec.gripper_index], previous[spec.gripper_index]
        if arm in scene.held:
            name, in_tcp = scene.held[arm]
            scene.boxes[name].pose = pose_compose(tcp, in_tcp)
            if command >= 0.5:
                del scene.held[arm]
                _drop(scene, name)
        elif command < 0.5 <= before:
            taken = {name for name, _ in scene.held.values()}
            free = {name: box for name, box in scene.boxes.items() if name not in taken}
            near = min(free, key=lambda n: np.linalg.norm(free[n].pose[:3] - tcp[:3]), default=None)
            if near is not None and np.linalg.norm(free[near].pose[:3] - tcp[:3]) <= _GRASP_RADIUS:
                scene.held[arm] = (near, pose_compose(pose_inverse(tcp), free[near].pose))
    if scene.success_step is None and _success(scene):
        scene.success_step = scene.step


def _drop(scene: _Scene, name: str) -> None:
    """Put a released box upright (same yaw) on the highest support below its center."""
    box = scene.boxes[name]
    held = {n for n, _ in scene.held.values()}
    support = TABLE_Z
    for other_name, other in scene.boxes.items():
        below = other_name != name and other_name not in held
        if below and np.hypot(*(other.pose[:2] - box.pose[:2])) <= other.half[0]:
            support = max(support, other.pose[2] + other.half[2])
    rotation = quat_to_matrix(box.pose[3:])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    box.pose = np.array(
        [*box.pose[:2], support + box.half[2], *quat_from_axis_angle([0.0, 0.0, 1.0], yaw)]
    )


def _success(scene: _Scene) -> bool:
    goal = scene.task.success
    box = scene.boxes[goal.box]
    if isinstance(goal, Lifted):
        return bool(box.pose[2] - box.start[2] >= goal.height)
    base = scene.boxes[goal.base]
    held = goal.box in {name for name, _ in scene.held.values()}
    centered = np.hypot(*(box.pose[:2] - base.pose[:2])) <= base.half[0]
    resting = abs(box.pose[2] - (base.pose[2] + base.half[2] + box.half[2])) < 0.005
    return bool(not held and centered and resting)


def _robot(scene: _Scene, profile: EmbodimentProfile) -> RobotState:
    arms = {}
    for arm, spec in profile.arms.items():
        start, stop = spec.action
        joints = [float(v) for v in scene.qpos[start:stop]]
        arms[arm] = ArmState(
            ee=Pose.from_list(_ee(scene.qpos, profile, arm)),
            tcp=Pose.from_list(_tcp(scene.qpos, profile, arm)),
            joints=joints,
            joints_cmd=joints,
            gripper=spec.gripper.opening(_gap(scene, profile, arm)),
            gripper_cmd=float(scene.qpos[spec.gripper_index]),
        )
    return RobotState(
        step=scene.step,
        step_limit=scene.step_limit,
        arms=arms,
        qpos_cmd=[float(v) for v in scene.qpos],
    )


def _camera(name: str, profile: EmbodimentProfile, scene: _Scene) -> CameraModel:
    spec = profile.cameras[name]
    fy = spec.height / 2 / math.tan(math.radians(_FOVY_DEG) / 2)
    K = ((fy, 0.0, spec.width / 2), (0.0, fy, spec.height / 2), (0.0, 0.0, 1.0))
    if spec.arm is None:
        cam2world = _HEAD
    else:
        cam2world = pose_to_matrix(_ee(scene.qpos, profile, spec.arm)) @ _WRIST
    return CameraModel(
        name=name, width=spec.width, height=spec.height, K=K, cam2world=cam2world.tolist()
    )


def render(
    camera: CameraModel,
    boxes: list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int]]],
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """RGB (H, W, 3) uint8 and depth (H, W) float32 in meters (0: no hit) of the table top and
    `boxes` (pose, half extents, color), by casting one ray per pixel center. Nothing nearer
    than 0.1 m is seen (a wrist camera's blind zone)."""
    K, cam2world = np.array(camera.K), np.array(camera.cam2world)
    v, u = np.mgrid[0 : camera.height, 0 : camera.width]
    rays = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3) @ np.linalg.inv(K).T
    origin, directions = cam2world[:3, 3], rays @ cam2world[:3, :3].T  # camera z of each ray is 1
    depth = np.full(len(rays), np.inf)
    rgb = np.tile(_SKY_RGB.astype(float), (len(rays), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (TABLE_Z - origin[2]) / directions[:, 2]
    hit = origin[:2] + t[:, None] * directions[:, :2]
    on_table = (
        (t >= _NEAR) & (np.abs(hit[:, 0]) <= _TABLE_HALF[0]) & (np.abs(hit[:, 1]) <= _TABLE_HALF[1])
    )
    depth[on_table] = t[on_table]
    rgb[on_table] = _TABLE_RGB * (0.55 + 0.45 * _LIGHT[2])
    for pose, half, color in boxes:
        rotation = quat_to_matrix(pose[3:])
        o, d = (origin - pose[:3]) @ rotation, directions @ rotation  # in the box frame
        with np.errstate(divide="ignore", invalid="ignore"):
            near, far = (-half - o) / d, (half - o) / d
        entry = np.minimum(near, far)
        t_in, t_out = np.nanmax(entry, axis=1), np.nanmin(np.maximum(near, far), axis=1)
        closer = (t_in <= t_out) & (t_in >= _NEAR) & (t_in < depth)
        normals = rotation[:, np.argmax(entry, axis=1)[closer]].T  # of the faces the rays enter
        rgb[closer] = np.array(color) * (0.55 + 0.45 * np.abs(normals @ _LIGHT))[:, None]
        depth[closer] = t_in[closer]
    depth[np.isinf(depth)] = 0.0
    shape = (camera.height, camera.width)
    return rgb.reshape(*shape, 3).astype(np.uint8), depth.reshape(shape).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve MockEnv on a control and an agent port.")
    parser.add_argument("--config", type=Path, required=True, help="a MockConfig YAML file")
    parser.add_argument("--ready-file", type=Path, help="where to write the planes' URLs")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config_file(args.config, MockConfig)
    anyio.run(serve_env, MockEnv(config), config.server, args.ready_file, control_token())


if __name__ == "__main__":
    main()
