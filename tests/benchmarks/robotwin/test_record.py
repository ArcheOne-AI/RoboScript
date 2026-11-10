"""Demo re-recording without the simulator: the recorder's replay on a stand-in `Simulator`, up to
the stop at the success latch. The recorder's modules read the expert data with h5py (the
simulator env's)."""

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ExecuteResult, StepState
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import TaskName
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import ArmState, RobotState
from roboscript.demos import BLOBS_DIR, Keypoint, KeypointSet, Provenance, Tolerance
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.learning import DemoPackage, save_package
from roboscript.motion import MockPlanner
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.demos import replay
from roboscript_robotwin.demos.validate import Validation, validate
from roboscript_robotwin.frames import Frame
from roboscript_robotwin.sim import referee
from roboscript_robotwin.sim.simulator import Opened

TASK = TaskName("lift_red")


def package(x: float, frame: int, success: int, grasp: bool = False) -> DemoPackage:
    """A package of MockEnv's lift at seed 3 with one left-arm keypoint at `x`: an approach, or
    with `grasp` the close."""
    keypoint = Keypoint(
        index=1,
        frame=frame,
        tcp=Pose.from_list([x, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]),
        gripper="closed" if grasp else "open",
        cue="close" if grasp else "stop",
        phase="grasp" if grasp else "approach",
        tolerance=Tolerance(),
    )
    keypoints = KeypointSet(
        task=TASK,
        seed=3,
        frame_steps=1,
        success_frame=success,
        config_hash="x",
        pointing={"left": (1.0, 0.0, 0.0)},
        arms={"left": [keypoint]},
    )
    return DemoPackage(task=TASK, seed=3, instruction="lift it", keypoints=keypoints)


async def run(tmp_path: Path, demo: DemoPackage) -> Validation:
    env = MockEnv(MockConfig(task=TASK, blob_dir=tmp_path / "blobs"))
    save_package(tmp_path / "package", demo)
    return await validate(
        env,
        lambda _: env,
        MockPlanner(),
        env.profile,
        tmp_path / "package",
        7,
        tmp_path / "referee",
        replay.ReplayConfig(),
        attempts=3,
        spacing=(0.01, 0.005),
        grip_threshold=0.5,
    )


class FakeSim:
    """What the recorder's `replay` uses of a `Simulator`: an episode whose success latches with
    the `latch`-th action, each action 7 physics steps, the box 1 cm further along y after each."""

    task = TASK
    profile = ROBOTWIN.profile()

    def __init__(self, latch: int | None) -> None:
        self.latch, self.sent, self.steps, self.closed = latch, [], 0, False

    def open(self, spec: EpisodeSpec) -> Opened:
        return Opened(True, "lift the box", 400, 1)

    @contextlib.contextmanager
    def stepping(self) -> Iterator[Callable[[], int]]:
        yield lambda: self.steps

    def observe(self, cameras: list[str]) -> tuple[dict[str, Frame], RobotState, str]:
        frames = {}
        for name in cameras:
            h, w = self.profile.cameras[name].height, self.profile.cameras[name].width
            rgb = np.full((h, w, 3), len(self.sent), dtype=np.uint8)
            frames[name] = Frame(np.eye(3), np.eye(4), rgb, np.full((h, w), 0.5, np.float32))
        return frames, self.state(), "lift the box"

    def execute(self, batch: ActionBatch) -> ExecuteResult:
        self.sent.append(batch.steps[0].qpos)
        self.steps += 7
        state = StepState(index=0, robot=self.state())
        return ExecuteResult(
            batch_id=batch.batch_id, states=[state], executed=1, stopped="completed"
        )

    def judge(self) -> tuple[bool, int | None, int, int | None, bool]:
        latched = self.latch is not None and len(self.sent) >= self.latch
        return latched, None, len(self.sent), 400, False

    def truth(self) -> referee.Truth:
        box = Pose(p=(0.0, 0.01 * len(self.sent), 0.76), q=(1.0, 0.0, 0.0, 0.0))
        touched: dict[Any, list[str]] = {"left": [], "right": []}
        return referee.Truth({"box": box}, {}, touched, {"box": "box"})

    def close(self) -> None:
        self.closed = True

    def state(self) -> RobotState:
        q = self.sent[-1] if self.sent else [0.0] * 14
        here = Pose(p=(0.1, 0.0, 0.9), q=(1.0, 0.0, 0.0, 0.0))
        arms = {
            arm: ArmState(
                ee=here,
                tcp=here,
                joints=q[i : i + 6],
                joints_cmd=q[i : i + 6],
                gripper=q[i + 6],
                gripper_cmd=q[i + 6],
            )
            for arm, i in (("left", 0), ("right", 7))
        }
        return RobotState(step=len(self.sent), step_limit=400, arms=arms, qpos_cmd=q)


def source(frames: int) -> Any:
    from roboscript_robotwin.sim.record import Source

    commands = np.array([[0.01 * t] * 6 + [1.0] + [0.0] * 6 + [1.0] for t in range(frames)])
    where = Provenance(source="test", path=Path("/x"), sha256="0" * 64, episode=7)
    return Source(commands, 15, "lift the box", where)


def test_a_replay_records_up_to_the_success_latch(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    from roboscript_robotwin.sim.record import replay

    sim = FakeSim(latch=3)
    spec = EpisodeSpec(task=TASK, seed=3, stage="demo", instruction="lift the box")
    recording = replay(sim, source(6), spec, "s1", 2, 0.5, tmp_path)  # type: ignore[arg-type]
    assert recording is not None and sim.closed and len(sim.sent) == 3  # stopped at the latch
    demo = recording.demo
    assert (demo.meta.frames, demo.meta.success_frame, demo.meta.frame_steps) == (4, 3, 15)
    assert demo.meta.replay == "s1" and demo.meta.instructions == ["lift the box"]
    np.testing.assert_array_equal(demo.source_frames, [0, 1, 2, 3])
    np.testing.assert_array_equal(demo.steps(), [0, 7, 14, 21])  # physics, not frame_steps
    assert demo.arms["left"].joints is not None
    np.testing.assert_allclose(demo.arms["left"].joints[:, 0], [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(recording.truth.poses[:, 0, 1], [0.0, 0.01, 0.02, 0.03])
    assert recording.truth.categories == {"box": "box"}
    store = LocalBlobStore(tmp_path / BLOBS_DIR)
    assert [f.alias for f in recording.frames] == ["frame_1", "frame_2", "frame_3", "frame_4"]
    last = Observation.model_validate_json(store.get_bytes(recording.frames[-1].observation))
    head = last.cameras["head_camera"]
    assert (last.frame_id, last.step) == ("demo-lift_red-3-3", 3)
    assert head.depth is not None and head.rgb.media_type == head.depth.media_type == "image/png"
    assert store.get_image(head.rgb)[0, 0, 0] == 3  # after the third action
    assert store.get_array(head.depth) == pytest.approx(np.full((480, 640), 0.5))
