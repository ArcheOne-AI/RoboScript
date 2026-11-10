"""roboscript.learning.demo: a demo package built from a synthetic re-recording (frames and
keyframes chosen, overlays drawn, blobs carried), validated, loaded into another blob store (only
the blobs it references, checked), and read by the Demo Reader with frame aliases that match the
keypoints'; its ground-truth layer, kept apart from the agent-facing part."""

import functools
from pathlib import Path

import numpy as np
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.camera import CameraModel
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import FrameId, TaskName
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.contracts.robot import ArmState, RobotState
from roboscript.demos import (
    BLOBS_DIR,
    CameraCalib,
    CameraTrack,
    Demo,
    DemoMeta,
    Keypoint,
    KeypointSet,
    Provenance,
    RecordingConfig,
    SceneTruth,
    Tolerance,
    arm_track,
    frame_alias,
    save_demo,
    save_frames,
)
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.learning import (
    DemoPackage,
    import_blobs,
    load_demo_truth,
    load_package,
    validate_package,
)
from roboscript.learning.demo import TRUTH_FILE
from roboscript.learning.demo import build_package as _build_package
from roboscript.testing.scenes import PERCEPTION

PROFILE = load_embodiment(MOCK_BIMANUAL)
build_package = functools.partial(_build_package, table=PERCEPTION.table)
"""`build_package` on the mock scenes' table."""
TASK = TaskName("lift_red")
N = 12
DOWN = (0.7071068, 0.0, 0.7071068, 0.0)
# Cameras 1.5 m up, looking straight down: the table fills the view.
LOOKING_DOWN = ((1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, -1.0, 1.5), (0, 0, 0, 1.0))
SIZES = {"overhead": (64, 48), "wrist_left": (32, 24), "wrist_right": (32, 24)}


def robot(t: int) -> RobotState:
    tcp = Pose.from_list([0.01 * t, 0.0, 0.85, *DOWN])
    arm = ArmState(
        ee=tcp, tcp=tcp, joints=[0.0] * 6, joints_cmd=[0.0] * 6, gripper=1.0, gripper_cmd=1.0
    )
    return RobotState(step=t, step_limit=100, arms={"left": arm, "right": arm}, qpos_cmd=[0.0] * 14)


def record(directory: Path) -> None:
    """A recording with observations: N frames, three RGB-D cameras on the table."""
    blobs = LocalBlobStore(directory / BLOBS_DIR)
    entries = []
    for t in range(N):
        cameras = {}
        for name, (w, h) in SIZES.items():
            K = ((30.0 * w / 64, 0.0, w / 2), (0.0, 30.0 * w / 64, h / 2), (0.0, 0.0, 1.0))
            camera = CameraModel(name=name, width=w, height=h, K=K, cam2world=LOOKING_DOWN)
            rgb = np.full((h, w, 3), 40 + 10 * t, dtype=np.uint8)
            cameras[name] = blobs.put_frame(camera, rgb, np.full((h, w), 0.76, dtype=np.float32))
        frame_id = FrameId(f"demo-{t}")
        seen = Observation(
            frame_id=frame_id, step=t, cameras=cameras, robot=robot(t), instruction="lift it"
        )
        blob = blobs.put_bytes(seen.model_dump_json().encode(), "application/json")
        entries.append(
            FrameEntry(frame_id=frame_id, alias=frame_alias(t), step=t, observation=blob)
        )
    states = [robot(t) for t in range(N)]
    offset = (0.1, 0.0, 0.0)
    meta = DemoMeta(
        task=TASK,
        seed=7,
        instructions=["lift it"],
        frames=N,
        frame_steps=15,
        ee_to_tcp={"left": offset, "right": offset},
        cameras={name: CameraTrack(width=w, height=h) for name, (w, h) in SIZES.items()},
        success_frame=9,
        replay="s1",
        observations=True,
        provenance=Provenance(source="synthetic", path=Path("/x"), sha256="0" * 64),
    )
    arms = {
        arm: arm_track([s.arms[arm].ee.as_list() for s in states], np.ones(N), offset)
        for arm in ("left", "right")
    }
    calib = {
        name: CameraCalib(np.tile(np.eye(3), (N, 1, 1)), np.tile(np.eye(4), (N, 1, 1)))
        for name in SIZES
    }
    save_demo(Demo(meta, arms, calib, np.arange(N)), directory)
    save_frames(directory, entries)


def keypoints(*frames: tuple[int, str, str]) -> KeypointSet:
    kps = [
        Keypoint(
            index=i,
            frame=frame,
            tcp=robot(frame).arms["left"].tcp,
            gripper="closed" if cue == "close" else "open",
            cue=cue,  # type: ignore[arg-type]
            phase=phase,  # type: ignore[arg-type]
            tolerance=Tolerance(),
        )
        for i, (frame, cue, phase) in enumerate(frames, start=1)
    ]
    return KeypointSet(
        task=TASK,
        seed=7,
        frame_steps=15,
        success_frame=9,
        config_hash="x",
        pointing={"left": (1.0, 0.0, 0.0)},
        arms={"left": kps},
    )


LIFT = keypoints((3, "stop", "approach"), (6, "close", "grasp"), (8, "stop", "lift"))


async def lift_package(tmp_path: Path) -> DemoPackage:
    """The synthetic recording (`tmp_path/recording`) packaged at `tmp_path/package`."""
    record(tmp_path / "recording")
    return await build_package(
        tmp_path / "recording",
        LIFT,
        "lift it",
        tmp_path / "package",
        RecordingConfig(every=5),
        PROFILE,
    )


async def truth_package(tmp_path: Path) -> DemoPackage:
    """The same, at `tmp_path/p`, with a ground-truth layer: a box sliding along x and a laptop
    whose hinge opens."""
    record(tmp_path / "recording")
    poses = np.zeros((N, 2, 7))
    poses[..., 3] = 1.0
    poses[:, 0, 0] = np.arange(N) * 0.01  # the box slides along x
    truth = SceneTruth(
        objects=["box", "laptop"],
        poses=poses,
        touched={"left": np.zeros((N, 2), bool)},
        joints={"laptop": np.linspace(0.0, 1.0, N)[:, None]},
        categories={"box": "box", "laptop": "laptop"},
    )
    config = RecordingConfig(every=5)
    return await build_package(
        tmp_path / "recording", LIFT, "lift it", tmp_path / "p", config, PROFILE, truth
    )


async def test_a_package_from_a_recording(tmp_path: Path) -> None:
    package = await lift_package(tmp_path)
    assert [e.alias for e in package.frames] == [frame_alias(t) for t in (0, 3, 5, 6, 8, 9, 10)]
    assert [k.caption for k in package.keyframes] == [
        "left K1 (approach)",
        "left K2 (grasp)",
        "left K3 (lift)",
    ]
    store = LocalBlobStore(tmp_path / "package" / BLOBS_DIR)
    head, wrist = package.keyframes[0].images
    assert store.get_image(head).shape == (48, 64, 3) and store.get_image(wrist).shape == (
        24,
        32,
        3,
    )
    # Overlays: the plain gray frame got marks and grid lines.
    assert len(np.unique(store.get_image(head).reshape(-1, 3), axis=0)) > 1
    assert load_package(tmp_path / "package") == package
    assert validate_package(tmp_path / "package") == []

    runner = LocalBlobStore(tmp_path / "runner-blobs")
    assert await import_blobs(tmp_path / "package", runner) == len(list(store.root.glob("*/*")))
    for entry in package.frames:
        seen = Observation.model_validate_json(runner.get_bytes(entry.observation))
        assert runner.exists(seen.cameras["overhead"].rgb)
    assert all(runner.exists(blob) for k in package.keyframes for blob in k.images)


async def test_the_ground_truth_layer_stays_apart(tmp_path: Path) -> None:
    package = await truth_package(tmp_path)
    layer = load_demo_truth(tmp_path / "p")
    assert layer is not None and layer.objects == {"box": "box", "laptop": "laptop"}
    assert [(f.frame, f.what) for f in layer.frames] == [
        (0, ["start"]),
        (3, ["left K1 (approach)"]),
        (6, ["left K2 (grasp)"]),
        (8, ["left K3 (lift)"]),
        (9, ["success"]),
    ]
    assert layer.frames[2].poses["box"].p == pytest.approx((0.06, 0.0, 0.0))
    assert layer.frames[4].joints["laptop"] == pytest.approx([9 / 11])
    assert "box" not in (tmp_path / "p" / "package.json").read_text(
        encoding="utf-8"
    )  # the agent-facing part
    assert load_package(tmp_path / "p") == package
    assert validate_package(tmp_path / "p") == []
    other = layer.model_copy(update={"frames": layer.frames[:2]})
    (tmp_path / "p" / TRUTH_FILE).write_text(other.model_dump_json(), encoding="utf-8")
    assert validate_package(tmp_path / "p") == ["the ground-truth layer lacks frames [6, 8, 9]"]
