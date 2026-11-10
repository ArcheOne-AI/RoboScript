"""roboscript_robotwin.demos on an expert episode in RoboTwin's layout, recorded the way RoboTwin's
collector saves frames. Needs h5py (the simulator env has it)."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roboscript.demos import KeypointConfig, demo_keypoints
from roboscript_robotwin.benchmark import ROBOTWIN

h5py = pytest.importorskip("h5py")
from roboscript_robotwin.demos.expert import SETTING, load_episode, read_images  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "examples" / "simBenchmarks" / "RoboTwin" / "scripts" / "select_demos.py"
PROFILE = ROBOTWIN.profile()
STEPS = 15
CAMERA_TO_WORLD = np.array(  # 1.1 m above the table centre, looking straight down
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
)
K = np.array([[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]])
TASKS = ("move_playingcard_away", "place_a2b_left")  # categories C5 and C4 in ROBOTWIN


def pick(dx: float) -> tuple[np.ndarray, np.ndarray]:
    """TCP positions and gripper commands of a top-down pick and place, at the frames RoboTwin's
    collector saves: each action's start (repeating the frame before), after its first step,
    every 15 steps and its end."""
    states = [np.array([0.0, 0.0, 0.9, 1.0])]  # x, y, z, gripper
    frames: list[int] = []

    def act(steps: list[np.ndarray]) -> None:
        first = len(states)
        states.extend(steps)
        frames.extend([first - 1, *range(first, len(states), STEPS), len(states) - 1])

    def move(d: tuple[float, float, float], n: int = 200) -> None:
        w = np.sin(np.linspace(0, np.pi, n + 2)[1:-1]) ** 2
        s = states[-1]
        act([np.r_[s[:3] + f * np.asarray(d), s[3]] for f in np.cumsum(w) / w.sum()])
        act([states[-1]] * 30)

    def gripper(to: float) -> None:
        s = states[-1]
        act([np.r_[s[:3], s[3] + (to - s[3]) * k / 200] for k in range(1, 201)])
        act([states[-1]] * 30)

    act([states[-1]] * 30)
    move((0.2 + dx, 0.0, -0.05))
    move((0.0, 0.0, -0.08), 120)
    gripper(0.0)
    move((0.0, 0.0, 0.1))
    move((0.15, 0.1, 0.0))
    move((0.0, 0.0, -0.08), 120)
    gripper(1.0)
    move((0.0, 0.0, 0.1))
    recorded = np.array(states)[frames]
    return recorded[:, :3], recorded[:, 3]


def write_task(root: Path, task: str, dxs: list[float]) -> list[int]:
    """RoboTwin's expert layout for one task, an episode per offset; returns the seeds."""
    base = root / task / SETTING
    for d in ("data", "instructions", "video"):
        (base / d).mkdir(parents=True)
    seeds = [3 * i + 1 for i in range(len(dxs))]
    (base / "seed.txt").write_text(" ".join(map(str, seeds)), encoding="utf-8")
    down = [0.70710678, 0.0, 0.70710678, 0.0]  # w, x, y, z: the gripper pointing down
    for i, dx in enumerate(dxs):
        tcp, grip = pick(dx)
        n = len(tcp)
        ee = np.c_[tcp + np.array([0.0, 0.0, 0.12]), np.tile(down, (n, 1))]  # above the TCP
        idle = np.tile([0.3, -0.3, 0.9, *down], (n, 1))
        buf = io.BytesIO()
        Image.new("RGB", (32, 24), (90, 140, 200)).save(buf, "JPEG")
        world2cam = np.tile(np.linalg.inv(CAMERA_TO_WORLD)[:3], (n, 1, 1)).astype(np.float32)
        with h5py.File(base / "data" / f"episode{i}.hdf5", "w") as f:
            f["endpose/left_endpose"], f["endpose/right_endpose"] = ee, idle
            f["endpose/left_gripper"], f["endpose/right_gripper"] = grip, np.ones(n)
            f["joint_action/left_arm"] = f["joint_action/right_arm"] = np.zeros((n, 6))
            f["observation/head_camera/rgb"] = np.array([buf.getvalue()] * n, dtype="S")
            f["observation/head_camera/intrinsic_cv"] = np.tile(K, (n, 1, 1)).astype(np.float32)
            f["observation/head_camera/extrinsic_cv"] = world2cam
        seen = [f"Pick the block and place it, take {i}.", "Move the block."]
        instructions = json.dumps({"seen": seen, "unseen": []})
        (base / "instructions" / f"episode{i}.json").write_text(instructions, encoding="utf-8")
        (base / "video" / f"episode{i}.mp4").write_bytes(b"not really an mp4 %d" % i)
    return seeds


def test_an_episode_as_a_demo(tmp_path: Path) -> None:
    seeds = write_task(tmp_path, TASKS[0], [0.0, 0.01])
    demo = load_episode(tmp_path, TASKS[0], 1, PROFILE)
    meta = demo.meta
    assert (meta.task, meta.seed, meta.frame_steps) == (TASKS[0], seeds[1], 15)
    assert meta.frames == len(pick(0)[0])
    assert meta.ee_to_tcp == {arm: spec.ee_to_tcp for arm, spec in PROFILE.arms.items()}
    assert meta.instructions[0] == "Pick the block and place it, take 1."
    path = tmp_path / TASKS[0] / SETTING / "data" / "episode1.hdf5"
    assert meta.provenance.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta.provenance.episode == 1
    np.testing.assert_allclose(demo.arms["left"].tcp[:, :3], pick(0.01)[0], atol=1e-6)
    np.testing.assert_allclose(demo.calib["head_camera"].cam2world[0], CAMERA_TO_WORLD, atol=1e-6)
    camera = meta.cameras["head_camera"]
    assert (camera.width, camera.height) == (32, 24)
    [image] = read_images(demo, "head_camera", [3])
    assert image.shape == (24, 32, 3) and abs(int(image[5, 5, 2]) - 200) < 8
    kps = demo_keypoints(demo, KeypointConfig(), KeypointConfig().tolerance)
    assert [k.phase for k in kps.arms["left"]] == [
        "approach", "grasp", "lift", "place", "release", "retreat"
    ]  # fmt: skip
