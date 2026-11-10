"""RoboTwin's expert episodes (`<root>/<task>/aloha-agilex_clean_50/`) as `Demo`s.

A task directory holds `data/episode<i>.hdf5`, `instructions/episode<i>.json` (`seen` and
`unseen` lists), `video/episode<i>.mp4` (the head camera) and `seed.txt`, the seeds of episodes
0, 1, ... in order (the collector kept only seeds its expert solved). In an episode file,
`endpose/<arm>_endpose` (N, 7) is `robot.get_<arm>_ee_pose()` and `endpose/<arm>_gripper` the
gripper command (equal to `joint_action/<arm>_gripper`); `joint_action/<arm>_arm` (N, 6) holds
joint drive targets; `observation/<camera>/` one JPEG per frame (`rgb`), `intrinsic_cv` and
`extrinsic_cv` (world to camera, OpenCV axes). RoboTwin's collector (`_take_picture` in
`take_dense_action` and `together_move_to_pose`) saved a frame at each action's start, which
repeats the frame before, one after the action's first physics step, then every `save_freq` = 15
steps (task_config/demo_clean.yml) and one at the action's end.

The TCP comes from the embodiment profile (`ArmProfile.ee_to_tcp`, 0.12 m along the ee's x axis,
the direction the gripper points): RoboTwin's `get_<arm>_tcp_pose()` is the ee pose moved
`gripper_bias` = 0.12 m along that axis with the same orientation (RoboTwin's
envs/robot/robot.py `_trans_endpose`; aloha-agilex config.yml: delta_matrix identity).
"""

import io
import json
from pathlib import Path
from typing import Final

import h5py
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from roboscript.contracts.ids import TaskName
from roboscript.demos import (
    CameraCalib,
    CameraTrack,
    Demo,
    DemoMeta,
    FrameSource,
    Provenance,
    arm_track,
)
from roboscript.demos.manifest import content_sha256
from roboscript.embodiments import EmbodimentProfile

SETTING: Final = "aloha-agilex_clean_50"
SOURCE: Final = "robotwin.expert_demo"
FRAME_STEPS: Final = 15


def task_dir(root: Path, task: str) -> Path:
    return root / task / SETTING


def seeds(root: Path, task: str) -> list[int]:
    """The seed of each episode, in episode order."""
    return [int(s) for s in (task_dir(root, task) / "seed.txt").read_text("utf-8").split()]


def video_path(root: Path, task: str, episode: int) -> Path:
    return task_dir(root, task) / "video" / f"episode{episode}.mp4"


def load_episode(root: Path, task: str, episode: int, profile: EmbodimentProfile) -> Demo:
    """Episode `episode` of `task`, recorded with the embodiment of `profile`."""
    base = task_dir(root, task)
    path = base / "data" / f"episode{episode}.hdf5"
    instructions = json.loads((base / "instructions" / f"episode{episode}.json").read_text("utf-8"))
    try:
        with h5py.File(path, "r") as f:
            arms = {
                arm: arm_track(
                    f[f"endpose/{arm}_endpose"][:],
                    f[f"endpose/{arm}_gripper"][:],
                    spec.ee_to_tcp,
                    f[f"joint_action/{arm}_arm"][:],
                )
                for arm, spec in profile.arms.items()
            }
            cameras, calib = {}, {}
            for name, obs in f["observation"].items():
                world2cam = np.zeros((len(obs["extrinsic_cv"]), 4, 4))
                world2cam[:, :3] = obs["extrinsic_cv"][:]
                world2cam[:, 3, 3] = 1.0
                calib[name] = CameraCalib(
                    K=obs["intrinsic_cv"][:].astype(float), cam2world=np.linalg.inv(world2cam)
                )
                width, height = Image.open(io.BytesIO(obs["rgb"][0])).size
                source = FrameSource(path=path, dataset=obs["rgb"].name.lstrip("/"), codec="jpeg")
                cameras[name] = CameraTrack(width=width, height=height, images=source)
        meta = DemoMeta(
            task=TaskName(task),
            seed=seeds(root, task)[episode],
            instructions=instructions["seen"],
            frames=len(next(iter(arms.values())).ee),
            frame_steps=FRAME_STEPS,
            ee_to_tcp={arm: spec.ee_to_tcp for arm, spec in profile.arms.items()},
            cameras=cameras,
            provenance=Provenance(
                source=SOURCE, path=path, sha256=content_sha256(path), episode=episode
            ),
        )
        return Demo(meta, arms, calib)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: not a RoboTwin expert episode: {error}") from None


def read_images(demo: Demo, camera: str, frames: list[int]) -> list[NDArray[np.uint8]]:
    """RGB images (H, W, 3) of `camera` at `frames`."""
    source = demo.meta.cameras[camera].images
    if source is None:
        raise ValueError(f"{demo.meta.task}: camera {camera} has no images")
    with h5py.File(source.path, "r") as f:
        data = f[source.dataset]
        return [np.asarray(Image.open(io.BytesIO(data[t])).convert("RGB")) for t in frames]
