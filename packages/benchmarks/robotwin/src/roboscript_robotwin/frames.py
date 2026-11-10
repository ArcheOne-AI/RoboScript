"""RoboTwin camera captures as the agent plane's `Observation`: the one place they become one, for
the env host, the demo recorder and P3's relay. The images go into the blob store as
`LocalBlobStore.put_frame` stores them (RGB as PNG, depth as a 16-bit PNG of millimetres), each
with its calibration, since the wrist cameras move. A depth pixel that hit nothing (the simulator's
inf, or past what the PNG holds) is stored as no return, 0. Agent side: no simulator code."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from roboscript.blobs import LocalBlobStore
from roboscript.blobs.codec import DEPTH_SCALE_M
from roboscript.contracts.camera import CameraModel
from roboscript.contracts.ids import FrameId, new_id
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import RobotState

MAX_DEPTH_M = DEPTH_SCALE_M * np.iinfo(np.uint16).max


@dataclass(frozen=True)
class Frame:
    """One camera's capture: intrinsics, OpenCV camera-to-world, RGB and depth in meters."""

    K: NDArray[np.float64]
    cam2world: NDArray[np.float64]
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]


def observation(
    frames: Mapping[str, Frame],
    robot: RobotState,
    instruction: str | None,
    blobs: LocalBlobStore,
    *,
    depth: bool = True,
    frame_id: FrameId | None = None,
) -> Observation:
    """The cameras' `frames` with the robot state as an observation, the images put into
    `blobs` (without depth when `depth` is false). Blocks on encoding and file writes."""
    cameras = {}
    for name, frame in frames.items():
        height, width = frame.rgb.shape[:2]
        camera = CameraModel(
            name=name,
            width=width,
            height=height,
            K=np.asarray(frame.K, dtype=float).tolist(),
            cam2world=np.asarray(frame.cam2world, dtype=float).tolist(),
        )
        cameras[name] = blobs.put_frame(camera, frame.rgb, no_hit(frame.depth) if depth else None)
    return Observation(
        frame_id=frame_id or FrameId(new_id("frame")),
        step=robot.step,
        cameras=cameras,
        robot=robot,
        instruction=instruction,
    )


def no_hit(depth: NDArray[np.floating]) -> NDArray[np.float32]:
    """Depth in meters with 0 where nothing was hit: not finite, negative or beyond `MAX_DEPTH_M`
    (a wrist camera pointing past the table sees nothing)."""
    depth = np.asarray(depth, dtype=np.float32)
    return np.where(np.isfinite(depth) & (depth >= 0) & (depth <= MAX_DEPTH_M), depth, 0.0).astype(
        np.float32
    )
