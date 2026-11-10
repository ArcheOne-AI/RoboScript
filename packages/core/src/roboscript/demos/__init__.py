"""Demonstrations (docs/concepts/learning.md): the format, keypoints, the choice of one
demonstration per task, its re-recording, and the keypoint referee of S1. Pure numpy; no
simulator.

- `demo`: `Demo` (per-frame arm tracks, camera calibration, image references; a re-recording's
  measured proprioception, replay mapping and observations) as `demo.json` + `arrays.npz`
  (+ `frames.jsonl` and `blobs/`).
- `keypoints`: per-arm keypoints from the TCP track and the gripper command.
- `selection`: offline quality ranking of a task's candidate episodes.
- `replay`: the frames of a recorded joint trajectory a re-recording sends.
- `compare`: a re-recording's keypoints against the replayed trajectory's.
- `disturbance`: what a demonstration moved besides its task, from the referee's ground truth.
- `referee`: an executed path against the keypoints, and the S1 pass rule.
- `manifest`: the chosen demonstration per task.
"""

from roboscript.demos.compare import (
    CompareConfig,
    KeypointDiff,
    KeypointPair,
    compare_keypoints,
)
from roboscript.demos.config import DemosConfig, RecordingConfig
from roboscript.demos.demo import (
    BLOBS_DIR,
    ArmTrack,
    CameraCalib,
    CameraTrack,
    Demo,
    DemoMeta,
    FrameSource,
    Provenance,
    arm_track,
    frame_alias,
    load_demo,
    load_frames,
    save_demo,
    save_frames,
    tcp_from_ee,
)
from roboscript.demos.disturbance import (
    Disturbance,
    DisturbanceConfig,
    ObjectMotion,
    SceneTruth,
    disturbance,
    load_truth,
    save_truth,
)
from roboscript.demos.keypoints import (
    Keypoint,
    KeypointConfig,
    KeypointSet,
    Tolerance,
    arm_keypoints,
    demo_keypoints,
    gripper_angle_deg,
    pointing_axis,
    timed_frames,
)
from roboscript.demos.manifest import DemoManifest, FileRef, TaskDemo
from roboscript.demos.referee import (
    ArmPath,
    ArmProgress,
    ObjectShape,
    RefereeConfig,
    Verdict,
    arm_progress,
    held_placements,
    judge,
    keypoint_objects,
    keypoint_sizes,
    object_shapes,
    symmetry_axes,
)
from roboscript.demos.replay import replay_frames
from roboscript.demos.selection import (
    EpisodeFeatures,
    RankedEpisode,
    SelectionConfig,
    episode_features,
    rank_episodes,
)

__all__ = [
    "BLOBS_DIR",
    "ArmPath",
    "ArmProgress",
    "ArmTrack",
    "CameraCalib",
    "CameraTrack",
    "CompareConfig",
    "Demo",
    "DemoManifest",
    "DemoMeta",
    "DemosConfig",
    "Disturbance",
    "DisturbanceConfig",
    "EpisodeFeatures",
    "FileRef",
    "FrameSource",
    "Keypoint",
    "KeypointConfig",
    "KeypointDiff",
    "KeypointPair",
    "KeypointSet",
    "ObjectMotion",
    "ObjectShape",
    "Provenance",
    "RankedEpisode",
    "RecordingConfig",
    "RefereeConfig",
    "SceneTruth",
    "SelectionConfig",
    "TaskDemo",
    "Tolerance",
    "Verdict",
    "arm_keypoints",
    "arm_progress",
    "arm_track",
    "compare_keypoints",
    "demo_keypoints",
    "disturbance",
    "episode_features",
    "frame_alias",
    "gripper_angle_deg",
    "held_placements",
    "judge",
    "keypoint_objects",
    "keypoint_sizes",
    "load_demo",
    "load_frames",
    "load_truth",
    "object_shapes",
    "pointing_axis",
    "rank_episodes",
    "replay_frames",
    "save_demo",
    "save_frames",
    "save_truth",
    "symmetry_axes",
    "tcp_from_ee",
    "timed_frames",
]
