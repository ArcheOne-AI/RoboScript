"""roboscript.demos.keypoints on synthetic motions recorded as a collector records them: pick and
place, the gripper threshold at 0.5, the order of a keypoint file, and the same keypoints at any
frame rate."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from roboscript.demos import (
    ArmTrack,
    Demo,
    Keypoint,
    KeypointConfig,
    KeypointSet,
    Tolerance,
    arm_keypoints,
    demo_keypoints,
    timed_frames,
)

CONFIG = KeypointConfig()
TOL = Tolerance()


def keypoints(track: ArmTrack, frame_steps: int = 1) -> list[Keypoint]:
    return arm_keypoints(track, frame_steps, CONFIG, TOL, timed_frames([track]))


def summary(kps: list[Keypoint]) -> list[tuple[str, str, str]]:
    return [(kp.cue, kp.phase, kp.gripper) for kp in kps]


PICK_AND_PLACE = [
    ("stop", "approach", "open"),
    ("close", "grasp", "closed"),  # the stop at the grasp pose gives way to the close
    ("stop", "lift", "closed"),
    ("stop", "place", "closed"),  # the last move before the release; setting down gives way
    ("open", "release", "open"),
    ("stop", "retreat", "open"),
]


@pytest.mark.parametrize("frame_steps", [1, 15])
def test_pick_and_place(pick_place: Callable[..., Any], frame_steps: int) -> None:
    kps = keypoints(pick_place().track(frame_steps), frame_steps)
    assert summary(kps) == PICK_AND_PLACE
    assert [kp.index for kp in kps] == [1, 2, 3, 4, 5, 6]
    assert all(kp.tolerance == TOL for kp in kps)
    grasp, lift = kps[1].tcp.p, kps[2].tcp.p
    np.testing.assert_allclose(grasp, [0.2, 0.0, 0.77], atol=1e-6)
    assert lift[2] - grasp[2] == pytest.approx(0.1, abs=1e-3)


def test_the_frame_rate_does_not_change_the_keypoints(pick_place: Callable[..., Any]) -> None:
    m = pick_place()
    dense, sparse = keypoints(m.track(1), 1), keypoints(m.track(15), 15)
    assert summary(dense) == summary(sparse)
    for a, b in zip(dense, sparse, strict=True):
        assert abs(m.frames(1)[a.frame] - m.frames(15)[b.frame]) <= 30  # physics steps
        np.testing.assert_allclose(a.tcp.p, b.tcp.p, atol=2e-3)


def test_a_gripper_at_exactly_the_threshold_is_open(motion: type) -> None:
    # Grippers pre-shaped to 0.5 on the way, then closed at the handles.
    m = motion().rest(20).move((0.2, 0.0, 0.0)).rest(20).gripper(0.5).rest(100).gripper(0.0)
    [close] = keypoints(m.rest(20).track())  # the stop where it closes gives way to the close
    assert (close.cue, close.phase) == ("close", "grasp")
    assert m.frames(1)[close.frame] == 20 + 200 + 20 + 200 + 100 + 1  # the first step below 0.5


def test_a_keypoint_file_keeps_its_order(
    pick_place: Callable[..., Any], make_demo: Callable[..., Demo]
) -> None:
    kps = demo_keypoints(make_demo({"left": pick_place()}), CONFIG, TOL)
    data = kps.model_dump(by_alias=True)
    del data["arms"]["left"][1]  # a keypoint deleted, the rest not renumbered
    with pytest.raises(ValidationError, match=r"numbered 1, 2, \.\.\. in order"):
        KeypointSet.model_validate(data)
    data = kps.model_dump(by_alias=True)
    data["arms"]["left"][0]["frame"] = 10**6
    with pytest.raises(ValidationError, match="in frame order"):
        KeypointSet.model_validate(data)
