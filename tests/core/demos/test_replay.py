"""roboscript.demos.replay: the frames of a joint trajectory each strategy sends."""

import numpy as np
import pytest

from roboscript.demos import replay_frames
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment

PROFILE = load_embodiment(MOCK_BIMANUAL)


def trajectory() -> np.ndarray:
    """20 frames: the left arm moves into frames 1-5 and rests, closes its gripper into frames
    8-10, moves again into 12-15; the right arm stays, its gripper open."""
    v = np.zeros((20, 14))
    v[:, 6] = v[:, 13] = 1.0
    v[1:6, 0] = np.linspace(0.02, 0.1, 5)
    v[6:, 0] = 0.1
    v[8:11, 6] = [0.7, 0.3, 0.0]
    v[11:, 6] = 0.0
    v[12:16, 1] = np.linspace(0.02, 0.08, 4)
    v[16:, 1] = 0.08
    return v


def test_strategies() -> None:
    v = trajectory()
    assert replay_frames("s1", v, PROFILE) == list(range(1, 20))
    assert replay_frames("s8", v, PROFILE) == [8, 16, 19]
    # Where the arm arrives at rest (frames 5 and 15), the frame before the gripper change and
    # its last (7 and 10), and the last frame.
    assert replay_frames("rest", v, PROFILE) == [5, 7, 10, 15, 19]
    with pytest.raises(ValueError, match="s<k>"):
        replay_frames("s0", v, PROFILE)
