"""roboscript.demos.disturbance: what a demonstration moved besides its task, from synthetic
ground truth, with settling and jitter in physics steps, and the referee file round trip."""

from pathlib import Path

import numpy as np
import pytest

from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import Arm
from roboscript.demos import (
    ArmTrack,
    Demo,
    DemoMeta,
    DisturbanceConfig,
    Provenance,
    SceneTruth,
    disturbance,
    load_truth,
    save_truth,
)

N = 40  # frames, 5 physics steps apart unless a test says otherwise
OBJECTS = ["cup", "lid", "ball", "plate", "block", "button"]
CONFIG = DisturbanceConfig(settle_steps=400)
HOLD: dict[Arm, float] = {"left": 0.2, "right": 0.2}  # a closed opening above this holds


def demo(
    left: np.ndarray, right: np.ndarray, openings: dict[str, np.ndarray], steps: np.ndarray
) -> Demo:
    def track(gripper: np.ndarray, opening: np.ndarray) -> ArmTrack:
        pose = np.tile([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0], (N, 1))
        return ArmTrack(ee=pose, tcp=pose, gripper=gripper, opening=opening)

    meta = DemoMeta(
        task=TaskName("t"),
        seed=0,
        instructions=["x"],
        frames=N,
        frame_steps=15,
        ee_to_tcp={"left": (0.1, 0.0, 0.0), "right": (0.1, 0.0, 0.0)},
        provenance=Provenance(source="synthetic", path=Path("/x"), sha256="0" * 64),
    )
    arms = {"left": track(left, openings["left"]), "right": track(right, openings["right"])}
    return Demo(meta, arms, physics_steps=np.r_[0, np.cumsum(steps)])


def scene(steps: np.ndarray | None = None) -> tuple[Demo, SceneTruth]:
    """The left arm grasps the cup at frame 10, carries it +1 cm a frame over frames 12-25 with
    a lid riding on it and a ball in it that falls out from frame 21, releases it at frame 26 (it
    settles 4 mm a frame into frames 27 and 28), then knocks the plate 1 cm a frame over frames
    30-32. A block jitters. The right arm, closed on nothing, presses a button 5 mm down at frame
    35. `steps`: the physics steps into each frame (default 5)."""
    left, right = np.ones(N), np.ones(N)
    left[10:26] = 0.0
    right[33:] = 0.0
    openings = {"left": np.where(left < 0.5, 0.4, 1.0), "right": np.where(right < 0.5, 0.18, 1.0)}
    position = np.zeros((N, len(OBJECTS), 3))
    step = np.zeros((N, len(OBJECTS), 3))
    step[12:26, 0:3, 0] = 0.01  # the cup carries the lid and the ball
    step[21:26, 2] = [0.0, 0.0, -0.05]  # the ball falls out
    step[27:29, 0, 2] = -0.004  # the cup settles
    step[30:33, 3, 1] = 0.01  # the plate knocked
    step[1:, 4, 0] = 0.0003  # jitter
    step[35, 5, 2] = -0.005  # the button pressed
    position = np.cumsum(step, axis=0)
    poses = np.concatenate([position, np.tile([1.0, 0.0, 0.0, 0.0], (N, len(OBJECTS), 1))], axis=2)
    touched = {
        "left": np.zeros((N, len(OBJECTS)), bool),
        "right": np.zeros((N, len(OBJECTS)), bool),
    }
    touched["left"][10:26, 0] = True
    touched["left"][29:33, 3] = True
    touched["right"][34:37, 5] = True
    drawer = np.zeros((N, 2))
    drawer[20:, 0], drawer[20:, 1] = 0.5, 0.4
    truth = SceneTruth(
        objects=OBJECTS,
        poses=poses,
        touched=touched,
        joints={"drawer": drawer},
        categories={"cup": "cup", "plate": "plate"},
    )
    return demo(left, right, openings, np.full(N - 1, 5) if steps is None else steps), truth


def test_what_was_moved_besides_the_task() -> None:
    d, truth = scene()
    result = disturbance(d, truth, CONFIG, HOLD)
    by = {o.name: o for o in result.objects}
    assert by["cup"].held and by["cup"].free_m == pytest.approx(0.0)  # carried, then settling
    assert by["lid"].free_m == pytest.approx(0.0)  # riding on the cup
    assert by["ball"].payload and not by["ball"].held  # rode, then went far on its own
    assert by["plate"].free_m == pytest.approx(0.03) and by["plate"].strike_m == pytest.approx(0.03)
    assert by["block"].free_m == 0.0  # jitter
    assert not by["button"].held and by["button"].strike_m == pytest.approx(0.005)  # a press
    assert result.disturbance_m == pytest.approx(0.035)  # the ball left out
    assert result.strike_m == pytest.approx(0.035)
    assert result.side_joints_rad == {"drawer": pytest.approx(0.4)}
    assert result.flags == [
        "arm strike 3.5 cm (most: plate 3.0 cm)",
        "disturbance 3.5 cm (most: plate 3.0 cm)",
        "drawer: 0.40 rad on joints other than the one the task moves",
    ]


def test_the_truth_round_trip(tmp_path: Path) -> None:
    _, truth = scene()
    save_truth(truth, tmp_path)
    back = load_truth(tmp_path)
    assert back.objects == OBJECTS
    assert back.categories == {"cup": "cup", "plate": "plate"}
    np.testing.assert_array_equal(back.poses, truth.poses)
    assert set(back.touched) == {"left", "right"}
    np.testing.assert_array_equal(back.touched["left"], truth.touched["left"])
    np.testing.assert_array_equal(back.joints["drawer"], truth.joints["drawer"])
