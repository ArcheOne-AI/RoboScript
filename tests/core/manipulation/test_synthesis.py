"""Specs to official points on synthetic proxies: the contact set by anchor, offset, closing axis
and yaw, mirroring a grasp for the other arm, and placement targets by anchor frame and kind."""

import math

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.manipulation import (
    GraspSpec,
    ObjectProxy,
    Part,
    PlaceSpec,
    PlaceTarget,
    grasp_candidates,
    mirror,
    target_frame,
)
from roboscript.manipulation.posemath import axangle

# A root turned a quarter about z.
BASE = Pose(p=(0.0, -0.6, 0.0), q=(0.7071068, 0.0, 0.0, 0.7071068))
POSES = load_embodiment(MOCK_BIMANUAL).poses  # the pose conventions


def rigid(rotation: np.ndarray, position: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, position
    return m


def proxy(
    size: tuple[float, float, float],
    symmetry: str = "pi",
    x_known: bool = False,
    yaw: float = 0.0,
    at: tuple[float, float, float] = (0.1, 0.0, 0.8),
    parts: dict[str, Part] | None = None,
) -> ObjectProxy:
    """A box whose frame (and box center) is at `at`, turned by `yaw` (rad)."""
    whole = Part(frame=np.eye(4), size=np.array(size), x_known=x_known)
    frame = rigid(axangle([0, 0, 1], yaw), at)
    return ObjectProxy("box", frame, symmetry, {"whole": whole, **(parts or {})})  # type: ignore[arg-type]


def create_box(kind: str, half: tuple[float, float, float]) -> list[np.ndarray]:
    """A box asset's contact points, scaled."""
    if kind == "default":
        raw = [
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
        ]
    else:
        raw = [
            [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, z], [0, 0, 0, 1]]
            if k == 0
            else [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, z], [0, 0, 0, 1]]
            if k == 1
            else [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, z], [0, 0, 0, 1]]
            if k == 2
            else [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, z], [0, 0, 0, 1]]
            for z in (0.7, -0.7)
            for k in range(4)
        ]
    out = []
    for m in raw:
        a = np.array(m, dtype=float)
        a[:3, 3] *= half
        out.append(a)
    return out


def test_anchor_offset_closing_and_yaw_set_the_contact() -> None:
    box = proxy((0.1, 0.04, 0.03), "none", x_known=True)
    spec = GraspSpec(
        approach="top",
        anchor="top_center",
        offset_m=(0.02, 0.0, -0.005),
        closing="minor",
        yaw_deg=30,
    )
    (only,) = grasp_candidates(spec, box, "left")
    assert only.matrix[:3, 3] == pytest.approx([0.02, 0.0, 0.015 - 0.005])
    assert only.matrix[:3, 1] == pytest.approx([0, 0, 1])  # y against the approach (down)
    turned = axangle([0, 0, -1], math.radians(30)) @ np.array([0.0, 1, 0])
    assert only.matrix[:3, 2] == pytest.approx(turned)  # minor (y) turned 30 deg about the approach
    (side,) = grasp_candidates(
        GraspSpec(approach="side", anchor="side_center", closing="major"),
        proxy((0.1, 0.04, 0.03), "none", x_known=True),
        "left",
    )[:1]
    assert side.matrix[:3, 3] == pytest.approx([0.0, -0.02, 0.0])  # the -y face
    assert side.matrix[:3, 1] == pytest.approx([0, -1, 0])  # approaching along +y
    assert side.matrix[:3, 2] == pytest.approx([1, 0, 0])  # x (10 cm) is longer than z (3 cm)


def test_mirroring_reflects_the_grasp_for_the_other_arm() -> None:
    box = proxy((0.1, 0.04, 0.03), "none", x_known=True, yaw=0.4)
    spec = GraspSpec(approach="top", offset_m=(0.01, 0.02, 0.0), yaw_deg=20, arm="right")
    (right,) = grasp_candidates(spec, box, "right")
    left, rolled = grasp_candidates(spec, box, "left")
    assert left.matrix[:3, 3] == pytest.approx(right.matrix[:3, 3] * [1, -1, 1])
    assert np.linalg.det(left.matrix[:3, :3]) == pytest.approx(1.0)
    assert mirror(left.matrix, np.eye(4)) == pytest.approx(right.matrix)
    # The same grasp rolled half a turn about the approach: the closing axis reversed.
    assert rolled.matrix[:3, 3] == pytest.approx(left.matrix[:3, 3])
    assert rolled.matrix[:3, 1] == pytest.approx(left.matrix[:3, 1])
    assert rolled.matrix[:3, 2] == pytest.approx(-left.matrix[:3, 2])
    # Unwritten for an arm, a spec serves both as written.
    both = GraspSpec(approach="top", offset_m=(0.01, 0.02, 0.0))
    assert np.allclose(
        grasp_candidates(both, box, "left")[0].matrix,
        grasp_candidates(both, box, "right")[0].matrix,
    )


def test_place_targets_by_anchor_frame_and_kind() -> None:
    plate = proxy((0.2, 0.1, 0.02), "pi", yaw=math.pi / 2, at=(0.1, 0.0, 0.75))
    on = PlaceSpec(target=PlaceTarget(kind="object", obj="plate", anchor="top_center"))
    frame = target_frame(on, plate, BASE)
    assert frame[:3, 3] == pytest.approx([0.1, 0.0, 0.76])
    assert frame[:3, 0] == pytest.approx([0, 1, 0])  # the plate's axes
    world = target_frame(on.model_copy(update={"frame": "world_axes"}), plate, BASE)
    assert world[:3, :3] == pytest.approx(np.eye(3)) and world[:3, 3] == pytest.approx(
        [0.1, 0.0, 0.76]
    )
    edge = PlaceSpec(target=PlaceTarget(kind="object", obj="plate", anchor="edge:front"))
    assert target_frame(edge, plate, BASE)[:3, 3] == pytest.approx(
        [0.1, -0.1, 0.76]
    )  # 20 cm along y
    fixed = PlaceSpec(target=PlaceTarget(kind="robot_frame"), offset_m=(0.52, 0.0, 0.75))
    base = target_frame(fixed, None, BASE)
    assert base[:3, 3] + base[:3, :3] @ np.array(fixed.offset_m) == pytest.approx(
        [0.0, -0.08, 0.75]
    )
    landmark = PlaceSpec(target=PlaceTarget(kind="landmark", landmark="table_center"))
    assert target_frame(landmark, Pose(p=(0.0, 0.0, 0.7), q=(1, 0, 0, 0)), BASE)[
        :3, 3
    ] == pytest.approx([0, 0, 0.7])
    with pytest.raises(ValueError, match="landmark"):
        target_frame(landmark, None, BASE)
