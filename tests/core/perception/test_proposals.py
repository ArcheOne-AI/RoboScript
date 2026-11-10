"""perception.proposals, overlay and grounding: proposals that cover every object a sensor could
see, the Set-of-Mark overlay and cards, and snapping 2D answers to proposals."""

import numpy as np
import pytest

from roboscript.contracts.ids import FrameId
from roboscript.perception.grounding import snap
from roboscript.perception.overlay import (
    attribute_card,
    place_labels,
    som_overlay,
)
from roboscript.perception.proposals import propose
from roboscript.perception.views import frame_geometry
from roboscript.testing.scenes import (
    HEAD,
    PERCEPTION,
    TABLE_Z,
    Box,
    Cylinder,
    capture,
    look_at,
    pinhole,
)

SHAPES = [
    Box((-0.15, 0.05, TABLE_Z + 0.025), (0.05, 0.05, 0.05), yaw_deg=20, rgb=(200, 30, 30)),
    Box((0.1, -0.05, TABLE_Z + 0.02), (0.12, 0.04, 0.04), yaw_deg=-35, rgb=(40, 170, 60)),
    Cylinder((0.0, 0.15, TABLE_Z + 0.06), radius=0.03, length=0.12, rgb=(40, 60, 200)),
    Cylinder(
        (-0.2, -0.12, TABLE_Z + 0.025), 0.025, 0.15, yaw_deg=30, lying=True, rgb=(220, 210, 40)
    ),
    Box((0.25, 0.1, TABLE_Z + 0.0005), (0.12, 0.12, 0.001), yaw_deg=10, rgb=(230, 130, 180)),
    Box((-0.3, 0.2, TABLE_Z + 0.0005), (0.1, 0.1, 0.001), rgb=(128, 128, 128)),  # a gray mat
]
HEAD_CAMERA = pinhole("overhead", HEAD)
SIDE = pinhole("wrist_left", look_at([-0.45, 0.0, 0.95], [0.0, 0.05, TABLE_Z]), 320, 240)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


@pytest.fixture(scope="module")
def world():  # type: ignore[no-untyped-def]
    images, index = capture(SHAPES, [HEAD_CAMERA, SIDE])
    return frame_geometry(FrameId("f"), 0, images, PERCEPTION), index


def test_proposals_cover_every_object_a_sensor_could_see(world) -> None:  # type: ignore[no-untyped-def]
    geometry, index = world
    proposals = propose(geometry, PERCEPTION)
    assert [p.id for p in proposals] == [1, 2, 3, 4, 5, 6]
    assert [p.kind for p in proposals] == ["solid"] * 4 + ["flat"] * 2
    assert all(p.camera == "overhead" for p in proposals)
    head = index["overhead"]
    for shape in range(6):  # the gray mat by its distance from the table's color
        best = max(iou(p.masks["overhead"], head == shape) for p in proposals)
        assert best > 0.9, (shape, best)
    sizes = [len(p.points) for p in proposals[:4]]
    assert sizes == sorted(sizes, reverse=True)
    for p in proposals:
        assert p.summary.n == len(p.points)
        assert set(p.masks) == {"overhead", "wrist_left"}
    side = index["wrist_left"]
    seen = [s for s in range(4) if (side == s).sum() > 500]
    assert seen
    for shape in seen:  # masks in the other view: the pixels near each proposal's points
        best = max(iou(p.masks["wrist_left"], side == shape) for p in proposals)
        assert best > 0.8, (shape, best)


def test_set_of_mark_overlay_and_cards(world) -> None:  # type: ignore[no-untyped-def]
    geometry, _ = world
    proposals = propose(geometry, PERCEPTION)
    rgb = geometry.view("overhead").image.rgb
    marks = {p.id: p.masks["overhead"] for p in proposals}
    image, anchors = som_overlay(rgb, marks)
    assert image.shape == rgb.shape and image.dtype == np.uint8
    assert set(anchors) == set(marks)
    for number, (u, v) in anchors.items():
        assert marks[number][v, u]  # deep inside its mask
    changed = (image != rgb).any(axis=2)
    assert changed.sum() < 0.1 * changed.size
    placed = place_labels(marks, (26, 24), rgb.shape[:2])
    boxes = [(u - 13, v - 12, u + 13, v + 12) for (u, v), _ in placed.values()]
    for number, ((u, v), target) in placed.items():
        assert target is None and marks[number][v, u]  # room on every object here
    for i, (a0, b0, a1, b1) in enumerate(boxes):  # labels never overlap
        for c0, d0, c1, d1 in boxes[i + 1 :]:
            assert a1 <= c0 or c1 <= a0 or b1 <= d0 or d1 <= b0
    card = attribute_card(rgb, proposals, geometry.table).splitlines()
    assert len(card) == 6 and all(line.startswith(f"mark {i}: ") for i, line in enumerate(card, 1))
    colors = [line.split(": ")[1].split(",")[0] for line in card]
    assert sorted(colors) == ["blue", "gray", "green", "pink", "red", "yellow"]
    red = card[colors.index("red")]
    assert "5.0 x 5.0 cm footprint, 5.0 cm tall" in red and "table position left/" in red


def test_snap_2d_answers_to_proposals(world) -> None:  # type: ignore[no-untyped-def]
    geometry, index = world
    proposals = propose(geometry, PERCEPTION)
    red = next(p for p in proposals if iou(p.masks["overhead"], index["overhead"] == 0) > 0.9)
    green = next(p for p in proposals if iou(p.masks["overhead"], index["overhead"] == 1) > 0.9)
    v, u = np.argwhere(red.masks["overhead"]).mean(axis=0)
    assert snap(proposals, "overhead", point=(u, v)) is red
    vs, us = np.nonzero(red.masks["overhead"])
    left = int(us.min())
    row = int(vs[us == left][0])
    assert snap(proposals, "overhead", point=(left - 4, row)) is red  # within 6 px
    assert snap(proposals, "overhead", point=(2, 2)) is None
    vs, us = np.nonzero(green.masks["overhead"])
    box = (us.min() - 5, vs.min() - 5, us.max() + 5, vs.max() + 5)
    assert snap(proposals, "overhead", box=box) is green
    assert snap(proposals, "overhead", box=(0, 0, 3, 3)) is None
    with pytest.raises(ValueError, match="a point or a box"):
        snap(proposals, "overhead")
