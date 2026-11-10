"""Spec fitting round trips: spec -> pose -> fit -> the same spec (written the canonical way), and
what a spec cannot express comes back as the residual."""

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
    fit_grasp,
    fit_place,
    grasp_candidates,
    placement_point_pose,
    target_frame,
    target_pose,
)
from roboscript.manipulation.fit import THIN_M
from roboscript.manipulation.posemath import axangle

BASE = Pose(p=(0.0, -0.6, 0.0), q=(0.7071068, 0.0, 0.0, 0.7071068))
POSES = load_embodiment(MOCK_BIMANUAL).poses  # the pose conventions
TO_GRASP = np.array([[0.0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def rigid(rotation: np.ndarray, position: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, position
    return m


def proxy(
    size: tuple[float, float, float], symmetry: str, x_known: bool = False, yaw: float = 0.3
) -> ObjectProxy:
    whole = Part(frame=np.eye(4), size=np.array(size), x_known=x_known)
    frame = rigid(axangle([0, 0, 1], yaw), (0.12, -0.05, 0.78))
    return ObjectProxy("thing", frame, symmetry, {"whole": whole})  # type: ignore[arg-type]


def tcp(box: ObjectProxy, contact: np.ndarray, tilt: float = 0.0) -> Pose:
    """The TCP of a grasp at a candidate (object frame), tilted about its closing axis."""
    world = box.pose @ contact @ TO_GRASP
    world[:3, :3] = world[:3, :3] @ axangle([0, 1, 0], tilt)
    return Pose.from_matrix(world)


BOXES = [
    proxy((0.12, 0.05, 0.04), "pi"),
    proxy((0.05, 0.05, 0.05), "half_pi"),
    proxy((0.07, 0.07, 0.1), "continuous"),
    proxy((0.12, 0.05, 0.04), "none", x_known=True),
]
SPECS = [
    GraspSpec(
        approach="top",
        anchor="top_center",
        offset_m=(0.01, -0.005, 0.0),
        closing="major",
        yaw_deg=10,
    ),
    GraspSpec(
        approach="side",
        anchor="side_center",
        offset_m=(0.0, 0.005, 0.0),
        closing="minor",
        yaw_deg=-5,
    ),
    GraspSpec(approach="top", anchor="center", closing="minor"),
    GraspSpec(
        approach="side", anchor="center", offset_m=(0.01, 0.0, 0.0), closing="major", yaw_deg=40
    ),
    GraspSpec(approach="side", anchor="bottom_center", offset_m=(0.0, -0.002, 0.004)),
]


def canonical(spec: GraspSpec, size: np.ndarray) -> GraspSpec:
    """A spec as the fit writes it: its offset in meters along the approach, in fractions of the
    part's extents across it."""
    depth = 2 if spec.approach == "top" else 1
    total = np.asarray(spec.offset_m) + np.asarray(spec.offset_frac) * size
    across = [k != depth and size[k] >= THIN_M for k in range(3)]
    metric = tuple(0.0 if a else float(v) for a, v in zip(across, total, strict=True))
    frac = tuple(float(v / n) if a else 0.0 for a, v, n in zip(across, total, size, strict=True))
    return spec.model_copy(update={"offset_m": metric, "offset_frac": frac})


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("box", BOXES)
def test_a_grasp_comes_back_as_its_spec(spec: GraspSpec, box: ObjectProxy) -> None:
    for candidate in grasp_candidates(spec, box, "right"):
        fit = fit_grasp(tcp(box, candidate.matrix), box, arm="right", purpose="pull")
        assert fit.position_error_m == pytest.approx(0.0, abs=1e-9)
        assert fit.angle_error_deg == pytest.approx(0.0, abs=1e-5)
        again = {c.label: c.matrix for c in grasp_candidates(fit.spec, box, "right")}
        assert again[fit.variant] == pytest.approx(candidate.matrix)
        # A symmetric turn may be written another way; on a square footprint a top grasp
        # closing along x is written as minor in the frame turned a quarter.
        square = box.whole.size[0] == box.whole.size[1] and spec.approach == "top"
        if candidate.label.endswith("/0") and not square:
            written = spec.model_copy(update={"arm": "right", "purpose": "pull"})
            expected = canonical(written, box.whole.size)
            numbers = {"offset_m": expected.offset_m, "offset_frac": expected.offset_frac}
            numbers |= {"yaw_deg": expected.yaw_deg, "tilt_deg": 0, "lean_deg": 0}
            assert fit.spec.model_copy(update=numbers) == expected
            assert fit.spec.offset_m == pytest.approx(expected.offset_m, abs=1e-9)
            assert fit.spec.offset_frac == pytest.approx(expected.offset_frac, abs=1e-9)
            assert fit.spec.yaw_deg == pytest.approx(spec.yaw_deg, abs=1e-7)
            assert (fit.spec.tilt_deg, fit.spec.lean_deg) == pytest.approx((0, 0), abs=1e-7)


def placed_at(spec: PlaceSpec, held: ObjectProxy, target: ObjectProxy | None) -> np.ndarray:
    """The held object's frame once its placement point is on the spec's target pose."""
    point = placement_point_pose(held, spec.placement_point, conventions=POSES)
    frame = target_frame(spec, target, BASE)
    goal = target_pose(spec, frame, held.pose @ point, placement_point=point).matrix()
    result: np.ndarray = goal @ np.linalg.inv(point)
    return result


def test_a_placement_comes_back_in_both_frames() -> None:
    cube = proxy((0.05, 0.05, 0.05), "half_pi")
    plate = proxy((0.2, 0.12, 0.02), "pi", yaw=1.2)
    spec = PlaceSpec(
        target=PlaceTarget(kind="object", obj="thing", anchor="top_center"),
        offset_m=(0.01, 0.02, 0.0),
        orient="yaw",
        yaw_deg=30,
    )
    placed = placed_at(spec, cube, plate)
    fits = fit_place(placed, cube, plate, BASE, conventions=POSES)
    assert set(fits) == {"target", "world_axes", "robot_frame"}
    # on the plate, turned 30 deg from it and from the world: a placement relative to the plate
    assert [name for name, fit in fits.items() if fit.consistent] == ["target", "world_axes"]
    fitted = fits["target"].spec
    assert fitted.model_copy(update={"offset_m": spec.offset_m, "yaw_deg": 30}) == spec
    assert fitted.offset_m == pytest.approx(spec.offset_m, abs=1e-9)
    assert fitted.yaw_deg == pytest.approx(30, abs=1e-7)
    for fit in fits.values():
        assert (fit.position_error_m, fit.angle_error_deg) == pytest.approx((0.0, 0.0), abs=1e-6)
        assert placed_at(fit.spec, cube, plate) == pytest.approx(placed)
    world = fits["world_axes"].spec
    offset = plate.pose[:3, :3] @ np.array(spec.offset_m)
    assert world.frame == "world_axes" and world.offset_m == pytest.approx(offset, abs=1e-9)
    plain = fit_place(placed, cube, plate, BASE, plain=True, conventions=POSES)
    assert list(plain) == ["target", "world_axes", "plain", "robot_frame"]
    assert plain["plain"].spec.target.anchor == "inside_bottom"
    assert plain["plain"].spec.frame == "world_axes"
    assert plain["plain"].spec.offset_m == pytest.approx((0.0, 0.0, 0.0))
