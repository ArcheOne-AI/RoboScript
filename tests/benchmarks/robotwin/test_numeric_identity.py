"""Bit-identical numbers from the RoboTwin bindings (docs/how-to/add-a-skillset.md).

The bindings hold constants and functions for the core (pose conventions, the planner's table
world and turn, the executor's tuning), and every number they give stays the same to the last
bit. This computes a fixed set of grasp and place poses, synthesis results, planner setups and
executor parameters and compares them, as `float.hex`, with `numeric_golden.json`.

Only `compute()` follows the code when it moves (the RoboTwin bindings, its embodiment profile's
pose conventions, `roboscript_robotwin.bindings`); the golden file never changes. To capture it
on a tree that has no golden yet:
`uv run python tests/benchmarks/robotwin/test_numeric_identity.py --capture`.
"""

import asyncio
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest

from roboscript.contracts.executor import with_overrides
from roboscript.contracts.geometry import Pose
from roboscript.embodiments import GraspPreference
from roboscript.manipulation import (
    GraspSpec,
    HingeSpec,
    ObjectProxy,
    Part,
    PlaceSpec,
    PlaceTarget,
    grasp_candidates,
    hinge_line,
    placement_point_pose,
    target_frame,
    target_pose,
)
from roboscript.manipulation.posemath import (
    CONTACT,
    PLACEMENT,
    arc_poses,
    axangle,
    choose_grasp,
    mat2quat,
    place_ee_pose,
    placed_object_pose,
    tilted_poses,
)
from roboscript.motion.config import ArmConfig
from roboscript.motion.curobo import planner_T_root, table_world
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.bindings import EXECUTOR
from roboscript_robotwin.bindings import PLANNER_PROFILE as PLANNER

pytestmark = pytest.mark.unit

GOLDEN: Final = Path(__file__).resolve().parent / "numeric_golden.json"
BASES: Final = {
    "left": Pose(p=(0.0, -0.65, 0.0), q=(0.707, 0.0, 0.0, 0.707)),  # aloha_agilex's planner base
    "right": Pose(p=(0.0, -0.65, 0.0), q=(0.707, 0.0, 0.0, 0.707)),
    "turned": Pose(p=(0.1, -0.6, 0.02), q=(0.9238795, 0.0, 0.0, 0.3826834)),
}
FRAME_BIASES: Final = ([0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.03, -0.01, 0.12])
PREFERENCES: Final = {
    "left": GraspPreference(
        tilt=(0.0, 1.0),
        top_down=(-0.61239, 0.353523, -0.61239, -0.353524),
        side=(-0.924, 0.0, 0.0, -0.383),
    ),
    "right": GraspPreference(
        tilt=(0.0, 1.0),
        top_down=(-0.353523, -0.61239, -0.353524, 0.61239),
        side=(-0.383, 0.0, 0.0, -0.924),
    ),
}
TCP: Final = 0.12
POSES: Final = ROBOTWIN.profile().poses  # RoboTwin's pose conventions, bound in its profile


def _rigid(rotation: Any, position: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, position
    return m


def _box(
    size: Sequence[float],
    symmetry: str = "pi",
    yaw: float = 0.0,
    at: Sequence[float] = (0.1, 0.0, 0.8),
    parts: dict[str, Part] | None = None,
    x_known: bool = False,
) -> ObjectProxy:
    whole = Part(frame=np.eye(4), size=np.array(size, dtype=float), x_known=x_known)
    frame = _rigid(axangle([0, 0, 1], yaw), at)
    return ObjectProxy("box", frame, symmetry, {"whole": whole, **(parts or {})})  # type: ignore[arg-type]


def _contacts(half: Sequence[float]) -> list[np.ndarray]:
    """RoboTwin's `create_box` default contact points, scaled, and two tilted ones."""
    raw = [
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
        [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
    ]
    out = []
    for m in raw:
        a = np.array(m, dtype=float)
        a[:3, 3] *= np.asarray(half, dtype=float)
        out.append(a)
    out.append(_rigid(axangle([1, 1, 0], 0.4) @ out[0][:3, :3], (0.01, -0.02, 0.03)))
    out.append(_rigid(axangle([0, 1, 1], -1.1) @ out[2][:3, :3], (-0.015, 0.005, -0.01)))
    return out


def _plannable(reject: int) -> Callable[[list[Pose]], Any]:
    """Refuses the first `reject` poses of every batch."""

    async def plannable(poses: list[Pose]) -> list[bool]:
        return [i >= reject for i in range(len(poses))]

    return plannable


def _floats(value: Any) -> Any:
    """Nested lists of `float.hex` strings (ints and bools kept, strings kept)."""
    if isinstance(value, Pose):
        return _floats([*value.p, *value.q])
    if isinstance(value, np.ndarray):
        return _floats(value.tolist())
    if isinstance(value, dict):
        return {str(k): _floats(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_floats(v) for v in value]
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float | np.floating):
        return float(value).hex()
    raise TypeError(f"cannot serialize {type(value).__name__}")


async def _grasps() -> dict[str, Any]:
    out: dict[str, Any] = {}
    cube = _box((0.05, 0.05, 0.05), "half_pi", yaw=0.3).with_frames(
        {CONTACT: _contacts((0.025, 0.025, 0.025))}
    )
    origin = Pose(p=(0.1, -0.1, 0.95), q=tuple(mat2quat(axangle([1, 0, 0], 2.5)).tolist()))  # type: ignore[arg-type]
    out["tilted_poses"] = tilted_poses(origin, [0.1, 0.0, 0.8], (0.0, 1.0), POSES)
    out["tilted_poses_neg"] = tilted_poses(origin, [0.12, 0.01, 0.82], (-0.5, 0.7), POSES)
    for arm in ("left", "right"):
        for reject in (0, 3, 9):
            for approach_m, grasp_offset_m in ((0.1, 0.0), (0.07, 0.02)):
                choice = await choose_grasp(
                    cube,
                    arm,
                    approach_m,
                    grasp_offset_m,
                    None,
                    PREFERENCES[arm],
                    TCP,
                    _plannable(reject),
                    POSES,
                )
                key = f"choose_grasp/{arm}/{reject}/{approach_m}/{grasp_offset_m}"
                out[key] = None if choice is None else [choice.pre, choice.grasp, choice.frame]
        listed = await choose_grasp(
            cube, arm, 0.1, 0.0, [1, 4], PREFERENCES[arm], TCP, _plannable(0), POSES
        )
        out[f"choose_grasp/{arm}/listed"] = (
            None if listed is None else [listed.pre, listed.grasp, listed.frame]
        )
    return out


def _places() -> dict[str, Any]:
    out: dict[str, Any] = {}
    held = _box((0.06, 0.04, 0.1), "none", yaw=0.7, at=(0.05, -0.1, 0.9)).with_frames(
        {PLACEMENT: [_rigid(np.diag([1.0, -1, -1]), (0.0, 0.0, -0.05))]}
    )
    targets = [
        Pose(p=(0.2, 0.05, 0.76), q=(1.0, 0.0, 0.0, 0.0)),
        Pose(p=(-0.15, 0.1, 0.8), q=tuple(mat2quat(axangle([0, 0, 1], 1.2)).tolist())),  # type: ignore[arg-type]
        Pose(p=(0.0, 0.2, 0.9), q=tuple(mat2quat(axangle([1, 0.2, 0], 0.5)).tolist())),  # type: ignore[arg-type]
    ]
    ees = [
        Pose(p=(0.05, -0.1, 1.02), q=tuple(mat2quat(axangle([1, 0, 0], 2.8)).tolist())),  # type: ignore[arg-type]
        Pose(p=(0.12, -0.08, 0.91), q=tuple(mat2quat(axangle([0, 1, 0], 1.4)).tolist())),  # type: ignore[arg-type]
    ]
    for i, target in enumerate(targets):
        for j, ee in enumerate(ees):
            for arm in ("left", "right"):
                for constrain in ("auto", "free", "align"):
                    for fp in (None, 0):
                        for axis in ("grasp", "target_z", (0.0, 1.0, 1.0)):
                            key = f"place_ee_pose/{i}/{j}/{arm}/{constrain}/{fp}/{axis}"
                            out[key] = place_ee_pose(
                                held,
                                arm,
                                target,
                                ee,
                                POSES,
                                constrain=constrain,  # type: ignore[arg-type]
                                align_axis=None if constrain != "align" else [[0, 1, 0], [1, 0, 0]],
                                placement_frame=fp,
                                approach_m=0.08,
                                approach_axis=axis,  # type: ignore[arg-type]
                            )
        for z in (True, False):
            out[f"placed_object_pose/{i}/{z}"] = placed_object_pose(
                held.get_pose(), target, "align", None, (1.0, 0.0, 0.0), "object", z, POSES
            )
    return out


def _arcs() -> dict[str, Any]:
    start = Pose(p=(0.1, -0.05, 0.95), q=tuple(mat2quat(axangle([1, 0, 0], 2.9)).tolist()))  # type: ignore[arg-type]
    return {
        f"arc_poses/{angle}/{steps}": arc_poses(
            ([0.15, 0.1, 0.8], [0.1, 0.0, 1.0]), start, angle, steps
        )
        for angle, steps in ((90.0, 9), (-45.0, 4), (30.0, 1))
    }


def _synthesis() -> dict[str, Any]:
    out: dict[str, Any] = {}
    lid = Part(
        frame=_rigid(np.eye(3), (0.0, 0.0, 0.03)), size=np.array([0.08, 0.06, 0.01]), x_known=True
    )
    boxes = {
        "cube": _box((0.05, 0.05, 0.05), "half_pi", yaw=0.2),
        "long": _box((0.06, 0.04, 0.2), "pi", yaw=-0.4, at=(-0.1, 0.05, 0.85)),
        "lidded": _box((0.1, 0.08, 0.06), "none", yaw=1.1, parts={"lid": lid}),
    }
    specs = {
        "top_major": GraspSpec(approach="top", closing="major"),
        "top_minor_yaw": GraspSpec(approach="top", closing="minor", yaw_deg=20.0),
        "side": GraspSpec(approach="side", closing="minor", offset_m=(0.0, 0.0, 0.02)),
        "side_tilt": GraspSpec(approach="side", closing="major", tilt_deg=15.0, lean_deg=-10.0),
    }
    for name, box in boxes.items():
        for spec_name, spec in specs.items():
            for arm in ("left", "right"):
                found = grasp_candidates(spec, box, arm)  # type: ignore[arg-type]
                out[f"grasp_candidates/{name}/{spec_name}/{arm}"] = [
                    [c.label, c.matrix] for c in found
                ]
        for f in ("bottom", "top", "center"):
            out[f"placement_point/{name}/{f}"] = placement_point_pose(box, f, conventions=POSES)
    out["placement_point/lid"] = placement_point_pose(
        boxes["lidded"], "part:lid", conventions=POSES
    )
    plate = boxes["long"]
    held = _rigid(axangle([0, 0, 1], 0.3) @ np.diag([1.0, -1, -1]), (0.0, 0.0, 1.0))
    on = PlaceTarget(kind="object", obj="box", anchor="top_center")
    for orient, extra in (
        ("yaw", {"yaw_deg": 90.0}),
        ("upright", {}),
        ("keep", {}),
        ("relative", {"rpy_deg": (10.0, -5.0, 30.0)}),
    ):
        spec = PlaceSpec(target=on, orient=orient, offset_m=(0.02, -0.01, 0.03), **extra)  # type: ignore[arg-type]
        frame = target_frame(spec, plate, BASES["left"])
        out[f"target_frame/{orient}"] = frame
        for symmetry in ("none", "pi", "half_pi"):
            out[f"target_pose/{orient}/{symmetry}"] = target_pose(spec, frame, held, symmetry)  # type: ignore[arg-type]
    fixed = PlaceSpec(target=PlaceTarget(kind="robot_frame"), offset_m=(0.52, 0.0, 0.75))
    out["target_frame/robot_frame"] = target_frame(fixed, None, BASES["turned"])
    for side, at in (("front", "bottom"), ("left", "upright"), ("back", "top")):
        spec = HingeSpec(part="lid", side=side, at=at)  # type: ignore[arg-type]
        out[f"hinge_line/{side}/{at}"] = list(hinge_line(spec, boxes["lidded"]))
    return out


def _planner() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, base in BASES.items():
        out[f"table_world/{name}"] = table_world(base, PLANNER.table)["cuboid"]["table"]
    for arm, yaw in PLANNER.yaw_rad.items():
        for i, bias in enumerate(FRAME_BIASES):
            out[f"planner_T_root/{arm}/{i}"] = planner_T_root(bias, yaw)
        out[f"arm_config/{arm}"] = ArmConfig(yaw_rad=yaw).model_dump(mode="json")
    return out


def _executor() -> dict[str, Any]:
    # fields whose default leaves the bindings' behaviour alone (`max_action_joint_step_rad` None)
    # are left out, as stored hashes leave them
    return {
        "executor": with_overrides(EXECUTOR, {}).model_dump(mode="json", exclude_defaults=True),
        "tcp_closing_axis": list(POSES.tcp_closing_axis),
    }


def compute() -> dict[str, Any]:
    values: dict[str, Any] = {}
    values.update(asyncio.run(_grasps()))
    values.update(_places())
    values.update(_arcs())
    values.update(_synthesis())
    values.update(_planner())
    values.update(_executor())
    return _floats(values)  # type: ignore[no-any-return]


def _same(value: Any) -> Any:
    """`value` with ints as floats (`float.hex`): a binding's data is typed, so a constant written
    `2` in code is `2.0`; every float is still compared bit for bit."""
    if isinstance(value, dict):
        return {k: _same(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_same(v) for v in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value).hex()
    return value


def test_the_numbers_are_bit_identical() -> None:
    golden = _same(json.loads(GOLDEN.read_text(encoding="utf-8")))
    now = _same(compute())
    assert sorted(now) == sorted(golden), "the computed set changed: compute() must not change"
    changed = [key for key in golden if golden[key] != now[key]]
    assert not changed, f"{len(changed)} values differ, e.g. {changed[:5]}"


if __name__ == "__main__":
    if sys.argv[1:] != ["--capture"]:
        raise SystemExit("usage: test_numeric_identity.py --capture")
    if GOLDEN.exists():
        raise SystemExit(f"{GOLDEN} exists: the golden is captured once, before the split")
    text = json.dumps(compute(), indent=1, sort_keys=True) + "\n"
    GOLDEN.write_text(text, encoding="utf-8")
    print(f"wrote {GOLDEN} ({math.ceil(len(text) / 1024)} KiB)")
