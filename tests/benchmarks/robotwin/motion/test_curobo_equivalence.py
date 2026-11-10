"""The cuRobo planner against RoboTwin's own planner and robot model, on a GPU.

RoboTwin's code is used read-only: its `Robot` is built as its tasks build it (`load_robot`), and
its `left_plan_path` / `left_plan_multi_path` are the references. Run in the simulator env:
`python -m pytest -m sim tests/benchmarks/robotwin/motion --keep-passed`; each test writes its rows
and summary as JSON to its artifacts directory. Assertions count outliers rather than bound rates,
and timings are only logged: the GPU is shared, and cuRobo is not bit-reproducible.
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import (
    BatchPlanRequest,
    Constraint,
    FKRequest,
    IKRequest,
    PlanRequest,
)
from roboscript.contracts.robot import Arm
from roboscript.motion import CuroboPlanner
from roboscript.spatial.pose import pose_compose
from roboscript.spatial.rotation import quat_angle_deg
from roboscript.testing.planner import PlannerConformance, PlannerTarget

pytestmark = [pytest.mark.sim, pytest.mark.gpu]

BASE = Pose(p=(0.0, -0.65, 0.0), q=(0.707, 0.0, 0.0, 0.707))  # config.yml robot_pose
ARMS: tuple[Arm, Arm] = ("left", "right")
# The range of each arm joint in expert demonstrations, 1st to 99th percentile.
BOX = {
    "left": (
        [-0.955, 0.0, 0.0, -1.61, -0.387, -1.946],
        [0.173, 2.576, 2.461, 1.325, 1.274, 1.473],
    ),
    "right": (
        [-0.132, 0.0, 0.0, -1.701, -1.017, -1.554],
        [1.066, 2.623, 2.457, 1.323, 1.048, 1.858],
    ),
}
LOCK = [1, 1, 1, 0, 0, 0]  # RoboTwin's constraint_pose for the approach of a grasp
FINGERS = {"left": ["fl_joint7", "fl_joint8"], "right": ["fr_joint7", "fr_joint8"]}
# Two poses per arm with room around them, for the conformance check.
SAMPLES = {
    "left": [[-0.3, 1.2, 1.1, 0.2, 0.3, -0.2], [-0.5, 1.6, 1.4, -0.3, 0.5, 0.4]],
    "right": [[0.3, 1.2, 1.1, -0.2, -0.3, 0.2], [0.5, 1.6, 1.4, 0.3, -0.5, -0.4]],
}


@dataclass
class Reference:
    robot: Any  # envs.robot.robot.Robot
    scene: Any  # the SAPIEN scene holding the robot, kept alive with it
    joint_names: list[str]

    def full(self, arm: Arm, joints: list[float]) -> np.ndarray:
        """The whole robot's active-joint vector, as `entity.get_qpos()` gives it (float32)."""
        names = getattr(self.robot, f"{arm}_arm_joints_name")
        q = np.zeros(len(self.joint_names), dtype=np.float32)
        q[[self.joint_names.index(n) for n in names]] = joints
        return q


@pytest.fixture(scope="module")
def reference(robotwin_root: Path) -> Iterator[Reference]:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(robotwin_root))  # RoboTwin's `envs` package
        with pytest.MonkeyPatch.context() as cwd:
            cwd.chdir(robotwin_root)  # `envs` reads asset lists relative to it on import
            import sapien.core as sapien
            from envs.robot.robot import Robot

            embodiment = f"{robotwin_root}/assets/embodiments/aloha-agilex/"
            config = yaml.safe_load(Path(embodiment, "config.yml").read_text(encoding="utf-8"))
            scene = sapien.Scene()
            robot = Robot(
                scene,
                need_topp=False,
                left_embodiment_config=config,
                right_embodiment_config=config,
                left_robot_file=embodiment,
                right_robot_file=embodiment,
                dual_arm_embodied=True,
            )
            robot.set_planner()  # as Base_Task.load_robot: planners, then joints
            robot.init_joints()
        names = [joint.get_name() for joint in robot.left_entity.get_active_joints()]
        yield Reference(robot, scene, names)


async def test_plans_match_robotwins_planner(
    planner: CuroboPlanner, reference: Reference, artifacts_dir: Path
) -> None:
    rows = []
    for arm in ARMS:
        qs = await _samples(planner, arm, 71, seed=ARMS.index(arm))
        ees = (await planner.fk(FKRequest(arm=arm, base=BASE, joints=qs))).ee
        cases: list[tuple[str, list[float], Pose, Constraint]] = [
            ("free", qs[i], ees[i + 1], "free") for i in range(50)
        ]
        cases += [  # the second phase of a grasp: 10 cm along the ee's x axis, orientation locked
            ("approach", qs[i], _ahead(ees[i], 0.10), "orientation") for i in range(50, 71)
        ]
        for kind, start, goal, constraint in cases:
            rows.append(await _compare(planner, reference, arm, kind, start, goal, constraint))
        rows += await _compare_batches(planner, reference, arm, qs)
    summary = _summarize(rows)
    _write(artifacts_dir / "plans.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))

    for group, stats in summary.items():
        assert stats["n"] - stats["agreeing"] <= 2, (group, stats)
        if "far_ends" in stats:  # both planned: ends more than 1 mm or 0.5 deg apart
            assert stats["far_ends"] <= 2, (group, stats)
            assert 0.95 <= stats["length_ratio_median"] <= 1.05, (group, stats)


async def test_fk_matches_robotwins_robot_model(
    planner: CuroboPlanner, reference: Reference, artifacts_dir: Path
) -> None:
    robot = reference.robot
    info = await planner.info()
    report = {}
    for arm in ARMS:
        rng = np.random.default_rng(10 + ARMS.index(arm))
        qs = rng.uniform(*BOX[arm], size=(200, 6)).tolist()
        links = info.arms[arm].links
        ours = await planner.fk(FKRequest(arm=arm, base=BASE, joints=qs, links=links))
        entity = getattr(robot, f"{arm}_entity")
        fingers = [reference.joint_names.index(n) for n in FINGERS[arm]]
        errors: dict[str, list[tuple[float, float]]] = {"ee": [], "tcp": [], "links": []}
        for i, q in enumerate(qs):
            full = reference.full(arm, q)
            full[fingers] = 0.04  # the opening the curobo ymls lock the fingers at
            entity.set_qpos(full)
            ee = Pose.from_list(getattr(robot, f"get_{arm}_ee_pose")())
            tcp = Pose.from_list(getattr(robot, f"get_{arm}_tcp_pose")())
            errors["ee"].append(_distance(ours.ee[i], ee))
            errors["tcp"].append(_distance(ours.tcp[i], tcp))
            for name in links:
                pose = entity.find_link_by_name(name).entity_pose
                truth = Pose(p=tuple(pose.p), q=tuple(pose.q))
                errors["links"].append(_distance(ours.links[name][i], truth))
        report[arm] = {
            key: {"max_mm": max(e[0] for e in v), "max_deg": max(e[1] for e in v), "n": len(v)}
            for key, v in errors.items()
        }
    _write(artifacts_dir / "fk.json", report)
    print(json.dumps(report, indent=1))
    for arm, parts in report.items():
        for part, stats in parts.items():
            assert stats["max_mm"] <= 1.0 and stats["max_deg"] <= 0.1, (arm, part, stats)


async def test_ik_round_trips(planner: CuroboPlanner, artifacts_dir: Path) -> None:
    report = {}
    for arm in ARMS:
        rng = np.random.default_rng(20 + ARMS.index(arm))
        qs = await _samples(planner, arm, 50, seed=30 + ARMS.index(arm))
        goals = (await planner.fk(FKRequest(arm=arm, base=BASE, joints=qs))).ee
        rows = []
        for q, goal in zip(qs, goals, strict=True):
            seed = (np.asarray(q) + rng.normal(0, 0.2, 6)).tolist()
            started = time.perf_counter()
            [solution] = await planner.ik(IKRequest(arm=arm, base=BASE, goals=[goal], seed=seed))
            elapsed = time.perf_counter() - started
            # A sample may collide with the robot itself; then its pose has no valid solution.
            stay = await planner.plan(PlanRequest(arm=arm, base=BASE, start=q, goal=goal))
            row: dict[str, Any] = {
                "success": solution.success,
                "valid": stay.reason != "invalid_start",
                "time_s": elapsed,
            }
            if solution.joints is not None:
                back = await planner.fk(FKRequest(arm=arm, base=BASE, joints=[solution.joints]))
                row["mm"], row["deg"] = _distance(back.ee[0], goal)
                row["reported_mm"] = 1000 * (solution.position_error_m or 0.0)
                row["same_branch"] = bool(np.abs(np.subtract(solution.joints, q)).max() < 0.05)
            rows.append(row)
        solved = [r for r in rows if r["success"]]
        valid = [r for r in rows if r["valid"]]
        report[arm] = {
            "success_rate": len(solved) / len(rows),
            "failures_valid": sum(not r["success"] for r in valid),
            "invalid_samples": len(rows) - len(valid),
            "max_mm": max(r["mm"] for r in solved),
            "max_deg": max(r["deg"] for r in solved),
            "same_branch_rate": sum(r["same_branch"] for r in solved) / len(solved),
            "time_median_s": float(np.median([r["time_s"] for r in rows])),
        }
    _write(artifacts_dir / "ik.json", report)
    print(json.dumps(report, indent=1))
    for arm, stats in report.items():
        assert stats["failures_valid"] <= 1, (arm, stats)
        assert stats["max_mm"] <= 1.0 and stats["max_deg"] <= 0.1, (arm, stats)


async def _compare(
    planner: CuroboPlanner,
    reference: Reference,
    arm: Arm,
    kind: str,
    start: list[float],
    goal: Pose,
    constraint: Constraint,
) -> dict[str, Any]:
    plan_path = getattr(reference.robot, f"{arm}_plan_path")
    lock = LOCK if constraint == "orientation" else None
    started = time.perf_counter()
    ref = plan_path(goal.as_list(), constraint_pose=lock, last_qpos=reference.full(arm, start))
    time_ref = time.perf_counter() - started
    request = PlanRequest(arm=arm, base=BASE, start=start, goal=goal, constraint=constraint)
    started = time.perf_counter()
    ours = await planner.plan(request)
    time_ours = time.perf_counter() - started
    row: dict[str, Any] = {
        "arm": arm,
        "kind": kind,
        "ref": ref["status"] == "Success",
        "ours": ours.status == "success",
        "reason": ours.reason,
        "time_ref_s": time_ref,
        "time_ours_s": time_ours,
    }
    if row["ref"] and row["ours"] and ours.path is not None:
        ref_path = np.asarray(ref["position"], dtype=float)
        ends = await planner.fk(FKRequest(arm=arm, base=BASE, joints=[ref_path[-1], ours.path[-1]]))
        row["final_joint_diff_rad"] = float(np.abs(ref_path[-1] - ours.path[-1]).max())
        row["final_pos_diff_mm"], row["final_rot_diff_deg"] = _distance(ends.ee[0], ends.ee[1])
        row["reach_ref_mm"] = _distance(ends.ee[0], goal)[0]
        row["reach_ours_mm"] = _distance(ends.ee[1], goal)[0]
        row["len_ref"], row["len_ours"] = len(ref_path), len(ours.path)
        if constraint == "orientation":
            for who, path in (("ref", ref_path.tolist()), ("ours", ours.path)):
                poses = (await planner.fk(FKRequest(arm=arm, base=BASE, joints=path))).ee
                row[f"line_dev_{who}_mm"] = _line_deviation(poses, poses[0], goal)
                row[f"drift_{who}_deg"] = max(float(quat_angle_deg(p.q, goal.q)) for p in poses)
    return row


async def _compare_batches(
    planner: CuroboPlanner, reference: Reference, arm: Arm, qs: list[list[float]]
) -> list[dict[str, Any]]:
    """`choose_best_pose` batches: 10 goals from one start, five times."""
    multi = getattr(reference.robot, f"{arm}_plan_multi_path")
    ees = (await planner.fk(FKRequest(arm=arm, base=BASE, joints=qs[:50]))).ee
    rows = []
    for b in range(5):
        start, goals = qs[60 + b], ees[10 * b : 10 * b + 10]
        started = time.perf_counter()
        ref = multi([g.as_list() for g in goals], last_qpos=reference.full(arm, start))
        time_ref = time.perf_counter() - started
        started = time.perf_counter()
        ours = await planner.plan_batch(
            BatchPlanRequest(arm=arm, base=BASE, start=start, goals=goals)
        )
        time_ours = time.perf_counter() - started
        for status, result in zip(ref["status"], ours, strict=True):
            rows.append(
                {
                    "arm": arm,
                    "kind": "batch",
                    "ref": status == "Success",
                    "ours": result.status == "success",
                    "reason": result.reason,
                    "time_ref_s": time_ref / 10,
                    "time_ours_s": time_ours / 10,
                }
            )
    return rows


async def _samples(planner: CuroboPlanner, arm: Arm, n: int, seed: int) -> list[list[float]]:
    """Joint vectors from the demos' joint box whose ee is at least 0.8 m high."""
    rng = np.random.default_rng(seed)
    found: list[list[float]] = []
    while len(found) < n:
        # float32 values, as drive targets come
        qs = rng.uniform(*BOX[arm], size=(64, 6)).astype(np.float32).astype(float)
        ees = (await planner.fk(FKRequest(arm=arm, base=BASE, joints=qs.tolist()))).ee
        found += [q.tolist() for q, ee in zip(qs, ees, strict=True) if ee.p[2] > 0.8]
    return found[:n]


def _ahead(pose: Pose, meters: float) -> Pose:
    return Pose.from_list(pose_compose(pose.as_list(), [meters, 0, 0, 1, 0, 0, 0]))


def _distance(a: Pose, b: Pose) -> tuple[float, float]:
    """Millimeters and degrees between two poses."""
    return 1000 * float(np.linalg.norm(np.subtract(a.p, b.p))), float(quat_angle_deg(a.q, b.q))


def _line_deviation(poses: list[Pose], start: Pose, goal: Pose) -> float:
    """The largest distance in mm of a path's ee from the segment start-goal."""
    a, b = np.asarray(start.p), np.asarray(goal.p)
    points = np.asarray([p.p for p in poses])
    t = np.clip((points - a) @ (b - a) / float((b - a) @ (b - a)), 0, 1)
    return 1000 * float(np.linalg.norm(points - (a + t[:, None] * (b - a)), axis=1).max())


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for arm in ARMS:
        for kind in ("free", "approach", "batch"):
            group = [r for r in rows if r["arm"] == arm and r["kind"] == kind]
            both = [r for r in group if "final_pos_diff_mm" in r]
            stats: dict[str, Any] = {
                "n": len(group),
                "agreeing": sum(r["ref"] == r["ours"] for r in group),
                "success_ref": sum(r["ref"] for r in group),
                "success_ours": sum(r["ours"] for r in group),
                "time_ref_median_s": float(np.median([r["time_ref_s"] for r in group])),
                "time_ours_median_s": float(np.median([r["time_ours_s"] for r in group])),
                "reasons": sorted({r["reason"] for r in group}),
            }
            if both:
                stats["far_ends"] = sum(
                    r["final_pos_diff_mm"] > 1.0 or r["final_rot_diff_deg"] > 0.5 for r in both
                )
                for key in ("final_pos_diff_mm", "final_rot_diff_deg", "final_joint_diff_rad"):
                    values = [r[key] for r in both]
                    stats[f"{key}_p95"] = float(np.percentile(values, 95))
                    stats[f"{key}_max"] = float(max(values))
                ratios = [r["len_ours"] / r["len_ref"] for r in both]
                stats["length_ratio_median"] = float(np.median(ratios))
                stats["length_ratio_range"] = [float(min(ratios)), float(max(ratios))]
                stats["reach_ours_mm_max"] = float(max(r["reach_ours_mm"] for r in both))
                for key in ("line_dev_ref_mm", "line_dev_ours_mm", "drift_ours_deg"):
                    if key in both[0]:
                        stats[f"{key}_max"] = float(max(r[key] for r in both))
                        stats[f"{key}_median"] = float(np.median([r[key] for r in both]))
            summary[f"{arm}/{kind}"] = stats
    return summary


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, default=float) + "\n", encoding="utf-8")


@pytest.fixture
def planner_target(planner: CuroboPlanner) -> PlannerTarget:
    return PlannerTarget(planner, BASE, SAMPLES, lock_drift_deg=2.0)


class TestCuroboPlannerConformance(PlannerConformance):
    """Runs after the comparisons above, which want the planner's seed state fresh."""
