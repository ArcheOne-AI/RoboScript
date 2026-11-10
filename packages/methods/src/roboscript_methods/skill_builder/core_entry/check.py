"""The checks themselves: one task's episodes, the API's calls in them, truth judging each call.

A benchmark gives a `Bench` (its profile, its executor binding, the probed workspace the pose
targets are drawn from, the back-to-origin tolerance) and a task's `DemoScene` (the episode spec
that opens it, the objects that may be grasped); `run_check` runs every group on the control
plane it is given, each in an episode of its own, the scene reset anew, and returns the report.
The API acts through an `Executor` on the episode's agent plane, the scene's other objects as its
obstacles; the control plane's truth (`gt_snapshot`) gives the objects (`gt_scene.measured`:
boxes fitted to their hulls as perception fits a point cloud) and judges every call.

The calls keep within the failures the benchmark ends an episode for (`GtSnapshot.end_rules`):
an object it fails for moving is lifted less (and placed from no higher), or not at all. Where
the benchmark still ends an episode (`JudgeResult.ended`), no action counts after it: the call's
row says why, the group stops there, and such rows are reported apart, not judged as the API's.
"""

import contextlib
import math
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from pydantic import SecretStr

from roboscript.contracts.env import (
    EndRule,
    EnvAgent,
    EnvControl,
    EpisodeOpened,
    EpisodeSpec,
    GtSnapshot,
)
from roboscript.contracts.executor import ExecutorConfig
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import TaskName
from roboscript.contracts.motion import Planner
from roboscript.contracts.receipt import ActionReceipt
from roboscript.contracts.robot import Arm, RobotState
from roboscript.demos import gt_scene
from roboscript.demos.gt_scene import obstacles
from roboscript.embodiments import EmbodimentProfile
from roboscript.manipulation.api import ActionApi, Built
from roboscript.manipulation.executor import Executor
from roboscript.manipulation.specs import GraspSpec, PlaceSpec, PlaceTarget
from roboscript.manipulation.workspace import Workspace, orientations
from roboscript.rpc import client
from roboscript.spatial.rotation import quat_angle_deg
from roboscript.toolsets.manipulation.arms import nearest_arm

ARMS: Final[tuple[Arm, Arm]] = ("left", "right")  # the two-arm naming convention of `Arm`
POSES: Final = 4
"""move_to_pose targets per arm."""
POSE_BAND_M: Final = (0.15, 0.35)
"""move_to_pose targets: TCP heights over the table top (clear of most demo-scene objects)."""
DISPLACEMENTS: Final = ((0.0, 0.0, 0.05), (0.04, 0.0, 0.0), (0.0, 0.04, 0.0), (-0.04, -0.04, -0.05))
LIFT_M: Final = 0.10
PLACE_APPROACH_M: Final = 0.1
"""A place's approach from above its target (`place_object`'s default `approach_m`)."""
RULE_MARGIN_M: Final = 0.01
"""How far within an end rule's limit a lift stays (a grasp may push its object aside first), and
how far past `Criteria.held_m` a lift must reach for truth to judge the hold."""
SEARCH_S: Final = 120.0
"""A grasp's or a placement's planning budget (`ActionApi.deadline`)."""


@dataclass(frozen=True)
class Criteria:
    """How truth judges one call, each measured on a benchmark's runs where its values leave a
    clear gap (`gaps`; the defaults judge where no gap is clear). `held_m`: the object's rise
    after the `LIFT_M` lift; `resting_m` and `placed_m`: a placed object's bottom centre from the
    target, vertically (resting, not held above or fallen over) and horizontally; `upright_deg`:
    its tilt from how it stood (None: not judged); `opened`: an open gripper's reading; `move_m`,
    `move_deg`: where an ok move's TCP must end (None: the binding's stop tolerance, what ok
    promises)."""

    held_m: float = 0.05
    resting_m: float = 0.015
    placed_m: float = 0.03
    upright_deg: float | None = 15.0
    opened: float = 0.9
    move_m: float | None = None
    move_deg: float | None = None

    @classmethod
    def load(cls, path: Path) -> "Criteria":
        """The `value` of each criterion a criteria file (`criteria.measure`'s) holds."""
        found = yaml.safe_load(path.read_text(encoding="utf-8"))["criteria"]
        return cls(**{f.name: found[f.name]["value"] for f in fields(cls) if f.name in found})


@dataclass(frozen=True)
class Bench:
    """What the check needs of a benchmark: its robot and executor binding, the probed reach the
    pose targets are drawn from (the benchmark's `reach` probe), and how close `return_home` must
    bring the arms to where they started (metres per axis, degrees: the benchmark's own value,
    given by its glue)."""

    profile: EmbodimentProfile
    executor: ExecutorConfig
    workspace: Workspace
    origin_tolerance: tuple[float, float]
    criteria: Criteria = Criteria()


@dataclass(frozen=True)
class DemoScene:
    """A task's demonstration scene: the spec every episode opens it with (the same scene each
    time), the objects that may be grasped (movable rigid bodies: no static, visual-only or
    articulated object), where the scene came from, for the report, and the objects the
    demonstration picked up (each is grasped, whatever its kind or size)."""

    spec: EpisodeSpec
    movable: frozenset[str]
    source: dict[str, Any]
    picked: tuple[str, ...] = ()


@dataclass
class Episode:
    opened: EpisodeOpened
    agent: EnvAgent
    api: ActionApi
    scene: gt_scene.Scene
    first: GtSnapshot
    start: RobotState
    ended: str | None = None  # why the benchmark ended the episode (`JudgeResult.ended`)


class Check:
    """One task's entry check: the control plane, the planner, the episodes."""

    def __init__(
        self, bench: Bench, demo: DemoScene, control: EnvControl, planner: Planner
    ) -> None:
        self.bench, self.demo, self.control, self.planner = bench, demo, control, planner
        self.task = TaskName(demo.spec.task)
        self.profile, self.config, self.criteria = bench.profile, bench.executor, bench.criteria
        self.measured: gt_scene.Scene | None = None

    @contextlib.asynccontextmanager
    async def episode(self) -> AsyncIterator[Episode]:
        """A fresh episode of the scene; its objects measured once per task (every episode
        resets the same scene: the boxes stay, the poses are this episode's)."""
        started = time.monotonic()
        opened = await self.control.open(self.demo.spec)
        try:
            agent: EnvAgent = connect(EnvAgent, opened.agent_url, opened.agent_token, timeout=600)
            measure = self.measured is None
            first = await self.control.gt_snapshot(opened.episode_id, geometry=measure)
            if self.measured is None:
                self.measured = gt_scene.measured(first)
            scene = self.measured.at(first.objects, {})
            took = time.monotonic() - started
            print(f"{time.strftime('%H:%M:%S')} {self.task} episode in {took:.0f} s", flush=True)
            executor = Executor(
                agent,
                opened.episode_id,
                self.planner,
                self.profile,
                self.config,
                table_top_z=first.table_top_z,
            )
            start = await agent.robot(opened.episode_id)
            yield Episode(opened, agent, ActionApi(executor), scene, first, start)
        finally:
            await self.control.close(opened.episode_id)

    async def run(self, ep: Episode, *built: Built) -> ActionReceipt:
        """One call through the API, the scene's objects (truth now) as the executor's obstacles."""
        robot = await ep.agent.robot(ep.opened.episode_id)
        here = await self.truth(ep)
        ep.api.executor.obstacles = obstacles(here, robot, 0.0)
        receipt = await ep.api.run(*built)
        ep.ended = ep.ended or (await self.control.judge(ep.opened.episode_id)).ended
        return receipt

    async def truth(self, ep: Episode) -> gt_scene.Scene:
        now = await self.control.gt_snapshot(ep.opened.episode_id)
        return ep.scene.at(now.objects, {})

    async def robot(self, ep: Episode) -> RobotState:
        return await ep.agent.robot(ep.opened.episode_id)

    # ---- the groups -----------------------------------------------------------------------------

    async def move_to_pose(self) -> list[dict[str, Any]]:
        rows = []
        async with self.episode() as ep:
            top = float(ep.first.table_top_z or 0.0)
            for arm in ARMS:
                for tcp in await self.sampled(ep, arm, top):
                    ee = self.profile.ee_pose(arm, tcp)
                    receipt = await self.run(ep, await ep.api.move_to_pose(arm, ee))
                    error = tcp_error((await self.robot(ep)).arms[arm].tcp, tcp)
                    row = {"arm": arm, "target_tcp": tcp.as_list(), **receipt_row(receipt), **error}
                    rows.append(marked(ep, row | {"honest": honest(receipt.ok, self.reached(row))}))
                    if ep.ended:
                        return rows
            rows.append(await self.return_home(ep))
        return rows

    def reached(self, row: dict[str, Any]) -> bool:
        """Whether a move's TCP ended where an ok receipt promises (`Criteria.move_m`, `_deg`)."""
        tolerance = move_tolerance(self.criteria, self.config)
        error = row["pos_m"] if "pos_m" in row else row["error_m"]
        return error <= tolerance[0] and row.get("deg", 0.0) <= tolerance[1]

    async def sampled(self, ep: Episode, arm: Arm, top: float) -> list[Pose]:
        """`POSES` TCP poses `arm` plans to: top-down at four yaws, at workspace cells over the
        table `POSE_BAND_M` high and clear of every object by the hand's length and the obstacle
        margin (the executor keeps clear of no object nearer a goal than that), in a fixed random
        order."""
        ws = self.bench.workspace
        index = np.unravel_index(np.asarray(ws.reachable[arm], dtype=int), ws.shape)
        centers = np.asarray(ws.low) + (np.stack(index, axis=1) + 0.5) * ws.cell_m
        low, high = top + POSE_BAND_M[0], top + POSE_BAND_M[1]
        centers = centers[(centers[:, 2] >= low) & (centers[:, 2] <= high)]
        hand = float(np.linalg.norm(self.profile.arms[arm].ee_to_tcp))
        clear = hand + self.config.obstacle_margin_m
        centers = np.array([c for c in centers if clearance(ep.scene, c) > clear])
        np.random.default_rng(0).shuffle(centers)
        down = orientations()[:4]
        plannable = await ep.api.plannable(arm)
        found: list[Pose] = []
        for center in centers[: 10 * POSES]:
            point = (float(center[0]), float(center[1]), float(center[2]))
            tcps = [Pose(p=point, q=o.q) for o in down]
            ok = await plannable([self.profile.ee_pose(arm, tcp) for tcp in tcps])
            found += [tcp for tcp, fine in zip(tcps, ok, strict=True) if fine][:1]
            if len(found) == POSES:
                break
        return found

    async def displacement(self) -> list[dict[str, Any]]:
        rows = []
        async with self.episode() as ep:
            for arms in [[arm] for arm in ARMS] + [list(ARMS)]:
                rows += await self.displace(ep, arms)
                if ep.ended:
                    return rows
            rows.append(await self.return_home(ep))
        return rows

    async def displace(self, ep: Episode, arms: list[Arm]) -> list[dict[str, Any]]:
        rows = []
        for shift in DISPLACEMENTS:
            before = await self.robot(ep)
            built = [await ep.api.move_by(arm, *shift) for arm in arms]
            receipt = await self.run(ep, *built)
            after = await self.robot(ep)
            for arm in arms:
                moved = np.subtract(after.arms[arm].tcp.p, before.arms[arm].tcp.p)
                error = float(np.linalg.norm(moved - np.asarray(shift)))
                row = {
                    "arm": arm,
                    "together": len(arms) > 1,
                    "shift": list(shift),
                    **receipt_row(receipt),
                    "error_m": round(error, 5),
                }
                rows.append(marked(ep, row | {"honest": honest(receipt.ok, self.reached(row))}))
            if ep.ended:
                break
        return rows

    async def gripper(self) -> list[dict[str, Any]]:
        rows = []
        async with self.episode() as ep:
            for arms in [[arm] for arm in ARMS] + [list(ARMS)]:
                rows += await self.grip(ep, arms)
                if ep.ended:
                    break
        return rows

    async def grip(self, ep: Episode, arms: list[Arm]) -> list[dict[str, Any]]:
        rows = []
        for close in (True, False):
            built = [ep.api.close_gripper(a) if close else ep.api.open_gripper(a) for a in arms]
            receipt = await self.run(ep, *built)
            state = await self.robot(ep)
            for arm in arms:
                spec = self.profile.arms[arm].gripper
                reading = state.arms[arm].gripper
                assert reading is not None  # the entry check needs measured fingers
                # closed on nothing reads `empty`; open reads 1
                reached = (
                    reading <= spec.holding_threshold if close else reading >= self.criteria.opened
                )
                row = {
                    "arm": arm,
                    "together": len(arms) > 1,
                    "close": close,
                    **receipt_row(receipt),
                    "reading": round(reading, 4),
                    "reached": reached,
                    "honest": honest(receipt.ok, reached),
                }
                rows.append(marked(ep, row))
            if ep.ended:
                break
        return rows

    async def grasp_and_place(self, label: str) -> dict[str, Any]:
        async with self.episode() as ep:
            proxy = ep.scene.proxy(label)
            box = proxy.pose @ proxy.whole.frame
            starts = {arm: ep.start.arms[arm].tcp.p for arm in ARMS}
            order = nearest_arm(box[:3, 3], starts)
            row: dict[str, Any] = {
                "label": label,
                "size_m": rounded(proxy.whole.size),
                "order": order,
            }
            rule = binding_rule(ep.first.end_rules, label)
            lift_m = LIFT_M if rule is None else min(LIFT_M, rule.limit_m - RULE_MARGIN_M)
            if rule is not None and lift_m < self.criteria.held_m + RULE_MARGIN_M:
                why = f"the benchmark ends the episode when it is {rule.kind} {rule.limit_m} m"
                return row | {"skipped": f"{why}: no lift truth can judge a hold by fits"}
            if rule is not None:
                row["lift_m"] = round(lift_m, 4)
            for arm in order:  # the arm rule: the nearer arm, the other where only it plans
                ep.api.deadline = time.monotonic() + SEARCH_S
                built = await ep.api.grasp_object(arm, proxy, GraspSpec())
                ep.api.deadline = None
                if built.refused is None:
                    break
                row.setdefault("refused", []).append(receipt_row(built.refused) | {"arm": arm})
            else:
                row |= {"grasped": False, "held": False, "planned": False}
                row["return_home"] = await self.return_home(ep)
                return row
            row |= {"arm": arm, "planned": True, "grasp_label": built.label}
            before = await self.truth(ep)
            grasp = await self.run(ep, built)
            lift = await self.run(ep, await ep.api.move_by(arm, z=lift_m))
            lifted = await self.truth(ep)
            rose = float(lifted.poses[label].p[2] - before.poses[label].p[2])
            held = rose >= self.criteria.held_m
            claimed = grasp.ok and lift.ok
            row |= {
                "grasp": receipt_row(grasp),
                "lift": receipt_row(lift),
                "rose_m": round(rose, 4),
                "held": held,
                "grasp_honest": honest(claimed, held),
            }
            if ep.ended:
                return marked(ep, row)
            if held:
                approach_m = (
                    PLACE_APPROACH_M if rule is None else lift_m
                )  # its approach within the rule too
                place = await self.place(ep, arm, label, box, proxy.whole.size, approach_m)
                row["place"] = marked(ep, place)
                if ep.ended:
                    return row
            row["return_home"] = await self.return_home(ep)
            return row

    async def place(
        self,
        ep: Episode,
        arm: Arm,
        label: str,
        picked: np.ndarray,
        size: np.ndarray,
        approach_m: float = PLACE_APPROACH_M,
    ) -> dict[str, Any]:
        """Put the held object back where it stood: its box's bottom centre, as picked."""
        target = picked[:3, 3] - picked[:3, 2] * size[2] / 2
        base = ep.api.base.matrix()
        offset = base[:3, :3].T @ (target - base[:3, 3])
        at = (float(offset[0]), float(offset[1]), float(offset[2]))
        spec = PlaceSpec(target=PlaceTarget(kind="robot_frame"), offset_m=at)
        held = (await self.truth(ep)).proxy(label)
        ep.api.deadline = time.monotonic() + SEARCH_S
        built = await ep.api.place_object(arm, held, spec, None, approach_m=approach_m)
        ep.api.deadline = None
        receipt = await self.run(ep, built)
        now = await self.truth(ep)
        proxy = now.proxy(label)
        box = proxy.pose @ proxy.whole.frame
        bottom = box[:3, 3] - box[:3, 2] * proxy.whole.size[2] / 2
        off = float(np.linalg.norm(bottom[:2] - target[:2]))
        dz = float(bottom[2] - target[2])
        tilt = math.degrees(math.acos(float(np.clip(box[:3, 2] @ picked[:3, 2], -1.0, 1.0))))
        reading = (await self.robot(ep)).arms[arm].gripper
        assert reading is not None
        row = {
            **receipt_row(receipt),
            "planned": built.refused is None,
            "label": built.label,
            "off_m": round(off, 4),
            "dz_m": round(dz, 4),
            "tilt_deg": round(tilt, 2),
            "gripper": round(reading, 4),
        }
        return row | place_verdict(row, self.criteria)

    async def return_home(self, ep: Episode) -> dict[str, Any]:
        built = [await ep.api.return_home(arm) for arm in ARMS]
        receipt = await self.run(ep, *built)
        state = await self.robot(ep)
        row: dict[str, Any] = {"call": "return_home", **receipt_row(receipt)}
        within = True
        axis_m, angle_deg = self.bench.origin_tolerance
        for arm in ARMS:
            now, then = state.arms[arm].ee, ep.start.arms[arm].ee
            axis = float(np.abs(np.subtract(now.p, then.p)).max())
            angle = float(quat_angle_deg(now.q, then.q))
            row[arm] = {"axis_m": round(axis, 4), "deg": round(angle, 2)}
            within = within and axis <= axis_m and angle <= angle_deg
        row |= {"within": within, "honest": honest(receipt.ok, within)}
        return marked(ep, row)

    def measured_boxes(self) -> set[str]:
        return set() if self.measured is None else set(self.measured.boxes)

    def graspable(self, scene: gt_scene.Scene, at_most: int) -> list[str]:
        """The objects to grasp: every one the demonstration picked up, then the movable objects
        (truth's order) whose box the open gripper spans across its minor side, at most
        `at_most` of these."""
        span = min(s for arm in ARMS if (s := self.profile.arms[arm].gripper.span_m) is not None)
        others = [
            label
            for label in scene.boxes
            if label not in self.demo.picked
            and label in self.demo.movable
            and float(min(scene.proxy(label).whole.size[:2])) < span
        ]
        return [*self.demo.picked, *others[:at_most]]


async def run_check(
    bench: Bench,
    demo: DemoScene,
    control: EnvControl,
    planner: Planner,
    objects: int,
) -> dict[str, Any]:
    """Every group in the task's demonstration scene, at most `objects` objects grasped: the
    report (its rows, the scene's objects, the metrics)."""
    c = Check(bench, demo, control, planner)
    report: dict[str, Any] = {"task": c.task, "scene": demo.source}
    started = time.monotonic()
    async with c.episode() as ep:
        report["step_limit"] = ep.opened.step_limit
        report["objects"] = {
            label: {
                "movable": label in demo.movable,
                "size_m": rounded(ep.scene.proxy(label).whole.size),
            }
            for label in ep.scene.boxes
        }
        labels = c.graspable(ep.scene, objects)
        report["end_rules"] = [rule.model_dump() for rule in ep.first.end_rules]
    report["graspable"] = labels
    for group in ("move_to_pose", "displacement", "gripper"):
        report[group] = await guarded(getattr(c, group)())
        print(f"{time.strftime('%H:%M:%S')} {c.task} {group} done", flush=True)
    report["grasp"] = []
    for label in labels:
        if label not in c.measured_boxes():
            note = "truth has no box for it (a garment or a fluid): no grasp to build"
            report["grasp"].append({"label": label, "picked": True, "skipped": note})
            continue
        row = await guarded(c.grasp_and_place(label))
        report["grasp"].append({**row, "label": label, "picked": label in demo.picked})
        print(f"{time.strftime('%H:%M:%S')} {c.task} grasp {label} done", flush=True)
    report["seconds"] = round(time.monotonic() - started, 1)
    report["metrics"] = metrics(report)
    return report


async def guarded(work: Any) -> Any:
    """A group's rows, or the error that ended it: one group's crash (a planner timeout, an
    episode past its limit) does not lose the others."""
    try:
        return await work
    except Exception as error:  # recorded in the report, judged as a failure
        return {"error": f"{type(error).__name__}: {error}"}


# ---- judging -------------------------------------------------------------------------------------


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    """A report's metrics: errors as median and max, the rest as `passed/of` shares. `grasp_held`
    counts the planned grasps, `grasp_success` every graspable object (unplanned ones fail);
    `demo_picked` the objects the demonstration picked up, each grasped and placed or failed
    (also when it could not be tried). A row of an episode the benchmark ended (`ended`) is not
    the API's outcome: none counts, `ended` counts them."""
    poses = rows_of(report, "move_to_pose", lambda r: "pos_m" in r and judged(r))
    shifts = rows_of(report, "displacement", lambda r: "error_m" in r and judged(r))
    grips = rows_of(report, "gripper", lambda r: "reached" in r and judged(r))
    rows = [r for r in report["grasp"] if judged(r)]
    picked = [r for r in rows if r.get("picked") and judged(r.get("place", {}))]
    grasps = [r for r in rows if "error" not in r and "skipped" not in r]
    planned = [r for r in grasps if r.get("planned")]
    places = [r["place"] for r in planned if "place" in r and judged(r["place"])]
    tilts = [p for p in places if p["upright"] is not None]  # where the criteria judge a tilt
    origins = [r for g in ("move_to_pose", "displacement") for r in rows_of(report, g, is_origin)]
    origins += [r["return_home"] for r in grasps if "return_home" in r]
    origins = [r for r in origins if judged(r)]
    ended = [
        r
        for g in ("move_to_pose", "displacement", "gripper")
        for r in rows_of(report, g, ended_row)
    ]
    ended += [
        r
        for g in report["grasp"]
        for r in (g, g.get("place"), g.get("return_home"))
        if r and ended_row(r)
    ]
    honest = [r["honest"] for r in poses + shifts + grips + places + origins]
    honest += [r["grasp_honest"] for r in planned]
    errors = [g for g in ("move_to_pose", "displacement", "gripper") if "error" in report[g]]
    errors += [f"grasp {i}" for i, r in enumerate(report["grasp"]) if "error" in r]
    held = sum(r["held"] for r in planned)
    return {
        "pose_pos_m": stats([r["pos_m"] for r in poses]),
        "pose_deg": stats([r["deg"] for r in poses]),
        "poses_ok": f"{sum(r['ok'] for r in poses)}/{len(poses)}",
        "displacement_m": stats([r["error_m"] for r in shifts]),
        "gripper_reached": f"{sum(r['reached'] for r in grips)}/{len(grips)}",
        "grasp_planned": f"{len(planned)}/{len(grasps)}",
        "grasp_held": f"{held}/{len(planned)}",
        "grasp_success": f"{held}/{len(grasps)}",
        "place_placed": f"{sum(p['placed'] for p in places)}/{len(places)}",
        "place_upright": f"{sum(p['upright'] for p in tilts)}/{len(tilts)}",
        "demo_picked": f"{sum(picked_and_placed(r) for r in picked)}/{len(picked)}",
        "return_home": f"{sum(r['within'] for r in origins)}/{len(origins)}",
        "honest": f"{sum(honest)}/{len(honest)}",
        "ended": len(ended),
        "errors": errors,
    }


def rows_of(report: dict[str, Any], group: str, keep: Any) -> list[dict[str, Any]]:
    rows = report.get(group)
    return [r for r in rows if keep(r)] if isinstance(rows, list) else []


def is_origin(row: dict[str, Any]) -> bool:
    return row.get("call") == "return_home"


def stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    found = np.asarray(values, dtype=float)
    return {"median": round(float(np.median(found)), 5), "max": round(float(found.max()), 5)}


def picked_and_placed(row: dict[str, Any]) -> bool:
    """Whether a grasp row held its object and placed it (truth)."""
    if "error" in row or "skipped" in row or not row.get("planned") or not row["held"]:
        return False
    return bool(row.get("place", {}).get("placed"))


def place_verdict(row: dict[str, Any], criteria: Criteria) -> dict[str, Any]:
    """A place's judgement from what truth measured: placed when the object rests at the target
    (one still held stands above it; the gripper's reading right after the release may still be
    opening, so it is kept but not judged); upright when it also stands as it was picked (a ball
    or an egg may roll and still be placed; None where the criteria set no tilt)."""
    placed = row["off_m"] <= criteria.placed_m and abs(row["dz_m"]) <= criteria.resting_m
    upright = None
    if criteria.upright_deg is not None:
        upright = placed and row["tilt_deg"] <= criteria.upright_deg
    return {"placed": placed, "upright": upright, "honest": honest(row["ok"], placed)}


def move_tolerance(criteria: Criteria, config: ExecutorConfig) -> tuple[float, float]:
    """Where an ok move's TCP must end: the criteria's, else the binding's stop tolerance."""
    metres = config.stop_tolerance_m if criteria.move_m is None else criteria.move_m
    degrees = config.stop_tolerance_deg if criteria.move_deg is None else criteria.move_deg
    return metres, degrees


def rejudge(report: dict[str, Any], criteria: Criteria, picked: set[str]) -> dict[str, Any]:
    """A report judged again by `criteria` on its own measurements (rises, offsets, readings,
    move errors; a move by the criteria's tolerance where they set one, else as judged), the
    demonstration's objects `picked` marked; its metrics recomputed."""
    for group in ("move_to_pose", "displacement"):
        for row in rows_of(report, group, lambda r: "pos_m" in r or "error_m" in r):
            if criteria.move_m is not None:
                there = row.get("pos_m", row.get("error_m")) <= criteria.move_m
                if criteria.move_deg is not None and "deg" in row:
                    there = there and row["deg"] <= criteria.move_deg
                row["honest"] = honest(row["ok"], there)
    for row in rows_of(report, "gripper", lambda r: "reached" in r and not r["close"]):
        row["reached"] = row["reading"] >= criteria.opened
        row["honest"] = honest(row["ok"], row["reached"])
    for row in report["grasp"]:
        row["picked"] = row["label"] in picked
        if "error" in row or "skipped" in row or not row.get("planned"):
            continue
        row["held"] = row["rose_m"] >= criteria.held_m
        row["grasp_honest"] = honest(row["grasp"]["ok"] and row["lift"]["ok"], row["held"])
        if "place" in row:
            row["place"] |= place_verdict(row["place"], criteria)
    report["metrics"] = metrics(report)
    return report


def clearance(scene: gt_scene.Scene, point: np.ndarray) -> float:
    """How far `point` is from the nearest object's box (0 inside one)."""
    nearest = math.inf
    for name in scene.boxes:
        proxy = scene.proxy(name)
        box = proxy.pose @ proxy.whole.frame
        local = np.linalg.inv(box) @ np.array([*point, 1.0])
        outside = np.maximum(np.abs(local[:3]) - proxy.whole.size / 2, 0.0)
        nearest = min(nearest, float(np.linalg.norm(outside)))
    return nearest


def judged(row: dict[str, Any]) -> bool:
    return "ended" not in row


def ended_row(row: dict[str, Any]) -> bool:
    return "ended" in row


def marked(ep: Episode, row: dict[str, Any]) -> dict[str, Any]:
    """A row made after the benchmark ended its episode, with why: no action counted after the
    end, so it is not the API's outcome."""
    return row if ep.ended is None else row | {"ended": ep.ended}


def binding_rule(rules: list[EndRule], label: str) -> EndRule | None:
    """The tightest end rule on `label`'s motion (a lift moves it as far as it rises)."""
    mine = [rule for rule in rules if rule.object == label]
    return min(mine, key=lambda rule: rule.limit_m) if mine else None


def honest(claimed: bool, true: bool) -> bool:
    """A receipt is dishonest only where it reports success that truth contradicts."""
    return true or not claimed


def receipt_row(receipt: ActionReceipt) -> dict[str, Any]:
    return {
        "reason": str(receipt.reason),
        "ok": receipt.ok,
        "steps": receipt.steps_used,
        "note": receipt.note[:300],
    }


def tcp_error(tcp: Pose, target: Pose) -> dict[str, Any]:
    pos = float(np.linalg.norm(np.subtract(tcp.p, target.p)))
    deg = float(quat_angle_deg(tcp.q, target.q))
    return {"pos_m": round(pos, 5), "deg": round(deg, 3)}


def rounded(values: Any) -> list[float]:
    return [round(float(v), 4) for v in values]


def connect(interface: type[Any], url: str, token: SecretStr | None, **timeouts: Any) -> Any:
    """A typed RPC client of `interface` (a protocol class)."""
    return client(interface, url, token=token, **({"timeout": 120} | timeouts))
