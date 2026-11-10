"""Conformance checks for motion planners (docs/concepts/embodied-core.md).

A planner conforms when its answers are consistent with each other and with the `Planner`
contract: FK and the TCP offset agree, IK round trips, plans start at their start and end at their
goal, orientation locks hold or are refused, failures say why, batches keep their order, and
requests that do not fit an arm are refused. The checks work on an object or an RPC client alike.

Use them from pytest by subclassing `PlannerConformance` as a `Test*` class and providing a
`planner_target` fixture, or without pytest via `roboscript.testing.precheck`. Checks marked
`unit` plan nothing.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import (
    BatchPlanRequest,
    FKRequest,
    IKRequest,
    Planner,
    PlanRequest,
)
from roboscript.contracts.robot import Arm
from roboscript.spatial.pose import pose_compose
from roboscript.spatial.rotation import quat_angle_deg
from roboscript.testing import unit


@dataclass(frozen=True)
class PlannerTarget:
    """`samples` holds, per arm to check, two or more joint vectors away from collisions and
    limits whose ee poses are reachable from each other and have room to move 2 cm up or down;
    `base` is the robot root's pose. The contract bounds no orientation lock's drift (in cuRobo
    the lock is a soft cost), so `lock_drift_deg`, if given, bounds it for these samples only."""

    planner: Planner
    base: Pose
    samples: Mapping[Arm, Sequence[Sequence[float]]]
    lock_drift_deg: float | None = None

    def arms(self) -> list[tuple[Arm, list[list[float]]]]:
        return [(arm, [list(map(float, q)) for q in qs]) for arm, qs in self.samples.items()]


class PlannerConformance:
    """The checks, as `test_*` methods taking a `PlannerTarget`."""

    @unit
    async def test_info(self, planner_target: PlannerTarget) -> None:
        info = await planner_target.planner.info()
        assert info.dt > 0
        for arm, qs in planner_target.arms():
            spec = info.arms[arm]
            assert len(spec.lower) == len(spec.upper) == len(spec.joints), spec
            assert len(qs) >= 2 and all(len(q) == len(spec.joints) for q in qs), (arm, qs)

    @unit
    async def test_requests_must_fit_the_arm(self, planner_target: PlannerTarget) -> None:
        planner, base = planner_target.planner, planner_target.base
        for arm, qs in planner_target.arms():
            bad = [
                FKRequest(arm=arm, base=base, joints=[qs[0][:-1]]),
                FKRequest(arm=arm, base=base, joints=[qs[0]], links=["no_such_link"]),
            ]
            for request in bad:
                try:
                    await planner.fk(request)
                except ProtocolError as error:
                    assert error.code == "invalid_request", error
                else:
                    raise AssertionError(f"{arm}: {request} was accepted")

    async def test_fk_places_the_tcp(self, planner_target: PlannerTarget) -> None:
        planner, base = planner_target.planner, planner_target.base
        info = await planner.info()
        for arm, qs in planner_target.arms():
            links = info.arms[arm].links[:1]
            fk = await planner.fk(FKRequest(arm=arm, base=base, joints=qs, links=links))
            assert len(fk.ee) == len(fk.tcp) == len(qs)
            assert all(len(fk.links[name]) == len(qs) for name in links)
            offset = [*info.arms[arm].ee_to_tcp, 1.0, 0.0, 0.0, 0.0]
            for ee, tcp in zip(fk.ee, fk.tcp, strict=True):
                _near(tcp, Pose.from_list(pose_compose(ee.as_list(), offset)), 1e-6, 1e-3, "TCP")

    async def test_fk_reports_collision_spheres_around_the_arm(
        self, planner_target: PlannerTarget
    ) -> None:
        planner, base = planner_target.planner, planner_target.base
        for arm, qs in planner_target.arms():
            fk = await planner.fk(FKRequest(arm=arm, base=base, joints=qs, spheres=True))
            assert len(fk.spheres) == len(qs), arm
            for spheres, tcp in zip(fk.spheres, fk.tcp, strict=True):
                assert spheres, f"{arm}: no collision spheres"
                # the hand is covered: some sphere lies within 10 cm of the TCP
                gaps = [math.dist(s[:3], tcp.p) - s[3] for s in spheres]
                assert min(gaps) < 0.1, (arm, min(gaps))
            plain = await planner.fk(FKRequest(arm=arm, base=base, joints=qs[:1]))
            assert plain.spheres == [], arm

    async def test_ik_round_trips(self, planner_target: PlannerTarget) -> None:
        planner, base = planner_target.planner, planner_target.base
        for arm, qs in planner_target.arms():
            goals = (await planner.fk(FKRequest(arm=arm, base=base, joints=qs))).ee
            solutions = await planner.ik(IKRequest(arm=arm, base=base, goals=goals, seed=qs[0]))
            assert len(solutions) == len(goals)
            for solution, goal in zip(solutions, goals, strict=True):
                assert solution.joints is not None, f"{arm}: no IK solution for a pose from FK"
                back = await planner.fk(FKRequest(arm=arm, base=base, joints=[solution.joints]))
                _near(back.ee[0], goal, 0.005, 3.0, "FK of an IK solution")

    async def test_plans_reach_their_goals(self, planner_target: PlannerTarget) -> None:
        planner, base = planner_target.planner, planner_target.base
        dt = (await planner.info()).dt
        for arm, qs in planner_target.arms():
            goal = (await planner.fk(FKRequest(arm=arm, base=base, joints=[qs[1]]))).ee[0]
            request = PlanRequest(arm=arm, base=base, start=qs[0], goal=goal, ee_path=True)
            result = await planner.plan(request)
            assert result.status == "success", f"{arm}: {result.reason} ({result.detail})"
            path = np.asarray(result.path)
            assert result.dt == dt and path.shape[1] == len(qs[0]) and len(path) >= 2
            assert np.abs(path[0] - qs[0]).max() < 1e-4, "a path starts at its start"
            assert result.ee_path is not None and len(result.ee_path) == len(path)
            _near(result.ee_path[-1], goal, 0.005, 3.0, "the end of a path")

    async def test_orientation_locks(self, planner_target: PlannerTarget) -> None:
        planner, base = planner_target.planner, planner_target.base
        for arm, qs in planner_target.arms():
            here = (await planner.fk(FKRequest(arm=arm, base=base, joints=[qs[0]]))).ee[0]
            results = []
            for dz in (0.02, -0.02):
                moved = Pose(p=(here.p[0], here.p[1], here.p[2] + dz), q=here.q)
                results.append(await planner.plan(_locked(arm, base, qs[0], moved, ee_path=True)))
            done = [r for r in results if r.status == "success"]
            assert done and done[0].ee_path is not None, f"{arm}: {results}"
            drift = max(float(quat_angle_deg(p.q, here.q)) for p in done[0].ee_path)
            bound = planner_target.lock_drift_deg
            assert bound is None or drift <= bound, f"{arm}: the locked move drifted {drift} deg"
            half = math.radians(30) / 2  # well beyond the ~5.7 deg a lock tolerates
            turn = [0.0, 0.0, 0.0, math.cos(half), 0.0, 0.0, math.sin(half)]
            turned = Pose.from_list(pose_compose(here.as_list(), turn))
            refused = await planner.plan(_locked(arm, base, qs[0], turned))
            assert refused.reason == "orientation_mismatch", f"{arm}: {refused}"

    async def test_failures_say_why(self, planner_target: PlannerTarget) -> None:
        planner, base = planner_target.planner, planner_target.base
        info = await planner.info()
        for arm, qs in planner_target.arms():
            goal = (await planner.fk(FKRequest(arm=arm, base=base, joints=[qs[1]]))).ee[0]
            far = Pose(p=(base.p[0], base.p[1], base.p[2] + 10.0), q=goal.q)
            missed = await planner.plan(PlanRequest(arm=arm, base=base, start=qs[0], goal=far))
            assert missed.reason == "unreachable" and missed.path is None, f"{arm}: {missed}"
            outside = [u + 1.0 for u in info.arms[arm].upper]
            invalid = await planner.plan(PlanRequest(arm=arm, base=base, start=outside, goal=goal))
            assert invalid.reason == "invalid_start", f"{arm}: {invalid}"

    async def test_batches_keep_their_order(self, planner_target: PlannerTarget) -> None:
        planner, base = planner_target.planner, planner_target.base
        for arm, qs in planner_target.arms():
            goal = (await planner.fk(FKRequest(arm=arm, base=base, joints=[qs[1]]))).ee[0]
            far = Pose(p=(base.p[0], base.p[1], base.p[2] + 10.0), q=goal.q)
            request = BatchPlanRequest(arm=arm, base=base, start=qs[0], goals=[goal, far, goal])
            results = await planner.plan_batch(request)
            assert [r.status for r in results] == ["success", "failed", "success"], results


def _locked(
    arm: Arm, base: Pose, start: list[float], goal: Pose, ee_path: bool = False
) -> PlanRequest:
    return PlanRequest(
        arm=arm, base=base, start=start, goal=goal, constraint="orientation", ee_path=ee_path
    )


def _near(a: Pose, b: Pose, meters: float, degrees: float, what: str) -> None:
    distance = float(np.linalg.norm(np.subtract(a.p, b.p)))
    angle = float(quat_angle_deg(a.q, b.q))
    assert distance <= meters and angle <= degrees, (
        f"{what}: {distance * 1000:.2f} mm and {angle:.3f} deg apart"
    )
