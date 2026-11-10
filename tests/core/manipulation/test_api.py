"""The action API on MockEnv with MockPlanner: specs on true-geometry proxies of MockEnv's boxes,
through the official pose math, executed and judged."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.ids import EpisodeId, TaskName
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.manipulation import (
    ActionApi,
    Executor,
    GraspSpec,
    ObjectProxy,
    Part,
    PlaceSpec,
    PlaceTarget,
)
from roboscript.manipulation.plan import Move
from roboscript.motion import MockPlanner
from roboscript.perception.registry import WHOLE
from roboscript.testing.skills import MOCK_EXECUTOR

TOP = GraspSpec(approach="top", closing="major")


@dataclass
class Rig:
    env: MockEnv
    episode: EpisodeId
    api: ActionApi

    async def box(self, name: str) -> ObjectProxy:
        """A mock box's true-geometry proxy: frame at its center, a 5 cm cube."""
        pose = (await self.env.gt_snapshot(self.episode)).objects[name]
        part = Part(frame=np.eye(4), size=np.full(3, 0.05), x_known=False)
        return ObjectProxy(name, pose.matrix(), "half_pi", {WHOLE: part})


async def make_rig(
    tmp_path: Path,
    task: str = "stack_red_on_blue",
    seed: int = 3,
    planner: MockPlanner | None = None,
) -> Rig:
    env = MockEnv(MockConfig(task=TaskName(task), blob_dir=tmp_path))
    opened = await env.open(EpisodeSpec(task=TaskName(task), seed=seed))
    executor = Executor(
        env, opened.episode_id, planner or MockPlanner(), env.profile, MOCK_EXECUTOR
    )
    api = ActionApi(executor)
    return Rig(env, opened.episode_id, api)


async def test_a_grasp_and_a_placement_by_specs_stack_the_boxes(tmp_path: Path) -> None:
    rig = await make_rig(tmp_path)
    red = await rig.box("red")
    arm = "left" if red.pose[0, 3] < 0 else "right"
    built = await rig.api.grasp_object(arm, red, TOP, approach_m=0.1)
    assert built.refused is None and built.grasp is not None and built.grasp.n_candidates == 4
    assert [type(s).__name__ for s in built.segments] == ["Move", "Move", "Grip"]
    grasped = await rig.api.run(built)
    assert grasped.reason == "ok" and grasped.gripper is not None and grasped.gripper.holding
    assert grasped.grasp == built.grasp and built.label in grasped.note
    assert built.label.startswith("grasp candidate") and "of 4 (top/" in built.label
    tcp = np.array((await rig.env.robot(rig.episode)).arms[arm].tcp.p)
    assert tcp == pytest.approx(red.pose[:3, 3], abs=1e-6)  # the TCP on the box's center
    lifted = await rig.api.run(await rig.api.move_by(arm, z=0.1))
    assert lifted.reason == "ok"
    place = PlaceSpec(target=PlaceTarget(kind="object", obj="blue", anchor="top_center"))
    built = await rig.api.place_object(
        arm, await rig.box("red"), place, await rig.box("blue"), stop_back_m=0.005, constrain="free"
    )
    assert [type(s).__name__ for s in built.segments] == ["Move", "Move", "Grip"]
    placed = await rig.api.run(built)
    assert placed.reason in ("ok", "stopped_short")
    assert (await rig.env.judge(rig.episode)).success


async def test_a_placement_that_stops_short_keeps_holding_instead_of_letting_go(
    tmp_path: Path,
) -> None:
    """The release waits for the place: an arm that stops beyond the contact tolerance keeps holding
    (here it stops at the pre-place, 8 cm short), one that gets there lets go."""
    rig = await make_rig(tmp_path)
    red = await rig.box("red")
    arm = "left" if red.pose[0, 3] < 0 else "right"
    assert (await rig.api.run(await rig.api.grasp_object(arm, red, TOP))).reason == "ok"
    assert (await rig.api.run(await rig.api.move_by(arm, z=0.1))).reason == "ok"
    executor = rig.api.executor
    run = executor.run

    async def short(*segments: Any, **kwargs: Any) -> Any:  # the place leg never comes
        cut = [s[:-1] if s and isinstance(s[-1], Move) and s[-1].contact else s for s in segments]
        return await run(*cut, **kwargs)

    place = PlaceSpec(target=PlaceTarget(kind="object", obj="blue", anchor="top_center"))
    built = await rig.api.place_object(arm, red, place, await rig.box("blue"), constrain="free")
    object.__setattr__(executor, "run", short)
    stopped = await rig.api.run(built)
    object.__setattr__(executor, "run", run)
    assert "kept holding: it stopped 8.0 cm" in stopped.note, stopped.note
    assert stopped.gripper is not None and stopped.gripper.holding
    assert not (await rig.env.judge(rig.episode)).success
    built = await rig.api.place_object(
        arm, await rig.box("red"), place, await rig.box("blue"), stop_back_m=0.005, constrain="free"
    )
    placed = await rig.api.run(built)
    assert "kept holding" not in placed.note
    assert placed.gripper is not None and not placed.gripper.holding
    assert (await rig.env.judge(rig.episode)).success


async def test_displacements_follow_the_official_semantics(tmp_path: Path) -> None:
    rig = await make_rig(tmp_path)
    start = (await rig.env.robot(rig.episode)).arms["left"].ee
    world = await rig.api.move_by("left", x=0.05, z=-0.05)
    target = world.segments[0].target  # type: ignore[union-attr]
    assert target.p == pytest.approx(np.add(start.p, (0.05, 0.0, -0.05)))
    assert world.segments[0].contact  # type: ignore[union-attr]  # lowering the TCP by 5 cm
    back = await rig.api.move_by("left", z=0.1, move_axis="arm")
    along = -0.1 * start.matrix()[:3, 0]  # back along the gripper's pointing direction
    assert back.segments[0].target.p == pytest.approx(np.add(start.p, along))  # type: ignore[union-attr]
    turned = await rig.api.move_by("left", quat=(0.0, 1.0, 0.0, 0.0))
    assert turned.segments[0].target.q == pytest.approx((0.0, 1.0, 0.0, 0.0))  # type: ignore[union-attr]
    assert (await rig.api.run(world)).reason == "ok"
    assert (await rig.api.run(await rig.api.return_home("left"))).reason == "ok"
    assert (await rig.env.robot(rig.episode)).arms["left"].ee.p == pytest.approx(start.p)


async def test_two_arms_move_together(tmp_path: Path) -> None:
    rig = await make_rig(tmp_path)
    left = await rig.api.move_by("left", y=0.05)
    right = await rig.api.move_by("right", y=0.05)
    done = await rig.api.run(left, right)
    assert done.reason == "ok" and done.steps_used == 1
    state = await rig.env.robot(rig.episode)
    assert state.arms["right"].ee.p == pytest.approx(right.segments[0].target.p)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="one action per arm"):
        await rig.api.run(left, left)


async def test_nothing_plannable_is_refused_without_moving(tmp_path: Path) -> None:
    rig = await make_rig(tmp_path)
    red = await rig.box("red")
    far = GraspSpec(approach="top", offset_m=(0.0, 0.0, 0.29))  # above the mock's reach
    far_box = red.at(np.array([0.0, 0.0, 1.3, 1.0, 0.0, 0.0, 0.0]))
    built = await rig.api.grasp_object("left", far_box, far)
    assert built.refused is not None and built.refused.reason == "unreachable"
    assert "no pre-grasp of the 4 candidates" in built.refused.note
    receipt = await rig.api.run(built)
    assert receipt is built.refused and (await rig.env.robot(rig.episode)).step == 0
    check = await rig.api.check_reach(await rig.api.move_by("left", z=0.05))
    assert (check.reachable, check.reason, check.steps) == (True, "ok", 1)
    assert set(check.targets) == {"left"} and (await rig.env.robot(rig.episode)).step == 0


def hinged(
    monkeypatch: pytest.MonkeyPatch, tcp: np.ndarray, pivot: tuple[float, float], stop_deg: float
) -> None:
    """MockEnv with the part held at `tcp` on a true vertical hinge at `pivot` (x, y): each step
    takes a holding arm's TCP to the hinge's arc (the hold's radius and height, turned at most
    `stop_deg` from it either way), the ee shifted with it; a command more than 2 cm off the
    arc tears the part out of the fingers."""
    from roboscript.envs import mock

    step = mock._step
    rel0 = tcp[:2] - pivot
    radius, angle0 = float(np.linalg.norm(rel0)), float(np.arctan2(rel0[1], rel0[0]))

    def on_hinge(scene: Any, qpos: np.ndarray, profile: Any) -> None:
        qpos = qpos.copy()
        for arm, spec in profile.arms.items():
            if arm not in scene.held:
                continue
            now = mock._tcp(qpos, profile, arm)[:3]
            rel = now[:2] - pivot
            turn = (np.arctan2(rel[1], rel[0]) - angle0 + np.pi) % (2 * np.pi) - np.pi
            angle = angle0 + np.clip(turn, *np.radians([-stop_deg, stop_deg]))
            on = np.array([*(pivot + radius * np.array([np.cos(angle), np.sin(angle)])), tcp[2]])
            if np.linalg.norm(on - now) > 0.02:
                qpos[spec.gripper_index] = 1.0
                continue
            begin = spec.action[0]
            qpos[begin : begin + 3] += on - now
        step(scene, qpos, profile)

    monkeypatch.setattr(mock, "_step", on_hinge)


def raised(m: np.ndarray, dz: float) -> np.ndarray:
    up = m.copy()
    up[2, 3] += dz
    return up
