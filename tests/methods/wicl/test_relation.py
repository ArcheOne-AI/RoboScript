"""wicl.relation: the demonstration's gripper at a keypoint, related to registered
objects by the specs that reproduce it. A spec gives a pose, the pose goes into a synthetic
demonstration (keypoints and the robot at their frames) on a synthetic registry, and relating
the keypoint gives the spec back: a grasp, a placement carried there by the recorded grasp."""

from typing import Final

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import FrameId
from roboscript.contracts.robot import Arm, RobotState
from roboscript.demos import Keypoint, KeypointSet, Tolerance
from roboscript.demos.keypoints import Cue, Phase
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.manipulation import (
    GraspSpec,
    ObjectProxy,
    PlaceSpec,
    PlaceTarget,
    grasp_candidates,
    placement_point_pose,
    target_frame,
    target_pose,
)
from roboscript.perception.registry import SceneRegistry, object_geometry
from roboscript.testing.scenes import TABLE_Z, Box, robot_state, surface, table_model
from roboscript_methods.wicl.relation import (
    frames,
    relate,
)

PROFILE: Final = load_embodiment(MOCK_BIMANUAL)  # a holding threshold of 0.1; its base: the world
BASE: Final = Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))
POSES: Final = load_embodiment(MOCK_BIMANUAL).poses
TO_GRASP: Final = np.array([[0.0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
GRASP: Final = GraspSpec(
    approach="top", anchor="top_center", closing="minor", offset_m=(0.01, 0.0, 0.0), yaw_deg=10
)
PLACE: Final = PlaceSpec(
    target=PlaceTarget(kind="object", obj="goal", anchor="top_center"),
    offset_m=(0.02, 0.01, 0.0),
    orient="yaw",
    yaw_deg=30,
)
HOLDING, OPEN, EMPTY = 0.5, 1.0, 0.0


def registry() -> SceneRegistry:
    """A long box (`tool`) and a flat one (`goal`), registered from their visible surfaces."""
    found = SceneRegistry()
    for name, box in (
        ("tool", Box((0.1, 0.0, TABLE_Z + 0.02), (0.12, 0.05, 0.04), yaw_deg=20)),
        ("goal", Box((-0.15, 0.1, TABLE_Z + 0.01), (0.2, 0.12, 0.02), yaw_deg=-10)),
    ):
        points = surface(box)
        points = points[points[:, 2] > TABLE_Z + 1e-6]
        found.register(name, points, frame=FrameId("f0"), step=0, table_z=TABLE_Z)
    return found


def proxy(scene: SceneRegistry, name: str) -> ObjectProxy:
    record = scene.named(name)
    assert record is not None
    return ObjectProxy.from_geometry(scene.geometry(record.id))


def tcp_of(obj: ObjectProxy, contact: np.ndarray) -> np.ndarray:
    """The TCP (world 4x4) of a grasp at a contact matrix (object frame)."""
    result: np.ndarray = obj.pose @ contact @ TO_GRASP
    return result


def raised(m: np.ndarray, dz: float) -> np.ndarray:
    up = m.copy()
    up[2, 3] += dz
    return up


def keypoint(i: int, frame: int, cue: Cue, phase: Phase, tcp: np.ndarray, closed: bool) -> Keypoint:
    return Keypoint(
        index=i,
        frame=frame,
        tcp=Pose.from_matrix(tcp),
        gripper="closed" if closed else "open",
        cue=cue,
        phase=phase,
        tolerance=Tolerance(),
    )


def demo(arms: dict[Arm, list[Keypoint]]) -> KeypointSet:
    return KeypointSet(
        task="lift_red",
        seed=0,
        frame_steps=1,
        success_frame=100,
        config_hash="test",
        pointing={"left": (1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)},
        arms=arms,
    )


def robots(
    keypoints: KeypointSet, openings: dict[Arm, dict[int, float]], still: dict[Arm, Pose]
) -> dict[int, RobotState]:
    """The robot at every keypoint's frame: each arm at its latest keypoint's TCP (or `still`)
    with the opening `openings` gives from that frame on."""
    found = {}
    for frame in frames(keypoints):
        state = robot_state(step=frame)
        arms = dict(state.arms)
        for arm in arms:
            before = [kp for kp in keypoints.arms.get(arm, []) if kp.frame <= frame]
            tcp = before[-1].tcp if before else still.get(arm, arms[arm].tcp)
            opened = [v for t, v in sorted(openings.get(arm, {}).items()) if t <= frame]
            arms[arm] = arms[arm].model_copy(
                update={"tcp": tcp, "gripper": opened[-1] if opened else OPEN}
            )
        found[frame] = state.model_copy(update={"arms": arms})
    return found


def pick_and_place(scene: SceneRegistry) -> tuple[KeypointSet, dict[int, RobotState], np.ndarray]:
    """The left arm grasps the tool as GRASP says, lifts it and places it as PLACE says; the
    tool's frame once placed."""
    tool, goal = proxy(scene, "tool"), proxy(scene, "goal")
    grasp = tcp_of(tool, grasp_candidates(GRASP, tool, "left")[0].matrix)
    point = placement_point_pose(tool, PLACE.placement_point, conventions=POSES)
    frame = target_frame(PLACE, goal, BASE)
    placed = target_pose(PLACE, frame, tool.pose @ point, tool.symmetry, point).matrix()
    placed = placed @ np.linalg.inv(point)  # the tool's frame placed
    release = placed @ np.linalg.inv(tool.pose) @ grasp
    kps = [
        keypoint(1, 10, "stop", "approach", raised(grasp, 0.1), False),
        keypoint(2, 20, "close", "grasp", grasp, True),
        keypoint(3, 30, "valley", "lift", raised(grasp, 0.1), True),
        keypoint(4, 40, "stop", "place", raised(release, 0.05), True),
        keypoint(5, 50, "open", "release", release, False),
        keypoint(6, 60, "stop", "retreat", raised(release, 0.1), False),
    ]
    keypoints = demo({"left": kps})
    return keypoints, robots(keypoints, {"left": {20: HOLDING, 50: OPEN}}, {}), placed


def test_a_grasp_comes_back_as_its_spec() -> None:
    scene = registry()
    keypoints, states, _ = pick_and_place(scene)
    tool = scene.named("tool")
    assert tool is not None
    found = relate(keypoints, "left K2", scene, states, PROFILE, table_model(), tool.id)
    assert found.grasp is not None and found.held is None and not found.place
    assert found.grasp.variant == "top/0"
    expected = GRASP.model_copy(update={"arm": "left"})
    fitted = found.grasp.spec
    numbers = {"offset_m": expected.offset_m, "offset_frac": expected.offset_frac, "yaw_deg": 10}
    numbers |= {"tilt_deg": 0, "lean_deg": 0, "reference_q": None}
    assert fitted.model_copy(update=numbers) == expected
    assert fitted.reference_q is not None  # its orientation in the robot base frame
    tool_proxy = ObjectProxy.from_geometry(object_geometry(tool, None))
    ((ours, theirs),) = [
        (a.matrix, b.matrix)
        for a in grasp_candidates(fitted, tool_proxy, "left")
        for b in grasp_candidates(GRASP, tool_proxy, "left")
        if a.label == b.label == "top/0"
    ]
    assert ours[:3, 3] == pytest.approx(theirs[:3, 3], abs=1e-6)  # the offset, maybe as fractions
    assert fitted.yaw_deg == pytest.approx(10, abs=1e-5)
    assert (fitted.tilt_deg, fitted.lean_deg) == pytest.approx((0, 0), abs=1e-6)
    assert (found.grasp.position_error_m, found.grasp.angle_error_deg) == pytest.approx(
        (0, 0), abs=1e-5
    )


def test_a_placement_comes_back_carried_by_the_recorded_grasp() -> None:
    scene = registry()
    keypoints, states, placed = pick_and_place(scene)
    tool, goal = scene.named("tool"), scene.named("goal")
    assert tool is not None and goal is not None
    point = placement_point_world(scene, placed)
    found = relate(keypoints, "left K5", scene, states, PROFILE, table_model(), goal.id)
    assert found.held == tool.id and found.grasp is None
    assert set(found.place) == {"plain", "target", "world_axes", "robot_frame"}
    # set down on the goal: a placement relative to it, not a constant point of the robot's
    assert {name for name, fit in found.place.items() if not fit.consistent} == {"robot_frame"}
    assert list(found.place)[-1] == "robot_frame"  # the consistent forms first
    fitted = found.place["target"].spec
    assert fitted.model_copy(
        update={"offset_m": PLACE.offset_m, "yaw_deg": 30}
    ) == PLACE.model_copy(update={"arm": "left"})
    assert fitted.offset_m == pytest.approx(PLACE.offset_m, abs=1e-6)
    turns = (fitted.yaw_deg - 30) % 180  # a two-fold box: either end first is the same
    assert min(turns, 180 - turns) == pytest.approx(0, abs=1e-5)
    for form in ("target", "world_axes"):
        fit = found.place[form]
        assert (fit.position_error_m, fit.angle_error_deg) == pytest.approx((0, 0), abs=1e-5)
    plain = found.place["plain"].spec  # on the goal: at its inside bottom, however it lay here
    assert (plain.target.anchor, plain.frame, plain.offset_m) == (
        "inside_bottom",
        "world_axes",
        (0.0, 0.0, 0.0),
    )
    alone = relate(keypoints, "left K5", registry(), states, PROFILE, table_model(), None)
    assert set(alone.place) == {"robot_frame"} and alone.obj is None
    base = alone.place["robot_frame"].spec.offset_m  # MockEnv's robot base is the world
    assert base == pytest.approx(point, abs=1e-6)
    # On the way to the release (the stop 5 cm above it, then the lift before it), no placement:
    # it is fitted at the open, K5; the held path is given as it was.
    above = relate(keypoints, "left K4", registry(), states, PROFILE, table_model(), goal.id)
    assert above.place == {} and above.release == "K5"
    lifted = relate(keypoints, "left K3", registry(), states, PROFILE, table_model(), goal.id)
    assert lifted.place == {} and lifted.release == "K5" and lifted.path is not None
    # The held path ends where the place leg begins: the 10 cm lift after the close.
    for relation in (found, above):
        assert relation.path is not None and len(relation.path.spec.poses) == 2
        lift = relation.path.spec.poses
        assert np.subtract(lift[1][:3], lift[0][:3]) @ np.subtract(lift[1][:3], lift[0][:3]) == (
            pytest.approx(0.01, abs=1e-6)
        )
        assert relation.path.travel_m == pytest.approx(0.1, abs=1e-4)


def placement_point_world(scene: SceneRegistry, placed: np.ndarray) -> np.ndarray:
    tool = proxy(scene, "tool")
    result: np.ndarray = (placed @ placement_point_pose(tool, "bottom", conventions=POSES))[:3, 3]
    return result
