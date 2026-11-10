"""`demo_relation` (docs/concepts/learning.md): the expert's TCP at a keypoint of the
demonstration in the frame of an object (or a part) the agent registered, as the GraspSpec or
PlaceSpec that reproduces it (`manipulation.fit`), with what the spec cannot reproduce.

On the demo seed the live scene is the one the demonstration started from, so besides perception
(the scene registry) it needs only the recorded proprioception: the keypoints' TCP poses and the
robot's state at the keypoints' frames. Objects start where they were first measured and follow the
recorded grasps by the registry's own grasp relations (`SceneRegistry.on_close`, `on_open`): at each
gripper event before the keypoint (closes before opens at one frame, as a hand-over goes), a close
whose opening settles at or above the holding threshold (read at the arm's next keypoint) takes the
registered object nearest the TCP, and an open leaves what the arm held where its grasp puts it,
settled (where the episode measured it again on the demonstration's frame after that open,
`release_frames`, there: as a live run measures an object it let go of); a close on an
articulated part takes that part alone. At a keypoint where its arm holds an
articulated part, the relation is a hinge turn: the HingeSpec on the part's box as measured and the
angle its TCP turned from the close to the keypoint (`fit_hinge`), and the held path from the close
(`held_path`: the TCP poses in the object's body frame, which `follow_path` repeats as a rigid
motion; no hinge needed); where the gripper turned too little for a hinge (a drawer slid out),
the held path alone. Where its arm holds an object (its held path too, in the object's frame at
the close), the relation is that object's placement at a target (a registered object or part: its
anchor nearest the placement point, in its axes and the world's; with no target, the robot base
frame), the orientation it is held at as the placement starts (for `keep`) taken at the last
keypoint before the place leg, each form with the TCP `place_object` would command with it at
`stop_back_m` 0 against the expert's, by `constrain` (`placed_gaps`: a form reproduces the held
object's pose on its own scene, which says nothing of what the call does with it); otherwise it
is a grasp of the object named, written as the other
arm's receiving grasp (x toward it) where that arm holds the object then, a press where the
keypoint's close holds nothing (with the push that follows it: the move to the arm's next keypoint,
still closed). A grasp is fitted where the gripper closes: a keypoint on the way to a later close of
its arm (an approach) is refused, naming the close. Every grasp comes in its pose form (the TCP in
the whole object's frame, `fit_grasp_pose`: no part box the next registration measures anew moves
it) and its plain form too, and every placement at a target (`manipulation.fit`). At a retreat (the
gripper leaving what it released) it is the world displacement from the arm's keypoint before. No
object ground truth is read.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final, cast

import numpy as np

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm, RobotState
from roboscript.demos.keypoints import Keypoint, KeypointSet
from roboscript.embodiments import EmbodimentProfile
from roboscript.manipulation import (
    GraspFit,
    HingeFit,
    ObjectProxy,
    PathFit,
    PlaceFit,
    contact_of,
    fit_grasp,
    fit_grasp_pose,
    fit_hinge,
    fit_path,
    fit_place,
    placement,
    target_frame,
)
from roboscript.manipulation.api import robot_base
from roboscript.manipulation.posemath import Constrain
from roboscript.manipulation.specs import MAX_PATH_POSES, PlaceSpec, Purpose
from roboscript.perception.registry import (
    PRESS_REACH,
    WHOLE,
    ObjectRecord,
    SceneRegistry,
    object_geometry,
)
from roboscript.perception.table import Table
from roboscript.spatial.rotation import quat_angle_deg

Released = Mapping[int, Mapping[str, Pose]]
"""Where an episode measured the objects the demonstration let go of, again on its frames
(`release_frames`): by the frame of the open, each object's pose (registry id -> its record's
`pose`, for the record's own fit)."""

NAME: Final = r"^(left|right) K[1-9][0-9]*$"
"""How a keypoint is named: its arm and number, `left K2`."""
PLACE_LEG: Final = ("place", "release")
_CLOSED_BELOW: Final = 0.5  # a gripper command below this is closed
_SAME_HEIGHT: Final = 0.001  # rests this close in height are a tie
"""The phases of a placement's own moves."""


@dataclass(frozen=True)
class Relation:
    """The relation at `keypoint` of `arm`: a grasp of `obj` (`grasp`), a placement of the
    object the arm holds (`held`) at `obj` (None: the robot base only) in each form (`place`), or
    a turn of `obj`'s articulated `part` about a hinge (`hinge`); a grasp's plain form too
    (`plain`: `fit_grasp`'s) and its pose form (`pose`: the TCP in the whole object's frame,
    `fit_grasp_pose`), and after a press the world displacement to the arm's next keypoint
    (`pushed`); where the arm holds something, the held path from its close (`path`); at a
    release the other arm already holds what it lets go, that arm's close (`handover`: a
    hand-over, `left K7`). Where the arm holds an object on its way to the open that ends the
    hold, no placement: the open (`release`, `K5`; `release_after`), where it is fitted."""

    arm: Arm
    keypoint: Keypoint
    obj: str | None
    held: str | None = None
    grasp: GraspFit | None = None
    pose: GraspFit | None = None
    place: dict[str, PlaceFit] = field(default_factory=dict)
    hinge: HingeFit | None = None
    part: str | None = None
    retreat: tuple[float, float, float] | None = None
    plain: GraspFit | None = None
    pushed: tuple[float, float, float] | None = None
    path: PathFit | None = None
    handover: str | None = None
    release: str | None = None


CONTACT_MOVE_M: Final = 0.02
"""An arm still holding what it closed on at the success frame, whose TCP moved this far since
its last keypoint, carried it on to a contact no gripper event marks (a held tool's strike, an
object pushed home): `contacts` gives it a keypoint there."""


def contacts(keypoints: KeypointSet, robots: Mapping[int, RobotState]) -> dict[Arm, Keypoint]:
    """Per arm, the contact keypoint the extraction has no event for (`CONTACT_MOVE_M`): its next
    number, at the success frame, the TCP the recording has there, phase `carry`, still closed
    (the referee keeps its own keypoints: this one only names where `demo_relation` fits the
    held object's placement and where S1's review compares it). Proprioception only; none
    without a success frame or the robot there."""
    final = keypoints.success_frame
    if final is None or final not in robots:
        return {}
    found: dict[Arm, Keypoint] = {}
    for arm, kps in keypoints.arms.items():
        grips = [kp for kp in kps if kp.grip]
        if not kps or not grips or grips[-1].cue != "close" or grips[-1].phase == "preshape":
            continue
        last, state = kps[-1], robots[final].arms.get(arm)
        if state is None or last.frame >= final:
            continue
        if float(np.linalg.norm(np.subtract(state.tcp.p, last.tcp.p))) < CONTACT_MOVE_M:
            continue
        found[arm] = last.model_copy(
            update={
                "index": last.index + 1,
                "frame": final,
                "since": last.frame,
                "tcp": state.tcp,
                "gripper": "closed",
                "cue": "stop",
                "phase": "carry",
            }
        )
    return found


def with_contacts(keypoints: KeypointSet, found: Mapping[Arm, Keypoint]) -> KeypointSet:
    """`keypoints` with each arm's contact keypoint (`contacts`) appended."""
    if not found:
        return keypoints
    arms = {
        arm: [*kps, *([found[arm]] if arm in found else [])] for arm, kps in keypoints.arms.items()
    }
    return keypoints.model_copy(update={"arms": arms})


def find(keypoints: KeypointSet, name: str) -> tuple[Arm, Keypoint]:
    """The keypoint `left K2` names (`NAME`); ValueError naming those there are."""
    side, number = name.split()
    arm, index = cast(Arm, side), int(number.removeprefix("K"))
    listed = keypoints.arms.get(arm, [])
    if not 1 <= index <= len(listed):
        there = ", ".join(f"{a} K1-K{len(kps)}" for a, kps in keypoints.arms.items())
        raise ValueError(f"the demonstration has no {name} (it has {there})")
    return arm, listed[index - 1]


def frames(keypoints: KeypointSet) -> list[int]:
    """The recording's frames a relation reads the robot at: every keypoint's."""
    return sorted({kp.frame for kps in keypoints.arms.values() for kp in kps})


def relate(
    keypoints: KeypointSet,
    name: str,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    table: Table | None,
    obj: str | None,
    part: str | None = None,
    placement_point: str = "bottom",
    approach: bool = True,
    released: Released | None = None,
) -> Relation:
    """The relation at keypoint `name` (see the module docstring). `registry` is changed (give a
    copy); `robots` holds the robot at every keypoint's frame (`frames`); `obj` is a registered
    object's id: the object grasped, or a placement's target (None: the robot base); `approach`:
    a placement's fits carry the line the expert came in along where `place_object`'s default
    line is off it (`with_approach`); `released`: see `replay_grips`."""
    arm, keypoint = find(keypoints, name)
    if keypoint.phase == "retreat":  # the gripper leaving what it released
        before = keypoints.arms[arm][keypoint.index - 2]
        move = np.subtract(keypoint.tcp.p, before.tcp.p)
        return Relation(
            arm, keypoint, None, retreat=(float(move[0]), float(move[1]), float(move[2]))
        )
    replay_grips(
        keypoints, registry, robots, profile, table, before=keypoint.frame, released=released
    )
    robot = _at(robots, arm, keypoint)
    turning = next((o for o in registry.objects.values() if arm in o.part_grasps), None)
    if turning is not None:
        if obj not in (None, turning.id):
            raise ValueError(f"at {name} the {arm} arm holds a part of another object")
        held_part = turning.part_grasps[arm].part
        close = next(
            kp
            for kp in reversed(keypoints.arms[arm])
            if kp.cue == "close" and kp.frame < keypoint.frame
        )
        arc = [kp.tcp for kp in keypoints.arms[arm] if close.frame <= kp.frame <= keypoint.frame]
        body = turning.model_copy(update={"part_grasps": {}, "part_poses": {}})
        path = held_path(keypoints.arms[arm], keypoint, _proxy(body, robot), arm)
        try:
            hinge = fit_hinge(arc, _proxy(body, robot), held_part)
        except ValueError:
            if path is None or len(arc) < 2:  # no move since the close: nothing to relate
                raise
            hinge = None  # it slid the part without turning (a drawer): the held path only
        return Relation(arm, keypoint, turning.id, hinge=hinge, part=held_part, path=path)
    held = next((o for o in registry.objects.values() if arm in o.grasps), None)
    if held is None:
        if obj is None:
            raise ValueError(f"the {arm} arm holds nothing at {name}: name the object it grasps")
        closing = next(
            (kp for kp in keypoints.arms[arm] if kp.frame >= keypoint.frame and kp.cue == "close"),
            None,
        )
        if closing is not None and closing is not keypoint:
            raise ValueError(
                f"{name} ({keypoint.phase}) is on the way to a grasp: the grasp is where the "
                f"gripper closes, {arm} {closing.name}; ask for that keypoint"
            )
        record = registry.get(obj)
        other: Arm = "right" if arm == "left" else "left"
        receiving = other in record.grasps
        purpose = _purpose(keypoints, arm, keypoint, robots, profile)
        span = profile.arms[arm].gripper.span_m
        if purpose != "press" and span is not None:
            _check_grasped(registry, record, robot, keypoint, part, span / 2, name)
        fitted = functools.partial(
            fit_grasp,
            keypoint.tcp,
            _proxy(record, robot),
            part=part or WHOLE,
            purpose=purpose,
            arm=arm,
            other_tcp=robot.arms[other].tcp.p if receiving else None,
            base=robot_base(profile).matrix(),
        )
        pushed = _pushed(keypoints.arms[arm], keypoint) if purpose == "press" else None
        posed = fit_grasp_pose(
            keypoint.tcp,
            _proxy(record, robot),
            purpose=purpose,
            arm=arm,
            other_tcp=robot.arms[other].tcp.p if receiving else None,
            base=robot_base(profile).matrix(),
        )
        return Relation(
            arm,
            keypoint,
            obj,
            grasp=fitted(),
            plain=fitted(plain=True),
            pose=posed,
            pushed=pushed,
        )
    if obj == held.id:
        raise ValueError(f"at {name} the {arm} arm holds that object: name the placement's target")
    carried = _proxy(held, robot)
    close = next(
        kp
        for kp in reversed(keypoints.arms[arm])
        if kp.cue == "close" and kp.frame < keypoint.frame
    )
    path = held_path(keypoints.arms[arm], keypoint, _proxy(held, _at(robots, arm, close)), arm)
    if (opening := release_after(keypoints.arms[arm], keypoint)) is not None:
        return Relation(arm, keypoint, obj, held=held.id, path=path, release=opening.name)
    before = place_start(keypoints.arms[arm], keypoint)
    now = _proxy(held, _at(robots, arm, before)).pose
    target = None if obj is None else _proxy(registry.get(obj), robot)
    fits = fit_place(
        carried.pose,
        carried,
        target,
        robot_base(profile),
        placement_point=placement_point,
        held_now=now,
        part=part,
        arm=arm,
        plain=True,
        conventions=profile.poses,
    )
    start = _at(robots, arm, before)  # where place_object starts: the place leg's start
    held_then = _proxy(held, start)
    fits = {
        name: replace(
            fit,
            tcp_gaps=placed_gaps(fit.spec, held_then, target, arm, start, keypoint.tcp, profile),
        )
        for name, fit in fits.items()
    }
    if approach:
        ee = profile.ee_pose(arm, robot.arms[arm].tcp)
        fits = with_approach(fits, keypoints.arms[arm], keypoint, carried, ee, target, profile)
    handover = _received(keypoints, arm, keypoint) if other_holds(held, arm) else None
    return Relation(arm, keypoint, obj, held=held.id, place=fits, path=path, handover=handover)


def release_after(keypoints: list[Keypoint], kp: Keypoint) -> Keypoint | None:
    """Where a placement at `kp`, a keypoint the arm holds something at, belongs instead: the
    arm's open that ends this hold (its next gripper event), when `kp` is not that open. None at
    the open itself and in a hold that ends without one (held in the air to the end), where a
    placement is fitted as it is."""
    if kp.cue == "open":
        return None
    after = next((k for k in keypoints if k.frame > kp.frame and k.grip), None)
    return after if after is not None and after.cue == "open" else None


def other_holds(record: ObjectRecord, arm: Arm) -> bool:
    """Whether the arm other than `arm` holds `record` too (a hand-over under way)."""
    return any(holder != arm for holder in record.grasps)


def _received(keypoints: KeypointSet, arm: Arm, kp: Keypoint) -> str | None:
    """At `arm`'s release `kp` of an object the other arm holds too: the other arm's last close
    up to it (`left K7`), where it took the object over; None at another keypoint."""
    if kp.cue != "open":
        return None
    other: Arm = "right" if arm == "left" else "left"
    closes = [k for k in keypoints.arms.get(other, []) if k.cue == "close" and k.frame <= kp.frame]
    return f"{other} {closes[-1].name}" if closes else None


APPROACH_DEG: Final = 20.0
"""A placement fit carries the expert's approach where `place_object`'s default line (the grasp's:
from the ee to the held object) lies further than this from the line the expert came in along."""


def release_line(keypoints: list[Keypoint], kp: Keypoint) -> np.ndarray | None:
    """The unit direction the TCP moved in from the arm's keypoint before `kp` to it (the
    straight move `approach_m` measures, wicl's `toolsets.demo._distances`); None after a
    gripper event or without a move."""
    before = keypoints[kp.index - 2] if kp.index > 1 else None
    if before is None or before.grip:
        return None
    move = np.subtract(kp.tcp.p, before.tcp.p)
    norm = float(np.linalg.norm(move))
    return None if norm < 1e-3 else move / norm


def with_approach(
    fits: Mapping[str, PlaceFit],
    keypoints: list[Keypoint],
    kp: Keypoint,
    held: ObjectProxy,
    ee: Pose,
    target: ObjectProxy | None,
    profile: EmbodimentProfile,
) -> dict[str, PlaceFit]:
    """`fits` with the line the expert came in along onto the release (`release_line`) as each
    one's `approach`, in its own frame (`synthesis.target_frame`: the target's axes, the world's,
    or the robot base's), where `place_object`'s default line, from the ee to the held object at
    the release, lies more than `APPROACH_DEG` off it (a handle threaded onto a rod, an object
    slid in sideways); else as they are. `approach_m` along it is the move's length (`_distances`):
    with `stop_back_m` 0 the call repeats the expert's last straight move."""
    line = release_line(keypoints, kp)
    if line is None:
        return dict(fits)
    default = held.pose[:3, 3] - np.asarray(ee.p, dtype=float)
    cos = float(np.dot(default, line) / (np.linalg.norm(default) + 1e-12))
    if np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) <= APPROACH_DEG:
        return dict(fits)
    base = robot_base(profile)
    out = {}
    for name, fit in fits.items():
        local = target_frame(fit.spec, target, base)[:3, :3].T @ line
        approach = (round(float(local[0]), 3), round(float(local[1]), 3), round(float(local[2]), 3))
        out[name] = replace(fit, spec=fit.spec.model_copy(update={"approach": approach}))
    return out


CONSTRAINS: Final[tuple[Constrain, ...]] = ("auto", "align", "free")
"""The `constrain` values of `place_object` a placement's commanded TCP is checked under."""
REPRODUCED: Final = (0.03, 20.0)
"""A commanded TCP this near the expert's (m, deg) reproduces it: S1's placement tolerances
(`learning.expert_diff.ReviewConfig` `place_m`, `place_deg`)."""


def reproducing(gaps: Mapping[str, tuple[float, float]]) -> list[str]:
    """The `constrain` values under which a placement's commanded TCP (`placed_gaps`) reproduces
    the expert's (within `REPRODUCED`), in `CONSTRAINS` order."""
    meters, degrees = REPRODUCED
    return [c for c in CONSTRAINS if c in gaps and gaps[c][0] <= meters and gaps[c][1] <= degrees]


def placed_tcp(
    spec: PlaceSpec,
    held: ObjectProxy,
    target: ObjectProxy | Pose | None,
    arm: Arm,
    robot: RobotState,
    profile: EmbodimentProfile,
    *,
    constrain: Constrain = "auto",
    distance: float = 0.0,
    honor_front: bool = True,
) -> Pose:
    """The TCP `place_object(spec, constrain=constrain)` commands `distance` back from its first
    option (the placement as written, unturned), the arm at `robot` holding `held` where its
    grasp holds it (`manipulation.api.placement`: what the call computes). A held object whose
    shape looks the same turned a half or quarter turn is taken as a pose grasp's close pins it
    (`ExecutorConfig.hold_frames`, `SceneRegistry.pin`): in the frame its hold was fitted in, no
    equivalent turn chosen for it."""
    found = placed_turns(spec, held, target, arm, robot, profile, constrain=constrain,
                         distances=(distance,), honor_front=honor_front)  # fmt: skip
    return found[0][0]


def placed_turns(
    spec: PlaceSpec,
    held: ObjectProxy,
    target: ObjectProxy | Pose | None,
    arm: Arm,
    robot: RobotState,
    profile: EmbodimentProfile,
    *,
    constrain: Constrain = "auto",
    distances: tuple[float, ...] = (0.0,),
    honor_front: bool = True,
) -> list[list[Pose]]:
    """Per option `place_object` tries for the placement as written (its equivalent turns, in the
    order the call tries them, the first unturned: `placed_tcp`'s), the TCPs it commands at each
    of `distances` back; one option where nothing is turned."""
    if (held.shape_symmetry or held.symmetry) in ("pi", "half_pi"):
        held = replace(held, symmetry="none")
    base = robot_base(profile)
    lateral = base.matrix()[:3, 1]
    ee = profile.ee_pose(arm, robot.arms[arm].tcp)
    found = placement(held, spec, target, arm, ee, base, lateral, constrain=constrain,
                      honor_front=honor_front, conventions=profile.poses)  # fmt: skip
    center = found.centers[0]
    return [
        [profile.tcp_pose(arm, found.ee(center, turn, 0, distance)) for distance in distances]
        for turn in found.turns
    ]


def placed_gaps(
    spec: PlaceSpec,
    held: ObjectProxy,
    target: ObjectProxy | Pose | None,
    arm: Arm,
    robot: RobotState,
    expert: Pose,
    profile: EmbodimentProfile,
) -> dict[str, tuple[float, float]]:
    """How far (m, deg) the TCP `place_object` commands with `spec` at `stop_back_m` 0 lies from the
    expert's (`expert`), by `constrain` (`CONSTRAINS`): a placement fitted on the held object's
    pose may still be commanded elsewhere (the official placement turns the object by its own
    rule unless `align`, and an object that looks the same turned takes the turn nearest how it
    is held)."""
    gaps = {}
    for constrain in CONSTRAINS:
        tcp = placed_tcp(spec, held, target, arm, robot, profile, constrain=constrain)
        gaps[constrain] = held_gap(tcp, expert, held, robot.arms[arm].tcp)
    return gaps


def held_gap(tcp: Pose, expert: Pose, held: ObjectProxy, now: Pose) -> tuple[float, float]:
    """How far (m, deg) a TCP lies from the expert's, as what it puts the held object at: the
    TCPs themselves, but for a round object (its measured shape looks the same turned about its
    own z, `continuous`) the held object's poses there (it held at `now` as `held` is), its
    position and the tilt of its z, since a turn about its own axis leaves it where the expert
    put it (an object let go turned about itself). A half or quarter turn of a shape is not
    taken as nothing: a measured footprint looks so for objects that are not (one with distinct
    ends)."""
    if (held.shape_symmetry or held.symmetry) != "continuous":
        return (
            float(np.linalg.norm(np.subtract(tcp.p, expert.p))),
            float(quat_angle_deg(tcp.q, expert.q)),
        )
    grip = np.linalg.inv(now.matrix()) @ held.pose
    ours, theirs = tcp.matrix() @ grip, expert.matrix() @ grip
    cos = float(np.clip(ours[:3, 2] @ theirs[:3, 2], -1.0, 1.0))
    return float(np.linalg.norm(ours[:3, 3] - theirs[:3, 3])), float(np.degrees(np.arccos(cos)))


def held_path(
    keypoints: list[Keypoint], kp: Keypoint, body: ObjectProxy, arm: Arm | None = None
) -> PathFit | None:
    """The held path (`manipulation.paths`) of an arm holding something at `kp`: its TCP at its
    keypoints from its last close before `kp` up to `kp`, or, at a placement's own move, up to
    the last keypoint before it (`place_start`: the place leg is the placement's), in the frame of
    `body` (what it holds, or the object whose part it holds, as it was at the close); None
    where the arm made no move since. Of a longer recording, `MAX_PATH_POSES` keypoints evenly
    spaced, the first and last kept."""
    close = next(kp_ for kp_ in reversed(keypoints) if kp_.cue == "close" and kp_.frame < kp.frame)
    end = place_start(keypoints, kp) if kp.phase in PLACE_LEG else kp
    held = [k.tcp for k in keypoints if close.frame <= k.frame <= end.frame]
    if len(held) < 2:
        return None
    if len(held) > MAX_PATH_POSES:
        picks = np.linspace(0, len(held) - 1, MAX_PATH_POSES).round().astype(int)
        held = [held[i] for i in picks]
    return fit_path(held, body, arm)


def _pushed(keypoints: list[Keypoint], close: Keypoint) -> tuple[float, float, float] | None:
    """The world displacement from a press's close to the arm's next keypoint while it stays
    closed (the push); None if there is none."""
    after = keypoints[close.index : close.index + 1]
    if not after or after[0].gripper != "closed":
        return None
    move = np.subtract(after[0].tcp.p, close.tcp.p)
    return float(move[0]), float(move[1]), float(move[2])


def place_start(keypoints: list[Keypoint], placed: Keypoint) -> Keypoint:
    """The keypoint the object is held at as the placement starts (what `keep` keeps): the last
    one before the place leg (the `place` and `release` keypoints up to `placed`), from the
    arm's close on."""
    close = next(kp for kp in reversed(keypoints) if kp.cue == "close" and kp.frame < placed.frame)
    carrying = [kp for kp in keypoints if close.frame <= kp.frame < placed.frame]
    return next((kp for kp in reversed(carrying) if kp.phase not in PLACE_LEG), close)


def _outside(proxy: ObjectProxy, tcp: Pose, part: str | None) -> float:
    """How far (m) the TCP is outside the measured box of `proxy` (of its `part`, if named)."""
    box = proxy.whole if part is None else proxy.part(part)
    local = np.linalg.inv(proxy.pose @ box.frame) @ contact_of(tcp)[:, 3]
    return float(np.linalg.norm(np.maximum(np.abs(local[:3]) - np.asarray(box.size) / 2, 0.0)))


def _check_grasped(
    registry: SceneRegistry,
    record: ObjectRecord,
    robot: RobotState,
    keypoint: Keypoint,
    part: str | None,
    margin: float,
    where: str,
) -> None:
    """Refuse a grasp of `record` at `keypoint` whose closing TCP is farther than `margin` (half
    the open fingers' span: what they can reach past a box and still close on it) outside its
    measured box (its `part`'s, if named), naming the registered object nearest the TCP: the
    demonstration's gripper closed on something else there, and a spec fitted on this object
    would carry that gap to every scene."""
    off = _outside(_proxy(record, robot), keypoint.tcp, part)
    if off <= margin:
        return
    others = [
        (_outside(_proxy(o, robot), keypoint.tcp, None), o.name)
        for o in registry.objects.values()
        if o.id != record.id
    ]
    nearest = min(others, default=None)
    shown = record.name if part is None else f"{record.name}'s {part}"
    named = (
        ""
        if nearest is None
        else f"; the registered object nearest it is {nearest[1]} ({nearest[0] * 100:.1f} cm "
        "outside its box): ask for that object"
    )
    raise ValueError(
        f"at {where} the gripper closes {off * 100:.1f} cm outside {shown}'s measured box, more "
        f"than the open fingers reach ({margin * 100:.1f} cm): it does not grasp {record.name} "
        f"there{named}"
    )


def _proxy(record: ObjectRecord, robot: RobotState) -> ObjectProxy:
    """The object as the registry has it at `robot`."""
    return ObjectProxy.from_geometry(object_geometry(record, robot))


def release_key(registry: SceneRegistry) -> dict[str, list[float]]:
    """The registry a `Released` was measured for: each object's first measured pose."""
    return {
        r.id: (r.history[0].pose if r.history else r.pose).as_list()
        for r in registry.objects.values()
    }


def released_poses(fact: Mapping[str, Any] | None, registry: SceneRegistry) -> Released | None:
    """The `DEMO_RELEASES` fact's poses (`tools.DEMO_RELEASES`) of the objects `registry` still
    has as they were when measured (the same id and first pose, `release_key`; an object
    registered again or replaced since drops out), else None (no fact)."""
    if fact is None:
        return None
    now, then = release_key(registry), fact.get("objects", {})
    out = {}
    for t, poses in fact.get("poses", {}).items():
        kept = {
            o: Pose.from_list(p) for o, p in poses.items() if o in now and then.get(o) == now[o]
        }
        if kept:
            out[int(t)] = kept
    return out


def released_tried(fact: Mapping[str, Any] | None, registry: SceneRegistry) -> set[tuple[int, str]]:
    """The releases (open frame, object) the `DEMO_RELEASES` fact measured, or tried to, for
    objects `registry` still has as they were then (`released_poses`)."""
    if fact is None:
        return set()
    now, then = release_key(registry), fact.get("objects", {})
    return {(int(t), o) for t, o in fact.get("tried", []) if o in now and then.get(o) == now[o]}


def release_frames(keypoints: KeypointSet) -> list[tuple[Arm, Keypoint, int]]:
    """Each open of the demonstration (arm, keypoint, in frame order) with the frame to measure
    what it let go of in: the arm's next keypoint (its gripper moved off) when that comes before
    the next gripper event of either arm, else the open's own frame."""
    events = sorted(
        ((a, kp) for a, kps in keypoints.arms.items() for kp in kps if kp.grip),
        key=lambda event: (event[1].frame, event[1].cue != "close"),
    )
    out = []
    for i, (a, kp) in enumerate(events):
        if kp.cue == "close":
            continue
        later = [e.frame for _, e in events[i + 1 :] if e.frame > kp.frame]
        after = keypoints.arms[a][kp.index :]  # the arm's keypoints after this one
        frame = kp.frame
        if after and (not later or after[0].frame < later[0]):
            frame = after[0].frame
        out.append((a, kp, frame))
    return out


def replay_grips(
    keypoints: KeypointSet,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    table: Table | None,
    *,
    before: int,
    released: Released | None = None,
) -> None:
    """Put `registry` (changed: give a copy) where the demonstration had it just before frame
    `before`: every object back where it was first measured, then each gripper event of an
    earlier frame replayed by the registry's grasp relations (see the module docstring); after an
    open, each object `released` gives for its frame that no arm holds is put where it was
    measured."""
    for record in list(registry.objects.values()):
        first = record.history[0].pose if record.history else record.pose
        registry.objects[record.id] = record.model_copy(
            update={"pose": first, "grasps": {}, "carried": False}
        )
    events = sorted(
        ((a, kp) for a, kps in keypoints.arms.items() for kp in kps if kp.grip),
        key=lambda event: (event[1].frame, event[1].cue != "close"),
    )
    for a, kp in events:
        if kp.frame >= before:
            break
        if kp.cue == "close":
            gripper = profile.arms[a].gripper
            robot = _at(robots, a, kp, opening=_settled(keypoints, a, kp, robots))
            registry.on_close(
                a,
                robot,
                threshold=gripper.holding_threshold,
                closing_axis=profile.poses.tcp_closing_axis,
                span_m=gripper.span_m,
            )
        else:
            registry.on_open(a, _at(robots, a, kp), table)
            for obj, pose in (released or {}).get(kp.frame, {}).items():
                record = registry.objects.get(obj)
                if record is not None and not record.grasps:
                    registry.objects[obj] = record.model_copy(update={"pose": pose})


def replay_presses(
    keypoints: KeypointSet,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    *,
    final: int,
) -> None:
    """The demonstration's presses (the done metric `press`) on `registry` as `replay_grips`
    left it. A rest is an arm stopped at a motion keypoint up to frame `final` or at `final`
    itself (a recording that stops at the latch ends mid-press) with its gripper closed on
    nothing (commanded closed, the measured opening below the holding threshold). Each rest goes
    to the registered object no arm holds that it reaches (`SceneRegistry.nearest_free`: over
    its footprint, above its perceived top by no more than that top's own uncertainty). A
    recording tells a push from a rest only by what it met, and the demonstration succeeded, so
    per object the press is its LOWEST such rest (the approach hovers above it; ties: the later
    frame), kept by `SceneRegistry.on_press`, in frame order. Objects are where the replay left
    them (a pressed object stays put)."""
    lowest: dict[str, tuple[float, int, Arm, Pose]] = {}
    for arm, frame, tcp in empty_rests(keypoints, robots, profile, final=final):
        found = registry.nearest_free(np.asarray(tcp.p), PRESS_REACH)
        if found is None:
            continue
        z = float(tcp.p[2])
        held = lowest.get(found[0])
        lower = held is None or z < held[0] - _SAME_HEIGHT
        later = held is not None and z <= held[0] + _SAME_HEIGHT and frame > held[1]
        if lower or later:
            lowest[found[0]] = (z, frame, arm, tcp)
    for _, _, arm, tcp in sorted(lowest.values(), key=lambda found: found[1]):
        registry.on_press(arm, tcp.p)


def empty_rests(
    keypoints: KeypointSet,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    *,
    final: int,
) -> list[tuple[Arm, int, Pose]]:
    """The demonstration's rests with a gripper closed on nothing, as `replay_presses` reads
    them (arm, frame, TCP): an arm stopped at a motion keypoint up to frame `final`, or at
    `final` itself, commanded closed with the measured opening below the holding threshold.
    `robots` has every such frame. Only proprioception decides: a demonstration that has one
    pushed something (press, click, toggle) with empty fingers."""
    rests = [
        (arm, kp.frame, kp.tcp)
        for arm, kps in keypoints.arms.items()
        for kp in kps
        if kp.frame <= final and not kp.grip
    ]
    rests += [(arm, final, state.tcp) for arm, state in robots[final].arms.items()]
    found = []
    for arm, frame, tcp in rests:
        state = robots[frame].arms[arm]
        opening, threshold = state.gripper, profile.arms[arm].gripper.holding_threshold
        if state.gripper_cmd >= _CLOSED_BELOW or opening is None or opening >= threshold:
            continue  # open, unmeasured, or holding something
        found.append((arm, frame, tcp))
    return found


def _at(
    robots: Mapping[int, RobotState], arm: Arm, kp: Keypoint, opening: float | None = None
) -> RobotState:
    """The robot at a keypoint's frame, the arm's TCP the keypoint's (and its opening, if
    given)."""
    robot = robots[kp.frame]
    state = robot.arms[arm].model_copy(update={"tcp": kp.tcp})
    if opening is not None:
        state = state.model_copy(update={"gripper": opening})
    return robot.model_copy(update={"arms": {**robot.arms, arm: state}})


def _settled(
    keypoints: KeypointSet, arm: Arm, close: Keypoint, robots: Mapping[int, RobotState]
) -> float | None:
    """The arm's opening once a close settled: at its next keypoint (its own if the last)."""
    later = next((kp for kp in keypoints.arms[arm] if kp.frame > close.frame), close)
    return robots[later.frame].arms[arm].gripper


def _purpose(
    keypoints: KeypointSet,
    arm: Arm,
    kp: Keypoint,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
) -> Purpose:
    """`press` where the keypoint's close settles holding nothing, else `pick`."""
    if kp.cue != "close":
        return "pick"
    opening = _settled(keypoints, arm, kp, robots)
    empty = opening is not None and opening < profile.arms[arm].gripper.holding_threshold
    return "press" if empty else "pick"
