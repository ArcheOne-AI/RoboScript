"""S1's spec dry run (docs/concepts/learning.md): before an attempt, does each grasp spec of the
Memory, as the Actor reads it (its Memory line read back, `memory.doc.as_read`: a field of any
grasp or placement spec the line loses is off), synthesized on runtime perception's registry of
the demo seed's scene, land on the expert's grasp? Privileged (S1 at tier P), no motion and no
planning: the expert's state only.

The registry is the one the Demo Reader stored, on the demonstration's start scene, where every fit
was made (never a later attempt's: a spec fitted on one registration and checked on another would
show the registration's difference, not the spec's), each object where perception first measured
it, with the parts registered on it. The check only reports: the Reviser after the attempt reads
the gaps beside the attempt's own evidence.

A spec fitted on the demonstration names the keypoint it was fitted at (its source,
`demo <arm> K<n>`); the object that keypoint closes on (the referee's `acted_on`) is the
registered object matched to it at the start (`diagnosis.match`: overlapping boxes, else centers
within `diagnosis.MATCH_M`). The spec's grasp candidates on that object's proxy (`synthesis`) are
compared with the expert's TCP at the keypoint: the candidate nearest it with its closing axis (the
contact frame's z, the TCP's y) within `ReviewConfig.contact_deg` of the expert's, else the
nearest. A spec whose best candidate lies beyond `ReviewConfig.spec_m` (or none comes) is off; one
that cannot be checked here (unfitted, its object or part not registered, held) says why.

Placement and held-path specs are checked on the demonstration's state at their keypoint: the
registry's objects replayed through the recorded grips to it (`demos.relation.replay_grips`), so
the held object is where the expert's grasp held it. A placement is what `place_object` would
command with the workflow's call (`demos.relation.placed_tcp`: its `constrain`, `approach_m` and
`stop_back_m`, as the stage's call line passes them, else the defaults), the arm where the place leg
starts: its place TCP against the expert's at the keypoint (within `ReviewConfig.place_m` and
`place_deg`), its TCP at `stop_back_m` 0 under each `constrain` beside it. A held path is what
`follow_path` would command from the close (`manipulation.paths`): each waypoint's TCP against
the expert's keypoint it was fitted from. Both give the poses the dry run then plans
(`targets`, each from the recorded robot at `starts`): a spec whose commanded TCPs do not plan
from the expert's own state is off too, with the planner's reason.
"""

import json
import logging
import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import BatchPlanRequest, Planner
from roboscript.contracts.robot import Arm, RobotState
from roboscript.demos import Keypoint, KeypointSet, ObjectShape
from roboscript.demos.referee import acted_on, keypoint_objects
from roboscript.embodiments import EmbodimentProfile
from roboscript.learning.demo import DemoTruth
from roboscript.manipulation.api import robot_base
from roboscript.manipulation.paths import choose_frame, hold_frames, motions
from roboscript.manipulation.posemath import Constrain
from roboscript.manipulation.proxy import ObjectProxy
from roboscript.manipulation.specs import (
    OTHER_ARM,
    GraspSpec,
    HingeSpec,
    PathSpec,
    PlaceSpec,
    SpecError,
)
from roboscript.manipulation.synthesis import grasp_candidates
from roboscript.memory import MemoryDoc
from roboscript.memory.calls import parse_calls
from roboscript.memory.doc import as_read, lost_fields
from roboscript.memory.doc import place_call_args as call_args
from roboscript.memory.params import (
    GraspSpecParam,
    PathSpecParam,
    PlaceSpecParam,
    jsonable,
)
from roboscript.perception.registry import WHOLE, ObjectRecord, SceneRegistry, object_geometry
from roboscript.perception.table import Table
from roboscript.spatial.rotation import quat_angle_deg
from roboscript_methods.wicl.diagnosis import match, truth_box
from roboscript_methods.wicl.expert_diff import ReviewConfig, center
from roboscript_methods.wicl.relation import (
    PLACE_LEG,
    Released,
    held_gap,
    place_start,
    placed_gaps,
    placed_turns,
    relate,
    release_after,
    replay_grips,
    reproducing,
)
from roboscript_methods.wicl.toolsets.demo import as_fitted, fit_spec

log = logging.getLogger(__name__)

FITTED: Final = re.compile(r"demo (left|right) K([1-9][0-9]*)")
REGISTRY: Final = "registry"
AIR_NOTE: Final = (
    "every candidate's fingers would close on air on the registered object (grasp_object refuses "
    "it before anything moves): move the contact onto measured material"
)
ORIGIN: Final = Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))


class SpecGap(BaseModel):
    """One spec against the expert at the keypoint it was fitted at: a grasp's best candidate
    (`candidates`, `offset`, `closing_deg`, `approach_deg`), or a placement's or held path's
    commanded TCPs (`kind`; `angle_deg` the turn off the expert's; `constrains`: a
    placement's TCP at `stop_back_m` 0 off the expert's by `constrain`, m and deg), with the poses
    the dry run plans (`targets`, the `arm`'s TCP, each from the recorded robot at the frame in
    `starts`) and the planner's verdicts (`plans`: None where it plans); `turns`: a placement's
    other options `place_object` tries in order (a symmetric object's equivalent turns), each its
    targets, planned in turn where the first does not plan, as the call does."""

    model_config = ConfigDict(frozen=True)

    parameter: str
    keypoint: str | None
    object: str | None
    status: Literal["off", "ok", "unchecked"]
    kind: Literal["grasp", "place", "path"] = "grasp"
    candidates: int = 0
    offset: tuple[float, float, float] | None = None
    distance_m: float | None = None
    closing_deg: float | None = None
    approach_deg: float | None = None
    angle_deg: float | None = None
    constrain: str | None = None
    constrains: dict[str, tuple[float, float]] = {}
    arm: Arm | None = None
    targets: list[Pose] = []
    starts: list[int] = []
    plans: list[str | None] = []
    turns: list[list[Pose]] = []
    note: str = ""


def last_registry(directory: Path) -> BlobRef | None:
    """The last registry an episode (in `directory`) stored; None without one."""
    found = last_fact(directory, REGISTRY)
    return None if found is None else BlobRef.model_validate(found["blob"])


def first_registry(directory: Path) -> BlobRef | None:
    """The first registry an episode (in `directory`) stored, as its identification left it;
    None without one."""
    found = last_fact(directory, REGISTRY, first=True)
    return None if found is None else BlobRef.model_validate(found["blob"])


def last_fact(directory: Path, name: str, *, first: bool = False) -> Any:
    """The last value (`first`: the first) an episode (in `directory`) stored for the scene fact
    `name`; None."""
    path = directory / "events.jsonl"
    if not path.is_file():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        data = event.get("data") or {}
        if event["kind"] == "scene.fact" and data.get("name") == name:
            found = data["value"]
            if first:
                break
    return found


Fit = GraspSpec | PlaceSpec | HingeSpec
OUTLINE_M: Final = 0.05
"""A close acts on an object whose outline (its center less half its width) lies this near the
TCP, where none lies near enough by its center (`acted_on`): a large object's handle."""


def grasp_objects(
    keypoints: Sequence[Keypoint],
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
) -> list[str | None]:
    """Per keypoint of one arm, the object it acts on (`acted_on`), and at a close that finds
    none by center, the object whose outline lies within `OUTLINE_M` of the TCP (a cabinet's
    handle, far from the cabinet's center)."""
    found = acted_on(keypoints, poses, shapes)
    for j, kp in enumerate(keypoints):
        if kp.cue != "close" or found[j] is not None:
            continue
        best, gap = None, OUTLINE_M
        for name, pose in poses.get(kp.frame, {}).items():
            shape = shapes.get(name)
            if shape is None or shape.size is None:
                continue
            reach = float(np.linalg.norm(center(pose, shape) - np.asarray(kp.tcp.p)))
            if (distance := reach - shape.size[0] / 2) < gap:
                best, gap = name, distance
        found[j] = best
    return found


def demo_fits(
    keypoints: KeypointSet,
    layer: DemoTruth,
    shapes: Mapping[str, ObjectShape],
    registry: SceneRegistry,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    table: Table | None,
    approach: bool = True,
    released: Released | None = None,
) -> dict[str, Fit | str]:
    """The spec the demonstration fits at each keypoint (`demos.relation.relate` on the Demo
    Reader's registry, as `demo_relation` gives it), for the S1 Reviser's `fit_spec`, or why
    none fits there: at a close, the grasp of the object it closes on (the referee's `acted_on`,
    matched to its registration: privileged, S1 only), its pose form first; where a held part
    turns, the hinge; where a held object is placed (the arm's open, or the last keypoint of a
    hold that ends without one), the placement at what it rests on (else the robot base), its
    first consistent form."""
    poses = {frame.frame: frame.poses for frame in layer.frames}
    start = layer.frames[0].poses
    out: dict[str, Fit | str] = {}
    for arm, kps in keypoints.arms.items():
        objects = grasp_objects(kps, poses, shapes)
        touches = keypoint_objects(kps, poses, shapes)
        for j, kp in enumerate(kps):
            name = f"{arm} {kp.name}"
            if kp.phase in ("preshape", "approach", "retreat"):
                out[name] = f"{name} is an {kp.phase} keypoint: no spec acts there"
                continue
            target = objects[j] if kp.cue == "close" else min(touches[j].onto, default=None)
            obj = None
            if target is not None and target in start:
                shape = shapes.get(target)
                record = registered(registry, center(start[target], shape), shape)
                obj = None if record is None else record.id
            if kp.cue == "close" and obj is None:
                out[name] = "the object it closes on is not registered on the demonstration's scene"
                continue
            try:
                # `relate` changes the registry it gets: a fresh copy each
                copy = registry.clone()
                relation = relate(keypoints, name, copy, robots, profile, table, obj,
                                  approach=approach, released=released)  # fmt: skip
            except (LookupError, ValueError) as error:
                out[name] = str(error)
                continue
            if relation.hinge is not None:
                out[name] = as_fitted("hinge", relation.hinge).spec
            elif relation.grasp is not None:
                out[name] = as_fitted("grasp", relation.pose or relation.grasp).spec
            elif relation.release is not None:
                out[name] = (
                    f"{name} is on the way to the release: a placement is fitted where the "
                    f"gripper opens, {arm} {relation.release} (fit_spec there)"
                )
            elif relation.place:
                fits = [as_fitted(form, fit) for form, fit in relation.place.items()]
                consistent = [f for f in fits if f.consistent] or fits
                made = [f for f in consistent if reproducing(f.tcp_gaps)]  # a call reproduces
                out[name] = (made or consistent)[0].spec
            elif relation.part is not None and relation.path is not None:
                out[name] = (
                    f"the arm moves {relation.part} there without a turn: its held path is a "
                    "path spec, not one fit_spec makes"
                )
            else:
                out[name] = "the gripper moves clear there (a retreat): no spec acts there"
    return out


def object_fits(
    keypoints: KeypointSet,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    table: Table | None,
    given: Mapping[str, Sequence[Mapping[str, Any]]],
    approach: bool = True,
    released: Released | None = None,
) -> dict[str, PlaceSpec]:
    """By keypoint, the placement the demonstration fits relative to the object the Demo
    Reader's own placement fits there named (`given`, `tools.DEMO_FITS`: the one object they
    name), its first consistent object-relative form (none where no form is consistent: the
    robot-frame fit stands for a new spec): what the Reviser's `fit_spec` refits a
    spec that targets an object with (it keeps its target kind, `memory.Editor`), where
    `demo_fits` gives another form (a placement in the air, which no support decides);
    `released`: see `demos.relation.replay_grips`."""
    out: dict[str, PlaceSpec] = {}
    for name, fits in given.items():
        named = {
            str((fit.get("target") or {}).get("obj"))
            for fit in fits
            if (fit.get("target") or {}).get("kind") == "object"
        }
        found = [registry.named(obj) for obj in named]
        if len(found) != 1 or found[0] is None:
            continue
        try:
            relation = relate(keypoints, name, registry.clone(), robots, profile, table,
                              found[0].id, approach=approach, released=released)  # fmt: skip
        except (LookupError, ValueError):
            continue
        forms = [
            as_fitted(form, fit)
            for form, fit in relation.place.items()
            if fit.spec.target.kind == "object"
        ]
        consistent = [f for f in forms if f.consistent]  # none consistent: no object fit
        made = [f for f in consistent if reproducing(f.tcp_gaps)]
        if consistent:
            spec = (made or consistent)[0].spec
            assert isinstance(spec, PlaceSpec)
            out[name] = spec
    return out


def spec_check(
    keypoints: KeypointSet,
    layer: DemoTruth,
    shapes: Mapping[str, ObjectShape],
    registry: SceneRegistry,
    memory: MemoryDoc,
    profile: EmbodimentProfile,
    config: ReviewConfig,
    tcps: Mapping[int, Mapping[Arm, Sequence[float]]] | None = None,
    robots: Mapping[int, RobotState] | None = None,
    given: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    only: Collection[str] | None = None,
    released: Released | None = None,
) -> list[SpecGap]:
    """Each grasp, placement and held-path spec of `memory` (those named in `only`, if given)
    on `registry` against the expert
    (see the module). `tcps`: the demonstration's TCP positions (world) by keypoint frame and
    arm, from its recorded proprioception, where a spec aiming at the other gripper (`x_toward:
    other_arm`) finds it as the expert had it then (without them, at the other arm's last
    keypoint up to then). `robots`: the recorded robot at every keypoint's frame, which
    placements and paths are checked on (without it they are not checked). `given`: the specs
    `demo_relation` gave the Demo Reader by keypoint (`tools.DEMO_FITS`), which name the
    registry's objects: a placement whose target the Memory names otherwise (its own role name)
    is checked at the object the fit it was made from names (`target_record`)."""
    poses = {frame.frame: frame.poses for frame in layer.frames}
    start = layer.frames[0].poses
    lateral = robot_base(profile).matrix()[:3, 1]
    out = []
    for name, param in memory.params.items():
        if only is not None and name not in only:
            continue
        if isinstance(param, PathSpecParam):
            out.append(path_check(name, param.value, param.source, keypoints, registry, robots,
                                  profile, config, released))  # fmt: skip
            continue
        if not isinstance(param, GraspSpecParam | PlaceSpecParam):
            continue
        # the spec as the Actor receives it: its Memory line, read back; a field the line
        # loses shows here as off (the Actor would have to invent it)
        try:
            read = as_read(param)
        except ValueError as error:
            note = f"its Memory line gives no valid spec: {error}"
            out.append(SpecGap(parameter=name, keypoint=None, object=None, status="off", note=note))
            continue
        if lost := lost_fields(param.value, read):
            note = f"its Memory line does not carry {', '.join(lost)}"
            out.append(SpecGap(parameter=name, keypoint=None, object=None, status="off", note=note))
            continue
        if isinstance(param, PlaceSpecParam):
            assert isinstance(read, PlaceSpec)
            out.append(
                place_check(name, read, param.source, keypoints, registry, robots, profile, config,
                            call_args(memory, name), given, released)
            )  # fmt: skip
            continue
        assert isinstance(read, GraspSpec)
        fitted = FITTED.fullmatch(param.source)
        if fitted is None or fitted[1] not in keypoints.arms:
            out.append(_unchecked(name, None, None, "not fitted on the demonstration"))
            continue
        arm, index = fitted[1], int(fitted[2])
        kps = keypoints.arms[arm]
        if index > len(kps):
            out.append(_unchecked(name, None, None, f"the demonstration has no {arm} K{index}"))
            continue
        kp, where = kps[index - 1], f"{arm} K{index}"
        obj = acted_on(kps, poses, shapes)[index - 1]
        if obj is None or obj not in start:
            out.append(_unchecked(name, where, None, "the expert closes on no known object"))
            continue
        record = registered(registry, center(start[obj], shapes.get(obj)), shapes.get(obj))
        if record is None:
            out.append(_unchecked(name, where, obj, "no object is registered at its place"))
            continue
        # where perception first measured it, before anything moved it (as `measured` proxies)
        unmoved = {"pose": record.fit.frame, "grasps": {}, "part_grasps": {}, "part_poses": {}}
        geometry = object_geometry(record.model_copy(update=unmoved))
        spec = read
        other_tcp = _other_tcp(keypoints, tcps, arm, kp.frame)
        if spec.x_toward == OTHER_ARM and other_tcp is None:
            note = "the other gripper's place at that keypoint is not recorded"
            out.append(_unchecked(name, where, obj, note))
            continue
        if spec.part != WHOLE and spec.part not in geometry.parts:
            note = f"its part {spec.part} is not registered in the registry read"
            out.append(_unchecked(name, where, obj, note))
            continue
        proxy = ObjectProxy.from_geometry(geometry)
        try:
            found = grasp_candidates(
                spec,
                proxy,
                arm,
                other_tcp=other_tcp,
                lateral=lateral,
                span_m=profile.arms[arm].gripper.span_m,
            )
        except SpecError as error:  # the spec cannot be made on this scene: off, and why
            note = f"the spec cannot be made on the registered object: {error}"
            out.append(SpecGap(parameter=name, keypoint=where, object=obj, status="off", note=note))
            continue
        if not found:
            out.append(
                SpecGap(
                    parameter=name,
                    keypoint=where,
                    object=obj,
                    status="off",
                    note="the spec yields no grasp candidate on the registered object",
                )
            )
            continue
        # the expert's TCP, mapped from its object at the keypoint onto the object at the start
        at = poses.get(kp.frame, start)[obj]
        expert = start[obj].matrix() @ np.linalg.inv(at.matrix()) @ kp.tcp.matrix()
        rows = []
        for candidate in found:
            world = proxy.pose @ candidate.matrix
            offset = world[:3, 3] - expert[:3, 3]
            closing = _axis_deg(world[:3, 2], expert[:3, 1])
            approach = _angle(-world[:3, 1], expert[:3, 0])
            rows.append((float(np.linalg.norm(offset)), closing, approach, offset))
        aligned = [row for row in rows if row[1] <= config.contact_deg]
        distance, closing, approach, offset = min(aligned or rows, key=lambda row: row[0])
        off = distance > config.spec_m or closing > config.contact_deg
        # grasp_object refuses a spec whose every candidate closes on air before anything moves:
        # near the expert or not, the Actor could not make it as written
        air = all(candidate.air for candidate in found)
        x, y, z = (float(v) for v in offset)
        out.append(
            SpecGap(
                parameter=name,
                keypoint=where,
                object=obj,
                status="off" if off or air else "ok",
                candidates=len(found),
                offset=(x, y, z),
                distance_m=distance,
                closing_deg=closing,
                approach_deg=approach,
                note=AIR_NOTE if air else "",
            )
        )
    return out


def _fitted_at(source: str, keypoints: KeypointSet) -> tuple[Arm, Keypoint] | str:
    """The keypoint a spec was fitted at (its source, `demo <arm> K<n>`), or why there is none."""
    fitted = FITTED.fullmatch(source)
    if fitted is None or fitted[1] not in keypoints.arms:
        return "not fitted on the demonstration"
    arm, index = cast(Arm, fitted[1]), int(fitted[2])
    if index > len(keypoints.arms[arm]):
        return f"the demonstration has no {arm} K{index}"
    return arm, keypoints.arms[arm][index - 1]


def _held_at(
    keypoints: KeypointSet,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    arm: Arm,
    frame: int,
    released: Released | None = None,
) -> tuple[SceneRegistry, ObjectRecord | None]:
    """The registry replayed through the recorded grips to just before `frame` (a copy), and
    what `arm` holds then."""
    copy = registry.clone()
    replay_grips(keypoints, copy, robots, profile, None, before=frame, released=released)
    held = next((o for o in copy.objects.values() if arm in o.grasps), None)
    return copy, held


def place_check(
    name: str,
    spec: PlaceSpec,
    source: str,
    keypoints: KeypointSet,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState] | None,
    profile: EmbodimentProfile,
    config: ReviewConfig,
    call: Mapping[str, float | str],
    given: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    released: Released | None = None,
) -> SpecGap:
    """A placement spec on the demonstration's state at its keypoint (see the module); one fitted
    on the way to its arm's open (`demos.relation.release_after`: a standoff above the release)
    is checked at that open, where a placement belongs, and the note says so."""
    fitted = _fitted_at(source, keypoints)
    if isinstance(fitted, str):
        return _unchecked(name, None, None, fitted, "place")
    arm, kp = fitted
    where = fitted_where = f"{arm} K{kp.index}"
    anchored = ""
    if (opening := release_after(keypoints.arms[arm], kp)) is not None:
        kp, where = opening, f"{arm} {opening.name}"
        anchored = (
            f"fitted at {fitted_where}, on the way to the release: checked at the open, {where}, "
            "where a placement belongs"
        )
    if robots is None:
        return _unchecked(name, where, None, "the recorded robot is not given", "place")
    copy, held = _held_at(keypoints, registry, robots, profile, arm, kp.frame, released)
    if held is None:
        return _unchecked(name, where, None, f"the {arm} arm holds nothing there", "place")
    target: ObjectProxy | None = None
    named = ""
    if spec.target.kind == "landmark":
        return _unchecked(name, where, held.name, "a landmark target needs the table", "place")
    start = place_start(keypoints.arms[arm], kp)
    robot = _robot_at(robots, arm, start)
    if spec.target.kind == "object":
        assert spec.target.obj is not None  # the kind requires it
        record, named = target_record(spec, where, copy, (given or {}).get(fitted_where, ()))
        if record is None:
            note = (
                f"no object is registered as {spec.target.obj!r} on the demonstration's scene, "
                f"and no fit at {where} names its object: {named}"
            )
            return SpecGap(parameter=name, keypoint=where, object=held.name, status="off",
                           kind="place", note=note)  # fmt: skip
        target = ObjectProxy.from_geometry(object_geometry(record, robot))
    carried = ObjectProxy.from_geometry(object_geometry(held, robot))
    constrain = cast(Constrain, call["constrain"])
    try:
        distances = (float(call["approach_m"]), float(call["stop_back_m"]))
        options = placed_turns(
            spec, carried, target, arm, robot, profile, constrain=constrain, distances=distances
        )
        pre, place = options[0]
        gaps = placed_gaps(spec, carried, target, arm, robot, kp.tcp, profile)
    except (SpecError, LookupError) as error:
        note = f"the placement cannot be made on the demonstration's state: {error}"
        return SpecGap(parameter=name, keypoint=where, object=held.name, status="off",
                       kind="place", note=note)  # fmt: skip
    offset = np.subtract(place.p, kp.tcp.p)
    # as `placed_gaps` judges: a round object turned about its own axis is where it should be
    distance, angle = held_gap(place, kp.tcp, carried, robot.arms[arm].tcp)
    off = distance > config.place_m or angle > config.place_deg
    return SpecGap(
        parameter=name,
        keypoint=where,
        object=held.name,
        status="off" if off else "ok",
        kind="place",
        offset=(float(offset[0]), float(offset[1]), float(offset[2])),
        distance_m=distance,
        angle_deg=angle,
        constrain=constrain,
        constrains=gaps,
        arm=arm,
        targets=[pre, place],
        starts=[start.frame, start.frame],
        turns=[list(option) for option in options[1:]],
        note="; ".join(n for n in (anchored, named) if n),
    )


NOOP_M: Final = 0.001
NOOP_DEG: Final = 1.0
"""An edit that moves what `place_object` commands (its pre-place and place TCP) less than this
on the demonstration's state changes no behaviour (`placement_effect`)."""


def placement_effect(
    keypoints: KeypointSet,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    config: ReviewConfig,
    given: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    released: Released | None = None,
) -> Callable[[MemoryDoc, MemoryDoc, Any], str | None]:
    """`memory.Editor`'s `effect` for S1's Reviser: why an edit of a placement spec's fields
    cannot change behaviour, where the TCPs `place_object` commands with the workflow's call on
    the demonstration's state (`place_check`, before and after the edit) stay within `NOOP_M`
    and `NOOP_DEG` (a yaw in `rpy_deg` under constrain auto, which turns the object its own
    way); None where it moves them, or where the check cannot run."""

    def check(doc: MemoryDoc, name: str) -> SpecGap | None:
        param = doc.params.get(name)
        if not isinstance(param, PlaceSpecParam):
            return None
        try:
            read = as_read(param)
        except ValueError:
            return None
        assert isinstance(read, PlaceSpec)
        try:
            gap = place_check(name, read, param.source, keypoints, registry, robots, profile,
                              config, call_args(doc, name), given, released)  # fmt: skip
        except Exception:  # a guard, not the revision: an edit it cannot check goes through
            log.exception("the placement effect check of %s failed", name)
            return None
        return gap if len(gap.targets) == 2 else None

    def effect(before: MemoryDoc, after: MemoryDoc, op: Any) -> str | None:
        name = getattr(op, "param", None)
        if op.op not in ("set_number", "set_category") or not isinstance(name, str):
            return None
        if op.field in (None, "arm", "mirror"):  # the check places with the keypoint's arm
            return None
        old, new = check(before, name), check(after, name)
        if old is None or new is None:
            return None
        pairs = list(zip(old.targets, new.targets, strict=True))
        moved = max(float(np.linalg.norm(np.subtract(a.p, b.p))) for a, b in pairs)
        turned = max(float(quat_angle_deg(a.q, b.q)) for a, b in pairs)
        if moved >= NOOP_M or turned >= NOOP_DEG:
            return None
        return (
            f"this edit of {name}.{op.field} leaves what place_object commands with the "
            f"workflow's call (constrain {new.constrain}) on the demonstration's state where it "
            f"was ({1000 * moved:.1f} mm, {turned:.1f} deg): it cannot change behaviour (under "
            "constrain auto or free the placement turns the object its own way about the "
            "vertical, whatever yaw the spec holds); change the call's constrain, refit the "
            "spec at its keypoint (fit_spec), or edit what the review names"
        )

    return effect


def dry_run_gate(
    check: Callable[[MemoryDoc, Collection[str]], list[SpecGap]],
    passing: Collection[str],
) -> Callable[[MemoryDoc, MemoryDoc, Any], str | None]:
    """`memory.Editor`'s `effect` for S1's Reviser: an edit that turns a spec the last dry run
    found reproducing the expert (`passing`) into one it finds off (`check`: the dry run of the
    named specs on a document, unplanned) is refused, with the gap. A spec is re-checked when
    its value changes, or the call of a stage that passes it does; a placement an edited call
    no longer passes to `place_object` is not (it would be checked under the call's defaults, as
    no call passes it), the placements that call passes in its stead are, under its arguments;
    a check that cannot run lets the edit through (a guard, not the revision)."""

    def effect(before: MemoryDoc, after: MemoryDoc, op: Any) -> str | None:
        touched = [
            name
            for name in passing
            if name in after.params
            and (
                name not in before.params
                or jsonable(before.params[name].value) != jsonable(after.params[name].value)
                or _calls_of(before, name) != _calls_of(after, name)
            )
            # a placement no place_object call passes any more was swapped out, not regressed
            and not (_placed(before, name) and not _placed(after, name))
        ]
        # the placements an edited call passes now in place of one the dry run found reproducing,
        # checked under that call's own arguments
        swapped = [name for name in passing if _placed(before, name) and not _placed(after, name)]
        stages = {s.name for s in before.workflow for n in swapped if _passes(s.call, n, True)}
        touched += [
            name
            for stage in after.workflow
            if stage.name in stages
            for name in after.params
            if _passes(stage.call, name, True)
            and not _placed(before, name, stage.name)
            and name not in touched
        ]
        if not touched:
            return None
        try:
            gaps = check(after, touched)
        except Exception:  # a guard, not the revision: an edit it cannot check goes through
            log.exception("the dry-run gate on %s failed", touched)
            return None
        off = [gap for gap in gaps if gap.status == "off"]
        if not off:
            return None
        names = ", ".join(gap.parameter for gap in off)
        return (
            f"this edit turns {names} from reproducing the expert in the spec dry run (on the "
            "demonstration's state, as the stage's call commands it) to off:\n"
            + "\n".join(render_specs(off)[1:])
            + "\nkeep what reproduced the expert and change what the attempt's evidence names, "
            "or make the edit so that the dry run still holds"
        )

    return effect


def _passes(call: str, name: str, place: bool = False) -> bool:
    """Whether a stage's call passes parameter `name` (`place`: as the placement of a
    `place_object`, its `place` argument or third positional one)."""
    braced = f"{{{name}}}"
    if not place:
        return braced in call
    return any(
        braced in c.keywords.get("place", c.values[2] if len(c.values) > 2 else "")
        for c in parse_calls(call)
        if c.name == "place_object"
    )


def _placed(doc: MemoryDoc, name: str, stage: str | None = None) -> bool:
    """Whether a `place_object` call of the workflow (of `stage`, if given) passes `name`."""
    return any(
        _passes(s.call, name, True) for s in doc.workflow if stage is None or s.name == stage
    )


def _calls_of(doc: MemoryDoc, name: str) -> list[str]:
    """The calls of the stages whose call passes parameter `name` (`{name}`)."""
    return [stage.call for stage in doc.workflow if f"{{{name}}}" in stage.call]


def target_record(
    spec: PlaceSpec, where: str, registry: SceneRegistry, fits: Sequence[Mapping[str, Any]]
) -> tuple[ObjectRecord | None, str]:
    """The registered object a placement's target names, and how it was found (a note; empty
    when by its own name): the Memory names roles, the Demo Reader's registry its own names for
    the same objects (`medium_rack` for the role `rack`), so a target not registered under its
    name is the object the fits `demo_relation` gave at its keypoint name, the fit the spec
    equals but for that name first, else the one object they all name."""
    assert spec.target.obj is not None
    found = registry.named(spec.target.obj)
    if found is not None:
        return found, ""
    same, others = [], []
    for fit in fits:
        obj = (fit.get("target") or {}).get("obj")
        if obj is None or registry.named(obj) is None:
            continue
        (same if _alike(PlaceSpec.model_validate(fit_spec(fit)), spec) else others).append(obj)
    names = list(dict.fromkeys(same)) or list(dict.fromkeys(others))
    if len(names) != 1:
        why = f"they name {', '.join(names)}" if names else "none names a registered object"
        return None, why
    how = "the fit it equals but for the name" if same else "the object every fit there names"
    note = (
        f"its target {spec.target.obj!r} (the Memory's role) read as the registered "
        f"{names[0]!r}, {how} at {where}"
    )
    return registry.named(names[0]), note


def path_check(
    name: str,
    spec: PathSpec,
    source: str,
    keypoints: KeypointSet,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState] | None,
    profile: EmbodimentProfile,
    config: ReviewConfig,
    released: Released | None = None,
) -> SpecGap:
    """A held path on the demonstration's state from its close (see the module): each
    waypoint's commanded TCP (the path's motion in the body frame `follow_path` would choose,
    from where the expert's gripper held at the close) against the expert's keypoint it was
    fitted from (evenly picked where the path keeps fewer poses than keypoints)."""
    fitted = _fitted_at(source, keypoints)
    if isinstance(fitted, str):
        return _unchecked(name, None, None, fitted, "path")
    arm, kp = fitted
    where = f"{arm} K{kp.index}"
    if robots is None:
        return _unchecked(name, where, None, "the recorded robot is not given", "path")
    kps = keypoints.arms[arm]
    close = next((k for k in reversed(kps) if k.cue == "close" and k.frame < kp.frame), None)
    if close is None:
        return _unchecked(name, where, None, f"the {arm} arm closes on nothing before it", "path")
    after = next((k for k in kps if k.frame > close.frame), kp)
    copy, held = _held_at(keypoints, registry, robots, profile, arm, after.frame, released)
    turning = next((o for o in copy.objects.values() if arm in o.part_grasps), None)
    body = held or turning
    if body is None:
        return _unchecked(name, where, None, f"the {arm} arm holds nothing after its close",
                          "path")  # fmt: skip
    robot = _robot_at(robots, arm, close)
    still = body.model_copy(update={"part_grasps": {}, "part_poses": {}})
    proxy = ObjectProxy.from_geometry(object_geometry(still, robot))
    frame, _, _ = choose_frame(hold_frames(proxy, spec.x_toward), spec, [close.tcp])
    grip = close.tcp.matrix()
    targets = [Pose.from_matrix(world @ grip) for world in motions(frame, spec)]
    end = place_start(kps, kp) if kp.phase in PLACE_LEG else kp
    held_kps = [k for k in kps if close.frame <= k.frame <= end.frame]
    if len(held_kps) > len(spec.poses):  # as `held_path` keeps them
        picks = np.linspace(0, len(held_kps) - 1, len(spec.poses)).round().astype(int)
        held_kps = [held_kps[i] for i in picks]
    expert = held_kps[1:]
    if len(expert) != len(targets):
        expert = [held_kps[-1]] * len(targets)  # a path fitted elsewhere: its end only
    gaps = [
        (float(np.linalg.norm(np.subtract(t.p, e.tcp.p))), float(quat_angle_deg(t.q, e.tcp.q)))
        for t, e in zip(targets, expert, strict=True)
    ]
    worst = max(
        range(len(gaps)), key=lambda i: gaps[i][0] / config.place_m + gaps[i][1] / config.place_deg
    )
    distance, angle = gaps[worst]
    offset = np.subtract(targets[worst].p, expert[worst].tcp.p)
    off = distance > config.place_m or angle > config.place_deg
    starts = [close.frame, *(k.frame for k in expert[:-1])]
    return SpecGap(
        parameter=name,
        keypoint=where,
        object=body.name,
        status="off" if off else "ok",
        kind="path",
        offset=(float(offset[0]), float(offset[1]), float(offset[2])),
        distance_m=distance,
        angle_deg=angle,
        arm=arm,
        targets=targets,
        starts=starts,
        note=f"waypoint {worst + 1} of {len(targets)} the farthest",
    )


def _alike(a: PlaceSpec, b: PlaceSpec) -> bool:
    """Whether two placements are the same but for their target's object (numbers as a Memory
    line rounds them: 1 mm, 0.1 deg)."""

    def same(x: Any, y: Any, key: str = "") -> bool:
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(same(x[k], y[k], k) for k in x)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(same(u, v, key) for u, v in zip(x, y, strict=True))
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return abs(float(x) - float(y)) <= (0.11 if key.endswith("_deg") else 1.1e-3)
        return bool(x == y)

    dumped = [s.model_dump(mode="json") for s in (a, b)]
    for d in dumped:
        d["target"] = {k: v for k, v in d["target"].items() if k != "obj"}
    return same(*dumped)


def _robot_at(robots: Mapping[int, RobotState], arm: Arm, kp: Keypoint) -> RobotState:
    """The recorded robot at a keypoint's frame, the arm's TCP the keypoint's."""
    robot = robots[kp.frame]
    state = robot.arms[arm].model_copy(update={"tcp": kp.tcp})
    return robot.model_copy(update={"arms": {**robot.arms, arm: state}})


async def planned(
    gaps: Sequence[SpecGap],
    planner: Planner,
    profile: EmbodimentProfile,
    robots: Mapping[int, RobotState],
) -> list[SpecGap]:
    """`gaps` with their targets planned (`with_plans`), each from the recorded robot's
    commanded joints at its start frame."""
    out = []
    for gap in gaps:
        if not gap.targets or gap.arm is None:
            out.append(gap)
            continue
        first = await _verdicts(planner, profile, robots, gap.arm, gap.targets, gap.starts)
        if any(first):  # as the call does: the next option (an equivalent turn) that plans
            for option in gap.turns:
                found = await _verdicts(planner, profile, robots, gap.arm, option, gap.starts)
                if not any(found):
                    gap, first = gap.model_copy(update={"targets": option}), found
                    break
        out.append(with_plans(gap, first))
    return out


async def _verdicts(
    planner: Planner,
    profile: EmbodimentProfile,
    robots: Mapping[int, RobotState],
    arm: Arm,
    targets: Sequence[Pose],
    starts: Sequence[int],
) -> list[str | None]:
    """The planner's verdict on each TCP target from the recorded robot's commanded joints at its
    start frame (None where it plans)."""
    spec = profile.arms[arm]
    base = spec.planner.base if spec.planner is not None else ORIGIN  # as the executor
    found: list[str | None] = []
    for tcp, frame in zip(targets, starts, strict=True):
        start = robots[frame].qpos_cmd[spec.action[0] : spec.action[1]]
        request = BatchPlanRequest(
            arm=arm, base=base, start=start, goals=[profile.ee_pose(arm, tcp)]
        )
        result = (await planner.plan_batch(request))[0]
        found.append(None if result.reason == "ok" else f"{result.reason}: {result.detail}"[:120])
    return found


def with_plans(gap: SpecGap, plans: Sequence[str | None]) -> SpecGap:
    """`gap` with the planner's verdicts on its targets: off where one does not plan."""
    failed = [(i, p) for i, p in enumerate(plans) if p is not None]
    if not failed:
        return gap.model_copy(update={"plans": list(plans)})
    i, reason = failed[0]
    what = ("the pre-place", "the place")[i] if gap.kind == "place" else f"waypoint {i + 1}"
    note = "; ".join(
        filter(None, [gap.note, f"{what} does not plan from the expert's state ({reason})"])
    )
    return gap.model_copy(update={"plans": list(plans), "status": "off", "note": note})


def _other_tcp(
    keypoints: KeypointSet,
    tcps: Mapping[int, Mapping[Arm, Sequence[float]]] | None,
    arm: str,
    frame: int,
) -> np.ndarray | None:
    """The other gripper's TCP (world) when `arm` is at its keypoint at `frame`: as recorded
    then (`tcps`), else at the other arm's last keypoint up to then; None when neither knows."""
    recorded = {a: p for a, p in (tcps or {}).get(frame, {}).items() if a != arm}
    if recorded:
        return np.asarray(next(iter(recorded.values())), dtype=float)
    others = [
        k.tcp.p
        for other, ks in keypoints.arms.items()
        if other != arm
        for k in ks
        if k.frame <= frame
    ]
    return np.asarray(others[-1], dtype=float) if others else None


def registered(
    registry: SceneRegistry, at: np.ndarray, shape: ObjectShape | None
) -> ObjectRecord | None:
    """The registered object that is the object centered at `at` (with `shape`) at the start
    (`diagnosis.match`), or None."""
    records = {record.id: record for record in registry.objects.values()}
    found = match([np.asarray(at, dtype=float)], [truth_box(np.asarray(at), shape)], records)
    return records[found[0]] if found else None


def _axis_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two lines (an axis either way)."""
    cos = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return math.degrees(math.acos(min(1.0, cos)))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _unchecked(
    name: str,
    where: str | None,
    obj: str | None,
    note: str,
    kind: Literal["grasp", "place", "path"] = "grasp",
) -> SpecGap:
    return SpecGap(
        parameter=name, keypoint=where, object=obj, status="unchecked", note=note, kind=kind
    )


def unreproducible(gaps: Sequence[SpecGap], config: ReviewConfig) -> dict[str, str]:
    """The placement specs no `constrain` reproduces on the demonstration's state (each one's TCP
    at `stop_back_m` 0 beyond `place_m` or `place_deg` of the expert's), with those gaps as text:
    what the review tells the Reviser to refit rather than tune (`expert_diff.review`)."""
    out = {}
    for gap in gaps:
        if gap.kind != "place" or not gap.constrains:
            continue
        if all(m > config.place_m or d > config.place_deg for m, d in gap.constrains.values()):
            out[gap.parameter] = ", ".join(
                f"{c} {100 * m:.1f} cm {d:.0f} deg" for c, (m, d) in gap.constrains.items()
            )
    return out


def dry_run_failed(error: str) -> str:
    """The Reviser's line for a spec dry run that failed: no spec was checked, which is no pass."""
    return (
        f"Spec dry run before this attempt: it failed ({error[:300]}), so no spec was checked "
        "against the expert; this is a fault of the check, not of a spec, and says nothing for "
        "or against any of them."
    )


def render_specs(gaps: Sequence[SpecGap]) -> list[str]:
    """The dry run as the Reviser reads it: each spec off the expert, with the gap."""
    lines = [
        "Spec dry run before this attempt (simulator truth, S1 only): each grasp spec "
        "synthesized on perception's registry of the demonstration's start scene (where it was "
        "fitted), against the expert's grasp it was fitted at; each placement and held path as "
        "its call would command it on the demonstration's state at its keypoint (the object "
        "where the expert's grasp held it), against the expert's TCP there, and planned from the "
        "expert's own state. A spec off here was off before the attempt ran: move it toward the "
        "expert where the attempt's evidence agrees (a placement no constrain reproduces, or "
        "one that does not plan, needs refitting at its keypoint with demo_relation)."
    ]
    for gap in gaps:
        head = f"- {gap.parameter} ({gap.keypoint or 'no keypoint'}"
        head += f" on {gap.object})" if gap.object else ")"
        if gap.status == "unchecked":
            lines.append(f"{head}: not checked, {gap.note}.")
        elif gap.distance_m is None:
            lines.append(f"{head}: off, {gap.note}.")
        elif gap.kind != "grasp":
            lines.append(f"{head}: {gap.status}; {_commanded_text(gap)}")
        else:
            assert gap.offset is not None
            cm = ", ".join(f"{100 * v:+.1f}" for v in gap.offset)
            lines.append(
                f"{head}: {gap.status}; its best of {gap.candidates} candidates lies ({cm}) cm = "
                f"{100 * gap.distance_m:.1f} cm from the expert's TCP (robot axes), closing axis "
                f"{gap.closing_deg:.0f} deg off, approach {gap.approach_deg:.0f} deg off"
                + (f"; {gap.note}." if gap.note else ".")
            )
    return lines


def _commanded_text(gap: SpecGap) -> str:
    """A placement's or held path's line: its commanded TCP off the expert's, how it plans."""
    assert gap.offset is not None and gap.distance_m is not None and gap.angle_deg is not None
    cm = ", ".join(f"{100 * v:+.1f}" for v in gap.offset)
    what = (
        f"place_object with constrain {gap.constrain} commands the TCP"
        if gap.kind == "place"
        else "follow_path commands the TCP (its farthest waypoint)"
    )
    text = (
        f"{what} ({cm}) cm = {100 * gap.distance_m:.1f} cm and {gap.angle_deg:.0f} deg off the "
        "expert's (robot axes)"
    )
    if gap.constrains:
        shown = ", ".join(
            f"{c} {100 * m:.1f} cm {d:.0f} deg" for c, (m, d) in gap.constrains.items()
        )
        text += f"; at stop_back_m 0 by constrain: {shown}"
    if gap.note:
        text += f"; {gap.note}"
    return text + "."
