"""The reuse step (reuse before writing): for each gap of a task's coverage, the other calls the
entries' fitters give for its segment, each run as the coverage step explains a segment
(`run.explain`), in order, until one explains it or the gap's budget (`TRIES`) is spent.

A grasp's alternatives are the fitted forms other than the pose form (`fit_grasp`: an anchor on
the object's box, then its plain form) and the entry's own candidates (`GraspSpec()`); a place's
or a carry's, first the place where the demonstration let go of the object, made as it was
(`released_place`: an operator lets go over a bin, a holder or a slot and the object drops in,
so its resting point is no pose the hand held it at), then a drop where the object rests inside a
container below its top (released over the container's centre, `DROP_M` above its rim: an
operator drops into a bin from beyond the table's edge), every other form `fit_place` gives, then
its plain form; a touch's, a press at the demonstration's pose (`fit_grasp_pose`, purpose `press`),
and a hold of it (purpose `pull`) that then follows the demonstration's held path (`fit_path` on the
TCP through the hold, `follow_path`) and opens: either explains it where the object ends, at the
hold's end, where and as turned the demonstration's did (the path first where the demonstration's
hold moved it).

After the core's, each written skill (`../builder/README.md`) whose `KINDS` hold the gap's
kind is one more alternative, fitted by its own `fit` (none where it returns None); `TRIES` stays
the core's budget.
"""

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm
from roboscript.demos import gt_scene
from roboscript.embodiments.profile import PoseConventions
from roboscript.manipulation.api import placement
from roboscript.manipulation.fit import fit_grasp, fit_grasp_pose, fit_place
from roboscript.manipulation.paths import fit_path
from roboscript.manipulation.posemath import Constrain
from roboscript.manipulation.proxy import ObjectProxy
from roboscript.manipulation.specs import MAX_PATH_POSES, GraspSpec, PlaceSpec
from roboscript.spatial.rotation import quat_angle_deg
from roboscript_methods.skill_builder.demo_coverage.fitting import (
    AbsoluteParams,
    Call,
    bottom,
    fit_error,
    fit_written,
)
from roboscript_methods.skill_builder.demo_coverage.run import SETTLE_FRAMES
from roboscript_methods.skill_builder.demo_coverage.segments import CutConfig, Recording, Segment

if TYPE_CHECKING:
    from roboscript_methods.skill_builder.builder.plugins import WrittenSkill

LOG = logging.getLogger(__name__)

TRIES: Final = 4
"""Alternatives tried per gap, at most: a fixed budget, the same for every gap."""
DROP_M: Final = 0.05
"""How far over a container's top an object resting inside it is released (a drop form)."""
CONSTRAINS: Final[tuple[Constrain, ...]] = ("auto", "align", "free")
"""`place_object`'s `constrain` values, in the order a released place's is chosen
(`constrained`)."""
REPRODUCED_M: Final = 0.03
REPRODUCED_DEG: Final = 20.0
"""A commanded ee this near the demonstration's reproduces it: S1's placement tolerances, those
its Demo Reader chooses a placement's `constrain` by (`wicl.relation.REPRODUCED`)."""
PLACE_APPROACH_M: Final = 0.1
"""A released place's pre-place distance (`place_object`'s default), along the demonstration's
approach (`approached`)."""
HELD_MOTION_M: Final = 0.01
HELD_MOTION_DEG: Final = 5.0
"""A touched object that moved more than this while held was moved by the hold (a push, a
turn): its held path is tried before a press."""


def alternatives(
    call: Call,
    recording: Recording,
    scene: gt_scene.Scene,
    base: Pose,
    conventions: PoseConventions,
    config: CutConfig,
    span_m: float | None,
    skills: Sequence["WrittenSkill"] = (),
    refused: list[dict[str, Any]] | None = None,
) -> list[Call]:
    """The calls to try for `call`'s segment besides `call` itself: the core's, at most `TRIES`,
    then one per written skill that fits it. A written fit that raises is skipped, logged: one
    skill's bug must not stop the round's rerun (the gate's cases report it); one the relative
    rule refused (`fitting.AbsoluteParams`) is also added to `refused` (`{segment, skill, why}`):
    the next round's builder rewrites it (`builder.clusters`, class `fit`)."""
    found = core_alternatives(call, recording, scene, base, conventions, config, span_m)
    for skill in skills:
        if call.segment.kind not in skill.kinds:
            continue
        try:
            fitted = fit_written(skill, call.segment, recording, scene, base, conventions)
        except Exception as error:  # the Agent's code
            LOG.warning("%s's fit on %s: %s: %s", skill.name, call.segment.name,
                        type(error).__name__, error)  # fmt: skip
            if isinstance(error, AbsoluteParams) and refused is not None:
                row = {"segment": call.segment.name, "skill": skill.name, "why": fit_error(error)}
                if row not in refused:
                    refused.append(row)
            continue
        if fitted is not None:
            found.append(fitted)
    return found


def core_alternatives(
    call: Call,
    recording: Recording,
    scene: gt_scene.Scene,
    base: Pose,
    conventions: PoseConventions,
    config: CutConfig,
    span_m: float | None,
) -> list[Call]:
    """The core entries' calls to try for `call`'s segment besides `call` itself, at most
    `TRIES`."""
    s = call.segment
    if s.obj is None or s.obj not in scene.boxes:
        return []
    found: list[Call] = []
    if s.kind in ("grasp", "touch"):
        made = s.rest if s.kind == "grasp" and s.rest is not None else s.end
        proxy = scene.at(recording.poses(made), {}).proxy(s.obj)
        tcp = Pose.from_list(recording.demo.arms[s.arm].tcp[made].tolist())
        other = None
        if s.handover:
            giver = next(a for a in recording.demo.arms if a != s.arm)
            other = recording.demo.arms[giver].tcp[made, :3]
        if s.kind == "touch":
            press = fit_grasp_pose(
                s.tcp, proxy, purpose="press", arm=s.arm, other_tcp=other, base=base.matrix()
            )
            pressed = Call(s, "grasp_object", grasp=press.spec, form=f"press {press.variant}")
            pull = fit_grasp_pose(
                s.tcp, proxy, purpose="pull", arm=s.arm, other_tcp=other, base=base.matrix()
            )
            held = fit_path(hold_tcps(recording, s), proxy, arm=s.arm)
            followed = Call(
                s,
                "grasp_object",
                grasp=pull.spec,
                form=f"held path {pull.variant}",
                residual_m=held.position_error_m,
                residual_deg=held.angle_error_deg,
                path=held.spec,
            )
            found += [followed, pressed] if moved(recording, s) else [pressed, followed]
        else:
            for plain in (False, True):
                fitted = fit_grasp(
                    tcp,
                    proxy,
                    arm=s.arm,
                    other_tcp=other,
                    base=base.matrix(),
                    plain=plain,
                    span_m=span_m,
                )
                form = f"{'plain ' if plain else ''}{fitted.variant}"
                found.append(
                    Call(
                        s,
                        "grasp_object",
                        grasp=fitted.spec,
                        form=form,
                        residual_m=fitted.position_error_m,
                        residual_deg=fitted.angle_error_deg,
                    )
                )
            found.append(Call(s, "grasp_object", grasp=GraspSpec(arm=s.arm), form="candidates"))
    elif s.kind in ("place", "carry") and s.rest is not None:
        found.append(released_place(s, recording, scene, base, conventions))
        held = scene.at(recording.poses(max(s.end - 1, 0)), {}).proxy(s.obj)
        rested = scene.at(recording.poses(s.rest), {})
        name = s.target  # the call's, as `fitting.fit` chose it (or an earlier step did)
        target = None if name is None else rested.proxy(name)
        placed = rested.proxy(s.obj).pose
        common = {"held_now": held.pose, "arm": s.arm, "conventions": conventions}
        forms = fit_place(placed, held, target, base, **common)
        if target is not None and (drop := dropped(rested, s.obj, name)) is not None:
            over = fit_place(drop, held, target, base, **common)  # released over its rim
            forms = {f"drop {k}": v for k, v in over.items()} | forms
        forms |= {
            f"plain {k}": v
            for k, v in fit_place(placed, held, target, base, plain=True, **common).items()
        }
        for form, fitted in forms.items():
            if form == call.form:
                continue
            found.append(
                Call(
                    s,
                    "place_object",
                    place=fitted.spec,
                    form=form,
                    residual_m=fitted.position_error_m,
                    residual_deg=fitted.angle_error_deg,
                )
            )
    return found[:TRIES]


def released_place(
    s: Segment,
    recording: Recording,
    scene: gt_scene.Scene,
    base: Pose,
    conventions: PoseConventions,
) -> Call:
    """A place's or a carry's call made as the demonstration made it: the held object's pose where
    the demonstration let go of it, from its target (the first form the demonstration bears
    out), under the `constrain` that commands the ee it let go from, held as it was where the
    place starts (`run.explain`'s replay: `SETTLE_FRAMES` after the close; `constrained`), its
    pre-place `PLACE_APPROACH_M` back along the way the demonstration's ee came onto the release
    (`approached`)."""
    assert s.obj is not None and s.rest is not None  # a place's
    let_go, begun = max(s.end - 1, 0), min(s.start + SETTLE_FRAMES, s.end)
    at = scene.at(recording.poses(let_go), {})
    held = at.proxy(s.obj)
    name = s.target  # the call's, as `fitting.fit` chose it (or an earlier step did)
    target = None if name is None else at.proxy(name)
    forms = fit_place(
        held.pose, held, target, base, held_now=held.pose, arm=s.arm, conventions=conventions
    )
    form, fitted = next(
        ((n, f) for n, f in forms.items() if f.consistent), next(iter(forms.items()))
    )
    ees = recording.demo.arms[s.arm].ee
    start = scene.at(recording.poses(begun), {}).proxy(s.obj)
    holding = Pose.from_list(ees[begun].tolist())
    expert = Pose.from_list(ees[let_go].tolist())
    common = {"arm": s.arm, "holding": holding, "base": base, "conventions": conventions}
    constrain, gaps = constrained(fitted.spec, start, target, expert=expert, **common)
    path = ees[s.start : let_go + 1]
    axis = approached(fitted.spec, start, target, path=path, constrain=constrain, **common)
    return Call(
        s,
        "place_object",
        place=fitted.spec,
        form=f"released {form}",
        residual_m=fitted.position_error_m,
        residual_deg=fitted.angle_error_deg,
        constrain=constrain,
        approach_m=PLACE_APPROACH_M,
        approach_axis=axis,
        gaps=gaps,
    )


def constrained(
    spec: PlaceSpec,
    held: ObjectProxy,
    target: ObjectProxy | None,
    *,
    arm: Arm,
    holding: Pose,
    expert: Pose,
    base: Pose,
    conventions: PoseConventions,
) -> tuple[Constrain, dict[str, tuple[float, float]]]:
    """The `constrain` under which `place_object` with `spec`, the arm's ee at `holding` holding
    `held`, commands at `stop_back_m` 0 the ee the demonstration let go from (`expert`;
    `manipulation.api.placement`: what the call computes, its first option): the first in
    `CONSTRAINS` order within `REPRODUCED_M` and `REPRODUCED_DEG`, else the one nearest in angle
    (`placement` turns the object by its own rule under `auto`, aligned with the arm's
    diagonal or the world's y, and by the least turn from how it is held under `align`); with how
    far (m, deg) each one commands it. A round object (its shape looks the same turned about its
    own z: `continuous`) is compared where each ee puts it, by its position and the tilt of its
    z: a turn about its own axis leaves it placed the same (S1's rule,
    `wicl.relation.held_gap`)."""
    lateral = base.matrix()[:3, 1]
    grip = np.linalg.inv(holding.matrix()) @ held.pose
    round_ = (held.shape_symmetry or held.symmetry) == "continuous"
    gaps = {}
    for constrain in CONSTRAINS:
        found = placement(
            held,
            spec,
            target,
            arm,
            holding,
            base,
            lateral,
            constrain=constrain,
            conventions=conventions,
        )
        commanded = found.ee(found.centers[0], found.turns[0], 0, 0.0)
        if round_:
            ours, theirs = commanded.matrix() @ grip, expert.matrix() @ grip
            cos = float(np.clip(ours[:3, 2] @ theirs[:3, 2], -1.0, 1.0))
            away = float(np.linalg.norm(ours[:3, 3] - theirs[:3, 3]))
            angle = math.degrees(math.acos(cos))
        else:
            away = float(np.linalg.norm(np.subtract(commanded.p, expert.p)))
            angle = float(quat_angle_deg(commanded.q, expert.q))
        gaps[constrain] = (round(away, 4), round(angle, 2))
    near = [c for c in CONSTRAINS if gaps[c][0] <= REPRODUCED_M and gaps[c][1] <= REPRODUCED_DEG]
    return (near[0] if near else min(CONSTRAINS, key=lambda c: gaps[c][1])), gaps


def approached(
    spec: PlaceSpec,
    held: ObjectProxy,
    target: ObjectProxy | None,
    *,
    arm: Arm,
    holding: Pose,
    path: np.ndarray,
    base: Pose,
    conventions: PoseConventions,
    constrain: Constrain,
) -> Literal["grasp"] | tuple[float, float, float]:
    """The way the demonstration's ee came onto the release (`path`: its ee poses up to the
    release, last), as `place_object`'s `approach_axis`: from its last pose at least
    `PLACE_APPROACH_M` back from the release toward the release, in the frame of the placement's
    target pose (the first option's centre, as the call computes it, `holding` holding `held`);
    `grasp` (`place_object`'s default, back along the gripper's approach) where the path never lies
    that far back or the spec moves along its own `approach`. The default backs a hold above a bin
    or a slot up out of the arm's reach where the operator came on from the side."""
    release = path[-1, :3]
    far = next((p for p in path[::-1] if np.linalg.norm(p[:3] - release) >= PLACE_APPROACH_M), None)
    if far is None or spec.approach is not None:
        return "grasp"
    lateral = base.matrix()[:3, 1]
    found = placement(
        held,
        spec,
        target,
        arm,
        holding,
        base,
        lateral,
        constrain=constrain,
        conventions=conventions,
    )
    onto = release - far[:3]
    local = found.centers[0].matrix()[:3, :3].T @ (onto / np.linalg.norm(onto))
    return (round(float(local[0]), 4), round(float(local[1]), 4), round(float(local[2]), 4))


def dropped(scene: gt_scene.Scene, held: str, container: str | None) -> np.ndarray | None:
    """Where `held` would be released into `container`, if it rests inside it below its top:
    its pose moved over the container's centre, its bottom `DROP_M` above the top."""
    if container is None:
        return None
    box = scene.proxy(container)
    frame = box.pose @ box.whole.frame
    top = float(frame[2, 3] + np.abs(frame[2, :3]) @ box.whole.size / 2)
    pose = np.array(scene.proxy(held).pose, dtype=float)
    low = float(bottom(scene, held)[2])
    if low >= top:
        return None
    pose[:2, 3] += frame[:2, 3] - bottom(scene, held)[:2]
    pose[2, 3] += top + DROP_M - low
    return pose


def hold_tcps(recording: Recording, s: Segment) -> list[Pose]:
    """The demonstration's TCP through a touch's hold, from the close to just before its open:
    about eight poses, at most `MAX_PATH_POSES`."""
    assert s.rest is not None
    tcp = recording.demo.arms[s.arm].tcp
    last = max(s.end + 1, s.rest - 1)
    frames = sorted({s.end, *range(s.end, last, max(1, (last - s.end) // 8)), last})
    if len(frames) > MAX_PATH_POSES:
        keep = np.linspace(0, len(frames) - 1, MAX_PATH_POSES).round().astype(int)
        frames = [frames[i] for i in keep]
    return [Pose.from_list(tcp[f].tolist()) for f in frames]


def moved(recording: Recording, s: Segment) -> bool:
    """Whether the touched object moved or turned during the hold (`HELD_MOTION_M`, `_DEG`)."""
    assert s.obj is not None and s.rest is not None
    track = recording.objects[s.obj]
    a, b = track[s.end], track[s.rest]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    turn = float(quat_angle_deg(a[3:], b[3:]))
    return float(np.linalg.norm(b[:3] - a[:3])) > HELD_MOTION_M or turn > HELD_MOTION_DEG


def unexplained(report: dict[str, Any]) -> set[str]:
    """`report`'s segments whose call did not explain them (an entry that failed, a fit out of
    tolerance)."""
    return {
        row["segment"]
        for row in report["explained"]
        if not row["explained"] and row["cause"] in ("entry fails", "fit out of tolerance")
    }


def gap_segments(report: dict[str, Any], calls: list[Call]) -> list[Call]:
    """The calls of `report`'s gaps a reuse can try: segments no skill fits (a touch; one whose
    object truth has no box for has no core alternative, a written skill may fit it) and those
    whose call did not explain them (an entry that failed, a fit out of tolerance)."""
    found = unexplained(report)
    return [c for c in calls if c.segment.name in found or c.skill is None]


def gap_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    """`gap_segments`' calls as `report` recorded them, in the same order (a sweep shards a
    task's reuse by these without a scene)."""
    found = unexplained(report)
    return [c for c in report["calls"] if c["segment"] in found or c["skill"] is None]
