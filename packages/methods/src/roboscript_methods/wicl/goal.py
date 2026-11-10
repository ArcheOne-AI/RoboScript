"""The demonstration's end as wicl's done check compares a scene with it (`DemoGoal`, the
method's own shape, which the core passes to wicl's finish gate unread: `read_goal` is its
`MethodRuntime.goal`, which the P3 host and `evaluate` call; S1 and S2 build theirs from the same
helpers):
its final views, press and frame, its end targets told which arm holds what and what was still
moving into the success latch, the lifts that latched it and which way to keep moving at a
finish short of the latch. From the demo package's recorded proprioception and views, no
object ground truth."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from roboscript.blobs import BlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import Arm
from roboscript.demos import KeypointSet, frame_alias
from roboscript.embodiments import EmbodimentProfile
from roboscript.learning.demo import DemoPackage, demo_robots
from roboscript.manipulation import robot_base
from roboscript.perception.criteria import EndTarget
from roboscript_methods.wicl.perception.done import ANY_LIFT
from roboscript_methods.wicl.perception.endstate import (
    MOVED_M,
    annotate_holds,
    annotate_motion,
)
from roboscript_methods.wicl.relation import empty_rests


@dataclass(frozen=True)
class DemoGoal:
    """The demonstration's end, as the done check compares a scene with it (`finish.DoneGate`):
    its final state by camera (`views`; None: not known), whether it pushed with empty fingers
    (`press`), its final frame (`reference`, whose table a lift is measured against), its end as
    targets (`end`), the roles whose lift latched its success (`latched`), the arms still moving
    into its latch with that motion (`latch`) and which way to keep moving at a finish short of
    the latch (`hint`, the one field the core reads: `agent.host.OpaqueGoal`). The defaults are
    an end the gate knows nothing of."""

    views: Mapping[str, BlobRef] | None = None
    press: bool = False
    reference: BlobRef | None = None
    end: Sequence[EndTarget] = ()
    latched: Sequence[str] = ()
    latch: Mapping[Arm, Vec3] = field(default_factory=dict)
    hint: str = ""


def demo_goal(value: object) -> DemoGoal:
    """`value` (an episode's or a demo view's goal, which the core passes through) as wicl's
    `DemoGoal`: one that knows nothing for None; any other shape is an error."""
    if value is None:
        return DemoGoal()
    if not isinstance(value, DemoGoal):
        raise TypeError(f"wicl's finish gate reads a DemoGoal, not {type(value).__name__}")
    return value


async def read_goal(
    package: DemoPackage,
    blobs: BlobStore,
    profile: EmbodimentProfile,
    end: Sequence[EndTarget],
) -> DemoGoal:
    """`package`'s end (its blobs in `blobs`), with `end` (its end as targets; none: not read)
    told which arm holds what (`holds_known`, read with `profile`)."""
    views = await end_views(package, blobs)
    press = await ends_pressed(package, blobs, profile)
    reference = end_observation(package)
    known = holds_known(end, package, profile) if end else []
    return DemoGoal(
        views=views,
        press=press,
        reference=reference,
        end=tuple(known),
        latched=tuple(latched_lifts(package, known)),
        latch=latch_motion(package.keypoints),
        hint=latch_hint(package.keypoints),
    )


def end_observation(package: DemoPackage) -> BlobRef:
    """The demonstration's final frame (its last frame, the success frame its recording stops
    at, the goal keyframe) as its `Observation`'s blob: where `check_done` finds the
    demonstration's table (`worldstate.DoneFact.reference`)."""
    final = package.keypoints.success_frame
    assert final is not None  # the package requires a success frame
    alias = frame_alias(final)
    entry = next((entry for entry in package.frames if entry.alias == alias), None)
    if entry is None:
        raise ValueError(f"the package of {package.task} lacks its success frame {alias}")
    return entry.observation


def held_ends(keypoints: KeypointSet) -> dict[Arm, tuple[Vec3, Vec3]]:
    """Each arm closed at the demonstration's end: its TCP (world) at its last closing keypoint
    and at its last keypoint (`perception.endstate.annotate_holds`). Recorded proprioception, no
    object ground truth."""
    out: dict[Arm, tuple[Vec3, Vec3]] = {}
    for arm, kps in keypoints.arms.items():
        if not kps or kps[-1].gripper != "closed":
            continue
        closes = [kp for kp in kps if kp.cue == "close"]
        close = closes[-1] if closes else kps[-1]
        out[arm] = (close.tcp.p, kps[-1].tcp.p)
    return out


def holds_known(
    targets: Sequence[EndTarget], package: DemoPackage, profile: EmbodimentProfile
) -> list[EndTarget]:
    """`targets` (the demonstration's end, `read_end_targets`) with each base target told the
    arm that holds its object at the end and whether it carried it there, from the package's
    keypoints and the robot's base (`perception.endstate.annotate_holds`), and the targets of a
    role an arm was still carrying into the success latch given that motion
    (`moving_at_latch`, `perception.endstate.annotate_motion`)."""
    base = robot_base(profile).matrix()
    held = annotate_holds(targets, held_ends(package.keypoints), base)
    return annotate_motion(held, moving_at_latch(package.keypoints))


RISING_M: Final = 0.005
"""An arm rises into the success latch when its last keypoint is this much higher than the one
before (m)."""


def moving_at_latch(keypoints: KeypointSet) -> dict[Arm, Vec3]:
    """The arms closed on something whose last keypoint is the success frame, each with its TCP
    displacement (world) from its keypoint before: how it was still moving what it holds when
    the official check latched (its recording stops there). Recorded proprioception, no object
    ground truth."""
    final = keypoints.success_frame
    out: dict[Arm, Vec3] = {}
    for arm, kps in keypoints.arms.items():
        if final is None or len(kps) < 2 or kps[-1].frame != final or kps[-1].gripper != "closed":
            continue
        last, before = kps[-1].tcp.p, kps[-2].tcp.p
        out[arm] = (last[0] - before[0], last[1] - before[1], last[2] - before[2])
    return out


LATCH_STEP_M: Final = 0.01
"""How far `latch_hint` asks each step along the motion into the latch to go: perception's
error on where an object is (`perception.criteria.TOL_FLOOR_M`), what the end's targets cannot
resolve."""


def latch_hint(keypoints: KeypointSet) -> str:
    """What S1's official finish refusal adds when the demonstration was still moving what an
    arm holds as its success latched (`moving_at_latch`, at least `MOVED_M`): a done check that
    passed is necessary, not sufficient (its end is the least that passes, and a hold that sits
    otherwise in the hand can leave the object short of it while every target passes), so keep
    going that way in `LATCH_STEP_M` steps; "" without such a motion. Recorded proprioception,
    no object ground truth."""
    parts = []
    for arm, (dx, dy, dz) in sorted(moving_at_latch(keypoints).items()):
        size = math.sqrt(dx * dx + dy * dy + dz * dz)
        if size < MOVED_M:
            continue
        ux, uy, uz = (v / size for v in (dx, dy, dz))
        step = [LATCH_STEP_M * v for v in (ux, uy, uz)]
        parts.append(
            f"the {arm} arm, and what it holds, the way the demonstration's {arm} arm was still "
            "moving "
            f"(world direction x {ux:+.2f}, y {uy:+.2f}, z {uz:+.2f}; a step of "
            f"{LATCH_STEP_M:g} m is x {step[0]:+.3f}, y {step[1]:+.3f}, z {step[2]:+.3f})"
        )
    if not parts:
        return ""
    return (
        ". A passing done check is necessary, not sufficient: the demonstration stopped the "
        "moment its success latched, so its end is the least that passes, and an object that "
        "sits a little otherwise in the hand can stay short of the latch while every target "
        "passes. Do not stop or move back: keep moving "
        + "; and ".join(parts)
        + ", one step at a time (a world move along all of that direction, such as nudge's x, y "
        "and lift for a held object; not a lift alone unless the direction is mostly up), "
        "finishing again after each"
    )


def latch_motion(keypoints: KeypointSet) -> dict[Arm, Vec3]:
    """The arms still moving what they hold into the success latch (`moving_at_latch`) by at
    least `MOVED_M`, each with that motion (world, m): what `check_done` tells an Actor stuck on
    end targets to try (`worldstate.DoneFact.latch`), since the end pose cannot show a strike or
    a push. Recorded proprioception, no object ground truth."""
    return {
        arm: d
        for arm, d in moving_at_latch(keypoints).items()
        if math.sqrt(sum(v * v for v in d)) >= MOVED_M
    }


def rising_at_latch(keypoints: KeypointSet) -> list[Arm]:
    """The arms moving into the success latch (`moving_at_latch`) higher, by `RISING_M`, than
    their keypoint before: still lifting when the official check latched."""
    return [arm for arm, d in moving_at_latch(keypoints).items() if d[2] >= RISING_M]


def latched_lifts(package: DemoPackage, targets: Sequence[EndTarget]) -> list[str]:
    """The roles whose lift caused the demonstration's success latch: held at its end by an arm
    still rising into the latch (`rising_at_latch`), by `targets`' base targets as
    `holds_known` annotated them (a lifted object stands over no anchor, so it has one). Their
    heights there are the threshold, not past it, so their lifts are minimums
    (`worldstate.DoneFact.latched`). With an arm rising but no base target that knows its arm
    (no targets read), the role is unknown: `ANY_LIFT`, every lift in the air."""
    rising = set(rising_at_latch(package.keypoints))
    if not rising:
        return []
    if not any(t.kind == "base" and t.arm is not None for t in targets):
        return [ANY_LIFT]
    return sorted({t.a for t in targets if t.kind == "base" and t.arm in rising})


async def end_views(package: DemoPackage, blobs: BlobStore) -> dict[str, BlobRef]:
    """The demonstration's final state as each camera saw it (its last frame, the success frame
    its recording stops at, the goal keyframe: what done looks like), by camera, from a store
    the package's blobs were imported into."""
    seen = Observation.model_validate_json(await blobs.aget_bytes(end_observation(package)))
    return {name: frame.rgb for name, frame in seen.cameras.items()}


async def ends_pressed(package: DemoPackage, blobs: BlobStore, profile: EmbodimentProfile) -> bool:
    """Whether the demonstration pushed something with empty fingers (press, click, toggle), by
    its own proprioception (`demos.relation.empty_rests` up to the success frame): a goal that
    may barely show in the final views, which `check_done` then also wants the episode's own
    press for. From a store the package's blobs were imported into."""
    final = package.keypoints.success_frame
    assert final is not None  # the package requires a success frame
    arms = package.keypoints.arms.values()
    frames = {kp.frame for kps in arms for kp in kps if kp.frame <= final}
    robots = await demo_robots(package, blobs, {*frames, final})
    missing = sorted({*frames, final} - robots.keys())
    if missing:
        raise ValueError(f"the package of {package.task} lacks its keypoint frame {missing[0]}")
    return bool(empty_rests(package.keypoints, robots, profile, final=final))
