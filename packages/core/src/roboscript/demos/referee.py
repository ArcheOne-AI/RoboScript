"""The keypoint referee, S1's judge side: how an executed path went through a demo's keypoints.

Per arm and in order, the current target passes once the TCP enters the referee's tolerance
(`RefereeConfig`), a position (the gripper's orientation is reported, never required, except at
a turn, below); a gripper
keypoint passes where the attempt's gripper makes the same change (a `preshape` anywhere), or,
judged by the gripper's state, where the arm reaches it with its gripper already as the keypoint
leaves it; a preshape at the start pose passes at the start (the arm is there already).
The demo's gripper keypoints cut an arm's keypoints into segments: when the attempt's gripper
changes, what the segment still has pending is missed, with its closest approach, so an arm that
hovered above a keypoint without ever entering its tolerance is caught. When
the gripper reverses right after a missed gripper keypoint, before anything of the next segment
but path keypoints passed (a close high up may be where the lift is), the attempt is retrying:
the missed segment reopens and a retry is counted, so the hints stay right for the rest of the
attempt. Keypoints after the last gripper keypoint stay
pending until they pass.

The pass rule (keypoints are a
reference that no key operation was skipped and that they came in order, not a precision gate)
widens the tolerance (`RefereeConfig`) and keeps what the order says under it: a motion keypoint
is reached only where the TCP is no nearer the arm's previous keypoint (a lift 4 cm above its
grasp is inside the tolerance at the grasp itself), and the next segment's own gripper event
also retries a missed one when it is neither at its keypoint nor nearer it than the missed one
(a close high up, on nothing, reaches the lift above it; the open that follows before the arm
moves on is its retry, not the release).

Kinds (`kinds`, from the demonstration's own data: its gripper events, success frame and
rotations, never its task): besides gripper keypoints, contacts (the success-frame keypoint)
and placements, a motion keypoint is a *path* keypoint, judged by order only: a lift or carry passes
as the arm reaches what follows it (its next keypoint, or its segment's gripper event), so it must
happen in its place in the sequence, and its position is information. A lift or carry that turns the
TCP (so the held object) by `RefereeConfig.turn_deg` or more from the previous keypoint is a *turn*,
a functional motion such as pouring or dumping, never a standoff: it keeps its position check and
must show the rotation: turned about the demonstration's axis, its way, by
`RefereeConfig.turn_share` of its angle. A gripper closed on nothing (a preshape) turns no object,
so its carries are no turns.

Standoffs (`standoffs`): the hover an arm makes before a contact (a gripper keypoint, or the
keypoint on the success frame, e.g. a press) is where the demonstration chose to pause, not part
of the task. Reaching the contact skips its pending standoffs (missed, with their closest
approach), and the verdict does not require them: a different approach distance is not a
different strategy. The approach is then checked only through the contact point itself. A
*retreat* (the arm moving away after a release) is the same clearance on the way out (by analogy
with the rules on standoffs and lifts): reported, not required, and reaching what follows it skips
it.

The pass rule: an attempt succeeds when it passes the benchmark's success
check and passes, in order, a prefix of the demo's keypoints; both arms' keypoints form one
time-ordered sequence of *moments*, and once the benchmark's success check latches, the remaining
keypoints do not count. A moment is the keypoints the demo moved both arms toward at once: their
motion segments (`Keypoint.segment`, from where the arm began moving toward a keypoint to it)
overlap, directly or through a chain (`moments`; keypoints without a recorded start are their frame
alone). Within a moment the arms' keypoints may pass in either order; across moments the order
holds. So every keypoint the attempt passed comes after the earlier moments' (at the recorded sample
it passed at, the same for both arms: `judge`'s `sources`), none of an earlier moment is still
unpassed, and none is missed: a missed keypoint was skipped or passed by before the latch (the
episode ends at the latch). What is still pending at the latch is the remaining part,
reported for information, except a gripper keypoint a retry reopened
(`KeypointProgress.retried`): the attempt gave up the operation and never did it again, so the
latch came without it (a missed close whose lift passed, then an open nearer the grasp than the
release, then success); a retry followed by the operation passes. The keypoint on the demo's
success frame (a press) counts like any other. Standoffs and retreats are not part of the sequence
(above), and path keypoints hold their place through what follows them on their arm. A placement in
a chain (ranking, stacking) is judged against the object placed before it (`held_placements`). The
order across the arms binds only keypoints on the same object (`keypoint_objects`, `Touches.binds`:
a hand-over, a stack); two arms working on different
objects, or on an object the ground truth does not find (reported as unknown), may do so in either
order.

Once the benchmark's success check has latched, every keypoint only has to show that its step
happened, in order; positions do not count where the
ground truth shows the step. A grasp or release is judged by its object: the demonstration's
object grasped or let go (at any distance; another object misses), or, where nothing moved
with the gripper, the TCP at that object as the gripper changed (`attempt_touches`: a hand-over);
only without either is it judged by position (a close on nothing is no evidence of a grasp).
Approaches and placements hold their place by order only (where a placed object landed is
information: the benchmark's success check judges it), the success frame's contact (a press) passes
at the latch, and turns and in-air steps are judged by what they did. Sliding is not a lift: a
lift is the held object raised off its support (`lift_ratio`: its center and its bottom risen a
share of the demonstration's own rise), at an in-air keypoint (`air_steps`) and over every hold
whose object the demonstration lifted, which catches a demonstration without a lift keypoint
(`held_lifts`). Skipping a step fails either way: every keypoint the demonstration
had before its own success frame is owed by the latch (pending there fails); the keypoint on that
frame is the latch's own moment (the latch may cut its step short, as it ends the attempt as soon
as the check holds), and those after it are left. An in-air step whose demonstration's object did
not move (an articulated part's root) shows nothing of that object: it is judged by order, or a
turn by the gripper. Without the benchmark's success, keypoints are judged by position within the
tolerances, which guide learning (the hints, a reviewing role's feedback). The order rules (moments,
the arms' order on one object, retries) hold throughout, except across objects: with the benchmark's
success, different objects go in any order and by either arm, each object's own steps still in
their order (`regrouped`, `judge`'s `free`); standoffs and retreats are not required.

What a gripper held (`attempt_holds`): an object lifted with the gripper and dropped before the
open (a hold lost on the way) was held, so that open lets go of it.

One more condition is judged by S1 from the attempt's calls, not here
(`learning.imitate.replaced_specs`): a Memory spec the Actor passed and had refused before
anything moved must be carried out as stored later in the attempt, by a call that ran it, in an
unhinted attempt. A pass made with a spec the Actor wrote in its place is the Actor's, not the
Memory's: S2 and P3 run the stored spec.

Judging needs the demo's success frame, which only a recording that checks success every physics
step has: keypoints of a benchmark's expert data serve the choice of a demonstration, not judging.

Paths must be dense enough that no tolerance fits between two samples: the caller interpolates
between executed steps in joint space.

Symmetry: a keypoint may come with a vertical axis (`axes`): that of the object its arm grasps
or holds there when the object is rotationally symmetric about the vertical (`symmetric`, from
the simulator's shape: the referee side may read ground truth). The keypoint then turns about
that axis to meet each sample (its position and orientation alike), so a grasp at another angle
around a round object matches; the distance from the axis, the height and the gripper's
orientation to the object stay strict, and so does everything about an object without that
symmetry (a grasp of a handled object is still one kind of grasp).
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from roboscript.contracts.env import GtSnapshot
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.robot import Arm
from roboscript.demos.demo import Array
from roboscript.demos.keypoints import Keypoint, KeypointSet, gripper_angle_deg
from roboscript.spatial.rotation import (
    quat_angle_deg,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_rotate,
)

T = TypeVar("T")
Status = Literal["passed", "missed", "pending"]
Kind = Literal[
    "grip", "standoff", "retreat", "placement", "turn", "contact", "path", "air", "motion"
]
SYMMETRY_AXIS_M: Final = 0.005
"""A keypoint this near its symmetry axis has no angle about it."""
SYMMETRY_SPREAD: Final = 0.1
"""How far a symmetric shape's outline may stick out beyond its median radius, as a share of it."""
SYMMETRY_SECTORS: Final = 9
"""How many of 12 sectors around its axis a symmetric shape's points reach at least."""
SYMMETRY_GRASP_M: Final = 0.15
"""How far from a close's TCP the center of the object it closes on may be."""


Size = tuple[float, float]
"""An object's width (its largest horizontal extent) and height, in meters (`ObjectShape.size`)."""


class RefereeConfig(BaseModel):
    """Where the referee counts a keypoint as reached: a vertical cylinder of radius `xy_m` and
    half-height `z_m` around it; widened to `size_share` of the width and the
    height of the object the arm acts on there when that is larger (`keypoint_sizes`; a
    placement: of its target). Keypoints are a reference that no key operation was skipped and
    that they came in order, not a precision gate, so the tolerance asks whether the operation
    happened there; a recipe may set other values (`recipes/wicl/protocols.yaml`). A keypoint's
    own `Tolerance` (from extraction) serves the review of re-recordings, not judging.
    `turn_deg`: a keypoint that turns the gripper this much from the previous one is a turn (see
    the module). A turn is shown by a turn about the
    demonstration's axis, in its sense, of at least `turn_share` of its angle
    (`_turn_shown`), which rejects an unturned gripper however small the demonstration's
    turn. An in-air step with a known held object is judged by that object
    (`air_steps`): it left its support in the demonstration's direction by `air_lift_m` (half the
    demonstration's rise where that is smaller), and at a turn it turned about the
    demonstration's axis, its way, by `air_turn_deg` (half the demonstration's turn where that is
    smaller); a demonstration's object that turns less than `air_turn_deg` shows no turn.

    Once the benchmark's success check latched, a lift shows by the held
    object raised off its support, not pushed along it (`lift_ratio`): where the demonstration's
    object rose `lift_floor_m` or more (at a lift or carry keypoint, or anywhere in a hold), the
    attempt's object's center, and its bottom where the demonstration's bottom rose that much,
    each rose `lift_share` of the demonstration's own rise, at most `lift_cap_m`. A
    demonstration's rise under `lift_floor_m` is within what a drag or a tilt makes too, so it
    shows no lift: the in-air check judges it (a rise of 1 cm, or the object moved 2 cm). A slide
    raises
    nothing, and a tipped object keeps its bottom on its support (an object 16 cm long tipped 20
    degrees about one end raises its center up to 2.7 cm, its bottom not at all)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    xy_m: PositiveFloat = 0.08
    z_m: PositiveFloat = 0.05
    size_share: NonNegativeFloat = 0.5
    turn_deg: Annotated[float, Field(gt=0, le=180)] = 45.0
    turn_share: Annotated[float, Field(gt=0, lt=1)] = 0.5
    air_lift_m: PositiveFloat = 0.01
    air_turn_deg: Annotated[float, Field(gt=0, le=180)] = 15.0
    lift_share: Annotated[float, Field(gt=0, le=1)] = 0.5
    lift_cap_m: PositiveFloat = 0.03
    lift_floor_m: PositiveFloat = 0.03

    def tolerance(self, size: Size | None) -> tuple[float, float]:
        """The (xy, z) tolerance at a keypoint acting on an object of `size` (`ObjectShape`)."""
        if size is None:
            return self.xy_m, self.z_m
        return max(self.xy_m, self.size_share * size[0]), max(self.z_m, self.size_share * size[1])


@dataclass(frozen=True, eq=False)
class ArmPath:
    """An executed path of one arm: TCP poses (N, 7) and the gripper command (N,)."""

    tcp: Array
    gripper: Array


class Approach(BaseModel):
    """The attempt relative to a keypoint at path sample `sample`: `offset_m` is attempt minus
    keypoint in world axes; `ratio` is the larger position error (xy, z) as a share of its
    tolerance, so the attempt is inside it when it is at most 1; `angle_deg` is information. At a
    turn, `ratio` is instead the share of the demonstration's turn the attempt failed to show
    (`turn_share`), not the position error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample: NonNegativeInt
    offset_m: Vec3
    distance_m: NonNegativeFloat
    angle_deg: NonNegativeFloat
    ratio: NonNegativeFloat


class KeypointProgress(BaseModel):
    """`at` is the sample where the keypoint passed or was missed; `nearest` is its closest
    approach within its segment so far, for a gripper keypoint where the gripper changed;
    `drop_m`, for a release judged by the held object (`held_placements`), how far the object
    fell as the gripper opened (information: where it came to rest is what is judged);
    `retried`, a gripper keypoint whose segment a retry reopened (`arm_progress`):
    one still pending at the latch fails the attempt (the operation it retried was not done);
    `same_object`, a gripper keypoint judged by the object the attempt's gripper
    change grasped or let go: whether it was the demonstration's (None: judged by position);
    `lifted_m`, a grasp whose object the demonstration lifted in its hold
    (`held_lifts`): how far the attempt's held object rose (its center) in the hold, recorded
    where that did not lift it off its support (missed)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: PositiveInt
    status: Status
    kind: Kind = "motion"
    at: NonNegativeInt | None = None
    nearest: Approach | None = None
    drop_m: float | None = None
    retried: bool = False
    same_object: bool | None = None
    lifted_m: float | None = None


class ArmProgress(BaseModel):
    """An arm's keypoints in order; `target` is the first pending one; `retries` counts reopened
    segments; `extra_events` counts the attempt's gripper changes that match no gripper keypoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keypoints: list[KeypointProgress]
    target: PositiveInt | None
    retries: NonNegativeInt = 0
    extra_events: NonNegativeInt = 0


class Verdict(BaseModel):
    """`reasons` holds the success problem and each arm's first failed required keypoint, or one
    line when passed; `informational` the keypoints not required, with their state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    reasons: list[str]
    informational: list[str]


STANDOFF_PHASES: Final = ("approach", "carry", "place")
"""The phases of a motion keypoint that may be a standoff (`standoffs`)."""


def standoffs(keypoints: Sequence[Keypoint], success_frame: int | None) -> list[int | None]:
    """Per keypoint of one arm, the position of the contact keypoint it is a standoff of, or
    None. A contact is a gripper keypoint or the keypoint on the success frame (where the task's
    effect happened: a press); a standoff is a motion keypoint in an approach, carry or place
    phase with only standoffs between it and a contact: a hover before the arm acts, whose
    distance is a choice of the demonstration, not part of the task."""
    out: list[int | None] = [None] * len(keypoints)
    contact: int | None = None
    for j in range(len(keypoints) - 1, -1, -1):
        kp = keypoints[j]
        if kp.grip or kp.frame == success_frame:
            contact = j
        elif kp.phase in STANDOFF_PHASES:
            out[j] = contact
        else:
            contact = None
    return out


DEFAULT: Final = RefereeConfig()
PATH_PHASES: Final = ("lift", "carry")
"""The phases of a motion keypoint judged by order only, or as a turn (`kinds`)."""


def kinds(
    keypoints: Sequence[Keypoint],
    success_frame: int | None,
    pointing: Vec3,
    config: RefereeConfig = DEFAULT,
) -> list[Kind]:
    """How the referee judges each keypoint of one arm, from the demonstration's own data (see
    the module): a gripper event (`grip`), a lift or carry that is a `turn` (the gripper, so the
    object it holds, turns by `config.turn_deg` or more from the previous keypoint), a
    `standoff`, a `retreat` (not required), a `placement` (place: by the held object where the
    ground truth allows), the `contact` on the success frame, another lift or carry as a `path`
    (order only), or a `motion` keypoint (an approach: by position)."""
    contact = standoffs(keypoints, success_frame)
    out: list[Kind] = []
    empty = False  # the gripper closed on nothing (a preshape) turns no object
    for j, kp in enumerate(keypoints):
        if kp.cue in ("close", "open"):
            empty = kp.cue == "close" and kp.phase == "preshape"
        turned = (
            j > 0
            and not empty
            and float(gripper_angle_deg(kp.tcp.q, keypoints[j - 1].tcp.q, pointing))
            >= config.turn_deg
        )
        if kp.grip:
            out.append("grip")
        elif kp.phase in PATH_PHASES and turned:
            out.append("turn")
        elif contact[j] is not None:
            out.append("standoff")
        elif kp.phase == "retreat":
            out.append("retreat")
        elif kp.phase in PLACEMENT:
            out.append("placement")
        elif kp.frame == success_frame:
            out.append("contact")
        elif kp.phase in PATH_PHASES:
            out.append("path")
        else:
            out.append("motion")
    return out


def arm_progress(
    keypoints: list[Keypoint],
    path: ArmPath,
    grip_threshold: float,
    pointing: Vec3,
    axes: Sequence[tuple[float, float] | None] | None = None,
    success_frame: int | None = None,
    config: RefereeConfig = DEFAULT,
    sizes: Sequence[Size | None] | None = None,
    objects: Sequence[str | None] | None = None,
    holds: Mapping[int, frozenset[str]] | None = None,
    air: Mapping[int, Array] | None = None,
    latched: bool = False,
    touched: Mapping[int, frozenset[str]] | None = None,
) -> ArmProgress:
    """Progress of one arm; `pointing` is its gripper axis (`KeypointSet.pointing`); `axes`: per
    keypoint, the vertical axis (world x, y) it may turn about (see the module), or None;
    `success_frame`: the demonstration's (`KeypointSet.success_frame`), which marks a contact
    (`standoffs`): without it only gripper keypoints are contacts; `sizes`: per keypoint, the
    size of the object the arm acts on there (`keypoint_sizes`), which widens its tolerance
    (`RefereeConfig.tolerance`). `objects` (per keypoint, the demonstration's
    object it acts on: `acted_on`) and `holds` (per path sample of a gripper change, what the
    attempt's gripper held from that close or up to that open: `attempt_holds`) judge a grasp or
    release by its object: the gripper change passes where it grasps or lets go the
    demonstration's object, however far its TCP from the keypoint, and misses where it holds
    other objects only; where either is unknown (nothing moved with the gripper, an articulated
    part), by position. `air` (per keypoint position, `air_steps`: an in-air
    step's ratio per path sample, from the held object) judges that keypoint by the object instead
    of the TCP, as an `air` keypoint: required, in its place, wherever the TCP is. With the
    official check `latched`, approaches and placements hold their
    place by order only (what follows them shows the step) and the success frame's contact passes
    at the latch; a gripper keypoint stays judged by its object where the ground truth shows what
    the attempt's gripper held (at any distance); where it held nothing that moved, it
    also passes where the TCP was at the demonstration's object (`touched`, `attempt_touches`: a
    hand-over, a grasp the benchmark's success check latched on before it moved), else by position
    (a close on nothing known is no evidence of a grasp); turns and in-air steps are judged by what
    they did."""
    kind = kinds(keypoints, success_frame, pointing, config)
    offset, angle, ratio = _errors(keypoints, path.tcp, pointing, axes, config, kind, sizes)
    count = len(keypoints)
    for j, row in (air or {}).items():
        kind[j], ratio[j] = "air", row
    step = latched
    ordered: tuple[Kind, ...] = ("path", "motion", "placement") if step else ("path",)
    inside = ratio <= 1
    if step:  # the contact: the latch itself shows it (below)
        inside[[j for j in range(count) if kind[j] == "contact"]] = False
    distance = np.linalg.norm(offset, axis=-1)
    # Wide tolerances overlap where keypoints are near (a lift 4 cm above its grasp): a motion
    # keypoint is reached only where the TCP is no nearer the arm's previous keypoint, so passing
    # it still says the arm moved on from there. An air keypoint is judged by its object, not
    # where the TCP is.
    for j in range(1, count):
        if not keypoints[j].grip and kind[j] != "air":
            inside[j] &= distance[j] <= distance[j - 1]
    # What a keypoint not reached by position waits for: a standoff its contact, a path
    # keypoint the next keypoint (`count` for the last: it passes as it becomes the target), a
    # retreat the next keypoint (the last one only its own position: it stays information).
    defer = standoffs(keypoints, success_frame)
    for j in range(count):
        if kind[j] in ordered:
            defer[j] = j + 1
        elif kind[j] == "retreat":
            defer[j] = j + 1 if j + 1 < count else None
        elif kind[j] in ("turn", "air"):  # the functional motion, not a hover
            defer[j] = None
    closed = path.gripper < grip_threshold
    status: list[Status] = ["pending"] * count
    at: list[int | None] = [None] * count
    best: list[int | None] = [None] * count
    start = target = retries = extra = 0  # start: the first keypoint of the current segment
    retried: set[int] = set()  # the gripper keypoints whose segments a retry reopened
    same: dict[int, bool] = {}  # gripper keypoints judged by their object
    reopen: tuple[int, int] | None = None  # a segment just closed by a missed gripper keypoint
    # The same, once only path keypoints passed since (a close high up, where the lift is): an
    # unexpected reversal still retries it, the next segment's own gripper event does not;
    # that event retries it too where it is neither at its keypoint nor nearer it
    # than the missed one: wide tolerances pass the lift above a close made high up (on
    # nothing), and the open that follows before the arm moves on is its retry, not the release.
    soft: tuple[int, int] | None = None

    def acting() -> int:
        """What the target waits for: itself, or past the standoffs, retreats and path keypoints it
        heads (reaching that passes the path keypoints and skips the others: missed, the verdict
        does not require them)."""
        now = target
        while now < count and (after := defer[now]) is not None:
            now = after
        return now

    def skip(to: int, i: int) -> None:
        for j in range(target, to):
            status[j], at[j] = ("passed" if kind[j] in ordered else "missed"), i

    for i in range(len(path.tcp)):
        end = next((j for j in range(start, count) if keypoints[j].grip), None)
        last = count - 1 if end is None else end
        for j in range(start, last + 1):
            if status[j] == "pending" and (best[j] is None or ratio[j, i] < ratio[j, best[j]]):
                best[j] = i
        # Targets reached at this sample count before a gripper change here closes the segment.
        while target <= last and not keypoints[target].grip:
            # The furthest keypoint reached along the chain the target waits for.
            chain, now = [target], target
            while now < count and (after := defer[now]) is not None:
                chain.append(now := after)
            reached = next(
                (j for j in reversed(chain) if j < count and not keypoints[j].grip
                 and inside[j, i]),
                target if now == count else None,
            )  # fmt: skip
            if reached is None:
                break
            skip(reached, i)
            status[reached], at[reached] = "passed", i
            target = reached + 1
            soft = (reopen or soft) if kind[reached] in ordered else None
            reopen = None
        if acting() == end and _satisfied(keypoints[end], ratio[end, i], closed, i):
            skip(end, i)
            status[end], at[end], best[end] = "passed", i, i
            start = target = end + 1
            reopen = soft = None
            continue
        if not i or closed[i] == closed[i - 1]:
            continue
        unexpected = end is None or keypoints[end].cue != ("close" if closed[i] else "open")
        due = end is not None and not unexpected and acting() == end
        arrived = due and _arrived(keypoints[end], ratio[end, i])  # type: ignore[index]
        if due and (match := _same(keypoints, objects, holds, closed, end, i)) is not None:
            arrived, same[end] = match, match  # type: ignore[index]
        elif due and step and touched is not None and not (holds or {}).get(i):
            # The gripper changed at the demonstration's object, nothing moving with it.
            demo = None if objects is None else objects[end]  # type: ignore[index]
            arrived = arrived or (demo is not None and demo in touched.get(i, frozenset()))
        away = (
            soft is not None
            and not unexpected
            and not arrived
            and distance[soft[1], i] < distance[end, i]  # type: ignore[index]
        )
        if (miss := reopen or (soft if unexpected or away else None)) is not None:
            # Reversed right after the miss: a retry of that segment, so what the miss
            # closed and the path keypoints passed since are pending again.
            first, gripped = miss
            retried.add(gripped)
            for j in range(first, count):
                if j > gripped or at[j] == at[gripped]:
                    status[j], at[j] = "pending", None
                    same.pop(j, None)
            start, target = first, status.index("pending", first)
            retries, reopen, soft = retries + 1, None, None
            continue
        if unexpected:
            extra += 1
            continue
        if arrived:
            status[end], at[end] = "passed", i
        best[end] = i
        for j in range(start, end + 1):
            if status[j] == "pending":  # path keypoints came in order: before this event
                status[j], at[j] = ("passed" if kind[j] in ordered else "missed"), i
        reopen, soft = (start, end) if status[end] == "missed" else None, None
        start = target = end + 1
    if step and (contact := acting()) < count and kind[contact] == "contact":
        # The contact on the demonstration's success frame (a press) happened where the
        # official check latched, the path's end; what the arm still waited for before it was
        # order only.
        last = len(path.tcp) - 1
        skip(contact, last)
        status[contact], at[contact], target = "passed", last, contact + 1
    return ArmProgress(
        keypoints=[
            KeypointProgress(
                index=kp.index,
                status=status[j],
                kind=kind[j],
                at=at[j],
                nearest=None if (b := best[j]) is None else _approach(offset, angle, ratio, j, b),
                retried=j in retried,
                same_object=same.get(j),
            )
            for j, kp in enumerate(keypoints)
        ],
        target=keypoints[target].index if target < count else None,
        retries=retries,
        extra_events=extra,
    )


def _same(
    keypoints: Sequence[Keypoint],
    objects: Sequence[str | None] | None,
    holds: Mapping[int, frozenset[str]] | None,
    closed: Array,
    end: int,
    i: int,
) -> bool | None:
    """Whether the attempt's gripper change at path sample `i` grasps or lets go the
    object of the gripper keypoint `end`; None where either side is unknown, or for a preshape
    (closed on nothing by design). An open the arm takes back, closing on what it let go again
    where the demonstration's next gripper keypoint is no close on it, is judged by position:
    the re-grasp retries that release (a placement corrected), as `arm_progress` reads it."""
    if objects is None or holds is None or keypoints[end].phase == "preshape":
        return None
    demo, held = objects[end], holds.get(i)
    if demo is None or not held:
        return None
    if not closed[i]:
        later = next((k for k in sorted(holds) if k > i and closed[k]), None)
        after = next((j for j in range(end + 1, len(keypoints)) if keypoints[j].grip), None)
        again = after is not None and keypoints[after].cue == "close" and objects[after] == demo
        if later is not None and holds[later] & held and not again:
            return None
    return demo in held


HELD_MOVED_M: Final = 0.02
"""How far an object moves while an attempt's gripper stays closed, at least, to be held by it
(`attempt_holds`)."""


def attempt_holds(
    gripper: Array,
    source: Array,
    tcp: Array,
    truth: Mapping[str, Array],
    grip_threshold: float,
) -> dict[int, frozenset[str]]:
    """What an attempt's gripper held (S1, privileged: the ground truth), per path sample of a
    gripper change: at a close, the objects that moved with the TCP until the next open, at that
    open the same; an object moved with it where, from some ground truth sample recorded with
    the gripper closed on, it moved `HELD_MOVED_M` or more while its offset in the TCP's frame
    changed by less than half that (carried, turned with the gripper, once the grasp settled;
    a container with what it holds, both), to the open, or, lifted `HELD_MOVED_M` with it, up to
    some sample before: an object lifted, carried and dropped before the open (a hold lost on
    the way) was held too, its release made already (a push, which lifts nothing, is no hold).
    Empty where nothing did (a close on nothing, an articulated part, whose root stays).
    `gripper`, `source`: the arm's path (`ArmPath.gripper`, `ArmPaths.sources`); `tcp`: the
    arm's recorded TCP (N, 7); `truth`: as `held_placements` takes it."""
    samples = truth.get("truth.sample")
    if samples is None:
        return {}
    closed = np.asarray(gripper) < grip_threshold
    changes = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    names = [key[6:] for key in truth if key != "truth.sample"]
    out: dict[int, frozenset[str]] = {}
    for c in changes:
        if not closed[c]:
            continue
        opens = changes[changes > c]
        first = int(source[c])
        last = int(source[opens[0]]) if len(opens) else len(tcp)
        ks = np.flatnonzero((samples >= first) & (samples < last))
        held: frozenset[str] = frozenset()
        if len(ks) >= 2:
            at = tcp[samples[ks].astype(int)]
            inverse = quat_conjugate(at[:, 3:])
            found = set()
            for name in names:
                p = truth[f"truth.{name}"][ks, :3]
                offset = quat_rotate(inverse, p - at[:, :3])
                for k in range(len(ks) - 1):
                    # the most it moved and drifted from sample k up to each later sample
                    moved = np.maximum.accumulate(np.linalg.norm(p[k:] - p[k], axis=1))
                    drift = np.maximum.accumulate(np.linalg.norm(offset[k:] - offset[k], axis=1))
                    rigid = drift < moved / 2
                    if moved[-1] >= HELD_MOVED_M and rigid[-1]:  # held to the open
                        found.add(name)
                        break
                    risen = np.maximum.accumulate(p[k:, 2] - p[k, 2]) >= HELD_MOVED_M
                    if np.any((moved >= HELD_MOVED_M) & rigid & risen):  # then fell
                        found.add(name)
                        break
            held = frozenset(found)
        out[int(c)] = held
        if len(opens):
            out[int(opens[0])] = held
    return out


TOUCH_M: Final = 0.02
"""How far outside the box of an object's size a TCP may be and still be at the object
(`attempt_touches`): the fingers' reach."""


def attempt_touches(
    path: ArmPath,
    source: Array,
    truth: Mapping[str, Array],
    shapes: Mapping[str, "ObjectShape"],
    grip_threshold: float,
) -> dict[int, frozenset[str]]:
    """S1, privileged (the ground truth): per path sample of an arm's gripper change, the
    objects its TCP was at: inside the box of the object's size (`ObjectShape.size`: its width
    both ways across and its height up, around its shape's center, as it stood at the start and
    turned with it since), grown by `TOUCH_M`, at the latest ground truth at or before the change
    (`source`: the recorded sample each path sample leads to; `truth` as `held_placements` takes
    it). What a close holding nothing that moved with it (`attempt_holds`) closed on: an object
    another gripper still holds (a hand-over), or one the benchmark's success check latched on
    before it moved."""
    samples = truth.get("truth.sample")
    if samples is None or not len(samples):
        return {}
    closed = np.asarray(path.gripper) < grip_threshold
    at = np.clip(np.searchsorted(samples, np.asarray(source), side="right") - 1, 0, None)
    out: dict[int, frozenset[str]] = {}
    for c in np.flatnonzero(closed[1:] != closed[:-1]) + 1:
        found = set()
        for name, shape in shapes.items():
            if shape.size is None or f"truth.{name}" not in truth:
                continue
            track = np.asarray(truth[f"truth.{name}"], dtype=float)
            now, start = track[int(at[c])], track[0]
            center = now[:3] + quat_rotate(now[3:], np.asarray(shape.center))
            turn = quat_multiply(now[3:], quat_conjugate(start[3:]))
            local = quat_rotate(quat_conjugate(turn), path.tcp[c, :3] - center)
            half = np.array([shape.size[0], shape.size[0], shape.size[1]]) / 2 + TOUCH_M
            if np.all(np.abs(local) <= half):
                found.add(name)
        out[int(c)] = frozenset(found)
    return out


def _arrived(kp: Keypoint, ratio: float) -> bool:
    """Whether a gripper event of the attempt is at the gripper keypoint `kp` (a preshape:
    anywhere)."""
    return kp.phase == "preshape" or ratio <= 1


def _satisfied(kp: Keypoint, ratio: float, closed: Array, i: int) -> bool:
    """Whether a gripper keypoint that is the target passes at sample `i` without a gripper
    change there: a preshape the arm is at from the start (sample 0), or the arm at the
    keypoint with its gripper already as the keypoint leaves it (judged by the gripper's state,
    not by an event: the attempt may have closed or opened it earlier)."""
    if ratio > 1 or (i and closed[i] != closed[i - 1]):
        return False
    return (i == 0 and kp.phase == "preshape") or bool(closed[i]) == (kp.gripper == "closed")


def judge(
    keypoints: KeypointSet,
    progress: dict[Arm, ArmProgress],
    success: bool,
    sources: Mapping[Arm, ArrayLike] | None = None,
    config: RefereeConfig = DEFAULT,
    objects: Mapping[Arm, Sequence["Touches"]] | None = None,
    arms: Mapping[Arm, Sequence[Keypoint]] | None = None,
    free: bool = False,
) -> Verdict:
    """The S1 pass rule (see the module docstring). `sources`: per arm, the recorded sample each
    point of its progress's path leads to (`ArmPaths.sources`), the clock the arms are ordered
    on; None when the arms' paths share one clock already. `objects` (per arm and keypoint,
    `keypoint_objects`) binds the order across the arms only between keypoints on the same object
    (`Touches.binds`); without it every keypoint of the other arm binds. `arms`: the keypoints
    `progress`
    judged per arm, the demonstration's (`keypoints.arms`) by default, else regrouped by object
    (`regrouped`), which `free` judges: with the benchmark's success, different
    objects in any order and by either arm, so that only
    keypoints on one object (`Touches.binds`: its own steps, a hand-over, a stack) bind each
    other's order across what the arms did; an arm's own keypoints came in its order already
    (`arm_progress`)."""
    if keypoints.success_frame is None:
        raise ValueError(
            f"{keypoints.task} seed {keypoints.seed}: the keypoints have no success frame; "
            "re-record the demonstration with a per-step success check"
        )
    reasons = [] if success else ["the task's success never latched"]
    informational: list[str] = []
    # Both arms' keypoints as one sequence in the demonstration's time, in moments (`moments`).
    # Standoffs and retreats mark the demo's time too, so they take part in finding the moments.
    layout = keypoints.arms if arms is None else arms
    every = [(arm, j, kp) for arm, kps in layout.items() for j, kp in enumerate(kps)]
    timed = [kp for *_, kp in every]
    of = {(arm, j): m for (arm, j, _), m in zip(every, moments(timed), strict=True)}
    sequence: list[tuple[Arm, Keypoint, KeypointProgress, int | None]] = []
    touches: list[Touches | None] = []
    places: list[tuple[Arm, int]] = []  # each entry of `sequence`: its arm and position there
    for arm, kps in layout.items():
        clock = None if sources is None else np.asarray(sources[arm])
        for j, (kp, p) in enumerate(zip(kps, progress[arm].keypoints, strict=True)):
            if p.kind in ("standoff", "retreat"):
                informational.append(f"{describe(arm, kp, p)} (not required)")
                continue
            time = p.at if clock is None or p.at is None else int(clock[p.at])
            sequence.append((arm, kp, p, time))
            places.append((arm, j))
            touches.append(objects[arm][j] if objects is not None else None)
            if (found := touches[-1]) is not None and not found.held:
                informational.append(
                    f"{arm} {kp.name} ({kp.phase}): object unknown, ordered only on its arm"
                )

    def bound(a: int, b: int) -> bool:
        """Whether keypoints `a` and `b` are ordered: one arm's always; the two arms' only
        where both are known to act on one object (`Touches.binds`; an unknown object
        binds nothing)."""
        ta, tb = touches[a], touches[b]
        if free:  # regrouped by object: only one object's keypoints bind
            return ta is not None and tb is not None and ta.binds(tb)
        return sequence[a][0] == sequence[b][0] or ta is None or tb is None or ta.binds(tb)

    moment = [of[place] for place in places]
    order = sorted(range(len(sequence)), key=lambda i: (moment[i], sequence[i][1].frame))
    passed = 0
    failure: str | None = None  # the first break of the rule
    open_: list[int] = []  # the keypoints not passed, in order
    lines: dict[int, str] = {}
    done: list[int] = []  # the keypoints passed in order
    for i in order:
        arm, kp, p, time = sequence[i]
        line = lines[i] = describe(arm, kp, p)
        before = [b for b in open_ if moment[b] < moment[i] and bound(b, i)]
        # What the demonstration did before its own success frame is owed by the latch;
        # a keypoint on that frame is the latch's own moment,
        # whose effect the benchmark's success check judged (the latch may cut the step short there,
        # as it ends the attempt as soon as the check holds), and those after it are left.
        owed = success and kp.frame < keypoints.success_frame
        if p.status == "pending" and owed:
            open_.append(i)
            skipped = _skipped(kp, p, touches[i])
            failure = failure or f"{line}: not done by the official success latch{skipped}"
            continue
        if p.kind == "path":
            # No time of its own: it passes with what follows it on its arm (`arm_progress`),
            # so that keypoint's place in the sequence holds its order.
            late = p.status == "pending" and success
            informational.append(
                f"{line} ({'after the success latch' if late else 'judged by order'})"
            )
            passed += p.status == "passed" and not before and failure is None
            continue
        if p.status != "passed":
            open_.append(i)
            if p.status == "missed":  # skipped or passed by before the latch
                failure = failure or line
            elif p.retried:  # a retry reopened it, and it was never redone
                failure = failure or f"{line} after a retry reopened it: the operation was not done"
            elif success:  # the episode ended at the latch before the attempt got to it
                informational.append(f"{line} (after the success latch)")
            continue
        assert time is not None  # a passed keypoint has its sample
        ahead = [a for a in done if moment[a] < moment[i] and bound(a, i) and sequence[a][3] > time]
        if before:
            after = f"yet {arm} {kp.name} ({kp.phase}), after it, passed"
            failure = failure or f"{lines[before[0]]}, {after}"
        elif ahead:
            first = max(ahead, key=lambda a: sequence[a][3] or 0)
            failure = failure or (
                f"{line} before {sequence[first][0]} {sequence[first][1].name}, which comes first"
            )
        else:
            passed += 1
            done.append(i)
    if not success and open_ and failure is None:
        failure = lines[open_[0]]  # what the attempt still owed
    if failure is not None:
        reasons.append(failure)
    if not reasons:
        return Verdict(
            passed=True,
            reasons=[
                f"success latched with the first {passed} of {len(sequence)} keypoints passed "
                "in order"
            ],
            informational=informational,
        )
    return Verdict(passed=False, reasons=reasons, informational=informational)


STEP_NAMES: Final[dict[str, str]] = {
    "grasp": "grasped",
    "release": "released",
    "lift": "lifted off its support",
    "carry": "carried",
    "place": "placed",
}
"""What a skipped step's phase says of its object (`_skipped`)."""


def _skipped(kp: Keypoint, p: KeypointProgress, touches: "Touches | None") -> str:
    """The step an owed keypoint shows, named for the Reviser (a skipped key
    step fails, and the review must say which): e.g. `: the demonstration's step 'grasp basket'
    never happened (the basket was never grasped)`; a held object that rose too little says so
    (`describe`); empty where its object is unknown."""
    if touches is None or len(touches.held) != 1:
        return ""
    obj = next(iter(touches.held))
    done = STEP_NAMES.get(kp.phase)
    if kp.cue == "close" and kp.phase != "preshape":
        done = "grasped"
    elif kp.cue == "open":
        done = "released"
    elif p.kind == "turn" or (p.kind == "air" and kp.phase not in ("lift", "carry")):
        done = "turned the demonstration's way"
    if done is None:
        return f": the demonstration's step '{kp.phase} {obj}' never happened"
    step = f": the demonstration's step '{kp.phase} {obj}' never happened"
    return f"{step} (the {obj} was never {done})"


def swapped(keypoints: KeypointSet, arms: Collection[Arm]) -> dict[Arm, Arm] | None:
    """The Memory never binds a hand: each arm with
    keypoints mapped to the other of the attempt's two `arms`, whose path the demonstration's
    keypoints of that arm are then judged on (`judge` on the progress so made); None without two
    arms. Only an attempt with the benchmark's success latched is judged so, and passes so only
    where it does not as it ran: the order and every contact keypoint still hold."""
    if len(arms) != 2:
        return None
    first, second = arms
    return {arm: second if arm == first else first for arm in keypoints.arms}


@dataclass(frozen=True)
class Regrouped:
    """The demonstration's keypoints regrouped by object onto the attempt's arms (`regrouped`):
    per attempt arm its keypoints (`arms`, whole blocks of the demonstration's, in the order the
    attempt grasped their objects) and, for each, the demonstration's arm and position it came
    from (`origin`); `moved`, per block that changed arm or place, a line for the verdict."""

    arms: dict[Arm, list[Keypoint]]
    origin: dict[Arm, list[tuple[Arm, int]]]
    moved: list[str]


def blocks(keypoints: Sequence[Keypoint], objects: Sequence[str | None]) -> list[list[int]] | None:
    """One arm's keypoints cut into blocks, each one object's: the positions from after the
    previous release through the next one (`open`), those after the last release joining the
    last block; None where a block grasps no known object (`objects`, `acted_on`: no close, a
    preshape alone, an object the ground truth does not find), which cannot move between arms."""
    out: list[list[int]] = []
    current: list[int] = []
    for j, kp in enumerate(keypoints):
        current.append(j)
        if kp.cue == "open":
            out.append(current)
            current = []
    if current:
        if out:
            out[-1].extend(current)
        else:
            out.append(current)
    for block in out:
        if block_object(keypoints, objects, block) is None:
            return None
    return out


def block_object(
    keypoints: Sequence[Keypoint], objects: Sequence[str | None], block: Sequence[int]
) -> str | None:
    """The object a block (`blocks`) grasps: its first close that is no preshape."""
    close = next(
        (j for j in block if keypoints[j].cue == "close" and keypoints[j].phase != "preshape"),
        None,
    )
    return None if close is None else objects[close]


def regrouped(
    keypoints: KeypointSet,
    objects: Mapping[Arm, Sequence[str | None]],
    paths: Mapping[Arm, ArmPath],
    sources: Mapping[Arm, ArrayLike],
    holds: Mapping[Arm, Mapping[int, frozenset[str]]],
    grip_threshold: float,
) -> Regrouped | None:
    """Different objects in any order, by either arm, each object's own steps still in their
    order: the demonstration's blocks (`blocks`, per arm
    by `objects`, `acted_on`) assigned to the attempt's arms (`paths`) in the order the attempt
    grasped their objects: each of an arm's closes that held objects (`holds`, `attempt_holds`;
    `sources` the recorded sample each path sample leads to, the arms' common clock) takes the
    first block not yet assigned of an object it held (the one the demonstration grasped first),
    a close again on the object that arm took last (a re-grasp) none. A block whose object the
    attempt never held stays on its own arm, after the rest, where it is owed and fails. None
    where a demonstration's block has no known object, or where nothing moved: each block on its
    own arm, in the demonstration's order (the arms as they ran, `judge`)."""
    found: dict[Arm, list[list[int]]] = {}
    for arm, kps in keypoints.arms.items():
        cut = blocks(kps, objects[arm])
        if cut is None:
            return None
        found[arm] = cut
    queue: dict[str, list[tuple[int, Arm, int]]] = {}  # object -> (frame, demo arm, block)
    for arm, cut in found.items():
        kps = keypoints.arms[arm]
        for b, block in enumerate(cut):
            obj = block_object(kps, objects[arm], block)
            assert obj is not None
            grasp = next(j for j in block if kps[j].cue == "close" and kps[j].phase != "preshape")
            queue.setdefault(obj, []).append((kps[grasp].frame, arm, b))
    for entries in queue.values():
        entries.sort()
    events: list[tuple[int, Arm, frozenset[str]]] = []
    for arm, path in paths.items():
        closed = np.asarray(path.gripper) < grip_threshold
        clock = np.asarray(sources[arm])
        for c, held in sorted(holds.get(arm, {}).items()):
            if closed[c] and held:
                events.append((int(clock[c]), arm, held))
    events.sort(key=lambda event: event[0])
    assigned: dict[Arm, list[tuple[Arm, int]]] = {arm: [] for arm in paths}
    last: dict[Arm, str] = {}  # the object each attempt arm took last
    holder: dict[str, Arm] = {}  # the attempt arm that took each object last
    for _, arm, held in events:
        open_ = [obj for obj in held if queue.get(obj)]
        if not open_:
            continue
        obj = min(open_, key=lambda o: queue[o][0][0])
        if last.get(arm) == obj and holder.get(obj) == arm:
            continue  # a re-grasp of what this arm holds or just set down
        _, demo_arm, b = queue[obj].pop(0)
        assigned[arm].append((demo_arm, b))
        last[arm], holder[obj] = obj, arm
    left = sorted(
        (keypoints.arms[demo_arm][found[demo_arm][b][0]].frame, demo_arm, b)
        for entries in queue.values()
        for _, demo_arm, b in entries
    )
    for _, demo_arm, b in left:
        target = demo_arm if demo_arm in assigned else next(iter(assigned))
        assigned[target].append((demo_arm, b))
    own = {arm: [(arm, b) for b in range(len(cut))] for arm, cut in found.items()}
    if all(assigned.get(arm, []) == blocks_ for arm, blocks_ in own.items()) and all(
        not done or arm in own for arm, done in assigned.items()
    ):
        return None
    arms: dict[Arm, list[Keypoint]] = {}
    origin: dict[Arm, list[tuple[Arm, int]]] = {}
    moved: list[str] = []
    for arm, taken in assigned.items():
        if not taken:
            continue
        for place, (demo_arm, b) in enumerate(taken):
            kps = keypoints.arms[demo_arm]
            block = found[demo_arm][b]
            arms.setdefault(arm, []).extend(kps[j] for j in block)
            origin.setdefault(arm, []).extend((demo_arm, j) for j in block)
            if demo_arm != arm or place != b:
                obj = block_object(kps, objects[demo_arm], block)
                moved.append(
                    f"the {arm} arm did the demonstration's {demo_arm} {kps[block[0]].name}-"
                    f"{kps[block[-1]].name} ({obj}) as its object number {place + 1}"
                )
    return Regrouped(arms=arms, origin=origin, moved=moved)


def permute(
    values: Mapping[Arm, Sequence[T]], origin: Mapping[Arm, Sequence[tuple[Arm, int]]]
) -> dict[Arm, list[T]]:
    """Per-keypoint values of the demonstration's arms (`values`) laid out as `origin`
    (`Regrouped.origin`) regrouped them."""
    return {arm: [values[a][j] for a, j in at] for arm, at in origin.items()}


def moments(keypoints: Sequence[Keypoint]) -> list[int]:
    """Each keypoint's moment in the demonstration's time, numbered 0, 1, ... in order: keypoints
    whose segments (`Keypoint.segment`, the arm's motion toward it) overlap, of both arms (one
    arm's never do), or through a chain of such overlaps, are one moment, which the arms moved
    toward at once."""
    out = [0] * len(keypoints)
    moment, end = -1, -1
    for i in sorted(range(len(keypoints)), key=lambda i: keypoints[i].segment):
        start, frame = keypoints[i].segment
        if start > end:
            moment += 1
        end = max(end, frame)
        out[i] = moment
    return out


def describe(arm: Arm, kp: Keypoint, p: KeypointProgress) -> str:
    """One line on a keypoint, e.g. `left K3 (grasp) missed: 3.1 cm off (dx +0.4, dy -0.2, dz
    +3.1 cm), 4 deg, 3.1x tolerance`, with the closest approach of a keypoint not passed."""
    text = f"{arm} {kp.name} ({kp.phase}) {p.status}"
    if p.drop_m is not None:
        text += f" (object dropped {100 * p.drop_m:.1f} cm on release)"
    if p.lifted_m is not None:
        return (
            f"{text}: the held object rose {100 * p.lifted_m:.1f} cm in the hold, never lifted "
            "off its support as the demonstration's was (slid or tipped, not lifted)"
        )
    if p.status == "passed" or p.nearest is None:
        return text
    n = p.nearest
    if p.kind == "air":
        made = 100 / max(n.ratio, 1e-3)
        return f"{text}: the held object made {made:.0f}% of the lift or turn needed"
    dx, dy, dz = (round(100 * v, 1) + 0.0 for v in n.offset_m)  # + 0.0: no -0.0
    return (
        f"{text}: {100 * n.distance_m:.1f} cm off (dx {dx:+.1f}, dy {dy:+.1f}, dz {dz:+.1f} cm),"
        f" {n.angle_deg:.0f} deg, {n.ratio:.1f}x tolerance"
    )


def _errors(
    keypoints: list[Keypoint],
    tcp: Array,
    pointing: Vec3,
    axes: Sequence[tuple[float, float] | None] | None = None,
    config: RefereeConfig = DEFAULT,
    kind: Sequence[Kind] = (),
    sizes: Sequence[Size | None] | None = None,
) -> tuple[Array, Array, Array]:
    """Per keypoint and sample: offset (K, N, 3), gripper angle and tolerance ratio (K, N)."""
    if not keypoints:
        return np.zeros((0, len(tcp), 3)), np.zeros((0, len(tcp))), np.zeros((0, len(tcp)))
    target = np.array([kp.tcp.as_list() for kp in keypoints])
    targets = np.broadcast_to(target[:, None, :], (len(keypoints), len(tcp), 7)).copy()
    spins = np.zeros((len(keypoints), len(tcp)))  # each target's turn about its symmetry axis
    for j, axis in enumerate(axes or ()):
        if axis is not None:
            spins[j] = _spin(target[j], tcp, np.asarray(axis, dtype=float))
            targets[j] = _turned(target[j], spins[j], np.asarray(axis, dtype=float))
    offset = tcp[None, :, :3] - targets[..., :3]
    angle = gripper_angle_deg(tcp[None, :, 3:], targets[..., 3:], pointing)
    # The keypoint standard is position and gripper state: the orientation
    # is reported, never required, except at a turn, which must show the rotation.
    sized = sizes if sizes is not None else [None] * len(keypoints)
    tolerances = np.array([config.tolerance(size) for size in sized])
    ratio = np.maximum(
        np.linalg.norm(offset[..., :2], axis=-1) / tolerances[:, None, 0],
        np.abs(offset[..., 2]) / tolerances[:, None, 1],
    )
    for j, k in enumerate(kind):
        if k != "turn":
            continue
        # The previous keypoint's orientation, turned with this one about the symmetry axis.
        spin = quat_from_axis_angle([0.0, 0.0, 1.0], spins[j])
        before = quat_multiply(spin, target[j - 1, 3:])
        shown = _turn_shown(before, targets[j, :, 3:], tcp[:, 3:], pointing)
        # The part of the turn still missing, as a share of what may be missing (<= 1: shown).
        missing = np.clip(1.0 - shown, 0.0, None) / (1.0 - config.turn_share)
        # An in-air step is judged by what it did, not where the TCP is: a turn
        # without its object's ground truth by the gripper's turn alone.
        ratio[j] = missing
    return offset, angle, ratio


def _turn_shown(before: Array, after: Array, tcp: Array, pointing: Vec3) -> Array:
    """Per sample, how much of the demonstration's turn from `before` to `after` (N, 4 each, in
    the world) the gripper orientation `tcp` (N, 4) has made from `before`: its rotation's twist
    about the demonstration's axis, in its sense, as a share of the demonstration's angle (1:
    all of it, 0: none, below 0: the other way). A gripper turned half a turn about `pointing`
    holds the same way, so each side takes whichever of the two holds turns the least (the
    demonstration) or the most (the attempt)."""
    half = np.array([0.0, *pointing])
    flipped = quat_multiply(after, half)
    # The demonstration's turn: the hold nearer `before`.
    near = quat_angle_deg(after, before) <= quat_angle_deg(flipped, before)
    turn = quat_multiply(np.where(near[:, None], after, flipped), quat_conjugate(before))
    turn *= np.where(turn[:, :1] < 0, -1.0, 1.0)  # w >= 0: the angle in [0, pi]
    sine = np.linalg.norm(turn[:, 1:], axis=1)
    angle = 2 * np.arctan2(sine, turn[:, 0])
    axis = turn[:, 1:] / np.maximum(sine, 1e-12)[:, None]
    shares = []
    for hold in (tcp, quat_multiply(tcp, half)):
        made = quat_multiply(hold, quat_conjugate(before))
        twist = 2 * np.arctan2(np.sum(made[:, 1:] * axis, axis=1), made[:, 0])
        twist = (twist + np.pi) % (2 * np.pi) - np.pi  # q and -q: the same rotation
        shares.append(twist / np.maximum(angle, 1e-9))
    shown: Array = np.maximum(*shares)
    return shown


def _spin(target: Array, tcp: Array, axis: Array) -> Array:
    """The turn (N,) about the vertical `axis` (x, y) that brings the keypoint `target` (7,) to
    face each sample of `tcp` (N, 7) around it."""
    have = target[:2] - axis
    want = tcp[:, :2] - axis
    turn = np.arctan2(have[0] * want[:, 1] - have[1] * want[:, 0], want @ have)
    spin: Array = np.where(np.hypot(*have) < SYMMETRY_AXIS_M, 0.0, turn)  # on the axis: any
    return spin


def _turned(target: Array, turn: Array, axis: Array) -> Array:
    """The keypoint `target` (7,) turned by `turn` (N,) about the vertical `axis` (x, y): (N, 7)."""
    have = target[:2] - axis
    c, s = np.cos(turn), np.sin(turn)
    out = np.empty((len(turn), 7))
    out[:, 0] = axis[0] + c * have[0] - s * have[1]
    out[:, 1] = axis[1] + s * have[0] + c * have[1]
    out[:, 2] = target[2]
    out[:, 3:] = quat_multiply(quat_from_axis_angle([0.0, 0.0, 1.0], turn), target[3:])
    return out


def symmetric(points: ArrayLike) -> bool:
    """Whether an object's shape (world points, e.g. its convex hulls' vertices, upright) is
    rotationally symmetric about a vertical axis: at every height its outline, the farthest point
    in each of 12 sectors around the axis, has no outward bump (none beyond `SYMMETRY_SPREAD` over
    their median), and the outline goes round (at least `SYMMETRY_SECTORS` sectors). Inner points (a
    hollow body's inside wall, a convex decomposition's pieces) do not count; a box's corners or a
    handle break it."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(p) < SYMMETRY_SECTORS:
        return False
    center = (p[:, :2].min(axis=0) + p[:, :2].max(axis=0)) / 2
    rel = p[:, :2] - center
    radius = np.hypot(rel[:, 0], rel[:, 1])
    outer = radius > 0.2 * radius.max()  # points near the axis say nothing about the shape
    sectors = np.floor((np.arctan2(rel[:, 1], rel[:, 0]) + np.pi) / (np.pi / 6)).astype(int) % 12
    if len(np.unique(sectors[outer])) < SYMMETRY_SECTORS:
        return False
    bands = np.digitize(p[:, 2], np.linspace(p[:, 2].min(), p[:, 2].max(), 6)[1:-1])
    for band in np.unique(bands):
        here = (bands == band) & outer
        outline = np.array([radius[here & (sectors == k)].max() for k in np.unique(sectors[here])])
        middle = float(np.median(outline)) if len(outline) else 0.0
        # Only outward bumps count: hull vertices are sparse, so a sector may hold only an
        # inner vertex and dip below the outline, which says nothing about the shape.
        if len(outline) >= SYMMETRY_SECTORS and outline.max() > (1 + SYMMETRY_SPREAD) * middle:
            return False
    return True


class ObjectShape(BaseModel):
    """What the referee knows of an object's shape: whether it is `symmetric` (see
    `symmetric`), its shape's center in the object's own frame and its `size` as it stands at the
    start (`width`; None: not recorded)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symmetric: bool
    center: Vec3
    size: tuple[NonNegativeFloat, NonNegativeFloat] | None = None


def object_shapes(snapshot: GtSnapshot) -> dict[str, ObjectShape]:
    """The shapes of a ground-truth snapshot's objects that have hulls (`geometry=True`)."""
    shapes = {}
    for name, hulls in snapshot.hulls.items():
        pose = snapshot.objects.get(name)
        points = np.array([v for hull in hulls for v in hull], dtype=float).reshape(-1, 3)
        if pose is None or not len(points):
            continue
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        local = quat_rotate(quat_conjugate(pose.q), center - np.asarray(pose.p))
        shapes[name] = ObjectShape(
            symmetric=symmetric(points),
            center=tuple(float(v) for v in local),  # type: ignore[arg-type]
            size=(width(points), float(np.ptp(points[:, 2]))),
        )
    return shapes


def width(points: ArrayLike) -> float:
    """The largest horizontal extent of world points (N, 3): their widest projection onto a
    horizontal direction, in 1-degree steps (within 0.02% of the exact one)."""
    xy = np.asarray(points, dtype=float).reshape(-1, 3)[:, :2]
    angles = np.radians(np.arange(180.0))
    spans = xy @ np.stack([np.cos(angles), np.sin(angles)])
    return float(np.max(spans.max(axis=0) - spans.min(axis=0)))


def symmetry_axes(
    keypoints: Sequence[Keypoint],
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
) -> list[tuple[float, float] | None]:
    """Per keypoint of one arm, the vertical axis it may turn about (`arm_progress`): that of the
    object the arm closes on next (up to its close) or holds (from its close to its open), when
    that object is symmetric; `poses`: the objects' ground-truth poses at the demonstration's
    keypoint frames. An object is the one nearest the TCP as the arm closes (within
    `SYMMETRY_GRASP_M`)."""
    axes: list[tuple[float, float] | None] = []
    for kp, obj in zip(keypoints, acted_on(keypoints, poses, shapes), strict=True):
        pose = poses.get(kp.frame, {}).get(obj or "")
        shape = shapes.get(obj or "")
        if pose is None or shape is None or not shape.symmetric:
            axes.append(None)
        else:
            center = np.asarray(pose.p) + quat_rotate(pose.q, shape.center)
            axes.append((float(center[0]), float(center[1])))
    return axes


def keypoint_sizes(
    keypoints: Sequence[Keypoint],
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
) -> list[Size | None]:
    """Per keypoint of one arm, the size of the object it acts on (`arm_progress`
    on): the one it closes on next or holds, as `symmetry_axes` finds it, else (no close follows:
    a press) the one nearest its TCP at the keypoint (within `SYMMETRY_GRASP_M`); None without
    one or its size."""
    sizes: list[Size | None] = []
    for kp, obj in zip(keypoints, acted_on(keypoints, poses, shapes), strict=True):
        obj = obj or _grasped(poses.get(kp.frame, {}), shapes, kp)
        shape = shapes.get(obj or "")
        sizes.append(None if shape is None else shape.size)
    return sizes


@dataclass(frozen=True)
class Touches:
    """The objects a keypoint acts on (`keypoint_objects`): `held`, the one its arm closes on or
    holds there (or, closing on nothing, the one nearest its TCP: a press), empty when none is
    found (unknown); `onto`, what a placement sets it on or in (`rests_on`)."""

    held: frozenset[str]
    onto: frozenset[str] = frozenset()

    def binds(self, other: "Touches") -> bool:
        """Whether two arms' keypoints are known to be on the same object, so their order holds
        (order only where both arms are known to act on one object): one
        handles what the other handles (a hand-over, both holding one long object) or sets something
        onto or into it (a stack). Two placements onto one tray, of different things, do not,
        nor two placed side by side, nor a keypoint whose object is unknown (empty `held`)."""
        return bool(self.held & (other.held | other.onto) or other.held & self.onto)


def keypoint_objects(
    keypoints: Sequence[Keypoint],
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
) -> list[Touches]:
    """Per keypoint of one arm, the objects it acts on (`Touches`), from the demonstration's
    ground truth: the object it closes on next or holds (`symmetry_axes`'s), else the one nearest
    its TCP (within `SYMMETRY_GRASP_M`), else none (unknown: it binds no order across the arms);
    a place or release also what the held object rests on or in once the demonstration released
    it (`rests_on`, at the first keypoint frame after the release), of the objects the
    demonstration leaves there (`held_placements`'s): a stack, a container, not a neighbor it was
    set down beside."""
    out = []
    for j, (kp, obj) in enumerate(zip(keypoints, acted_on(keypoints, poses, shapes), strict=True)):
        obj = obj or _grasped(poses.get(kp.frame, {}), shapes, kp)
        onto: frozenset[str] = frozenset()
        release = next((k for k in keypoints[j:] if k.cue == "open"), None)
        if obj is not None and kp.phase in PLACEMENT and release is not None:
            later = _moved_later(poses, shapes, release.frame, obj)
            frame = min((f for f in poses if f > release.frame), default=release.frame)
            here = poses.get(frame, {})
            onto = frozenset(
                name
                for name in here
                if name != obj and name not in later and rests_on(here, shapes, obj, name)
            )
        out.append(Touches(frozenset() if obj is None else frozenset({obj}), onto))
    return out


REST_ABOVE_M: Final = 0.005
"""How far above an object's bottom another object's bottom must be to rest on or in it (at most
half its height): one set down beside it on the same surface is level with it."""
REST_TOP_M: Final = 0.01
"""How far above an object's top another object's bottom may be and still rest on it."""


def rests_on(
    poses: Mapping[str, Pose], shapes: Mapping[str, ObjectShape], held: str, target: str
) -> bool:
    """Whether `held` rests on or in `target` (a support: a stack, a plate, a basket): its
    center is over the target's footprint (within half the target's width of its center: a
    resting object's center of mass is over its support), and its bottom lies between just above
    the target's bottom (`REST_ABOVE_M`, or half its height for a thin one) and just above its
    top (`REST_TOP_M`). Heights are the shapes' as they stood at the start (`ObjectShape.size`),
    around the shapes' centers; False without both poses and sizes. Two objects set side by side
    on one surface are level with each other, so neither rests on the other however near."""
    a, b = shapes.get(held), shapes.get(target)
    if held not in poses or target not in poses or a is None or b is None:
        return False
    if a.size is None or b.size is None:
        return False
    here, there = _center(poses[held], a), _center(poses[target], b)
    if float(np.hypot(*(here[:2] - there[:2]))) > b.size[0] / 2:
        return False
    bottom, low = here[2] - a.size[1] / 2, there[2] - b.size[1] / 2
    return low + min(REST_ABOVE_M, b.size[1] / 2) <= bottom <= low + b.size[1] + REST_TOP_M


def acted_on(
    keypoints: Sequence[Keypoint],
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
) -> list[str | None]:
    """Per keypoint of one arm, the object it closes on next (up to its close) or holds (from its
    close to its open)."""
    out: list[str | None] = []
    held: str | None = None
    for j, kp in enumerate(keypoints):
        obj = held
        if obj is None:
            close = next((k for k in keypoints[j:] if k.cue == "close"), None)
            obj = None if close is None else _grasped(poses.get(close.frame, {}), shapes, close)
        if kp.cue == "close":
            held = obj
        out.append(obj)
        if kp.cue == "open":
            held = None
    return out


PLACEMENT: Final = ("place", "release")
"""The phases judged by the held object in S1 (`held_placements`)."""
TARGET_M: Final = 0.25
"""How far from the held object, at the demonstration's release, its target may be."""
MOVED_M: Final = 0.02
"""How far the demonstration moves an object relative to the one it released, after the release,
for it not to be that release's target (`held_placements`): an object placed later is placed
relative to this one, not this one relative to it; a container carried off with what it holds
still is its target."""


def held_placements(
    keypoints: Sequence[Keypoint],
    progress: ArmProgress,
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
    truth: Mapping[str, Array],
    source: Array,
    config: RefereeConfig = DEFAULT,
    latched: bool = False,
) -> ArmProgress:
    """`progress` with each place and release keypoint of the arm judged by the object it holds
    instead of its TCP (S1, privileged: the ground truth): that object's center relative to its
    target's (the object nearest it as the demonstration released it) against the
    demonstration's, at the attempt's release (the ground truth after the batch that opened the
    gripper, against the demonstration's first frame after its release: both at rest; for a
    place keypoint, the batch before it, against the keypoint's frame), within the referee's
    position tolerance (`config`). A different grasp offset then moves neither, and a release
    from higher up that lands the object where the demonstration's did passes; how far it fell
    is recorded (`KeypointProgress.drop_m`). `poses`: the demonstration's ground truth at its
    keypoints; `truth`: the attempt's, `truth.sample` (K,) and `truth.<object>` (K, 7) as
    `Trajectory.arrays` records it; `source`: the recorded sample each point of the progress's
    path leads to.

    The target is an object the demonstration leaves where it is after the release: in a chain
    (ranking, stacking) a placement is judged against the object placed before it, in world
    axes relative to that object's center at the release. The chain's first placement, whose
    only neighbors the demonstration moves later (they are placed relative to it), is judged by
    its height alone: where it lies is set by what follows, and the benchmark's success check, which
    the verdict requires, judges the result. A release `arm_progress` judged by its
    object (`KeypointProgress.same_object`) keeps that status: where the object landed is
    recorded, and the benchmark's success check judges it. With the benchmark's success check
    `latched`, a keypoint `progress` passed (a placement in its order) stays
    passed: where its object landed is information."""
    samples = truth.get("truth.sample")
    if samples is None:
        return progress
    judged = list(progress.keypoints)
    held: str | None = None
    for j, kp in enumerate(keypoints):
        if kp.cue == "close":
            held = _grasped(poses.get(kp.frame, {}), shapes, kp)
        if held is None or kp.phase not in PLACEMENT:
            if kp.cue == "open":
                held = None
            continue
        release = next((k for k in range(j, len(keypoints)) if keypoints[k].cue == "open"), None)
        point = None if release is None else judged[release].at
        if release is None or point is None:
            continue
        at = int(source[point])  # the recorded sample of the release
        where = poses.get(keypoints[release].frame, {})
        # a chain: the target is an object the demonstration leaves there
        later = _moved_later(poses, shapes, keypoints[release].frame, held)
        target = _nearest_other(where, shapes, held, later)
        root = target is None and _nearest_other(where, shapes, held) is not None
        target = target or (held if root else None)  # a root: its own center, height only
        after = np.flatnonzero(samples >= at) if kp.cue == "open" else np.flatnonzero(samples < at)
        if target is None or not len(after) or f"truth.{held}" not in truth:
            continue
        k = int(after[0] if kp.cue == "open" else after[-1])
        # A released object is compared where it came to rest on both sides: the gripper opens
        # at the release keypoint, the object settles by the demonstration's next frame.
        frame = kp.frame
        if kp.cue == "open":
            frame = min((f for f in poses if f > kp.frame), default=kp.frame)
        demo = poses.get(frame, {})
        if held not in demo or target not in demo or f"truth.{target}" not in truth:
            continue
        now = {name: Pose.from_list(truth[f"truth.{name}"][k].tolist()) for name in (held, target)}
        # The tolerance grows with the target (a root: with itself): placed
        # anywhere on or in it that the demonstration's offset allows is the same placement.
        xy_m, z_m = config.tolerance(None if (shape := shapes.get(target)) is None else shape.size)
        if root:
            error = _center(now[held], shapes.get(held)) - _center(demo[held], shapes.get(held))
            ratio = abs(float(error[2])) / z_m
        else:
            expected = _center(demo[held], shapes.get(held)) - _center(
                demo[target], shapes.get(target)
            )
            found = _center(now[held], shapes.get(held)) - _center(now[target], shapes.get(target))
            error = found - expected
            ratio = max(float(np.hypot(error[0], error[1])) / xy_m, abs(float(error[2])) / z_m)
        shown = int(np.flatnonzero(source <= samples[k])[-1])  # its path point
        approach = Approach(
            sample=shown,
            offset_m=(float(error[0]), float(error[1]), float(error[2])),
            distance_m=float(np.linalg.norm(error)),
            angle_deg=0.0,
            ratio=ratio,
        )
        status: Status = "passed" if ratio <= 1 else "missed"
        if judged[j].same_object is not None:  # a release judged by its object
            status = judged[j].status
        elif latched and judged[j].status == "passed":
            status = "passed"  # the step happened (a placement in its order)
        # `at` stays where the arm got there, the time the arms are ordered by (`judge`): the
        # object settles later.
        reached = judged[j].at if judged[j].at is not None else shown
        update: dict[str, object] = {"status": status, "at": reached, "nearest": approach}
        if kp.cue == "open" and len(before := np.flatnonzero(samples < at)):
            held_before = Pose.from_list(truth[f"truth.{held}"][int(before[-1])].tolist())
            shape = shapes.get(held)
            fell = _center(held_before, shape)[2] - _center(now[held], shape)[2]
            update["drop_m"] = float(fell)
        judged[j] = judged[j].model_copy(update=update)
        if kp.cue == "open":
            held = None
    return progress.model_copy(update={"keypoints": judged})


AIR_RATIO_MAX: Final = 1e3
"""The ratio of an in-air step that made none of what it needs (`air_steps`)."""


def air_steps(
    keypoints: Sequence[Keypoint],
    objects: Sequence[str | None],
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
    truth: Mapping[str, Array],
    source: Array,
    success_frame: int | None,
    pointing: Vec3,
    config: RefereeConfig = DEFAULT,
    path: ArmPath | None = None,
    holds: Mapping[int, frozenset[str]] | None = None,
    grip_threshold: float = 0.5,
    latched: bool = False,
) -> dict[int, Array]:
    """An in-air step is judged by "the step happened", not by where the TCP was: per lift, carry
    or turn keypoint of one arm (`kinds`)
    whose object (`objects`, `acted_on`) both the demonstration's ground truth (`poses`) and the
    attempt's (`truth`, as `held_placements` takes it) have, its ratio per path sample (<= 1: the
    step happened by then; `source`: the recorded sample each path sample leads to). Both sides
    are measured from the object's pose at its start (the demo seed's scene is the same): the
    demonstration's rise of the object's center at the keypoint gives the direction, and the
    attempt's object must have risen that way by `config.air_lift_m` (half the demonstration's
    rise where smaller); a demonstration's object that rose less than that was carried, which the
    attempt shows by moving it `HELD_MOVED_M`. At a turn, where the demonstration's object turned
    by `config.air_turn_deg` or more, the attempt's must also have turned about the
    demonstration's axis, its way, by `config.air_turn_deg` (half the demonstration's turn where
    smaller). Whichever arm holds the object, the step happened (the Memory binds no hand).
    The ground truth is recorded between actions only, so an action that lifts and sets down
    within itself shows none of it: where this arm's `path` held the object (`holds`,
    `attempt_holds`: it moved with the TCP, its offset kept), the object follows the TCP from the
    latest ground truth sample of that hold (`_held_track`). With the benchmark's
    check `latched`, a lift (the demonstration's object rose `config.lift_floor_m` or more) shows
    by the object raised off its support (`lift_ratio`)."""
    samples = truth.get("truth.sample")
    if samples is None or not len(samples):
        return {}
    # The ground-truth sample each path sample shows: the latest at or before its recorded one.
    at = np.clip(np.searchsorted(samples, np.asarray(source), side="right") - 1, 0, None)
    kind = kinds(keypoints, success_frame, pointing, config)
    out: dict[int, Array] = {}
    for j, kp in enumerate(keypoints):
        obj = objects[j]
        if kind[j] not in ("path", "turn") or obj is None or f"truth.{obj}" not in truth:
            continue
        frames = sorted(f for f, here in poses.items() if obj in here)
        if not frames or obj not in poses.get(kp.frame, {}):
            continue
        shape = shapes.get(obj)
        start, there = poses[frames[0]][obj], poses[kp.frame][obj]
        track = np.asarray(truth[f"truth.{obj}"], dtype=float)[at]  # (N, 7)
        if path is not None and holds:
            track = _held_track(track, at, samples, source, path, holds, obj, grip_threshold)
        centers = track[:, :3] + quat_rotate(
            track[:, 3:], np.asarray(shape.center if shape else (0.0, 0.0, 0.0))
        )
        first = centers[0]
        rise = float(_center(there, shape)[2] - _center(start, shape)[2])
        step = latched
        if step and _still(start, there, shape, config):
            # The demonstration's object did not move by this step (an articulated
            # part's root stays): its pose shows nothing of the step, the gripper's does.
            continue
        # A lift at the demonstration's success frame is cut short by the latch, which
        # ends the attempt as soon as the check holds: begun is done (the in-air check's 1 cm).
        if step and rise >= config.lift_floor_m and kp.frame != success_frame:
            heights, bottoms = _heights(track, shape)
            demo = _heights(np.array([start.as_list(), there.as_list()]), shape)
            raised = float(demo[1][1] - demo[1][0])
            ratio = lift_ratio(heights - heights[0], bottoms - bottoms[0], rise, raised, config)
        else:
            if abs(rise) >= config.air_lift_m:
                need = min(config.air_lift_m, abs(rise) / 2)
                made = (centers[:, 2] - first[2]) * np.sign(rise)
            else:
                need, made = HELD_MOVED_M, np.linalg.norm(centers - first, axis=1)
            ratio = need / np.clip(made, need / AIR_RATIO_MAX, None)
        if kind[j] == "turn":
            turn = quat_multiply(np.asarray(there.q), quat_conjugate(np.asarray(start.q)))
            turn *= -1.0 if turn[0] < 0 else 1.0
            sine = float(np.linalg.norm(turn[1:]))
            angle = float(np.degrees(2 * np.arctan2(sine, turn[0])))
            if angle >= config.air_turn_deg:
                axis = turn[1:] / sine
                made_q = quat_multiply(track[:, 3:], quat_conjugate(track[0, 3:]))
                twist = 2 * np.arctan2(made_q[:, 1:] @ axis, made_q[:, 0])
                twist = np.degrees((twist + np.pi) % (2 * np.pi) - np.pi)  # q and -q: one turn
                need_deg = min(config.air_turn_deg, angle / 2)
                turned = need_deg / np.clip(twist, need_deg / AIR_RATIO_MAX, None)
                ratio = np.maximum(ratio, turned)
        out[j] = ratio
    return out


def _held_track(
    track: Array,
    at: Array,
    samples: Array,
    source: Array,
    path: ArmPath,
    holds: Mapping[int, frozenset[str]],
    obj: str,
    grip_threshold: float,
) -> Array:
    """`track` (the object's latest ground truth per path sample, `at` its index) with, between a
    close of `path` that held `obj` and the next open, the object carried rigidly with the TCP from
    the latest ground-truth sample since the close (where there is one yet)."""
    out = track.copy()
    closed = np.asarray(path.gripper) < grip_threshold
    changes = sorted(holds)
    for n, c in enumerate(changes):
        if not closed[c] or obj not in holds[c]:
            continue
        end = changes[n + 1] if n + 1 < len(changes) else len(path.tcp)
        for i in range(c, end):
            k = int(at[i])
            if samples[k] < source[c]:
                continue  # no ground truth since the close yet
            # The path point of that sample: the last one leading to it (its measured TCP).
            ref = int(np.flatnonzero(np.asarray(source) == samples[k])[-1])
            tcp_ref, tcp_now = path.tcp[ref], path.tcp[i]
            inverse = quat_conjugate(tcp_ref[3:])
            offset = quat_rotate(inverse, track[i, :3] - tcp_ref[:3])
            spin = quat_multiply(inverse, track[i, 3:])
            out[i, :3] = tcp_now[:3] + quat_rotate(tcp_now[3:], offset)
            out[i, 3:] = quat_multiply(tcp_now[3:], spin)
    return out


def _still(start: Pose, there: Pose, shape: ObjectShape | None, config: RefereeConfig) -> bool:
    """Whether an object's pose barely changed from `start` to `there`: risen less than
    `config.air_lift_m`, moved less than `HELD_MOVED_M`, turned less than `config.air_turn_deg`."""
    moved = _center(there, shape) - _center(start, shape)
    return (
        abs(float(moved[2])) < config.air_lift_m
        and float(np.linalg.norm(moved)) < HELD_MOVED_M
        and quat_angle_deg(np.asarray(there.q), np.asarray(start.q)) < config.air_turn_deg
    )


def lift_ratio(
    center: Array, bottom: Array, rise: float, raised: float, config: RefereeConfig = DEFAULT
) -> Array:
    """Per sample, a lift like the demonstration's (<= 1: shown), from how far the
    attempt's object rose since its start, its center (`center`, N) and its bottom (`bottom`, N,
    `_heights`), against the demonstration's center rise `rise` (at least `config.lift_floor_m`)
    and bottom rise `raised`: each rose `config.lift_share` of the demonstration's, at most
    `config.lift_cap_m`. The bottom counts only where the demonstration's bottom rose
    `config.lift_floor_m` or more (a long object turned as it rose may show no bottom rise on the
    box of its size, `_heights`): the center then shows the lift alone."""

    def part(made: Array, demo: float) -> Array:
        need = min(config.lift_share * demo, config.lift_cap_m)
        out: Array = need / np.clip(made, need / AIR_RATIO_MAX, None)
        return out

    ratio = part(center, rise)
    if raised >= config.lift_floor_m:
        ratio = np.maximum(ratio, part(bottom, raised))
    return ratio


def _heights(track: Array, shape: ObjectShape | None) -> tuple[Array, Array]:
    """Per pose of an object's track (N, 7, world): the height of its shape's center and of its
    bottom, the lowest point of the box of its size (`ObjectShape.size`: its width both ways
    across, its height up) as it stood at the track's first pose, turned with it since (a turn
    about the vertical lowers nothing; a tipped object's bottom stays down); the center's height
    for both without a size."""
    centers = track[:, :3] + quat_rotate(
        track[:, 3:], np.asarray(shape.center if shape else (0.0, 0.0, 0.0))
    )
    if shape is None or shape.size is None:
        return centers[:, 2], centers[:, 2]
    w, x, y, z = quat_multiply(track[:, 3:], quat_conjugate(track[0, 3:])).T
    up = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)
    half = np.array([shape.size[0], shape.size[0], shape.size[1]]) / 2
    bottoms: Array = centers[:, 2] - np.abs(up) @ half
    return centers[:, 2], bottoms


def held_lifts(
    keypoints: Sequence[Keypoint],
    progress: ArmProgress,
    objects: Sequence[str | None],
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
    truth: Mapping[str, Array],
    source: Array,
    path: ArmPath,
    holds: Mapping[int, frozenset[str]] | None,
    grip_threshold: float,
    config: RefereeConfig = DEFAULT,
    latched: bool = False,
    success_frame: int | None = None,
) -> ArmProgress:
    """Sliding is not a lift: with the benchmark's success check
    `latched`: `progress` with each passed grasp (a close, not a preshape) whose
    object (`objects`, `acted_on`) the demonstration lifted while holding it, its center rising
    `config.lift_floor_m` or more from the start at a keypoint frame up to its release (`poses`;
    the success frame's included: a demonstration may have no lift keypoint, its success frame
    showing the object up), missed where the attempt's hold never lifted it so (`lift_ratio`): the
    samples its gripper was closed from that grasp (its path sample) to its next gripper keypoint
    passed (or the end), so a grasp redone after an open the order did not count (a close on
    air, then the object) is judged by the redone one; `lifted_m` records how far its center
    rose. Only the demonstration up to its `success_frame` counts (what came after is not
    owed), and a rise it shows only at that frame was cut short in the attempt by the latch,
    which ends it as soon as the check holds: begun is done (the in-air check's 1 cm, or half the
    rise). The attempt's object follows its ground truth, or the TCP between samples while this
    arm held it (`air_steps`); `source` maps each path sample to its recorded one. One mechanism
    for every hold: where a lift keypoint follows, it adds that keypoint's place in the order;
    here, the grasp itself was a holding, not a pushing."""
    samples = truth.get("truth.sample")
    if not latched or samples is None or not len(samples):
        return progress
    at = np.clip(np.searchsorted(samples, np.asarray(source), side="right") - 1, 0, None)
    closed = np.asarray(path.gripper) < grip_threshold
    judged = list(progress.keypoints)
    for j, kp in enumerate(keypoints):
        obj, p = objects[j], judged[j]
        if kp.cue != "close" or kp.phase == "preshape" or p.status != "passed" or p.at is None:
            continue
        frames = sorted(f for f, here in poses.items() if obj in here)
        release = next((k.frame for k in keypoints[j + 1 :] if k.cue == "open"), None)
        last = success_frame if success_frame is not None else max(frames, default=0)
        if release is not None:
            last = min(last, release)
        hold = [f for f in frames if kp.frame < f <= last]
        if obj is None or f"truth.{obj}" not in truth or not hold:
            continue
        shape = shapes.get(obj)
        demo = _heights(np.array([poses[f][obj].as_list() for f in [frames[0], *hold]]), shape)
        rise, raised = (float(np.max(h[1:] - h[0])) for h in demo)
        if rise < config.lift_floor_m:
            continue
        # Raised only at the success frame: the latch cuts the attempt's lift short as soon as
        # the check holds, so begun is done (the in-air check's 1 cm, or half the rise).
        early = [k for k, f in enumerate(hold, 1) if f != success_frame]
        cut = float(np.max(demo[0][early] - demo[0][0], initial=0.0)) < config.lift_floor_m
        track = np.asarray(truth[f"truth.{obj}"], dtype=float)[at]
        if holds:
            track = _held_track(track, at, samples, source, path, holds, obj, grip_threshold)
        end = next(
            (
                q.at
                for k, q in zip(keypoints[j + 1 :], judged[j + 1 :], strict=True)
                if k.grip and q.status == "passed" and q.at is not None
            ),
            len(closed),
        )
        hold = np.flatnonzero(closed[p.at : end + 1]) + p.at
        heights, bottoms = (h - h[0] for h in _heights(track, shape))
        if cut:
            need = min(config.air_lift_m, rise / 2)
            ratio = need / np.clip(heights[hold], need / AIR_RATIO_MAX, None)
        else:
            ratio = lift_ratio(heights[hold], bottoms[hold], rise, raised, config)
        if len(ratio) and float(ratio.min()) <= 1:
            continue
        made = float(np.max(heights[hold], initial=0.0))
        judged[j] = p.model_copy(update={"status": "missed", "lifted_m": made})
    return progress.model_copy(update={"keypoints": judged})


def _center(pose: Pose, shape: ObjectShape | None) -> Array:
    return np.asarray(pose.p) + (0.0 if shape is None else quat_rotate(pose.q, shape.center))


def _nearest_other(
    poses: Mapping[str, Pose],
    shapes: Mapping[str, ObjectShape],
    held: str,
    excluded: Collection[str] = (),
) -> str | None:
    """The object nearest `held` (its target), within `TARGET_M`, other than `excluded`."""
    if held not in poses:
        return None
    here = _center(poses[held], shapes.get(held))
    best, gap = None, TARGET_M
    for name, pose in poses.items():
        distance = float(np.linalg.norm(_center(pose, shapes.get(name)) - here))
        if name != held and name not in excluded and distance < gap:
            best, gap = name, distance
    return best


def _moved_later(
    poses: Mapping[int, Mapping[str, Pose]],
    shapes: Mapping[str, ObjectShape],
    frame: int,
    held: str,
) -> set[str]:
    """The objects the demonstration moves by `MOVED_M` or more relative to `held` after `frame`."""
    here = poses.get(frame, {})
    if held not in here:
        return set()

    def offset(at: Mapping[str, Pose], name: str) -> Array:
        return _center(at[name], shapes.get(name)) - _center(at[held], shapes.get(held))

    moved = set()
    for f, later in poses.items():
        if f <= frame or held not in later:
            continue
        for name in later:
            if (
                name != held
                and name in here
                and np.linalg.norm(offset(later, name) - offset(here, name)) >= MOVED_M
            ):
                moved.add(name)
    return moved


def _grasped(
    poses: Mapping[str, Pose], shapes: Mapping[str, ObjectShape], close: Keypoint
) -> str | None:
    """The object whose shape's center is nearest the TCP of a close, within reach."""
    best, gap = None, SYMMETRY_GRASP_M
    for name, pose in poses.items():
        shape = shapes.get(name)
        center = np.asarray(pose.p) + (0.0 if shape is None else quat_rotate(pose.q, shape.center))
        distance = float(np.linalg.norm(center - np.asarray(close.tcp.p)))
        if distance < gap:
            best, gap = name, distance
    return best


def _approach(offset: Array, angle: Array, ratio: Array, j: int, i: int) -> Approach:
    dx, dy, dz = offset[j, i].tolist()
    return Approach(
        sample=i,
        offset_m=(dx, dy, dz),
        distance_m=float(np.linalg.norm(offset[j, i])),
        angle_deg=float(angle[j, i]),
        ratio=float(ratio[j, i]),
    )
