"""A demonstration cut into the segments the coverage step fits: each arm's motion up to its gripper
events (the core's keypoints, `roboscript.demos`), with the objects each involves, read from the
recording's truth.

An arm's segment begins where it starts moving toward the event (the `since` of the first
keypoint after its previous event: a demonstration may wait, as an operator does while a
support arm demonstrates) and ends at the event. A close is a `grasp` of the object that rose
while the gripper held (of those near the TCP, the nearest), else of the object that went with
the gripper (`carried`: held at one point of it while it travelled, off its support; of those,
the one held nearest its origin: an object handed over in the air rises little from where the
receiving hand closes, a box held at its rim lies far from its origin), else a `touch` of the
nearest object near the TCP (a press, a pull, a push: none rose, none went with it off its
support), unless the other arm holds that object then, closed on it `min_hold_frames` before
(`handed`; both closing at once is a two-handed lift): a hand-over's receiving grasp, made
where the giver lifted it, need not rise again. A gripper that stood closed partway, then
closes further on an object that rises (or goes with it), closed there (`deeper`: an
operator's gripper rests half-closed, below the close threshold, between holds, from the start
too; only a close that took nothing so is moved). An open after a grasp is the `place` of what
it held, which came to rest on or beside its `target` (the nearest other object there), or,
where the other arm holds it then (a grasp of its own), the `release` of a hand-over, whose
receiving grasp is marked `handover`; the giver first moves, holding it, to where it is just
before the receiving close (it brings the object there; marked `handover` too). A close with no
object near (in the air: a preshape) and one that opens again within `min_hold_frames` lifting
nothing (an operator's blip) are not events.

The demonstration is also cut where the arm pauses (it comes to rest or slows to turn back: a `stop`
or a `valley` keypoint): an arm empty-handed between an open and its next close that pauses back
where it started (within `home_m` per axis and `home_deg`: the benchmark's back-to-origin tolerance)
goes `home` there (an opponent may wait for it, as in a turn-taking game). An arm's last motion,
from its last event to the recording's end, is a `carry` of what it still holds to where it ends (an
insertion: kept, not released), `home` where the demonstration brings the arm back to where it
started, else a `move` to each pause and to where it ends (an arm that never grips presses or pushes
so).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm
from roboscript.demos.demo import ArmTrack, Demo
from roboscript.demos.keypoints import Keypoint, KeypointConfig, demo_keypoints
from roboscript.spatial.rotation import quat_angle_deg, quat_conjugate, quat_rotate

Kind = Literal["grasp", "touch", "place", "release", "carry", "move", "home"]


@dataclass(frozen=True, eq=False)
class Recording:
    """One demonstration as the coverage step reads it: the core's `Demo` (arm tracks, gripper
    commands), each object's pose per frame as recorded (truth, the env's world frame), the
    joint-position step sent after each frame but the last (`commands[t]` takes frame t to t + 1: a
    replay sends them in order) and where they came from."""

    demo: Demo
    objects: dict[str, NDArray[np.float64]]
    commands: NDArray[np.float64]
    source: dict[str, object] = field(default_factory=dict)

    def poses(self, frame: int) -> dict[str, Pose]:
        """Every recorded object's pose at `frame`; an object with none there (not finite: a
        fluid, one not yet brought in) is left out."""
        return {
            label: Pose.from_list(track[frame].tolist())
            for label, track in self.objects.items()
            if np.isfinite(track[frame]).all()
        }


@dataclass(frozen=True)
class Segment:
    """One arm's motion over frames [`start`, `end`] and the event that ends it: a `grasp` or a
    `touch` of `obj` (the TCP then at `tcp`; a grasp's hold is made at `rest`, a touch's ends
    there; `handover`: a grasp of what the other arm holds, or a move that brings what it holds
    to the other arm), the `place` of `obj`, which rested from frame `rest` on or beside `target`
    (None: no object near), its `release` to the other arm, its `carry` to where it is at `rest`
    (kept: not released), a `move` with no event, or `home`: back to the arm's start pose, which
    `tcp` then is (a demonstration may end an arm exactly there; a policy's recording stops
    anywhere within the benchmark's tolerance)."""

    arm: Arm
    kind: Kind
    start: int
    end: int
    tcp: Pose
    obj: str | None = None
    rest: int | None = None
    target: str | None = None
    handover: bool = False
    closed: bool = False  # a move: the demonstration's gripper closed through it (a press)
    carries: tuple[str, ...] = ()
    """A move or a home: what the demonstration carried with the arm through it though the cut
    read no grasp of it (`carried`: a close the keypoints took for a preshape); the chained
    judge holds the call to bringing it along (`run.left_behind`)."""

    @property
    def name(self) -> str:
        what = f" {self.obj}" if self.obj else ""
        what += " from the other arm" if self.handover and self.kind == "grasp" else ""
        onto = f" at {self.target}" if self.target else ""
        return f"{self.arm} {self.kind}{what}{onto} [{self.start}, {self.end}]"

    @property
    def key(self) -> str:
        """The name without the target: what the cut alone decides. A place's target is chosen
        on the measured boxes (`fitting.supporting`), which each step measures again; where two
        supports lie nearly as close, a step can choose the other (a long block set across two),
        so the steps after the coverage fit with its targets (`fitting.report_targets`)."""
        return replace(self, target=None).name


@dataclass(frozen=True)
class CutConfig:
    """`keypoints`: the core's extraction parameters; `near_m`: how close to the TCP at a close an
    object must be to be the arm's; `rise_m`: how far it must rise while held to be held (the
    gripper probe's rule); a released object rests from the first of `settle_frames` frames it
    moves less than `settle_m` in each; `target_m`: the farthest another object may be from where
    it rests to be its target; `take_m`: how far a grasped object moves once the hold is made
    (`taken`); `min_hold_frames`: a close that opens again sooner, lifting nothing, took no hold
    (the fingers settle in the executor binding's `settle_holds`); `home_m`, `home_deg`: where an
    arm counts as back where it started (the benchmark's own back-to-origin tolerance, which its
    glue passes: no default); `regrip`: how much further a gripper that stood still closed partway
    for `min_hold_frames` must close to close anew (`deeper`). An object that did not rise `rise_m`
    from where a close began, or lay farther than `near_m` then, is still taken where it went with
    the gripper (`carried`): the TCP stayed within `rigid_m` of one point in the object's frame
    while the object travelled `carry_m` and was lifted `rise_m` above the lowest it lay before (off
    its support), that point within `reach_m` of the object's origin (the recording has no geometry:
    a long handle, a box held at its rim)."""

    keypoints: KeypointConfig
    home_m: float
    home_deg: float
    near_m: float = 0.12
    rise_m: float = 0.03
    settle_m: float = 5e-4
    settle_frames: int = 5
    target_m: float = 0.15
    take_m: float = 0.003
    min_hold_frames: int = 10
    regrip: float = 0.2
    carry_m: float = 0.05
    rigid_m: float = 0.005
    reach_m: float = 0.2

    def __post_init__(self) -> None:
        for name in ("near_m", "rise_m", "carry_m", "rigid_m", "reach_m", "home_m", "home_deg"):
            if not getattr(self, name) > 0:
                raise ValueError(f"CutConfig.{name} must be positive, got {getattr(self, name)}")


def cut(recording: Recording, config: CutConfig) -> list[Segment]:
    """Every arm's segments, in the order their events end (frame, then arm)."""
    demo = recording.demo
    keys = demo_keypoints(demo, config.keypoints, config.keypoints.tolerance)
    last = demo.meta.frames - 1
    lifted = {  # the closes whose object rose: a giver's holds, which a receiving grasp may take
        arm: holds_of(recording, demo.arms[arm].tcp, gripper_events(
            recording, demo.arms[arm], [k for k in kps if k.grip], last, config), last, config)
        for arm, kps in keys.arms.items()
    }  # fmt: skip
    events = {
        arm: gripper_events(
            recording,
            demo.arms[arm],
            [k for k in kps if k.grip],
            last,
            config,
            [h for other, mine in lifted.items() if other != arm for h in mine],
        )
        for arm, kps in keys.arms.items()
    }
    holds = {
        arm: holds_of(
            recording,
            demo.arms[arm].tcp,
            found,
            last,
            config,
            [h for other, mine in lifted.items() if other != arm for h in mine],
        )
        for arm, found in events.items()
    }
    found: list[Segment] = []
    for arm, kps in keys.arms.items():
        tcp = demo.arms[arm].tcp
        origin = Pose.from_list(tcp[0])  # a home's target: the start pose itself (`Segment`)
        others = [h for other, mine in holds.items() if other != arm for h in mine]
        grip = demo.arms[arm].gripper

        def shut(frame: int, grip: NDArray[np.float64] = grip) -> bool:
            return bool(grip[frame] < config.keypoints.grip_threshold)

        before, held, kept = 0, None, False  # kept: the open ends a close of the cut's
        for i, event in enumerate(events[arm]):
            until = events[arm][i + 1].frame if i + 1 < len(events[arm]) else last
            start = began(kps, before)
            if event.cue == "open" and not kept:  # closed in the air (pressing): its pauses
                for k in pauses(kps, before, event.frame):
                    found.append(
                        Segment(
                            arm, "move", began(kps, before), k.frame, k.tcp, closed=shut(k.frame)
                        )
                    )
                    before = k.frame
            kept = event.cue == "close"
            if event.cue == "close":
                obj, rose = held_object(recording, tcp, event.frame, until, config)
                given = handed(others, obj, event.frame, config)  # a hand-over's receiving grasp
                kind: Kind = "grasp" if rose or given else "touch"
                # a grasp is made where its object starts to go with the gripper; a touch's
                # effect is where its hold ends
                grasped = kind == "grasp"
                ends = taken(recording, obj, event.frame, until, config) if grasped else until
                found.append(
                    Segment(arm, kind, start, event.frame, event.tcp, obj, ends, handover=given)
                )
                held = obj if grasped else None
            elif held is not None and holding(others, held, event.frame):
                # brought, held, to where the other arm takes it: ends just before that close
                took = min(c for o, c, e in others if o == held and c <= event.frame < e) - 1
                if took > before and brought(tcp[before], tcp[took], config):
                    at = Pose.from_list(tcp[took])
                    moving = began(kps, before)
                    found.append(
                        Segment(arm, "move", moving, took, at, handover=True, closed=shut(took))
                    )
                    before = took
                found.append(
                    Segment(
                        arm,
                        "release",
                        began(kps, before),
                        event.frame,
                        event.tcp,
                        held,
                        handover=True,
                    )
                )
                held = None
            elif held is not None:
                rest = resting(recording.objects[held], event.frame, config)
                target = beside(recording, held, rest, config)
                found.append(
                    Segment(arm, "place", start, event.frame, event.tcp, held, rest, target)
                )
                held = None
            before = event.frame
            if event.cue == "open":  # empty-handed until its next close: home on the way?
                stops = pauses(kps, before, until)
                back = [k for k in stops if back_home(tcp[0], tcp[k.frame], config)]
                for k in stops if back else []:  # its pauses up to the first one home
                    kind = "home" if k is back[0] else "move"
                    at = origin if kind == "home" else k.tcp
                    found.append(
                        Segment(arm, kind, began(kps, before), k.frame, at, closed=shut(k.frame))
                    )
                    before = k.frame
                    if k is back[0]:
                        break
        end = Pose.from_list(tcp[last])
        if held is not None:  # still holding at the end: brought to where it ends, kept
            target = beside(recording, held, last, config)
            found.append(Segment(arm, "carry", began(kps, before), last, end, held, last, target))
            continue
        home = back_home(tcp[0], tcp[last], config)
        for k in pauses(kps, before, last):  # where it went and stood: a press, a push
            rests = home and not any(j.frame > k.frame for j in kps)  # its last: back home
            found.append(
                Segment(
                    arm,
                    "home" if rests else "move",
                    began(kps, before),
                    k.frame,
                    origin if rests else k.tcp,
                    closed=shut(k.frame),
                )
            )
            before = k.frame
        if any(k.frame > before for k in kps):  # on from its last pause
            found.append(
                Segment(
                    arm,
                    "home" if home else "move",
                    began(kps, before),
                    last,
                    origin if home else end,
                    closed=shut(last),
                )
            )
    found = [replace(s, carries=carries(recording, s, holds[s.arm], config))
             if s.kind in ("move", "home") else s for s in found]  # fmt: skip
    return sorted(found, key=lambda s: (s.end, s.arm))


def pauses(kps: list[Keypoint], after: int, before: int) -> list[Keypoint]:
    """The arm's pauses between two frames: where it came to rest (a `stop`) or slowed to turn
    back (a `valley`: a press's bottom, a push's end)."""
    return [k for k in kps if k.cue in ("stop", "valley") and after < k.frame < before]


Hold = tuple[str, int, int]
"""An arm's hold: the object, the frame its gripper closed on it and the frame it opened."""


def taken(recording: Recording, obj: str | None, close: int, until: int, config: CutConfig) -> int:
    """The frame a grasp's hold was made: the last before its object, from the close on, moved
    `take_m` (an operator's close starts while the arm still descends: the fingers meet the
    object later and deeper); the close where it never moved."""
    assert obj is not None
    track = recording.objects[obj][close : until + 1, :3]
    gone = np.flatnonzero(np.linalg.norm(track - track[0], axis=1) > config.take_m)
    return close + max(int(gone[0]) - 1, 0) if len(gone) else close


def holds_of(
    recording: Recording,
    tcp: NDArray[np.float64],
    events: list[Keypoint],
    last: int,
    config: CutConfig,
    given: Sequence[Hold] = (),
) -> list[Hold]:
    """The arm's grasps: each close whose object rose while it held (or went with it:
    `carried`), or that the other arm held then (`given`, its holds: a hand-over), until its
    next open."""
    found = []
    for i, event in enumerate(events):
        if event.cue != "close":
            continue
        until = events[i + 1].frame if i + 1 < len(events) else last
        obj, rose = held_object(recording, tcp, event.frame, until, config)
        if obj is not None and (rose or handed(given, obj, event.frame, config)):
            found.append((obj, event.frame, until))
    return found


def carries(
    recording: Recording, s: Segment, held: Sequence[Hold], config: CutConfig
) -> tuple[str, ...]:
    """The objects that went with `s`'s arm over its frames (`carried`) but one a grasp of the
    cut's holds then (`held`, the arm's holds: a giver's move bringing what it holds to the
    other arm, which its release goes on to judge)."""
    tcp = recording.demo.arms[s.arm].tcp
    return tuple(sorted(label for label, track in recording.objects.items()
                        if not holding(held, label, s.end - 1)
                        and carried(track, tcp, s.start, s.end, config) is not None))  # fmt: skip


def holding(holds: Sequence[Hold], obj: str | None, frame: int) -> bool:
    """Whether one of `holds` (the other arm's) holds `obj` at `frame`."""
    return any(held == obj and close <= frame < opened for held, close, opened in holds)


def handed(holds: Sequence[Hold], obj: str | None, frame: int, config: CutConfig) -> bool:
    """Whether a close at `frame` on `obj` takes it from the other arm (`holds`, its): that arm
    holds it then and closed on it `min_hold_frames` before (it held it). Both arms closing on
    one object at once (a long block lifted by both) is a two-handed hold: each arm's
    own grasp, no hand-over."""
    return any(held == obj and close + config.min_hold_frames <= frame < opened
               for held, close, opened in holds)  # fmt: skip


def brought(start: NDArray[np.float64], end: NDArray[np.float64], config: CutConfig) -> bool:
    """Whether a TCP moved from `start` to `end` (the keypoints' least move or turn)."""
    k = config.keypoints
    moved = float(np.linalg.norm(end[:3] - start[:3])) >= k.min_move_m
    return moved or float(quat_angle_deg(start[3:], end[3:])) >= k.min_turn_deg


def back_home(first: NDArray[np.float64], last: NDArray[np.float64], config: CutConfig) -> bool:
    """Whether a TCP at `last` is back where it was at `first` (per axis and in angle)."""
    axis = float(np.abs(last[:3] - first[:3]).max())
    return axis <= config.home_m and float(quat_angle_deg(first[3:], last[3:])) <= config.home_deg


def gripper_events(
    recording: Recording,
    track: ArmTrack,
    events: list[Keypoint],
    last: int,
    config: CutConfig,
    given: Sequence[Hold] = (),
) -> list[Keypoint]:
    """The arm's gripper events but a close in the air and a blip (`CutConfig.min_hold_frames`),
    each with the open that ends it; a close whose object near it did not rise moved to where the
    gripper closes further on what rises (`deeper`; asked before `carried`), and one added there
    before the first open of a gripper closed from the start. A close on what the other arm
    holds then (`given`) took it."""
    tcp = track.tcp
    if events and events[0].cue == "open" and track.gripper[0] < config.keypoints.grip_threshold:
        start = events[0].model_copy(
            update={"frame": 0, "since": None, "cue": "close", "gripper": "closed"}
        )
        if (moved := deeper(recording, track, start, events[0].frame, config)) is not None:
            events = [moved, *events]
    kept: list[Keypoint] = []
    skip = False
    for i, event in enumerate(events):
        if skip:  # the open of a close left out
            skip = False
            continue
        if event.cue == "close":
            until = events[i + 1].frame if i + 1 < len(events) else last
            # a further close on what rises comes before what went with the gripper (`carried`):
            # a half-closed rest's object goes with it only once the gripper closes further
            obj, rose = held_object(recording, tcp, event.frame, until, config, carry=False)
            if not rose and not handed(given, obj, event.frame, config):
                moved = deeper(recording, track, event, until, config)
                if moved is not None:
                    kept.append(moved)
                    continue
                obj, rose = held_object(recording, tcp, event.frame, until, config)
            if not rose and (obj is None or until - event.frame < config.min_hold_frames):
                skip = i + 1 < len(events)
                continue
        kept.append(event)
    return kept


def deeper(
    recording: Recording, track: ArmTrack, close: Keypoint, until: int, config: CutConfig
) -> Keypoint | None:
    """`close` moved to where its gripper, after standing still closed partway for
    `min_hold_frames`, closes `regrip` further before `until` (its open) on an object that then
    rises: the frame its command stops falling there (an operator's gripper may rest
    half-closed, below the close threshold, between objects and close fully on each); None
    where it never does so."""
    grip = track.gripper
    still = 0
    t = close.frame + 1
    while t < until:
        if grip[t] >= grip[t - 1]:
            still = still + 1 if grip[t] == grip[t - 1] else 0
            t += 1
            continue
        end = t
        while end + 1 < until and grip[end + 1] < grip[end]:
            end += 1
        if still >= config.min_hold_frames and grip[end] <= grip[t - 1] - config.regrip:
            _, rose = held_object(recording, track.tcp, end, until, config)
            if rose:
                tcp = Pose.from_list(track.tcp[end].tolist())
                return close.model_copy(update={"frame": end, "since": None, "tcp": tcp})
        still, t = 0, end + 1
    return None


def began(kps: list[Keypoint], after: int) -> int:
    """Where the arm starts moving toward its next keypoint after frame `after` (that keypoint's
    `since`, not before `after`: a close `deeper` moved is no keypoint, the gripper's change
    before it may begin the next one); `after` itself where none follows."""
    following = [k for k in kps if k.frame > after]
    return max(following[0].segment[0], after) if following else after


def held_object(
    recording: Recording,
    tcp: NDArray[np.float64],
    frame: int,
    until: int,
    config: CutConfig,
    carry: bool = True,
) -> tuple[str | None, bool]:
    """The object a close at `frame` took (the nearest to the TCP of those near it that rose
    `rise_m` before `until`, else the one held nearest its origin of those that went with the
    gripper: `carried`; `carry` false: not those), and whether it took one; else the nearest
    near it (a touch), if any."""
    near = []
    for label, track in recording.objects.items():
        gap = float(np.linalg.norm(track[frame, :3] - tcp[frame, :3]))
        if gap <= config.near_m:
            rise = float(track[frame : until + 1, 2].max() - track[frame, 2])
            near.append((gap, label, rise >= config.rise_m))
    risen = sorted(n for n in near if n[2])
    if risen:
        return risen[0][1], True
    went = (
        []
        if not carry
        else sorted(
            (at, label)
            for label, track in recording.objects.items()
            if (at := carried(track, tcp, frame, until, config)) is not None
        )
    )
    if went:
        return went[0][1], True
    return (min(near)[1] if near else None), False


def carried(
    track: NDArray[np.float64],
    tcp: NDArray[np.float64],
    frame: int,
    until: int,
    config: CutConfig,
) -> float | None:
    """Whether the object of `track` went with the gripper between frames `frame` and `until`:
    over a run of frames in which the TCP stayed within `rigid_m` of one point in the object's
    frame (the fingers on it: a pushed object turns and slides under them, a carried one does
    not), the object travelled `carry_m` and rose `rise_m` above the lowest it lay before the run
    (off its support: what a hand passes on in the air, what a hand lowers after the other let
    go). The distance of that point from the object's origin where the run began, within
    `reach_m`; None where no run did so."""
    lowest = np.fmin.accumulate(track[:, 2])  # NaN-free once the object has a pose
    poses = track[frame : until + 1]
    seen = np.asarray(np.isfinite(poses).all(axis=1))
    held = np.zeros((len(poses), 3))  # the TCP in the object's frame
    held[seen] = quat_rotate(quat_conjugate(poses[seen, 3:]), tcp[frame : until + 1][seen, :3]
                             - poses[seen, :3])  # fmt: skip
    found: float | None = None
    start, top = -1, -np.inf
    for t in range(len(poses)):
        if not seen[t]:
            start = -1
            continue
        if start < 0 or float(np.linalg.norm(held[t] - held[start])) > config.rigid_m:
            start, top = t, -np.inf
        top = max(top, float(poses[t, 2]))
        reach = float(np.linalg.norm(held[start]))
        if (
            reach <= config.reach_m
            and float(np.linalg.norm(poses[t, :3] - poses[start, :3])) >= config.carry_m
            and top - float(lowest[frame + start]) >= config.rise_m
        ):
            found = reach if found is None else min(found, reach)
    return found


def resting(track: NDArray[np.float64], frame: int, config: CutConfig) -> int:
    """The first frame from `frame` on that begins `settle_frames` frames in each of which the
    object moved less than `settle_m` (the recording's last frame if none does)."""
    moved = np.linalg.norm(np.diff(track[frame:, :3], axis=0), axis=1) < config.settle_m
    for t in range(len(moved) - config.settle_frames + 1):
        if moved[t : t + config.settle_frames].all():
            return frame + t
    return len(track) - 1


def beside(recording: Recording, held: str, rest: int, config: CutConfig) -> str | None:
    """The other object nearest where `held` rests from frame `rest`, within `target_m`."""
    at = recording.objects[held][rest, :3]
    gaps = [
        (float(np.linalg.norm(track[rest, :3] - at)), label)
        for label, track in recording.objects.items()
        if label != held
    ]
    near = [g for g in gaps if g[0] <= config.target_m]
    return min(near)[1] if near else None
