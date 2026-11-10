"""The privileged S1 failure review (docs/concepts/learning.md): a failed S1 attempt against the
expert's state on the same seed, stage by stage, on the control plane, for the S1 Reviser and the
hinted Actor only (tier P; never an unhinted attempt, an S2 Actor or P3).

By design: what the demonstration does is reachable, so a failure is the
wrong arm, a low-level API problem, or the Actor or Memory diverging from the expert; in S1, where
privileged data may be used, a review with the expert's ground truth names which. It reads the
expert's STATE only, never its API calls: per arm the demonstration's keypoints (TCP pose, gripper
event, phase, the referee's kind) with the objects' poses at each keypoint (the package's
ground-truth layer), and the attempt's recorded samples (`physics.npz`: TCP, gripper command and
opening per action step, the objects' poses after every batch), its calls and receipts and, for
the keypoints it missed, the referee's progress.

Stages are the demonstration's gripper events (a close on an object, an open) and its turned
lifts and carries and its contact (the referee's `turn` and `contact` kinds), in frame order; a
held object's contact past its arm's last keypoint (`demos.relation.contacts`, given in
`keypoints`) is compared by the held object against what it meets, at the attempt's nearest
approach. An expert close is matched with the attempt's close on the same object, counted per
object (one the attempt closed on nothing is the object nearest its TCP, within
`ReviewConfig.near_m`); an expert open with the attempt's next open of the arm that closed on that
object. The first stage that
diverges, and how, is the verdict:

- `wrong_arm`: the attempt did the stage with the other arm;
- `transcription`: the stage is off as `memory_diverged` says, and the call's spec differs from
  the Memory's spec of that kind its own workflow stage passes (`Stage.spec_diff`, field by
  field; a stage that passes none has no spec to transcribe), which the attempt had not passed
  by name and had refused (unreachable, plan_failed) earlier: the Actor did not pass the Memory
  as stored, so the Memory is not what diverged (a stored spec already refused is what failed:
  `memory_diverged`, with that refusal in the note);
- `displaced`: the object a stage acts on (a release: its target) lay `MOVED_M` or more from
  where the expert had it, and an earlier call moved it while no gripper of the attempt held it,
  away from the expert's place (a retreat after a release dragging the released object): a
  `move` stage before that stage names the call and the move; the stage that then missed it is
  not where the attempt diverged;
- `memory_diverged`: its contact (the TCP at the close, in the object's frame; the call's
  commanded TCP when the stage failed before closing) or its release (the TCP at the open and the
  held object as last held before it, relative to the target; where it came to rest after, as it
  may have fallen, is only a note) is beyond the tolerance from the expert's, and the
  expert's pose, mapped into the attempt's scene, plans from the attempt's state (or the plan
  check did not run);
- `expert_unplannable`: the expert's pose does not plan from the attempt's state nor from its
  start, or the call failed with a target within the tolerance of it (unreachable, plan_failed):
  a low-level problem of the execution stack, the "framework problem" channel; never when the
  object lay `MOVED_M` or more from where the expert found it or the attempt came in more than
  `contact_deg` off the expert's approach (the state, not the stack, differed: `memory_diverged`);
- `grasp_slip`: the close was at the expert's pose, yet nothing was held, or the hold was lost
  during the grasp's own call;
- `hold_lost`: the grasp held (the referee saw the object move with it, or the measured opening
  stayed at the embodiment's `Gripper.holding_threshold` or more: an articulated part's root
  never moves), and the hold broke during a later call, the one in whose turn the opening fell
  (else whose receipt says `lost_hold`): a held-motion stage (`move`) after the close, with the
  TCP's displacement and turn from the close, the expert's to its next keypoint and to its last
  held one against the attempt's to where the hold broke, in the object's frame at the close
  (`Motion`); the call's parameters are the ones to move, not the grasp's;
- `execution_failed`: the same lost hold, while the attempt's TCP had kept to the expert's held
  path (every sample from the settled close to the break, and the TCP the breaking call
  commanded, within `ReviewConfig.contact_m` of the polyline of the expert's held keypoints, and
  the turn at the break within `contact_deg` of the expert's at the nearest one, all from the
  close in the object's frame): the motion was the
  expert's and the execution stack lost the hold executing it, so the Memory is not what to edit;
  the stage gives where it broke, the opening, the deviation and how far the object (an articulated
  part's body) moved;
- `path_frame`: a held path's call (`follow_path`) whose receipt says its start lay more than
  `contact_m` from where the grasp held (`path.start_m`): the frame the object was given at
  execution was not the hold's (a symmetric object's turn), so the expert's motion ran turned or
  reversed; the stack's, never a path or grasp edit, and its parameters are no evidence about the
  Memory (`NOT_MEMORY`);
- `approach_diverged`: a release whose TCP at the open and held object matched the expert's,
  but which came in along another line than the expert's last straight move onto it (beyond
  `contact_deg`): the placement's `approach` (and `approach_m`), not its offset;
- `unreproducible`: a placement off the expert's by a spec the spec dry run found no
  `constrain` reproduces on the demonstration's own state: refit it, do not tune it;
- `missing_stage`: the attempt never did the stage (a turned lift the referee missed, a close it
  never made, without a failing call to blame);
- `end_state_gap`: every stage matched, and the objects ended off the demonstration's final state
  (relative to what they rest on there).

Before the stages, the attempt's identification (`identification`: its first registry, as the
one-time identification left it, against the objects' shapes on this same seed): an object the
demonstration acts on that the registry has as an object `IDENTITY_RATIO` times its width or
more (or as little as 1 / `IDENTITY_RATIO`, unless cut at the image border), or as part of one
it registered with another such object, or (one it grasps) not at all, and a role the Memory's
workflow or specs use that nothing was registered for, make the verdict `identification`,
whatever the stages say: they were measured on the wrong objects, so their gaps are no evidence
about the specs (an object merged with its container reads as a large grasp error).

The contact names the Memory parameter behind it: the spec the call named (`{name}`), or whose
value it passed (fields other than `arm` and `reference_q` equal), else the specs the call's
workflow stage references.
"""

import functools
import itertools
import json
import math
import re
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, get_args

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt, PositiveFloat

from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import BatchPlanRequest, Planner
from roboscript.contracts.robot import Arm
from roboscript.demos import KeypointSet, ObjectShape, RefereeConfig
from roboscript.demos.keypoints import Keypoint, gripper_angle_deg
from roboscript.demos.referee import (
    ArmProgress,
    acted_on,
    attempt_holds,
    keypoint_objects,
    kinds,
    width,
)
from roboscript.embodiments import EmbodimentProfile
from roboscript.learning.demo import DemoTruth
from roboscript.manipulation.specs import GraspSpec, HingeSpec, PathSpec, PlaceSpec, PlaceTarget
from roboscript.memory import MemoryDoc, call_arguments, parse_objects, template_refs
from roboscript.memory.doc import field_diff
from roboscript.perception.registry import SceneRegistry
from roboscript.spatial.rotation import (
    quat_angle_deg,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_rotate,
)
from roboscript.toolsets.manipulation.actions import ArmAction
from roboscript_methods.wicl.diagnosis import (
    CONTAINED,
    MATCH_M,
    overlap,
    start_box,
    start_center,
    truth_box,
)

EXPERT_DIFF: Final = "expert_diff.json"
"""Beside the referee's verdict in an attempt's `referee/`."""
HARD: Final = ("unreachable", "plan_failed", "empty_grasp", "lost_hold")
REFUSED: Final = ("unreachable", "plan_failed")
"""Receipts of a call refused before it moved: the spec it passed is what failed."""
SPEC_IGNORED: Final = ("arm", "reference_q")
"""Spec fields a call may change from the Memory's value (the arm it runs on, an ordering hint)."""
SPEC_ARGUMENTS: Final = ("grasp", "place", "hinge", "path")
"""The arguments that take a spec, written out or by its Memory parameter's name."""
MOVED_M: Final = 0.01
MOVED_DEG: Final = 10.0
"""An object the demonstration moved this far (or turned) is part of its end state."""

WORLD: Final = Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))


Verdict = Literal[
    "memory_diverged",
    "transcription",
    "displaced",
    "execution_failed",
    "wrong_arm",
    "expert_unplannable",
    "grasp_slip",
    "hold_lost",
    "missing_stage",
    "end_state_gap",
    "path_frame",
    "approach_diverged",
    "unreproducible",
    "identification",
]
IDENTITY_RATIO: Final = 1.5
IDENTITY_M: Final = 0.05
"""A registered object is another object than the one it matches when its width differs from
that object's by `IDENTITY_RATIO` times and by `IDENTITY_M` (depth noise on a small object is
neither): the width is the referee's (`demos.referee.width`), of its points and of the shape."""
NOT_MEMORY: Final = ("execution_failed", "path_frame")
"""Verdicts that say the Memory reproduced the expert where the attempt failed: the parameters
they list are no evidence about the Memory (`imitate._named`)."""
Array = NDArray[np.float64]
PlanCheck = Callable[[Arm, Pose, int], Awaitable[str | None]]
"""Whether `arm` plans to a TCP pose from the attempt's state at a recorded sample: None when it
plans, else the planner's reason."""


class ReviewConfig(BaseModel):
    """How S1 reviews a failed attempt against the expert, and which Reviser edits it refuses.

    Review: `enabled` turns it on. A contact is off beyond `contact_m` (`held_m` for a close that
    held to the end: a hold a little elsewhere still carries) or `contact_deg` (orientation, a
    half turn about the gripper's pointing axis counting as none), a placement beyond `place_m`
    or `place_deg`; a close acts on the object whose outline is nearest its TCP, within `near_m`.

    Dry runs: `dry_run`: before a new attempt, each grasp spec is synthesized on perception's
    registry of the demonstration's start and checked within `spec_m` of the expert's grasp, each
    placement and held path as its call would command it on the demonstration's state, within
    `place_m` and `place_deg` of the expert's TCP and planned from the expert's state
    (`learning.spec_check`); the gaps go to the Reviser after that attempt. `reader_dry_run`: the
    Demo Reader's finish is refused up to this many times while the dry run of its draft finds a
    spec off, with the gaps (0: only the first Reviser reads them, after attempt 1).
    `fit_approach`: the Reviser's `fit_spec` placements carry the expert's approach line
    (`demos.relation.with_approach`).

    Reviser edit guards. Each refuses one kind of edit the evidence at hand does not support, and
    acts only in the situation it names; they are on together by design.
    - `revert_guard`: no setting a target the previous revision changed back without new
      evidence, so revisions do not oscillate (`memory.Editor`).
    - `repeat_guard`: no setting a target back to a value attempts on this seed failed with and
      none passed with, since it was already tried (`memory.Editor`'s `failed`).
    - `fit_reverts`: a parameter the review's first divergence does not name may go back to the
      demonstration's fit of it past the two guards above, the fit being the reference
      (`memory.Editor`'s `first`).
    - `lock_passed_objects`: no rewriting OBJECTS while it is as it was when an attempt passed
      on this seed, since it was right then.
    - `noop_edits`: placement-spec edits that leave what `place_object` commands with the
      workflow's call on the demonstration's state as it was are refused: they cannot change
      behaviour.
    - `split_votes`: the Demo Reader's and the Reviser's finish is refused while roles the
      identification's backends split on (one's outvoted answer showed the other's object) have
      no `order:` line each among one group, so an ambiguous binding is stated
      (`opening.split_refusal`).
    - `dry_run_gate`: an edit that turns a spec the last dry run found reproducing the expert
      into one it finds off is refused, with the gap (`spec_check.dry_run_gate`; unplanned, the
      geometry only), since it moves away from a known match.
    - `identity_guard`: after a review whose verdict is `identification`, edits to the specs the
      Memory has are refused, since their gaps were measured on the wrong objects
      (`imitate.identity_guard`).
    - `settled_guard`: after a review in which no stage diverged and the closes matched and held,
      of an attempt that ended on a finish the official check did not confirm, edits to those
      grasps' parameters are refused, since the grasps were not the fault (`settled`,
      `imitate.settled_guard`).
    - `latch_guard`: after an attempt that ended on such a finish, of a demonstration still
      moving what it held into its success latch, edits that lower, pull back or band a
      placement or lift are refused, since the fault is the stop, not the motion
      (`imitate.latch_guard`).
    - `edit_streak`: after this many revisions in a row edited one target without the next
      attempt passing more of the demonstration's keypoints (the referee's prefix), its further
      edits are refused but a refit, so a stuck target is refit rather than nudged
      (`memory.review.EditStreak`; 0: off).
    - `after_first`: edits to a parameter the review names only at stages after its first
      divergence whose calls the Actor improvised are refused, since later stages follow from
      the first (`memory.review.AfterFirst`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    revert_guard: bool = True
    contact_m: PositiveFloat = 0.02
    held_m: PositiveFloat = 0.04
    spec_m: PositiveFloat = 0.03
    dry_run: bool = True
    contact_deg: PositiveFloat = 30.0
    place_m: PositiveFloat = 0.03
    place_deg: PositiveFloat = 20.0
    near_m: PositiveFloat = 0.15
    fit_approach: bool = True
    lock_passed_objects: bool = True
    noop_edits: bool = True
    split_votes: bool = True
    repeat_guard: bool = True
    reader_dry_run: NonNegativeInt = 2
    dry_run_gate: bool = True
    identity_guard: bool = True
    settled_guard: bool = True
    latch_guard: bool = True
    fit_reverts: bool = True
    edit_streak: NonNegativeInt = 3
    after_first: bool = True


class Gap(BaseModel):
    """Attempt against expert, both relative to the same reference (an object, a target, the
    world): `attempt` and `expert` positions in robot axes (x right, y forward, z up) from the
    reference's center, `offset` their difference (attempt minus expert), `angle_deg` the turn
    between the orientations (a half turn about the pointing axis counts as none for a gripper),
    `twist_deg` the part of it about the gripper's pointing axis (the closing axis turned; folded
    to 0-90), `local` the offset in the reference's own frame."""

    model_config = ConfigDict(frozen=True)

    reference: str
    attempt: tuple[float, float, float]
    expert: tuple[float, float, float]
    offset: tuple[float, float, float]
    distance_m: NonNegativeFloat
    angle_deg: NonNegativeFloat
    twist_deg: NonNegativeFloat | None = None
    local: tuple[float, float, float] | None = None


class Motion(BaseModel):
    """A held motion (see the module): the TCP's displacement (m) and turn (deg, about an axis)
    from the close, in the frame of the object as it was at the close (`reference`; its x, y, z)
    and along the robot's axes (`*_world`): the expert's to its next keypoint (`expert_to`) and to
    its last held one (`expert_end`), the attempt's from its settled close to where the hold broke
    (`attempt`), and to the TCP the breaking call commanded (`commanded`, where it said); the
    opening the hold broke at; how far the attempt's TCP strayed from the expert's held path
    (`path_m`), the expert's held keypoint nearest the break (`nearest`), and how far the object
    moved from the close to the break (`body_moved_m`: an articulated part's body that slid)."""

    model_config = ConfigDict(frozen=True)

    reference: str
    expert_to: str | None = None
    expert: tuple[float, float, float] | None = None
    expert_world: tuple[float, float, float] | None = None
    expert_turn_deg: float | None = None
    expert_axis: tuple[float, float, float] | None = None
    expert_end: str | None = None
    expert_end_local: tuple[float, float, float] | None = None
    expert_end_turn_deg: float | None = None
    attempt: tuple[float, float, float]
    attempt_world: tuple[float, float, float]
    attempt_turn_deg: float
    attempt_axis: tuple[float, float, float] | None = None
    commanded: tuple[float, float, float] | None = None
    commanded_turn_deg: float | None = None
    opening: float | None = None
    path_m: float | None = None
    nearest: str | None = None
    body_moved_m: float | None = None


class Stage(BaseModel):
    """One stage of the demonstration and what the attempt did there (see the module)."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    keypoint: str
    phase: str
    event: Literal["close", "open", "turn", "contact", "move"]
    object: str | None
    attempt_arm: Arm | None = None
    status: Literal["matched", "failed", "missing", "missed", "unreached"] = "matched"
    contact: Gap | None = None
    commanded: Gap | None = None
    world: Gap | None = None
    place: Gap | None = None
    motion: Motion | None = None
    approach_deg: float | None = None
    standoff_m: tuple[float, float | None] | None = None
    opening: float | None = None
    held: bool | None = None
    call: str | None = None
    stage: str | None = None
    """The workflow stage the call's rationale named."""
    receipt: str | None = None
    parameters: list[str] = []
    spec_diff: list[str] = []
    invented: list[str] = []
    """The optional arguments the call passed (not at their default) that its stage's Memory
    call does not name: the Actor added them (`invented_arguments`)."""
    plan: str | None = None
    verdict: Verdict | None = None
    note: str = ""
    handover: bool = False
    """A hand-over of the expert's (the receiving arm's close) the attempt skipped
    (`skipped_handovers`): restore the hand-over stages, not a position."""
    turned_only: bool = False
    """A close at the expert's position, turned beyond `ReviewConfig.contact_deg` only, that held
    is no divergence by itself (`review` blames it only for a later
    failure of its hold, `_turn_blamed`)."""


class EndRow(BaseModel):
    """An object the demonstration moved: where it ended against the demonstration's end."""

    model_config = ConfigDict(frozen=True)

    object: str
    gap: Gap


class Misidentified(BaseModel):
    """An identification error (see the module): the task objects the demonstration acts on that
    one registered object holds (`objects`, the simulator's names), the role it was registered
    for (None: none holds them, `near` the role registered nearest), its width and theirs
    (`measured_m`, `expert_m`: the first object's)."""

    model_config = ConfigDict(frozen=True)

    objects: list[str]
    role: str | None = None
    near: str | None = None
    measured_m: float | None = None
    expert_m: float | None = None


class ExpertDiff(BaseModel):
    """The review: every stage, the first that diverged (`first`, an index into `stages`; None:
    none did), its verdict, and the end state (when no stage diverged); `identification`: the
    identification errors, which make the verdict `identification` (`first` is still the first
    stage that diverged), and `unregistered`: the Memory's roles it registered nothing for."""

    model_config = ConfigDict(frozen=True)

    stages: list[Stage]
    first: int | None
    verdict: Verdict | None
    end: list[EndRow] = []
    success: bool
    identification: list[Misidentified] = []
    unregistered: list[str] = []


@dataclass(frozen=True)
class Call:
    """A model call of the attempt: its turn, tool, arguments, receipt (`result`) and text."""

    turn: int
    name: str
    arguments: Mapping[str, Any]
    result: Mapping[str, Any]
    text: str

    @property
    def reason(self) -> str | None:
        found = self.result.get("reason")
        return str(found) if found is not None else None

    @property
    def stage(self) -> str | None:
        rationale = self.arguments.get("rationale")
        return str(rationale.get("stage")) if isinstance(rationale, Mapping) else None

    def parts(self) -> dict[str, Mapping[str, Any]]:
        """Each arm's action in `together`: its `moves` by arm; none for another call."""
        moves = self.arguments.get("moves")
        if not isinstance(moves, list):
            return {}
        return {
            str(m["arm"]): m["action"]
            for m in moves
            if isinstance(m, Mapping) and isinstance(m.get("action"), Mapping)
        }

    def part(self, arm: str | None) -> Mapping[str, Any] | None:
        """`arm`'s action in `together` (`parts`)."""
        return self.parts().get(arm) if arm else None

    def commanded(self) -> tuple[Arm, Pose] | None:
        """The arm the receipt is about and its commanded TCP, when it says."""
        arm, commanded = self.result.get("arm"), self.result.get("commanded")
        if arm not in ("left", "right") or not isinstance(commanded, Mapping):
            return None
        tcp = commanded.get("tcp")
        if not isinstance(tcp, Mapping):
            return None
        return arm, Pose(p=tuple(tcp["p"]), q=tuple(tcp["q"]))


def attempt_calls(directory: Path) -> list[Call]:
    """The model's calls of an attempt in order (`turns.jsonl`), with their receipts
    (`tool_calls.jsonl`)."""
    results: dict[str, Mapping[str, Any]] = {}
    path = directory / "tool_calls.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                results[str(record.get("call_id"))] = record
    calls: list[Call] = []
    turns = directory / "turns.jsonl"
    if not turns.is_file():
        return calls
    for line in turns.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        turn = json.loads(line)
        for exchange in turn["exchanges"]:
            for call in exchange["calls"]:
                found = results.get(str(call["call_id"]), {})
                content = found.get("content") or []
                text = " ".join(str(c.get("text", "")) for c in content if isinstance(c, Mapping))
                arguments = call["arguments"] if isinstance(call["arguments"], Mapping) else {}
                calls.append(
                    Call(
                        turn=int(turn["turn"]),
                        name=str(call["name"]),
                        arguments=arguments,
                        result=found.get("result") or {},
                        text=text,
                    )
                )
    return calls


@dataclass(frozen=True)
class _Event:
    """An attempt's gripper change: the sample, the arm, close or open, the TCP, the opening
    after it, what it held (`attempt_holds`) and the object it acts on."""

    sample: int
    arm: Arm
    cue: Literal["close", "open"]
    tcp: Pose
    opening: float | None
    held: frozenset[str]
    object: str | None


class _Scene:
    """The attempt's ground truth: object poses at the recorded truth samples."""

    def __init__(self, recorded: Mapping[str, np.ndarray], shapes: Mapping[str, ObjectShape]):
        self.samples = np.asarray(recorded.get("truth.sample", np.zeros(0)), dtype=int)
        self.poses = {
            key[6:]: np.asarray(value, dtype=float)
            for key, value in recorded.items()
            if key.startswith("truth.") and key != "truth.sample"
        }
        self.shapes = shapes

    def pose(self, name: str, sample: int, after: bool = False) -> Pose | None:
        """`name`'s pose at the last truth sample at or before `sample` (`after`: the first at or
        after it, as an object comes to rest after a release)."""
        if name not in self.poses or not len(self.samples):
            return None
        if after:
            later = np.flatnonzero(self.samples >= sample)
            k = int(later[0]) if len(later) else len(self.samples) - 1
        else:
            earlier = np.flatnonzero(self.samples <= sample)
            k = int(earlier[-1]) if len(earlier) else 0
        return Pose.from_list(self.poses[name][k])

    def nearest(self, point: Array, sample: int, within: float) -> str | None:
        """The object whose center is nearest `point`, where it is at `sample` or where it was
        at the start (a close on air where an object stood before a push), within `within`."""
        best, gap = None, within
        for name in self.poses:
            for at in (sample, 0):
                pose = self.pose(name, at)
                assert pose is not None
                shape = self.shapes.get(name)
                distance = float(np.linalg.norm(center(pose, shape) - point))
                if shape is not None and shape.size is not None:  # from its outline, not center
                    distance = max(0.0, distance - shape.size[0] / 2)
                if distance < gap:
                    best, gap = name, distance
        return best

    def moved(self, name: str, sample: int) -> float:
        """How far `name` moved from its start by `sample`."""
        now, start = self.pose(name, sample), self.pose(name, 0)
        if now is None or start is None:
            return 0.0
        return float(np.linalg.norm(np.subtract(now.p, start.p)))


def center(pose: Pose, shape: ObjectShape | None) -> Array:
    """An object's center (its shape's, else its pose's position) in the world."""
    offset = np.zeros(3) if shape is None else quat_rotate(pose.q, shape.center)
    return np.asarray(pose.p, dtype=float) + offset


def attempt_events(
    recorded: Mapping[str, np.ndarray],
    scene: _Scene,
    arms: Sequence[Arm],
    threshold: float,
    near_m: float,
) -> list[_Event]:
    """The attempt's gripper changes, in sample order."""
    truth = {key: value for key, value in recorded.items() if key.startswith("truth.")}
    events = []
    for arm in arms:
        gripper = np.asarray(recorded[f"{arm}.gripper_cmd"], dtype=float)
        tcp = np.asarray(recorded[f"{arm}.tcp"], dtype=float)
        opening = np.asarray(recorded.get(f"{arm}.gripper", np.full(len(gripper), np.nan)))
        source = np.arange(len(gripper), dtype=float)
        holds = attempt_holds(gripper, source, tcp, truth, threshold) if truth else {}
        closed = gripper < threshold
        for c in np.flatnonzero(closed[1:] != closed[:-1]) + 1:
            held = holds.get(int(c), frozenset())
            obj = sorted(held)[0] if len(held) == 1 else None
            if obj is None:
                obj = scene.nearest(tcp[c, :3], int(c), near_m)
            measured = float(opening[c])
            events.append(
                _Event(
                    sample=int(c),
                    arm=arm,
                    cue="close" if closed[c] else "open",
                    tcp=Pose.from_list(tcp[c]),
                    opening=None if math.isnan(measured) else measured,
                    held=held,
                    object=obj,
                )
            )
    return sorted(events, key=lambda e: (e.sample, e.arm))


def _relative(reference: Pose, shape: ObjectShape | None, pose: Pose) -> tuple[Array, Array]:
    """`pose` relative to an object: the position from its center in robot axes, and the
    orientation in its frame."""
    offset = np.asarray(pose.p) - center(reference, shape)
    return offset, quat_multiply(quat_conjugate(reference.q), pose.q)


def _twist_deg(a: Array, b: Array, axis: Array) -> float:
    """The turn about `axis` (in the frame of `a`) from orientation `a` to `b`, folded to 0-90
    (a parallel gripper closes the same way turned half round)."""
    r = quat_multiply(quat_conjugate(a), b)
    angle = abs(math.degrees(2 * math.atan2(float(np.dot(r[1:], axis)), float(r[0]))))
    angle %= 180.0
    return min(angle, 180.0 - angle)


def contact_gap(
    name: str,
    shape: ObjectShape | None,
    expert_obj: Pose,
    expert_tcp: Pose,
    attempt_obj: Pose,
    attempt_tcp: Pose,
    pointing: Sequence[float],
) -> Gap:
    """A TCP against the expert's, each relative to the object it acts on (its pose then); about
    a symmetric object's vertical axis the expert's is turned to the attempt's side first, as the
    referee does."""
    symmetric = shape is not None and shape.symmetric
    if symmetric:  # its own turn about the vertical is arbitrary: world axes at its center
        expert_obj = Pose(p=expert_obj.p, q=WORLD.q)
        attempt_obj = Pose(p=attempt_obj.p, q=WORLD.q)
    e_off, e_q = _relative(expert_obj, shape if not symmetric else None, expert_tcp)
    a_off, a_q = _relative(attempt_obj, shape if not symmetric else None, attempt_tcp)
    if symmetric and np.hypot(*e_off[:2]) > 1e-3:
        yaw = math.atan2(a_off[1], a_off[0]) - math.atan2(e_off[1], e_off[0])
        turn = quat_from_axis_angle([0.0, 0.0, 1.0], yaw)
        e_off = quat_rotate(turn, e_off)
        e_q = quat_multiply(turn, e_q)  # the expert's TCP turned about the vertical with it
    offset = a_off - e_off
    axis = np.asarray(pointing, dtype=float)
    local = quat_rotate(quat_conjugate(attempt_obj.q), offset)
    return Gap(
        reference=name,
        attempt=_vec(a_off),
        expert=_vec(e_off),
        offset=_vec(offset),
        distance_m=float(np.linalg.norm(offset)),
        angle_deg=float(gripper_angle_deg(a_q, e_q, _vec(pointing))),
        twist_deg=_twist_deg(e_q, a_q, axis / np.linalg.norm(axis)),
        local=_vec(local),
    )


def object_gap(
    name: str,
    target: str | None,
    shapes: Mapping[str, ObjectShape],
    expert: tuple[Pose, Pose | None],
    attempt: tuple[Pose, Pose | None],
    start: Pose | None = None,
) -> Gap:
    """An object's pose against the expert's, relative to its target (what it rests on; None:
    the world): the center's offset from the target's center, and its turn in the target's frame
    (about the vertical only, for an object symmetric about it)."""
    shape = shapes.get(name)
    target_shape = shapes.get(target) if target is not None else None

    def relative(pose: Pose, base: Pose | None) -> tuple[Array, Array]:
        here = center(pose, shape)
        if base is None:
            return here, np.asarray(pose.q, dtype=float)
        return here - center(base, target_shape), quat_multiply(quat_conjugate(base.q), pose.q)

    e_off, e_q = relative(*expert)
    a_off, a_q = relative(*attempt)
    angle = float(_angle(e_q, a_q))
    if shape is not None and shape.symmetric and start is not None:
        angle = _tilt_deg(start, expert[0], attempt[0])
    offset = a_off - e_off
    return Gap(
        reference=target or "world",
        attempt=_vec(a_off),
        expert=_vec(e_off),
        offset=_vec(offset),
        distance_m=float(np.linalg.norm(offset)),
        angle_deg=angle,
    )


def _angle(a: Array, b: Array) -> float:
    r = quat_multiply(quat_conjugate(a), b)
    return math.degrees(2 * math.atan2(float(np.linalg.norm(r[1:])), abs(float(r[0]))))


def _tilt_deg(start: Pose, expert: Pose, attempt: Pose) -> float:
    """How far apart two poses hold a symmetric object's axis (the one vertical at `start`; its
    turn about that axis is free)."""
    axis = quat_rotate(quat_conjugate(start.q), [0.0, 0.0, 1.0])
    e, a = quat_rotate(expert.q, axis), quat_rotate(attempt.q, axis)
    cos = float(np.clip(np.dot(e, a) / (np.linalg.norm(e) * np.linalg.norm(a)), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _vec(v: Sequence[float] | Array) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in v)
    return x, y, z


def spec_parameters(doc: MemoryDoc | None, call: Call) -> list[str]:
    """The Memory parameters behind a call: specs whose value it passed (fields other than
    `SPEC_IGNORED` equal), numbers it passed by value, else the specs its stage references."""
    if doc is None:
        return []
    passed: list[Mapping[str, Any]] = []
    numbers: list[float] = []
    named: list[str] = []

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, str) and key in SPEC_ARGUMENTS:
            name = value.strip("{}")
            param = doc.params.get(name)
            if param is not None and param.kind.endswith("_spec") and name not in named:
                named.append(name)  # by name: the stored spec, exactly
        elif isinstance(value, Mapping):
            if key != "rationale":
                passed.append(value)
                for k, v in value.items():
                    walk(v, k)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and key.endswith("_m"):
            numbers.append(float(value))

    walk(call.arguments)
    found = [f"{name} ({doc.params[name].kind}, by name)" for name in named]
    for name, param in doc.params.items():
        if not isinstance(param.value, BaseModel) or not param.kind.endswith("_spec"):
            continue
        value = param.value.model_dump(mode="json")
        for given in passed:
            shared = [k for k in value if k in given and k not in SPEC_IGNORED]
            if len(shared) >= 3 and all(_same(value[k], given[k]) for k in shared):
                found.append(f"{name} ({param.kind})")
                break
    if not found and call.stage is not None:
        stages = [s for s in doc.workflow if s.name == call.stage]
        for stage in stages:
            for name in sorted(template_refs(stage.call, doc.params)):
                if doc.params[name].kind.endswith("_spec"):
                    found.append(f"{name} ({doc.params[name].kind}, by its stage)")
    for name, param in doc.params.items():
        if param.kind == "number" and any(abs(float(param.value) - n) < 1e-9 for n in numbers):
            found.append(f"{name} (number)")
    return found


_KINDS: Final = {"grasp": GraspSpec, "place": PlaceSpec, "hinge": HingeSpec, "path": PathSpec}
SPEC_TOLERANCE: Final = {"deg": 0.5, "other": 0.002}
"""How far a call's spec number may lie from the Memory's and still be the Memory's (the
Memory's rounding, a refit's noise): degrees for `*_deg` fields, else (meters, unit-vector and
quaternion components, fractions) the other."""
ALIAS: Final = re.compile(r"^obj_[0-9]+$")


def _differs(memory_spec: BaseModel, written: BaseModel) -> list[tuple[str, str, str]]:
    """The fields the call's spec differs from the Memory's in beyond `SPEC_TOLERANCE` (the arm
    aside; a target the call names by an object's alias compared by its other fields)."""
    out = []
    for field, mine, theirs in field_diff(memory_spec, written, skip=("arm",)):
        a, b = getattr(memory_spec, field), getattr(written, field)
        tol = SPEC_TOLERANCE["deg" if field.endswith("_deg") else "other"]
        close = (
            _numbers(a)
            and _numbers(b)
            and np.shape(a) == np.shape(b)
            and float(np.max(np.abs(np.subtract(a, b)))) <= tol
        )
        aliased = (
            isinstance(a, PlaceTarget)
            and isinstance(b, PlaceTarget)
            and ALIAS.match(b.obj or "") is not None
            and a.model_copy(update={"obj": b.obj}) == b
        )
        if not close and not aliased:
            out.append((field, mine, theirs))
    return out


def _numbers(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    return isinstance(value, tuple | list) and all(_numbers(v) for v in value)  # a path's poses


def spec_diff(doc: MemoryDoc | None, call: Call | None, arm: Arm | None) -> list[str]:
    """Where the spec `call` passed for `arm` (written out) differs from the Memory's spec behind
    it, field by field (the arm aside: the acting arm mirrors a spec): the Memory spec its
    workflow stage passes of that kind nearest it (the fewest fields apart). Empty for a spec
    passed by name (it ran as stored), or where the call's stage passes no spec of that kind (a
    spec of another stage, such as another object's grasp, is not what the call transcribed)."""
    found = _written(doc, call, arm)
    if found is None or not found[2]:
        return []
    assert doc is not None
    _, written, staged = found
    diffs = {n: _differs(doc.params[n].value, written) for n in staged}
    name = min(staged, key=lambda n: len(diffs[n]))
    return [
        f"{name}.{field}: the call passed {theirs}, the Memory holds {mine}"
        for field, mine, theirs in diffs[name]
    ]


def _written(
    doc: MemoryDoc | None, call: Call | None, arm: Arm | None
) -> tuple[str, BaseModel, list[str]] | None:
    """The spec `call` wrote out for `arm` (the first of its spec arguments that is no name):
    its kind's parameter label, the spec, and the Memory specs of that kind its workflow stage
    passes; None without one (or one that does not validate)."""
    if doc is None or call is None:
        return None
    for scope in (call.part(arm), call.arguments):
        if not isinstance(scope, Mapping):
            continue
        for key, kind in _KINDS.items():
            given = scope.get(key)
            if not isinstance(given, Mapping):
                continue
            try:
                written = kind.model_validate(given)
            except ValueError:
                return None
            label = f"{key}_spec"
            staged = [
                n
                for st in doc.workflow
                if st.name == call.stage
                for n in template_refs(st.call, doc.params)
                if n in doc.params and doc.params[n].kind == label
            ]
            return label, written, staged
    return None


Refused = Callable[[int], Mapping[str, Call]]
"""The Memory specs the attempt passed by name in a call refused before it moved (`REFUSED`), in
turns before the given one: each with the first such call."""


def named_specs(call: Call) -> list[str]:
    """The Memory specs a call passed by name (`{name}`), its own arguments or an arm's in
    `together`."""
    found: list[str] = []
    for scope in (call.arguments, *call.parts().values()):
        if not isinstance(scope, Mapping):
            continue
        for key in SPEC_ARGUMENTS:
            value = scope.get(key)
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                found.append(value.strip("{}"))
    return found


PLACING_FIELDS: Final[dict[type[BaseModel], frozenset[str]]] = {
    GraspSpec: frozenset(
        {
            "part",
            "anchor",
            "offset_m",
            "offset_frac",
            "closing",
            "approach",
            "tilt_deg",
            "yaw_deg",
            "lean_deg",
            "purpose",
            "x_toward",
            "approach_axis",  # a pose grasp's (null otherwise, so equal)
            "closing_axis",
        }
    ),
    PlaceSpec: frozenset(
        {
            "target",
            "frame",
            "offset_m",
            "orient",
            "yaw_deg",
            "rpy_deg",
            "placement_point",
            "placement_point_at",
            "region_m",
            "x_toward",
        }
    ),
    HingeSpec: frozenset({"part", "side", "at", "offset_m"}),
    PathSpec: frozenset({"poses", "x_toward"}),
}
"""The fields of a spec that place its contact (or placement, or the hinge or path it follows):
a spec written out is the Memory's only where every one of them lies within `SPEC_TOLERANCE` of
the stored value. `x_toward` turns the part frame, so it is one of them, with one exception
(`narrowed`): a stored spec without it tries the part frame's turns, and a call that names one
picks among those turns, so it cannot put a stored spec's contact somewhere the stored spec
would not try. A grasp's only: a placement's null `x_toward` is its frame as
measured."""
TRANSCRIBED_FIELDS: Final = 1
"""A spec written out is the Memory's (transcribed) when its placing fields (`PLACING_FIELDS`)
match the stored spec's and at most this many of its other fields but `SPEC_IGNORED` lie apart
from it beyond `SPEC_TOLERANCE`: an Actor copying a Memory line drops or adds a hint, while a
spec it writes in its place moves the contact, which even one placing field does."""


def written_specs(call: Call) -> list[tuple[str, BaseModel]]:
    """The specs a call wrote out (that validate), its own arguments or an arm's in `together`:
    each with its argument's name (`grasp`, `place`, `hinge`, `path`)."""
    found: list[tuple[str, BaseModel]] = []
    for scope in (call.arguments, *call.parts().values()):
        if not isinstance(scope, Mapping):
            continue
        for key, kind in _KINDS.items():
            given = scope.get(key)
            if not isinstance(given, Mapping):
                continue
            try:
                found.append((key, kind.model_validate(given)))
            except ValueError:
                continue
    return found


def narrowed(stored: BaseModel, written: BaseModel, field: str) -> bool:
    """Whether `written` differs from `stored` in `field` only by naming the `x_toward` the
    stored grasp leaves open (see `PLACING_FIELDS`); a placement's or a path's null `x_toward`
    is the frame as measured, not either way, so naming one there turns it."""
    return (
        field == "x_toward"
        and isinstance(stored, GraspSpec)
        and getattr(stored, field, None) is None
        and getattr(written, field, None) is not None
    )


def transcribed(stored: BaseModel, written: BaseModel) -> bool:
    """Whether `written` is the stored spec `stored` written out: every placing field within
    `SPEC_TOLERANCE` (`PLACING_FIELDS`, an `x_toward` it only narrows aside) and at most
    `TRANSCRIBED_FIELDS` other fields apart, `SPEC_IGNORED` aside."""
    if type(written) is not type(stored):
        return False
    placing = PLACING_FIELDS[type(stored)]
    apart = [f for f, *_ in _differs(stored, written) if f not in SPEC_IGNORED]
    if any(f in placing and not narrowed(stored, written, f) for f in apart):
        return False
    return len(apart) <= TRANSCRIBED_FIELDS


def spec_changes(stored: BaseModel, written: BaseModel) -> list[str]:
    """Where `written` differs from the stored spec beyond `SPEC_TOLERANCE` (`SPEC_IGNORED`
    aside): `field stored -> written`."""
    return [
        f"{field} {mine} -> {theirs}"
        for field, mine, theirs in _differs(stored, written)
        if field not in SPEC_IGNORED
    ]


def memory_specs(doc: MemoryDoc, call: Call) -> list[str]:
    """The Memory specs a call passed: by name (`named_specs`), or written out as stored
    (`transcribed`), its own arguments or an arm's in `together`."""
    found = [name for name in named_specs(call) if name in doc.params]
    for _, written in written_specs(call):
        for name, param in doc.params.items():
            if name in found or not isinstance(param.value, type(written)):
                continue
            if transcribed(param.value, written):
                found.append(name)
    return found


def _transcribed(
    verdict: Verdict | None,
    update: dict[str, Any],
    call: Call | None,
    refused: Refused,
    memory: MemoryDoc | None,
    arm: Arm | None,
) -> Verdict | None:
    """A stage off with a call whose spec differs from its stage's Memory spec is a
    transcription, unless the attempt had passed that spec by name and had it refused: then the
    stored spec is what failed (`memory_diverged`, the refusal in the note). A spec the call
    wrote where its stage passes none of that kind is said in the note: no Memory spec stands
    behind it to move."""
    if extra := invented_arguments(memory, call):
        update["invented"] = extra
    if verdict != "memory_diverged":
        return verdict
    if not update["spec_diff"]:
        found = _written(memory, call, arm)
        if found is not None and not found[2]:
            note = (
                f"the call wrote its {found[0]} out and its workflow stage "
                f"({call.stage if call else None}) passes no {found[0]} of the Memory: no "
                "stored spec stands behind this stage (a demonstration fit for this keypoint "
                "would)"
            )
            update["note"] = "; ".join(filter(None, [update.get("note"), note]))
        return verdict
    name = str(update["spec_diff"][0]).split(".", 1)[0]
    before = refused(_turn(call)).get(name)
    if before is None:
        return "transcription"
    note = (
        f"the Memory's {name}, passed by name to {before.name} in turn {before.turn}, was refused "
        f"({before.reason}): the stored spec is what failed, and this call wrote another after it"
    )
    update["note"] = "; ".join(filter(None, [update.get("note"), note]))
    return verdict


_ACTIONS: Final[dict[str, type[BaseModel]]] = {
    get_args(model.model_fields["action"].annotation)[0]: model
    for model in get_args(get_args(ArmAction)[0])
}
"""The action tools' argument models by tool name, for their defaults."""


def invented_arguments(memory: MemoryDoc | None, call: Call | None) -> list[str]:
    """The optional arguments `call` passed at other than their default that its stage's Memory
    call (the workflow stage its rationale names) does not name for that tool."""
    if memory is None or call is None or (model := _ACTIONS.get(call.name)) is None:
        return []
    stage = next((st for st in memory.workflow if st.name == call.stage), None)
    named = None if stage is None else call_arguments(stage.call).get(call.name)
    if named is None:
        return []
    fields = model.model_fields
    return [
        key
        for key, value in call.arguments.items()
        if value is not None
        and key not in named
        and key in fields
        and not fields[key].is_required()
        and not _same(value, fields[key].get_default(call_default_factory=True))
    ]


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(a == b)


@dataclass(frozen=True)
class _Expert:
    arm: Arm
    kp: Keypoint
    kind: str
    object: str | None
    onto: frozenset[str]
    standoff: Keypoint | None


def expert_stages(
    keypoints: KeypointSet,
    layer: DemoTruth,
    shapes: Mapping[str, ObjectShape],
    referee: RefereeConfig,
) -> list[_Expert]:
    """The demonstration's stages in frame order: gripper events, turned lifts and carries, and
    its contact."""
    poses = {frame.frame: frame.poses for frame in layer.frames}
    out = []
    for arm, kps in keypoints.arms.items():
        objects = acted_on(kps, poses, shapes)
        touches = keypoint_objects(kps, poses, shapes)
        kind = kinds(kps, keypoints.success_frame, keypoints.pointing[arm], referee)
        for j, kp in enumerate(kps):
            if kp.cue not in ("close", "open") and kind[j] not in ("turn", "contact"):
                continue
            if kp.cue == "close" and kp.phase == "preshape":
                continue  # a close on nothing before moving: no stage
            standoff = kps[j - 1] if j > 0 and kps[j - 1].phase == "approach" else None
            onto = touches[j].onto
            if kp.cue == "open" and not onto and j > 0:
                onto = touches[j - 1].onto  # the placement before the release
            out.append(_Expert(arm, kp, kind[j], objects[j], onto, standoff))
    return sorted(out, key=lambda e: (e.kp.frame, e.arm))


async def review(
    keypoints: KeypointSet,
    layer: DemoTruth,
    shapes: Mapping[str, ObjectShape],
    recorded: Mapping[str, np.ndarray],
    calls: Sequence[Call],
    progress: Mapping[Arm, ArmProgress] | None,
    success: bool,
    memory: MemoryDoc | None,
    *,
    referee: RefereeConfig,
    config: ReviewConfig,
    grip_threshold: float,
    holding: Mapping[Arm, float],
    plan: PlanCheck | None = None,
    unreproducible: Mapping[str, str] | None = None,
    registry: SceneRegistry | None = None,
) -> ExpertDiff:
    """The attempt against the expert (see the module). `recorded`: the attempt's
    `physics.npz`; `progress`: the referee's per arm (its missed turns); `plan`: the plan check
    (None: not run); `unreproducible`: the placement specs the spec dry run found no constrain
    reproduces on the demonstration's state, each with its gaps by constrain
    (`spec_check.unreproducible`); `registry`: the attempt's first, for its identification
    (None: not checked); `holding`: per arm, the measured opening at or above which a close
    holds something (the embodiment's `Gripper.holding_threshold`)."""
    poses = {frame.frame: frame.poses for frame in layer.frames}
    scene = _Scene(recorded, shapes)
    # every arm the attempt recorded: one the expert left idle may still act (the wrong arm)
    arms: list[Arm] = [k.split(".")[0] for k in recorded if k.endswith(".gripper_cmd")]  # type: ignore[misc]
    events = attempt_events(recorded, scene, arms, grip_threshold, config.near_m)
    turns = np.asarray(recorded.get("turn", np.zeros(1)), dtype=int)
    experts = expert_stages(keypoints, layer, shapes, referee)
    used: set[int] = set()  # attempt events matched
    closed_by: dict[tuple[Arm, str | None], _Event] = {}  # matched closes per expert arm, object
    last = 0  # the sample of the last matched event

    def start_of(turn: int) -> int:
        before = np.flatnonzero(turns < turn)
        return int(before[-1]) if len(before) else 0

    charged: set[int] = set()  # turns of calls a held-motion stage was charged with

    def failing_after(sample: int) -> tuple[Call, int] | None:
        """The first hard failure after `sample` that no held-motion stage was charged with,
        with the sample its turn started from."""
        for call in calls:
            if call.turn in charged:
                continue
            if call.reason in HARD and (start := start_of(call.turn)) >= sample:
                return call, start
        return None

    def lost_between(arm: Arm, start: int, end: int | None, own: bool) -> list[Call]:
        """The calls whose receipts say `arm` lost its hold, in turns starting from `start` up
        to (not with) `end` (None: to the end): those `arm` acted in (`own`), or those only
        another arm acted in (a hold knocked loose by the other arm's call)."""
        return [
            c
            for c in calls
            if c.reason == "lost_hold"
            and c.result.get("arm") == arm
            and (arm in _acting(c)) == own
            and start_of(c.turn) >= start
            and (end is None or start_of(c.turn) < end)
        ]

    def next_event(sample: int, arm: Arm) -> int | None:
        """The sample of `arm`'s next close after `sample` (its next hold)."""
        found = [
            ev.sample for ev in events if ev.sample > sample and ev.arm == arm and ev.cue == "close"
        ]
        return min(found) if found else None

    def call_at(sample: int) -> Call | None:
        """The model's action in the turn that recorded `sample`."""
        turn = int(turns[sample]) if sample < len(turns) else -1
        acting = [c for c in calls if c.turn == turn and c.result.get("reason") is not None]
        return acting[-1] if acting else None

    def placing(arm: Arm, turn: int, since: int = 0) -> Call | None:
        """The last call up to `turn` that placed with `arm` (passed a placement spec), in a turn
        after the one that recorded sample `since` (the close that took what it releases: a
        placement before it moved another object)."""
        after = int(turns[since]) if 0 < since < len(turns) else -1
        found = [
            c
            for c in calls
            if after < c.turn <= turn
            and arm in _acting(c)
            and any(
                isinstance(scope, Mapping) and "place" in scope
                for scope in (c.arguments, c.part(arm))
            )
        ]
        return found[-1] if found else None

    def refused(turn: int) -> dict[str, Call]:
        found: dict[str, Call] = {}
        for c in calls:
            if c.turn < turn and c.reason in REFUSED:
                for name in named_specs(c):
                    found.setdefault(name, c)
        return found

    holds = _holds(events)

    def displacing(e: _Expert, obj: str | None, before: int) -> Stage | None:
        """The `displaced` stage (see the module) for `obj` at `e`'s stage, whose call's turn
        starts after sample `before` (`start_of`); None when it lay where the expert had it, or
        no earlier call moved it away while unheld."""
        expert = poses.get(e.kp.frame, {}).get(obj) if obj is not None else None
        now = scene.pose(obj, before) if obj is not None else None
        if obj is None or expert is None or now is None or _displaced(expert, now) < MOVED_M:
            return None
        goal = np.asarray(expert.p, dtype=float)
        best: tuple[float, int, Pose, Pose] | None = None
        for turn in sorted({int(turns[int(t)]) for t in scene.samples if 0 < int(t) <= before}):
            span = np.flatnonzero(turns == turn)
            first, last = int(span[0]), min(int(span[-1]), before)
            if any(a <= last and first < b for a, b in holds.get(obj, [])):
                continue  # an arm of the attempt held it during the turn
            start, end = scene.pose(obj, first - 1), scene.pose(obj, last)
            assert start is not None and end is not None
            away = float(np.linalg.norm(np.subtract(end.p, goal))) - float(
                np.linalg.norm(np.subtract(start.p, goal))
            )
            if away >= MOVED_M and (best is None or away > best[0]):
                best = (away, last, start, end)
        if best is None:
            return None
        _, last, start, end = best
        call = call_at(last)
        if call is None:
            return None
        moved = np.subtract(end.p, start.p)
        stage = f", stage {call.stage}" if call.stage else ""
        note = (
            f"{obj} moved {100 * float(np.linalg.norm(moved)):.1f} cm {_cm(moved)} during "
            f"{call.name} (turn {call.turn}{stage}) while no gripper held it, away from where the "
            f"expert had it at {e.arm} {e.kp.name}; it lay "
            f"{100 * _displaced(expert, now):.1f} cm from there when that stage came"
        )
        return Stage(
            arm=e.arm,
            keypoint=f"before {e.kp.name}",
            phase="unheld",
            event="move",
            object=obj,
            status="failed",
            call=call.name,
            receipt=call.reason,
            parameters=spec_parameters(memory, call),
            verdict="displaced",
            note=note,
        )

    before_row: dict[int, Stage] = {}  # a `displaced` stage, by the row it goes before

    async def planned(arm: Arm, tcp: Pose, sample: int) -> str:
        if plan is None:
            return "not checked"
        why = await plan(arm, tcp, sample)
        if why is None:
            return "plans from the attempt's state"
        if sample > 0 and (first := await plan(arm, tcp, 0)) is None:
            return f"does not plan from the attempt's state ({why}); plans from its start"
        return f"does not plan from the attempt's state ({why})" + (
            "" if sample == 0 else f" nor from its start ({first})"
        )

    rows: list[Stage | None] = []  # None: not reached yet (decided below)
    pending: list[tuple[int, _Expert, Stage]] = []
    for e in experts:
        kp, obj = e.kp, e.object
        base = Stage(
            arm=e.arm,
            keypoint=kp.name,
            phase=kp.phase,
            event="close"
            if kp.cue == "close"
            else "open"
            if kp.cue == "open"
            else ("turn" if e.kind == "turn" else "contact"),
            object=obj,
        )
        if kp.cue == "close":
            # the attempt's closes on the object, and the expert's arm's closes on nothing near
            # any object (fingers shut in the air short of it)
            mine = [
                ev
                for i, ev in enumerate(events)
                if ev.cue == "close"
                and i not in used
                and (ev.object == obj or (ev.object is None and ev.arm == e.arm))
            ]
            # a close of the expert's arm first (two arms on one object: each its own)
            match = next((ev for ev in mine if ev.arm == e.arm), mine[0] if mine else None)
            if match is None:
                pending.append((len(rows), e, base))
                rows.append(None)
                continue
            used.add(events.index(match))
            last = max(last, match.sample)
            closed_by[(e.arm, obj)] = match
            track = np.asarray(recorded[f"{match.arm}.tcp"], dtype=float)
            hold = _hold(recorded, turns, match, events, holding[match.arm])
            row, breaking = await _close(
                base,
                e,
                match,
                track,
                poses,
                scene,
                shapes,
                keypoints,
                config,
                call_at,
                planned,
                memory,
                lost_between,
                next_event,
                start_of,
                hold,
                refused,
            )
            closing = call_at(match.sample)
            begin = start_of(closing.turn) if closing is not None else match.sample
            # a close off the expert's on an object an earlier call had moved away: that call's
            if row.verdict not in (None, "wrong_arm") and (moved := displacing(e, obj, begin)):
                before_row[len(rows)] = moved
            rows.append(row)
            # with the other arm the close is already the wrong arm's, and its track is not the
            # expert's arm's to compare a held motion with
            if breaking is not None and match.arm == e.arm:
                charged.add(breaking.turn)
                rows.append(
                    _held_motion(
                        e,
                        match,
                        breaking,
                        hold,
                        track,
                        poses,
                        scene,
                        keypoints,
                        memory,
                        config,
                        turns,
                    )
                )
        elif kp.cue == "open":
            grasp = closed_by.get((e.arm, obj)) or next(
                (ev for (_, o), ev in closed_by.items() if o == obj), None
            )
            after = [
                (i, ev)
                for i, ev in enumerate(events)
                if ev.cue == "open"
                and i not in used
                and grasp is not None
                and ev.arm == grasp.arm
                and ev.sample > grasp.sample
            ]
            if not after:
                pending.append((len(rows), e, base))
                rows.append(None)
                continue
            i, match = after[0]
            used.add(i)
            last = max(last, match.sample)
            track = np.asarray(recorded[f"{match.arm}.tcp"], dtype=float)
            # where the released object came to rest: before the next gripper event, else the end
            later = [ev.sample for ev in events if ev.sample > match.sample]
            rest_at = min(later) - 1 if later else len(turns) - 1
            rows.append(
                _open(
                    base,
                    e,
                    match,
                    track,
                    rest_at,
                    poses,
                    scene,
                    shapes,
                    keypoints,
                    config,
                    call_at,
                    functools.partial(placing, since=grasp.sample),
                    memory,
                    refused,
                    grasp.sample,
                )
            )
        else:
            status = None
            if progress is not None and e.arm in progress:
                index = kp.index - 1
                found = progress[e.arm].keypoints
                status = found[index].status if index < len(found) else None
            if status is None and e.kind == "contact" and e.object is not None:
                # a contact past the arm's last keypoint (`demos.relation.contacts`): no referee
                # keypoint, so the review compares the held object against what it meets
                rows.append(_held_contact(base, e, poses, scene, shapes, config))
            elif status == "missed":
                rows.append(_missed(base, e, keypoints, progress))
            elif status == "pending" and not success:
                pending.append((len(rows), e, base))
                rows.append(None)
            else:
                rows.append(base)
    # The stages the attempt did not reach: the first hard failure after the last stage it did
    # belongs to the first of them its arm does (the expert's other arm may have had a stage in
    # between that the failure cut off), else to the first; without one they were skipped.
    failed = failing_after(last)
    blamed = None
    if failed is not None and pending:
        arm = failed[0].result.get("arm")
        blamed = next((j for j, e, _ in pending if e.arm == arm), pending[0][0])
    for j, e, base in pending:
        if j == blamed:
            assert failed is not None
            cut = await _failed(
                base, e, failed, poses, scene, shapes, planned, memory, keypoints, config, refused
            )
            rows[j] = cut
            if cut.verdict != "wrong_arm" and (moved := displacing(e, _reference(e), failed[1])):
                before_row[j] = moved
        elif failed is not None or charged:  # cut off by a failure, or a hold that broke
            rows[j] = base.model_copy(update={"status": "unreached"})
        else:
            rows[j] = base.model_copy(
                update={"status": "missing", "verdict": "missing_stage", "note": "never done"}
            )
    done = [
        stage
        for j, row in enumerate(rows)
        if row is not None
        for stage in ([before_row[j]] if j in before_row else []) + [row]
    ]
    done = [_reproducible(row, unreproducible or {}) for row in done]
    done = skipped_handovers(done, experts, events)
    first = next((i for i, row in enumerate(done) if row.verdict is not None), None)
    if first is not None and (turned := _turn_blamed(done, first)) is not None:
        done, first = turned
    end: list[EndRow] = []
    verdict = done[first].verdict if first is not None else None
    if first is None and not success:
        end = end_rows(keypoints, layer, shapes, scene, config, experts)
        if end:
            verdict = "end_state_gap"
    wrong: list[Misidentified] = []
    unregistered: list[str] = []
    if registry is not None and not success:
        roles = [line.role for line in parse_objects(memory.objects)[0]] if memory else []
        acted = {e.object for e in experts if e.object is not None}
        acted |= {o for e in experts for o in e.onto}
        grasped = {e.object for e in experts if e.kp.cue == "close" and e.object is not None}
        wrong = identification(registry, scene, shapes, sorted(acted), grasped)
        names = {record.name for record in registry.objects.values()}
        # the roles the workflow or its specs use (a placement's target is in its spec)
        used = "" if memory is None else memory.model_dump_json(include={"workflow", "params"})
        unregistered = [
            role
            for role in roles
            if role not in names and re.search(rf"\b{re.escape(role)}\b", used)
        ]
        if wrong or unregistered:
            verdict = "identification"
    return ExpertDiff(
        stages=done,
        first=first,
        verdict=verdict,
        end=end,
        success=success,
        identification=wrong,
        unregistered=unregistered,
    )


def identification(
    registry: SceneRegistry,
    scene: "_Scene",
    shapes: Mapping[str, ObjectShape],
    acted: Sequence[str],
    grasped: Collection[str],
) -> list[Misidentified]:
    """The identification errors of an attempt's first `registry` (see the module): each task
    object the demonstration acts on (`acted`) goes to the registered object nearest it at the
    start among those whose center lies within `diagnosis.MATCH_M` of its (horizontally), else
    among those whose start box holds its center (in 3D: a block resting on another is not in
    it); a registered object holding two of them, or one of a width off theirs (`IDENTITY_RATIO`,
    `IDENTITY_M`), is an error, and so is one the expert grasps (`grasped`) that none holds nor
    overlaps as `diagnosis.match` would (a registration of a large object's front face); one it
    only places onto may lie off the table, where no camera sees it (a role left unregistered
    is `ExpertDiff.unregistered`). Objects without a shape with a size are not
    checked."""
    records = list(registry.objects.values())
    starts = {r.id: start_center(r) for r in records}
    boxes = {r.id: start_box(r) for r in records}
    held: dict[str, list[str]] = {}
    out: list[Misidentified] = []
    for name in acted:
        shape, pose = shapes.get(name), scene.pose(name, 0)
        if pose is None or shape is None or shape.size is None:
            continue
        at = center(pose, shape)
        flat = {r.id: float(np.linalg.norm((at - starts[r.id])[:2])) for r in records}
        inside = [
            r.id for r in records if np.all(at >= boxes[r.id][0]) and np.all(at <= boxes[r.id][1])
        ]
        near = [r.id for r in records if flat[r.id] <= MATCH_M]
        found = min(near or inside, key=flat.__getitem__, default=None)
        if found is not None:
            held.setdefault(found, []).append(name)
            continue
        box = truth_box(at, shape)
        if box is not None and any(overlap(box, b)[1] >= CONTAINED for b in boxes.values()):
            continue  # registered in part: its size is not comparable
        if name not in grasped:
            continue
        nearest = min(records, key=lambda r: flat[r.id], default=None)
        out.append(
            Misidentified(
                objects=[name], near=None if nearest is None else nearest.name,
                expert_m=shape.size[0],
            )
        )  # fmt: skip
    by_id = {r.id: r for r in records}
    for obj, names in held.items():
        record = by_id[obj]
        measured = width(np.asarray(record.points))
        expert = shapes[names[0]].size
        assert expert is not None
        larger = measured >= IDENTITY_RATIO * expert[0] and measured - expert[0] >= IDENTITY_M
        smaller = (
            not record.partially_visible
            and measured * IDENTITY_RATIO <= expert[0]
            and expert[0] - measured >= IDENTITY_M
        )
        if len(names) > 1 or larger or smaller:
            out.append(
                Misidentified(objects=names, role=record.name, measured_m=measured,
                              expert_m=expert[0])
            )  # fmt: skip
    return out


def handovers(experts: Sequence[_Expert]) -> list[tuple[_Expert, _Expert]]:
    """The demonstration's hand-overs: each release of an object (its giver's open) the other arm
    had closed on after the giver last did, with that close (the receiver's)."""
    found = []
    for e in experts:
        if e.kp.cue != "open" or e.object is None:
            continue
        closes = [
            c
            for c in experts
            if c.kp.cue == "close" and c.object == e.object and c.kp.frame <= e.kp.frame
        ]
        mine = [c for c in closes if c.arm == e.arm]
        theirs = [c for c in closes if c.arm != e.arm]
        if mine and theirs and mine[-1].kp.frame < theirs[-1].kp.frame:
            found.append((e, theirs[-1]))
    return found


def skipped_handovers(
    stages: list[Stage], experts: Sequence[_Expert], events: Sequence[_Event]
) -> list[Stage]:
    """`stages` with each hand-over of the demonstration's (`handovers`) the attempt skipped
    named as such: its giver held the object, its receiver never did. The receiver's close is
    the missing stage (`handover`), with the order to restore; the receiver's later stages on
    that object, and the giver's release of it, are judged by it (no verdict of their own: a
    release compared with a hand-over's pose that never happened says nothing about a
    placement's numbers)."""
    rows = list(stages)
    for give, take in handovers(experts):
        obj = give.object
        held = {ev.arm for ev in events if ev.cue == "close" and obj in ev.held}
        if give.arm not in held or take.arm in held:
            continue
        at = f"{take.arm} {take.kp.name}"
        note = (
            f"the expert handed {obj} from the {give.arm} arm to the {take.arm} arm here (the "
            f"{take.arm} arm closed on it at {at}, then the {give.arm} arm let go at "
            f"{give.arm} {give.kp.name}); this attempt never put {obj} in the {take.arm} gripper: "
            "it skipped the hand-over. Restore the hand-over stages in the workflow, in this "
            f"order: place {obj} with the {give.arm} arm without opening (demo_relation at "
            f"{give.arm} {give.kp.name} gives that placement and says it is a hand-over), the "
            f"{take.arm} arm's grasp ({at}), then open_gripper on the {give.arm} arm. This is a "
            "missing step: do not edit positions or numbers for it"
        )
        later = {
            e.kp.name
            for e in experts
            if e.arm == take.arm and e.object == obj and e.kp.frame > take.kp.frame
        }
        for i, row in enumerate(rows):
            if row.object != obj:
                continue
            if row.arm == take.arm and row.keypoint == take.kp.name:
                rows[i] = row.model_copy(
                    update={"status": "missing", "verdict": "missing_stage", "handover": True,
                            "held": False, "parameters": [], "spec_diff": [], "note": note,
                            "contact": None, "commanded": None, "world": None}
                )  # fmt: skip
            elif (row.arm == give.arm and row.keypoint == give.kp.name) or (
                row.arm == take.arm and row.keypoint in later
            ):
                rows[i] = row.model_copy(
                    update={"verdict": None, "parameters": [],
                            "note": f"judged by the skipped hand-over at {at}"}
                )  # fmt: skip
    return rows


async def _close(
    base: Stage,
    e: _Expert,
    match: _Event,
    track: Array,
    poses: Mapping[int, Mapping[str, Pose]],
    scene: _Scene,
    shapes: Mapping[str, ObjectShape],
    keypoints: KeypointSet,
    config: ReviewConfig,
    call_at: Callable[[int], Call | None],
    planned: Callable[[Arm, Pose, int], Awaitable[str]],
    memory: MemoryDoc | None,
    lost_between: Callable[[Arm, int, int | None, bool], list[Call]],
    next_event: Callable[[int, Arm], int | None],
    start_of: Callable[[int], int],
    hold: "_Held",
    refused: "Refused",
) -> tuple[Stage, Call | None]:
    """The close against the expert's, and the later call the hold broke in (None: it held on,
    never held, or broke in this close's own call)."""
    kp, obj = e.kp, e.object
    call = call_at(match.sample)
    # held: the referee saw the object move with the gripper, or the settled opening says so
    # (an articulated part's root never moves with it)
    held = bool(match.held) or hold.holding
    update: dict[str, Any] = {
        "attempt_arm": match.arm,
        "opening": match.opening if hold.settled is None else hold.settled,
        "held": held,
        "call": None if call is None else call.name,
        "stage": None if call is None else call.stage,
        "receipt": None if call is None else call.reason,
        "parameters": [] if call is None else spec_parameters(memory, call),
        "spec_diff": spec_diff(memory, call, match.arm),
    }
    expert_obj = poses.get(kp.frame, {}).get(obj) if obj is not None else None
    # where the object was before the close (the truth after the batch has the close's push)
    attempt_obj = scene.pose(obj, match.sample - 1) if obj is not None else None
    # the object away from where the expert found it, or the approach off the expert's line:
    # the state differed, so a close short of the target is not the stack's (no "low-level")
    displaced = _displaced(expert_obj, attempt_obj)
    if expert_obj is not None and attempt_obj is not None and obj is not None:
        gap = contact_gap(
            obj,
            shapes.get(obj),
            expert_obj,
            kp.tcp,
            attempt_obj,
            match.tcp,
            keypoints.pointing[e.arm],
        )
        update["contact"] = gap
        update["world"] = contact_gap(
            "world", None, WORLD, kp.tcp, WORLD, match.tcp, keypoints.pointing[e.arm]
        )
        if (moved := scene.moved(obj, match.sample - 1)) >= MOVED_M:
            update["note"] = f"{obj} had moved {100 * moved:.1f} cm from its start before the close"
        if displaced >= MOVED_M:
            update["note"] = "; ".join(
                filter(
                    None,
                    [
                        update.get("note"),
                        f"{obj} lay {100 * displaced:.1f} cm from where the expert closed on it",
                    ],
                )
            )
        if e.standoff is not None:
            direction = np.subtract(kp.tcp.p, e.standoff.tcp.p)
            given = _argument(call, match.arm, "approach_m") if call is not None else None
            update["standoff_m"] = (float(np.linalg.norm(direction)), given)
            if (approach := _approach(track, match.sample)) is not None:
                cos = float(np.dot(direction, approach) / (np.linalg.norm(direction) + 1e-12))
                update["approach_deg"] = math.degrees(math.acos(float(np.clip(cos, -1, 1))))
    contact: Gap | None = update.get("contact")
    # a hold is lost in the call in whose turn the measured opening fell (else the first of
    # this arm's own calls whose receipt says so, before its next close): this close's own call
    # makes it this stage's, a later one a held-motion stage's; one lost in a call of the other
    # arm alone (a receiving grasp knocking it loose) is that call's stage's
    begin = start_of(_turn(call)) if call is not None else match.sample
    receipts = lost_between(match.arm, begin, next_event(match.sample, match.arm), True)
    breaking = call_at(hold.drop) if hold.drop is not None else None
    if breaking is None and receipts:
        breaking = receipts[0]
    lost = breaking is not None and (call is None or breaking.turn == call.turn)
    later = breaking if breaking is not None and not lost else None
    if later is not None:
        note = f"it held; the hold broke later, during {later.name} (the next stage)"
        update["note"] = "; ".join(filter(None, [update.get("note"), note]))
    other = _other(match.arm)
    if call is not None and any(
        c.turn == call.turn for c in lost_between(other, begin, None, False)
    ):
        note = f"the {other} arm lost its hold during this call (this arm's action knocked it)"
        update["note"] = "; ".join(filter(None, [update.get("note"), note]))
    approach_off = (update.get("approach_deg") or 0.0) > config.contact_deg
    state_differs = displaced >= MOVED_M or approach_off
    reach = config.held_m if held and not lost else config.contact_m
    off = contact is not None and (
        contact.distance_m > reach or contact.angle_deg > config.contact_deg
    )
    # A grasp at the expert's position that held, turned only about the pointing axis, is no
    # divergence while what follows goes fine; `review` blames it for a later failure of that hold
    # (`_turn_blamed`).
    turned_only = off and contact is not None and contact.distance_m <= reach and held and not lost
    if turned_only and match.arm == e.arm and not state_differs:
        off = False
        update["turned_only"] = True
    # where the call sent this arm: a close far from it is the executor's, not the Memory's
    commanded = call.commanded() if call is not None else None
    missed_target = False
    if commanded is not None and commanded[0] == match.arm:
        target = commanded[1]
        if expert_obj is not None and attempt_obj is not None and obj is not None:
            update["commanded"] = contact_gap(
                obj,
                shapes.get(obj),
                expert_obj,
                kp.tcp,
                attempt_obj,
                target,
                keypoints.pointing[e.arm],
            )
        short = float(np.linalg.norm(np.subtract(match.tcp.p, target.p)))
        if short > config.contact_m:
            missed_target = True
            why = (
                "it met something on the way: the object lay away from the expert's place or "
                "the approach came in off the expert's line"
                if state_differs
                else "the executor closed before reaching it: low-level"
            )
            update["note"] = "; ".join(
                filter(
                    None,
                    [
                        update.get("note"),
                        f"the arm closed {100 * short:.1f} cm from the TCP the call commanded "
                        f"({why})",
                    ],
                )
            )
    sent: Gap | None = update.get("commanded")
    sent_ok = sent is not None and (
        sent.distance_m <= config.contact_m and sent.angle_deg <= config.contact_deg
    )
    verdict: Verdict | None = None
    if match.arm != e.arm:
        verdict = "wrong_arm"
    elif off and missed_target and sent_ok and not state_differs:
        verdict = "expert_unplannable"  # the Memory aimed at the expert's pose; the stack missed
    elif off:
        mapped = _mapped(expert_obj, kp.tcp, attempt_obj)
        update["plan"] = await planned(e.arm, mapped, max(match.sample - 1, 0))
        unplannable = update["plan"].startswith("does not plan")
        verdict = "expert_unplannable" if unplannable and not state_differs else "memory_diverged"
    elif not held or lost:
        verdict = "grasp_slip"
        slip = "lost the hold during the grasp's own call" if held else "held nothing"
        update["note"] = "; ".join(filter(None, [update.get("note"), slip]))
    verdict = _transcribed(verdict, update, call, refused, memory, match.arm)
    return base.model_copy(update={**update, "verdict": verdict}), later


@dataclass(frozen=True)
class _Held:
    """The attempt's hold after a close by its measured opening: the opening the close settled
    at (the end of its call's turn; None: not measured), the sample it settled at, whether that
    holds something, and the first later sample before the arm's next gripper event where it fell
    below the holding opening (None: it held on, or never held), with the opening there."""

    settled: float | None
    settled_at: int
    holding: bool
    drop: int | None
    dropped: float | None = None


def _hold(
    recorded: Mapping[str, np.ndarray],
    turns: Array,
    match: _Event,
    events: Sequence[_Event],
    threshold: float,
) -> _Held:
    found = recorded.get(f"{match.arm}.gripper")
    end = [ev.sample for ev in events if ev.arm == match.arm and ev.sample > match.sample]
    turn = int(turns[match.sample]) if match.sample < len(turns) else None
    same = np.flatnonzero(turns == turn) if turn is not None else np.zeros(0, dtype=int)
    settled_at = int(same[-1]) if len(same) else match.sample
    if found is None or settled_at >= len(found) or math.isnan(float(found[settled_at])):
        return _Held(None, settled_at, False, None)
    opening = np.asarray(found, dtype=float)
    settled = float(opening[settled_at])
    if settled < threshold:
        return _Held(settled, settled_at, False, None)
    stop = min(end) if end else len(opening)
    below = np.flatnonzero(opening[settled_at + 1 : stop] < threshold)
    if not len(below):
        return _Held(settled, settled_at, True, None)
    drop = settled_at + 1 + int(below[0])
    return _Held(settled, settled_at, True, drop, float(opening[drop]))


def _held_motion(
    e: _Expert,
    match: _Event,
    breaking: Call,
    hold: _Held,
    track: Array,
    poses: Mapping[int, Mapping[str, Pose]],
    scene: _Scene,
    keypoints: KeypointSet,
    memory: MemoryDoc | None,
    config: ReviewConfig,
    turns: Array,
) -> Stage:
    """The held motion after the close `match` the hold broke in (`breaking`; see the module):
    the expert's from its close to its next keypoints while it held, the attempt's from its
    settled close to the sample the hold broke at (else the end of the breaking call's turn);
    `execution_failed` where the attempt's TCP kept to the expert's held path up to the break, and
    the TCP the breaking call commanded lies on it too, else `hold_lost`."""
    kp, obj = e.kp, e.object
    kps = keypoints.arms[e.arm]
    held: list[Keypoint] = []
    for k in kps[kp.index :]:
        if k.cue == "open":
            break
        held.append(k)
    reference = obj or "world"
    expert_ref = poses.get(kp.frame, {}).get(obj) if obj is not None else None
    attempt_ref = scene.pose(obj, match.sample - 1) if obj is not None else None
    expert_ref = expert_ref or WORLD
    attempt_ref = attempt_ref or WORLD
    to = hold.drop if hold.drop is not None else min(hold.settled_at + 1, len(track) - 1)
    start = Pose.from_list(track[hold.settled_at])
    attempt = _moved(attempt_ref, start, Pose.from_list(track[to]))
    found: dict[str, Any] = {}
    if held:
        local, world, deg, axis = _moved(expert_ref, kp.tcp, held[0].tcp)
        found |= {
            "expert_to": held[0].name,
            "expert": local,
            "expert_world": world,
            "expert_turn_deg": deg,
            "expert_axis": axis,
        }
        if len(held) > 1:
            local, _, deg, _ = _moved(expert_ref, kp.tcp, held[-1].tcp)
            found |= {"expert_end": held[-1].name, "expert_end_local": local}
            found["expert_end_turn_deg"] = deg
    commanded = breaking.commanded()
    sent: Array | None = None
    if commanded is not None and commanded[0] == match.arm:
        local, _, deg, _ = _moved(attempt_ref, start, commanded[1])
        found |= {"commanded": local, "commanded_turn_deg": deg}
        sent = np.asarray(local)
    on_path = False
    if held:  # the expert's held path from its close, against the attempt's from its own
        path = [np.zeros(3)] + [np.asarray(_moved(expert_ref, kp.tcp, k.tcp)[0]) for k in held]
        track_from = range(hold.settled_at, to + 1)
        deviation = max(
            _to_polyline(np.asarray(_moved(attempt_ref, start, Pose.from_list(track[i]))[0]), path)
            for i in track_from
        )
        here = np.asarray(attempt[0])
        k = min(range(len(held)), key=lambda i: float(np.linalg.norm(path[i + 1] - here)))
        turn_gap = abs(attempt[2] - _moved(expert_ref, kp.tcp, held[k].tcp)[2])
        # where the breaking call was going counts too: a move off the path broken early is not
        # the expert's motion
        if sent is not None:
            deviation = max(deviation, _to_polyline(sent, path))
        on_path = deviation <= config.contact_m and turn_gap <= config.contact_deg
        found |= {"path_m": deviation, "nearest": held[k].name}
    if obj is not None:  # to the end of the breaking call's turn (the body may slide on)
        span = np.flatnonzero(turns == breaking.turn)
        end = int(span[-1]) if len(span) else to
        a, b = scene.pose(obj, hold.settled_at), scene.pose(obj, max(end, to))
        if a is not None and b is not None:
            found["body_moved_m"] = float(np.linalg.norm(np.subtract(b.p, a.p)))
    motion = Motion(
        reference=reference,
        attempt=attempt[0],
        attempt_world=attempt[1],
        attempt_turn_deg=attempt[2],
        attempt_axis=attempt[3],
        opening=hold.dropped,
        **found,
    )
    target = held[0] if held else kp
    verdict: Verdict = "execution_failed" if on_path else "hold_lost"
    frame = path_frame(breaking, config)
    why = "its opening fell below holding" if hold.drop is not None else "its receipt says so"
    note = f"the hold broke during {breaking.name} ({why})"
    if on_path:
        note += (
            f", with the TCP within {100 * found['path_m']:.1f} cm of the expert's held path "
            f"(nearest {found['nearest']}): an execution failure of the expert's own motion"
        )
    return Stage(
        arm=e.arm,
        keypoint=target.name,
        phase=target.phase,
        event="move",
        object=obj,
        attempt_arm=match.arm,
        status="failed",
        motion=motion,
        call=breaking.name,
        receipt=breaking.reason,
        parameters=spec_parameters(memory, breaking),
        spec_diff=spec_diff(memory, breaking, match.arm),
        verdict="path_frame" if frame is not None else verdict,
        note=note if frame is None else f"{note}; {frame}",
    )


def _to_polyline(point: Array, path: Sequence[Array]) -> float:
    """The distance from `point` to the polyline through `path`."""
    best = float(np.linalg.norm(point - path[0]))
    for a, b in itertools.pairwise(path):
        d = b - a
        t = float(np.clip(np.dot(point - a, d) / (float(np.dot(d, d)) or 1.0), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (a + t * d))))
    return best


def _moved(
    reference: Pose, a: Pose, b: Pose
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], float, tuple[float, float, float] | None
]:
    """The TCP's move from `a` to `b`: its displacement in `reference`'s frame and along the
    robot's axes, the angle it turned (deg) and the axis it turned about, in `reference`'s
    frame (None: no turn)."""
    world = np.subtract(b.p, a.p)
    inverse = quat_conjugate(reference.q)
    turn = quat_multiply(b.q, quat_conjugate(a.q))
    w = float(np.clip(abs(turn[0]), -1.0, 1.0))
    deg = math.degrees(2 * math.acos(w))
    vector = np.asarray(turn[1:], dtype=float) * (1.0 if turn[0] >= 0 else -1.0)
    norm = float(np.linalg.norm(vector))
    axis = _vec(quat_rotate(inverse, vector / norm)) if norm > 1e-9 and deg >= 0.5 else None
    return _vec(quat_rotate(inverse, world)), _vec(world), deg, axis


def _acting(call: Call) -> set[str]:
    """The arms a call acted with: its `arm`, or the arms `together` gave actions."""
    arms = set(call.parts())
    if isinstance(call.arguments.get("arm"), str):
        arms.add(call.arguments["arm"])
    return arms


def _other(arm: Arm) -> Arm:
    return "right" if arm == "left" else "left"


def _reference(e: _Expert) -> str | None:
    """What a stage's pose is measured against: its object; a release's target (None: world)."""
    if e.kp.cue == "open":
        return sorted(e.onto)[0] if e.onto else None
    return e.object


def _holds(events: Sequence[_Event]) -> dict[str, list[tuple[int, int]]]:
    """Per object, the samples the attempt held it over: from a close that held it to its arm's
    next open (end exclusive)."""
    out: dict[str, list[tuple[int, int]]] = {}
    for ev in events:
        if ev.cue != "close":
            continue
        opens = [
            o.sample for o in events if o.arm == ev.arm and o.cue == "open" and o.sample > ev.sample
        ]
        end = min(opens) if opens else 2**62
        for name in ev.held:
            out.setdefault(name, []).append((ev.sample, end))
    return out


def _displaced(expert_obj: Pose | None, attempt_obj: Pose | None) -> float:
    """How far the attempt's object lay from where the expert's was at the stage (same seed)."""
    if expert_obj is None or attempt_obj is None:
        return 0.0
    return float(np.linalg.norm(np.subtract(attempt_obj.p, expert_obj.p)))


def _turn(call: Call | None) -> int:
    return -1 if call is None else call.turn


APPROACH_M: Final = 0.02
"""The attempt's approach is the direction from its last sample this far from its close."""


def _approach(track: Array, sample: int, length: float = APPROACH_M) -> Array | None:
    """The unit direction the attempt's TCP came into its close (or open) along: from its last
    recorded sample `length` away."""
    here = track[sample, :3]
    for i in range(sample - 1, -1, -1):
        step = here - track[i, :3]
        if (norm := float(np.linalg.norm(step))) >= length:
            return step / norm
    return None


def _argument(call: Call, arm: Arm, key: str) -> float | None:
    """A number the call passed for `arm` (its own arguments, or its arm's part of `together`)."""
    for scope in (call.part(arm), call.arguments):
        if isinstance(scope, Mapping) and isinstance(scope.get(key), (int, float)):
            return float(scope[key])
    return None


def _mapped(expert_obj: Pose | None, expert_tcp: Pose, attempt_obj: Pose | None) -> Pose:
    """The expert's TCP relative to its object, put on the attempt's object pose."""
    if expert_obj is None or attempt_obj is None:
        return expert_tcp
    relative = np.linalg.inv(expert_obj.matrix()) @ expert_tcp.matrix()
    return Pose.from_matrix(attempt_obj.matrix() @ relative)


def _open(
    base: Stage,
    e: _Expert,
    match: _Event,
    track: Array,
    rest_at: int,
    poses: Mapping[int, Mapping[str, Pose]],
    scene: _Scene,
    shapes: Mapping[str, ObjectShape],
    keypoints: KeypointSet,
    config: ReviewConfig,
    call_at: Callable[[int], Call | None],
    placing: Callable[[Arm, int], Call | None],
    memory: MemoryDoc | None,
    refused: "Refused",
    closed_at: int = 0,
) -> Stage:
    """A release against the expert's, by what the attempt controlled at the open: its TCP
    (relative to the target where each was then), the line it came in along, and the held
    object's last held pose; where the object then came to rest (at sample `rest_at`) is a note
    (it may have fallen: the motion before the open, not the fall, is what a spec sets), unless
    it came to rest where the expert's did (`_landed`): then the release did not diverge. An
    open off the expert's whose arm barely moved since its close (at sample `closed_at`) where
    the expert's carried it says the move to the target is missing (`_unmoved`)."""
    kp, obj = e.kp, e.object
    call = call_at(match.sample)
    update: dict[str, Any] = {
        "attempt_arm": match.arm,
        "call": None if call is None else call.name,
        "stage": None if call is None else call.stage,
        "receipt": None if call is None else call.reason,
        "parameters": [] if call is None else spec_parameters(memory, call),
        "spec_diff": spec_diff(memory, call, match.arm),
    }
    placed: Call | None = call
    if not any("place_spec" in given for given in update["parameters"]):
        # an open of its own call: the placement before it set where it opened (one since the
        # close that took what it releases: never another object's placement)
        placed = placing(match.arm, _turn(call) if call is not None else int(1e9))
        if placed is not None and placed is not call:
            update["parameters"] += [
                f"{given.split(' ')[0]} (place_spec, by the {placed.name} before the open)"
                for given in spec_parameters(memory, placed)
                if "place_spec" in given
            ]
    target = sorted(e.onto)[0] if e.onto else None
    here = poses.get(kp.frame, {})
    pointing = keypoints.pointing[e.arm]
    expert_target = here.get(target) if target is not None else None
    attempt_target = scene.pose(target, match.sample - 1) if target is not None else None
    if target is not None and expert_target is not None and attempt_target is not None:
        tcp = contact_gap(
            target, shapes.get(target), expert_target, kp.tcp, attempt_target, match.tcp, pointing
        )
    else:
        tcp = contact_gap("world", None, WORLD, kp.tcp, WORLD, match.tcp, pointing)
    update["contact"] = tcp
    tcp_off = tcp.distance_m > config.place_m or tcp.angle_deg > config.place_deg
    held_off = False
    if obj is not None and obj in here:
        attempt = scene.pose(obj, match.sample - 1)  # the last held pose, before the open
        if attempt is not None:
            gap = object_gap(
                obj,
                target,
                shapes,
                (here[obj], expert_target),
                (attempt, attempt_target),
                scene.pose(obj, 0),
            )
            update["place"] = gap
            held_off = gap.distance_m > config.place_m or gap.angle_deg > config.place_deg
        rest = _rest_note(obj, target, kp, max(rest_at, match.sample), poses, scene, shapes)
        if rest:
            update["note"] = rest
    approach = _release_approach(e, keypoints, track, match.sample)
    if approach is not None:
        given = _argument(call, match.arm, "approach_m") if call is not None else None
        update["standoff_m"] = (approach[0], given)
        update["approach_deg"] = approach[1]
    approach_off = approach is not None and approach[1] > config.contact_deg
    if (tcp_off or held_off) and (unmoved := _unmoved(e, keypoints, track, closed_at, match)):
        note = (
            f"the {match.arm} arm moved {100 * unmoved[0]:.1f} cm from its close to the open, the "
            f"expert's {100 * unmoved[1]:.1f} cm: the open came {100 * tcp.distance_m:.1f} cm "
            "from the expert's release, a missing move to the target (transport or placement) "
            "before the open, not a placement spec's numbers"
        )
        update["note"] = "; ".join(filter(None, [note, update.get("note")]))
        if placed is not None and placed is not call:  # that placement was not this release's
            update["parameters"] = [p for p in update["parameters"] if "before the open" not in p]
    landed = _landed(obj, target, kp, rest_at, poses, scene, shapes, config)
    verdict: Verdict | None = None
    if match.arm != e.arm:
        verdict = "wrong_arm"
    elif landed and (tcp_off or held_off or approach_off):
        note = (
            f"{obj} came to rest where the expert's did (within {100 * config.place_m:.0f} cm "
            f"and {config.place_deg:.0f} deg), so the release did not diverge whatever the "
            "gripper's pose at the open"
        )
        update["note"] = note  # in place of the rest note, which judges by the open
    elif tcp_off:
        verdict = "memory_diverged"
        turned = _half_turned(e, match, here, scene, shapes, config)
        if turned:
            update["note"] = "; ".join(filter(None, [turned, update.get("note")]))
    elif approach_off:
        assert approach is not None
        verdict = "approach_diverged"
        update["note"] = "; ".join(
            filter(None, [_approach_note(approach, expert_target, target), update.get("note")])
        )
    elif held_off:
        verdict = "memory_diverged"
        note = (
            _frontless(memory, update["parameters"], update.get("place"))
            or _half_turned(e, match, here, scene, shapes, config)
            or (
                "the TCP was at the expert's at the open, the held object was not: it sat in "
                "the gripper otherwise than the expert's (the grasp behind it, not the placement)"
            )
        )
        update["note"] = "; ".join(filter(None, [note, update.get("note")]))
    if placed is not None and placed is not call and (extra := invented_arguments(memory, placed)):
        update["invented"] = extra  # the placement that set where it opened
    verdict = _transcribed(verdict, update, call, refused, memory, match.arm)
    return base.model_copy(update={**update, "verdict": verdict})


def _unmoved(
    e: _Expert, keypoints: KeypointSet, track: Array, closed_at: int, match: _Event
) -> tuple[float, float] | None:
    """How far the attempt's TCP and the expert's went from their close to the open (m), where
    the expert's carried what it held (`UNMOVED_M` at least) and the attempt's went less than
    half as far; else None."""
    close = next(
        (k for k in reversed(keypoints.arms[e.arm]) if k.cue == "close" and k.frame < e.kp.frame),
        None,
    )
    if close is None or not 0 <= closed_at < len(track):
        return None
    theirs = float(np.linalg.norm(np.subtract(e.kp.tcp.p, close.tcp.p)))
    ours = float(np.linalg.norm(np.subtract(match.tcp.p, track[closed_at][:3])))
    return (ours, theirs) if theirs >= UNMOVED_M and ours < theirs / 2 else None


UNMOVED_M: Final = 0.05
"""An expert's release this far from its close carried what it held (`_unmoved`)."""


HALF_TURN_DEG: Final = 150.0
"""A held object this far or more off the expert's at the open is about half a turn round
(`_frontless`)."""


def _frontless(memory: MemoryDoc | None, parameters: Sequence[str], gap: Gap | None) -> str:
    """Where the held object was about half a turn off the expert's at the open (`HALF_TURN_DEG`)
    and the placement specs behind the open (`parameters`, as `spec_parameters` names them) name
    no front (`x_toward` null): the object's frame x may come out at either end from one
    measurement to the next, so a spec without a front places it either way round, and no turn
    of the grasp changes that (a gripper half-turned about its pointing axis holds the object
    the same way); else empty."""
    if memory is None or gap is None or gap.angle_deg < HALF_TURN_DEG:
        return ""
    names = [given.split(" ")[0] for given in parameters if "(place_spec" in given]
    frontless = [
        name
        for name in names
        if isinstance(spec := getattr(memory.params.get(name), "value", None), PlaceSpec)
        and spec.x_toward is None
    ]
    if not frontless:
        return ""
    return (
        f"the held object was {gap.angle_deg:.0f} deg off the expert's at the open, about half a "
        f"turn round, and {', '.join(frontless)} names no front (x_toward null): the object's "
        "frame x can come out at either end from one measurement to the next, so the placement "
        "puts it either way round. Name the front the placement turns toward in its x_toward "
        "(a decided part, far_end, or low_end / high_end: the end of its long side that stands "
        "lower or higher, which locate_part reports and every measurement finds again), and the "
        "same front in the grasp spec of that object; turning the grasp does not fix it: a "
        "gripper half-turned about its pointing axis holds the object the same way"
    )


def _half_turned(
    e: _Expert,
    match: _Event,
    here: Mapping[str, Pose],
    scene: _Scene,
    shapes: Mapping[str, ObjectShape],
    config: ReviewConfig,
) -> str:
    """Where the held object sat in the gripper near the expert's way (the object in the TCP's
    frame within `place_m`, turned less than half as far as the TCP) but the TCP itself was
    turned off the expert's beyond `place_deg` (a gripper looks the same half-turned about its
    pointing axis, the object it holds does not): the placement turned it, not the grasp; else
    empty."""
    obj = e.object
    shape = shapes.get(obj) if obj is not None else None
    if obj is None or obj not in here or (shape is not None and shape.symmetric):
        return ""  # a round object turned about its axis is where it was put
    attempt = scene.pose(obj, match.sample - 1)
    if attempt is None:
        return ""
    theirs = np.linalg.inv(e.kp.tcp.matrix()) @ here[obj].matrix()
    ours = np.linalg.inv(match.tcp.matrix()) @ attempt.matrix()
    moved = float(np.linalg.norm(ours[:3, 3] - theirs[:3, 3]))
    turned = float(quat_angle_deg(Pose.from_matrix(ours).q, Pose.from_matrix(theirs).q))
    off = float(quat_angle_deg(match.tcp.q, e.kp.tcp.q))
    if moved > config.place_m or off <= config.place_deg or 2 * turned >= off:
        return ""  # the grasp's own turn explains as much
    return (
        f"the object sat in the gripper near the expert's way ({100 * moved:.1f} cm, "
        f"{turned:.0f} deg), the TCP at the open was turned {off:.0f} deg from the expert's: the "
        "placement turned the held object (its spec's turn or front, or the turn place_object "
        "took), not the grasp"
    )


def _landed(
    obj: str | None,
    target: str | None,
    kp: Keypoint,
    rest_at: int,
    poses: Mapping[int, Mapping[str, Pose]],
    scene: _Scene,
    shapes: Mapping[str, ObjectShape],
    config: ReviewConfig,
) -> bool:
    """Whether the released object came to rest (at sample `rest_at`) where the expert's did
    (its next frame), within `place_m` and `place_deg` (about the vertical only for a symmetric
    object): a release that put the object where the expert's went is no divergence."""
    if obj is None:
        return False
    later = min((f for f in poses if f > kp.frame), default=kp.frame)
    settled = poses.get(later, poses.get(kp.frame, {}))
    rest = scene.pose(obj, rest_at)
    if obj not in settled or rest is None:
        return False
    expert_target = settled.get(target) if target is not None else None
    attempt_target = scene.pose(target, rest_at) if target is not None else None
    gap = object_gap(obj, target, shapes, (settled[obj], expert_target),
                     (rest, attempt_target), scene.pose(obj, 0))  # fmt: skip
    # where the expert's own object tumbled after its open (turned beyond `place_deg` in its
    # fall), how it lands turned is the fall's, not the release's: where it rests is the measure
    held = poses.get(kp.frame, {}).get(obj)
    tumbled = held is not None and quat_angle_deg(held.q, settled[obj].q) > config.place_deg
    return gap.distance_m <= config.place_m and (tumbled or gap.angle_deg <= config.place_deg)


def _release_approach(
    e: _Expert, keypoints: KeypointSet, track: Array, sample: int
) -> tuple[float, float, Array, Array] | None:
    """The line a release came in along: the expert's from its keypoint before (its length,
    and the direction), against the attempt's TCP over the same length before its open (at least
    `APPROACH_M`); the angle between them (deg); None without a keypoint before or a move."""
    kps = keypoints.arms[e.arm]
    if e.kp.index < 2:
        return None
    move = np.subtract(e.kp.tcp.p, kps[e.kp.index - 2].tcp.p)
    length = float(np.linalg.norm(move))
    if length < APPROACH_M:
        return None
    expert = move / length
    attempt = _approach(track, sample, max(APPROACH_M, min(length, 2 * APPROACH_M)))
    if attempt is None:
        return None
    cos = float(np.clip(np.dot(expert, attempt), -1.0, 1.0))
    return length, math.degrees(math.acos(cos)), expert, attempt


def _approach_note(
    approach: tuple[float, float, Array, Array], target_pose: Pose | None, target: str | None
) -> str:
    """The expert's release line as a place spec's `approach` and `approach_m` would give it."""
    length, deg, expert, attempt = approach
    text = (
        f"the TCP was at the expert's at the open, but came in {deg:.0f} deg off the expert's "
        f"line: the expert moved {100 * length:.1f} cm along {_unit(expert)} (robot axes) onto "
        f"it, the attempt along {_unit(attempt)}"
    )
    if target_pose is not None and target is not None:
        local = quat_rotate(quat_conjugate(target_pose.q), expert)
        text += f", which is {_unit(local)} in {target}'s own frame"
    return (
        text + ": set the place spec's approach to the expert's line (in its frame: the target's "
        "for frame target, the world's for world_axes) and approach_m to that length, "
        "or refit the placement at this keypoint (demo_relation gives its approach)"
    )


def _unit(v: Sequence[float] | Array) -> str:
    return "(" + ", ".join(f"{float(c):+.2f}" for c in v) + ")"


def _rest_note(
    obj: str,
    target: str | None,
    kp: Keypoint,
    rest_at: int,
    poses: Mapping[int, Mapping[str, Pose]],
    scene: _Scene,
    shapes: Mapping[str, ObjectShape],
) -> str:
    """Where the released object came to rest (at sample `rest_at`) against where the expert's
    did (its next frame), when that is off: a note, not the release's measure."""
    later = min((f for f in poses if f > kp.frame), default=kp.frame)
    settled = poses.get(later, poses.get(kp.frame, {}))
    rest = scene.pose(obj, rest_at)
    if obj not in settled or rest is None:
        return ""
    expert_target = settled.get(target) if target is not None else None
    attempt_target = scene.pose(target, rest_at) if target is not None else None
    expert, attempt = (settled[obj], expert_target), (rest, attempt_target)
    gap = object_gap(obj, target, shapes, expert, attempt, scene.pose(obj, 0))
    if gap.distance_m <= MOVED_M and gap.angle_deg <= MOVED_DEG:
        return ""
    return (
        f"after the open {obj} came to rest {100 * gap.distance_m:.1f} cm and "
        f"{gap.angle_deg:.0f} deg from where the expert's did (it moved on its own after the "
        "release: judged by the open above, not by this)"
    )


def _held_contact(
    base: Stage,
    e: _Expert,
    poses: Mapping[int, Mapping[str, Pose]],
    scene: _Scene,
    shapes: Mapping[str, ObjectShape],
    config: ReviewConfig,
) -> Stage:
    """A held object's contact with what it meets (the object nearest it at the expert's
    contact): the attempt's nearest approach to the expert's pose against it, over its samples;
    beyond `place_m` or `place_deg` it diverged."""
    held, here = e.object, poses.get(e.kp.frame, {})
    assert held is not None
    others = [o for o in here if o != held]
    if held not in here or not others:
        return base.model_copy(update={"note": "nothing for the held object to meet"})
    at = center(here[held], shapes.get(held))
    target = min(others, key=lambda o: float(np.linalg.norm(center(here[o], shapes.get(o)) - at)))
    best: tuple[Gap, int] | None = None
    for sample in (int(t) for t in scene.samples):
        mine, theirs = scene.pose(held, sample), scene.pose(target, sample)
        if mine is None or theirs is None:
            continue
        gap = object_gap(
            held, target, shapes, (here[held], here[target]), (mine, theirs), scene.pose(held, 0)
        )
        if best is None or gap.distance_m < best[0].distance_m:
            best = (gap, sample)
    if best is None:
        return base.model_copy(update={"note": f"the attempt never recorded {held} and {target}"})
    gap, sample = best
    off = gap.distance_m > config.place_m or gap.angle_deg > config.place_deg
    note = (
        f"{held} against {target} at the expert's contact; the attempt's nearest approach to it "
        f"(sample {sample}); a placement fitted at this keypoint reproduces it"
    )
    return base.model_copy(
        update={
            "status": "matched",
            "place": gap,
            "note": note,
            "verdict": "memory_diverged" if off else None,
        }
    )


def _missed(
    base: Stage, e: _Expert, keypoints: KeypointSet, progress: Mapping[Arm, ArmProgress] | None
) -> Stage:
    """A turned lift or carry, or the contact, the referee says the attempt passed by."""
    kps = keypoints.arms[e.arm]
    j = e.kp.index - 1
    note = ""
    if j > 0:
        before = kps[j - 1]
        rise = e.kp.tcp.p[2] - before.tcp.p[2]
        turn = float(gripper_angle_deg(e.kp.tcp.q, before.tcp.q, keypoints.pointing[e.arm]))
        moved = float(np.linalg.norm(np.subtract(e.kp.tcp.p, before.tcp.p)))
        note = (
            f"the expert moved {100 * moved:.1f}"
            f" cm from its {before.name} ({before.phase}), {100 * rise:+.1f} cm up, turning the "
            f"gripper {turn:.0f} deg"
        )
    if progress is not None:
        p = progress[e.arm].keypoints[j]
        if p.nearest is not None:
            dx, dy, dz = (100 * v for v in p.nearest.offset_m)
            note += (
                f"; the attempt came no nearer than {100 * p.nearest.distance_m:.1f} cm "
                f"(dx {dx:+.1f}, dy {dy:+.1f}, dz {dz:+.1f} cm), {p.nearest.angle_deg:.0f} deg"
            )
    return base.model_copy(
        update={"status": "missed", "verdict": "missing_stage", "note": note.lstrip("; ")}
    )


async def _failed(
    base: Stage,
    e: _Expert,
    failed: tuple[Call, int],
    poses: Mapping[int, Mapping[str, Pose]],
    scene: _Scene,
    shapes: Mapping[str, ObjectShape],
    planned: Callable[[Arm, Pose, int], Awaitable[str]],
    memory: MemoryDoc | None,
    keypoints: KeypointSet,
    config: ReviewConfig,
    refused: "Refused",
) -> Stage:
    """A stage the attempt did not reach because a call failed on the way: its commanded TCP
    against the expert's, relative to what the stage acts on (a release: its target)."""
    call, turns_start = failed
    kp, obj = e.kp, e.object
    update: dict[str, Any] = {
        "status": "failed",
        "call": call.name,
        "stage": call.stage,
        "receipt": call.reason,
        "parameters": spec_parameters(memory, call),
        "spec_diff": spec_diff(memory, call, call.result.get("arm")),
    }
    commanded = call.commanded()
    expert_obj = poses.get(kp.frame, {}).get(obj) if obj is not None else None
    reference = _reference(e)
    if kp.cue == "open":  # a release: where it puts the object, relative to its target or world
        expert_obj = poses.get(kp.frame, {}).get(reference) if reference is not None else None
    attempt_obj = scene.pose(reference, turns_start) if reference is not None else None
    verdict: Verdict
    mapped = _mapped(expert_obj, kp.tcp, attempt_obj)
    displaced = _displaced(expert_obj, attempt_obj)
    if displaced >= MOVED_M:  # the state differed: not the stack's to blame
        update["note"] = f"{reference} lay {100 * displaced:.1f} cm from where the expert had it"
    if commanded is None:  # refused before any target was commanded (nothing plannable)
        update["plan"] = await planned(e.arm, mapped, turns_start)
        said = "the call commanded no pose: " + _clip(call.text, 240)
        update["note"] = "; ".join(filter(None, [update.get("note"), said]))
        if update["plan"].startswith("does not plan") and displaced < MOVED_M:
            verdict = "expert_unplannable"
        elif call.result.get("arm") not in (None, e.arm):
            verdict = "wrong_arm"
        else:
            verdict = "memory_diverged"
    else:
        arm, tcp = commanded
        update["attempt_arm"] = arm
        if expert_obj is not None and attempt_obj is not None and reference is not None:
            gap = contact_gap(
                reference,
                shapes.get(reference),
                expert_obj,
                kp.tcp,
                attempt_obj,
                tcp,
                keypoints.pointing[e.arm],
            )
        else:
            gap = contact_gap("world", None, WORLD, kp.tcp, WORLD, tcp, keypoints.pointing[e.arm])
        update["commanded"] = gap
        if reference is not None:
            update["world"] = contact_gap(
                "world", None, WORLD, kp.tcp, WORLD, tcp, keypoints.pointing[e.arm]
            )
        off = gap.distance_m > config.contact_m or gap.angle_deg > config.contact_deg
        update["plan"] = await planned(e.arm, mapped, turns_start)
        unplannable = update["plan"].startswith("does not plan") and displaced < MOVED_M
        if arm != e.arm:
            verdict = "wrong_arm"
        elif unplannable or (
            not off and displaced < MOVED_M and call.reason in ("unreachable", "plan_failed")
        ):
            verdict = "expert_unplannable"
        elif off or displaced >= MOVED_M:
            verdict = "memory_diverged"
        else:
            verdict = "grasp_slip"
    if verdict != "wrong_arm" and (frame := path_frame(call, config)) is not None:
        verdict = "path_frame"
        update["note"] = "; ".join(filter(None, [update.get("note"), frame]))
    verdict = _transcribed(verdict, update, call, refused, memory, call.result.get("arm"))
    return base.model_copy(update={**update, "verdict": verdict})


def _reproducible(row: Stage, found: Mapping[str, str]) -> Stage:
    """A placement that diverged by a spec no constrain reproduces (`unreproducible`): the
    spec, not its numbers, is what to change."""
    if row.verdict not in ("memory_diverged", "approach_diverged"):
        return row
    named = [p.split(" ")[0] for p in row.parameters]
    spec = next((name for name in named if name in found), None)
    if spec is None:
        return row
    note = (
        f"the spec dry run on the demonstration's state found {spec} reproduced by no constrain "
        f"(place_object's TCP at stop_back_m 0 off the expert's: {found[spec]})"
    )
    return row.model_copy(
        update={"verdict": "unreproducible", "note": "; ".join(filter(None, [note, row.note]))}
    )


def path_start(call: Call) -> float | None:
    """How far (m) a held path's start lay from where the grasp held as `follow_path` began
    (its receipt's `path.start_m`); None for another call."""
    path = call.result.get("path")
    found = path.get("start_m") if isinstance(path, Mapping) else None
    return float(found) if isinstance(found, (int, float)) else None


def path_frame(call: Call, config: ReviewConfig) -> str | None:
    """Why a held path's call failed by its frame (`path_frame`), not its spec: its start lay
    beyond `contact_m` from the hold it set off from. The path's poses are the expert's motion in
    the held object's frame; a start that far from the hold means the frame chosen for the
    object at execution was not the one the hold was taken in (a symmetric object's turn), so
    the commanded motion is the expert's turned, and no path or grasp edit fixes that."""
    start = path_start(call)
    if start is None or start <= config.contact_m:
        return None
    return (
        f"{call.name}'s path started {1000 * start:.0f} mm from where the grasp held (the "
        "receipt's start): the frame the object was given at execution was not the hold's, so "
        "the commanded motion is the expert's turned or reversed; an execution/frame failure, "
        "not the path's or the grasp's spec"
    )


TURN_BLAMED: Final = ("memory_diverged", "expert_unplannable", "hold_lost", "execution_failed")
"""The later failures a turned-only grasp (`Stage.turned_only`) is blamed for: what it holds failed
to be placed (an unreachable or colliding plan, a placement off) or the hold broke."""


def _turn_blamed(stages: list[Stage], first: int) -> tuple[list[Stage], int] | None:
    """By design: a grasp turned from the expert's that held is no divergence
    by itself, but where a later stage of its hold fails (`TURN_BLAMED`: its held motion, its
    release, a call that failed while holding it, before the arm let it go), the grasp's turn is
    the first divergence, `memory_diverged` with the later failure named; the stages with it, and
    the grasp's index. None where no such grasp holds what failed."""
    failed = stages[first]
    if failed.verdict not in TURN_BLAMED or failed.object is None:
        return None
    for i in range(first - 1, -1, -1):
        row = stages[i]
        if row.arm != failed.arm or row.object != failed.object:
            continue
        if row.event == "open":
            return None  # let go before the failure: not this hold's
        if row.event != "close":
            continue
        if not row.turned_only:
            return None
        gap = row.contact
        turn = "" if gap is None else f"{gap.angle_deg:.0f} deg"
        if gap is not None and gap.twist_deg:
            turn += f" ({gap.twist_deg:.0f} deg about the pointing axis)"
        why = (
            f"the grasp was at the expert's position, turned {turn}, and held, but the later "
            f"stage {failed.keypoint} ({failed.phase}, {failed.event}) failed ({failed.verdict}): "
            "the turn is the likely cause"
        )
        blamed = row.model_copy(
            update={
                "verdict": "memory_diverged",
                "note": "; ".join(filter(None, [row.note, why])),
            }
        )
        later = f"follows from the grasp at {row.keypoint}, turned from the expert's"
        out = list(stages)
        out[i] = blamed
        out[first] = failed.model_copy(
            update={"note": "; ".join(filter(None, [failed.note, later]))}
        )
        return out, i
    return None


def end_rows(
    keypoints: KeypointSet,
    layer: DemoTruth,
    shapes: Mapping[str, ObjectShape],
    scene: _Scene,
    config: ReviewConfig,
    experts: Sequence[_Expert],
) -> list[EndRow]:
    """The objects the demonstration moved that ended off its final state."""
    if not layer.frames or not len(scene.samples):
        return []
    start, final = layer.frames[0].poses, layer.frames[-1].poses
    targets = {e.object: sorted(e.onto)[0] for e in experts if e.object and e.onto}
    rows = []
    last = int(scene.samples[-1])
    for name, pose in final.items():
        moved = float(np.linalg.norm(np.subtract(pose.p, start[name].p))) >= MOVED_M or (
            _angle(np.asarray(pose.q), np.asarray(start[name].q)) >= MOVED_DEG
        )
        attempt = scene.pose(name, last)
        if not moved or attempt is None:
            continue
        target = targets.get(name)
        gap = object_gap(
            name,
            target,
            shapes,
            (pose, final.get(target) if target else None),
            (attempt, scene.pose(target, last) if target else None),
            start[name],
        )
        if gap.distance_m > config.place_m or gap.angle_deg > config.place_deg:
            rows.append(EndRow(object=name, gap=gap))
    return rows


def plan_check(planner: Planner, profile: EmbodimentProfile, qpos: Array) -> PlanCheck:
    """`PlanCheck` through `planner`, from the attempt's commanded joints (`qpos_cmd`)."""

    async def check(arm: Arm, tcp: Pose, sample: int) -> str | None:
        spec = profile.arms[arm]
        if spec.planner is None:
            return "no planner assets"
        start = qpos[min(sample, len(qpos) - 1), spec.action[0] : spec.action[1]].tolist()
        goal = profile.ee_pose(arm, tcp)
        request = BatchPlanRequest(arm=arm, base=spec.planner.base, start=start, goals=[goal])
        result = (await planner.plan_batch(request))[0]
        return None if result.reason == "ok" else f"{result.reason}: {result.detail}"[:120]

    return check


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _cm(v: Sequence[float]) -> str:
    return "(" + ", ".join(f"{100 * c:+.1f}" for c in v) + ") cm"


def _gap_text(gap: Gap, what: str) -> str:
    if gap.reference == "world":  # no absolute positions: only how far off
        return (
            f"{what} off the expert's in the world by {_cm(gap.offset)} = "
            f"{100 * gap.distance_m:.1f} cm, turned {gap.angle_deg:.0f} deg"
        )
    text = (
        f"{what} relative to {gap.reference} (robot axes, from its center): attempt "
        f"{_cm(gap.attempt)}, expert {_cm(gap.expert)}; off {_cm(gap.offset)} = "
        f"{100 * gap.distance_m:.1f} cm, turned {gap.angle_deg:.0f} deg"
    )
    if gap.twist_deg is not None:
        text += f" (closing axis {gap.twist_deg:.0f} deg about the pointing axis)"
    if gap.local is not None:
        text += f"; in {gap.reference}'s own frame off {_cm(gap.local)}"
    return text


VERDICT_TEXT: Final[dict[str, str]] = {
    "memory_diverged": "the Memory (or the Actor) diverged from the expert here; the expert's "
    "pose is reachable: move the named parameter toward the expert's pose",
    "transcription": "the call did not pass its stage's Memory spec as stored (the fields below "
    "differ), so the Memory is not what diverged: have the workflow pass the spec by its name "
    "({name}) rather than change the spec",
    "displaced": "an earlier call moved the object while no gripper held it, away from where the "
    "expert had it (below: which call, how far): that call's motion is where the attempt "
    "diverged, not this stage's spec; change how that stage moves clear of the object and leave "
    "the next stage's spec as it is",
    "execution_failed": "the grasp held and the motion kept to the expert's held path, yet the "
    "hold broke (below: where, the opening, how far the object moved): an execution failure of "
    "the execution stack, reported to the framework; the Memory reproduced the expert here, so "
    "do not change its parameters for this",
    "wrong_arm": "the attempt used the other arm than the expert: fix the rule that picks the arm",
    "expert_unplannable": "the expert's pose fails with the execution stack from this state: a "
    "low-level problem (reported to the framework), or an earlier stage left the arm or object "
    "elsewhere",
    "grasp_slip": "the close was at the expert's pose, yet the hold failed (nothing held or lost "
    "in the grasp's own call)",
    "hold_lost": "the grasp held, and the hold broke during the motion after it: the motion "
    "diverged from the expert's, not the grasp; move the motion's parameters (its direction, "
    "distance, turn or hinge) toward the expert's held motion below and leave the grasp spec as "
    "it is",
    "missing_stage": "the attempt skipped a stage the expert did: add it to the workflow",
    "end_state_gap": "every stage matched, but the objects ended off the demonstration's end",
    "path_frame": "the held path set off far from where the grasp held (below: how far): the "
    "frame the object was given when the path ran was not the one the hold was taken in, so the "
    "expert's motion ran turned or reversed; an execution/frame failure of the execution stack, "
    "reported to the framework; do not change the path's or the grasp's spec for this",
    "approach_diverged": "the release was at the expert's TCP, but the held object came onto it "
    "along another line (below: both lines): set the placement's approach (and approach_m) to the "
    "expert's line, and leave its offset and orientation as they are",
    "identification": "the identification at the start registered the wrong objects for the "
    "roles (below: which, measured against the demonstration's objects on this seed), so every "
    "stage after it was measured on the wrong object and its gaps are no evidence about the "
    "specs: fix how OBJECTS describes the roles (what kind of object each is, how it differs "
    "from the others: size, shape, what it holds) and leave the grasp and placement specs, their "
    "frames and x_toward as they are",
    "unreproducible": "the placement spec behind this stage cannot reproduce the expert's pose "
    "under any constrain on the demonstration's own state (the dry run, below): its numbers are "
    "not the lever, so do not tune its offset or rpy; refit it at its keypoint with fit_spec "
    "(another form: orient yaw where relative is not reproduced), or pass the constrain the dry "
    "run found closest",
}


def identification_line(found: Misidentified) -> str:
    """One identification error in the Reviser's words: the role, what it measured and what the
    demonstration's objects measure (their simulator names are never shown)."""
    expert = "" if found.expert_m is None else f"{100 * found.expert_m:.0f} cm"
    if found.role is None:
        near = "" if found.near is None else f" (the nearest registered is {found.near})"
        return f"an object the demonstration acts on, {expert} wide, was not registered{near}"
    text = f"{found.role} measured {100 * (found.measured_m or 0.0):.0f} cm wide"
    if len(found.objects) > 1:
        return (
            text + f": it holds {len(found.objects)} of the demonstration's objects at once (one "
            f"{expert} wide): it is them merged, not one object"
        )
    return text + f"; the demonstration's object there is {expert} wide"


def stage_line(row: Stage) -> str:
    """One stage in a line: what the expert did and how far the attempt was."""
    line = f"{row.arm} {row.keypoint} ({row.phase}, {row.event}"
    line += f" on {row.object})" if row.object else ")"
    if row.attempt_arm is not None and row.attempt_arm != row.arm:
        line += f": the attempt used the {row.attempt_arm} arm"
    gap = row.contact or row.place or row.commanded
    if gap is not None:
        line += f": off {100 * gap.distance_m:.1f} cm, {gap.angle_deg:.0f} deg"
    if row.event == "move" and row.call is not None:
        if row.verdict == "displaced":
            line += f": moved while not held, during {row.call}"
        else:
            line += f": the hold broke during {row.call}"
    if row.status != "matched":
        line += f" [{row.status}]"
    if row.verdict is not None:
        line += f" -> {row.verdict}"
    return line


def settled(diff: ExpertDiff, latch_miss: bool) -> list[str]:
    """The Memory parameters of the closes that matched the expert and held, when no stage
    diverged and the attempt failed at its end (`latch_miss`: a finish the official check did not
    confirm): those grasps did what the expert's did, so the failure is no evidence about them."""
    if not latch_miss or diff.success or diff.first is not None:
        return []
    if diff.verdict == "identification":
        return []
    return sorted(
        {
            given.split(" ")[0]
            for row in diff.stages
            if row.event == "close" and row.status == "matched" and row.held
            for given in row.parameters
        }
    )


def render(diff: ExpertDiff, latch_miss: bool = False) -> list[str]:
    """The review as the S1 Reviser reads it (typed guidance, appended to the try's log): the
    first diverging stage in full; for a failed attempt none of whose stages diverged beyond the
    tolerances, the stage nearest to diverging (the largest gap) in full all the same, since a
    gap within the tolerance can still be what failed (an empty grasp a little off), unless the
    attempt ended on a finish the official check did not confirm (`latch_miss`) and the closes
    matched and held (`settled`): then that is said plainly."""
    lines = [
        "Expert comparison (simulator truth, S1 only; the demonstration's state on this same "
        "seed, never its calls):"
    ]
    if diff.first is None and not diff.end and (diff.success or not diff.stages):
        lines.append("- every stage matched the expert within tolerance")
        return lines
    if diff.verdict == "identification":
        lines.append(f"- Identification error. {VERDICT_TEXT['identification']}.")
        lines += [f"  - {identification_line(found)}" for found in diff.identification]
        lines += [
            f"  - {role}: nothing was registered for it, though the workflow uses it"
            for role in diff.unregistered
        ]
        lines.append(
            "- Stages (measured on the objects as identified, not evidence about the specs): "
            + "; ".join(stage_line(row) for row in diff.stages)
            + "."
        )
        return lines
    if diff.first is not None:
        row = diff.stages[diff.first]
        lines.append(
            f"- First divergence: stage {diff.first + 1} of {len(diff.stages)}, "
            f"{stage_line(row)}. {VERDICT_TEXT[row.verdict or 'missing_stage']}."
        )
        lines += _detail(row)
    elif not diff.success and (nearest := _nearest(diff.stages)) is not None:
        row = diff.stages[nearest]
        lines.append(
            "- No stage diverged beyond the tolerances; the one nearest to diverging: stage "
            f"{nearest + 1} of {len(diff.stages)}, {stage_line(row)}."
        )
        lines += _detail(row)
        if found := settled(diff, latch_miss):
            lines.append(
                "- Every grasp matched the expert within tolerance and held, and the attempt "
                "ended on a finish the official check did not confirm: the failure is at the end "
                "(the task's success never latched), not in a grasp. The grasp parameters "
                f"({', '.join(found)}) are not evidence of the failure: their edits are refused "
                "on this attempt; look at the end state, the done check and the last stage."
            )
    for row in diff.stages:  # whatever diverged first: a skipped hand-over is always said
        if row.handover and (diff.first is None or diff.stages[diff.first] is not row):
            lines.append(f"- Hand-over skipped: {row.note}.")
    for end in diff.end:
        lines.append(f"- End state: {_gap_text(end.gap, end.object)}.")
    lines.append("- Stages: " + "; ".join(stage_line(row) for row in diff.stages) + ".")
    lines.append("- Gaps measured here are privileged measurements: one edit may close one.")
    return lines


def _gap_of(row: Stage) -> Gap | None:
    return row.contact or row.place or row.commanded


def _nearest(stages: Sequence[Stage]) -> int | None:
    """The stage with the largest measured gap (None: none measured)."""
    gaps = [(gap.distance_m, i) for i, row in enumerate(stages) if (gap := _gap_of(row))]
    return max(gaps)[1] if gaps else None


def _detail(row: Stage) -> list[str]:
    """One stage in full: its call, each gap, the approach, the hold, the plan check, the
    Memory parameters behind it, and the gap per spec field."""
    lines = []
    if row.call is not None:
        lines.append(f"- The call: {row.call} ({row.receipt or 'no receipt'}).")
    at = "the open" if row.event == "open" else "the close"
    if row.contact is not None:
        lines.append(f"- {_gap_text(row.contact, f'TCP at {at}')}.")
    if row.commanded is not None:
        lines.append(f"- {_gap_text(row.commanded, 'commanded TCP')}.")
    if row.world is not None:
        what = f"TCP at {at}" if row.contact is not None else "commanded TCP"
        lines.append(f"- {_gap_text(row.world, what)}.")
    if row.place is not None:
        what = "held object as last held before the open" if row.event == "open" else "held object"
        lines.append(f"- {_gap_text(row.place, what)}.")
    if row.motion is not None:
        lines += _motion_lines(row.motion)
    if row.standoff_m is not None:
        expert, given = row.standoff_m
        text = f"- Approach: the expert came {100 * expert:.1f} cm straight in"
        if given is not None:
            distance = "pre-place" if row.event == "open" else "pre-grasp"
            text += f", the call's {distance} distance was {100 * given:.1f} cm"
        if row.approach_deg is not None:
            text += f"; the attempt came in {row.approach_deg:.0f} deg off the expert's line"
        lines.append(text + ".")
    if row.opening is not None:
        held = "holding" if row.held else "holding nothing"
        lines.append(
            f"- The attempt's gripper after the close: opening {row.opening:.3f}, "
            f"{held}; the expert held {row.object}."
        )
    if row.plan is not None:
        lines.append(f"- The expert's pose, mapped into this attempt's scene: {row.plan}.")
    if row.parameters:
        lines.append(f"- Produced by the Memory parameters: {', '.join(row.parameters)}.")
    if row.spec_diff:
        lines.append(
            "- The call's spec differs from the Memory's: " + "; ".join(row.spec_diff) + "."
        )
    if row.invented:
        lines.append(
            f"- The call passed {', '.join(row.invented)}, which its stage's Memory call does not "
            "name: the Actor added it, so what it did is not the Memory's (a Memory that needs it "
            "writes it in the call)."
        )
    if (only := _orientation_only(row)) is not None:
        lines.append(only)
    elif (field := _field_line(row)) is not None:
        lines.append(field)
    if row.note:
        lines.append(f"- {row.note}.")
    return lines


def _turned(deg: float | None, axis: Sequence[float] | None) -> str:
    if deg is None or deg < 0.5:
        return "without turning"
    about = "" if axis is None else " about " + "(" + ", ".join(f"{v:+.2f}" for v in axis) + ")"
    return f"turning {deg:.0f} deg{about}"


def _motion_lines(m: Motion) -> list[str]:
    """A held motion as the Reviser reads it: the expert's against the attempt's."""
    frame = f"{m.reference}'s frame at the close (its own x, y, z)"
    lines = []
    if m.expert is not None:
        text = (
            f"- Held motion, expert: from the close to {m.expert_to} the TCP moved {_cm(m.expert)} "
            f"in {frame}, {_turned(m.expert_turn_deg, m.expert_axis)}"
        )
        if m.expert_world is not None:
            text += f"; along the robot's axes {_cm(m.expert_world)}"
        if m.expert_end is not None and m.expert_end_local is not None:
            text += (
                f"; to {m.expert_end}, its last held keypoint, {_cm(m.expert_end_local)}, "
                f"{_turned(m.expert_end_turn_deg, None)}"
            )
        lines.append(text + ".")
    else:
        lines.append("- Held motion, expert: it released without moving the held object.")
    text = (
        f"- Held motion, this attempt: from its settled close to where the hold broke the TCP "
        f"moved {_cm(m.attempt)} in {frame}, {_turned(m.attempt_turn_deg, m.attempt_axis)}; "
        f"along the robot's axes {_cm(m.attempt_world)}"
    )
    if m.commanded is not None:
        text += f"; the call commanded {_cm(m.commanded)}, {_turned(m.commanded_turn_deg, None)}"
    if m.opening is not None:
        text += f"; the opening fell to {m.opening:.3f}"
    lines.append(text + ".")
    if m.path_m is not None:
        text = (
            "- Execution: up to the break (and where its call was going) the TCP stayed within "
            f"{100 * m.path_m:.1f} cm of the expert's held path from its close, nearest "
            f"{m.nearest}"
        )
        if m.body_moved_m is not None:
            moved = 100 * m.body_moved_m
            text += f"; {m.reference} moved {moved:.1f} cm from the close to the end of that call"
        lines.append(text + ".")
    return lines


ORIENTATION_M: Final = 0.02
"""A gap whose position is within this is one of orientation only (`ReviewConfig.contact_m`)."""


def _orientation_only(row: Stage) -> str | None:
    """For a gap whose position is within `ORIENTATION_M` and whose orientation is off by more
    than 30 deg: say it is the orientation alone, so the offset is left as it is; nothing for a
    grasp turned only that held and was not blamed for a later failure (`Stage.turned_only`)."""
    if row.turned_only and row.verdict is None:
        return None
    gap = row.contact or row.commanded or row.place
    if gap is None or gap.distance_m > ORIENTATION_M or gap.angle_deg <= ORIENTATION_DEG:
        return None
    twist = f" ({gap.twist_deg:.0f} deg about the pointing axis)" if gap.twist_deg else ""
    return (
        f"- Orientation only: the position is within {100 * gap.distance_m:.1f} cm of the "
        f"expert's, the orientation {gap.angle_deg:.0f} deg off{twist}: keep offset_m as it is "
        "and turn the spec (rotate_spec for a pose grasp; yaw_deg, tilt_deg or lean_deg for "
        "another; a placement's orient or rpy_deg)."
    )


ORIENTATION_DEG: Final = 30.0
"""An orientation off by more than this counts (`ReviewConfig.contact_deg`)."""


def _field_line(row: Stage) -> str | None:
    """The gap in the terms of the spec fields behind it: the move toward the expert in the
    reference object's own frame (a grasp spec's `offset_m` frame for a whole-object grasp; a
    named part's frame differs, so no value is computed for it) and the turn about the
    pointing axis (`yaw_deg`); for a release, the TCP's offset from the target's (a place
    spec's `offset_m` in the target's frame)."""
    gap = row.contact or row.commanded
    if gap is None or gap.local is None or gap.reference == "world":
        return None
    toward = _cm(tuple(-c for c in gap.local))
    if row.event == "open":
        return (
            f"- Per spec field: to meet the expert the release moves by {toward} in "
            f"{gap.reference}'s own frame (the place spec's offset_m where its frame is "
            f"{gap.reference}'s), turned {gap.angle_deg:.0f} deg."
        )
    text = (
        f"- Per spec field: to meet the expert the contact moves by {toward} in "
        f"{gap.reference}'s own frame (the grasp spec's offset_m for a whole-object or pose "
        "grasp; a named part's frame differs)"
    )
    if gap.twist_deg is not None:
        text += (
            f", and turns {gap.twist_deg:.0f} deg about the pointing axis (yaw_deg; rotate_spec "
            "about the approach for a pose grasp)"
        )
    return text + "."


def progress_line(diff: ExpertDiff) -> str:
    """The first divergence so far, for the hinted Actor after an action."""
    if diff.first is None:
        return "Expert comparison so far: no stage diverged from the expert."
    row = diff.stages[diff.first]
    gap = row.contact or row.place or row.commanded
    text = f"Expert comparison so far: {stage_line(row)}"
    if gap is not None:
        text += f"; {_gap_text(gap, 'yours')}"
    return text + "."


def load(directory: Path) -> ExpertDiff | None:
    path = directory / EXPERT_DIFF
    return ExpertDiff.model_validate_json(path.read_bytes()) if path.is_file() else None
