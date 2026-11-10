"""What the executor takes (per-arm segments) and what it plans (steps plus what to check them by).

An action is a list of segments per arm, run pair by pair: the k-th segments of both arms execute in
the same `take_action` calls, an arm without a k-th segment holding its drive target.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from roboscript.contracts.action import ActionStep
from roboscript.contracts.geometry import Pose
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.robot import Arm, RobotState


@dataclass(frozen=True)
class Move:
    """Move the arm's ee to `target` (world). A free move follows a planned path; a straight one
    keeps to the line from where the arm is (IK along it: the approach of a grasp). A contact
    move is meant to touch, so ending short of the target is expected and not reported, unless
    it holds something and ends far from it (a place that did not get there). A bounded
    move is a displacement, clamped to the configured translation and rotation."""

    target: Pose
    straight: bool = False
    contact: bool = False
    bounded: bool = False
    past_m: float = 0.0
    """A placement's: how far past where the held object is to be placed its target lies (a
    negative `stop_back_m`, pushing it on): its stop there on the way is no shortfall."""
    free_fallback: bool = False
    """A straight move that, where no straight way reaches its target (IK along the line, then
    the orientation-locked plans), is planned as a free move instead (a placement's last leg)."""


@dataclass(frozen=True)
class Grip:
    """Command the gripper to `value` (normalized: 1 open, 0 closed). A grasp is a close meant to
    hold something: closing on nothing is then `empty_grasp`. A pull's hold may slip at the end
    of the pull, and a close that is no grasp (a press) may pinch something it then drops:
    losing either is noted, not a failure."""

    value: float
    grasp: bool = False
    pull: bool = False


Segment = Move | Grip


@dataclass(frozen=True)
class MoveCheck:
    """A planned move: it ends with batch step `end`, commanding ee `target` (TCP `tcp`) along the
    planned TCP positions `path` (n, 3)."""

    arm: Arm
    end: int
    target: Pose
    tcp: Pose
    path: NDArray[np.float64]
    contact: bool
    straight: bool = False
    past_m: float = 0.0
    free_fallback: bool = False


@dataclass(frozen=True)
class GripCheck:
    """A planned gripper command; `end` is its last batch step (the settle, or the last hold of
    an opening)."""

    arm: Arm
    end: int
    value: float
    grasp: bool
    pull: bool = False


@dataclass(frozen=True)
class Plan:
    """A planned action: the steps to execute from `before` and what its receipt checks, or
    `refused`, the receipt of an action that cannot run (nothing executes; a pair refused for
    the arms' gap keeps its planned moves)."""

    before: RobotState
    steps: list[ActionStep] = field(default_factory=list)
    moves: list[MoveCheck] = field(default_factory=list)
    grips: list[GripCheck] = field(default_factory=list)
    arms: list[Arm] = field(default_factory=list)  # the arms the action commands
    notes: list[str] = field(default_factory=list)
    clamped: bool = False
    refused: ActionReceipt | None = None
    lenient: frozenset[Arm] = frozenset()  # arms whose hold may slip, when the action starts
    placed: frozenset[Arm] = frozenset()  # arms whose last move placed what they hold, at the end


def tcp_gap(plan: Plan, samples: int = 50) -> tuple[float, float]:
    """How far apart the left and right TCPs are (meters) when `plan` starts and where they come
    closest while it runs, along the planned TCP paths: the moves of a pair run at shared path
    fractions (in the same steps), `samples` of them compared. A moving arm starts where its
    planned path does (its drive target, where the plan starts); an arm without a move stays
    where `plan.before` measured it."""
    at = {arm: np.array([state.tcp.p]) for arm, state in plan.before.arms.items()}
    for move in reversed(plan.moves):  # the first move of each arm wins
        at[move.arm] = move.path[:1]
    start = closest = _nearest(at)
    for end in sorted({move.end for move in plan.moves}):
        paths = dict(at)
        for move in plan.moves:
            if move.end == end:
                index = np.linspace(0.0, len(move.path) - 1, samples)
                paths[move.arm] = np.stack(
                    [np.interp(index, np.arange(len(move.path)), axis) for axis in move.path.T],
                    axis=1,
                )
        closest = min(closest, _nearest(paths))
        at = {arm: path[-1:] for arm, path in paths.items()}
    return start, closest


def _nearest(positions: dict[Arm, NDArray[np.float64]]) -> float:
    """The smallest distance between the left and right positions (n, 3) at equal indices (one
    position stands for every index)."""
    return float(np.linalg.norm(positions["left"] - positions["right"], axis=1).min())


class Reach(BaseModel):
    """A planned action, nothing moved: whether it would run (`reachable`, also when clamped),
    the reason its receipt would give (ok, clamped, or why it is refused), the TCP target of each
    arm's last move (after clamping; a refused target when refused), the steps it would send
    (a close adds at most a few settling holds) and a note."""

    reachable: bool
    reason: ReceiptReason
    targets: dict[Arm, Pose]
    steps: int
    note: str

    @classmethod
    def of(cls, refused: ActionReceipt, targets: dict[Arm, Pose] | None = None) -> "Reach":
        """A refusal as a reach check's result."""
        found = dict(targets or {})
        if refused.arm is not None and refused.commanded is not None:
            found[refused.arm] = refused.commanded.tcp
        return cls(
            reachable=False, reason=refused.reason, targets=found, steps=0, note=refused.note
        )


def reach(plan: Plan) -> Reach:
    """What a planned action would do (`Reach`)."""
    targets = {move.arm: move.tcp for move in plan.moves}  # the last move of each arm
    if plan.refused is not None:
        return Reach.of(plan.refused, targets)
    reason = ReceiptReason.clamped if plan.clamped else ReceiptReason.ok
    return Reach(
        reachable=True,
        reason=reason,
        targets=targets,
        steps=len(plan.steps),
        note="; ".join(plan.notes),
    )
