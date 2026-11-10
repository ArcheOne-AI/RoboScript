"""Which arm does which of the Memory's stages in this episode (`plan_arms`,
docs/concepts/embodied-core.md): one assignment over the workflow's holds (`memory.chains`).
Holds open at the same time take different arms, a hand-over's receiving hold the other arm, and
an arm the instruction names keeps the stage it does. Each hold's arm comes from the single-object
rule the toolset evaluates (`ArmChoice`): the demonstration's arm on its own scene (`DEMO_ARMS`),
the instruction's arm (a semantic reading, `perception.identify`, never a word match here), the arm
holding the object, the Memory's `arm side:` (`ARM_SIDES`) for an object whose x is known, else the
nearer arm unless only the other plans the grasp (with carry and placement). A Memory keeps the
rule, never the arm: the side an object lies on changes with the scene."""

import re
from collections.abc import Mapping, Sequence
from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.robot import Arm
from roboscript.contracts.tools import ContentPart, TextPart

__all__ = [
    "ArmChoice",
    "ArmPlan",
    "HoldArm",
    "demo_arm",
    "named_arms",
    "nearest_arm",
    "render_plan",
]

Why = Literal[
    "instruction",
    "holding",
    "nearer",
    "reach",
    "place",
    "neither",
    "only",
    "both_named",
    "demonstration",
    "stage",
    "direction",
    "end",
]
DEMO_MATCH_M: Final = 0.1
"""A demonstrated grasp is of the object whose measured start lies this near its TCP at most
(horizontally), the nearest such."""
_NAMED: Final = re.compile(r"\b(left|right)[- ](?:hand|arm|gripper)\b(?![- ]side)", re.IGNORECASE)
"""An arm the instruction names: `left arm`, `right hand`, `left-hand gripper` (not a `left-hand
side`)."""


class ArmChoice(BaseModel):
    """The arm to use and why: `instruction` (it names the arm), `holding` (that arm holds the
    object), `nearer` (its start is nearer the object), `reach` (only it plans the grasp), `place`
    (only it plans the grasp, the carry and the placement given), `neither` (no arm plans them
    all: a hand-over; `arm` is the one that plans the grasp, nearer first), `direction` (the
    role's `arm side:` and where its x points give it), `end` (both arms plan the grasp, carry
    and placement given, and this one's leaves the object nearer where the done criteria want it
    relative to another object) or `only` (the robot has one arm) or
    `stage` (the workflow grasps the object with both arms in turn, `candidates` in order, and
    this is the stage's arm); `side`: the `arm side:` rule the chosen arm follows where the
    object's x is decided (on the demonstration's scene: what the Memory's OBJECTS line should
    say);
    `distance_m` is each arm's start from the object's center, `reachable` each checked arm's
    grasp plan, `note` what else matters."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    why: Why
    candidates: list[Arm] = []
    distance_m: dict[Arm, float] = {}
    reachable: dict[Arm, bool] = {}
    side: str | None = None
    note: str = ""


def named_arms(instruction: str | None) -> list[Arm]:
    """The arms the instruction names, in the order it names them first (none: it names none)."""
    found: list[Arm] = []
    for match in _NAMED.finditer(instruction or ""):
        arm: Arm = "left" if match.group(1).lower() == "left" else "right"
        if arm not in found:
            found.append(arm)
    return found


def demo_arm(
    grasps: Sequence[Mapping[str, object]], starts: Mapping[str, Sequence[float]], obj: str
) -> tuple[Arm, str] | None:
    """The arm the demonstration first grasped `obj` with, and where (`DEMO_ARMS`' grasps, in
    frame order): each grasp is of the object whose start (`starts`, by id: its center as first
    measured, on the demonstration's scene) lies nearest its TCP, within `DEMO_MATCH_M`
    horizontally; None when no grasp is of `obj`."""
    for grasp in grasps:
        tcp = np.asarray(grasp["tcp"], dtype=float)[:2]
        gaps = {o: float(np.linalg.norm(np.asarray(c)[:2] - tcp)) for o, c in starts.items()}
        found = min(gaps, key=gaps.__getitem__, default=None)
        if found == obj and gaps[found] <= DEMO_MATCH_M:
            return grasp["arm"], str(grasp["keypoint"])  # type: ignore[return-value]
    return None


def nearest_arm(center: Sequence[float], starts: Mapping[Arm, Sequence[float]]) -> list[Arm]:
    """The arms by the horizontal distance of their start TCP from `center`, nearest first."""
    point = np.asarray(center, dtype=float)[:2]
    return sorted(
        starts, key=lambda arm: float(np.linalg.norm(np.asarray(starts[arm])[:2] - point))
    )


FIXED: Final = frozenset({"instruction", "holding", "stage", "done"})
"""Why an arm a plan cannot change: the instruction names it, the arm holds the role now, the
workflow's per-stage arms of a hand-over or a regrasp stand, or the hold is done."""
_RANK: Final = {
    "instruction": 6,
    "holding": 6,
    "done": 6,
    "stage": 5,
    "demonstration": 4,
    "direction": 3,
    "end": 2,
    "place": 2,
    "reach": 2,
}
"""How strongly a single-object choice holds when two holds open at once ask for one arm: the
weaker one changes arm."""


def rank(why: str) -> int:
    return _RANK.get(why, 1)


class HoldArm(BaseModel):
    """One hold of the plan: its role, the arm, why (`ArmChoice.why`, `receiver` for a
    hand-over's receiving hold, `apart` for one moved to the other arm because another hold is
    open at the same time, `agent` for the Agent's choice), the stages it spans, each arm's
    check (`ok`, why it does not plan, or `unchecked`) and the `arm side:` rule the arm follows."""

    model_config = ConfigDict(frozen=True)

    role: str
    arm: Arm
    why: str
    stages: list[str] = []
    checked: dict[Arm, str] = {}
    side: str | None = None
    note: str = ""


class ArmPlan(BaseModel):
    """Which arm does which stage in this episode (`plan_arms`): each stage's arms (in the
    order its holds open), each hold, and what else matters. `override`: the Agent's reason
    where it chose arms itself."""

    model_config = ConfigDict(frozen=True)

    stages: dict[str, list[Arm]] = {}
    holds: list[HoldArm] = []
    override: str | None = None
    note: str = ""

    def fact(self) -> dict[str, object]:
        """The plan as the `ARM_PLAN` fact (`tools.frames.ARM_PLAN`)."""
        roles: dict[str, list[str]] = {}
        for hold in self.holds:
            roles.setdefault(hold.role, []).append(hold.arm)
        return {
            "stages": {stage: list(arms) for stage, arms in self.stages.items()},
            "roles": roles,
            "holds": [
                h.model_dump(mode="json", include={"role", "arm", "stages", "why"})
                for h in self.holds
            ],
            "override": self.override,
        }


_WHY: Final = {
    "instruction": "the instruction names it",
    "holding": "it holds the object now",
    "done": "this hold is done",
    "nearer": "its start is nearer the object",
    "reach": "only it plans the grasp",
    "place": "only it grasps, carries and places it",
    "neither": "neither arm does the whole hold alone: a hand-over",
    "only": "the robot has no other arm",
    "both_named": "the instruction names both arms and pairs neither with this object",
    "demonstration": "the demonstration did this hold with it, on this same scene",
    "stage": "the workflow does this stage of the object with it",
    "direction": "the object's x points the way its arm side says this arm takes",
    "end": "its grasp, carry and placement leave the object nearest where the done criteria want",
    "receiver": "it receives the object from the other arm (a hand-over)",
    "apart": "another hold is open at the same time on the other arm",
    "agent": "your choice",
}


def render_plan(plan: ArmPlan) -> list[ContentPart]:
    lines = []
    for n, hold in enumerate(plan.holds, 1):
        checks = ", ".join(f"{a}: {c}" for a, c in sorted(hold.checked.items()))
        line = (
            f"Hold {n}, {hold.role} ({', '.join(hold.stages) or 'no stage'}): the {hold.arm} arm, "
            f"{_WHY.get(hold.why, hold.why)}."
        )
        if checks:
            line += f" Checked: {checks}."
        if hold.side is not None:
            line += f" Its x relative to this arm: arm side: {hold.side}."
        if hold.note:
            line += f" Note: {hold.note}."
        lines.append(line)
    if plan.stages:
        lines.append(
            "Stage arms: "
            + "; ".join(f"{stage}: {' and '.join(arms)}" for stage, arms in plan.stages.items())
            + ". In a stage's calls, <arm> is its arm and <arm of <role>> the arm of the role's "
            "newest hold."
        )
    if plan.override:
        lines.append(f"Chosen by you: {plan.override}.")
    if plan.note:
        lines.append(f"Note: {plan.note}.")
    return [TextPart(text="\n".join(lines) or "Nothing to plan.")]
