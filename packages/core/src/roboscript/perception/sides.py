"""Which sides of the robot a Task Memory may name (docs/concepts/learning.md,
docs/concepts/perception.md).

A side of the robot (its left, its right) in a Memory comes from one of two places. The task
itself names it: every instruction of the task says "to the left of", "tip ends left", "from
left to right", and the Memory's TASK statement says it too; such a side is fixed in the robot's
frame. Or the demonstration's arm put it there: the right arm set the can down on its own side
of the pot, and on a scene where the left arm acts the can ends on the other side. A side of the
second kind is written relative to the acting arm (`ARM_SIDE`, `OTHER_SIDE`; a done criterion's
`arm`, an end target's `arm` and `mirror`), never as left or right, so that it follows whichever
arm acts. `task_sides` says whether the TASK statement names a side (the linter's `side_bound`
and the done check's end targets read it).
"""

import re
from typing import Final, Literal

from roboscript.contracts.robot import Arm

ARM_SIDE: Final = "arm_side"
"""The side of the robot of the arm that acts on the object (left for the left arm)."""
OTHER_SIDE: Final = "other_side"
"""The side of the other arm."""
ArmRelative = Literal["arm_side", "other_side"]

_SIDE: Final = re.compile(
    r"(?<![^\W_])(?:left|right)(?![^\W_])"
    r"(?![- ]?(?:arms?|hands?|grippers?|handed)(?![^\W_]))",
    re.IGNORECASE,
)
"""left or right as a side (not the left arm, hand or gripper)."""


def names_side(text: str) -> bool:
    """Whether `text` names a side of the robot (left, right) other than as an arm."""
    return _SIDE.search(text) is not None


def task_sides(statement: str) -> bool:
    """Whether a TASK statement names a side: its sides are the task's, fixed in the robot's
    frame; any other side in the Memory came from the demonstration's arm."""
    return names_side(statement)


def resolve(side: str, arm: Arm | None) -> str | None:
    """An arm-relative side (`ARM_SIDE`, `OTHER_SIDE`) as the robot's `left` or `right` for the
    arm `arm` acting (None: not known, so undecidable); any other side as it is."""
    if side not in (ARM_SIDE, OTHER_SIDE):
        return side
    if arm is None:
        return None
    if side == ARM_SIDE:
        return arm
    return "right" if arm == "left" else "left"
