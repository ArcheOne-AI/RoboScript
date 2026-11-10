"""The `manipulation` toolset's results and what a model reads of them
(docs/concepts/agent-runtime.md): a receipt in a few lines (its reason and whether that is a
failure, commanded against measured TCP, a followed path's waypoints and holds, the gripper, the
steps, the note) and a reach check. A receipt's `tcp_path` stays in the event log; it is never
rendered."""

from collections.abc import Sequence
from typing import Final

from roboscript.contracts.receipt import (
    HARD_FAILURES,
    ActionReceipt,
    PathReport,
    ReceiptReason,
)
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.manipulation import Reach

SOFT: Final = frozenset(
    {ReceiptReason.clamped, ReceiptReason.stopped_short, ReceiptReason.deviated}
)
"""Reasons that are information for the model: the action ran, and nothing ends."""
_REFUSED: Final = frozenset({ReceiptReason.invalid_args, ReceiptReason.rejected_repeat})
_HOLDING: Final = {True: "holding something", False: "holding nothing", None: "holding unknown"}


def render_receipt(receipt: ActionReceipt) -> list[ContentPart]:
    return [TextPart(text=receipt_text(receipt))]


def render_reach(result: Reach) -> list[ContentPart]:
    if result.reachable:
        lines = [f"Reachable ({result.reason}): {result.steps} steps planned. Nothing moved."]
    else:
        outcome = "a hard failure" if result.reason in HARD_FAILURES else "refused"
        lines = [
            f"Not reachable ({result.reason}): run as it is, it would be {outcome}. Nothing moved."
        ]
    lines += [f"{arm} TCP target {_xyz(pose.p)}." for arm, pose in result.targets.items()]
    if result.note:
        lines.append(f"Note: {result.note}.")
    return [TextPart(text="\n".join(lines))]


def receipt_text(receipt: ActionReceipt) -> str:
    lines = [f"Receipt: {receipt.reason} ({_kind(receipt.reason)})."]
    arm = receipt.arm
    if receipt.commanded is not None:
        line = f"{arm} TCP commanded {_xyz(receipt.commanded.tcp.p)}"
        if receipt.achieved is not None and receipt.deviation_m is not None:
            line += (
                f", measured {_xyz(receipt.achieved.tcp.p)}: {receipt.deviation_m * 1000:.0f} mm "
                f"and {receipt.achieved.orient_err_deg:.1f} deg off"
            )
        lines.append(line + ".")
    if receipt.gripper is not None:
        gripper = receipt.gripper
        opening = "not measured" if gripper.measured is None else f"{gripper.measured:.3f}"
        lines.append(
            f"{arm} gripper: commanded {gripper.commanded:.2f}, opening {opening}, "
            f"{_HOLDING[gripper.holding]}."
        )
    if receipt.path is not None:
        lines.append(path_text(receipt.path))
    remaining = "no step limit" if receipt.steps_left is None else f"{receipt.steps_left} left"
    lines.append(f"Steps: {receipt.steps_used} used, {remaining}.")
    if receipt.note:
        lines.append(f"Note: {receipt.note}.")
    return "\n".join(lines)


def path_text(path: PathReport) -> str:
    """A followed path in a line: waypoints reached, each arm's deviation from each (mm), its
    hold."""
    parts = [f"Path: reached {path.reached} of {path.waypoints} waypoints"]
    for arm, hold in path.holds.items():
        off = ", ".join(f"{v * 1000:.0f}" for v in path.deviation_m.get(arm, []))
        held = "hold kept"
        if not hold.kept:
            assert hold.tcp is not None  # a lost hold says where
            held = f"hold lost after waypoint {hold.lost_at} at {_xyz(hold.tcp)}"
        parts.append(f"{arm} off by [{off}] mm, {held}")
    if path.start_m > 0.005:
        parts.append(f"the grasp was {path.start_m * 1000:.0f} mm from the path's start")
    return "; ".join(parts) + "."


def _kind(reason: ReceiptReason) -> str:
    if reason in HARD_FAILURES:
        return "hard failure"
    if reason in SOFT:
        return "information, not a failure"
    if reason in _REFUSED:
        return "refused, nothing moved"
    if reason == ReceiptReason.budget_exhausted:
        return "the step limit is reached"
    return "done"


def _xyz(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{round(float(v), 3) + 0.0:.3f}" for v in values) + ")"
