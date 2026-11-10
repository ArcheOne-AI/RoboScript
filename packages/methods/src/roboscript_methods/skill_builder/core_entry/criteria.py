"""The per-call criteria (`check.Criteria`) measured on entry-check reports by the gap rule
(`gaps.widest_gap`): for each, the quantity truth measured in every call it judges, the gap and
the threshold where the gap is clear; where none is, the value that judges instead, and why."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final

from roboscript_methods.skill_builder.core_entry.gaps import widest_gap


@dataclass(frozen=True)
class Quantity:
    what: str
    values: Callable[[dict[str, Any]], Iterable[float]]
    log: bool
    floor: float
    otherwise: float | None
    """The value without a clear gap (None: the criterion judges nothing, or as the binding)."""
    why: str


def _grasps(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [g for g in report["grasp"] if "error" not in g and "skipped" not in g and g["planned"]]


def _places(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [g["place"] for g in _grasps(report) if "place" in g]


def _rows(report: dict[str, Any], group: str) -> list[dict[str, Any]]:
    rows = report[group]
    return rows if isinstance(rows, list) else []


def _moves(report: dict[str, Any]) -> Iterable[float]:
    for row in _rows(report, "move_to_pose") + _rows(report, "displacement"):
        if row.get("ok") and ("pos_m" in row or "error_m" in row):
            yield row.get("pos_m", row.get("error_m"))


QUANTITIES: Final[dict[str, Quantity]] = {
    "held_m": Quantity(
        "each planned grasp's object, its rise after the lift (m)",
        lambda r: (g["rose_m"] for g in _grasps(r)),
        False,
        0.0,
        0.05,
        "the default",
    ),
    "opened": Quantity(
        "the gripper's reading after each open and each close on nothing",
        lambda r: (row["reading"] for row in _rows(r, "gripper") if "reading" in row),
        False,
        0.0,
        0.9,
        "the default",
    ),
    "resting_m": Quantity(
        "each place, the object's bottom centre from the target's height (|dz|, m)",
        lambda r: (abs(p["dz_m"]) for p in _places(r)),
        True,
        0.001,
        0.015,
        "the default",
    ),
    "placed_m": Quantity(
        "each place, the object's bottom centre from the target, horizontally (m)",
        lambda r: (p["off_m"] for p in _places(r)),
        True,
        0.001,
        0.03,
        "the default: a place judged without an offset would pass an object resting anywhere",
    ),
    "upright_deg": Quantity(
        "each place, the object's tilt from how it stood when picked (deg)",
        lambda r: (p["tilt_deg"] for p in _places(r)),
        True,
        1.0,
        None,
        "not judged: round objects come to rest at any tilt",
    ),
    "move_m": Quantity(
        "each move whose receipt said ok, its TCP from the target (m)",
        _moves,
        True,
        0.001,
        None,
        "the binding's stop tolerance, what an ok receipt promises",
    ),
    "move_deg": Quantity(
        "each move_to_pose whose receipt said ok, its TCP's rotation from the target (deg)",
        lambda r: (
            row["deg"] for row in _rows(r, "move_to_pose") if row.get("ok") and "deg" in row
        ),
        True,
        0.1,
        None,
        "the binding's stop tolerance, what an ok receipt promises",
    ),
}


def measure(reports: list[dict[str, Any]], decimals: int = 4) -> dict[str, dict[str, Any]]:
    """Each criterion's record: the quantity, its scale, how many values, the widest gap; the
    `value` judging (the gap's threshold where it is clear, else `otherwise`) and `measured`."""
    found = {}
    for name, q in QUANTITIES.items():
        values = [float(v) for report in reports for v in q.values(report)]
        gap = widest_gap(values, q.log, q.floor)
        record: dict[str, Any] = {
            "quantity": q.what,
            "scale": f"log10, floor {q.floor}" if q.log else "linear",
            "values": len(values),
        }
        if gap is not None:
            record |= gap.record()
        if gap is not None and gap.clear:
            record |= {"value": round(gap.threshold, decimals), "measured": True}
        else:
            record |= {"value": q.otherwise, "measured": False, "instead": q.why}
        found[name] = record
    return found
