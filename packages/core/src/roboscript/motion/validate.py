"""Request checks every planner makes before planning: the arm exists and vectors fit it."""

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.robot import Arm

T = TypeVar("T")


def arm_of(arms: Mapping[Arm, T], arm: Arm) -> T:
    if arm not in arms:
        raise ProtocolError("invalid_request", f"no {arm} arm; this planner has {sorted(arms)}")
    return arms[arm]


def check_joints(arm: Arm, dof: int, vectors: Iterable[Sequence[float]]) -> None:
    for vector in vectors:
        if len(vector) != dof:
            raise ProtocolError(
                "invalid_request", f"the {arm} arm has {dof} joints, got a vector of {len(vector)}"
            )


def check_links(arm: Arm, known: Sequence[str], asked: Iterable[str]) -> None:
    if unknown := sorted(set(asked) - set(known)):
        raise ProtocolError(
            "invalid_request", f"the {arm} arm has no links {unknown}; it has {list(known)}"
        )
