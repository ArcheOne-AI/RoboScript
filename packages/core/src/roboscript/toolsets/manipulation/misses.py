"""Empty closes in a row on one grasp target with nothing measured in between.

An arm's last `grasp_object` names what it aims at (an object, or a registered part of it). Each
close of that arm that ends holding nothing counts one miss; the count runs on while the scene
registry stays the version it was at the previous miss (no `remeasure`, `register`, part
registration or refinement stored anything in between; `locate_part` registers a part on the
object's registered points, not a new measurement of the object), and starts again at one
otherwise. A hold or a new aim clears it. From `MissConfig.after` misses on, the receipt says
so and gives the point the grasp spec aims the TCP at (where the last `grasp_object` put it,
while the registry is as it was then; else the target's box center as last measured) against
where the fingers closed: the action itself is not blocked.
"""

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.robot import Arm

STATE: Final = "misses"  # per arm: `_Aim`, in the manipulation toolset's episode state


class MissConfig(BaseModel):
    """`after`: from how many empty closes in a row on one target, with the scene registry
    unchanged in between, the receipt asks for a new measurement (None: never)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    after: int | None = Field(default=3, ge=1)


@dataclass
class Aim:
    """An arm's grasp target (`obj`, its `part`; None: the whole), the TCP point its last grasp
    aimed at and the registry version then, and the empty closes in a row on it."""

    obj: str
    part: str | None
    point: tuple[float, float, float] | None = None
    point_version: int | None = None
    count: int = 0
    version: int | None = None


def aim(
    state: dict[str, Any],
    arm: Arm,
    obj: str,
    part: str | None,
    point: ArrayLike | None,
    version: int,
) -> None:
    """`arm`'s grasp aims at `obj` (`part` of it; None: the whole), its spec putting the TCP at
    `point` (None: not built) with the registry at `version`: a new target starts the count
    again."""
    aims: dict[Arm, Aim] = state.setdefault(STATE, {})
    old = aims.get(arm)
    if old is None or (old.obj, old.part) != (obj, part):
        old = aims[arm] = Aim(obj, part)
    if point is not None:
        p = np.asarray(point, dtype=float)
        old.point, old.point_version = (float(p[0]), float(p[1]), float(p[2])), version
    else:
        old.point, old.point_version = None, None


def closed(state: dict[str, Any], arm: Arm, holding: bool | None, version: int) -> Aim | None:
    """`arm` closed, holding `holding`, with the scene registry at `version`: its target with
    the empty closes in a row on it, None without a target or on a hold or an unknown hold
    (which clears the count)."""
    target: Aim | None = state.get(STATE, {}).get(arm)
    if target is None:
        return None
    if holding is not False:
        target.count, target.version = 0, None
        return None
    unchanged = target.version is not None and target.version == version
    target.count = target.count + 1 if unchanged else 1
    target.version = version
    return target


def note(
    arm: Arm,
    name: str,
    part: str | None,
    count: int,
    located: ArrayLike,
    at: ArrayLike,
    aimed: bool,
) -> str:
    """The receipt's note after `count` empty closes in a row on `name` (its `part`): the point
    its grasp spec aims the TCP at (`aimed`; else its box center as last measured: `located`)
    against where this close was (`at`, the TCP)."""
    located = np.asarray(located, dtype=float)[:3]
    at = np.asarray(at, dtype=float)[:3]
    off = np.round(located - at, 3) + 0.0  # no "-0.000"
    what = name if part is None else f"{name}'s {part}"
    loc = ", ".join(f"{v:.3f}" for v in located)
    tcp = ", ".join(f"{v:.3f}" for v in at)
    where = (
        f"its grasp spec aims the TCP at ({loc}) on its last measurement"
        if aimed
        else f"its box center was last measured at ({loc})"
    )
    return (
        f"{arm}: {count} empty closes in a row on {what} with nothing measured in between; "
        f"{where}, this close at ({tcp}), {off[0]:+.3f}, {off[1]:+.3f}, {off[2]:+.3f} m away: "
        "it may have moved; measure it again (remeasure; if that is not sure, observe it and "
        "register it again) before closing again"
    )
