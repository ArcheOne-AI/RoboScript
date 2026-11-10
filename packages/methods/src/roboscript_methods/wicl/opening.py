"""wicl's open-time hook (`learning.opening.Opener`; docs/concepts/perception.md,
docs/concepts/learning.md): the one-time identification as an episode opens, which the
`EpisodeDriver` runs for every episode and the P3 host for its own (`Identification`).

When the Hub episode has the system-only `identify` tool (the perception toolset is mounted: an
Actor's or the Demo Reader's episode on a live scene), the runner calls it on the system channel
before the brain runs, with the episode's live instruction and the Memory's OBJECTS roles (with
none yet, the objects the instruction is about). Its outcome, rendered as the system call's
result (which the event log keeps), goes into the role's brief, and the episode's wall clock
restarts after it: identifying is not the role's time. A failure ends the episode, classified:
`infra_error` and `timeout` are infrastructure (in S1 and S2 the episode is void and run again;
a caller may first try again in place, `Retries`, as the P3 host does), among them a service
failing under `identify`: a vision or motion server error (a 5xx, e.g. a CUDA out-of-memory), a
model API's 5xx, a connection that failed, a service not ready (`rpc.service_failures_as_infra`);
anything else, a refused request, `invalid_arguments`, a tool's own error (a model refusing),
`internal_error` (a bug) or a malformed OBJECTS section, is a configuration error, which stops
the run. The brain runs only after a success. The P3 host runs the same hook
(`agent_host.opening_hook`). The roles the workflow grasps or places go with it
(`memory.carried_roles`: the identification never registers a flat patch for one), and with
`Retries.unresolved` (the P3 host's 1) the roles it left without an object are identified once
more on a new observation (a `perception.reidentify` event names them), both answers in the
brief.

A later episode of the same scene (the tries of an S2 seed) may start from an identification
instead (`Reuse`, saved as the episode's `identify.json`): when its roles and instruction are the
ones the episode would identify (`reusable`) and its first frame still looks like the one the
identification saw (`scene_drift`: a cheap image difference, at most `DRIFT`), its frames, facts
and aliases seed the Hub episode, its rendering is the brief, and the reuse is logged
(`perception.reuse`) naming its source; a scene that drifted is identified again, and that is
logged too.
"""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

import anyio
import numpy as np
from pydantic import BaseModel, ConfigDict

from roboscript.blobs import BlobStore
from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.events import EventKind
from roboscript.contracts.hub import EpisodeSession
from roboscript.contracts.observation import Observation
from roboscript.contracts.tools import TextPart
from roboscript.hub import Hub
from roboscript.learning.opening import (
    IDENTIFY,
    INFRA_CODES,
    Drift,
    Opening,
    Retries,
    Reuse,
)
from roboscript.memory import (
    MemoryDoc,
    carried_roles,
    object_orders,
    object_roles,
    object_towards,
    order_text,
)
from roboscript.records import read_events
from roboscript.tools import IDENTIFIED
from roboscript.toolsets.perception import identify_objects

SCALE: Final = 8


def reuse_key(
    roles: Mapping[str, str],
    instruction: str,
    towards: Mapping[str, str] | None = None,
    orders: Mapping[str, str] | None = None,
) -> str:
    """What an identification is of: its roles, its instruction and the roles' `x toward:` and
    `order:` lines."""
    of: dict[str, object] = {"roles": dict(roles), "instruction": instruction}
    if towards:
        of["towards"] = dict(towards)
    if orders:
        of["orders"] = dict(orders)
    text = canonical_json(of)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def roles_of(memory: MemoryDoc | None) -> dict[str, str]:
    """The Memory's OBJECTS roles (none without a Memory); ValueError for a malformed line."""
    return {} if memory is None else object_roles(memory)


def towards_of(memory: MemoryDoc | None) -> dict[str, str]:
    """The Memory's OBJECTS `x toward:` words (none without a Memory)."""
    return {} if memory is None else object_towards(memory)


def orders_of(memory: MemoryDoc | None) -> dict[str, str]:
    """The Memory's OBJECTS `order:` lines as text (none without a Memory)."""
    if memory is None:
        return {}
    return {role: order_text(order) for role, order in object_orders(memory).items()}


def reusable(reuse: Reuse | None, memory: MemoryDoc | None, instruction: str | None) -> bool:
    """Whether an episode reading `memory` under `instruction` may start from `reuse`: the same
    roles and instruction (a Reviser that changed OBJECTS makes the next try identify again)."""
    if reuse is None or instruction is None:
        return False
    try:
        found = reuse_key(roles_of(memory), instruction, towards_of(memory), orders_of(memory))
        return reuse.key == found
    except ValueError:
        return False


async def scene_drift(reuse: Reuse, first: Observation, blobs: BlobStore) -> Drift:
    """How far `first`, a try's first frame, is from the frame the identification of `reuse`
    saw: the mean absolute difference of their images at 1/8 scale, in the camera they share
    that differs most."""
    fact = reuse.seed.facts.get(IDENTIFIED)
    entry = next((f for f in reuse.seed.frames if fact and f.frame_id == fact["frame"]), None)
    if entry is None:
        return Drift(source=reuse.source, difference=None)
    seen = Observation.model_validate_json(await blobs.aget_bytes(entry.observation))
    found = []
    for camera, frame in first.cameras.items():
        before = seen.cameras.get(camera)
        if before is None:
            continue
        now, then = await blobs.aget_array(frame.rgb), await blobs.aget_array(before.rgb)
        found.append(await anyio.to_thread.run_sync(image_difference, now, then))
    shared = [difference for difference in found if difference is not None]
    return Drift(source=reuse.source, difference=max(shared) if shared else None)


async def open_identification(
    hub: Hub,
    session: EpisodeSession,
    *,
    instruction: str | None,
    memory: MemoryDoc | None,
    source: str,
    reuse: Reuse | None = None,
    drift: Drift | None = None,
    retries: Retries | None = None,
) -> Opening:
    """Identify once as `session` opens (tried again in place after an infrastructure failure
    as `retries` allows), or take `reuse` (the Hub episode was opened with its seed; see
    `reusable` and `scene_drift`, whose outcome `drift` is logged); then restart the episode's
    clock, or end the episode after a failure. `source` names this episode for a later reuse."""
    opened = await identify_tried(hub, session, instruction, memory, source, reuse, drift, retries)
    if opened.failed:
        await hub.end_episode(session.episode_id, opened.end_reason, opened.note)
    elif opened.status != "skipped":
        hub.restart_clock(session.episode_id, "identification")
    return opened


async def identify_tried(
    hub: Hub,
    session: EpisodeSession,
    instruction: str | None,
    memory: MemoryDoc | None,
    source: str,
    reuse: Reuse | None = None,
    drift: Drift | None = None,
    retries: Retries | None = None,
) -> Opening:
    """Identify (or take `reuse`), tried again in place after an infrastructure failure as
    `retries` allows; the episode is left as it is, whatever the outcome."""
    retries = retries or Retries()
    opened = await _identify(
        hub, session, instruction, memory, source, reuse, drift, retries.unresolved
    )
    for attempt in range(1, retries.attempts):
        if opened.status != "infra":
            break
        await anyio.sleep(retries.backoff_s * 2 ** (attempt - 1))
        opened = await _identify(
            hub, session, instruction, memory, source, None, None, retries.unresolved
        )
        if opened.status == "infra":
            opened = opened.model_copy(update={"note": f"{opened.note} (try {attempt + 1})"})
    return opened


async def _identify(
    hub: Hub,
    session: EpisodeSession,
    instruction: str | None,
    memory: MemoryDoc | None,
    source: str,
    reuse: Reuse | None,
    drift: Drift | None,
    unresolved: int = 0,
) -> Opening:
    if IDENTIFY not in session.system_tools or instruction is None:
        return Opening(status="skipped")
    try:
        roles = roles_of(memory)
    except ValueError as error:
        return Opening(status="config", note=f"the Memory's OBJECTS section: {error}")
    shift = None if drift is None else drift.difference
    if reuse is not None:
        data = {"source": reuse.source, "key": reuse.key, "drift": shift}
        hub.note(session.episode_id, EventKind.PERCEPTION_REUSE, data)
        note = f"reused from {reuse.source}"
        return Opening(status="reused", note=note, content=reuse.content, reuse=reuse)
    note = ""
    if drift is not None:  # a reuse the scene drifted from
        data = {"source": drift.source, "drift": shift, "reused": False}
        hub.note(session.episode_id, EventKind.PERCEPTION_REUSE, data)
        how = "no camera in common" if shift is None else f"image difference {shift:.1f}"
        note = f"identified again: the scene differs from {drift.source}'s ({how})"
    towards, orders = towards_of(memory), orders_of(memory)
    carried = set() if memory is None else carried_roles(memory)
    outcome = await identify_objects(
        hub, session.episode_id, instruction, roles or None, towards, orders=orders, carried=carried
    )
    content = list(outcome.content)
    for _ in range(unresolved):
        if outcome.is_error or not (missing := unresolved_roles(outcome.structured)):
            break
        data = {"roles": missing}
        hub.note(session.episode_id, EventKind.PERCEPTION_REIDENTIFY, data)
        outcome = await identify_objects(
            hub,
            session.episode_id,
            instruction,
            roles or None,
            towards,
            orders=orders,
            carried=carried,
        )
        if not outcome.is_error:
            head = f"Identified again for {', '.join(missing)}, which the first look left open:"
            content += [TextPart(text=head), *outcome.content]
    if outcome.is_error:
        text = " ".join(part.text for part in outcome.content if isinstance(part, TextPart))
        status: Literal["infra", "config"] = (
            "infra" if outcome.error_code in INFRA_CODES else "config"
        )
        return Opening(status=status, note=f"identify: {outcome.error_code}: {text}")
    made = Reuse(
        source=source,
        key=reuse_key(roles, instruction, towards, orders),
        seed=hub.snapshot(session.episode_id),
        content=content,
    )
    return Opening(status="identified", note=note, content=content, reuse=made)


def unresolved_roles(result: Mapping[str, object] | None) -> list[str]:
    """The roles an identification's result left without an object: not found, or ambiguous
    with nothing registered (a role not in the instruction, `not_named`, is not open)."""
    rows = [] if result is None else result.get("roles") or []
    assert isinstance(rows, list)
    return [
        str(r["role"])
        for r in rows
        if r.get("obj") is None and r.get("reason") in ("not_found", "ambiguous")
    ]


def image_difference(now: np.ndarray, then: np.ndarray) -> float | None:
    """The mean absolute difference of two images (0-255) at 1/`SCALE` scale, `scene_drift`'s
    measure; None for images of different sizes."""
    a = np.asarray(now, dtype=np.float32)[::SCALE, ::SCALE, :3]
    b = np.asarray(then, dtype=np.float32)[::SCALE, ::SCALE, :3]
    return float(np.abs(a - b).mean()) if a.shape == b.shape else None


SPLIT_M: Final = 0.03
"""An outvoted answer this near another role's chosen object showed that object."""


ALIKE_SHARE: Final = 0.2
"""Two objects whose sorted extents each differ by at most this share of the larger look alike:
a split vote between look-alikes is which-is-which, one between objects of other shapes (a box
around a basket's content) is a backend's miss that the vote already outvoted."""


class Split(BaseModel):
    """A split vote between the identification's backends: `role`'s outvoted answer (`by`)
    showed the object `other` was bound to (at `at`, world): which is which is the backends'
    call, a coin on another scene."""

    model_config = ConfigDict(frozen=True)

    role: str
    other: str
    by: str
    at: tuple[float, float, float]


def split_votes(directory: Path) -> list[Split]:
    """The split votes of every identification an episode (in `directory`) made so far, from its
    event log (each `identify` result's roles: the chosen answer first, then the others, with
    the backends outvoted), between objects that look alike (`ALIKE_SHARE`)."""
    path = directory / "events.jsonl"
    if not path.is_file():
        return []
    found: dict[tuple[str, str], Split] = {}
    for event in read_events(path):
        if event.kind != EventKind.TOOL_RESULT or event.data.get("name") != IDENTIFY:
            continue
        roles = (event.data.get("result") or {}).get("roles") or []
        chosen = {
            r["role"]: np.asarray(r["candidates"][0]["center"], dtype=float)
            for r in roles
            if r.get("candidates") and r["candidates"][0].get("center") is not None
        }
        sizes = {r["role"]: r.get("size") for r in roles}
        for r in roles:
            if not r.get("dissent") and r.get("reason") != "ambiguous":
                continue
            for answer in r.get("candidates", [])[1:]:
                if answer.get("center") is None:
                    continue
                at = np.asarray(answer["center"], dtype=float)
                for other, center in chosen.items():
                    if (
                        other != r["role"]
                        and float(np.linalg.norm(center - at)) <= SPLIT_M
                        and _alike(sizes.get(r["role"]), sizes.get(other))
                    ):
                        x, y, z = (round(float(v), 3) for v in at)
                        key = (r["role"], other)
                        found[key] = Split(
                            role=r["role"], other=other, by=str(answer["backend"]), at=(x, y, z)
                        )
    return list(found.values())


def _alike(a: Sequence[float] | None, b: Sequence[float] | None) -> bool:
    """Whether two measured objects look alike by their extents (`ALIKE_SHARE`); unmeasured ones
    are taken to."""
    if a is None or b is None:
        return True
    return all(
        abs(x - y) <= ALIKE_SHARE * max(x, y, 1e-9)
        for x, y in zip(sorted(a), sorted(b), strict=True)
    )


def split_refusal(
    memory: MemoryDoc, splits: list[Split], before: MemoryDoc | None = None
) -> str | None:
    """Why a Memory may not stand as it is after split votes (`split_votes`): the roles a split
    vote confused must say which is which, by an `order:` line each among one group (where the
    demonstration took them by where they lay), or, against `before` (the Memory a revision
    started from), by a rewritten description of either (a measured property: the larger, the
    taller); a role the Memory does not name (an identification under the episode's own role
    names) asks for at least one group of two ordered roles. None when they do. A bounded
    refusal (the finish's refusals): where neither remedy fits, the finish goes through."""
    if not splits:
        return None
    orders = object_orders(memory)
    roles = object_roles(memory) if memory.objects.strip() else {}
    was = object_roles(before) if before is not None and before.objects.strip() else {}
    groups: dict[str, int] = {}
    for order in orders.values():
        groups[order.group] = groups.get(order.group, 0) + 1
    open_ = []
    for split in splits:
        if split.role in roles and split.other in roles:
            a, b = orders.get(split.role), orders.get(split.other)
            reworded = any(
                name in was and was[name] != roles[name] for name in (split.role, split.other)
            )
            if (a is None or b is None or a.group != b.group) and not reworded:
                open_.append(split)
        elif not any(count >= 2 for count in groups.values()):
            open_.append(split)
    if not open_:
        return None
    shown = "; ".join(
        f"{s.role}'s outvoted answer ({s.by}) showed the object {s.other} got, at {s.at}"
        for s in open_
    )
    return (
        "the identification split between its backends on which of two look-alike objects is "
        f"which ({shown}): on another scene that is a coin. Say which is which in OBJECTS: where "
        "the demonstration took them by where they lay, an order: line each among one group "
        '("order: <rank> among <group> by <key>", key distance to the robot or the robot\'s '
        "left to right, as the demonstration's first frame has them); where a measured property "
        "sets the role (the larger, the taller), that property in their descriptions; then finish"
    )


class Identification:
    """wicl's open-time hook (`learning.opening.Opener`): the functions above."""

    def reusable(
        self, reuse: Reuse | None, memory: MemoryDoc | None, instruction: str | None
    ) -> bool:
        return reusable(reuse, memory, instruction)

    async def drift(self, reuse: Reuse, first: Observation, blobs: BlobStore) -> Drift:
        return await scene_drift(reuse, first, blobs)

    async def open(
        self,
        hub: Hub,
        session: EpisodeSession,
        *,
        instruction: str | None,
        memory: MemoryDoc | None,
        source: str,
        reuse: Reuse | None = None,
        drift: Drift | None = None,
        retries: Retries | None = None,
    ) -> Opening:
        return await open_identification(
            hub,
            session,
            instruction=instruction,
            memory=memory,
            source=source,
            reuse=reuse,
            drift=drift,
            retries=retries,
        )

    async def identify(
        self,
        hub: Hub,
        session: EpisodeSession,
        *,
        instruction: str | None,
        memory: MemoryDoc,
        retries: Retries | None = None,
    ) -> Opening:
        return await identify_tried(
            hub, session, instruction, memory, session.episode_id, retries=retries
        )
