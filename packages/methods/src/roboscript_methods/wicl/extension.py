"""wicl's own settings of an episode (`Episode.method`, which the core carries to the method's
hooks without reading) and the hooks that read them: the privileged finish guard
(`DoneGate.guard`), the reading of a judged try's end (`read_ends`, `MethodRuntime.judged`) and
the demonstration's end read on frames alone (`demo_relations`, `demo_end_targets`).

A judged try with `WiclEpisode.relations` reads every done metric among the registered roles by
perception alone into `relations.json`, with the env's outcome (what S1's done criteria are
written from and S2's Reviser corrects them by); where the frames seed the demonstration
(`DEMO`), also at its final state into `relations_demo.json` (what the criteria aim at) and as
targets into `end_targets.json`.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from roboscript.agent.history import text_of
from roboscript.contracts.env import EnvControl, GtSnapshot
from roboscript.contracts.errors import InfraError, RSError
from roboscript.contracts.hub import EpisodeSeed, OpenEpisode
from roboscript.contracts.ids import EpisodeId, PhaseName, ToolName
from roboscript.hub import Hub
from roboscript.learning.demo import DemoTruth
from roboscript.learning.opening import IDENTIFY
from roboscript.records import write_atomic
from roboscript.spatial.rotation import quat_angle_deg
from roboscript.tools import DEMO
from roboscript.toolsets.perception import identify_objects

if TYPE_CHECKING:
    from roboscript.learning.episode import Episode, EpisodeDriver

log = logging.getLogger(__name__)

RELATIONS: Final = "relations.json"
DEMO_RELATIONS: Final = "relations_demo.json"
"""The same metrics at the demonstration's final state, where the episode has it (`DEMO`)."""
END_TARGETS: Final = "end_targets.json"
"""The demonstration's end as targets (`perception.endstate`), read beside
`relations_demo.json`."""
UNCONFIRMED: Final = "finish_unconfirmed"
"""How a try ends whose accepted `finish` the env's success check did not confirm (S2 under done
criteria, past its latch refusals, `PracticeConfig.latch_finish`; S1 past its refusals): a
failed try (fail-fast), whatever the done criteria said; the Reviser corrects them from it."""
READ_RELATIONS: Final = ToolName("read_relations")
READ_END_TARGETS: Final = ToolName("read_end_targets")


@dataclass(frozen=True)
class WiclEpisode:
    """`official_finish` (S1; S2 under `PracticeConfig.latch_finish`) refuses a `finish` while the
    env's success latch has not fired, up to the episode's `finish_refusals` times, with
    `finish_hint` (which way to keep moving: the demonstration's motion into its latch) and, with
    `finish_target` (S1 only, privileged), how the scene's ground truth differs from its success
    frame (the demonstration's end, same seed); without one only the latched flag is read, which
    an Agent has in P3 too (the benchmark's loop ends an episode at its latch). `relations` (S1
    Actors, and S2 tries under done criteria) reads the try's end (`read_ends`). `demo_end`: the
    demonstration's ground-truth layer, for the diagnosis alone (`Episode.diagnose`), which
    `diagnosis.explain_try` reads (each object's tilt against the tilt the demonstration ended
    its own with, so an object the task stands up is not read as fallen over)."""

    official_finish: bool = False
    finish_target: DemoTruth | None = None
    finish_hint: str = ""
    relations: bool = False
    demo_end: DemoTruth | None = None


NONE: Final = WiclEpisode()


def settings(method: object | None) -> WiclEpisode:
    """wicl's settings from an episode's `Episode.method` (the defaults: none given)."""
    return method if isinstance(method, WiclEpisode) else NONE


def finish_guard(episode: "Episode", control: EnvControl, env_episode: EpisodeId) -> Any:
    """The privileged finish guard of an `official_finish` episode (None: none)."""
    own = settings(episode.method)
    if not own.official_finish:
        return None
    target, hint = own.finish_target, own.finish_hint
    return lambda _: not_done(control, env_episode, target, hint)


async def not_done(
    control: EnvControl, env_episode: EpisodeId, target: DemoTruth | None = None, hint: str = ""
) -> str | None:
    """S1's privileged finish check: a refusal while the env's success check is false, with
    `hint` (the way the demonstration moved into its latch: `goal.latch_hint`) and how the ground
    truth now differs from the demonstration's end (`end_gap`)."""
    try:
        judged = await control.judge(env_episode)
    except InfraError as error:
        log.warning("the env could not judge %s: %s", env_episode, error)
        judged = None
    if judged is None or judged.success:
        return None
    text = (
        "the task is not complete yet (the official success check is false): look at the scene "
        "again and go on"
    ) + hint
    if target is None:
        return text
    try:
        snapshot = await control.gt_snapshot(env_episode)
    except RSError as error:
        log.warning("no ground truth for %s: %s", env_episode, error)
        return text
    gaps = end_gap(target, snapshot)
    return text + (". Against the demonstration's end: " + "; ".join(gaps) if gaps else "")


def end_gap(target: DemoTruth, snapshot: GtSnapshot) -> list[str]:
    """How `snapshot` differs from `target`'s success frame (its last frame without one), object
    by object in its category's words: the position offset (world, cm), the turn (degrees) and
    articulated joints' positions; an object the snapshot lacks is left out."""
    frames = [frame for frame in target.frames if "success" in frame.what] or target.frames[-1:]
    if not frames:
        return []
    end, gaps = frames[-1], []
    for name, pose in end.poses.items():
        now = snapshot.objects.get(name)
        if now is None:
            continue
        word = target.objects.get(name) or "object"
        dx, dy, dz = (100 * (a - b) for a, b in zip(now.p, pose.p, strict=True))
        turn = float(quat_angle_deg(np.asarray(now.q), np.asarray(pose.q)))
        line = (
            f"the {word} is {np.linalg.norm([dx, dy, dz]):.1f} cm off "
            f"(x {dx:+.1f}, y {dy:+.1f}, z {dz:+.1f}) and turned {turn:.0f} deg"
        )
        moving = [link.joint.position for link in snapshot.links.get(name, {}).values()]
        wanted = end.joints.get(name)
        if wanted is not None and len(wanted) == len(moving) and moving:
            line += ", joints " + ", ".join(
                f"{a:.2f} (end {b:.2f})" for a, b in zip(moving, wanted, strict=True)
            )
        gaps.append(line)
    return gaps


async def read_ends(hub: Hub, episode: "Episode", episode_id: EpisodeId, success: bool) -> None:
    """A judged try's end (`MethodRuntime.judged`) under `WiclEpisode.relations`: `relations.json`,
    the scene the try ended in as perception reads it, and whether the env's success check
    passed; with the demonstration in its frames, `relations_demo.json` and `end_targets.json`;
    nothing past a reading that failed (it does not change the try)."""
    if not settings(episode.method).relations:
        return
    read = await hub.system_call(episode_id, READ_RELATIONS, {})
    if read.is_error or read.structured is None:
        log.warning(
            "%s: the relations could not be read: %s", episode.directory, text_of(read.content)
        )
        return
    data = json.dumps({"success": success, **read.structured}, indent=2) + "\n"
    write_atomic(episode.directory / RELATIONS, data.encode())
    if episode.frames is None or DEMO not in episode.frames.facts:
        return
    read = await hub.system_call(episode_id, READ_RELATIONS, {"at": "demo_end"})
    if read.is_error or read.structured is None:
        log.warning(
            "%s: the demonstration's final relations could not be read: %s",
            episode.directory,
            text_of(read.content),
        )
        return
    data = json.dumps(read.structured, indent=2) + "\n"
    write_atomic(episode.directory / DEMO_RELATIONS, data.encode())
    if READ_END_TARGETS not in hub.mounted:
        return
    read = await hub.system_call(episode_id, READ_END_TARGETS, {})
    if read.is_error or read.structured is None:
        log.warning(
            "%s: the demonstration's end targets could not be read: %s",
            episode.directory,
            text_of(read.content),
        )
        return
    data = json.dumps(read.structured, indent=2) + "\n"
    write_atomic(episode.directory / END_TARGETS, data.encode())


async def demo_relations(
    driver: "EpisodeDriver", seed: EpisodeSeed, *, phase: PhaseName
) -> dict[str, Any] | None:
    """`read_relations` at the demonstration's final state on frames alone, no env and no
    brain: an Actor episode of `phase` seeded with `seed` (an earlier try's frames and
    registry, the demonstration's frames and `DEMO` fact), for an S1 whose attempts never read
    it; None where the reading tool is not mounted
    (nothing can read it here). A reading that fails raises (`RuntimeError`): the failure may be
    transient, so the caller must not record it as "nothing to read"."""
    hub = driver.hub
    if READ_RELATIONS not in hub.mounted:
        return None
    spec = OpenEpisode(stage=phase, role="actor", embodiment=driver.embodiment, seed=seed)
    session = await hub.open_episode(spec)
    episode_id = session.episode_id
    try:
        read = await hub.system_call(episode_id, READ_RELATIONS, {"at": "demo_end"})
    finally:
        await hub.close_episode(episode_id)
    if read.is_error or read.structured is None:
        why = text_of(read.content)
        log.warning("the demonstration's final relations could not be read: %s", why)
        raise RuntimeError(f"the demonstration's final relations could not be read: {why}")
    return dict(read.structured)


async def demo_end_targets(
    driver: "EpisodeDriver",
    seed: EpisodeSeed,
    instruction: str,
    roles: Mapping[str, str] | None,
    orders: Mapping[str, str],
    carried: set[str],
    frame: str,
    *,
    phase: PhaseName,
) -> dict[str, Any] | None:
    """The demonstration's end as targets (`read_end_targets`) on its recording alone, no env
    and no brain: an Actor episode of `phase` seeded with `seed` (the package's frames and its
    `DEMO` fact) identifies the Memory's `roles` (with their `orders`; `carried`: the roles
    its workflow grasps or places) in the recording's first frame `frame` with the
    demonstration's `instruction` (perception only, as any episode's opening does; without
    the roles' `x toward` words, whose two VLM questions a role decide nothing the targets
    use: S2 and P3 drop yaw targets, `tools.done.end_for`), then reads the targets at its
    last keyframe. None where the tools are not mounted; a reading
    that fails raises (`RuntimeError`: it may be transient, never "nothing to read")."""
    hub = driver.hub
    if READ_END_TARGETS not in hub.mounted or IDENTIFY not in hub.mounted:
        return None
    spec = OpenEpisode(stage=phase, role="actor", embodiment=driver.embodiment, seed=seed)
    session = await hub.open_episode(spec)
    episode_id = session.episode_id
    try:
        found = await identify_objects(
            hub,
            episode_id,
            instruction,
            roles,
            orders=orders,
            carried=carried,
            frame=frame,
        )
        if found.is_error:
            why = text_of(found.content)
            raise RuntimeError(f"the demonstration's objects could not be identified: {why}")
        read = await hub.system_call(episode_id, READ_END_TARGETS, {})
    finally:
        await hub.close_episode(episode_id)
    if read.is_error or read.structured is None:
        why = text_of(read.content)
        raise RuntimeError(f"the demonstration's end targets could not be read: {why}")
    return dict(read.structured)
