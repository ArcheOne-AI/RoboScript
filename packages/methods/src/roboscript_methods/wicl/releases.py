"""Where the objects the demonstration let go of are, measured again on its frames
(`demo_releases`): the replays of the demonstration's grips (`relation.replay_grips`'
`released`) start a regrasp from where the object was seen. Used by the demo toolset's
`demo_relation` and the done toolset's demonstration end, on the scene toolsets' `SceneAccess`."""

import functools
from collections.abc import Mapping

import anyio
import numpy as np

from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import RobotState
from roboscript.demos import KeypointSet, frame_alias
from roboscript.embodiments import EmbodimentProfile
from roboscript.perception.measure import measure
from roboscript.perception.registry import ObjectRecord, SceneRegistry
from roboscript.perception.table import Table
from roboscript.spatial.pose import pose_from_matrix, pose_to_matrix
from roboscript.tools import DEMO_RELEASES, ToolContext
from roboscript.toolsets.scene import SceneAccess
from roboscript_methods.wicl.relation import (
    Released,
    release_frames,
    release_key,
    released_poses,
    released_tried,
    replay_grips,
)


async def demo_releases(
    scene: SceneAccess,
    ctx: ToolContext,
    keypoints: KeypointSet,
    registry: SceneRegistry,
    robots: Mapping[int, RobotState],
    profile: EmbodimentProfile,
    table: Table | None,
) -> Released:
    """Where each object the demonstration let go of is, measured again
    (`perception.measure`, stored as `remeasure` stores) on the demonstration's frame after
    the open (`demos.relation.release_frames`), its grips replayed on `registry` (not
    changed) up to there: what a live run's `place_object` does after it opens, so the
    replays (`replay_grips`' `released`) start a regrasp from where the object was seen, not
    from where its grasp would have left it. Kept in the `DEMO_RELEASES` fact per object
    (`released_poses`): only releases of objects registered or replaced since are measured
    again; an object not measured there stays where the replay puts it."""
    fact = ctx.frames.fact(DEMO_RELEASES)
    known = released_poses(fact, registry) or {}
    tried = released_tried(fact, registry)
    released: dict[int, dict[str, Pose]] = {t: dict(poses) for t, poses in known.items()}
    trust = scene.config.pipeline.measure.trust
    measured = False
    for arm, kp, at in release_frames(keypoints):
        if at not in robots or kp.frame not in robots:
            continue
        copy = registry.clone()
        run = functools.partial(
            replay_grips, keypoints, copy, robots, profile, table, before=kp.frame + 1,
            released=released,
        )  # fmt: skip
        await anyio.to_thread.run_sync(run)
        before = registry.clone()
        run = functools.partial(
            replay_grips, keypoints, before, robots, profile, table, before=kp.frame,
            released=released,
        )  # fmt: skip
        await anyio.to_thread.run_sync(run)
        held = [o.id for o in before.objects.values() if arm in o.grasps]
        free = [o for o in held if o in copy.objects and not copy.objects[o].grasps]
        free = [o for o in free if (kp.frame, o) not in tried]
        if not free:
            continue
        try:
            seen = await scene.frame(ctx, frame_alias(at))
        except ToolError:
            continue
        geometry = await scene.geometry(ctx, seen)
        for obj in free:
            tried.add((kp.frame, obj))
            measured = True
            run = functools.partial(
                measure, copy, geometry, obj, robot=robots[at], pipeline=scene.config.pipeline
            )
            found = await anyio.to_thread.run_sync(run)
            usable = found.reason == "ok" or (
                found.reason == "relocated" and found.confidence >= trust
            )
            scene.log(ctx)(
                "demo_release", obj=obj, open=kp.frame, frame=at, reason=found.reason,
                shift_m=round(found.shift_m, 4), stored=usable,
            )  # fmt: skip
            if usable:
                old = copy.objects[obj]
                released.setdefault(kp.frame, {})[obj] = same_frame(found.record, old)
    if not measured:
        return released
    ctx.frames.set_fact(
        DEMO_RELEASES,
        {
            "objects": release_key(registry),
            "tried": [[t, o] for t, o in sorted(tried)],
            "poses": {
                str(t): {o: p.as_list() for o, p in poses.items()} for t, poses in released.items()
            },
        },
    )
    return released


def same_frame(measured: ObjectRecord, old: ObjectRecord) -> Pose:
    """`measured`'s placement as the pose of `old`'s record (its own fit's frame): the world
    transform that takes `measured`'s fit onto where it was measured, applied to `old`'s fit."""
    moved = pose_to_matrix(measured.pose.as_list()) @ np.linalg.inv(measured.fit.frame.matrix())
    return Pose.from_list(pose_from_matrix(moved @ old.fit.frame.matrix()).tolist())
