"""Fitting a written skill on the demonstration as perceived (`toolsets.skill_fit`): a fitter's
answer, its params or why none, and a fit holding an absolute position is refused."""

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.demos import ArmTrack, Demo, DemoMeta
from roboscript.demos.demo import Provenance
from roboscript.perception.registry import SceneRegistry
from roboscript.testing.scenes import table_model
from roboscript.tools.skill_fit import RELATIVE_RULE, FitSegment
from roboscript_methods.wicl.relation import replay_grips
from roboscript_methods.wicl.toolsets.skill_fit import (
    perceived,
    run_fit,
    segment_of,
)
from tests.methods.wicl.test_relation import BASE, POSES, PROFILE, pick_and_place, registry

FRAMES = 70


def recorded(keypoints_tcp: dict[int, np.ndarray]) -> Demo:
    """The left arm's TCP every frame, held at each keypoint's from that frame on."""
    tcp = np.zeros((FRAMES, 7))
    marks = sorted(keypoints_tcp)
    for frame in range(FRAMES):
        before = [m for m in marks if m <= frame]
        tcp[frame] = Pose.from_matrix(keypoints_tcp[before[-1] if before else marks[0]]).as_list()
    track = ArmTrack(ee=tcp.copy(), tcp=tcp, gripper=np.ones(FRAMES))
    meta = DemoMeta(
        task="lift_red",
        seed=0,
        instructions=["lift"],
        frames=FRAMES,
        frame_steps=1,
        ee_to_tcp={"left": (0.0, 0.0, 0.0)},
        provenance=Provenance(source="test", path="demo", sha256="0" * 64),
    )
    return Demo(meta=meta, arms={"left": track})


def snapshots(scene: SceneRegistry, keypoints, states) -> list[tuple[int, SceneRegistry]]:  # type: ignore[no-untyped-def]
    events = sorted({kp.frame for kps in keypoints.arms.values() for kp in kps if kp.grip})
    found = []
    for first in [0, *(frame + 1 for frame in events)]:
        at = scene.clone()
        replay_grips(keypoints, at, states, PROFILE, table_model(), before=first)
        found.append((first, at))
    return found


def test_a_fitter_answers_with_params_or_why_not() -> None:
    scene = registry()
    keypoints, states, _ = pick_and_place(scene)
    demo = recorded({kp.frame: kp.tcp.matrix() for kp in keypoints.arms["left"]})
    recording, start = perceived(keypoints, scene, snapshots(scene, keypoints, states), demo)
    segment = segment_of(keypoints, demo, "move", "left K3", "left K2", None, None, None, False)

    def rise(seg: FitSegment, rec, sc, base, conv):  # type: ignore[no-untyped-def]
        track = rec.demo.arms[seg.arm].tcp
        return {"dz": np.float64(track[seg.end, 2] - track[seg.start, 2])}

    def truth(seg: FitSegment, rec, sc, base, conv):  # type: ignore[no-untyped-def]
        return rec.commands  # the recording's commands: not what perception gives

    fit = run_fit("rise", rise, segment, recording, start, BASE, POSES)
    assert fit.params == {"dz": pytest.approx(0.1)}
    assert run_fit("none", lambda *a: None, segment, recording, start, BASE, POSES).why == (
        "it does not apply to this segment"
    )
    refused = run_fit("truth", truth, segment, recording, start, BASE, POSES)
    assert refused.params is None and "AttributeError" in (refused.why or "")


def test_a_fit_holding_an_absolute_position_is_refused() -> None:
    scene = registry()
    keypoints, states, _ = pick_and_place(scene)
    demo = recorded({kp.frame: kp.tcp.matrix() for kp in keypoints.arms["left"]})
    recording, start = perceived(keypoints, scene, snapshots(scene, keypoints, states), demo)
    tool, goal = scene.named("tool"), scene.named("goal")
    assert tool is not None and goal is not None
    segment = segment_of(keypoints, demo, "place", "left K5", None, None, tool.id, goal.id, False)

    def where(seg: FitSegment, rec, sc, base, conv):  # type: ignore[no-untyped-def]
        return {"goal": list(seg.tcp.p)}  # the world's frame: it moves with the demonstration

    def offset(seg: FitSegment, rec, sc, base, conv):  # type: ignore[no-untyped-def]
        goal = sc.at(rec.poses(seg.start), {}).proxy(seg.target).pose[:3, 3]
        return {"above": (np.asarray(seg.tcp.p) - goal).tolist()}  # relative to the target

    refused = run_fit("where", where, segment, recording, start, BASE, POSES)
    assert refused.params is None and RELATIVE_RULE in (refused.why or "")
    assert "goal[0]" in (refused.why or "") and "goal[2]" not in (refused.why or "")
    assert run_fit("offset", offset, segment, recording, start, BASE, POSES).params is not None
