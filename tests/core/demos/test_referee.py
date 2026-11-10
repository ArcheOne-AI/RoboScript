"""roboscript.demos.referee: executed paths against a demo's keypoints, and the S1 pass rule."""

from collections.abc import Callable
from typing import Any

import numpy as np

from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import TaskName
from roboscript.demos import (
    ArmPath,
    ArmProgress,
    Demo,
    Keypoint,
    KeypointConfig,
    KeypointSet,
    RefereeConfig,
    Tolerance,
    demo_keypoints,
)
from roboscript.demos import arm_progress as _arm_progress
from roboscript.demos import held_placements as _held_placements
from roboscript.demos import judge as _judge
from roboscript.demos.referee import (
    KeypointProgress,
    Touches,
    Verdict,
)
from roboscript.spatial.rotation import (
    quat_from_axis_angle,
)

CONFIG = KeypointConfig()
DOWN = quat_from_axis_angle([0.0, 1.0, 0.0], np.pi / 2)
X = (1.0, 0.0, 0.0)  # the synthetic gripper points along its x axis
# The mechanism tests use tight tolerances (2 cm xy, 1 cm z), and their centimeters are those
# tolerances'. The defaults have their own tests.
STRICT = RefereeConfig(xy_m=0.02, z_m=0.01)


def arm_progress(*args: Any, config: RefereeConfig = STRICT, **kwargs: Any) -> ArmProgress:
    return _arm_progress(*args, config=config, **kwargs)


def judge(
    keypoints: KeypointSet,
    progress: dict[Any, ArmProgress],
    success: bool,
    sources: Any = None,
    config: RefereeConfig = STRICT,
    objects: Any = None,
) -> Verdict:
    return _judge(keypoints, progress, success, sources, config, objects)


def held_placements(*args: Any, config: RefereeConfig = STRICT) -> ArmProgress:
    return _held_placements(*args, config=config)


def keyset(demo: Demo) -> KeypointSet:
    return demo_keypoints(demo, CONFIG, CONFIG.tolerance)


def progress(kps: list[Keypoint], motion: Any) -> ArmProgress:
    track = motion.track()
    return arm_progress(kps, ArmPath(track.tcp, track.gripper), CONFIG.grip_threshold, X)


def statuses(p: ArmProgress) -> list[str]:
    return [k.status for k in p.keypoints]


def test_the_demo_itself_passes(
    pick_place: Callable[..., Any], make_demo: Callable[..., Demo]
) -> None:
    m = pick_place()
    kset = keyset(make_demo({"left": m}, success_frame=len(m.frames(1)) - 1))
    p = progress(kset.arms["left"], m)
    assert statuses(p) == ["passed"] * 6
    assert (p.target, p.retries, p.extra_events) == (None, 0, 0)
    verdict = judge(kset, {"left": p}, success=True)
    assert verdict.passed
    assert verdict.reasons == ["success latched with the first 3 of 3 keypoints passed in order"]
    # The hovers before the grasp and the release are standoffs and the move away after the
    # release a retreat, all outside the sequence; the lift is judged by order.
    assert [k.kind for k in p.keypoints] == [
        "standoff", "grip", "path", "standoff", "grip", "retreat",
    ]  # fmt: skip
    assert verdict.informational == [
        "left K1 (approach) passed (not required)",
        "left K4 (place) passed (not required)",
        "left K6 (retreat) passed (not required)",
        "left K3 (lift) passed (judged by order)",
    ]


def test_keypoints_pass_only_in_order(
    pick_place: Callable[..., Any], make_demo: Callable[..., Demo]
) -> None:
    # Lifted before closing: the lift, which is no standoff, cannot pass before the grasp, and
    # closing up there misses the grasp.
    kps = keyset(make_demo({"left": pick_place()})).arms["left"]
    p = progress(kps, pick_place(grasp_dz=0.02))
    assert statuses(p)[:2] == ["passed", "missed"]


def test_the_attempts_latch_ends_the_sequence_wherever_the_demos_did(
    pick_place: Callable[..., Any], make_demo: Callable[..., Demo]
) -> None:
    """Once the official check latches, the keypoints the demo did from its success frame on do not
    count; those before it are owed."""
    m = pick_place()
    kps = keyset(make_demo({"left": m})).arms["left"]
    track = m.track()
    cut = kps[3].frame + 10  # holding the object over the target, the success latches
    p = arm_progress(kps, ArmPath(track.tcp[:cut], track.gripper[:cut]), 0.5, X)
    assert statuses(p) == ["passed"] * 4 + ["pending"] * 2
    assert p.target == 5

    def verdict(success_frame: int, success: bool = True) -> Any:
        demo = make_demo({"left": m}, success_frame=success_frame)
        return judge(keyset(demo), {"left": p}, success)

    for frame in (kps[3].frame + 1, kps[4].frame - 1, kps[4].frame):
        on = verdict(frame)
        assert on.passed
        assert any(
            line.startswith("left K5 (release) pending: ")
            and line.endswith("(after the success latch)")
            for line in on.informational
        )
    owed = verdict(kps[5].frame)
    assert not owed.passed and owed.reasons[0].startswith("left K5 (release) pending: ")
    failed = verdict(kps[3].frame + 1, success=False)
    assert failed.reasons[0] == "the task's success never latched"
    assert failed.reasons[1].startswith("left K5 (release) pending: ")  # what it still owed


def _two_arms(frames: dict[str, list[tuple[int, ...]]], success_frame: int) -> KeypointSet:
    """Two arms' grasps and releases at the given demo frames: (frame, cue) per keypoint, or
    (since, frame, cue) with the frame its arm began moving toward it."""
    arms = {
        arm: [
            Keypoint(
                index=k,
                since=since[0] if since else None,
                frame=frame,
                tcp=Pose(p=(0.0, 0.0, 0.8), q=tuple(DOWN)),  # type: ignore[arg-type]
                gripper="closed" if cue == "close" else "open",
                cue=cue,  # type: ignore[arg-type]
                phase="grasp" if cue == "close" else "release",
                tolerance=Tolerance(),
            )
            for k, (*since, frame, cue) in enumerate(kps, 1)
        ]
        for arm, kps in frames.items()
    }
    return KeypointSet(
        task=TaskName("synthetic"), seed=0, frame_steps=1, success_frame=success_frame,
        config_hash="0", pointing=dict.fromkeys(arms, X), arms=arms,  # type: ignore[arg-type]
    )  # fmt: skip


def _progress(*keypoints: tuple[str, int | None]) -> ArmProgress:
    """An arm's progress: (status, path sample) per keypoint."""
    return ArmProgress(
        keypoints=[
            KeypointProgress(index=k, status=status, kind="grip", at=at)  # type: ignore[arg-type]
            for k, (status, at) in enumerate(keypoints, 1)
        ],
        target=None,
    )


# The demo: the left arm grasps (frame 10), then the right (20), the left releases (30), the
# success latches (40), and the right releases (50).
HANDOFF = {"left": [(10, "close"), (30, "open")], "right": [(20, "close"), (50, "open")]}


def test_both_arms_keypoints_form_one_time_ordered_sequence() -> None:
    """Both arms' keypoints form one time-ordered sequence. Each arm's path has its own samples;
    `sources` puts them on the attempt's recorded clock."""
    kset = _two_arms(HANDOFF, success_frame=40)
    left = _progress(("passed", 5), ("passed", 30))
    right = _progress(("passed", 20), ("passed", 40))
    sources = {"left": np.arange(100), "right": np.arange(100)}
    ordered = judge(kset, {"left": left, "right": right}, success=True, sources=sources)
    assert ordered.passed
    assert ordered.reasons == ["success latched with the first 4 of 4 keypoints passed in order"]
    # The right arm grasped before the left: each arm is in order, the sequence is not.
    early = _progress(("passed", 2), ("passed", 40))
    swapped = judge(kset, {"left": left, "right": early}, success=True, sources=sources)
    assert not swapped.passed
    assert swapped.reasons == ["right K1 (grasp) passed before left K1, which comes first"]
    # Only the clock tells: the right arm's path is sampled four times as densely, so its
    # samples 20 and 160 are recorded samples 5 (with the left's grasp) and 40; 16 is 4, before.
    dense = {"left": np.arange(100), "right": np.arange(400) // 4}
    right = _progress(("passed", 20), ("passed", 160))
    assert judge(kset, {"left": left, "right": right}, success=True, sources=dense).passed
    soon = _progress(("passed", 16), ("passed", 160))
    assert not judge(kset, {"left": left, "right": soon}, success=True, sources=dense).passed
    assert judge(kset, {"left": left, "right": soon}, success=True).passed  # on its own samples
    # Keypoints on the same demo frame are one moment: either arm may come first.
    together = _two_arms({"left": [(10, "close")], "right": [(10, "close")]}, success_frame=10)
    first, second = _progress(("passed", 9)), _progress(("passed", 3))
    assert judge(together, {"left": first, "right": second}, success=True).passed


def test_a_skipped_keypoint_followed_by_a_reached_one_fails() -> None:
    """An attempt passes, in order, a prefix: a keypoint not passed ends the prefix, so one after
    it in the sequence, on either arm, may not have passed."""
    kset = _two_arms(HANDOFF, success_frame=40)
    left = _progress(("pending", None), ("pending", None))  # never grasped
    right = _progress(("passed", 20), ("pending", None))
    verdict = judge(kset, {"left": left, "right": right}, success=True)
    assert not verdict.passed
    assert verdict.reasons == ["left K1 (grasp) pending: not done by the official success latch"]


def test_missed_before_the_latch_fails_pending_after_the_demos_success_does_not_count() -> None:
    """Keypoints after the demo's success frame do not count once the official check latches;
    every one before it is owed. Missed means the attempt skipped it or went past it before the
    latch (the episode ends there), which breaks the prefix whatever follows."""
    kset = _two_arms(HANDOFF, success_frame=40)
    left = _progress(("passed", 5), ("passed", 30))
    rest = judge(kset, {"left": left, "right": _progress(("passed", 20), ("pending", None))}, True)
    assert rest.passed
    assert rest.informational == ["right K2 (release) pending (after the success latch)"]
    # The latch came before the demo's: the keypoints the demo did before its success are owed.
    grasped = _progress(("passed", 5), ("pending", None))
    none = _progress(("pending", None), ("pending", None))
    early = judge(kset, {"left": grasped, "right": none}, True)
    assert early.reasons == ["right K1 (grasp) pending: not done by the official success latch"]
    # Not a prefix: the left released before the right grasped, as the demo did not.
    assert not judge(kset, {"left": left, "right": none}, True).passed
    assert not judge(kset, {"left": left, "right": _progress(("passed", 20), ("pending", None))},
                     False).passed  # fmt: skip
    missed = judge(kset, {"left": left, "right": _progress(("passed", 20), ("missed", 45))}, True)
    assert not missed.passed and missed.reasons == ["right K2 (release) missed"]


def test_the_arms_order_binds_only_keypoints_on_the_same_object() -> None:
    """The two arms' order is required only where both act on the
    same object (a hand-over, a stack); on different objects either arm may go first; each arm's
    own keypoints stay in order. The demo grasps with the left (frame 10), then the right (20)."""
    kset = _two_arms({"left": [(10, "close")], "right": [(20, "close")]}, success_frame=20)
    left, right = _progress(("passed", 30)), _progress(("passed", 10))  # the right first
    cube, ball = Touches(frozenset({"cube"})), Touches(frozenset({"ball"}))
    config = RefereeConfig()
    apart = {"left": [cube], "right": [ball]}
    assert judge(kset, {"left": left, "right": right}, True, None, config, apart).passed
    same = {"left": [cube], "right": [cube]}
    verdict = judge(kset, {"left": left, "right": right}, True, None, config, same)
    assert verdict.reasons == ["right K1 (grasp) passed before left K1, which comes first"]
    onto = {"left": [cube], "right": [Touches(frozenset({"ball"}), frozenset({"cube"}))]}
    assert not judge(kset, {"left": left, "right": right}, True, None, config, onto).passed
    tray = Touches(frozenset({"cube"}), frozenset({"tray"}))
    both = {"left": [tray], "right": [Touches(frozenset({"ball"}), frozenset({"tray"}))]}
    assert judge(kset, {"left": left, "right": right}, True, None, config, both).passed
    # Without the objects every pair is ordered.
    assert not judge(kset, {"left": left, "right": right}, True, None, config).passed


UP = (1.0, 0.0, 0.0, 0.0)


REF = RefereeConfig()
