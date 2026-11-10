"""S1 -> S2 -> evaluate end to end on MockEnv (CPU) with scripted brains and real success
latches: a demonstration recorded with the oracle lift (0.1 m), a Demo Reader whose Memory v0
lifts too little (0.02 m), the keypoint referee, hinted attempts with keypoint progress, an S1
revision, unhinted passes to v_S1, S2 converging from it, and the frozen Memory evaluated."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from roboscript.agent import read_lines
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import EpisodeSpec, ObserveRequest
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import BatchId, new_id
from roboscript.contracts.observation import FrameEntry
from roboscript.demos import BLOBS_DIR, Keypoint, KeypointSet, Tolerance
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import MockEnv
from roboscript.learning import (
    DemoPackage,
    DemoTruth,
    Evaluate,
    EvaluateResult,
    Keyframe,
    save_package,
)
from roboscript.learning.demo import TRUTH_FILE, TruthFrame
from roboscript.testing.protocols import (
    LIFT_RED,
    editing,
    finishing,
    lifting_actor,
)
from roboscript_methods.wicl.imitate import (
    Imitate,
    ImitateConfig,
    ImitateResult,
    exits,
)
from roboscript_methods.wicl.practice import Practice, PracticeResult

DEMO_SEED = 7
READER_OPS = [
    {
        "op": "set_text",
        "section": "objects",
        "text": "- box: the box the instruction names.",
        "why": "demo",
    },
    {
        "op": "define_param",
        "name": "lift_height",
        "param": {
            "kind": "number",
            "value": 0.02,
            "unit": "m",
            "bounds": [0.01, 0.2],
            "trust": 0.05,
        },
        "why": "the expert lifted the box",
    },
    {
        "op": "add_stage",
        "stage": {
            "name": "lift",
            "goal": "the box held above the table",
            "arm": "the arm on the box's side",
            "call": "lift(color, height_m={lift_height})",
            "done": "lifted(box, {lift_height})",
        },
        "why": "the demonstration's one stage",
    },
]
FIX = [{"op": "set_number", "param": "lift_height", "value": 0.1, "why": "success never latched"}]


async def record_demo(env: MockEnv, blobs: LocalBlobStore, directory: Path) -> DemoPackage:
    """The oracle lift by 0.1 m at the demo seed, observed before and after every step."""
    opened = await env.open(EpisodeSpec(task=LIFT_RED, seed=DEMO_SEED))
    episode = opened.episode_id
    x, y, z = (await env.gt_snapshot(episode)).objects["red"].p
    arm = "left" if x < 0 else "right"
    spec = load_embodiment(MOCK_BIMANUAL).arms[arm]
    qpos = list((await env.robot(episode)).qpos_cmd)

    def at(ee_z: float, grip: float) -> list[float]:
        q = list(qpos)
        q[spec.action[0] : spec.action[0] + 6] = [x, y, ee_z, 0.0, math.pi / 2, 0.0]
        q[spec.gripper_index] = grip
        return q

    steps = [
        ActionStep(qpos=at(z + 0.2, 1.0)),
        ActionStep(qpos=at(z + 0.1, 1.0)),
        ActionStep(qpos=at(z + 0.1, 0.0), until_gripper_stable=True),
        ActionStep(qpos=at(z + 0.2, 0.0)),
    ]
    observations = [await env.observe(episode, ObserveRequest())]
    for step in steps:
        await env.execute(episode, ActionBatch(batch_id=BatchId(new_id("b")), steps=[step]))
        observations.append(await env.observe(episode, ObserveRequest()))
    judged = await env.judge(episode)
    await env.close(episode)
    assert judged.success
    frames = []
    for n, observation in enumerate(observations, 1):
        blob = await blobs.aput_bytes(observation.model_dump_json().encode(), "application/json")
        frames.append(
            FrameEntry(
                frame_id=observation.frame_id,
                alias=f"frame_{n}",
                step=observation.step,
                observation=blob,
            )
        )

    def keypoint(index: int, frame: int, cue: Any, phase: Any, gripper: Any) -> Keypoint:
        tcp = observations[frame].robot.arms[arm].tcp
        return Keypoint(
            index=index,
            frame=frame,
            tcp=tcp,
            gripper=gripper,
            cue=cue,
            phase=phase,
            tolerance=Tolerance(),
        )

    keypoints = KeypointSet(
        task=LIFT_RED,
        seed=DEMO_SEED,
        frame_steps=1,
        success_frame=4,  # the lift latched success: its keypoint is not required
        config_hash="synthetic",
        pointing={"left": (1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)},
        arms={
            arm: [
                keypoint(1, 1, "stop", "approach", "open"),
                keypoint(2, 3, "close", "grasp", "closed"),
                keypoint(3, 4, "stop", "lift", "closed"),
            ]
        },
    )
    keyframes = []
    for kp in keypoints.arms[arm]:
        rgb = await blobs.aget_array(observations[kp.frame].cameras["overhead"].rgb)
        image = await blobs.aput_image(rgb)
        keyframes.append(Keyframe(caption=f"{arm} {kp.name} ({kp.phase})", images=[image]))
    package = DemoPackage(
        task=LIFT_RED,
        seed=DEMO_SEED,
        instruction=opened.instruction or "",
        keypoints=keypoints,
        keyframes=keyframes,
        frames=frames,
    )
    save_package(directory, package)
    carried = LocalBlobStore(directory / BLOBS_DIR)  # a package carries what it references
    refs = [image for keyframe in keyframes for image in keyframe.images]
    refs += [entry.observation for entry in frames]
    refs += [b for o in observations for c in o.cameras.values() for b in (c.rgb, c.depth) if b]
    for ref in refs:
        carried.put_bytes(await blobs.aget_bytes(ref), ref.media_type)
    red = Pose.from_list([x, y, z, 1.0, 0.0, 0.0, 0.0])
    truth = DemoTruth(
        task=LIFT_RED,
        seed=DEMO_SEED,
        objects={"red": "box"},
        frames=[TruthFrame(frame=0, what=["start"], poses={"red": red})],
    )
    (directory / TRUTH_FILE).write_text(truth.model_dump_json(), encoding="utf-8")
    return package


async def test_s1_then_s2_then_evaluate(setup: Any) -> None:
    demo = setup.run_dir.parent / "demo"
    package = await record_demo(setup.env, setup.blobs, demo)
    brains = {
        "demo_reader": editing(lambda _: READER_OPS),
        "actor": lifting_actor,
        "reviser": editing(lambda _: FIX),
    }
    async with setup.runner(brains) as runner:
        results = await runner.run_tasks(
            {
                LIFT_RED: [
                    (Imitate(), {"demo": str(demo)}),
                    (Practice(), None),
                    (Evaluate(), {"seeds": [20000, 20001, 20002]}),
                ]
            }
        )
    s1, s2, evaluated = results[LIFT_RED]  # type: ignore[misc]
    assert isinstance(s1, ImitateResult) and isinstance(s2, PracticeResult)
    assert isinstance(evaluated, EvaluateResult)
    # S1: attempt 1 (hinted) lifts 0.02 m, the latch never fires; the Reviser makes it 0.1 m (v001);
    # attempts 2-4 (unhinted, whatever the hinted one scored) pass: v_S1 = v001.
    assert (s1.passed, s1.attempts, s1.hinted, s1.unhinted, s1.version) == (True, 4, 1, 3, 1)
    root = setup.run_dir / LIFT_RED / "S1" / str(DEMO_SEED)
    first = json.loads(
        (root / "attempt1" / "referee" / "keypoints.json").read_text(encoding="utf-8")
    )
    assert first["verdict"]["passed"] is False
    # The lift keypoint is on the demonstration's success frame, so it is judged too.
    reasons = first["verdict"]["reasons"]
    assert reasons[0] == "the task's success never latched"
    assert len(reasons) == 2 and reasons[1].startswith("right K3 (lift) pending: 8.0 cm off")
    # S1 is privileged: its first finish short of the official success is refused once; S2's
    # finish never asks the simulator.
    refused = "the task is not complete yet (the official success check is false)"
    assert refused in (root / "attempt1" / "events.jsonl").read_text(encoding="utf-8")
    with np.load(root / "attempt1" / "physics.npz") as physics:  # S1 records ground truth
        assert "truth.sample" in physics and "truth.red" in physics
    s2_physics = list((setup.run_dir / LIFT_RED / "S2").rglob("physics.npz"))
    assert s2_physics and all("truth.sample" not in np.load(path) for path in s2_physics)
    s2_logs = list((setup.run_dir / LIFT_RED / "S2").rglob("events.jsonl"))
    assert s2_logs and not any(refused in log.read_text(encoding="utf-8") for log in s2_logs)
    passed = json.loads(
        (root / "attempt2" / "referee" / "keypoints.json").read_text(encoding="utf-8")
    )
    assert passed["verdict"]["passed"] is True
    [turn] = [t for t in read_lines(root / "attempt1" / "turns.jsonl") if t["action"] == "lift"]
    [lift] = [c for e in turn["exchanges"] for c in e["calls"] if c["name"] == "lift"]
    notes = lift["content"][-1]["text"]
    assert notes.startswith("Keypoint progress") and "passed K1, K2" in notes and "K3" in notes
    hinted = json.loads((root / "attempt1" / "config.json").read_text(encoding="utf-8"))
    unhinted = json.loads((root / "attempt2" / "config.json").read_text(encoding="utf-8"))
    assert hinted["rules"]["failure_turns"] == 40  # its only rule: no fail-fast
    assert hinted["seeded_frames"] == len(package.frames)
    assert unhinted["rules"]["fail_fast"] is True and unhinted["seeded_frames"] == 0
    reader = json.loads((root / "reader" / "config.json").read_text(encoding="utf-8"))
    # The demonstration's keypoints (`demo_relation`'s) reach the Demo Reader and hinted
    # attempts only, as the recording's frames do; every S1 episode gets the arms its grasps
    # took (`plan_arms`'s default on the demonstration's scene).
    assert reader["seeded_facts"] == hinted["seeded_facts"] == ["demo", "demo_arms"]
    assert unhinted["seeded_facts"] == ["demo_arms"]
    assert reader["env"]["repeat"] is False
    assert unhinted["env"]["repeat"] is True  # the demo seed's setup check passed already
    brief = (root / "reviser" / "r1" / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "success never latched" in brief and "Ground truth after the try" in brief
    # The privileged review against the expert's state: the failed attempt's, for S1's Reviser
    # (and the hinted Actor after each action); nothing of it reaches S2.
    review = json.loads((root / "attempt1" / "referee" / "expert_diff.json").read_bytes())
    assert review["verdict"] == "missing_stage"  # the lift never reached the expert's K3
    assert not (root / "attempt2" / "referee" / "expert_diff.json").exists()  # passed
    assert "Expert comparison (simulator truth" in brief and "right K3 (lift" in brief
    assert "Expert comparison so far" in notes
    s2_briefs = list((setup.run_dir / LIFT_RED / "S2").rglob("llm_calls.jsonl"))
    assert s2_briefs
    assert not any("Expert comparison" in p.read_text(encoding="utf-8") for p in s2_briefs)
    # The package's ground-truth layer: the Demo Reader, hinted attempts and the Reviser only.
    layer = "The demonstration's ground truth"
    for seen in ("reader", "attempt1", "reviser/r1"):
        assert layer in (root / seen / "llm_calls.jsonl").read_text(encoding="utf-8"), seen
    for unseen in ("attempt2", "attempt3", "attempt4"):
        assert layer not in (root / unseen / "llm_calls.jsonl").read_text(encoding="utf-8"), unseen
    shown = {
        name: json.loads((root / name / "config.json").read_text(encoding="utf-8"))["protocol"][
            "demo_truth"
        ]
        for name in ("reader", "attempt1", "attempt2", "reviser/r1")
    }
    assert shown == {"reader": True, "attempt1": True, "attempt2": False, "reviser/r1": True}
    exit = json.loads((root / "exit.json").read_text(encoding="utf-8"))
    assert (exit["passed"], exit["version"], exit["demo_truth"]) == (True, 1, True)
    # S2 from v_S1 (0.1 m): three first-try passes (seed 10001 is unsolvable) and freeze.
    assert (s2.reason, s2.k, s2.version, s2.frozen) == ("converged", 3, 1, True)
    # evaluate: the frozen Memory on three validation seeds.
    assert (evaluated.version, evaluated.n, evaluated.successes) == (1, 3, 3)
    assert evaluated.wilson95[0] > 0.4 and evaluated.wilson95[1] == 1.0
    events = [
        json.loads(line)
        for line in (setup.run_dir / LIFT_RED / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    starts = [e["data"]["protocol"] for e in events if e["data"].get("step") == "start"]
    assert starts == ["imitate", "practice", "evaluate"]
    reviews = [e["data"] for e in events if e["data"].get("step") == "expert_diff"]
    assert [(r["attempt"], r["verdict"]) for r in reviews] == [(1, "missing_stage")]
    # The Reviser's ledger is model-visible input: logged as shown (one per revision).
    ledgers = [e["data"] for e in events if e["kind"] == "reviser.ledger"]
    assert [entry["attempt"] for entry in ledgers] == [1]
    assert ledgers[0]["lines"] == [] and len(ledgers[0]["sha256"]) == 64


async def test_two_unhinted_failures_insert_a_hinted_attempt(setup: Any) -> None:
    """The Actor gives up in the 3rd and 4th attempts (unhinted): the 5th is hinted; the
    hinted pass does not count, and the exit is 3 passes among the last 5 unhinted attempts."""
    demo = setup.run_dir.parent / "demo"
    await record_demo(setup.env, setup.blobs, demo)
    good = [
        op if op["op"] != "define_param" else {**op, "param": {**op["param"], "value": 0.1}}
        for op in READER_OPS
    ]
    brains = {
        "demo_reader": editing(lambda _: good),
        "actor": lambda e: finishing(e) if e.extra["attempt"] in (3, 4) else lifting_actor(e),
        "reviser": editing(lambda _: []),
    }
    async with setup.runner(brains) as runner:
        s1 = await runner.run(Imitate(), LIFT_RED, {"demo": str(demo)})
    assert isinstance(s1, ImitateResult)
    # 1 hinted pass; 2 unhinted pass, 3 and 4 fail; 5 hinted (inserted); 6, 7 unhinted pass:
    # 2, 6, 7 are 3 of the last 5 unhinted (2, 3, 4, 6, 7), which 3 in a row would not be.
    assert (s1.passed, s1.attempts, s1.hinted, s1.version) == (True, 7, 2, 0)
    root = setup.run_dir / LIFT_RED / "S1" / str(DEMO_SEED)
    shown = [
        json.loads((root / f"attempt{n}" / "config.json").read_text(encoding="utf-8"))["protocol"][
            "hinted"
        ]
        for n in range(1, 8)
    ]
    assert shown == [True, False, False, False, True, False, False]
    assert sorted(path.name for path in (root / "reviser").iterdir()) == ["r3", "r4"]


def test_the_exit_window() -> None:
    config = ImitateConfig(demo=Path("/demo"))
    assert (config.max_attempts, config.exit_passes, config.exit_window) == (60, 3, 5)
    assert exits([True, False, True, False, True], config)
    assert not exits([True, False, True, False, False, True], config)  # 2 of the last 5
    assert exits([False, True, True, True], config)
    with pytest.raises(ValueError, match="never exits"):
        ImitateConfig(demo=Path("/demo"), exit_passes=4, exit_window=3)


async def test_s1_gives_up_after_its_attempts(setup: Any) -> None:
    demo = setup.run_dir.parent / "demo"
    await record_demo(setup.env, setup.blobs, demo)
    brains = {
        "demo_reader": editing(lambda _: READER_OPS),
        "actor": lifting_actor,
        "reviser": editing(lambda _: []),
    }
    async with setup.runner(brains) as runner:
        s1 = await runner.run(Imitate(), LIFT_RED, {"demo": str(demo), "max_attempts": 3})
    assert isinstance(s1, ImitateResult)
    # The hinted first attempt fails, and so do the unhinted ones that follow it.
    assert (s1.passed, s1.reason, s1.attempts, s1.hinted) == (False, "attempts exhausted", 3, 1)
