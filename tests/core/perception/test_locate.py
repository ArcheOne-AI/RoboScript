"""perception.locate end to end on MockEnv: observe, analyze, propose, locate (logged), act, check
predicates, re-measure. Ground truth (MockEnv's control plane) is read only to assert."""

import math
from pathlib import Path

import numpy as np
import pytest

from roboscript.contracts.action import ActionBatch, ActionStep, ExecuteResult
from roboscript.contracts.env import EpisodeSpec, ObserveRequest
from roboscript.contracts.events import EventKind
from roboscript.contracts.ids import BatchId, EpisodeId, TaskName, new_id
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import TABLE_Z, MockConfig, MockEnv
from roboscript.perception.frames import Frame, Frames
from roboscript.perception.locate import StepLog, associate, locate, refresh, scene
from roboscript.perception.overlay import attribute_card
from roboscript.perception.predicates import check
from roboscript.perception.proposals import Proposal, propose
from roboscript.perception.registry import SceneRegistry
from roboscript.perception.views import FrameGeometry, analyze
from roboscript.records import EventWriter, read_events
from roboscript.testing.scenes import PERCEPTION
from roboscript.tools.frames import FrameStore

TASK = TaskName("stack_red_on_blue")
HOLD = load_embodiment(MOCK_BIMANUAL).arms["left"].gripper.holding_threshold
"""MockEnv's holding threshold, from its embodiment profile."""


class Episode:
    """MockEnv with one open episode, the agent side's frame store and qpos commands."""

    def __init__(self, tmp_path: Path, seed: int = 3) -> None:
        self.env = MockEnv(MockConfig(task=TASK, blob_dir=tmp_path / "blobs"))
        self.store = FrameStore(self.env.blobs, lambda frame_id: frame_id, lambda kind, data: None)
        self.frames = Frames(self.store, self.env.blobs)
        self.seed = seed
        home = [-0.2, 0.95, 0.0, math.pi / 2, 0.0, 1.0]
        self.qpos = [-0.25, *home, 0.25, *home]

    async def open(self) -> None:
        self.id = (await self.env.open(EpisodeSpec(task=TASK, seed=self.seed))).episode_id

    async def look(self) -> tuple[Frame, FrameGeometry]:
        await self.store.put(await self.env.observe(self.id, ObserveRequest()))
        frame = await self.frames.frame("live")
        return frame, await analyze(frame, PERCEPTION)

    async def move(self, tcp: list[float], grip: float) -> ExecuteResult:
        """The left arm's TCP to `tcp` (pointing down: the ee is 0.1 m above it)."""
        self.qpos[:7] = [tcp[0], tcp[1], tcp[2] + 0.1, 0.0, math.pi / 2, 0.0, grip]
        batch = ActionBatch(
            batch_id=BatchId(new_id("batch")),
            steps=[ActionStep(qpos=self.qpos, until_gripper_stable=grip < 0.5)],
            stop_on_success=False,
        )
        return await self.env.execute(self.id, batch)

    async def truth(self, box: str) -> np.ndarray:
        return np.asarray((await self.env.gt_snapshot(self.id)).objects[box].p)


def by_color(geometry: FrameGeometry, proposals: list[Proposal], color: str) -> Proposal:
    """What the agent does with the Set-of-Mark card: pick the mark of that color."""
    card = attribute_card(geometry.view("overhead").image.rgb, proposals, geometry.table)
    [line] = [line for line in card.splitlines() if line.split(": ")[1].startswith(color)]
    return next(p for p in proposals if line.startswith(f"mark {p.id}: "))


async def test_locate_act_and_check_on_mock_env(tmp_path: Path, artifacts_dir: Path) -> None:
    episode = Episode(tmp_path)
    await episode.open()
    _, geometry = await episode.look()
    assert set(geometry.views) == {"overhead", "wrist_left", "wrist_right"}
    assert geometry.table.z == pytest.approx(TABLE_Z, abs=1e-4)
    proposals = propose(geometry, PERCEPTION)
    assert len(proposals) == 3
    registry = SceneRegistry()
    with EventWriter(artifacts_dir / "events.jsonl", "perception") as events:
        log = StepLog(events, EpisodeId("ep-test"))
        red = locate(registry, geometry, "red block", by_color(geometry, proposals, "red"), log=log)
        blue = locate(
            registry, geometry, "blue block", by_color(geometry, proposals, "blue"), log=log
        )
    logged = list(read_events(artifacts_dir / "events.jsonl"))
    assert [e.kind for e in logged] == [EventKind.PERCEPTION_STEP] * 6
    assert [e.data["step"] for e in logged] == ["lift", "fit", "register"] * 2
    assert logged[2].data == {
        "step": "register",
        "object": "o1",
        "name": "red block",
        "confidence": 1.0,
    }
    assert logged[0].episode_id == "ep-test" and logged[0].data["points"] > 1000
    for record, box in ((red, "red"), (blue, "blue")):
        truth = await episode.truth(box)
        np.testing.assert_allclose(record.fit.center, truth, atol=2e-3)
        np.testing.assert_allclose(record.fit.obb.size, [0.05, 0.05, 0.05], atol=2e-3)

    # Grasp the red block with the left arm where perception says it is: the grasp relation, not
    # a re-detection, tells the registry what is held and where it goes.
    table = geometry.table
    center = list(red.fit.center)
    await episode.move([center[0], center[1], center[2] + 0.1], 1.0)
    await episode.move(center, 1.0)
    closed = await episode.move(center, 0.0)
    steps: list[tuple[str, dict[str, object]]] = []

    def decided(step: str, **data: object) -> None:
        steps.append((step, data))

    held = registry.on_close("left", closed.states[-1].robot, threshold=HOLD, log=decided)
    assert held is not None and held.id == red.id
    await episode.move([center[0], center[1], center[2] + 0.12], 0.0)
    frame, _ = await episode.look()
    now = scene(registry, frame.robot, table)
    assert check(now, "holding", arm="left", threshold=HOLD).value is True
    assert check(now, "lifted", obj=red.id, height=0.1).value is True
    assert check(now, "near", arm="left", obj=red.id, distance=0.03).value is True
    np.testing.assert_allclose(now.object(red.id).center, await episode.truth("red"), atol=3e-3)

    # Put it on the blue block and open: it settles on the blue block's top.
    target = list(blue.fit.top_center)
    await episode.move([target[0], target[1], target[2] + 0.06], 0.0)
    released = await episode.move([target[0], target[1], target[2] + 0.06], 1.0)
    settled = registry.on_open("left", released.states[-1].robot, table, log=decided)
    assert settled is not None and settled.id == red.id
    assert [s for s, _ in steps] == ["attach", "detach", "settle"]
    assert steps[0][1]["object"] == red.id and steps[2][1]["support"] == blue.id
    await episode.move([target[0], target[1], target[2] + 0.2], 1.0)
    assert (await episode.env.judge(episode.id)).success
    frame, _ = await episode.look()
    after = scene(registry, frame.robot, table)
    np.testing.assert_allclose(after.object(red.id).center, await episode.truth("red"), atol=3e-3)
    on = check(after, "on", a=red.id, b=blue.id, tol=0.015, min_overlap=0.5)
    assert on.value is True, on.reason
    assert check(after, "moved", obj=red.id, distance=0.01).value is True
    assert (
        check(after, "moved", obj=blue.id, distance=0.01).value is False
    )  # never grasped: as registered
    assert registry.get(red.id).step == 0  # moved by the grasp relation, never re-measured


async def test_refresh_finds_a_moved_object(tmp_path: Path, artifacts_dir: Path) -> None:
    episode = Episode(tmp_path, seed=5)
    await episode.open()
    _, geometry = await episode.look()
    registry = SceneRegistry()
    proposals = propose(geometry, PERCEPTION)
    for color in ("red", "green", "blue"):
        locate(registry, geometry, f"{color} block", by_color(geometry, proposals, color))
    red = registry.get("o1")
    center = list(red.fit.center)
    await episode.move([center[0], center[1], center[2] + 0.1], 1.0)
    await episode.move(center, 1.0)
    await episode.move(center, 0.0)
    await episode.move([center[0], center[1], center[2] + 0.08], 0.0)
    dropped = [center[0] - 0.03, center[1] + 0.02, center[2] + 0.08]
    await episode.move(dropped, 0.0)
    await episode.move(dropped, 1.0)  # released: it falls onto the table, 3.6 cm from before
    await episode.move([dropped[0], dropped[1], dropped[2] + 0.2], 1.0)
    _, geometry = await episode.look()
    proposals = propose(geometry, PERCEPTION)
    with EventWriter(artifacts_dir / "events.jsonl", "perception") as events:
        log = StepLog(events, EpisodeId("ep-test"))
        record = refresh(registry, geometry, red.id, proposals, log=log)
        with pytest.raises(LookupError, match="o2 is not where it was"):
            refresh(registry, geometry, "o2", [], log=log)
    np.testing.assert_allclose(record.fit.center, await episode.truth("red"), atol=2e-3)
    steps = [
        (e.data["step"], e.data.get("region")) for e in read_events(artifacts_dir / "events.jsonl")
    ]
    assert steps[0][0] == "associate" and steps[0][1] is not None
    assert [s for s, _ in steps] == ["associate", "lift", "fit", "associate"] and steps[-1][
        1
    ] is None
    # Association is one to one: two objects never take the same proposal.
    expected = {i: registry.geometry(i) for i in registry.objects}
    matched = associate(expected, proposals)
    assert sorted(matched) == ["o1", "o2", "o3"]
    assert len({p.id for p, _ in matched.values()}) == 3
    assert associate(expected, proposals[:1]).keys() <= {"o1", "o2", "o3"}
    assert len(associate(expected, proposals[:1])) == 1
