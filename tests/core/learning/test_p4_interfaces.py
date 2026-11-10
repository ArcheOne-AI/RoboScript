"""The core's interfaces for P4 (docs/concepts/learning.md), method-neutral: artifact overlays
around the Actor's prompt, the read-only view of an evaluation's records, and the seed guard of a
reflection and its evaluation. P4 itself is `roboscript_methods.rsi` (tests/methods/rsi)."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import anyio
import pytest
from pydantic import Field, ValidationError

from roboscript.agent import call
from roboscript.agent.overlay import overlay_loader
from roboscript.contracts.artifact import Artifact, ArtifactLayer, ArtifactRef
from roboscript.contracts.camera import CameraModel
from roboscript.contracts.ids import TaskName
from roboscript.learning.eval_view import DIAGNOSIS, EvalRecordView, read_eval_records
from roboscript.learning.seeds import SeedRange, SeedSplit, check_reflection
from roboscript.memory.artifact import KIND
from roboscript.records import EventWriter
from roboscript.testing.host import LIFT_RED, mock_host
from roboscript.testing.phases import TestMethod

RATIONALE = {"stage": "wait", "scene": "three boxes", "progress": "at rest", "plan": "stay"}
SCRIPT = [[call("hold", rationale=RATIONALE, steps=3)], [call("finish", note="done")]]
KEY = "lift_red.p3.seed0"


@pytest.fixture
def events(artifacts_dir: Path) -> Iterator[EventWriter]:
    with EventWriter(artifacts_dir / "events.jsonl", "agent_host") as writer:
        yield writer


class Notes(Artifact):
    """An artifact an Actor reads beside its Task Memory: one line for every task, one per task."""

    kind: ClassVar[str] = "notes"
    every: str = ""
    per_task: dict[str, str] = Field(default_factory=dict)

    def context(self, task: TaskName | None) -> str:
        return self.every if task is None else self.per_task.get(task, "")


def _ref(kind: str, name: str, mark: str = "0") -> ArtifactRef:
    return ArtifactRef(kind=kind, name=name, version="v000", hash=mark * 64)


NOTES = Notes(every="# NOTES: every task", per_task={"lift_red": "# NOTES: lift_red"})


# Overlays


async def test_an_overlay_renders_its_layers_around_the_task_memory_in_stack_order() -> None:
    memory, notes = _ref(KIND, "lift_red"), _ref("notes", "n")
    layers = [
        ArtifactLayer(ref=notes),
        ArtifactLayer(ref=memory, task=LIFT_RED),
        ArtifactLayer(ref=notes, task=LIFT_RED),
        ArtifactLayer(ref=_ref("notes", "other", "1"), task=LIFT_RED),
    ]
    loaded: list[ArtifactRef] = []

    async def load(ref: ArtifactRef) -> Artifact:
        loaded.append(ref)
        return NOTES if ref.name == "n" else Notes()

    overlay = await overlay_loader(load, KIND)(LIFT_RED, layers)
    assert overlay.before == ("# NOTES: every task",)
    assert overlay.after == ("# NOTES: lift_red",)  # the empty part is left out
    assert overlay.layers == (layers[0], layers[2])
    assert loaded == [notes, layers[3].ref]  # each artifact once
    with pytest.raises(ValueError, match="one task_memory layer, not 0"):
        await overlay_loader(load, KIND)(LIFT_RED, [layers[0]])


# The evaluation records' view


def _seed_of(key: str, index: int, summary: Any) -> int:
    return 100 + index


async def test_the_view_of_a_hosts_records(tmp_path: Path, events: EventWriter) -> None:
    async with mock_host(tmp_path, events, SCRIPT, TestMethod, key=KEY) as host:
        with anyio.fail_after(60):
            await host.client.episode(seed=7)
            await host.client.episode(seed=8)
    lost = host.records / KEY / "ep001" / "summary.json"
    summary = json.loads(lost.read_text(encoding="utf-8"))
    (host.records / KEY / "ep000" / DIAGNOSIS).write_text('{"miss": "0.02 m"}', encoding="utf-8")
    episodes = read_eval_records(host.records, _seed_of)
    assert [(e.task, e.seed, e.success) for e in episodes] == [
        (LIFT_RED, 100, False),
        (LIFT_RED, 101, False),
    ]
    view = EvalRecordView.of(episodes)
    first = view.episodes[0]
    kinds = [part.kind for part in first.parts]
    assert kinds[0] == "instruction" and first.parts[0].content == "raise the red block"
    assert {"keyframe", "tool_call", "receipt", "finish"} <= set(kinds)
    assert "diagnosis" not in kinds  # D and the official success by default
    hold = next(p for p in first.parts if p.kind == "tool_call")
    assert hold.content["name"] == "hold" and hold.step == 0  # type: ignore[index]
    finish = next(p for p in first.parts if p.kind == "finish")
    assert finish.content == {"note": "done"} and finish.step == 3
    privileged = EvalRecordView.of(episodes, privileged=True)
    assert privileged.episodes[0].parts[-1].kind == "diagnosis"
    # an episode infrastructure lost is no failure of the agent's: left out
    lost.write_text(json.dumps({**summary, "infra": True}), encoding="utf-8")
    assert [e.seed for e in read_eval_records(host.records, _seed_of)] == [100]
    # so is one whose host died before its summary: no summary.json
    lost.unlink()
    assert [e.seed for e in read_eval_records(host.records, _seed_of)] == [100]


# The seed guard


def test_a_reflection_reads_and_its_evaluation_runs_only_their_blocks() -> None:
    split = SeedSplit(
        blocks={"p3": SeedRange(start=100, count=2), "p4_eval": SeedRange(start=200, count=2)},
        reflects={"p4_eval": "p3"},
    )
    check_reflection(split, "p4_eval", [100, 101], [200, 201])
    with pytest.raises(ValueError, match=r"seeds \[102\] lie outside seed block 'p3'"):
        check_reflection(split, "p4_eval", [101, 102])
    with pytest.raises(ValueError, match=r"seeds \[101\] lie outside seed block 'p4_eval'"):
        check_reflection(split, "p4_eval", [100], [101])
    with pytest.raises(ValueError, match="reflects on no block"):
        check_reflection(split, "p3", [100])


def test_a_reflection_on_its_own_evaluation_block_is_transductive_only() -> None:
    blocks = {"p3": SeedRange(start=100, count=2), "p4_eval": SeedRange(start=101, count=2)}
    with pytest.raises(ValidationError, match="set transductive"):
        SeedSplit(blocks=blocks, reflects={"p4_eval": "p3"})
    split = SeedSplit(blocks=blocks, reflects={"p4_eval": "p3"}, transductive=True)
    check_reflection(split, "p4_eval", [100, 101], [101, 102])


def _official(block: int) -> SeedRange:
    return SeedRange(start=100000 * (1 + block), count=100000)


BASE = SeedSplit(blocks={"p3_1": _official(1)})


def test_the_chain_refuses_leaks() -> None:
    blocks = {f"p3_{b}": _official(b) for b in (1, 2, 3, 4)}
    with pytest.raises(ValidationError, match=r"final blocks \['p3_3'\] are read"):
        SeedSplit(blocks=blocks, reflects={"p3_2": "p3_1", "p3_4": "p3_3"}, final=["p3_3"])
    with pytest.raises(ValidationError, match="also a round's evaluation block"):
        SeedSplit(blocks=blocks, reflects={"p3_2": "p3_1"}, final=["p3_2"])
    with pytest.raises(ValidationError, match="final names a block twice"):
        SeedSplit(blocks=blocks, reflects={"p3_2": "p3_1"}, final=["p3_3", "p3_3"])
    with pytest.raises(ValidationError, match=r"unknown blocks \['p3_9'\]"):
        SeedSplit(blocks=blocks, reflects={"p3_2": "p3_1"}, final=["p3_9"])
    with pytest.raises(ValidationError, match="not in the order of their evaluation blocks"):
        SeedSplit(blocks=blocks, reflects={"p3_3": "p3_1", "p3_2": "p3_1"})
    with pytest.raises(ValidationError, match="reflect on no block"):
        SeedSplit(blocks=blocks, reflects={"p3_2": []})
    # an earlier round read the block a later round evaluates on: its document carries it
    later = {"p3_2": "p3_3", "p3_3": "p3_2"}
    with pytest.raises(ValidationError, match="'p3_3' is read by its own or an earlier round"):
        SeedSplit(blocks=blocks, reflects=later)
    with pytest.raises(ValidationError, match="'p3_2' is read by its own or an earlier round"):
        SeedSplit(blocks=blocks, reflects={"p3_2": "p3_2"})
    SeedSplit(blocks=blocks, reflects=later, transductive=True)  # by opt-in only
    overlapping = {**blocks, "p3_5": SeedRange(start=300050, count=10)}
    with pytest.raises(ValidationError, match=r"'p3_2' and 'p3_5' overlap \["):
        SeedSplit(blocks=overlapping, reflects={"p3_3": "p3_1"}, final=["p3_5"])
    # a final block overlapping a read block is no reflection's evaluation: never allowed
    with pytest.raises(ValidationError, match=r"'p3_1' and 'p3_5' overlap \["):
        SeedSplit(
            blocks={**blocks, "p3_5": SeedRange(start=200010, count=5)},
            reflects={"p3_2": "p3_1"},
            final=["p3_5"],
            transductive=True,
        )


def _camera(name: str) -> CameraModel:
    k = ((1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
    pose = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0, 0, 0, 1.0))
    return CameraModel(name=name, width=2, height=2, K=k, cam2world=pose)
