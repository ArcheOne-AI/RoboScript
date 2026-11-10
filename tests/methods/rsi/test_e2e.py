"""P4 end to end on MockEnv with scripted brains (docs/concepts/learning.md): P3 evaluates a
frozen Task Memory through the Agent host; `p4_reflect` reads the read-only view of its records
and writes an RSI document through its roles (the reflector stages a trap for its task, all its
episodes having failed; the curator a GLOBAL entry, the lint refusing one that names the task; the
reviewer accepts them);
`p4_eval` evaluates again on another seed block, the Actor reading the Task Memory, the RSI
document's GLOBAL part and its task's part, in that order. And what is refused: records from
outside the reflected block, a privileged view the config did not open, a privileged episode of
`p4_eval`."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from roboscript.agent import ScriptedBrain, call, read_lines
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import TaskName
from roboscript.hub import Hub
from roboscript.learning.agent_host import AgentHostConfig, host_overlay
from roboscript.learning.eval_view import DIAGNOSIS, EvalRecordView, read_eval_records
from roboscript.learning.seeds import SeedRange, SeedSplit, check_reflection
from roboscript.memory import FileMemoryStore, build_manifest
from roboscript.memory.artifact import manifest_stack
from roboscript.records import EventWriter
from roboscript.testing.host import LIFT_RED, mock_host
from roboscript.testing.phases import TestMethod
from roboscript.tools import LocalProvider
from roboscript_methods.rsi.doc import EpisodeKey
from roboscript_methods.rsi.lint import LintContext
from roboscript_methods.rsi.method import EVAL, RsiConfig, attach, context_stack
from roboscript_methods.rsi.recipe import load_rsi_recipe
from roboscript_methods.rsi.reflect import EDITS, RECORD, Reflected, Reflection, reflect
from roboscript_methods.rsi.store import FileRsiStore
from roboscript_methods.rsi.toolset import RsiTools

METHOD = attach(TestMethod)
P3_KEY = "lift_red.p3.seed0"
P4_KEY = "lift_red.p4.seed1"
SPLIT = SeedSplit(
    blocks={"p3": SeedRange(start=100, count=2), "p4_eval": SeedRange(start=200, count=2)},
    reflects={"p4_eval": "p3"},
)
RATIONALE = {"stage": "wait", "scene": "three boxes", "progress": "at rest", "plan": "stay"}
ACTOR = [[call("hold", rationale=RATIONALE, steps=3)], [call("finish", note="done")]]

LEAKY = {
    "op": "add",
    "task": None,
    "when": "lifting in lift_red",
    "do": "go slowly",
    "check": "it rises",
    "evidence": [{"block": "p3", "task": "lift_red", "episode": 0}],
}
GLOBAL = {
    "op": "add",
    "task": None,
    "when": "a hold leaves the scene as it was",
    "do": "look at the scene before holding again",
    "check": "the next frame differs",
    "evidence": [{"block": "p3", "task": "lift_red", "episode": 0, "step": 3}],
}
OWN = {
    "op": "add",
    "task": "lift_red",
    "kind": "trap",
    "when": "the red block still rests on the table",
    "do": "lift it before you finish",
    "check": "it rests above the table",
    "evidence": [{"block": "p3", "task": "lift_red", "episode": 1}],
}
# the official index of each episode, in the view's order (a benchmark's numbering, given as data)
KEYS = [EpisodeKey(block="p3", task=LIFT_RED, episode=i) for i in range(2)]
# one task in the mock run: GLOBAL takes evidence from one task here
BASE = load_rsi_recipe(Path("recipes/rsi"))
RECIPE = BASE.model_copy(update={"rules": BASE.rules.model_copy(update={"global_min_tasks": 1})})
SCRIPTS = {
    "reflector": [
        [call("rsi_episode", key="p3/lift_red/ep000")],
        [call("rsi_doc")],
        [call("rsi_edit", edit=OWN)],
        [call("finish", note="one trap. Gaps: none")],
    ],
    "curator": [
        [call("rsi_edit", edit=LEAKY)],
        [call("rsi_edit", edit=GLOBAL)],
        [call("finish", note="one GLOBAL entry")],
    ],
    "reviewer": [
        [call("rsi_verdict", proposal="P-curator-01", accept=True, reason="shown")],
        [call("rsi_verdict", proposal="P-lift_red-01", accept=True, reason="shown")],
        [call("finish", note="2 accepted")],
    ],
}


@pytest.fixture
def events(artifacts_dir: Path) -> Iterator[EventWriter]:
    with EventWriter(artifacts_dir / "events.jsonl", "p4") as writer:
        yield writer


def _p3_seed(key: str, index: int, summary: Any) -> int:
    return 100 + index  # the mock client's slot runs one seed after another from 100


async def _p3(run: Path, events: EventWriter) -> Path:
    """P3: two episodes of lift_red's frozen Memory, seeds 100 and 101; its records."""
    async with mock_host(run, events, ACTOR, TestMethod, key=P3_KEY, records="p3") as host:
        with anyio.fail_after(60):
            for seed in (100, 101):
                await host.client.episode(seed=seed)
    return host.records


async def _reflect(
    run: Path,
    events: EventWriter,
    view: EvalRecordView,
    split: SeedSplit = SPLIT,
    config: RsiConfig | None = None,
    keys: list[EpisodeKey] = KEYS,
) -> Reflected:
    tools = RsiTools()
    blobs = LocalBlobStore(run / "blobs")
    hub = Hub([LocalProvider(tools)], blobs=blobs, phases=METHOD.phases, events=events)
    memory = (await FileMemoryStore(run / "memory").load(LIFT_RED)).doc
    reflection = Reflection(
        hub=hub,
        tools=tools,
        brain=lambda role, label: ScriptedBrain(SCRIPTS[role]),
        blobs=blobs,
        store=FileRsiStore(run / "rsi"),
        directory=run / "p4_reflect",
        recipe=RECIPE,
        context=LintContext(vocabulary=RECIPE.lint, tasks=[LIFT_RED], memories={LIFT_RED: memory}),
        events=events,
    )
    async with hub:
        with anyio.fail_after(60):
            return await reflect(reflection, config or RsiConfig(), view, split, round=1, keys=keys)


async def test_p3_then_reflect_then_p4_eval(tmp_path: Path, events: EventWriter) -> None:
    run = tmp_path / "run"
    p3 = await _p3(run, events)
    (p3 / P3_KEY / "ep000" / DIAGNOSIS).write_text('{"miss": "the block"}', encoding="utf-8")

    # p4_reflect: the view holds D parts and the official success; the diagnosis stays out
    view = EvalRecordView.of(read_eval_records(p3, _p3_seed))
    assert [e.seed for e in view.episodes] == [100, 101]
    reflected = await _reflect(run, events, view)
    doc = reflected.doc
    assert [(e.id, e.do) for e in doc.global_entries] == [
        ("G-01", "look at the scene before holding again")
    ]
    assert [(e.id, e.do) for e in doc.task_entries[LIFT_RED]] == [
        ("T-lift_red-01", "lift it before you finish")
    ]
    assert {e.status for _, e in doc.entries()} == {"probation"}
    store = FileRsiStore(run / "rsi")
    assert await store.frozen("G1") == reflected.ref
    assert await store.load_round(1) == doc and await store.rounds() == [1]
    assert await store.load(reflected.ref) == doc
    record = json.loads((run / "p4_reflect" / RECORD).read_text(encoding="utf-8"))
    assert (record["round"], record["tiers"], record["seeds"]) == (1, ["D"], [100, 101])
    assert record["episodes"] == ["p3/lift_red/ep000", "p3/lift_red/ep001"]
    assert record["triage"] == {LIFT_RED: "trap_only"}  # both episodes failed
    assert record["rsi"] == {"name": "G1", "hash": doc.content_hash()}
    assert record["proposals"] == {"applied": 2}
    logged = read_lines(run / "p4_reflect" / EDITS)
    assert [(e["kind"], e.get("id"), e.get("status")) for e in logged] == [
        ("proposal", "P-curator-01", "applied"),
        ("proposal", "P-lift_red-01", "applied"),
        ("refused_at_staging", None, None),
    ]
    assert "global_task" in logged[2]["reason"]
    assert [r["end"] for r in record["roles"]] == ["finish"] * 4  # with consolidation
    roles = run / "p4_reflect" / "reflect"
    turns = read_lines(roles / "curator" / "curator" / "pass1" / "turns.jsonl")
    shown = turns[0]["exchanges"][0]["calls"][0]["content"][0]["text"]
    assert shown.startswith("Refused: the document would refuse it")
    calls = (roles / "reflector" / LIFT_RED / "pass1" / "llm_calls.jsonl").read_text(
        encoding="utf-8"
    )
    assert "p3/lift_red/ep000 seed 100 [baseline]: FAILURE" in calls  # the brief names the key
    assert "the block" not in calls  # no diagnosis

    # p4_eval: on the other block, the Actor reads the Task Memory, GLOBAL, then its task's part
    manifest = await build_manifest(FileMemoryStore(run / "memory"), [LIFT_RED])
    stack_file = run / "stack.json"
    stack = context_stack(manifest_stack(manifest), reflected.ref)
    stack_file.write_text(stack.model_dump_json(), encoding="utf-8")
    host_config = AgentHostConfig.model_validate(
        {
            "manifest": str(run / "manifest.json"),
            "method": "wicl",
            "recipe": "recipes/wicl",
            "episodes": {"records": str(run / "p4"), "embodiment": "mock_bimanual"},
            "toolsets": [{"name": "sensing"}],
            "overlay": {
                "stack": str(stack_file),
                "stores": {"rsi": {"name": "rsi", "options": {"root": str(run / "rsi")}}},
            },
        }
    )
    loaded, overlay = host_overlay(host_config)
    assert loaded == stack
    async with mock_host(
        run,
        events,
        ACTOR,
        METHOD,
        key=P4_KEY,
        records="p4",
        stack=lambda _: stack,
        overlay=overlay,
    ) as host:
        with anyio.fail_after(60):
            for seed in (200, 201):
                await host.client.episode(seed=seed)
    check_reflection(SPLIT, EVAL, record["seeds"], [200, 201])
    for index in (0, 1):
        episode = host.records / P4_KEY / f"ep{index:03d}"
        context = next(
            line for line in read_lines(episode / "llm_calls.jsonl") if line["kind"] == "context"
        )
        prompt = context["instructions"]
        assert "(p3)" in prompt  # the Hub episode and the prompt rules: the base's eval phase
        order = [
            prompt.index("# Memory"),
            prompt.index("# EXPERIENCE: GLOBAL"),
            prompt.index("look at the scene before holding again"),
            prompt.index("# EXPERIENCE: TASK lift_red"),
            prompt.index("lift it before you finish"),
        ]
        assert order == sorted(order)
        assert "go slowly" not in prompt and "p3/lift_red" not in prompt  # no refused or evidence
        assert "helpful" not in prompt and "probation" not in prompt  # no counts or status
        config = json.loads((episode / "config.json").read_text(encoding="utf-8"))
        assert [layer["task"] for layer in config["overlay"]] == [None, LIFT_RED]
    p3_prompt = next(
        line
        for line in read_lines(p3 / P3_KEY / "ep000" / "llm_calls.jsonl")
        if line["kind"] == "context"
    )["instructions"]
    assert "EXPERIENCE" not in p3_prompt and "(p3)" in p3_prompt


async def test_every_episode_needs_its_key(tmp_path: Path, events: EventWriter) -> None:
    run = tmp_path / "run"
    view = EvalRecordView.of(read_eval_records(await _p3(run, events), _p3_seed))
    with pytest.raises(ValueError, match="give every episode of the view its key"):
        await _reflect(run, events, view, keys=KEYS[:1])
    with pytest.raises(ValueError, match=r"key p3/lift_red/ep000 given twice"):
        await _reflect(run, events, view, keys=[KEYS[0], KEYS[0]])
    other = EpisodeKey(block="p3", task=TaskName("push_green"), episode=0)
    with pytest.raises(ValueError, match=r"key p3/push_green/ep000 of another task"):
        await _reflect(run, events, view, keys=[other, KEYS[1]])
    assert not (run / "rsi").exists()


async def test_a_privileged_view_needs_the_diagnosis_config(
    tmp_path: Path, events: EventWriter
) -> None:
    run = tmp_path / "run"
    p3 = await _p3(run, events)
    (p3 / P3_KEY / "ep000" / DIAGNOSIS).write_text('{"miss": "the block"}', encoding="utf-8")
    privileged = EvalRecordView.of(read_eval_records(p3, _p3_seed), privileged=True)
    with pytest.raises(ValueError, match="the truth diagnosis is read when `diagnosis` says so"):
        await _reflect(run, events, privileged)
    with pytest.raises(ValueError, match="the truth diagnosis is read when"):
        await _reflect(run, events, EvalRecordView.of([]), config=RsiConfig(diagnosis=True))
    await _reflect(run, events, privileged, config=RsiConfig(diagnosis=True))
    calls = (
        run / "p4_reflect" / "reflect" / "reflector" / LIFT_RED / "pass1" / "llm_calls.jsonl"
    ).read_text(encoding="utf-8")
    assert "the block" in calls  # the upper-bound group reads it
    record = json.loads((run / "p4_reflect" / RECORD).read_text(encoding="utf-8"))
    assert (record["tiers"], record["config"]["diagnosis"]) == (["D", "P"], True)


async def test_p4_eval_opens_at_tier_d_only(tmp_path: Path) -> None:
    async with Hub([], blobs=LocalBlobStore(tmp_path), phases=METHOD.phases) as hub:
        with pytest.raises(ProtocolError, match="phase 'p4_eval' runs at tier D only"):
            await hub.open_episode(OpenEpisode(stage=EVAL, role="actor", tier="P"))
        with pytest.raises(ProtocolError, match="phase 'p4_reflect' runs at tier D only"):
            await hub.open_episode(OpenEpisode(stage="p4_reflect", role="reflector", tier="P"))
