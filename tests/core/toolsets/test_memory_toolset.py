"""The memory toolset through the Hub: who sees it, a Reviser editing its cache within budget, the
edit schema, and a method's edit check refusing an edit before the editor sees it."""

import itertools
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import ToolError
from roboscript.contracts.hub import CallOutcome, OpenEpisode
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import ToolCall
from roboscript.hub import Hub
from roboscript.memory import (
    EditCosts,
    EditOp,
    EditRecord,
    MemoryDoc,
    MemoryRecipe,
    open_cache,
)
from roboscript.records import EventWriter, read_events
from roboscript.testing.phases import TEST_PHASES
from roboscript.tools import LocalProvider, ToolContext, check_strict, to_responses_tool
from roboscript.toolsets.memory import CLAMPED, MemoryTools

WHY = {"why": "the gripper stopped short of the handle"}
_IDS = itertools.count(1)


@pytest.fixture
def events(artifacts_dir: Path) -> Iterator[EventWriter]:
    with EventWriter(artifacts_dir / "events.jsonl", "hub") as writer:
        yield writer


@pytest.fixture
def tools() -> MemoryTools:
    return MemoryTools()


@pytest.fixture
async def hub(tmp_path: Path, events: EventWriter, tools: MemoryTools) -> AsyncIterator[Hub]:
    blobs = LocalBlobStore(tmp_path / "blobs")
    async with Hub(
        [LocalProvider(tools)], blobs=blobs, phases=TEST_PHASES, events=events
    ) as started:
        yield started


async def call(hub: Hub, episode: EpisodeId, tool: str, **arguments: Any) -> CallOutcome:
    request = ToolCall(call_id=f"call-{next(_IDS)}", name=tool, arguments=arguments, origin="model")
    return await hub.call(episode, request)


def text(outcome: CallOutcome) -> str:
    return "\n".join(part.text for part in outcome.content)  # type: ignore[union-attr]


async def test_only_demo_readers_and_revisers_see_the_tools(hub: Hub) -> None:
    seen = {}
    for phase in TEST_PHASES:
        for role in phase.visibility:
            session = await hub.open_episode(OpenEpisode(stage=phase.name, role=role))
            seen[phase.name, role] = [t.name for t in session.tools if t.meta.toolset == "memory"]
    both = ["memory_read", "memory_edit"]
    assert seen == {
        (phase.name, role): both if phase.name in ("s1", "s2") and role != "actor" else []
        for phase in TEST_PHASES
        for role in phase.visibility
    }


async def test_a_reviser_edits_its_cache_within_budget(
    hub: Hub, tools: MemoryTools, recipe: MemoryRecipe, memory: MemoryDoc, artifacts_dir: Path
) -> None:
    session = await hub.open_episode(OpenEpisode(stage="s2", role="reviser"))
    episode = session.episode_id
    cache = open_cache(
        memory.replace(version=2), budget=2, seed=10004, rules=recipe.rules("s2"), lint=recipe.lint
    )
    tools.attach(episode, cache)
    shown = text(await call(hub, episode, "memory_read"))
    assert shown.startswith(
        "Budget: 2 of 2 units left. Costs: set_number 0, set_category 1, rotate_spec 0, add_"
    )
    assert (
        "\nThis revision may change 3 more numeric targets, each within its trust radius.\n"
        f"{CLAMPED}\n\n"
        "# TASK MEMORY: lift_red   (draft, S1, -, cache of v002)\n\n"
        "## PARAMETERS\n- pre_grasp = 0.1 m  (bounds [0.05, 0.15]"
    ) in shown
    assert shown.endswith("\n\n## SCRATCH\n")
    edits = [
        {"op": "set_number", "param": "pre_grasp", "value": 0.12, **WHY},
        {
            "op": "set_category",
            "param": "grasp_handle",
            "field": "approach",
            "value": "side",
            **WHY,
        },
        {"op": "set_number", "param": "pre_grasp", "value": 0.2, **WHY},
        {
            "op": "add_lesson",
            "when": "the tool lies diagonal",
            "do": "grasp nearer the head",
            **WHY,
        },
        {"op": "edit_lesson", "lesson": "L1", "do": "approach from the top", **WHY},
        {"op": "scratch", "text": "the handle looked shorter than measured"},
        {"op": "set_text", "section": "statement", "text": "Beat the base.", **WHY},
    ]
    outcomes = [await call(hub, episode, "memory_edit", op=op) for op in edits]
    assert not any(outcome.is_error for outcome in outcomes)  # refusals are results
    assert [text(outcome) for outcome in outcomes] == [
        "Accepted set_number (cost 0; 2 budget units left).",
        "Accepted set_category (cost 1; 1 budget unit left).",
        "Rejected set_number: pre_grasp: value 0.2 is outside its bounds [0.05, 0.15]",
        "Accepted add_lesson (cost 1; 0 budget units left).",
        "Rejected edit_lesson: edit_lesson costs 1; 0 of 2 units are left",
        "Accepted scratch (cost 0; 0 budget units left).",
        "Rejected set_text: set_text costs 2; 0 of 2 units are left",
    ]
    empty = await call(hub, episode, "memory_edit", op={"op": "edit_lesson", "lesson": "L1", **WHY})
    assert (empty.is_error, empty.error_code) == (True, "invalid_arguments")
    assert (cache.used, cache.doc.params["pre_grasp"].value) == (2, 0.12)
    assert cache.doc.scratch == ["the handle looked shorter than measured"]
    changes = text(await call(hub, episode, "memory_read", section="diff"))
    assert changes.startswith("Budget: 0 of 2 units left.")
    assert changes.endswith(
        '\n\n## CHANGES\n- params.pre_grasp: 0.1 -> 0.12\n- params.grasp_handle.approach: "top" '
        '-> "side"\n- lessons.L2: added\n'
    )
    logged = [e for e in read_events(artifacts_dir / "events.jsonl") if e.kind == "memory.edit"]
    assert [e.data["accepted"] for e in logged] == [True, True, False, True, False, True, False]
    assert {e.episode_id for e in logged} == {episode}
    assert [EditRecord.model_validate(e.data) for e in logged] == cache.records
    assert tools.detach(episode) is cache


def test_the_memory_edit_schema() -> None:
    spec = {spec.name: spec for spec in MemoryTools.specs()}["memory_edit"]
    check_strict(spec.input_schema)
    branches = spec.input_schema["properties"]["op"]["anyOf"]
    assert sorted(b["properties"]["op"]["enum"][0] for b in branches) == sorted(
        EditCosts.model_fields
    )
    define = next(b for b in branches if b["properties"]["op"]["enum"] == ["define_param"])
    kinds = [b["properties"]["kind"] for b in define["properties"]["param"]["anyOf"]]
    assert kinds == [
        {"type": "string", "enum": [kind]}
        for kind in (
            "number",
            "category",
            "grasp_spec",
            "place_spec",
            "hinge_spec",
            "path_spec",
            "done_spec",
        )
    ]
    assert to_responses_tool(spec)["strict"] is True
    assert (spec.meta.kind, spec.meta.uses_steps, spec.meta.needs_rationale) == (
        "memory",
        False,
        False,
    )


def refuse_lessons(ctx: ToolContext, op: EditOp) -> None:
    """A method's edit check (`toolsets.memory.EditCheck`) that refuses one kind of edit."""
    if op.op == "add_lesson":
        raise ToolError("this method takes no lessons")


async def test_a_methods_edit_check_refuses_an_edit_before_the_editor_sees_it(
    tmp_path: Path, events: EventWriter, recipe: MemoryRecipe, memory: MemoryDoc
) -> None:
    tools = MemoryTools(checks=[refuse_lessons])
    blobs = LocalBlobStore(tmp_path / "blobs")
    async with Hub([LocalProvider(tools)], blobs=blobs, phases=TEST_PHASES, events=events) as hub:
        episode = (await hub.open_episode(OpenEpisode(stage="s2", role="reviser"))).episode_id
        cache = open_cache(
            memory.replace(version=2), budget=2, seed=1, rules=recipe.rules("s2"), lint=recipe.lint
        )
        tools.attach(episode, cache)
        lesson = {"op": "add_lesson", "when": "it slips", "do": "grip lower", **WHY}
        refused = await call(hub, episode, "memory_edit", op=lesson)
        assert refused.is_error and "this method takes no lessons" in text(refused)
        assert cache.used == 0 and cache.records == []  # the editor never saw it
        number = {"op": "set_number", "param": "pre_grasp", "value": 0.12, **WHY}
        assert text(await call(hub, episode, "memory_edit", op=number)).startswith("Accepted")
