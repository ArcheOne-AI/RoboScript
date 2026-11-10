"""wicl's edit check on the memory toolset (`check_fitted`, `toolsets.memory.EditCheck`): with a
demonstration in the episode, a spec is defined as `demo_relation` fitted it."""

import itertools
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.hub import CallOutcome, OpenEpisode
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import ToolCall
from roboscript.hub import Hub
from roboscript.manipulation import GraspSpec
from roboscript.memory import Editor, MemoryDoc, MemoryRecipe, unfitted
from roboscript.records import EventWriter
from roboscript.testing.phases import TEST_PHASES
from roboscript.tools import DEMO, DEMO_FITS, LocalProvider
from roboscript.toolsets.memory import MemoryTools
from roboscript_methods.wicl.toolsets.demo import check_fitted

WHY = {"why": "the gripper stopped short of the handle"}
_IDS = itertools.count(1)


@pytest.fixture
def tools() -> MemoryTools:
    return MemoryTools(checks=[check_fitted])


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


async def test_with_a_demonstration_specs_are_defined_as_fitted(
    hub: Hub, tools: MemoryTools, recipe: MemoryRecipe
) -> None:
    """With a demonstration in the episode, a spec is defined as demo_relation fitted it there
    (its arm and search may change), and keeps where; the draft lint flags one that is not."""
    session = await hub.open_episode(OpenEpisode(stage="s1", role="demo_reader"))
    episode = session.episode_id
    draft = MemoryDoc(task="stack_two", instruction="Stack the blocks.")
    editor = Editor(draft, rules=recipe.rules("s1"), lint=recipe.lint)
    tools.attach(episode, editor)
    frames = hub.store.get(episode).frames
    fitted = {"approach": "top", "anchor": "top_center", "offset_m": [0.0, 0.0, -0.017]}
    frames.set_fact(DEMO, {"arms": {}})
    frames.set_fact(
        DEMO_FITS, {"left K2": [GraspSpec.model_validate(fitted).model_dump(mode="json")]}
    )

    def define(name: str, value: dict[str, Any], where: str | None) -> dict[str, Any]:
        param = {"kind": "grasp_spec", "value": value}
        return {"op": "define_param", "name": name, "param": param, "fitted": where, **WHY}

    unfitted_op = await call(hub, episode, "memory_edit", op=define("a", fitted, None))
    other = {**fitted, "offset_m": [0.0, 0.0, 0.0]}
    wrong = await call(hub, episode, "memory_edit", op=define("b", other, "left K2"))
    elsewhere = await call(hub, episode, "memory_edit", op=define("c", fitted, "left K3"))
    right = await call(
        hub, episode, "memory_edit", op=define("d", {**fitted, "arm": "right"}, "left K2")
    )
    assert text(unfitted_op).startswith("fit this spec on the demonstration first")
    assert text(wrong).startswith("demo_relation fitted no such spec at left K2")
    assert text(elsewhere).startswith("demo_relation fitted no such spec at left K3")
    assert text(right).startswith("Accepted define_param")
    assert editor.doc.params["d"].source == "demo left K2" and not unfitted(editor.doc)
    reviser = (await hub.open_episode(OpenEpisode(stage="s1", role="reviser"))).episode_id
    doc = MemoryDoc(task="stack_two", instruction="Stack the blocks.")
    plain = Editor(doc, rules=recipe.rules("s1"), lint=recipe.lint)
    tools.attach(reviser, plain)  # without a demonstration a spec may be defined as it is
    assert text(await call(hub, reviser, "memory_edit", op=define("e", other, None))).startswith(
        "Accepted"
    )
    [finding] = unfitted(plain.doc)
    assert (finding.rule, finding.where, finding.level) == ("unfitted_spec", "params.e", "error")
