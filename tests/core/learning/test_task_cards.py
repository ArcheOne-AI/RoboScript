"""A task's success condition card (docs/concepts/embodied-core.md) as the runner's `evaluate`
reads it: right under the instruction, recorded as a `task.card` event and in `config.json`;
without a card, prompts and records hold nothing of it."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from roboscript.agent import read_lines
from roboscript.contracts.events import EventKind
from roboscript.envs.cards import LEAD, TaskCard
from roboscript.learning import Evaluate
from roboscript.records import EventWriter, read_events
from roboscript.testing.protocols import LIFT_RED, acting, lift_memory

CARD = TaskCard(
    task=LIFT_RED,
    version=2,
    source="mock run_reward",
    source_sha256="b" * 64,
    text="The box ends lifted clear of the table, resting on nothing.",
)
BLOCK = f"\n\n{LEAD}\n{CARD.text}"


def context(directory: Path) -> str:
    lines = read_lines(directory / "llm_calls.jsonl")
    prompt: str = next(line for line in lines if line["kind"] == "context")["instructions"]
    return prompt


def config(directory: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return loaded


async def test_evaluate_shows_the_card_under_the_instruction(setup: Any) -> None:
    await setup.start(lift_memory(0.06))
    await setup.store.freeze(LIFT_RED, 0, reason="for the test")
    actor = acting(lambda e: "pass")
    async with setup.runner({"actor": actor}) as runner:
        assert runner.context(LIFT_RED).card is None
        await runner.run(Evaluate(), LIFT_RED, {"seeds": [20000], "name": "plain"})
    async with setup.runner({"actor": actor}, cards={LIFT_RED: CARD}) as runner:
        assert runner.context(LIFT_RED).card == CARD
        await runner.run(Evaluate(), LIFT_RED, {"seeds": [20000], "name": "card"})
    plain = setup.run_dir / LIFT_RED / "eval" / "plain" / "20000" / "try1"
    carded = setup.run_dir / LIFT_RED / "eval" / "card" / "20000" / "try1"
    instruction = json.loads((plain / "summary.json").read_text(encoding="utf-8"))["instruction"]
    assert instruction in context(plain) and BLOCK not in context(plain)
    assert context(carded) == context(plain).replace(instruction, instruction + BLOCK, 1)
    assert "card" not in config(plain)  # records without a card have no such keys
    assert config(carded)["card"] == CARD.record()
    assert not [e for e in read_events(plain / "events.jsonl") if e.kind == EventKind.TASK_CARD]
    [shown] = [e for e in read_events(carded / "events.jsonl") if e.kind == EventKind.TASK_CARD]
    assert shown.data == CARD.record() and shown.episode_id is not None
    summary = json.loads((carded / "summary.json").read_text(encoding="utf-8"))
    assert summary["instruction"] == instruction  # the instruction itself is unchanged
    assert summary["success"] is True


@pytest.fixture
def events(artifacts_dir: Path) -> Iterator[EventWriter]:
    with EventWriter(artifacts_dir / "events.jsonl", "agent_host") as writer:
        yield writer
