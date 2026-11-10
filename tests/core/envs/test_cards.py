"""Success condition cards (docs/concepts/embodied-core.md): the check that keeps a scene's values,
simulator state and code out, and how a card goes under the instruction."""

import json
from pathlib import Path

import pytest

from roboscript.envs.cards import (
    LEAD,
    MAX_CHARS,
    check_card,
    read_card,
    shown,
    task_text,
)

SHA = "a" * 64
TEXT = (
    "Each mug hangs from the holder by its handle on a hook: it rests there with nothing holding "
    "it, not merely touching the holder. Both mugs count, in any order, and 2 hooks may hold them."
)


def write(directory: Path, name: str, text: str = TEXT, **meta: object) -> Path:
    """`<name>.md` with a card of task `name` (`meta` replaces front matter)."""
    head = {"task": name, "version": 1, "source": "bench abc123 run_reward", "source_sha256": SHA}
    head.update(meta)
    lines = "\n".join(f"{key}: {json.dumps(value)}" for key, value in head.items())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(f"---\n{lines}\n---\n{text}\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("The mug ends 0.02 above the hook.", "numbers"),
        ("Within 2 cm of the hook.", "units"),
        ("Tilted less than 15°.", "units"),
        ("At least 15 balls.", "numbers"),
        ("Lower than -3.", "numbers"),
        ("The cup ends at (3, 4).", "tuples"),
        ("The same as in seed 3.", "simulator state"),
        ("The /World/mug prim rests on the hook.", "simulator state"),
        ("As in tasks/hang_mugs.py.", "simulator state"),
        ("```\ncheck\n```", "code"),
        ("success when mug_z == hook_z", "code"),
        ("run_reward(env) is true", "code"),
        ("x" * (MAX_CHARS + 1), "characters"),
    ],
)
def test_a_scene_s_values_state_and_code_are_refused(text: str, why: str) -> None:
    problems = check_card(text)
    assert problems and any(why in problem for problem in problems), problems


def test_a_card_goes_under_the_instruction(tmp_path: Path) -> None:
    card = read_card(write(tmp_path, "lift_red"))
    assert task_text("Lift the red cube.", None) == "Lift the red cube."
    text = task_text("Lift the red cube.", card)
    assert text == f"Lift the red cube.\n\n{LEAD}\n{TEXT}"
    assert shown(card, f"# Task\n{text}\n# Memory") is card
    assert shown(card, "# Task\nLift the red cube.") is None and shown(None, text) is None
