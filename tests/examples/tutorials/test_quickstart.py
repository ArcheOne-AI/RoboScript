"""The quickstart episode runs, ends by `finish`, and leaves the records the README names."""

from pathlib import Path

import pytest
from quickstart_mock.run import run

pytestmark = pytest.mark.unit


async def test_the_scripted_episode_runs_to_finish(tmp_path: Path) -> None:
    outcome = await run(tmp_path)
    assert (outcome.end, outcome.turns, outcome.success) == ("finish", 2, False)
    assert outcome.events["tool.call"] >= 3 and outcome.events["model.request"] == 3
    assert (tmp_path / "events.jsonl").is_file()
    assert {"turns.jsonl", "llm_calls.jsonl"} <= {p.name for p in (tmp_path / "episode").iterdir()}
