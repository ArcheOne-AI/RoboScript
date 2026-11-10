"""The replay tape (roboscript.records.replay): the comparator finds the first divergence, and
what follows the success latch is not compared."""

import json
from pathlib import Path
from typing import Any

from roboscript.contracts.geometry import Pose
from roboscript.records.replay import (
    Tape,
    TapeConfig,
    compare,
)

BASE = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def recording(tmp: Path, name: str = "a", **values: Any) -> Tape:
    return Tape(TapeConfig(mode="record", directory=tmp / name, scope="job", **values))


def _copy_with(source: Path, target: Path, change: Any) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            out = target / path.relative_to(source)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(path.read_bytes())
    for path in target.rglob("compare.*.jsonl"):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text(
            "".join(json.dumps(change(line)) + "\n" for line in lines), encoding="utf-8"
        )


def test_the_comparator_finds_the_first_divergence(tmp_path: Path) -> None:
    tape = recording(tmp_path)
    tape.observe("episode", {"stage": "s2", "role": "actor", "tier": "D"})
    tape.observe("tool", {"name": "observe", "content": [{"type": "text", "text": "a view"}]})
    batch = {"batch_id": "batch-0123456789ab", "steps": [{"qpos": [0.1, 0.2]}]}
    tape.observe("env.agent.execute", {"request": {"batch": batch}, "ok": {"stopped": "completed"}})
    tape.observe("tool", {"name": "lift", "content": [{"type": "text", "text": "lifted"}]})
    assert compare(tmp_path / "a", tmp_path / "a", client_phases=["p3"]).equal

    def moved(line: dict[str, Any]) -> dict[str, Any]:
        if line["kind"] == "env.agent.execute":
            line["data"]["request"]["batch"]["steps"][0]["qpos"][1] = 0.25
            line["data"]["request"]["batch"]["batch_id"] = "<stripped>"
        return line

    _copy_with(tmp_path / "a", tmp_path / "c", moved)
    found = compare(tmp_path / "a", tmp_path / "c", client_phases=["p3"])
    assert not found.equal and len(found.divergences) == 1
    divergence = found.divergences[0]
    assert (divergence.scope, divergence.index, divergence.label) == ("job", 2, "env_request")
    assert "episode s2/actor" in divergence.report and "[1] tool observe" in divergence.report
    assert "-       0.2" in divergence.report and "+       0.25" in divergence.report

    _copy_with(tmp_path / "a", tmp_path / "d", lambda line: line)
    path = next((tmp_path / "d").rglob("compare.*.jsonl"))
    kept = path.read_text(encoding="utf-8").splitlines(keepends=True)[:3]
    path.write_text("".join(kept), encoding="utf-8")
    short = compare(tmp_path / "a", tmp_path / "d", client_phases=["p3"])
    assert short.divergences[0].label == "length" and "tool lift" in short.divergences[0].report


def test_what_follows_the_success_latch_is_not_compared(tmp_path: Path) -> None:
    def stream(name: str, after: int) -> None:
        tape = recording(tmp_path, name)
        tape.observe("episode", {"stage": "p3", "role": "actor", "tier": "D"})
        tape.observe("env.agent.execute", {"request": {}, "ok": {"stopped": "success"}})
        for _ in range(after):  # the agent's calls before it learns the episode ended
            tape.observe("env.agent.robot", {"request": {}, "ok": {"step": 9}})
            tape.observe("tool", {"name": "observe", "content": []})
        tape.observe("env.control.judge", {"request": {}, "ok": {"success": True}})

    stream("a", 3)
    stream("b", 0)
    found = compare(tmp_path / "a", tmp_path / "b", client_phases=["p3"])
    assert found.equal, found.text()
    assert found.kinds == {"episode": 1, "env.agent.execute": 1, "env.control.judge": 1}
    assert "6 recorded and 0 replayed records not compared" in found.text()
