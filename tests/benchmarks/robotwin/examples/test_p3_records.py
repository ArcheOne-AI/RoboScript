"""A P3 block's records as the pipeline reads them: a block whose per-episode outcomes could not be
read does not count as done once episodes ended, and the report counts only the host summaries
that count (not those of attempts infrastructure lost before, carried as history)."""

import json
from pathlib import Path
from typing import Any

import pytest

from roboscript.contracts.ids import ManifestId
from roboscript_robotwin.eval import launch
from roboscript_robotwin.eval import report as p3_report
from roboscript_robotwin.eval.blocks import block_done
from roboscript_robotwin.eval.launch import P3Run


def _summary(out: Path, slot: str, success: bool, infra: bool = False) -> None:
    folder = out / "host" / slot / "ep000"
    folder.mkdir(parents=True)
    summary = {"official": None if infra else {"success": success}, "infra": infra}
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_unread_outcomes_keep_a_block_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "block1"
    _summary(out, "a", True)

    def broken(_: Path) -> Any:
        raise ValueError("episode 3 after 1")

    monkeypatch.setattr(launch, "_ended", broken)
    run = P3Run(task="lift_red", seed=1, manifest=tmp_path / "m.json", out_dir=out)
    result = launch._collect(
        run, "demo_clean", ManifestId("mf-000000000000"), None, True, 0, launch._Launched(out)
    )
    assert result.outcomes == [] and result.outcomes_error == "ValueError: episode 3 after 1"
    assert not block_done(out)  # an ended episode, but which were lost is unknown
    monkeypatch.undo()
    launch._collect(
        run, "demo_clean", ManifestId("mf-000000000000"), None, True, 0, launch._Launched(out)
    )
    assert block_done(out)  # read: nothing lost


def test_the_report_leaves_lost_slots_out(tmp_path: Path) -> None:
    out = tmp_path / "clean" / "block1"
    _summary(out, "a", True)
    _summary(out, f"b.{launch.LOST}-block1-aborted1", False, infra=True)
    run = P3Run(task="lift_red", seed=1, manifest=tmp_path / "m.json", out_dir=out)
    launch._collect(
        run, "demo_clean", ManifestId("mf-000000000000"), None, True, 0, launch._Launched(out)
    )
    found = p3_report._p3(tmp_path, False, lambda mode: 1)
    block = found["blocks"]["clean/block1"]
    assert (block["host_seen"], block["host_ended"], block["infra"]) == (1, 1, 0)
    assert found["modes"]["clean"]["rate"] == 1.0
