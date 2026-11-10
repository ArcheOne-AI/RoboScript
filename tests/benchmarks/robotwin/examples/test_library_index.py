"""P3 in library mode in the pipeline (`run.with_library_index`): the library is every task's
frozen Memory in the run's variant, never one task's manifest alone, and P3 waits until each task
has frozen one or ended without (S1 not passed, S2 not converged)."""

import sys
from pathlib import Path

import pytest
from pipeline_fakes import config as pipeline_config
from robotwin_pipeline.run import LIBRARY_INDEX, with_library_index

from roboscript.contracts.ids import TaskName
from roboscript.memory import FileMemoryStore, VersionSource, build_manifest, write_manifest
from roboscript.memory.library import read_index
from roboscript.orchestrate.jobs import RunKey, RunState
from roboscript.stack import Interpreters
from roboscript.testing.memory import sample_memory
from roboscript_robotwin.eval.blocks import MANIFEST
from roboscript_robotwin.eval.launch import P3Config

TASKS = ["lift_red", "stack_two", "beat_block_hammer"]


def launcher(**host: object) -> P3Config:
    python = Path(sys.executable)
    return P3Config(
        interpreters=Interpreters(agent=python, sim=python),
        sim_pythonpath=[],
        host={"memory_access": "library", **host},
    )


async def freeze(unit: Path, task: str) -> None:
    store = FileMemoryStore(unit)
    await store.commit(sample_memory(task), status="s1_draft", source=VersionSource(kind="s1"))
    await store.freeze(task, 0, reason="for the test")
    write_manifest(unit / MANIFEST, await build_manifest(store, [TaskName(task)]))


async def test_the_library_spans_every_task_of_the_variant(tmp_path: Path) -> None:
    config = pipeline_config(tmp_path, TASKS, {"default": {}})
    stamp = tmp_path / "runs" / "stamp"
    units = {t: RunKey(TaskName(t), "default").unit_dir(stamp) for t in TASKS}
    key = RunKey(TaskName(TASKS[0]), "default")
    await freeze(units[TASKS[0]], TASKS[0])
    with pytest.raises(ValueError, match=r"\['stack_two', 'beat_block_hammer'\] have none yet"):
        with_library_index(config, key, units[TASKS[0]], launcher())
    await freeze(units[TASKS[1]], TASKS[1])
    units[TASKS[2]].mkdir(parents=True)
    RunState(task=TaskName(TASKS[2]), variant="default", status="not_passed").save(units[TASKS[2]])
    given = with_library_index(config, key, units[TASKS[0]], launcher())
    path = units[TASKS[0]] / LIBRARY_INDEX
    assert given.host["memory_library"]["index"] == str(path)
    index = read_index(path)
    assert [e.task for e in index.entries] == TASKS[:2]  # S1 not passed: no Memory to give
    assert {e.store.options["root"] for e in index.entries} == {
        str(units[TASKS[0]]),
        str(units[TASKS[1]]),
    }
