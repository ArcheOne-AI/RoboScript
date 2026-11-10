"""wicl's operations on a run's records as a scheduler keeps them: where a task's S1 lies
(`S1`); the record an S1 that did not pass leaves when its fallback lets its variants start
(`FALLBACK`, `write_fallback`); how a variant's S2 stopped (`s2_stop`) and the record of how it
ended for the evaluation (`S2_EXIT`); and the protocol parameters an S1 or S2 job runs with over the
recipe's (`imitate_values`, `practice_values`).

The functions that read a run's config take it as `OpsConfig`: a `typing.Protocol` of the members
they read (the protocol parameters of a task, its demo package, the recipe, S1's fallback
switch and rule, the end-state switch). It is in-process only, so it is a structural type, not
a pydantic contract; any pipeline config with those members satisfies it."""

import json
import logging
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from roboscript.contracts.ids import TaskName
from roboscript.learning import ProtocolRecipe
from roboscript.memory import FileMemoryStore
from roboscript.memory.store import FROZEN_FILE
from roboscript.orchestrate.jobs import RunKey
from roboscript.records import write_atomic
from roboscript_methods.wicl.imitate import EXIT
from roboscript_methods.wicl.practice import STOP

log = logging.getLogger(__name__)

S1: Final = "S1"
"""The directory, and the unit of the key, of a task's shared S1."""
FALLBACK: Final = "s1_fallback.json"
"""In a task's S1 directory once S1 ran out of attempts and its fallback found a Memory
(`write_fallback`): the version its variants start S2 from."""


class OpsConfig(Protocol):
    """What these operations read of a run's config: the parameters of protocol `name` for a task
    (and a variant), as the run sets them over the recipe's (`protocol`), the task's demo package
    (`demo_of`), the recipe (`recipe`, its protocols' defaults), whether an S1 that runs out of
    attempts yields its best Memory (`s1_fallback`) and under which rule (`s1_fallback_rule`,
    `ImitateConfig.fallback_rule` in `roboscript_methods.wicl.imitate`), and whether S2 and P3
    compare with the demonstration's end (`done_goal`)."""

    @property
    def s1_fallback(self) -> bool: ...

    @property
    def s1_fallback_rule(self) -> Literal["success", "progress"]: ...

    @property
    def done_goal(self) -> bool: ...

    def recipe(self) -> ProtocolRecipe: ...

    def protocol(self, name: str, task: str, variant: str | None = None) -> dict[str, Any]: ...

    def demo_of(self, task: str) -> Path: ...


def seed_variant(s1_dir: Path, run_dir: Path, task: str) -> None:
    """Give a variant its task's S1, once: the Memory versions S1 made and its passed
    `exit.json`, where S2's gate finds them."""
    target = FileMemoryStore(run_dir).folder(task)
    if not target.exists():
        staging = target.with_name(".memory-staging")
        shutil.rmtree(staging, ignore_errors=True)  # a copy that a dead worker left half done
        shutil.copytree(FileMemoryStore(s1_dir).folder(task), staging)
        staging.rename(target)
    for passed in (s1_dir / task / "S1").glob(f"*/{EXIT}"):
        copy = run_dir / task / "S1" / passed.parent.name / EXIT
        if not copy.is_file():
            write_atomic(copy, passed.read_bytes())


def write_fallback(s1_dir: Path, task: TaskName, fallback: Mapping[str, Any]) -> None:
    """The record of an S1 that did not pass but found a fallback: its variants start from it."""
    record = {
        "version": fallback["version"],
        "hash": fallback["hash"],
        "at": time.time(),
        "converged": False,
        "s1_exit": "best_memory",
    }
    if "hinted_only" in fallback:  # its official successes all came with the hint
        record["hinted_only"] = fallback["hinted_only"]
    write_atomic(s1_dir / FALLBACK, (json.dumps(record) + "\n").encode())
    log.warning("%s: its variants start from S1's fallback v%03d", task, fallback["version"])


def last_version(run_dir: Path, task: TaskName) -> tuple[int, str] | None:
    """The version a variant's Memory ends at: the frozen one, else its latest (number, hash);
    None without one. Read from the files alone (a dry run reads it too)."""
    folder = FileMemoryStore(run_dir).folder(task)
    if (frozen := folder / FROZEN_FILE).is_file():
        record = json.loads(frozen.read_bytes())
        return int(record["version"]), str(record["hash"])
    metas = sorted(folder.glob("v[0-9][0-9][0-9]/meta.json"))
    if not metas:
        return None
    meta = json.loads(metas[-1].read_bytes())
    return int(meta["version"]), str(meta["hash"])


UNCONVERGED: Final = "not converged"
"""A practice's stop reason at its seed cap (`roboscript_methods.wicl.practice.Stop`)."""


def s2_stop(run_dir: Path, task: TaskName) -> dict[str, Any] | None:
    """How a variant's S2 stopped, from its files: its freeze record (written first) or its
    `stop.json` (`reason`, and `k` when the latter has it); None while it has not stopped."""
    stop = run_dir / task / "S2" / STOP
    if stop.is_file():
        found = json.loads(stop.read_bytes())
        return {"reason": found["reason"], "k": found.get("k")}
    frozen = FileMemoryStore(run_dir).folder(task) / FROZEN_FILE
    if frozen.is_file():
        return {"reason": json.loads(frozen.read_bytes())["reason"], "k": None}
    return None


S2_EXIT: Final = "s2_exit.json"
"""How a variant's S2 ended for the evaluation (a pipeline's S2 pick), in its S2
folder: `s2_exit` `converged`, `best_memory` (stopped at the cap, the version with the best
first-try record frozen, `picked`) or `none` (stopped at the cap with no first-try success: no
P3, it counts 0)."""


def unconverged_note(run_dir: Path, task: TaskName) -> str:
    """The note a `not_converged` job carries."""
    stop = s2_stop(run_dir, task) or {}
    k = "" if stop.get("k") is None else f" ({stop['k']} counted seeds)"
    exit_ = run_dir / task / "S2" / S2_EXIT
    if exit_.is_file() and json.loads(exit_.read_bytes())["s2_exit"] == "none":
        return (
            f"S2 stopped at its seed cap{k} with no first-try success: no Memory to evaluate, "
            "it counts 0 in P3 (p3.unconverged best_memory)"
        )
    return (
        f"S2 stopped at its seed cap{k} without converging: no P3 runs (p3.unconverged stop), "
        "P3 blocks done before do not count"
    )


def practice_values(config: OpsConfig, key: RunKey) -> dict[str, Any]:
    """A variant's S2 parameters, with its task's demo package when its `demo_context` shows the
    demonstration (and P3 then shows it at the same level), and as `done_goal` (`check_done`
    compares with its final state) unless the config's `done_goal` is off."""
    values = config.protocol("practice", key.task, key.variant)
    if values.get("demo_context", "none") != "none":
        values["demo"] = str(config.demo_of(key.task))
    if config.done_goal:
        values["done_goal"] = str(config.demo_of(key.task))
    return values


def imitate_values(config: OpsConfig, task: TaskName) -> dict[str, Any]:
    """The `imitate` parameters the task's S1 runs (resumes) with over the recipe's: the config's,
    with its demo package and the fallback switch and rule."""
    values = config.protocol("imitate", task)
    values["demo"] = str(config.demo_of(task))
    values["resume"] = True
    values["fallback"] = config.s1_fallback
    values["fallback_rule"] = config.s1_fallback_rule
    return values
