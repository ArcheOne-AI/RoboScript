"""wicl's job bodies by phase (`Wicl.kinds`, `roboscript.contracts.jobs.JobKind`), as a
pipeline's worker runs them (`roboscript.orchestrate.worker`), on what they need of the pipeline
(`LearnHost`).

- `imitate` (`ImitateJob`, stage `s1`): the task's S1, resumed where it was cut short, or its
  result once it has exited. `done` once it exits, or, with `s1_fallback`, once a fallback was
  found (the job writes `ops.FALLBACK`), else `not_passed`.
- `practice` (`PracticeJob`, stage `s2`): the unit gets its task's S1 (`ops.seed_variant`), then
  S2, resumed at its ledger position once it has begun (a stopped one returns its result), from
  the version its task's S1 fallback named, if any; it must freeze the Memory. Then
  the pipeline's own step after S2 (`LearnHost.after_practice`), and `not_converged` when the unit
  may not go on to its evaluation (`LearnHost.gated`, with `ops.unconverged_note`), `queued` at
  the evaluation's stage when the pipeline evaluates, else `done`."""

import json
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Final, Protocol

from roboscript.contracts.ids import TaskName
from roboscript.contracts.jobs import JobEnd, JobRequest
from roboscript.learning import ProtocolRunner
from roboscript.memory import FileMemoryStore
from roboscript.orchestrate.jobs import RunKey
from roboscript_methods.wicl.imitate import EXIT, Imitate, ImitateResult
from roboscript_methods.wicl.ops import (
    FALLBACK,
    S1,
    OpsConfig,
    imitate_values,
    practice_values,
    seed_variant,
    unconverged_note,
    write_fallback,
)
from roboscript_methods.wicl.practice import LEDGER, START, Practice, PracticeResult

DONE: Final = JobEnd(status="done")


class LearnHost(Protocol):
    """What wicl's job bodies need of the pipeline that runs them: its config (`config`), a job's
    runner on an env server of its own (`serve`, under the job's
    `made`), its step after S2 (`after_practice`), whether a unit may not go on to its evaluation
    (`gated`) and the stage a unit's evaluation records (`evaluation_stage`; None: the launch
    evaluates nothing)."""

    config: OpsConfig
    evaluation_stage: str | None

    def serve(
        self, request: JobRequest, made: dict[str, Any] | None
    ) -> AbstractAsyncContextManager[ProtocolRunner]: ...

    async def after_practice(self, runner: ProtocolRunner, key: RunKey) -> None: ...

    def gated(self, run_dir: Path, task: TaskName) -> bool: ...


class ImitateJob:
    """S1 (`imitate`)."""

    stage = "s1"

    async def run(
        self, host: LearnHost, request: JobRequest, made: dict[str, Any] | None
    ) -> JobEnd:
        key = RunKey.of(request)
        run_dir = key.unit_dir(request.directory)
        async with host.serve(request, made) as runner:
            imitated = await imitate(host.config, runner, key, run_dir)
            if imitated.passed:
                return DONE
            exits = sorted((run_dir / key.task / "S1").glob(f"*/{EXIT}"))
            found = json.loads(exits[-1].read_bytes()).get("fallback") if exits else None
            if host.config.s1_fallback and found is not None:
                write_fallback(run_dir, key.task, found)
                return DONE
            return JobEnd(status="not_passed")


class PracticeJob:
    """S2 (`practice`)."""

    stage = "s2"

    async def run(
        self, host: LearnHost, request: JobRequest, made: dict[str, Any] | None
    ) -> JobEnd:
        key = RunKey.of(request)
        run_dir = key.unit_dir(request.directory)
        seed_variant(request.directory / key.task / S1, run_dir, key.task)
        async with host.serve(request, made) as runner:
            practiced = await practice(host.config, runner, key, run_dir)
            if await FileMemoryStore(run_dir).frozen(key.task) is None:
                raise RuntimeError(f"S2 stopped ({practiced.reason}) without freezing the Memory")
            await host.after_practice(runner, key)
        if host.gated(run_dir, key.task):  # no Memory to evaluate, or it is held back
            return JobEnd(status="not_converged", note=unconverged_note(run_dir, key.task))
        if host.evaluation_stage is not None:
            return JobEnd(status="queued", stage=host.evaluation_stage)
        return DONE


async def imitate(
    config: OpsConfig, runner: ProtocolRunner, key: RunKey, run_dir: Path
) -> ImitateResult:
    """S1, resumed where it was cut short, or its result once it has exited."""
    folder = run_dir / key.task / "S1"
    if exits := sorted(folder.glob(f"*/{EXIT}")):
        return ImitateResult.model_validate_json(exits[-1].read_bytes())
    values = imitate_values(config, key.task)
    return ImitateResult.model_validate(await runner.run(Imitate(), key.task, values))


async def practice(
    config: OpsConfig, runner: ProtocolRunner, key: RunKey, run_dir: Path
) -> PracticeResult:
    """S2, resumed at its ledger position once it has begun (a stopped practice returns its
    result); from the version its task's S1 fallback named (`ops.FALLBACK`), if any."""
    folder = run_dir / key.task / "S2"
    values = practice_values(config, key)
    values["resume"] = (folder / START).is_file() or (folder / LEDGER).is_file()
    if (fell := run_dir.parent / S1 / FALLBACK).is_file():
        values["start"] = json.loads(fell.read_bytes())["hash"]
    return PracticeResult.model_validate(await runner.run(Practice(), key.task, values))
