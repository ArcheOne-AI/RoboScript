"""RoboTwin's evaluation job body (`P3Jobs.kinds`, `roboscript.contracts.jobs.JobKind`), as a
pipeline's worker runs it (`roboscript.orchestrate.worker`), on what it needs of the pipeline
(`EvalHost`).

`evaluate` (`EvaluateJob`, stage `p3`): the unit's P3 blocks of the modes the job was given (the
settings' by default), through the pipeline (`EvalHost.evaluate`). A job that leaves blocks of the
settings' modes to run (the clean ones before spare randomized blocks) ends `queued`, so that a
launch again takes them up; a lane is done with its own blocks (the unit's job, only once the
lane's are done too)."""

from collections.abc import Sequence
from typing import Any, Protocol, cast

from roboscript.contracts.jobs import JobEnd, JobRequest
from roboscript.orchestrate.jobs import RunKey
from roboscript_robotwin.eval.blocks import P3Settings, owned_modes, p3_pending
from roboscript_robotwin.eval.launch import Mode


class EvalHost(Protocol):
    """What RoboTwin's evaluation job needs of the pipeline that runs it: its P3 settings (`p3`)
    and `evaluate`, which runs a job's blocks of `modes` under the job's `made`."""

    p3: P3Settings

    async def evaluate(
        self, request: JobRequest, modes: Sequence[Mode], made: dict[str, Any] | None
    ) -> None: ...


class EvaluateJob:
    """P3 (`evaluate`)."""

    stage = "p3"

    async def run(self, host: EvalHost, request: JobRequest, made: dict[str, Any] | None) -> JobEnd:
        key = RunKey.of(request)
        run_dir = key.unit_dir(request.directory)
        mine = cast(list[Mode], request.modes) or host.p3.modes
        await host.evaluate(request, mine, made)
        owned = owned_modes(host.p3, key) if key.lane is not None else None
        return JobEnd(status="queued" if p3_pending(host.p3, run_dir, owned) else "done")
