"""One pipeline job in a process of its own, so that a crash, a kill for memory or a blocked call
ends only that job: the scheduler runs `<wiring.worker> <directory> <phase> <task> <variant>
<gpu> <services JSON> [--modes <P3 modes>] [--lane <lane>]`
(`robotwin_pipeline.schedule`), whose script calls `main` with the same wiring. The job
records its end in its `pipeline.json` (`roboscript.orchestrate.worker.run_job`); the exit
status is 0 whenever it did."""

import os
from collections.abc import Sequence
from pathlib import Path

from roboscript.contracts.jobs import JobRequest
from roboscript.orchestrate import worker
from roboscript.orchestrate.jobs import RunKey
from roboscript.records.replay import TapeConfig
from roboscript_robotwin.eval.blocks import LANE_DIRS
from robotwin_pipeline.config import ReplaySettings, load_directory
from robotwin_pipeline.run import BLOBS, Host, Wiring, evaluation_factory, job_dir, method


def tape_environ(
    replay: ReplaySettings, directory: Path, key: RunKey, phase: str
) -> dict[str, str]:
    """The environment that switches a job's replay tape on: its scope is its directory under
    the pipeline's and its phase, the run root and the code SHA are volatile (they differ between
    the recorded run and its replay), the masks go to the pipeline's blob store."""
    scope = f"{job_dir(directory, key).relative_to(directory).as_posix()}/{phase}"
    volatile = {"run_root": str(directory.resolve())}
    if sha := os.environ.get("RS_SOURCE_SHA"):
        volatile["source_sha"] = sha
    tape = TapeConfig(
        mode=replay.mode,
        directory=replay.directory,
        source=replay.source,
        scope=scope,
        blob_dir=directory.resolve() / BLOBS,
        volatile=volatile,
    )
    return tape.environ()


def main(wiring: Wiring, argv: Sequence[str]) -> int:
    """Run the job `argv` names (`roboscript.orchestrate.worker.command`) by its phase: wicl's
    `imitate` and `practice` (`METHOD`'s kinds), RoboTwin's `evaluate` (`EVALUATION`'s), on the
    pipeline (`run.Host`) of its directory."""
    kinds = worker.job_kinds(method().kinds, evaluation_factory().kinds)

    def host(request: JobRequest) -> Host:
        config = load_directory(request.directory)
        if config.replay is not None:  # before anything tapes: the job's processes inherit it
            key = RunKey.of(request)
            os.environ.update(tape_environ(config.replay, request.directory, key, request.phase))
        return Host(config, wiring, request.directory)

    return worker.main(
        kinds, sorted(LANE_DIRS), host, argv, prog="pipeline worker", description=__doc__
    )
