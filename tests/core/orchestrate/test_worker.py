"""The generic worker (`roboscript.orchestrate.worker`): a job run by its kind from the command
line, recording how it ended (its status, its stage from then on, the note)."""

from pathlib import Path
from typing import Any

import anyio
import pytest

from roboscript.contracts.ids import TaskName
from roboscript.contracts.jobs import JobEnd, JobRequest
from roboscript.contracts.provenance import SOURCE_ENV
from roboscript.orchestrate import worker
from roboscript.orchestrate.jobs import Failure, RunKey, RunState


class Host:
    """A pipeline whose jobs keep their records in `<root>/<task>/<unit>[/<lane>]`."""

    heartbeat_s = 0.01

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ran: list[JobRequest] = []

    def job_dir(self, key: RunKey) -> Path:
        unit = self.root / key.task / key.variant
        return unit if key.lane is None else unit / key.lane

    def definition(self, key: RunKey) -> dict[str, Any]:
        return {"made": key.name}

    def failure(self, error: BaseException) -> Failure:
        return "infra" if isinstance(error, ConnectionError) else "error"


class Learn:
    stage = "learn"

    async def run(self, host: Host, request: JobRequest, made: dict[str, Any] | None) -> JobEnd:
        host.ran.append(request)
        await anyio.sleep(0.05)  # long enough to beat its heartbeat
        assert made == {"made": f"{request.task}/{request.unit}"}
        return JobEnd(status="queued", stage="check", note="evaluate next")


class Broken:
    stage = "check"

    async def run(self, host: Host, request: JobRequest, made: dict[str, Any] | None) -> JobEnd:
        raise ConnectionError("the service went away\nmore")


KINDS = {"learn": Learn(), "check": Broken()}


def test_a_job_runs_by_its_kind_and_records_its_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SOURCE_ENV, "abc1234")
    host = Host(tmp_path)
    argv = worker.command(["w"], tmp_path, "learn", RunKey(TaskName("red"), "default"), 3, {})[1:]
    assert worker.main(KINDS, ["spare"], lambda request: host, argv, prog="worker") == 0
    [request] = host.ran
    assert (request.task, request.unit, request.gpu, request.modes) == ("red", "default", 3, None)
    assert RunKey.of(request) == RunKey(TaskName("red"), "default")
    assert RunKey.of(request).unit_dir(tmp_path) == tmp_path / "red" / "default"
    state = RunState.load(tmp_path / "red" / "default", RunKey(TaskName("red"), "default"))
    assert (state.status, state.stage, state.error, state.launches) == (
        "queued",
        "check",
        "evaluate next",
        1,
    )
    assert (tmp_path / "red" / "default" / ".roboscript" / "heartbeat").is_file()
