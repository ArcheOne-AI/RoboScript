"""The worker side of dispatch: `python -m roboscript.orchestrate.dispatch.agent <command>`, run by
the coordinator over the transport in the worker's code directory, with one JSON request on stdin
(`roboscript.contracts.dispatch`) and one `AgentReply` on stdout. It is shipped with the code
copy, so a worker needs nothing installed besides it.

- `start`: check the code version, config provenance and pinned files; write the unit's config
  once; launch the unit detached (a session of its own), its jobs pointed at the worker's
  gateway URL (`RS_GATEWAY_URL`) if the dispatch names one, and record its exact pid and start
  token in `.dispatch/unit.json`; a unit whose process is alive is left as it is.
- `status`: whether that exact process is alive, and the newest of the unit's status files.
- `stop`: SIGTERM to that exact process (never a pattern), then, after the grace, SIGKILL to it
  and its descendants (the pipeline stops what it started on SIGTERM and stays resumable).
- `manifest`, `remove`, `purge`, `digest`: the move of a finished task and the drain of a unit.

The agent writes and deletes only inside the unit directory, which must lie in the worker's run
root (`UnitPaths`). It never reads a token: the unit reads the worker's `.env` itself."""

import contextlib
import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ValidationError

from roboscript.contracts.dispatch import (
    AgentReply,
    DigestRequest,
    ManifestRequest,
    PurgeRequest,
    RemoveReply,
    RemoveRequest,
    StartReply,
    StartRequest,
    StatusReply,
    StatusRequest,
    StopReply,
    StopRequest,
    UnitPaths,
    UnitRecord,
)
from roboscript.orchestrate.dispatch.config import provenance
from roboscript.orchestrate.dispatch.files import (
    blob_path,
    manifest,
    named_blobs,
    path_digest,
    tree_files,
)
from roboscript.records import write_atomic

RECORD: Final = ".dispatch/unit.json"
LAUNCH_LOG: Final = ".dispatch/launch.log"
OWN: Final = ".dispatch"


class Refused(Exception):
    """A request the agent will not carry out (wrong version, config, path, state)."""


def code_sha(root: Path) -> tuple[str, bool]:
    """The code copy's revision and whether it is dirty: a git checkout's HEAD (dirty when a
    tracked file changed), else the copy's `.roboscript/source.sha` (`-dirty` suffix)."""
    if (root / ".git").exists():
        head = _git(root, "rev-parse", "HEAD")
        changed = _git(root, "status", "--porcelain", "--untracked-files=no")
        return head, bool(changed)
    recorded = root / ".roboscript" / "source.sha"
    if not recorded.is_file():
        raise Refused(
            f"{root} is neither a git checkout nor a synced copy (.roboscript/source.sha)"
        )
    sha = recorded.read_text(encoding="utf-8").strip()
    return sha.removesuffix("-dirty"), sha.endswith("-dirty")


def sha_matches(have: str, frozen: str) -> bool:
    """One revision written in full or as a prefix (at least 7 hex digits)."""
    return min(len(have), len(frozen)) >= 7 and (have.startswith(frozen) or frozen.startswith(have))


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise Refused(f"git {' '.join(args)} in {root}: {done.stderr.strip()}")
    return done.stdout.strip()


def start_token(pid: int) -> str | None:
    """What tells this process from a later one with its pid (its start time), or None when no
    live process (zombies count as dead) has the pid."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.parent.parent.joinpath("self").exists():  # Linux
        try:
            fields = stat.read_text(encoding="utf-8").rpartition(")")[2].split()
        except (FileNotFoundError, ProcessLookupError):
            return None
        return None if fields[0] in ("Z", "X") else fields[19]
    done = subprocess.run(
        ["ps", "-o", "stat=", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True
    )
    line = done.stdout.strip()
    if done.returncode != 0 or not line or line.startswith("Z"):
        return None
    return line.split(None, 1)[1]


def _record(unit: UnitPaths) -> UnitRecord | None:
    path = Path(unit.unit_dir) / RECORD
    return UnitRecord.model_validate_json(path.read_bytes()) if path.is_file() else None


def _alive(record: UnitRecord | None) -> bool:
    return record is not None and start_token(record.pid) == record.start_token


def start(request: StartRequest, root: Path) -> StartReply:
    sha, dirty = code_sha(root)
    if dirty or not sha_matches(sha, request.sha):
        state = "a dirty tree at " if dirty else ""
        raise Refused(f"{root} runs {state}{sha}, not the dispatch's {request.sha}")
    if provenance(request.config, request.local_keys) != request.provenance:
        raise Refused("the config's provenance does not match the dispatch's (sent wrong)")
    for name, expected in request.pinned.items():
        if (found := path_digest(root / name)) != expected:
            raise Refused(f"{name}: {'missing' if found is None else 'differs from the dispatch'}")
    unit_dir = Path(request.unit.unit_dir)
    record = _record(request.unit)
    if record is not None:
        if record.provenance != request.provenance or not sha_matches(record.sha, request.sha):
            raise Refused(f"{unit_dir} ran another version or config: start a new unit")
        if sorted(record.tasks) != sorted(request.tasks):
            raise Refused(f"{unit_dir} runs tasks {record.tasks}, not {request.tasks}")
        if _alive(record):
            return StartReply(pid=record.pid, started=False, launches=record.launches)
    unit_dir.mkdir(parents=True, exist_ok=True)
    config = unit_dir / request.config_file
    if not config.is_file():
        write_atomic(config, yaml.safe_dump(request.config, sort_keys=False).encode())
    argv = [part.replace("{dir}", str(unit_dir)) for part in request.argv]
    env = {**os.environ, **request.env, "RS_SOURCE_SHA": sha}
    if request.gateway_url is not None:  # its jobs call this gateway (`gateway_url`)
        env["RS_GATEWAY_URL"] = request.gateway_url
    log = unit_dir / LAUNCH_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        out.write(f"$ {' '.join(argv)}  # {_utc()}\n".encode())
        out.flush()
        process = subprocess.Popen(
            argv,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # outlives the ssh session; stopped by exact pid only
        )
    token = start_token(process.pid)
    if token is None:
        raise Refused(f"{argv[0]} exited at once; see {log}")
    launches = 1 if record is None else record.launches + 1
    new = UnitRecord(
        pid=process.pid,
        start_token=token,
        sha=sha,
        provenance=request.provenance,
        tasks=request.tasks,
        launches=launches,
        launched_at=_utc(),
    )
    write_atomic(unit_dir / RECORD, (new.model_dump_json(indent=1) + "\n").encode())
    return StartReply(pid=process.pid, started=True, launches=launches)


def status(request: StatusRequest) -> StatusReply:
    unit_dir = Path(request.unit.unit_dir)
    record = _record(request.unit)
    found = [unit_dir / name for name in request.status_files if (unit_dir / name).is_file()]
    newest = max(found, key=lambda p: p.stat().st_mtime, default=None)
    data: dict[str, Any] = json.loads(newest.read_bytes()) if newest is not None else {}
    return StatusReply(
        exists=unit_dir.is_dir(),
        alive=_alive(record),
        pid=None if record is None else record.pid,
        launches=0 if record is None else record.launches,
        rows=data.get("rows", []),
        updated=data.get("updated"),
    )


def stop(request: StopRequest) -> StopReply:
    record = _record(request.unit)
    if not _alive(record):
        return StopReply(pid=None if record is None else record.pid, was_alive=False, killed=False)
    assert record is not None
    os.kill(record.pid, signal.SIGTERM)
    deadline = time.monotonic() + request.grace_s
    while time.monotonic() < deadline:
        if not _alive(record):
            return StopReply(pid=record.pid, was_alive=True, killed=False)
        time.sleep(0.5)
    for pid in [*_descendants(record.pid), record.pid]:
        with contextlib.suppress(ProcessLookupError):  # ended on its own meanwhile
            os.kill(pid, signal.SIGKILL)
    return StopReply(pid=record.pid, was_alive=True, killed=True)


def _descendants(pid: int) -> list[int]:
    """`pid`'s descendants, deepest first."""
    done = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="], capture_output=True, text=True)
    children: dict[int, list[int]] = {}
    for line in done.stdout.splitlines():
        child, parent = map(int, line.split())
        children.setdefault(parent, []).append(child)
    found: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        found.append(child)
        stack.extend(children.get(child, []))
    return found[::-1]


def manifest_of(request: ManifestRequest) -> dict[str, Any]:
    unit_dir = Path(request.unit.unit_dir)
    blobs = unit_dir / request.blobs if request.blobs is not None else None
    found = manifest(unit_dir / request.path if request.path else unit_dir, request.exclude, blobs)
    return found.model_dump(mode="json")


def remove(request: RemoveRequest) -> RemoveReply:
    unit_dir = Path(request.unit.unit_dir)
    task_dir = unit_dir / request.task
    blobs = unit_dir / request.blobs if request.blobs is not None else None
    found = manifest(task_dir, (), blobs)
    if found.digest != request.digest:
        raise Refused(f"{task_dir} changed since its manifest: move it again")
    removed = 0
    for name in found.files:
        if not any(fnmatch.fnmatch(name, pattern) for pattern in request.keep):
            (task_dir / name).unlink()
            removed += 1
    for directory in sorted(task_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    gone = 0
    if blobs is not None and request.blob_min_bytes is not None and found.blobs:
        skip = (request.task, request.blobs, OWN)
        others = [p for p in unit_dir.iterdir() if p.is_dir() and p.name not in skip]
        still = set().union(*(named_blobs(p, tree_files(p), blobs) for p in others))
        for blob in found.blobs:
            path = blob_path(blobs, blob)
            large = path.is_file() and path.stat().st_size >= request.blob_min_bytes
            if blob not in still and large:
                path.unlink()
                gone += 1
    return RemoveReply(files=removed, blobs=gone)


def purge(request: PurgeRequest) -> dict[str, Any]:
    unit_dir = Path(request.unit.unit_dir)
    if _alive(_record(request.unit)):
        raise Refused(f"{unit_dir} still runs: stop it first")
    if unit_dir.is_dir():
        shutil.rmtree(unit_dir)
    if unit_dir.parent != Path(request.unit.run_root):
        with contextlib.suppress(OSError):  # the dispatch's directory, once its last unit drained
            unit_dir.parent.rmdir()
    return {"purged": str(unit_dir)}


def digests(request: DigestRequest, root: Path) -> dict[str, Any]:
    return {name: path_digest(root / name) for name in request.paths}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


COMMANDS: Final[dict[str, tuple[type[BaseModel], Callable[..., Any]]]] = {
    "start": (StartRequest, start),
    "status": (StatusRequest, status),
    "stop": (StopRequest, stop),
    "manifest": (ManifestRequest, manifest_of),
    "remove": (RemoveRequest, remove),
    "purge": (PurgeRequest, purge),
    "digest": (DigestRequest, digests),
}
NEEDS_ROOT: Final = frozenset({"start", "digest"})


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    if len(given) != 1 or given[0] not in COMMANDS:
        print(f"usage: agent {{{','.join(COMMANDS)}}} < request.json", file=sys.stderr)
        return 2
    model, handler = COMMANDS[given[0]]
    try:
        request = model.model_validate_json(sys.stdin.buffer.read())
        args = (request, Path.cwd()) if given[0] in NEEDS_ROOT else (request,)
        result = handler(*args)
        payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        reply = AgentReply(ok=True, reply=payload)
    except (Refused, ValidationError) as error:
        reply = AgentReply(ok=False, kind="refused", error=str(error))
    except Exception as error:  # reported to the coordinator, which decides
        reply = AgentReply(ok=False, kind="failed", error=f"{type(error).__name__}: {error}")
    sys.stdout.write(reply.model_dump_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
