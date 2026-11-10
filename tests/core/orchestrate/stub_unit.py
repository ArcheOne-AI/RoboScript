"""A stand-in for a pipeline launch, as dispatch drives one: `stub_unit.py --resume <dir>` reads
`<dir>/pipeline.yaml` (its `tasks`, `gpus` and `stub` settings) and runs each task's jobs `S1`
then `default`, `stub.ticks` ticks of `stub.tick_s` each, writing the pipeline's directory shape:
`<task>/<job>/pipeline.json` (status), `runs.txt` (a line per execution), `events.jsonl` naming a
large blob of its own and a small one every task shares (in `blobs/`), and `status.json` (the
pipeline's `rows`). A resumed launch skips jobs whose `pipeline.json` says done (tombstones).
`STUB_TICK_S` in the environment overrides `stub.tick_s` (a slow worker); with `STUB_HOLD`
naming a file, the launch stays alive after its last job until the file is gone. `stub.crash_after`:
exit 1 once (marker `stub.crash_marker`) after that many ticks in all."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

JOBS = ("S1", "default")


def put_blob(blobs: Path, data: bytes) -> str:
    name = hashlib.sha256(data).hexdigest()
    path = blobs / name[:2] / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return name


def state(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "pipeline.json"
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"status": "queued", "ticks": 0}
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def rows(directory: Path, tasks: list[str], gpu: int) -> list[dict[str, Any]]:
    found = []
    for task in tasks:
        for job in JOBS:
            s = state(directory / task / job)
            found.append(
                {
                    "run": f"{task}/{job}",
                    "status": s["status"],
                    "stage": "s1" if job == "S1" else "s2",
                    "gpu": gpu,
                }
            )
    return found


def main() -> int:
    assert sys.argv[1] == "--resume", sys.argv
    directory = Path(sys.argv[2])
    config = yaml.safe_load((directory / "pipeline.yaml").read_text(encoding="utf-8"))
    stub = config.get("stub", {})
    tasks, gpu = config["tasks"], config["gpus"][0]
    blobs = directory / "blobs"
    shared = put_blob(blobs, b"shared")
    ticks = 0
    while True:
        pending = False
        for task in tasks:
            for job in JOBS:
                job_dir = directory / task / job
                s = state(job_dir)
                if s["status"] == "done":
                    continue
                pending = True
                if s["status"] != "running":
                    job_dir.mkdir(parents=True, exist_ok=True)
                    with (job_dir / "runs.txt").open("a") as out:
                        out.write(f"run by {directory}\n")
                s["status"], s["ticks"] = "running", s["ticks"] + 1
                big = put_blob(blobs, f"{task}/{job}/{s['ticks']}/".encode() * 20000)
                with (job_dir / "events.jsonl").open("a") as out:
                    out.write(
                        json.dumps({"tick": s["ticks"], "image": {"id": big}, "shared": shared})
                        + "\n"
                    )
                if s["ticks"] >= stub.get("ticks", 2):
                    s["status"] = "done"
                write_json(job_dir / "pipeline.json", s)
                break  # S1 before the variant
        write_json(
            directory / "status.json",
            {
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rows": rows(directory, tasks, gpu),
            },
        )
        if not pending:
            write_json(
                directory / "summary.json",
                {
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "rows": rows(directory, tasks, gpu),
                },
            )
            hold = os.environ.get("STUB_HOLD")
            while hold and Path(hold).exists():  # a test keeps it alive until it lets go
                time.sleep(0.05)
            return 0
        ticks += 1
        marker = stub.get("crash_marker")
        if marker and ticks >= stub.get("crash_after", 1) and not Path(marker).exists():
            Path(marker).write_text("crashed once", encoding="utf-8")
            return 1
        time.sleep(float(os.environ.get("STUB_TICK_S", stub.get("tick_s", 0.05))))


if __name__ == "__main__":
    raise SystemExit(main())
