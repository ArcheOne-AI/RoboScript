"""`python -m roboscript.orchestrate.dispatch`: run a dispatch, resume it, read or stop it.

    ... run configs/dispatch.yaml [--set poll_s=30]   assign, start the units and run to the end
    ... resume <root>/<name>                          go on with a dispatch (its dispatch.yaml)
    ... status <root>/<name>                          print the last status table
    ... stop <root>/<name>                            stop every unit by its exact pid (resumable)

Run it detached on the coordinator machine, from its code copy (at the dispatch's sha):
`setsid nohup .venv/bin/python -m roboscript.orchestrate.dispatch run ... > dispatch.log 2>&1 &`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import anyio

from roboscript.orchestrate.dispatch.config import load_dispatch
from roboscript.orchestrate.dispatch.coordinator import Coordinator, status_table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["run", "resume", "status", "stop"])
    ap.add_argument("path", type=Path, help="a dispatch config (run) or a dispatch directory")
    ap.add_argument("--set", action="append", default=[], help="a config override key=value")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "status":  # read only: the running coordinator writes it
        print(status_table(json.loads((args.path / "status.json").read_bytes())))
        return 0
    path = args.path if args.command == "run" else args.path / "dispatch.yaml"
    coordinator = Coordinator(load_dispatch(path, args.set))
    if args.command in ("run", "resume"):
        found = anyio.run(coordinator.run)
        print(status_table(found))
        return 0 if set(found["tasks"]) <= {"moved"} else 1
    for line in anyio.run(coordinator.stop_all):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
