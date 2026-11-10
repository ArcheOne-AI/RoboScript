"""A stand-in for the cuRobo planner service in launcher tests: MockPlanner behind the same
`Planner` RPC, started as the service is (`--config`, `--ready-file`, `--port`)."""

import argparse
from functools import partial
from pathlib import Path

import anyio

from roboscript.motion import MockPlanner
from roboscript.motion.service import run


class WarmMockPlanner(MockPlanner):
    """MockPlanner as a served planner: nothing to warm up, and it never fails for good."""

    @property
    def ready(self) -> bool:
        return True

    @property
    def failure(self) -> str | None:
        return None

    async def start(self) -> None:
        """Nothing to warm up."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    anyio.run(partial(run, WarmMockPlanner(), port=args.port, ready_file=args.ready_file))


if __name__ == "__main__":
    main()
