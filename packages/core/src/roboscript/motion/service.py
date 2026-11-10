"""The planner service: `python -m roboscript.motion --config <file or name> [--planner NAME]
[--ready-file PATH]`.

`--planner` is a planner registered in `roboscript.planners` (where the core is installed) or
an import path `package.module:Class` (the simulator env, where it is only on PYTHONPATH and
nothing is registered); the default is cuRobo's by import path. A servable planner class has a
`Config` model, read from `--config`, and is built from it.

It serves `Planner` over the typed RPC on 127.0.0.1 (port 0 unless `--port`) and writes its ready
file as soon as it listens, then builds and warms up the planners; `/readyz` answers 200 once they
are warm, calls before that fail as `not_ready`; after an error on the planner thread that may
have left the planner's state inconsistent (its `failure`) the service stops serving and exits
(`rpc.ServiceFault`), so that its supervisor restarts it. It runs in the simulator env on the GPU
that `CUDA_VISIBLE_DEVICES` names and serves any number of episodes, one call at a time (clients:
`contracts.motion.TIMEOUTS`). Requests carry all they need, but cuRobo's seed state is shared: each
plan or IK call advances it for every episode, so a plan depends on the calls before it.
"""

import argparse
import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Final, Protocol

import anyio
from pydantic import BaseModel

from roboscript.contracts.motion import Planner
from roboscript.kernel import entrypoints
from roboscript.rpc import make_app, routes, serve_while
from roboscript.settings import load_config_ref

GROUP: Final = "roboscript.planners"
CUROBO: Final = "roboscript.motion.curobo:CuroboPlanner"
"""cuRobo's planner by import path: what the service serves by default, in any env."""


def load_planner(ref: str) -> type[Any]:
    """The planner class `ref` names: `package.module:Class`, else a `roboscript.planners`
    name. A servable one has a `Config` model."""
    found = entrypoints.load_ref(ref) if ":" in ref else entrypoints.load(GROUP, ref)
    config = getattr(found, "Config", None)
    if not (isinstance(config, type) and issubclass(config, BaseModel)):
        raise TypeError(f"planner {ref} has no Config model: it cannot be served from a config")
    return found  # type: ignore[no-any-return]


class WarmPlanner(Planner, Protocol):
    """A planner that must warm up before it serves; `failure` names what failed for good (None
    while it is sound)."""

    @property
    def ready(self) -> bool: ...

    @property
    def failure(self) -> str | None: ...

    async def start(self) -> None: ...


async def run(planner: WarmPlanner, *, port: int = 0, ready_file: Path | None = None) -> None:
    """Serve `planner` until SIGINT or SIGTERM, warming it up meanwhile, or until it fails for
    good (`rpc.ServiceFault`)."""
    if not hasattr(planner, "failure"):  # checked here, not first after a warm-up of minutes
        name = type(planner).__name__
        raise TypeError(f"{name} has no `failure`: a served planner is a WarmPlanner")
    app = make_app(routes(Planner, planner, token=None), ready=lambda: planner.ready)
    await serve_while(
        app,
        planner.start,
        name="the planner",
        port=port,
        ready_file=ready_file,
        fault=lambda: planner.failure,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m roboscript.motion", description=__doc__)
    parser.add_argument("--config", required=True, help="a YAML file or a name under configs/")
    parser.add_argument("--planner", default=CUROBO, help="a registered name or module:Class")
    parser.add_argument("--ready-file", type=Path, help="written once the service listens")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    planner_class = load_planner(args.planner)
    planner = planner_class(load_config_ref(args.config, planner_class.Config))
    anyio.run(partial(run, planner, port=args.port, ready_file=args.ready_file))
