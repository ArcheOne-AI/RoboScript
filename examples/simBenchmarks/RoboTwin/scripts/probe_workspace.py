"""Probe aloha_agilex's reach with the cuRobo planner's IK and store the workspace table
(roboscript.manipulation.workspace) that the executor's pre-check reads.

Run from the repository root in the simulator env, on the GPU CUDA_VISIBLE_DEVICES names:

    python examples/simBenchmarks/RoboTwin/scripts/probe_workspace.py --out
    data/workspace/aloha_agilex.json

It starts one motion service (configs/motion.example.yaml) through the stack supervisor, probes
TCP positions on a grid over the table (by default 3 cm cells from (-0.6, -0.5, 0.74) to
(0.6, 0.4, 1.3), world) with a few gripper orientations each, and writes the table; set
`workspace:` in the executor's config to its path.
"""

import argparse
import sys
from pathlib import Path

import anyio
import httpx

from roboscript.contracts.motion import Planner
from roboscript.manipulation import probe
from roboscript.rpc import client
from roboscript.settings import repo_root
from roboscript.stack import Interpreters, ServiceSpec, StackConfig, up
from roboscript_robotwin.benchmark import ROBOTWIN


async def run(
    out: Path, low: tuple[float, float, float], high: tuple[float, float, float], cell: float
) -> None:
    root = repo_root()
    python = Path(sys.executable)
    stack = StackConfig(
        interpreters=Interpreters(agent=python, sim=python),
        sim_pythonpath=[root / "packages/core/src", *ROBOTWIN.pythonpath],
        run_dir=out.parent / "probe",
        services=[
            ServiceSpec(
                name="motion",
                module="roboscript.motion",
                interpreter="sim",
                args=[
                    "--config",
                    str(root / "examples/simBenchmarks/RoboTwin/configs/motion.example.yaml"),
                ],
                ready_timeout_s=300,
            )
        ],
    )
    profile = ROBOTWIN.profile()
    seed = {arm: [0.0] * len(spec.joints) for arm, spec in profile.arms.items()}
    async with up(stack) as running, httpx.AsyncClient(trust_env=False) as http:
        planner = client(Planner, running.endpoints["motion"], token=None, http=http, timeout=300)
        # type: ignore[type-abstract]
        table = await probe(planner, profile, low, high, cell, seed)
    table.save(out)
    reached = {arm: len(cells) for arm, cells in table.reachable.items()}
    print(f"{out}: {table.shape} cells of {cell} m, reachable {reached}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--low", type=float, nargs=3, default=(-0.6, -0.5, 0.74))
    parser.add_argument("--high", type=float, nargs=3, default=(0.6, 0.4, 1.3))
    parser.add_argument("--cell", type=float, default=0.03)
    options = parser.parse_args()
    out = options.out if options.out.is_absolute() else repo_root() / options.out
    anyio.run(run, out, tuple(options.low), tuple(options.high), options.cell)


if __name__ == "__main__":
    main()
