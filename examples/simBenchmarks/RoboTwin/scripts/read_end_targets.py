"""Read the demonstration's end targets for a pipeline's variants (`learning.practice
.read_end_targets`) without restarting them: for each task, its variant's Memory as it stands (the
frozen version, else its latest) identified on the demo package's recording, then
`read_end_targets`, into `<run>/<task>/<variant>/<task>/S2/end_targets.json`. A running S2 picks
the file up when it resumes; P3 reads it (`robotwin_pipeline.run.p3_demo`). No env and no brain:
only the perception toolsets, planning with a running motion service and cross-checking with a
running vision service (no GPU process of its own), configured as the run's (`--configs`: the
checkout whose `configs/scene.yaml` it read). A file under the Memory's roles stays unless
`--overwrite`. Nothing else of the run is written: the readings' events go to `--records`; with
`--out` the files go there too (`<out>/<task>/end_targets.json`), read fresh, to compare before
overwriting.

    PYTHONPATH=examples/simBenchmarks/RoboTwin .venv/bin/python
    examples/simBenchmarks/RoboTwin/scripts/read_end_targets.py \\
        --run runs/<name>/<stamp> --motion http://127.0.0.1:<p1> --vision http://127.0.0.1:<p2> \\
        [--configs <run's checkout>] [--task T ...] [--variant default] [--overwrite] [--jobs 4] \\
        [--dry-run]

Without --task: every task whose variant has an S2 and no end targets under its Memory's roles.
Run from the repository root in the agent venv, with RS_GATEWAY_TOKEN set.
"""

import argparse
import functools
import json
import logging
import sys
from pathlib import Path

import anyio
from robotwin_pipeline import load_directory
from robotwin_pipeline.run import BENCHMARKS, BLOBS

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.ids import TaskName
from roboscript.gateway import openai_client
from roboscript.gateway.client import gateway_url
from roboscript.learning import LearningConfig, ProtocolRunner
from roboscript.learning.demo import import_blobs, load_package
from roboscript.memory import FileMemoryStore, MemoryDoc, object_roles, read_doc
from roboscript.perception.config import PerceptionConfig
from roboscript.settings import load_config, runs_path
from roboscript.tools import LocalProvider
from roboscript.toolsets.geometry import Geometry
from roboscript.toolsets.perception import Perception
from roboscript.toolsets.scene import SceneConfig
from roboscript.toolsets.worldstate import WorldState
from roboscript_methods.wicl.extension import END_TARGETS
from roboscript_methods.wicl.ops import last_version
from roboscript_methods.wicl.practice import read_end_targets
from roboscript_methods.wicl.toolsets.done import Done
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.config import with_perception_binding

log = logging.getLogger("read_end_targets")


def scene_config(root: Path | None) -> SceneConfig:
    """The scene toolsets' config: `configs/scene.yaml` under `root` (the repository's), else the
    defaults, as the run's toolsets read it."""
    try:
        scene = load_config("scene", SceneConfig, root=root)
    except FileNotFoundError:
        scene = SceneConfig()
    pipeline = with_perception_binding(scene.pipeline.model_dump(exclude_unset=True))
    return scene.model_copy(update={"pipeline": PerceptionConfig.model_validate(pipeline)})


def memory_doc(run_dir: Path, task: str) -> MemoryDoc:
    """The variant's Memory as it stands: its frozen version, else its latest."""
    found = last_version(run_dir, TaskName(task))
    assert found is not None
    return read_doc(FileMemoryStore(run_dir).folder(task) / f"v{found[0]:03d}")


def pending(directory: Path, tasks: list[str], variant: str, overwrite: bool) -> list[str]:
    """The tasks whose variant has a Memory and an S2, and no end targets under that Memory's
    roles (all of them with `overwrite`)."""
    found = []
    for task in tasks:
        run_dir = directory / task / variant
        if last_version(run_dir, TaskName(task)) is None or not (run_dir / task / "S2").is_dir():
            continue
        path = run_dir / task / "S2" / END_TARGETS
        if overwrite or not path.is_file():
            found.append(task)
            continue
        doc = memory_doc(run_dir, task)
        if json.loads(path.read_bytes()).get("roles") != sorted(object_roles(doc)):
            found.append(task)
    return found


async def run(args: argparse.Namespace) -> int:
    directory = runs_path(args.run).resolve()
    config = load_directory(directory)
    fresh = args.overwrite or args.out is not None
    tasks = pending(directory, args.task or list(config.tasks), args.variant, fresh)
    print(f"to read: {', '.join(tasks) or 'nothing'}")
    if args.dry_run or not tasks:
        return 0
    blobs = LocalBlobStore(directory / BLOBS)  # the vision service's store too
    scene = scene_config(args.configs).model_copy(
        update={"planner": args.motion, "vision": args.vision}
    )
    providers = [
        LocalProvider(Perception(), scene, ROBOTWIN.skillsets),
        LocalProvider(Geometry(), scene, ROBOTWIN.skillsets),
        LocalProvider(WorldState(), scene, ROBOTWIN.skillsets),
        LocalProvider(Done(), scene, ROBOTWIN.skillsets),
    ]
    # the readings' own events go beside the run, never into its files
    records = args.records or directory.with_name(f"{directory.name}-read_end_targets")
    records.mkdir(parents=True, exist_ok=True)
    learning = LearningConfig.model_validate({**config.learning, "run_dir": records})
    runner = ProtocolRunner(
        learning,
        providers=providers,
        blobs=blobs,
        controls={},
        embodiment=BENCHMARKS[config.benchmark][0].embodiment,
        models=functools.partial(openai_client, base_url=gateway_url(args.configs)),
    )
    failed = 0
    limit = anyio.Semaphore(args.jobs)

    async def one(task: str) -> None:
        nonlocal failed
        async with limit:
            run_dir = directory / task / args.variant
            version, _ = last_version(run_dir, TaskName(task)) or (0, "")
            doc = memory_doc(run_dir, task)
            path = config.demo_of(task)
            await import_blobs(path, blobs)
            targets, why = await read_end_targets(
                runner.driver,
                run_dir / task / "S2" if args.out is None else args.out / task,
                load_package(path),
                doc,
                attempts=args.attempts,
                timeout_s=args.timeout_s,
                overwrite=fresh,
            )
            if targets is None:
                failed += 1
                print(f"{task}: not read: {why}")
            else:
                print(f"{task}: {len(targets)} targets (v{version:03d})")

    async with runner, anyio.create_task_group() as group:
        for task in tasks:
            group.start_soon(one, task)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True, help="the pipeline directory")
    ap.add_argument("--motion", help="a running motion service's URL (the self-filter)")
    ap.add_argument("--vision", help="a running vision service's URL (the cross-check)")
    ap.add_argument("--task", action="append", help="only these tasks")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--overwrite", action="store_true", help="read again where a file stands")
    ap.add_argument("--jobs", type=int, default=4, help="readings at once")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--timeout-s", type=float, default=2700.0, help="per attempt")
    ap.add_argument(
        "--configs", type=Path, help="the checkout whose configs (scene, gateway) the run used"
    )
    ap.add_argument(
        "--records",
        type=Path,
        help="where the readings' events go (default: <run>-read_end_targets)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="write <out>/<task>/end_targets.json instead of into the variants (a comparison)",
    )
    ap.add_argument("--dry-run", action="store_true", help="list what would be read")
    args = ap.parse_args(argv)
    if not args.dry_run and (args.motion is None or args.vision is None):
        ap.error("--motion and --vision name the running services the reading uses")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return anyio.run(run, args)


if __name__ == "__main__":
    sys.exit(main())
