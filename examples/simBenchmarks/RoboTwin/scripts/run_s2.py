"""Run S2 `practice` on RoboTwin tasks in parallel (docs/concepts/learning.md), Python API only.

One env server per task (the simulator env, round robin over --gpus), the cuRobo planner service
(configs/motion.yaml, on the first GPU) and, with --vision-python, the vision service
(configs/vision.yaml: GroundingDINO, SAM 3, on --vision-gpu, else the least loaded of
--gpu-pool, by default CUDA_VISIBLE_DEVICES) start under `roboscript.stack`; the model gateway is
the persistent one (configs/gateway.yaml, RS_GATEWAY_TOKEN). The toolsets mounted: sensing,
manipulation (its planner is always the stack's motion service; configs/manipulation.yaml, whose
table fallback is for the scenes practiced on, applies until perception estimates the table), and
the scene toolsets (perception, geometry, worldstate; configs/scene.yaml if present) with the
planner (the self-filter) and the vision service (the cross-check) filled in, so that every Actor
episode identifies its objects once as it opens and every action's grasp relations update them;
`--no-perception` leaves the scene toolsets out. Every task starts from the Memory its S1 passed
with in this run directory (`<run_dir>/<task>/S1/<seed>/exit.json`), or from the version
`--start` names, and ends frozen or paused; `--resume` continues a practice that started.
Settings come from configs/learning.yaml and configs/manipulation.yaml (copy
configs/learning.example.yaml and, from examples/simBenchmarks/RoboTwin/configs,
manipulation.example.yaml); `--set` changes practice parameters (`--set max_seeds=5`), `--learning`
the learning config (`--learning max_infra_reruns=1`), `--manipulation` the toolset's
(`--manipulation table_top_z=0.71`).

    uv run python examples/simBenchmarks/RoboTwin/scripts/run_s2.py --task beat_block_hammer [--task
    ...] \\
        --sim-python /path/to/envs/roboscript-sim/bin/python [--gpus 0,1] \\
        [--vision-python /path/to/envs/roboscript-vision/bin/python [--vision-gpu 2]] \\
        [--start v003] [--resume]

Run from the repository root in the agent venv. The toolsets are `toolsets()`'s, which puts them
under the prompt-hygiene test (tests/benchmarks/robotwin/test_prompt_hygiene.py).
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import anyio
import httpx
import yaml

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.ids import TaskName
from roboscript.envs import control_token
from roboscript.gateway import openai_client
from roboscript.kernel import compose
from roboscript.learning import LearningConfig, ProtocolRunner, control_client
from roboscript.perception.config import PerceptionConfig
from roboscript.settings import load_config, repo_root
from roboscript.stack import Interpreters, ServiceSpec, StackConfig, least_loaded, up
from roboscript.tools import LocalProvider, ToolProvider
from roboscript.toolsets.geometry import Geometry
from roboscript.toolsets.manipulation import Manipulation, ManipulationConfig
from roboscript.toolsets.perception import Perception
from roboscript.toolsets.scene import SceneConfig
from roboscript.toolsets.sensing import Sensing
from roboscript.toolsets.worldstate import WorldState
from roboscript_methods.wicl.toolsets.demo import Demo, DemoConfig
from roboscript_methods.wicl.toolsets.done import Done
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.config import (
    LINT_OVERLAYS,
    RoboTwinConfig,
    with_perception_binding,
    with_perception_place,
)


def toolsets(
    manipulation: ManipulationConfig,
    endpoints: Mapping[str, str] | None = None,
    *,
    perception: bool = True,
    demo: bool = False,
    vlm_model: str | None = None,
) -> list[ToolProvider]:
    """The toolsets of a practice besides `memory`, which the runner mounts itself: sensing,
    manipulation (after every action its hook, `toolsets.scene.after_action`, keeps the grasp
    relations), with `demo` the demo toolset (for stacks that run S1: its episodes alone see
    it; S2-only stacks leave it out) and, unless `perception` is off, the scene toolsets
    (perception, geometry, worldstate and wicl's done; configs/scene.yaml if present) with
    the stack's planner and vision service (`endpoints`: service name -> URL; none given: unset,
    as when only their texts are read), their VLM asking `vlm_model` when given (else the
    config's). Each has RoboTwin's skillsets mounted (`ROBOTWIN.skillsets`)."""
    skillsets = ROBOTWIN.skillsets
    given = endpoints or {}
    scene = None
    if perception:
        scene = scene_config().model_copy(
            update={"planner": given.get("motion"), "vision": given.get("vision")}
        )
        if vlm_model is not None:
            scene = scene.model_copy(
                update={"vlm": scene.vlm.model_copy(update={"model": vlm_model})}
            )
    # the demo toolset measures a release again on the demonstration's frame as worldstate does
    demo_config = None if scene is None else DemoConfig(scene=scene)
    mounted: list[ToolProvider] = [
        LocalProvider(Sensing(), None, skillsets),
        LocalProvider(Manipulation(), manipulation, skillsets),
        *([LocalProvider(Demo(), demo_config, skillsets)] if demo else []),
    ]
    if scene is not None:
        mounted += [
            LocalProvider(Perception(), scene, skillsets),
            LocalProvider(Geometry(), scene, skillsets),
            LocalProvider(WorldState(), scene, skillsets),
            LocalProvider(Done(), scene, skillsets),
        ]
    return mounted


def scene_config() -> SceneConfig:
    """The scene toolsets' config: configs/scene.yaml, else the defaults, with RoboTwin's words."""
    try:
        scene = load_config("scene", SceneConfig)
    except FileNotFoundError:
        scene = SceneConfig()
    pipeline = with_perception_binding(scene.pipeline.model_dump(exclude_unset=True))
    return scene.model_copy(update={"pipeline": PerceptionConfig.model_validate(pipeline)})


def manipulation_config(
    planner: str, overrides: Sequence[str] = (), root: Path | None = None
) -> ManipulationConfig:
    """`configs/manipulation.yaml` under `root` (the repository's) with `overrides`, planning with
    the motion service at `planner`, the stack's, and finding the demonstration's table with
    RoboTwin's anchor camera and table area (`with_perception_place`)."""
    stack = f"planner={json.dumps({'url': planner})}"
    found = load_config(
        "manipulation", ManipulationConfig, root=root, overrides=[stack, *overrides]
    )
    table = with_perception_place(found.table_perception.model_dump(exclude_unset=True))
    return found.model_copy(update={"table_perception": PerceptionConfig.model_validate(table)})


def stack_config(
    config: LearningConfig,
    tasks: list[TaskName],
    sim: Path,
    gpus: list[int],
    vision: Path | None = None,
    vision_gpu: int | None = None,
) -> StackConfig:
    """An env server per task (its config written under the run directory), the planner and,
    with a `vision` python, the vision service on `vision_gpu`, sharing the stack's blob
    directory with the runner."""
    blobs = config.run_dir / "blobs"
    services = []
    for n, task in enumerate(tasks):
        env = RoboTwinConfig(task=task, blob_dir=blobs)
        path = config.run_dir / "configs" / f"env-{task}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(env.model_dump(mode="json")), encoding="utf-8")
        services.append(
            ServiceSpec(
                name=f"env-{task}",
                module=ROBOTWIN.env_server,
                interpreter="sim",
                args=["--config", str(path)],
                gpu=gpus[n % len(gpus)],
                ready_timeout_s=600,
            )
        )
    services.append(
        ServiceSpec(
            name="motion",
            module="roboscript.motion",
            interpreter="sim",
            args=["--config", "motion"],
            gpu=gpus[0],
            ready_timeout_s=600,
        )
    )
    root = repo_root()
    if vision is not None:
        if vision_gpu is None:
            raise ValueError("the vision service needs its GPU")
        from roboscript_vision.config import VisionConfig  # the vision package, config only

        shared = {**load_config("vision", VisionConfig).model_dump(mode="json"), "blob_dir": None}
        path = config.run_dir / "configs" / "vision.yaml"
        path.write_text(yaml.safe_dump(shared), encoding="utf-8")
        services.append(
            ServiceSpec(
                name="vision",
                module="roboscript_vision",
                interpreter="vision",
                args=["--config", str(path)],
                gpu=vision_gpu,
                ready_timeout_s=600,
            )
        )
    return StackConfig(
        interpreters=Interpreters(agent=Path(sys.executable), sim=sim, vision=vision),
        sim_pythonpath=[
            root / "packages/core/src",
            root / "packages/vision/src",
            *ROBOTWIN.pythonpath,
        ],
        vision_pythonpath=[root / "packages/core/src", root / "packages/vision/src"],
        run_dir=config.run_dir,
        blob_dir=blobs,
        services=services,
    )


async def practice(args: argparse.Namespace) -> int:
    config = load_config("learning", LearningConfig, overrides=args.learning)
    if config.lint_overlay is None:
        overlay = LearningConfig.model_validate(
            {"run_dir": config.run_dir, "lint_overlay": LINT_OVERLAYS["memory"]}
        )
        config = config.model_copy(update={"lint_overlay": overlay.lint_overlay})
    tasks = [TaskName(task) for task in args.task]
    if unknown := sorted(set(tasks) - set(ROBOTWIN.tasks)):
        raise SystemExit(f"RoboTwin has no tasks {unknown}")
    values = compose({"resume": args.resume, "start": args.start}, overrides=args.set)
    if not args.gpus:
        raise SystemExit("name the GPUs: --gpus or CUDA_VISIBLE_DEVICES")
    pool = args.gpu_pool or args.gpus
    if args.vision_gpu is not None and args.vision_gpu not in pool:
        raise SystemExit(f"--vision-gpu {args.vision_gpu} is not one of --gpu-pool {pool}")
    vision_gpu = args.vision_gpu
    if args.vision_python is not None and vision_gpu is None:
        vision_gpu = await least_loaded(pool)
    stack = stack_config(config, tasks, args.sim_python, args.gpus, args.vision_python, vision_gpu)
    token = control_token()
    async with up(stack) as running, httpx.AsyncClient(trust_env=False) as http:
        controls = {
            task: control_client(running.urls[f"env-{task}"]["env.control"], token, config, http)
            for task in tasks
        }
        manipulation = manipulation_config(running.endpoints["motion"], args.manipulation)
        runner = ProtocolRunner(
            config,
            providers=toolsets(manipulation, running.endpoints, perception=not args.no_perception),
            blobs=LocalBlobStore(running.blob_dir),
            controls=controls,
            embodiment=ROBOTWIN.embodiment,
            models=openai_client,
        )
        async with runner:
            results = await runner.run_tasks({task: [("practice", values)] for task in tasks})
    failed = 0
    for task, result in results.items():
        if isinstance(result, Exception):
            failed += 1
            print(f"{task}: stopped by {type(result).__name__}: {result}")
        else:
            print(f"{task}: {result[0].model_dump_json()}")
    return 1 if failed else 0


def _gpu_list(text: str) -> list[int]:
    return [int(gpu) for gpu in text.split(",") if gpu]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", action="append", required=True, help="a RoboTwin task")
    ap.add_argument("--sim-python", type=Path, required=True, help="the simulator env's python")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ap.add_argument(
        "--gpus", type=_gpu_list, default=visible, help="e.g. 0,1 (default: CUDA_VISIBLE_DEVICES)"
    )
    ap.add_argument(
        "--vision-python", type=Path, help="the vision env's python: start the vision service"
    )
    ap.add_argument(
        "--vision-gpu", type=int, help="the vision service's GPU (default: the least loaded)"
    )
    ap.add_argument(
        "--gpu-pool",
        type=_gpu_list,
        default=visible,
        help="the GPUs the vision service may go on (default: CUDA_VISIBLE_DEVICES, else --gpus)",
    )
    ap.add_argument(
        "--no-perception", action="store_true", help="mount no scene toolsets (no identify)"
    )
    ap.add_argument("--resume", action="store_true", help="continue a practice that started")
    ap.add_argument("--start", help="the Memory version to start from (default: S1's exit)")
    ap.add_argument("--set", action="append", default=[], help="a practice parameter key=value")
    ap.add_argument("--learning", action="append", default=[], help="a learning config key=value")
    ap.add_argument(
        "--manipulation", action="append", default=[], help="a manipulation config key=value"
    )
    return anyio.run(practice, ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
