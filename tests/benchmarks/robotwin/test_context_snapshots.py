"""Golden snapshots of what a model is sent as tools and instructions, on the stacks the RoboTwin
pipeline and its P3 host mount: `scripts/run_s2.py`'s `toolsets()` and the aloha_agilex embodiment
(`snapshot_harness` says what a snapshot holds and how to regenerate it)."""

import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, cast

import pytest
import yaml
from robotwin_snapshot_tasks import ROBOTWIN_TASKS
from snapshot_harness import (
    PIPELINE_VARIANTS,
    RECIPE,
    Prompt,
    Stack,
    Variant,
    build_runner,
    capture,
    compare,
    pipeline_prompts,
)

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.env import EnvControl
from roboscript.embodiments import load_embodiment
from roboscript.learning.agent_host import ToolsetSpec, actor_prompt, toolset_provider
from roboscript.settings import repo_root
from roboscript.tools import LocalProvider
from roboscript.tools.provider import provider_guides
from roboscript.toolsets.manipulation import ManipulationConfig, PlannerConfig
from roboscript.toolsets.privileged import Privileged
from roboscript_methods.wicl.method import Wicl
from roboscript_robotwin.benchmark import HERE as ROBOTWIN_HERE
from roboscript_robotwin.benchmark import ROBOTWIN

pytestmark = pytest.mark.unit

SNAPSHOTS: Final = Path(__file__).resolve().parent / "snapshots"
PLANNED: Final = ManipulationConfig(planner=PlannerConfig(url="http://127.0.0.1:18300"))
P3_TOOLSETS: Final = ("sensing", "manipulation", "perception", "geometry", "worldstate", "done")
"""The toolsets of `examples/simBenchmarks/RoboTwin/configs/p3.example.yaml`'s Agent host."""


def _run_s2() -> Any:
    """`scripts/run_s2.py`, whose `toolsets()` the pipeline's RoboTwin wiring mounts."""
    path = repo_root() / "examples" / "simBenchmarks" / "RoboTwin" / "scripts" / "run_s2.py"
    spec = importlib.util.spec_from_file_location("run_s2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robotwin_stack(tmp: Path, manipulation: ManipulationConfig = PLANNED) -> Stack:
    """The RoboTwin pipeline's S1/S2 stack: the pipeline's toolsets (sensing, manipulation, demo,
    perception, geometry, worldstate), the S1-only truth toolset, and the memory toolset the
    runner adds."""
    truth = LocalProvider(Privileged(cast(EnvControl, None)))  # listed only, never called
    providers = [*_run_s2().toolsets(manipulation, demo=True), truth]
    runner, blobs = build_runner(tmp, providers, ROBOTWIN.embodiment)
    for task in ROBOTWIN_TASKS:
        info = yaml.safe_load((ROBOTWIN_HERE / "tasks.yaml").read_text("utf-8"))["tasks"]
        assert task.name in info, task.name
    return Stack(
        "robotwin",
        runner.providers,
        runner.method.phases,
        blobs,
        PIPELINE_VARIANTS,
        pipeline_prompts(runner),
        ROBOTWIN_TASKS,
    )


def p3_stack(tmp: Path) -> Stack:
    """The P3 Agent host's: the five toolsets of its config, loaded as the host loads them, and
    its Actor prompt."""
    planner = {"planner": PLANNED.planner.model_dump(mode="json")}  # the launcher fills it in
    profile = load_embodiment(ROBOTWIN.embodiment)
    providers = [
        toolset_provider(
            ToolsetSpec(
                name=name,
                config=planner if name == "manipulation" else {},
                skillsets=ROBOTWIN.skillsets,  # the launcher mounts them (`eval.launch`)
            ),
            profile,
        )
        for name in P3_TOOLSETS
    ]
    runtime = Wicl.runtime(RECIPE)
    assert runtime.host_phase is not None
    prompts = runtime.prompts({}, provider_guides(providers))  # as `serve_host` makes them
    prompt, _ = actor_prompt(prompts, profile, runtime.host_phase)
    return Stack(
        "robotwin_p3host",
        providers,
        Wicl.phases,
        LocalBlobStore(tmp / "blobs"),
        [Variant("p3", "D", "actor", "actor")],
        [Prompt("actor_p3", "actor", "p3", "D", lambda t: prompt(t.instruction, t.memory))],
        ROBOTWIN_TASKS,
    )


STACKS: Final[Mapping[str, Callable[[Path], Stack]]] = {
    "robotwin": robotwin_stack,
    "robotwin_p3host": p3_stack,
}


@pytest.mark.parametrize("name", sorted(STACKS))
async def test_the_model_sees_what_the_snapshots_say(name: str, tmp_path: Path) -> None:
    files = await capture(STACKS[name](tmp_path / "a"))
    again = await capture(STACKS[name](tmp_path / "b"))
    assert files == again, (
        "rebuilding the same stack changes what the model sees: not deterministic"
    )
    compare(SNAPSHOTS, name, files)


async def test_tool_lists_do_not_depend_on_the_toolsets_configs(tmp_path: Path) -> None:
    """The snapshots mount the toolsets with one config; their specs are the toolsets' own, so a
    deployment's config (planner address, table height) cannot change what a model sees."""
    other = PLANNED.model_copy(
        update={"planner": PlannerConfig(url="http://10.0.0.9:1"), "table_top_z": 0.71}
    )
    one = await capture(robotwin_stack(tmp_path / "a"))
    two = await capture(robotwin_stack(tmp_path / "b", other))
    assert one == two
