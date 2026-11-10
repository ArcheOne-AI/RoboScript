"""The core's runner opens episodes on an env that resets by initial state only: the BEHAVIOR
skeleton on its fake official module, run by `ProtocolRunner` with the core's `evaluate` protocol
and a scripted Actor. The episodes are reset as the env's descriptor says
(`TaskContext.reset_by`, from `ProtocolRunner(offered=...)`)."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import fakes
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.ids import TaskName
from roboscript.envs.skeleton import SkeletonEnv
from roboscript.learning import Evaluate, EvaluateResult, LearningConfig, ProtocolRunner
from roboscript.memory import FileMemoryStore, VersionSource
from roboscript.testing.phases import TestMethod
from roboscript.testing.protocols import finishing, lift_memory
from roboscript.tools import LocalProvider
from roboscript.toolsets.sensing import Sensing
from roboscript_behavior.benchmark import BEHAVIOR
from roboscript_behavior.config import BehaviorConfig
from roboscript_behavior.sim.host import make_env

TASK = TaskName("turning_on_radio")


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SkeletonEnv]:
    fakes.install_behavior(monkeypatch)
    config = BehaviorConfig(task=TASK, blob_dir=tmp_path / "blobs", once_per_reset=False)
    served = make_env(config, LocalBlobStore(tmp_path / "blobs"))
    await served.start()
    served.agent_url = "http://agent"
    try:
        yield served
    finally:
        served.shutdown()


def _runner(env: SkeletonEnv, run_dir: Path, offered: EnvCapabilities | None) -> ProtocolRunner:
    return ProtocolRunner(
        LearningConfig(run_dir=run_dir, method="wicl", recipe=Path("recipes/wicl")),
        providers=[LocalProvider(Sensing())],
        blobs=env.blobs,
        controls={TASK: env},
        embodiment=BEHAVIOR.embodiment,
        agent_env=lambda binding: env,
        brains={"actor": finishing},
        method=TestMethod,
        offered=offered,
    )


async def _frozen(run_dir: Path) -> None:
    store = FileMemoryStore(run_dir)
    await store.commit(
        lift_memory(0.06, TASK), status="edited", source=VersionSource(kind="edited")
    )
    await store.freeze(TASK, 0, reason="manual")


async def test_evaluate_opens_episodes_by_the_envs_initial_state(
    env: SkeletonEnv, artifacts_dir: Path
) -> None:
    run_dir = artifacts_dir / "run"
    await _frozen(run_dir)
    async with _runner(env, run_dir, BEHAVIOR.capabilities) as runner:
        result = await runner.run(Evaluate(), TASK, {"seeds": [3], "prerun_attempts": 1})
    assert isinstance(result, EvaluateResult) and (result.n, result.unsolvable) == (1, [])
    written = run_dir / TASK / "eval/v000/3/try1/config.json"
    config = json.loads(written.read_text(encoding="utf-8"))
    assert config["env"]["reset_by"] == "initial_state" and config["env"]["seed"] == 3
