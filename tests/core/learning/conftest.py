"""Fixtures for the learning tests: MockEnv's lift_red in-process (seed 10001 unsolvable), runners
over it with scripted brains and `MockPlanner` for the referee's FK, and the store S2 starts from
(roboscript.testing.protocols)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from roboscript.agent.session import ModelClients
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.ids import TaskName
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.cards import TaskCard
from roboscript.envs.mock import MOCK, MockConfig
from roboscript.learning import Hooks, LearningConfig, ProtocolRunner, RoleConfig
from roboscript.learning.episode import BrainFactory
from roboscript.learning.method import Declaration
from roboscript.memory import FileMemoryStore, MemoryDoc, Version, VersionSource
from roboscript.motion import MockPlanner
from roboscript.testing.phases import TestMethod
from roboscript.testing.protocols import LIFT_RED, Lifter, ScriptedEnv
from roboscript.tools import LocalProvider, ToolProvider
from roboscript.tools.skillsets import guides
from roboscript.toolsets.sensing import Sensing

RECIPE = Path(__file__).resolve().parents[3] / "recipes" / "wicl"


@dataclass
class Setup:
    """Scripted MockEnvs by task (lift_red first), a blob store and a run directory."""

    envs: dict[TaskName, ScriptedEnv]
    blobs: LocalBlobStore
    run_dir: Path
    tmp: Path
    lifters: dict[TaskName, Lifter] = field(default_factory=dict)
    method: type[Declaration] = TestMethod  # the runners' (`ProtocolRunner(method=...)`)

    @property
    def env(self) -> ScriptedEnv:
        return self.envs[LIFT_RED]

    @env.setter
    def env(self, env: ScriptedEnv) -> None:
        self.envs[LIFT_RED] = env

    def add(self, config: MockConfig) -> ScriptedEnv:
        """Another task's env."""
        env = self.envs[config.task] = ScriptedEnv(config, self.blobs)
        return env

    def runner(
        self,
        brains: Mapping[str, BrainFactory],
        *,
        hooks: Hooks | None = None,
        lifter: bool = True,
        max_infra_reruns: int = 2,
        infra_backoff_s: Sequence[float] = (),
        models: ModelClients | None = None,
        roles: Mapping[str, RoleConfig] | None = None,
        extra: Sequence[ToolProvider] = (),
        recipe: Path = RECIPE,
        cards: Mapping[TaskName, TaskCard] | None = None,
    ) -> ProtocolRunner:
        config = LearningConfig(
            run_dir=self.run_dir,
            method="wicl",
            recipe=recipe,
            max_infra_reruns=max_infra_reruns,
            infra_backoff_s=list(infra_backoff_s),
        )
        providers: list[ToolProvider] = [LocalProvider(Sensing()), *extra]
        if lifter:  # one Lifter serves every task's env: it acts on the episode's own env
            providers.append(LocalProvider(Lifter(self.envs)))
        return ProtocolRunner(
            config,
            providers=providers,
            blobs=self.blobs,
            controls=dict(self.envs),
            embodiment=MOCK_BIMANUAL,
            agent_env=lambda binding: next(
                env for env in self.envs.values() if env.episodes.current == binding.episode_id
            ),
            brains=brains,
            hooks=hooks,
            method=self.method,
            models=models,
            roles=roles,
            planner=MockPlanner(),
            guides=guides(MOCK.skillsets),  # the Mock's guide; its toolsets mount no skillset
            cards=cards,
        )

    @property
    def store(self) -> FileMemoryStore:
        return FileMemoryStore(self.run_dir)

    async def start(self, doc: MemoryDoc) -> Version:
        """Commit `doc` as the S1 draft S2 starts from (v000 in a new run)."""
        return await self.store.commit(doc, status="s1_draft", source=VersionSource(kind="s1"))


@pytest.fixture
def setup(tmp_path: Path, artifacts_dir: Path) -> Setup:
    return make_setup(tmp_path, artifacts_dir)


def make_setup(tmp_path: Path, artifacts_dir: Path) -> Setup:
    """`setup`'s value (the methods' tests make it with their own method)."""
    blobs = LocalBlobStore(tmp_path / "blobs")
    config = MockConfig(task=LIFT_RED, blob_dir=tmp_path / "blobs", unsolvable_seeds=[10001])
    return Setup(
        envs={LIFT_RED: ScriptedEnv(config, blobs)},
        blobs=blobs,
        run_dir=artifacts_dir / "run",
        tmp=tmp_path,
    )
