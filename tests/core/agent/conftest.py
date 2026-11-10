"""Fixtures for the agent tests: MockEnv in-process behind a Hub with the sensing toolset and the
Mock toolset, and sessions over them that record into the test's directory."""

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from roboscript.agent import AgentSession, Limits, Recorder
from roboscript.agent.session import ModelClients
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.env import EpisodeSpec, ObserveRequest
from roboscript.contracts.errors import InfraError
from roboscript.contracts.hub import Budgets, EpisodeSeed, OpenEpisode
from roboscript.contracts.ids import EpisodeId, TaskName
from roboscript.contracts.observation import Observation
from roboscript.contracts.phase import PhaseRules
from roboscript.contracts.robot import RobotState
from roboscript.contracts.tools import ContentPart, EnvBinding
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.records import EventWriter
from roboscript.testing.mock import Mock
from roboscript.testing.phases import TEST_PHASES
from roboscript.tools import LocalProvider
from roboscript.toolsets.sensing import Sensing


class FlakyEnv(MockEnv):
    """MockEnv whose agent-plane calls named in `broken` fail as infrastructure."""

    def __init__(self, config: MockConfig, blobs: LocalBlobStore) -> None:
        super().__init__(config, blobs)
        self.broken: set[str] = set()

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation:
        if "observe" in self.broken:
            raise InfraError("sim_fault", "the simulator stopped")
        return await super().observe(episode_id, request)

    async def robot(self, episode_id: EpisodeId) -> RobotState:
        if "robot" in self.broken:
            raise InfraError("sim_fault", "the simulator stopped")
        return await super().robot(episode_id)


@dataclass
class Rig:
    """A started Hub over sensing and the Mock toolset in front of MockEnv (task lift_red)."""

    hub: Hub
    env: FlakyEnv
    mock: Mock
    blobs: LocalBlobStore
    events: EventWriter
    root: Path

    async def session(
        self,
        stage: str = "s1",
        role: str = "actor",
        instructions: str = "Lift the red box.",
        brief: Sequence[ContentPart] | None = None,
        limits: Limits | None = None,
        budgets: Budgets | None = None,
        rules: PhaseRules | None = None,
        models: ModelClients | None = None,
        toolsets: list[str] | None = None,
        seed: EpisodeSeed | None = None,
        env: bool = True,
    ) -> AgentSession:
        binding = None
        if env:
            opened = await self.env.open(EpisodeSpec(task=TaskName("lift_red"), seed=3))
            binding = EnvBinding(
                url="inproc://mock", episode_id=opened.episode_id, token=opened.agent_token
            )
        spec = OpenEpisode(
            stage=stage,
            role=role,
            env=binding,
            embodiment=str(MOCK_BIMANUAL),
            budgets=budgets or Budgets(),
            rules=rules,
            toolsets=toolsets,
            seed=seed,
        )
        episode = await self.hub.open_episode(spec)
        return AgentSession(
            episode=episode,
            hub=self.hub,
            instructions=instructions,
            blobs=self.blobs,
            recorder=Recorder(self.root / episode.episode_id, self.events),
            brief=brief,
            limits=limits or Limits(),
            models=models,
        )


@pytest.fixture
def events(artifacts_dir: Path) -> Iterator[EventWriter]:
    with EventWriter(artifacts_dir / "events.jsonl", "orchestrator") as writer:
        yield writer


@pytest.fixture
async def rig(tmp_path: Path, artifacts_dir: Path, events: EventWriter) -> AsyncIterator[Rig]:
    blobs = LocalBlobStore(tmp_path / "blobs")
    config = MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs")
    env, mock = FlakyEnv(config, blobs), Mock()
    providers = [LocalProvider(Sensing()), LocalProvider(mock, {"act_seconds": 0.0})]
    async with Hub(
        providers, blobs=blobs, phases=TEST_PHASES, events=events, env_client=lambda _: env
    ) as hub:
        yield Rig(hub, env, mock, blobs, events, artifacts_dir / "episodes")
