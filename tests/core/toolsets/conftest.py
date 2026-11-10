"""Fixtures for the toolset tests: MockEnv in-process behind a Hub, and model calls."""

import itertools
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import EpisodeId, TaskName
from roboscript.contracts.tools import EnvBinding, ToolCall
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.records import EventWriter
from roboscript.testing.phases import dev_phases
from roboscript.tools import LocalProvider, Toolset


@dataclass
class Rig:
    """A Hub over some toolsets in front of MockEnv (task lift_red)."""

    env: MockEnv
    blobs: LocalBlobStore
    events: EventWriter

    def hub(self, *toolsets: Toolset | LocalProvider) -> Hub:
        """A Hub over the toolsets (a toolset alone takes its default config), under
        `TEST_PHASES` with a test's own toolsets in phase `dev`."""
        providers = [t if isinstance(t, LocalProvider) else LocalProvider(t) for t in toolsets]
        return Hub(
            providers,
            blobs=self.blobs,
            phases=dev_phases([provider.name for provider in providers]),
            events=self.events,
            env_client=lambda _: self.env,
        )

    async def episode(
        self, hub: Hub, stage: str = "s1", role: str = "actor", seed: int = 3
    ) -> tuple[EpisodeId, EpisodeId]:
        """Open an env episode and a Hub episode bound to it (results also as JSON):
        (Hub id, env id)."""
        opened = await self.env.open(EpisodeSpec(task=TaskName("lift_red"), seed=seed))
        binding = EnvBinding(
            url="inproc://mock", episode_id=opened.episode_id, token=opened.agent_token
        )
        spec = OpenEpisode(
            stage=stage, role=role, env=binding, embodiment=str(MOCK_BIMANUAL), structured=True
        )
        session = await hub.open_episode(spec)
        return session.episode_id, opened.episode_id


@pytest.fixture
def events(artifacts_dir: Path) -> Iterator[EventWriter]:
    with EventWriter(artifacts_dir / "events.jsonl", "orchestrator") as writer:
        yield writer


@pytest.fixture
async def rig(tmp_path: Path, events: EventWriter) -> AsyncIterator[Rig]:
    blobs = LocalBlobStore(tmp_path / "blobs")
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs"), blobs=blobs)
    yield Rig(env=env, blobs=blobs, events=events)


@pytest.fixture
def model() -> Callable[..., ToolCall]:
    """`model("observe", views=[...])`: a call as a model makes it."""
    ids = itertools.count(1)

    def make(tool: str, /, **arguments: Any) -> ToolCall:
        return ToolCall(call_id=f"call-{next(ids)}", name=tool, arguments=arguments, origin="model")

    return make
