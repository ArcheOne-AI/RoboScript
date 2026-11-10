"""A benchmark's own evaluation client for tests of the Agent host (docs/concepts/learning.md):
`MockClient` drives `agent.host` episodes as such a client does, with `MockEnv` as its
simulator: `reset`, then one turn per call of its policy (submit an observation, poll until the
batch comes, run it, report the states), holding still to the step limit once the agent is done,
and `end` with the outcome; then it waits until the host has recorded the episode.

`mock_host` starts such a host on MockEnv for a method's evaluation phase (P3, or P4's
`p4_eval`): lift_red's frozen Memory, sensing and `hold`, scripted Actor brains, and optionally an
artifact stack with its overlay loader (`agent.overlay`)."""

import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anyio

from roboscript.agent.host import AgentHostService, HostConfig
from roboscript.agent.overlay import OverlayLoader
from roboscript.agent.scripted import Script, ScriptedBrain
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.artifact import ArtifactStack
from roboscript.contracts.env import EpisodeSpec, JudgeResult, ObserveRequest
from roboscript.contracts.host import AgentHost, OfficialOutcome, Poll
from roboscript.contracts.ids import BatchId, EpisodeId, ManifestId, TaskName, new_id
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs import BridgeConfig, Bridges
from roboscript.envs.cards import TaskCard
from roboscript.envs.mock import MOCK, MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.learning.agent_host import actor_prompt, frozen_memory, host_phase
from roboscript.learning.method import Declaration
from roboscript.memory import FileMemoryStore, MemoryManifest, VersionSource, build_manifest
from roboscript.memory.artifact import MemoryArtifacts
from roboscript.records import EventWriter
from roboscript.testing.hold import Hold
from roboscript.testing.memory import sample_memory
from roboscript.tools import LocalProvider
from roboscript.tools.skillsets import guides as skillset_guides
from roboscript.toolsets.sensing import Sensing

LIFT_RED: Final = TaskName("lift_red")


@dataclass
class MockClient:
    """The client of one episode slot (`key`) of `host`, evaluating `manifest` on `env`'s task."""

    env: MockEnv
    host: AgentHost
    key: str
    task: TaskName
    manifest: ManifestId
    poll_s: float = 0.01

    async def episode(self, seed: int = 3) -> JudgeResult:
        """One episode with the mock's paraphrased instruction; the benchmark's verdict."""
        spec = EpisodeSpec(task=self.task, seed=seed, options={"instructions": "paraphrase"})
        opened = await self.env.open(spec)
        await self.host.reset(self.key, self.task, self.manifest)
        while not (judge := await self.env.judge(opened.episode_id)).success:
            assert judge.step_limit is not None
            if judge.steps_used >= judge.step_limit:
                break
            await self.turn(opened.episode_id)
        outcome = OfficialOutcome(
            success=judge.success, steps=judge.steps_used, step_limit=judge.step_limit
        )
        await self.host.end(self.key, outcome)
        await self._answer("done")  # the host has recorded the episode
        return judge

    async def turn(self, episode: EpisodeId) -> None:
        """One call of the policy: at least one counted action."""
        await self.host.submit(self.key, await self.env.observe(episode, ObserveRequest()))
        answer = await self._answer("batch", "done")
        if answer.batch is not None:
            await self.host.report(self.key, await self.env.execute(episode, answer.batch))
            return
        robot = await self.env.robot(episode)  # the agent is done: hold still to the limit
        assert robot.step_limit is not None
        steps = [ActionStep(qpos=robot.qpos_cmd)] * (robot.step_limit - robot.step)
        await self.env.execute(episode, ActionBatch(batch_id=BatchId(new_id("b")), steps=steps))

    async def _answer(self, *statuses: str) -> Poll:
        """Poll until the host answers with one of `statuses`."""
        while True:
            answer = await self.host.poll(self.key)
            if answer.status in statuses:
                return answer
            await anyio.sleep(self.poll_s)


@dataclass(frozen=True)
class MockHost:
    """A started host, its client on slot `key`, the manifest it evaluates, its records."""

    service: AgentHostService
    client: MockClient
    manifest: MemoryManifest
    records: Path


@contextlib.asynccontextmanager
async def mock_host(
    directory: Path,
    events: EventWriter,
    script: Script,
    method: type[Declaration],
    *,
    key: str,
    records: str = "host",
    stack: Callable[[MemoryManifest], ArtifactStack] | None = None,
    overlay: OverlayLoader | None = None,
    cards: Mapping[TaskName, TaskCard] | None = None,
) -> AsyncIterator[MockHost]:
    """A host over sensing and `hold` evaluating lift_red's frozen Memory (committed and frozen
    under `directory/memory` once; later hosts on that directory reuse it), opening its episodes in
    `method`'s host phase with its runtime's prompts and finish gate, each Actor a scripted brain
    on `script`; with the stack `stack` makes of the manifest and its `overlay`, and the tasks'
    success condition `cards`. Records go to
    `directory/<records>`."""
    blobs = LocalBlobStore(directory / "blobs")
    store = FileMemoryStore(directory / "memory")
    if await store.frozen(LIFT_RED) is None:
        await store.commit(
            sample_memory(LIFT_RED), status="s1_draft", source=VersionSource(kind="s1")
        )
        await store.freeze(LIFT_RED, 0, reason="for the test")
    manifest = await build_manifest(store, [LIFT_RED])
    runtime = method.runtime(Path("."))
    phase = host_phase(method, runtime)
    bridges = Bridges()
    hub = Hub(
        [LocalProvider(Sensing()), LocalProvider(Hold())],
        blobs=blobs,
        phases=method.phases,
        events=events,
        env_client=bridges.client,
    )
    config = HostConfig(
        records=directory / records,
        embodiment=str(MOCK_BIMANUAL),
        bridge=BridgeConfig(observe_timeout_s=5, report_timeout_s=5),
        latch_finish=False,
    )
    guides = skillset_guides(MOCK.skillsets)  # the Mock's guide: its toolsets mount no skillset
    prompt, digest = actor_prompt(
        runtime.prompts({}, guides), load_embodiment(MOCK_BIMANUAL), phase
    )
    service = AgentHostService(
        config,
        hub=hub,
        bridges=bridges,
        blobs=blobs,
        events=events,
        load=frozen_memory(MemoryArtifacts(memory=store), runtime.frozen),
        manifest=manifest,
        brain=lambda: ScriptedBrain(script),
        prompt=prompt,
        phase=phase,
        prompt_digest=digest,
        finish=runtime.finish.host,
        stack=None if stack is None else stack(manifest),
        overlay=overlay,
        cards=cards,
    )
    env = MockEnv(MockConfig(task=LIFT_RED, blob_dir=directory / "blobs"), blobs)
    async with hub:
        await service.start()  # out of the service's task group: its errors come as they are
        async with service:
            client = MockClient(env, service, key, LIFT_RED, manifest.id)
            yield MockHost(service, client, manifest, directory / records)
