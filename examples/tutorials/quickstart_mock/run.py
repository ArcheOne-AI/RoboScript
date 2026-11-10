"""Quickstart: one episode on the Mock benchmark, driven by a scripted brain.

No model, GPU or simulator: `MockEnv` is a CPU-only env with all three env planes, and
`ScriptedBrain` answers each model call with the next reply of a script. The episode goes through
the same Hub, visibility, guards and event log a real one does.

Run it:  uv run python -m quickstart_mock.run   (from examples/tutorials)
"""

import asyncio
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from roboscript.agent import AgentSession, Limits, Recorder, ScriptedBrain, call
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.env import EpisodeSpec
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import TaskName
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.tools import EnvBinding
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.records import EventWriter, read_events
from roboscript.testing.hold import Hold
from roboscript.tools import LocalProvider
from roboscript.toolsets.sensing import Sensing

# A phase is the method's declaration of who sees which tools, in which order, at which tier.
# Here one role, `actor`, sees two tools and `finish`, and works at tier D (no ground truth).
PHASE = PhaseSpec.model_validate(
    {
        "name": "tutorial",
        "kind": "learn",
        "visibility": {"actor": ["sensing.get_robot_state", "hold.hold", "hub.finish"]},
        "tiers": {"actor": "D"},
    }
)
# Every physical call carries a rationale: where in the workflow the model is and why it acts.
RATIONALE = {"stage": "wait", "scene": "a red box", "progress": "starting", "plan": "hold still"}
SCRIPT = [
    [call("get_robot_state")],
    [call("hold", rationale=RATIONALE, steps=3)],
    [call("finish", note="nothing to do")],
]


class Outcome(NamedTuple):
    end: str  # how the episode ended
    turns: int
    success: bool  # the env's verdict (holding still does not lift the box)
    events: dict[str, int]  # the event log by kind
    directory: Path


async def run(directory: Path) -> Outcome:
    """Play the scripted episode, leaving its records under `directory`."""
    blobs = LocalBlobStore(directory / "blobs")
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=directory / "blobs"), blobs=blobs)
    log = directory / "events.jsonl"
    # The Hub mounts toolsets as providers; `LocalProvider` runs one in this process.
    providers = [LocalProvider(Sensing()), LocalProvider(Hold())]
    with EventWriter(log, "quickstart") as events:
        async with Hub(
            providers, blobs=blobs, phases=[PHASE], events=events, env_client=lambda _: env
        ) as hub:
            # The control plane (the runner's) opens the env episode and hands the agent a token.
            opened = await env.open(EpisodeSpec(task=TaskName("lift_red"), seed=0))
            binding = EnvBinding(
                url="inproc://mock", episode_id=opened.episode_id, token=opened.agent_token
            )
            episode = await hub.open_episode(
                OpenEpisode(
                    stage="tutorial", role="actor", env=binding, embodiment=str(MOCK_BIMANUAL)
                )
            )
            session = AgentSession(
                episode=episode,
                hub=hub,
                instructions="Lift the red box.",
                blobs=blobs,
                recorder=Recorder(directory / "episode", events),
                limits=Limits(),
            )
            result = await ScriptedBrain(SCRIPT).run(session)
            verdict = await env.judge(opened.episode_id)
            await env.close(opened.episode_id)
    kinds = Counter(event.kind for event in read_events(log))
    return Outcome(result.end.reason, result.turns, verdict.success, dict(kinds), directory)


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="quick-"))
    outcome = asyncio.run(run(directory))
    print(f"ended: {outcome.end} after {outcome.turns} turns; env success: {outcome.success}")
    for kind, count in sorted(outcome.events.items()):
        print(f"  {kind}: {count}")
    print(f"records: {outcome.directory}")


if __name__ == "__main__":
    main()
