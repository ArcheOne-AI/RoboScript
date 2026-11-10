"""`ReplayBrain` (docs/concepts/agent-runtime.md, docs/concepts/operations.md): the
reference brain answered by a recorded run instead of a model.

It is `ResponsesBrain` with the same configuration (so its requests are the recorded run's, byte
for byte where the code did not change) whose model client the process's replay tape answers
(`roboscript.records.replay.replay_client`): no gateway, no token, no network. A request the
recording has no answer for ends the run with the diff against the nearest recorded request
(`ReplayMiss`), which tells a changed prompt from a field the canonical form missed. Perception's
VLM calls are answered by the same tape through the gateway client factory, so a replayed run
needs `RS_REPLAY=replay` in any case; this brain only makes the brain's side independent of the
gateway's configuration.
"""

from functools import partial

from roboscript.agent.loop import run_loop
from roboscript.agent.responses import ResponsesBrain, ResponsesConfig, respond
from roboscript.agent.session import AgentSession, BrainResult
from roboscript.records.replay import replay_client


class ReplayBrain(ResponsesBrain):
    """`ResponsesBrain` on the replay tape (see the module docstring)."""

    Config = ResponsesConfig

    async def run(self, session: AgentSession) -> BrainResult:
        options = self.options(session.episode_id)
        async with replay_client(session.episode_id) as client:
            return await run_loop(session, partial(respond, client), options, self.config)
