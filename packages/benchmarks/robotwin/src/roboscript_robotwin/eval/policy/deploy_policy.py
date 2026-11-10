"""RoboScript's policy for RoboTwin's official evaluation (docs/concepts/learning.md), linked as
`ROBOTWIN_ROOT/policy/RoboScript` by `roboscript_robotwin.eval.install`.

Both official processes import this module, so its top level imports nothing heavy (numpy at
most); the rest is imported where it is used. In `script/policy_model_server.py`, `get_model`
returns the relay (`roboscript_robotwin.eval.relay.Relay`), a thin client of the Agent host that
runs the agent. In `script/eval_policy_client.py`, every `eval()` is one agent turn: it polls first
(the agent may be done already), then submits the observation (the whitelisted camera frames,
measured joints and openings, the step count and limit, the instruction) and polls every `POLL_S`
until the turn's actions come (each round trip is short, far from the client's hard-coded 30 s
socket timeout), runs them with `take_action(qpos)` and a `get_obs()` after each (so that the
official video shows every step), and reports the states. Once the agent acts no more (`finish`,
a failure, an unreachable host, a report that did not arrive) each `eval()` holds still for a
chunk of actions (the relay's `hold`), since the official loop ends only at the step limit or the
success latch, and its own `get_obs()` between chunks keeps the video going; so every `eval()`
takes at least one action. When the episode is over it tells the host the outcome, and prints
`wire.INFRA_LOST` when infrastructure lost it. `TASK_ENV` is
used only through `roboscript_robotwin.sim.shim` (tests/benchmarks/robotwin checks that).
"""

import time
from typing import Any

POLL_S = 0.2
"""Between polls: a round trip each (a few ms to the local host), far below the official
client's 30 s socket timeout; short, since the actions wait for the next poll."""


def get_model(usr_args: dict[str, Any]) -> Any:
    """The relay to the Agent host (`deploy_policy.yml` and the launcher's overrides)."""
    from roboscript_robotwin.eval.relay import Relay

    return Relay.from_args(usr_args)


def eval(TASK_ENV: Any, model: Any, observation: dict[str, Any]) -> None:
    """One agent turn (see the module docstring); `observation` is the loop's `get_obs()`."""
    from roboscript_robotwin.eval import wire
    from roboscript_robotwin.sim import shim

    profile = wire.profile()
    if shim.progress(TASK_ENV)[0] == 0:  # an episode's first turn: no loss noted yet
        shim.clear_infra(TASK_ENV)
    answer = model.call(func_name="poll")
    if answer["status"] == "pending":
        frames = shim.frames_of(observation, list(profile.cameras))
        robot = shim.robot_state(TASK_ENV, profile)
        packed = wire.pack(frames, robot, shim.instruction(TASK_ENV))
        answer = model.call(func_name="submit", obs=packed)
        while answer["status"] == "pending":
            time.sleep(POLL_S)
            answer = model.call(func_name="poll")
    if answer["status"] == "batch":
        result = shim.run_batch(TASK_ENV, profile, wire.batch(answer), refresh=True)
        answer = model.call(func_name="report", obs=result.model_dump(mode="json"))
    if answer["status"] != "ok":  # done or failed: the agent acts no more in this episode
        if answer.get("infra"):
            shim.mark_infra(TASK_ENV)
        shim.hold(TASK_ENV, profile, answer["hold"])
    taken, limit, success = shim.progress(TASK_ENV)
    if success or (limit is not None and taken >= limit):
        outcome = {"success": success, "steps": taken, "step_limit": limit}
        model.call(func_name="end", obs=outcome)
        if shim.infra(TASK_ENV) and not success:  # a success that latched after it stands
            print(wire.INFRA_LOST, flush=True)


def reset_model(model: Any) -> None:
    """Begin the next episode. `script/eval_policy.py` (policy and simulator in one process) calls
    this; the client and server pair calls the relay's `reset_model` directly."""
    model.reset_model()
