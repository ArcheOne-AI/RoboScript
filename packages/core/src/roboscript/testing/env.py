"""Conformance checks for env servers (docs/concepts/embodied-core.md and
docs/concepts/agent-runtime.md).

An env conforms when its planes behave as the contract says: discovery, the episode lifecycle,
per-episode agent tokens, calibrated observations, proprioception consistent with its embodiment
profile (the arms it declares, by name), step accounting, idempotent and exclusive `execute`,
ground truth on the control plane only, errors for closed episodes, resets only of the kinds it
declares, each reset opened once where it declares so (`reruns` False), and a terminal score
exactly where it judges by one. The checks talk RPC, as the orchestrator and toolsets do.

The optional faces are checked as the env declares them in `discover()`: the expert pre-run's
check (`EnvCapabilities.expert`) and the ground-truth check (`truth`) do not apply to an env that
declares neither (`not_applicable`: skipped under pytest), and an env without an expert must say
it ran no pre-run (`prerun_attempts` 0).

The other checks reopen a reset: an env that opens each once runs them as a copy that allows
reruns (and declares them in `discover()`), and the once-per-reset check on its own
configuration.

Use them from pytest by subclassing `EnvConformance` as a `Test*` class, setting the capability
flags and providing an `env_target` fixture; or without pytest via `roboscript.testing.precheck`.
Checks marked `unit` open no episode; the others (`integration`) do.
"""

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import anyio
import numpy as np
from pydantic import SecretStr

from roboscript.contracts import versions
from roboscript.contracts.action import ActionBatch, ActionStep, ExecuteResult
from roboscript.contracts.env import (
    EnvAgent,
    EnvControl,
    EnvPublic,
    EpisodeOpened,
    EpisodeSpec,
    ObserveRequest,
)
from roboscript.contracts.errors import EpisodeEnded, InfraError, ProtocolError, RSError
from roboscript.contracts.ids import BatchId, EpisodeId, TaskName, new_id
from roboscript.embodiments import EmbodimentProfile
from roboscript.testing import not_applicable, unit

Connect = Callable[[type[Any], str, SecretStr | None], Any]


@dataclass(frozen=True)
class EnvTarget:
    """How the checks reach an env: `connect(interface, url, token)` returns an RPC client (as
    `roboscript.rpc.client` does); `url` is the control port, `token` its bearer token. `seed`
    must be solvable for `task`, the task the env serves (its expert pre-run gets three tries,
    since planning may not be deterministic); `unsolvable_seed`,
    if given, never is (checked only on an env with an expert)."""

    url: str
    token: SecretStr
    connect: Connect
    profile: EmbodimentProfile
    task: TaskName
    seed: int
    unsolvable_seed: int | None = None
    reset_by: Literal["seed", "initial_state"] = "seed"
    """How the checks' episodes are reset (`EpisodeSpec.reset_by`): `seed` and
    `unsolvable_seed` are initial-state indexes for an env that resets by those only."""


class EnvConformance:
    """The checks, as `test_*` methods taking an `EnvTarget`; subclasses set the flags."""

    has_depth: ClassVar[bool] = True
    step_limit: ClassVar[bool] = True  # eval-mode episodes have a step limit
    exhaust_step_limit: ClassVar[bool] = False  # run one episode to its limit (slow on real sims)
    busy_steps: ClassVar[int] = 10  # steps of the batch that a concurrent `execute` must meet
    records: ClassVar[bool] = False  # the env records episodes (`EpisodeSpec.record`)
    traces_expert: ClassVar[bool] = False  # the env traces its expert (`EpisodeSpec.trace_expert`)

    @unit
    async def test_discover(self, env_target: EnvTarget) -> None:
        found = await _public(env_target).discover()
        for plane in ("env.public", "env.control", "env.agent"):
            served = versions.parse(f"{plane}/{found.contracts[plane]}")
            assert versions.compatible(
                versions.parse(f"{plane}/{versions.VERSIONS[plane]}"), served
            )
        assert found.embodiment == env_target.profile.name
        sensing = found.capabilities.sensing
        assert sensing is not None
        assert set(sensing.cameras) <= set(env_target.profile.cameras)
        assert sensing.depth == self.has_depth
        assert found.config_hash
        assert found.capabilities.reset, "an env says how its episodes are reset"
        assert env_target.reset_by in found.capabilities.reset, env_target.reset_by

    async def test_refuses_a_reset_it_does_not_offer(self, env_target: EnvTarget) -> None:
        # integration: on an env that does not refuse, the open goes through and opens an episode
        offered = (await _public(env_target).discover()).capabilities.reset
        for kind in ("seed", "initial_state"):
            if kind not in offered:
                spec = EpisodeSpec(task=env_target.task, seed=env_target.seed, reset_by=kind)
                await _raises(ProtocolError, "invalid_request", _control(env_target).open(spec))

    @unit
    async def test_control_plane_needs_its_token(self, env_target: EnvTarget) -> None:
        for token in (None, SecretStr(secrets.token_urlsafe(16))):
            control = env_target.connect(EnvControl, env_target.url, token)
            await _raises(ProtocolError, "unauthorized", control.judge(EpisodeId("ep-none")))
        unknown = _control(env_target).judge(EpisodeId("ep-000000000000"))
        await _raises(ProtocolError, "not_found", unknown)

    async def test_lifecycle(self, env_target: EnvTarget) -> None:
        control = _control(env_target)
        offered = (await _public(env_target).discover()).capabilities
        opened = await _open(env_target)
        assert opened.solvable and opened.instruction and opened.agent_url
        if not offered.expert:
            assert opened.prerun_attempts == 0, "an env without an expert runs no pre-run"
        assert opened.embodiment == env_target.profile.name
        assert set(opened.cameras) <= set(env_target.profile.cameras)
        assert (opened.step_limit is not None) == self.step_limit
        verdict = await control.judge(opened.episode_id)
        assert (verdict.success, verdict.success_step, verdict.steps_used) == (False, None, 0)
        assert verdict.step_limit == opened.step_limit
        scored = "terminal_score" in offered.success
        assert (verdict.score is not None) == scored, "a score iff judged so"
        await control.close(opened.episode_id)
        await _raises(EpisodeEnded, "episode_ended", control.judge(opened.episode_id))
        reused = await _open(env_target, instruction="reuse this instruction")
        assert reused.instruction == "reuse this instruction"
        other_task = EpisodeSpec(
            task=TaskName("no_such_task"), seed=env_target.seed, reset_by=env_target.reset_by
        )
        await _raises(ProtocolError, "invalid_request", control.open(other_task))
        undeclared = EpisodeSpec(
            task=env_target.task,
            seed=env_target.seed,
            options={"no_such_option": True},
            reset_by=env_target.reset_by,
        )
        await _raises(ProtocolError, "invalid_request", control.open(undeclared))
        if not self.records:
            recorded = EpisodeSpec(
                task=env_target.task,
                seed=env_target.seed,
                record=True,
                reset_by=env_target.reset_by,
            )
            await _raises(ProtocolError, "invalid_request", control.open(recorded))

    async def test_expert_prerun(self, env_target: EnvTarget) -> None:
        control = _control(env_target)
        if not (await _public(env_target).discover()).capabilities.expert:
            not_applicable("the env declares no expert pre-run (EnvCapabilities.expert)")
        opened = await _open(env_target)
        assert opened.solvable and 1 <= opened.prerun_attempts <= 3
        if env_target.unsolvable_seed is not None:
            skipped = await _open(env_target, seed=env_target.unsolvable_seed)
            assert not skipped.solvable and skipped.instruction is None
            await _raises(RSError, None, control.judge(skipped.episode_id))

    async def test_agent_tokens_are_per_episode(self, env_target: EnvTarget) -> None:
        first = await _open(env_target)
        agent = _agent(env_target, first)
        await agent.robot(first.episode_id)
        for token in (None, SecretStr(secrets.token_urlsafe(16))):
            other = env_target.connect(EnvAgent, first.agent_url, token)
            await _raises(ProtocolError, "unauthorized", other.robot(first.episode_id))
        on_agent_port = env_target.connect(EnvControl, first.agent_url, env_target.token)
        await _raises(ProtocolError, "not_found", on_agent_port.judge(first.episode_id))
        second = await _open(env_target)  # closes the first episode
        await _raises(ProtocolError, "unauthorized", agent.robot(second.episode_id))
        await _raises(EpisodeEnded, "episode_ended", agent.robot(first.episode_id))
        await _agent(env_target, second).robot(second.episode_id)

    async def test_robot_state_matches_the_profile(self, env_target: EnvTarget) -> None:
        opened = await _open(env_target)
        state = await _agent(env_target, opened).robot(opened.episode_id)
        profile = env_target.profile
        assert state.step == 0 and state.step_limit == opened.step_limit
        assert set(state.arms) == set(profile.arms), "the arms the profile declares, by name"
        assert len(state.qpos_cmd) == profile.action_dim
        for arm, arm_state in state.arms.items():
            spec = profile.arms[arm]
            start, stop = spec.action
            assert arm_state.joints_cmd == state.qpos_cmd[start:stop]
            assert arm_state.gripper_cmd == state.qpos_cmd[spec.gripper_index]
            tcp = profile.tcp_pose(arm, arm_state.ee).as_list()
            assert np.allclose(tcp, arm_state.tcp.as_list(), atol=1e-3), (arm, "ee_to_tcp")
            assert arm_state.joints is not None and arm_state.gripper is not None
            assert np.allclose(arm_state.joints, arm_state.joints_cmd, atol=1e-3), "at rest"
            assert abs(arm_state.gripper - arm_state.gripper_cmd) < 0.05, "at rest"

    async def test_observe(self, env_target: EnvTarget) -> None:
        opened = await _open(env_target)
        agent = _agent(env_target, opened)
        first = await agent.observe(opened.episode_id, ObserveRequest())
        assert set(first.cameras) == set(opened.cameras)
        assert first.robot == await agent.robot(opened.episode_id) and first.step == 0
        for name, frame in first.cameras.items():
            spec = env_target.profile.cameras[name]
            camera = frame.camera
            assert (camera.width, camera.height) == (spec.width, spec.height)
            assert (frame.rgb.dtype, frame.rgb.shape) == ("uint8", (spec.height, spec.width, 3))
            if self.has_depth:
                assert frame.depth is not None
                assert (frame.depth.dtype, frame.depth.shape) == (
                    "float32",
                    (spec.height, spec.width),
                )
            K, cam2world = np.array(camera.K), np.array(camera.cam2world)
            assert (
                K[0, 0] > 0
                and K[1, 1] > 0
                and 0 < K[0, 2] < spec.width
                and 0 < K[1, 2] < spec.height
            )
            rotation = cam2world[:3, :3]
            assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
            if spec.arm is not None:  # a wrist camera rides near its arm's ee
                ee = np.array(first.robot.arms[spec.arm].ee.p)
                assert np.linalg.norm(cam2world[:3, 3] - ee) < 0.3, name
        only_head = await agent.observe(
            opened.episode_id, ObserveRequest(cameras=[next(iter(first.cameras))], depth=False)
        )
        assert len(only_head.cameras) == 1 and next(iter(only_head.cameras.values())).depth is None
        await _raises(
            ProtocolError,
            "invalid_request",
            agent.observe(opened.episode_id, ObserveRequest(cameras=["no_such_camera"])),
        )

    async def test_execute_counts_steps_and_is_idempotent(self, env_target: EnvTarget) -> None:
        opened = await _open(env_target)
        agent = _agent(env_target, opened)
        rest = await agent.robot(opened.episode_id)
        hold = _batch([ActionStep(qpos=rest.qpos_cmd)])
        done = await agent.execute(opened.episode_id, hold)
        assert (done.stopped, done.executed, len(done.states)) == ("completed", 1, 1)
        assert done.states[0].index == 0 and done.states[0].robot.step == 1
        assert await agent.execute(opened.episode_id, hold) == done  # a replay, not a new step
        assert (await agent.robot(opened.episode_id)).step == 1
        other_steps = hold.model_copy(update={"steps": [*hold.steps, *hold.steps]})
        await _raises(
            ProtocolError, "invalid_request", agent.execute(opened.episode_id, other_steps)
        )
        too_short = _batch([ActionStep(qpos=rest.qpos_cmd[:-1])])
        await _raises(ProtocolError, "invalid_request", agent.execute(opened.episode_id, too_short))
        settle = _batch(
            [ActionStep(qpos=rest.qpos_cmd, until_gripper_stable=True, max_repeats=3)] * 2
        )
        result = await agent.execute(opened.episode_id, settle)
        assert result.stopped == "completed" and result.executed == len(result.states)
        assert 2 <= result.executed <= 8
        assert [s.robot.step for s in result.states] == list(range(2, 2 + result.executed))
        verdict = await _control(env_target).judge(opened.episode_id)
        assert verdict.steps_used == 1 + result.executed

    async def test_concurrent_execute_is_busy(self, env_target: EnvTarget) -> None:
        opened = await _open(env_target)
        agent = _agent(env_target, opened)
        qpos = (await agent.robot(opened.episode_id)).qpos_cmd
        batches = [
            _batch([ActionStep(qpos=qpos)] * self.busy_steps),
            _batch([ActionStep(qpos=qpos)]),
        ]
        outcomes: list[ExecuteResult | BaseException] = []

        async def run(batch: ActionBatch) -> None:
            try:
                outcomes.append(await agent.execute(opened.episode_id, batch))
            except InfraError as error:
                outcomes.append(error)

        async with anyio.create_task_group() as tg:
            for batch in batches:
                tg.start_soon(run, batch)
        busy = [o for o in outcomes if isinstance(o, InfraError)]
        assert len(busy) == 1 and busy[0].code == "busy" and busy[0].retryable
        retried = [
            b
            for b in batches
            if not any(isinstance(o, ExecuteResult) and o.batch_id == b.batch_id for o in outcomes)
        ]
        assert (await agent.execute(opened.episode_id, retried[0])).stopped == "completed"
        assert (await agent.robot(opened.episode_id)).step == self.busy_steps + 1

    async def test_step_limit(self, env_target: EnvTarget) -> None:
        if not self.exhaust_step_limit:
            return
        opened = await _open(env_target)
        assert opened.step_limit is not None
        agent = _agent(env_target, opened)
        qpos = (await agent.robot(opened.episode_id)).qpos_cmd
        result = await agent.execute(
            opened.episode_id, _batch([ActionStep(qpos=qpos)] * (opened.step_limit + 1))
        )
        assert (result.stopped, result.executed) == ("step_limit", opened.step_limit)
        again = await agent.execute(opened.episode_id, _batch([ActionStep(qpos=qpos)]))
        assert (again.stopped, again.executed) == ("step_limit", 0)

    async def test_ground_truth_is_on_the_control_plane(self, env_target: EnvTarget) -> None:
        control = _control(env_target)
        if not (await _public(env_target).discover()).capabilities.truth:
            not_applicable("the env declares no ground truth (EnvCapabilities.truth)")
        first = await _open(env_target)
        diagnosis = await control.diagnose(first.episode_id)
        assert diagnosis.objects and not diagnosis.success
        for truth in diagnosis.objects:
            assert truth.displacement_m < 0.005, f"{truth.name} moved before any action"
        snapshot = await control.gt_snapshot(first.episode_id)
        assert snapshot.step == 0 and set(snapshot.objects) == {t.name for t in diagnosis.objects}
        again = await _open(env_target)  # the same seed rebuilds the same scene
        repeated = await control.diagnose(again.episode_id)
        for a, b in zip(diagnosis.objects, repeated.objects, strict=True):
            assert a.name == b.name
            assert np.allclose(a.initial.as_list(), b.initial.as_list(), atol=1e-6), a.name

    async def test_expert_trace(self, env_target: EnvTarget) -> None:
        control = _control(env_target)
        spec = EpisodeSpec(
            task=env_target.task,
            seed=env_target.seed,
            prerun_attempts=3,
            trace_expert=True,
            reset_by=env_target.reset_by,
        )
        if not self.traces_expert:
            await _raises(ProtocolError, "invalid_request", control.open(spec))
            return
        opened = await control.open(spec)
        assert opened.solvable
        trace = await control.expert_trace(opened.episode_id)
        assert trace.episode_id == opened.episode_id
        moves = [call for call in trace.calls if call.kind == "move" and not call.refused]
        assert moves, trace.calls
        for call in moves:
            assert call.actions and set(call.api) == set(call.actions) == set(call.openings), call
            assert set(call.actions) <= set(env_target.profile.arms), call
        untraced = await _open(env_target)
        await _raises(ProtocolError, "invalid_request", control.expert_trace(untraced.episode_id))

    async def test_a_reset_declared_once_is_not_reopened(self, env_target: EnvTarget) -> None:
        if (await _public(env_target).discover()).capabilities.reruns:
            return
        control = _control(env_target)
        opened = await _open(env_target)
        await control.close(opened.episode_id)
        await _raises(ProtocolError, "invalid_request", _open(env_target))

    async def test_closed_episode_errors(self, env_target: EnvTarget) -> None:
        control = _control(env_target)
        opened = await _open(env_target)
        agent = _agent(env_target, opened)
        qpos = (await agent.robot(opened.episode_id)).qpos_cmd
        await control.close(opened.episode_id)
        episode = opened.episode_id
        for call in (
            agent.robot(episode),
            agent.observe(episode, ObserveRequest()),
            agent.execute(episode, _batch([ActionStep(qpos=qpos)])),
            control.judge(episode),
            control.diagnose(episode),
            control.gt_snapshot(episode),
            control.expert_trace(episode),
            control.close(episode),
        ):
            await _raises(EpisodeEnded, "episode_ended", call)


def _public(target: EnvTarget) -> EnvPublic:
    return cast(EnvPublic, target.connect(EnvPublic, target.url, None))


def _control(target: EnvTarget) -> EnvControl:
    return cast(EnvControl, target.connect(EnvControl, target.url, target.token))


def _agent(target: EnvTarget, opened: EpisodeOpened) -> EnvAgent:
    return cast(EnvAgent, target.connect(EnvAgent, opened.agent_url, opened.agent_token))


async def _open(
    target: EnvTarget, *, seed: int | None = None, instruction: str | None = None
) -> EpisodeOpened:
    """Open `seed` with one expert pre-run, or `target.seed` with up to three (see `EnvTarget`)."""
    spec = EpisodeSpec(
        task=target.task,
        seed=target.seed if seed is None else seed,
        instruction=instruction,
        prerun_attempts=3 if seed is None else 1,
        reset_by=target.reset_by,
    )
    return await _control(target).open(spec)


def _batch(steps: list[ActionStep]) -> ActionBatch:
    return ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps)


async def _raises(kind: type[RSError], code: str | None, call: Any) -> None:
    try:
        await call
    except kind as error:
        assert code is None or error.code == code, f"expected {code}, got {error.code}: {error}"
        return
    raise AssertionError(f"expected {kind.__name__}({code})")
