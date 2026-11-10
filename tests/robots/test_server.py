"""The robot server over the fake backend: speed caps over ticks, the e-stop, and the heartbeat
watchdog. (The planes' contract-level behavior is `roboscript_robots.testing.RobotConformance`,
run in `test_conformance.py`.)"""

import math
from pathlib import Path

import anyio
import anyio.lowlevel
import numpy as np
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.capabilities import ManualReset, OpenSpec
from roboscript.contracts.errors import EpisodeEnded
from roboscript.contracts.ids import BatchId, new_id
from roboscript_robots import single_arm
from roboscript_robots.base import (
    Clock,
    FakeBackend,
    RobotConfig,
    RobotServer,
    ScriptedOperator,
)
from roboscript_robots.testing import FakeTime


def _batch(*steps: list[float], stable: bool = False) -> ActionBatch:
    return ActionBatch(
        batch_id=BatchId(new_id("batch")),
        steps=[ActionStep(qpos=list(s), until_gripper_stable=stable) for s in steps],
    )


def _moved(config: RobotConfig, **changes: float) -> list[float]:
    """The home vector with the named action indexes (`i3=0.1`) changed."""
    qpos = config.home_vector()
    for key, value in changes.items():
        qpos[int(key[1:])] = value
    return qpos


async def _server(
    tmp_path: Path, config: RobotConfig = single_arm.FAKE_CONFIG, **backend: float
) -> tuple[RobotServer, FakeBackend, FakeTime, ScriptedOperator]:
    fake_time, operator = FakeTime(), ScriptedOperator([True])
    hardware = FakeBackend(config, **backend)
    server = RobotServer(
        config,
        hardware,
        operator,
        LocalBlobStore(tmp_path),
        clock=Clock(fake_time.now, fake_time.sleep),
    )
    await server.start()
    return server, hardware, fake_time, operator


async def _open(server: RobotServer):
    task = next(iter(server.config.tasks))
    return await server.open(OpenSpec(task=task, reset=ManualReset(prompt="set the scene")))


# Speed caps ----------------------------------------------------------------------------------


async def test_a_far_target_is_stretched_over_ticks_and_never_commanded_faster_than_the_caps(
    tmp_path: Path,
) -> None:
    server, hardware, _, _ = await _server(tmp_path)
    opened = await _open(server)
    limits = server.config.limits
    hardware.commands.clear()
    target = _moved(server.config, i0=0.4, i1=0.3)  # 0.5 m of tcp travel at 1 m/s
    result = await server.execute(opened.episode_id, _batch(target))
    assert result.executed == 1
    ticks = np.array(hardware.commands)
    assert len(ticks) >= math.ceil(0.5 / (limits.max_tcp_speed_m_s / limits.control_hz))
    per_tick = np.abs(np.diff(ticks[:, :3], axis=0))
    assert np.all(
        np.linalg.norm(per_tick, axis=1) <= limits.max_tcp_speed_m_s / limits.control_hz + 1e-9
    )
    assert np.all(per_tick <= limits.max_joint_speed_rad_s / limits.control_hz + 1e-9)
    assert hardware.commands[-1] == target, "arrives exactly on the target"


# Workspace and stops -------------------------------------------------------------------------


# E-stop --------------------------------------------------------------------------------------


async def test_the_estop_interrupts_a_batch_in_flight_and_halts_the_hardware(
    tmp_path: Path,
) -> None:
    server, hardware, _, _ = await _server(tmp_path)
    opened = await _open(server)
    far = _batch(_moved(server.config, i0=0.5), _moved(server.config, i0=-0.5))
    outcome = []

    async def run() -> None:
        outcome.append(await server.execute(opened.episode_id, far))

    async with anyio.create_task_group() as tg:
        tg.start_soon(run)
        while len(hardware.commands) < 20:  # the move is under way
            await anyio.lowlevel.checkpoint()
        await server.estop("someone pressed it")
    (result,) = outcome
    assert result.stopped == "error" and "emergency stop: someone pressed it" in (
        result.error or ""
    )
    assert result.executed == 0 < len(far.steps)
    assert hardware.halts >= 1
    sent = len(hardware.commands)
    with pytest.raises(EpisodeEnded, match="emergency stop"):
        await server.execute(opened.episode_id, _batch(server.config.home_vector()))
    assert len(hardware.commands) == sent, "nothing is commanded after the e-stop"
    await server.estop("again")  # idempotent; the first reason stays
    assert server.safety.estopped == "someone pressed it"


# Heartbeat -----------------------------------------------------------------------------------


async def test_the_watchdog_halts_the_robot_on_silence_even_when_no_action_runs(
    tmp_path: Path,
) -> None:
    server, hardware, fake_time, _ = await _server(tmp_path)
    opened = await _open(server)
    halts = hardware.halts
    fake_time.advance(server.config.limits.heartbeat_timeout_s * 1.5)
    await server.enforce()
    assert hardware.halts == halts + 1
    await server.enforce()
    assert hardware.halts == halts + 1, "halted once per stop"
    with pytest.raises(EpisodeEnded, match="no heartbeat"):
        await server.execute(opened.episode_id, _batch(server.config.home_vector()))


# Manual reset, judgment, steps ----------------------------------------------------------------


# What the server cannot do yet ----------------------------------------------------------------


# Process entry --------------------------------------------------------------------------------
