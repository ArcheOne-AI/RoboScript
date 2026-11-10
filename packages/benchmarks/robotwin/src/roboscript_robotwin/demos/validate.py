"""Replay validation of a recorded demonstration (docs/concepts/learning.md): at the demo seed the
expert runs again with its official API calls traced, the calls replay through our executor
(`roboscript_robotwin.demos.replay`), and the package's keypoints judge the executed path
the way S1's referee judges an attempt (the TCP along the joint line between the executed steps, by
FK through the motion service). An attempt passes with the official success latched and every
required keypoint passed. The planner is stochastic, so a demonstration gets several attempts and
passes with the first that does; when none does, its candidate is replaced. Each attempt's trace,
ground truth, goes to `referee/expert_trace<n>.json` and the outcome to `referee/validation.json`,
with the sha256 of the `package.json` it judged by: the calls serve this check only, never a
package. The services come from `replay.motion_services` and `replay.env_server`.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import anyio
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.env import EnvAgent, EnvControl, EpisodeOpened
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.motion import Planner
from roboscript.demos import Verdict, arm_progress, judge
from roboscript.demos.manifest import content_sha256
from roboscript.demos.paths import ArmPaths
from roboscript.embodiments import EmbodimentProfile
from roboscript.learning import DemoPackage, load_package
from roboscript.learning.demo import PACKAGE_FILE
from roboscript.learning.episode import Trajectory
from roboscript_robotwin.demos.replay import ReplayConfig, replay

VALIDATION: Final = "validation.json"


class ValidationAttempt(BaseModel):
    """One replay: `success` is official (None: the expert pre-run failed at the seed),
    `failure` the reason for a replay that did not succeed, `verdict` the keypoint referee's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    success: bool | None
    failure: str | None
    verdict: Verdict | None
    calls: NonNegativeInt | None = None
    take_actions: NonNegativeInt | None = None
    seconds: float


class Validation(BaseModel):
    """`referee/validation.json`: the candidate (the expert data's `episode` and its seed), the
    sha256 of the package's `package.json`, and the attempts; it passed with the first attempt
    that did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode: NonNegativeInt
    seed: NonNegativeInt
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    attempts: list[ValidationAttempt]


class _Recorded:
    """An env agent plane that keeps the robot state after every executed step."""

    def __init__(self, env: EnvAgent) -> None:
        self._env = env
        self.trajectory: Trajectory | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        if self.trajectory is None:
            self.trajectory = Trajectory(await self._env.robot(episode_id))
        result = await self._env.execute(episode_id, batch)
        await self.trajectory.add(batch, result)
        return result


async def validate(
    control: EnvControl,
    agent_of: Callable[[EpisodeOpened], EnvAgent],
    planner: Planner,
    profile: EmbodimentProfile,
    directory: Path,
    episode: int,
    referee: Path,
    config: ReplayConfig,
    *,
    attempts: int,
    spacing: tuple[float, float],
    grip_threshold: float,
) -> Validation:
    """Validate the package in `directory` (of the expert data's `episode`) on an env server of
    its task, with up to `attempts` replays; the traces and the outcome go to `referee`. The
    referee's path `spacing` (rad of joint motion, m of TCP travel) and `grip_threshold` are as in
    `ImitateConfig`."""
    digest, package = await anyio.to_thread.run_sync(_package, directory)
    tried: list[ValidationAttempt] = []
    for n in range(1, attempts + 1):
        attempt, trace = await _attempt(
            control, agent_of, planner, profile, package, config, spacing, grip_threshold
        )
        tried.append(attempt)
        _keep(referee / f"expert_trace{n}.json", trace)
        if attempt.passed:
            break
    result = Validation(
        episode=episode,
        seed=package.seed,
        package_sha256=digest,
        passed=tried[-1].passed,
        attempts=tried,
    )
    _keep(referee / VALIDATION, result.model_dump(mode="json"))
    return result


async def _attempt(
    control: EnvControl,
    agent_of: Callable[[EpisodeOpened], EnvAgent],
    planner: Planner,
    profile: EmbodimentProfile,
    package: DemoPackage,
    config: ReplayConfig,
    spacing: tuple[float, float],
    grip_threshold: float,
) -> tuple[ValidationAttempt, dict[str, Any] | None]:
    """One replay and its verdict, with the trace it replayed (None: the pre-run failed)."""
    started = time.monotonic()
    recorded: list[_Recorded] = []

    def tracked(opened: EpisodeOpened) -> EnvAgent:
        recorded.append(_Recorded(agent_of(opened)))
        return recorded[-1]  # type: ignore[return-value]

    row = await replay(control, tracked, planner, profile, package.task, package.seed, config)
    trace = row.pop("trace", None)
    trajectory = recorded[0].trajectory if recorded else None
    if trajectory is None:  # unsolvable, or nothing ran
        failure = row.get("failure") if row["solvable"] else "unsolvable"
        seconds = time.monotonic() - started
        missing = ValidationAttempt(
            passed=False, success=row.get("success"), failure=failure, verdict=None, seconds=seconds
        )
        return missing, trace
    paths = ArmPaths(planner, profile, *spacing)
    await paths.extend(trajectory.arrays())
    executed = paths.paths()
    keypoints = package.keypoints
    progress = {
        arm: arm_progress(
            kps,
            executed[arm],
            grip_threshold,
            keypoints.pointing[arm],
            success_frame=keypoints.success_frame,
        )
        for arm, kps in keypoints.arms.items()
    }
    verdict = judge(keypoints, progress, row["success"], paths.sources())
    attempt = ValidationAttempt(
        passed=verdict.passed,
        success=row["success"],
        failure=row["failure"],
        verdict=verdict,
        calls=row["calls"],
        take_actions=row["take_actions"],
        seconds=time.monotonic() - started,
    )
    return attempt, trace


def _package(directory: Path) -> tuple[str, DemoPackage]:
    return content_sha256(directory / PACKAGE_FILE), load_package(directory)


def _keep(path: Path, data: dict[str, Any] | None) -> None:
    """Ground truth under `referee/` (nothing when there is none)."""
    if data is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def validation_text(result: Validation | None) -> str:
    """One line on a validation: pending, or passed / failed with its attempts."""
    if result is None:
        return "pending"
    last = result.attempts[-1]
    reasons = [] if last.verdict is None else last.verdict.reasons
    if result.passed:
        return f"passed at attempt {len(result.attempts)}: " + "; ".join(reasons)
    why = [*([last.failure] if last.failure else []), *reasons]
    return f"failed {len(result.attempts)} attempts; the last: " + "; ".join(why)
