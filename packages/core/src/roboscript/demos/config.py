"""`configs/demos.yaml`: keypoint extraction, the choice of demonstrations, and their re-recording
into the packages S1 learns from."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt

from roboscript.demos.compare import CompareConfig
from roboscript.demos.disturbance import DisturbanceConfig
from roboscript.demos.keypoints import KeypointConfig
from roboscript.demos.replay import STRATEGY
from roboscript.demos.selection import SelectionConfig


class RecordingConfig(BaseModel):
    """Re-recording a chosen demonstration: replay `strategies` in order until one latches
    success, `tail_holds` hold actions at most after the last frame, once the candidate's expert
    pre-run passed `prerun_checks` runs in a row; `prerun_attempts` expert pre-runs per open; a
    re-run of the expert (tasks whose recorded trajectory may not match the scene) gets
    `expert_attempts`. The package holds every `every`-th frame and each keypoint's
    and the success frame, and `keyframes` keyframes at most (drawn as the sensing toolset draws
    its views); the review video runs at `video_fps`; the replay validation gets
    `validation_attempts` (its planner is stochastic), its keypoint referee the path spacing of
    S1's (`path_step_rad` of joint motion, `path_step_m` of TCP travel)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategies: list[Annotated[str, Field(pattern=f"^({STRATEGY})$")]] = Field(
        default_factory=lambda: ["s1", "s8", "rest"], min_length=1
    )
    tail_holds: NonNegativeInt = 4
    prerun_checks: PositiveInt = 3
    prerun_attempts: PositiveInt = 3
    expert_attempts: PositiveInt = 3
    every: PositiveInt = 5
    keyframes: Annotated[int, Field(ge=1, le=24)] = 24
    video_fps: PositiveInt = 15
    validation_attempts: PositiveInt = 3
    path_step_rad: PositiveFloat = 0.01
    path_step_m: PositiveFloat = 0.005


class DemosConfig(BaseModel):
    """`disturbance` has no default: how long a release settles depends on the benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keypoints: KeypointConfig = KeypointConfig()
    selection: SelectionConfig = SelectionConfig()
    recording: RecordingConfig = RecordingConfig()
    compare: CompareConfig = CompareConfig()
    disturbance: DisturbanceConfig
