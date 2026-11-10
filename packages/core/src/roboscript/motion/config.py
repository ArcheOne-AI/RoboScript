"""Configuration of the cuRobo planner service.

The robot comes from an embodiment profile: each arm's joints, ee-to-TCP offset, curobo yml and
base pose, and the asset directory. The env's planning setup (its collision table, its
arms' yaws) is its planner binding, given as data (`planner_profile`,
`contracts.skills.PlannerProfile`). The config names the arms to plan for and may override any
profile value per arm.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PositiveInt,
    field_validator,
)

from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.robot import Arm
from roboscript.contracts.skills import PlannerProfile
from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.settings import repo_root


def _inside(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"must be a relative path without '..', got {path}")
    return path


Inside = Annotated[Path, AfterValidator(_inside)]


class ArmConfig(BaseModel):
    """An arm to plan for: what the planner adds to its profile, and overrides of that."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    yaw_rad: FiniteFloat | None = None
    """The turn about z that `CuroboPlanner.plan_path` gives targets after adding `frame_bias`;
    None: the planner profile's for the arm (none if it names none)."""
    curobo_yml: Inside | None = None
    """Overrides the profile's `planner.curobo`, relative to the asset directory."""
    joints: list[str] | None = Field(default=None, min_length=1)
    """Overrides the profile's joints; checked against the yml's planned joints."""
    ee_to_tcp: Vec3 | None = None
    base: Pose | None = None
    """Overrides the profile's `planner.base`: the root pose the planner frame hangs off."""


@dataclass(frozen=True)
class ArmSetup:
    """One arm as the planner builds it, every value resolved."""

    curobo_yml: Path
    joints: list[str]
    ee_to_tcp: tuple[float, float, float]
    base: Pose
    yaw_rad: float
    stops: list[tuple[float, float]] | None = None
    """The joints' physical stops (the profile's), planned within where narrower than the
    URDF's limits; none when the config overrides the profile's joints."""


class CuroboConfig(BaseModel):
    """The planner of one embodiment on one GPU."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    embodiment: str
    """A profile registered in `roboscript.embodiments`, or its YAML file: absolute, or relative
    to the repository root."""
    arms: dict[Arm, ArmConfig] = Field(min_length=1)
    planner_profile: PlannerProfile
    """The env's planning setup (its binding): the collision table, the arms' yaws."""
    assets_root: Path | None = None
    """Absolute; overrides the profile's `assets_root`, the directory of the curobo ymls."""
    device: Literal["cuda:0"] = "cuda:0"
    """The first visible GPU: choose it with `CUDA_VISIBLE_DEVICES` (cuRobo puts some tensors on
    the default CUDA device)."""
    batch_size: PositiveInt = 10
    """Goals per batch-planner call, the count it warms up for."""
    warmup: bool = True
    """Warm up the planners; without it the first calls are slow and the seed state differs."""

    @field_validator("embodiment")
    @classmethod
    def _profile_ref(cls, ref: str) -> str:
        if ref.endswith((".yaml", ".yml")):
            path = Path(ref)
            if not path.is_absolute():
                _inside(path)
        return ref

    @field_validator("assets_root")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        if path is not None and not path.is_absolute():
            raise ValueError(f"must be an absolute path, got {path}")
        return path

    def profile(self) -> EmbodimentProfile:
        if not self.embodiment.endswith((".yaml", ".yml")):
            return load_embodiment(self.embodiment)  # a registered name
        return load_embodiment(repo_root() / self.embodiment)  # an absolute path stays as it is

    def resolve(self) -> tuple[Path, dict[Arm, ArmSetup]]:
        """The asset directory, which must exist, and each configured arm's setup."""
        profile = self.profile()
        directory = self.assets_root or profile.asset("")
        if not directory.is_dir():
            raise FileNotFoundError(f"the asset directory {directory} does not exist")
        setups = {}
        for arm, cfg in self.arms.items():
            if arm not in profile.arms:
                raise ValueError(f"embodiment {profile.name} has no {arm} arm")
            spec, planner = profile.arms[arm], profile.arms[arm].planner
            yml = cfg.curobo_yml or (planner.curobo if planner else None)
            base = cfg.base or (planner.base if planner else None)
            if yml is None or base is None:
                raise ValueError(f"{arm} arm: give curobo_yml and base (the profile has none)")
            joints = cfg.joints or spec.joints
            setups[arm] = ArmSetup(
                curobo_yml=directory / yml,
                joints=joints,
                ee_to_tcp=cfg.ee_to_tcp or spec.ee_to_tcp,
                base=base,
                yaw_rad=self.planner_profile.yaw_rad.get(arm, 0.0)
                if cfg.yaw_rad is None
                else cfg.yaw_rad,
                stops=spec.stops if joints == spec.joints else None,
            )
        return directory, setups
