"""The BEHAVIOR env server's config (`python -m roboscript_behavior.sim.host --config <file>`)."""

from pydantic import model_validator
from typing_extensions import Self

from roboscript.envs.skeleton import SkeletonConfig
from roboscript_behavior.benchmark import BEHAVIOR


class BehaviorConfig(SkeletonConfig):
    """One env process serves one `task` (an activity). The challenge opens each instance once
    (`EnvCapabilities.reruns` False), which the server enforces by default; `once_per_reset:
    false` turns it off for a local copy that reloads an instance."""

    @model_validator(mode="after")
    def _known_task(self) -> Self:
        if self.task not in BEHAVIOR.tasks:
            raise ValueError(f"the BEHAVIOR sample table has no task {self.task!r}")
        return self
