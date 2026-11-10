"""The RoboTwin env server's config (`python -m roboscript_robotwin.sim.host --config <file>`)."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from roboscript.contracts.ids import TaskName
from roboscript.envs import ServerConfig
from roboscript.settings import path_variable
from roboscript_robotwin.benchmark import ROBOTWIN

PERCEPTION_WORDS: Final = {
    "scene_words": "tabletop",
    "anchor_words": "a robot's head camera",
    "camera_words": {
        "left_camera": "a camera on the robot's left wrist",
        "right_camera": "a camera on the robot's right wrist",
    },
}
"""What RoboTwin's scenes and cameras are in the VLM prompts' words
(`PerceptionConfig.scene_words`, `anchor_words`, `camera_words`): a config that sets them keeps
its own."""
PERCEPTION_TABLE: Final = {"xy_box": (-0.62, 0.62, -0.37, 0.37), "z_range": (0.60, 0.80)}
"""Where RoboTwin's table is searched for (`PerceptionConfig.table`): the nominal table area
(x0, x1, y0, y1) and the heights of its top."""
PERCEPTION_ANCHOR: Final = "head_camera"
"""The camera RoboTwin's frames are anchored on (`PerceptionConfig.anchor`)."""
LINT_OVERLAYS: Final = {
    "memory": "examples/simBenchmarks/RoboTwin/run/wicl/memory_lint.yaml",
    "rsi": "examples/simBenchmarks/RoboTwin/run/rsi/lint.yaml",
}
"""RoboTwin's lint vocabulary over the recipes' generic rules, relative to the repository root."""
ROBOTWIN_ROOT: Final = "ROBOTWIN_ROOT"
"""The path variable this package declares: RoboTwin's checkout, from the environment or `.env`
(`roboscript.settings.path_variable`); its embodiment's `assets_root` is under it."""


def with_perception_place(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    """Perception pipeline values with RoboTwin's anchor camera and table area where they set
    none (`PERCEPTION_ANCHOR`, `PERCEPTION_TABLE`, each table key on its own)."""
    table = pipeline.get("table") or {}
    return {"anchor": PERCEPTION_ANCHOR, **pipeline, "table": {**PERCEPTION_TABLE, **table}}


def with_perception_binding(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    """Perception pipeline values with RoboTwin's perception binding where they set none: its
    `PERCEPTION_WORDS` and `with_perception_place`."""
    return with_perception_place({**PERCEPTION_WORDS, **pipeline})


def robotwin_root() -> Path | None:
    """RoboTwin's checkout (`ROBOTWIN_ROOT`); None where it is unset."""
    return path_variable(ROBOTWIN_ROOT)


EXPERT_DEMO_ROOT: Final = "RS_EXPERT_DEMO_ROOT"
"""The path variable this package declares for the expert demonstrations, laid out as
`<task>/<task config>/data/episode<N>.hdf5`."""


def expert_demo_root() -> Path | None:
    """The expert demonstrations (`RS_EXPERT_DEMO_ROOT`); None where it is unset."""
    return path_variable(EXPERT_DEMO_ROOT)


class RoboTwinConfig(BaseModel):
    """One env process serves one `task` (each task object holds its own robot and planners, so
    switching tasks would grow GPU memory); it builds and renders it once before reporting ready.
    `mode` picks our task config (`rs_agentic_clean.yml` or `rs_agentic_randomized.yml`); frames
    go to `blob_dir`, by default the stack's (`RS_BLOB_DIR`); `robotwin_root` defaults to
    `ROBOTWIN_ROOT`, both from the environment or `.env`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    blob_dir: Path | None = None
    mode: Literal["clean", "randomized"] = "clean"
    robotwin_root: Path | None = None
    server: ServerConfig = ServerConfig()

    @model_validator(mode="after")
    def _known_task(self) -> Self:
        if self.task not in ROBOTWIN.tasks:
            raise ValueError(f"RoboTwin has no task {self.task!r}")
        return self
