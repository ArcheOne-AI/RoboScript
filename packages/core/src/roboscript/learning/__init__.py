"""The multi-stage harness (docs/concepts/learning.md): S1 `imitate`, S2 `practice` with freeze,
and `evaluate`, run per task by the `ProtocolRunner`, which alone holds the control plane (seeds,
env episodes, budgets, referees, Memory versions).

- `runner`: `ProtocolRunner` (the Hub, the memory toolset, env control planes, the store, the
  recipe, brains per role; `run`, `run_tasks`), `load_protocol`, `brain_factory` and
  `brain_maker` (the brains of S1, S2 and P3), `control_client`
- `episode`: `EpisodeDriver` (one episode end to end: env open, Hub episode, brain, judge and
  diagnosis, the episode directory; `run_counted` re-runs infrastructure outcomes),
  `Episode`, `EpisodeOutcome`, `Trajectory`
- `evaluate`: a protocol (entry-point group `roboscript.protocols`; `imitate` and `practice`
  are `roboscript_methods.wicl`'s); `set_aside` moves what an interrupted run left out of
  the way
- `ledger`: `TryRecord` (one try as a protocol records it) and `wilson`
- `hooks`: `Hooks`, the debug hooks; `demo`: `DemoPackage`, what S1 reads of a demonstration,
  its ground-truth layer `DemoTruth`, its motion starts `DemoSegments` (`load_segments`,
  `with_segments`), `build_package` and `validate_package` from a re-recording, `import_blobs`
- `config`: `LearningConfig` (`configs/learning.yaml`), `ProtocolRecipe` (`protocols.yaml`)
- `context`: `TaskContext`, `LearningProtocol`; `events`: `EventRouter`
- `agent_host`: the Agent host's process for P3 (`python -m roboscript.learning.agent_host`),
  which a benchmark's own evaluation client drives
"""

from roboscript.learning.config import (
    BrainSpec,
    LearningConfig,
    ProtocolRecipe,
    RoleConfig,
    load_protocol_recipe,
)
from roboscript.learning.context import LearningProtocol, TaskContext
from roboscript.learning.demo import (
    DemoPackage,
    DemoSegments,
    DemoTruth,
    Keyframe,
    Operation,
    build_package,
    demo_truth,
    goal_frames,
    import_blobs,
    keyframe_plan,
    load_demo_truth,
    load_package,
    load_segments,
    operation_plan,
    package_frames,
    package_images,
    save_package,
    validate_package,
    with_segments,
)
from roboscript.learning.episode import (
    Episode,
    EpisodeDriver,
    EpisodeOutcome,
    InfraExhausted,
    Trajectories,
    Trajectory,
    set_aside,
)
from roboscript.learning.evaluate import Evaluate, EvaluateConfig, EvaluateResult, evaluate
from roboscript.learning.events import EventRouter
from roboscript.learning.hooks import Hooks
from roboscript.learning.ledger import TryRecord, wilson
from roboscript.learning.runner import (
    ProtocolRunner,
    brain_factory,
    brain_maker,
    control_client,
    load_protocol,
)

__all__ = [
    "BrainSpec",
    "DemoPackage",
    "DemoSegments",
    "DemoTruth",
    "Episode",
    "EpisodeDriver",
    "EpisodeOutcome",
    "Evaluate",
    "EvaluateConfig",
    "EvaluateResult",
    "EventRouter",
    "Hooks",
    "InfraExhausted",
    "Keyframe",
    "LearningConfig",
    "LearningProtocol",
    "Operation",
    "ProtocolRecipe",
    "ProtocolRunner",
    "RoleConfig",
    "TaskContext",
    "Trajectories",
    "Trajectory",
    "TryRecord",
    "brain_factory",
    "brain_maker",
    "build_package",
    "control_client",
    "demo_truth",
    "evaluate",
    "goal_frames",
    "import_blobs",
    "keyframe_plan",
    "load_demo_truth",
    "load_package",
    "load_protocol",
    "load_protocol_recipe",
    "load_segments",
    "operation_plan",
    "package_frames",
    "package_images",
    "save_package",
    "set_aside",
    "validate_package",
    "wilson",
    "with_segments",
]
