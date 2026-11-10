"""The action API (docs/concepts/embodied-core.md): grasp, place, displacement and pose builders
over perceived objects and agent-written specs, executed through qpos waypoints with receipts.

- `specs`: GraspSpec, PlaceSpec, PlaceTarget, HingeSpec, PathSpec, what the agent writes
  instead of asset annotations.
- `proxy`: `ObjectProxy`, the object surface the pose functions read, over measured geometry
  (`from_geometry`).
- `synthesis`: specs to contact and placement frames: grasp candidates (and their mirror for the
  other arm), placement points, placement targets (mirrored with `for_arm`) and regions, hinge
  lines.
- `posemath`: the `SceneObject` surface (a pose and frames by kind), `approach_pose`,
  `first_plannable_tilt`, `choose_grasp`, `place_ee_pose` and `arc_poses`.
- `fit`: the spec that reproduces an observed grasp or placement (`demo_relation`).
- `api`: `ActionApi`, the builders (`grasp_object`, `place_object`, ...),
  `rotate_about` (a hinge), `run` (one action, or two at once) and `check_reach` (a `Reach`: which
  arms could make the actions, without moving).
- `paths`: held paths (`PathSpec`): the demonstration's TCP poses while holding, in the held
  object's body frame (`fit_path`), and the rigid motions that `ActionApi.follow_path` repeats.
- `articulation`: a joint (`Joint`) estimated from the robot's own motion: circle and line fits
  to achieved TCP positions (`estimate`), which `ActionApi` follows on a held part's pull.
- `Executor` turns per-arm segments (`Move` to an ee pose, free or straight, contact or not,
  bounded or not; `Grip`) into one batch of `take_action(qpos)` steps on the env's agent plane,
  planned with a `Planner` from the arm's drive target and sent as the fewest waypoints keeping
  the TCP within 3 mm of the planned path, after its pre-checks (the table, the probed
  `workspace`, the arms' gap moving together, measured by `tcp_gap`); `receipts.receipt` builds
  its `ActionReceipt`.
"""

from roboscript.contracts.executor import ExecutorConfig
from roboscript.manipulation.api import (
    ARC_STEP_DEG,
    ActionApi,
    Built,
    Follow,
    GraspCall,
    Placement,
    PlanningLog,
    PlanningTimeout,
    placement,
    robot_base,
)
from roboscript.manipulation.articulation import Joint, estimate
from roboscript.manipulation.executor import Executor
from roboscript.manipulation.fit import (
    GraspFit,
    HingeFit,
    PlaceFit,
    contact_of,
    fit_grasp,
    fit_grasp_pose,
    fit_hinge,
    fit_place,
)
from roboscript.manipulation.paths import PathFit, body_frames, fit_path, hold_frames
from roboscript.manipulation.plan import (
    Grip,
    GripCheck,
    Move,
    MoveCheck,
    Plan,
    Reach,
    Segment,
    reach,
    tcp_gap,
)
from roboscript.manipulation.posemath import CONTACT, PLACEMENT, SceneObject
from roboscript.manipulation.proxy import ObjectProxy, Part
from roboscript.manipulation.receipts import chained, receipt
from roboscript.manipulation.specs import (
    FAR_END,
    GEOMETRIC_ENDS,
    HIGH_END,
    LOW_END,
    OTHER_ARM,
    GraspSpec,
    HingeSpec,
    PathSpec,
    PlaceSpec,
    PlaceTarget,
    SpecError,
)
from roboscript.manipulation.synthesis import (
    Candidate,
    equivalent_turns,
    for_arm,
    grasp_candidates,
    hinge_axes,
    hinge_line,
    mirror,
    placement_point_pose,
    region_points,
    target_frame,
    target_pose,
    turned,
)
from roboscript.manipulation.workspace import Workspace, load_workspace, probe

__all__ = [
    "ARC_STEP_DEG",
    "CONTACT",
    "FAR_END",
    "GEOMETRIC_ENDS",
    "HIGH_END",
    "LOW_END",
    "OTHER_ARM",
    "PLACEMENT",
    "ActionApi",
    "Built",
    "Candidate",
    "Executor",
    "ExecutorConfig",
    "Follow",
    "GraspCall",
    "GraspFit",
    "GraspSpec",
    "Grip",
    "GripCheck",
    "HingeFit",
    "HingeSpec",
    "Joint",
    "Move",
    "MoveCheck",
    "ObjectProxy",
    "Part",
    "PathFit",
    "PathSpec",
    "PlaceFit",
    "PlaceSpec",
    "PlaceTarget",
    "Placement",
    "Plan",
    "PlanningLog",
    "PlanningTimeout",
    "Reach",
    "SceneObject",
    "Segment",
    "SpecError",
    "Workspace",
    "body_frames",
    "chained",
    "contact_of",
    "equivalent_turns",
    "estimate",
    "fit_grasp",
    "fit_grasp_pose",
    "fit_hinge",
    "fit_path",
    "fit_place",
    "for_arm",
    "grasp_candidates",
    "hinge_axes",
    "hinge_line",
    "hold_frames",
    "load_workspace",
    "mirror",
    "placement",
    "placement_point_pose",
    "probe",
    "reach",
    "receipt",
    "region_points",
    "robot_base",
    "target_frame",
    "target_pose",
    "tcp_gap",
    "turned",
]
