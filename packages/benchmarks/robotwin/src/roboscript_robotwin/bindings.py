"""RoboTwin's bindings for the core's entries, as data (docs/how-to/add-a-skillset.md): what its
skillset `robotwin` binds (`roboscript_robotwin.skills`), importable without the tools (the
simulator side and the motion service's configs read them too). The pose conventions are its
embodiment profile's (`embodiments/aloha_agilex.yaml`, `poses`)."""

from typing import Final

from roboscript.contracts.executor import ExecutorConfig
from roboscript.contracts.skills import Bindings, PlannerProfile, PlannerTable, SupportSurface

PLANNER_PROFILE: Final = PlannerProfile(
    table=PlannerTable(size=(0.7, 2.0, 0.04), top_z=0.74),
    yaw_rad={"left": -0.02, "right": -0.01},
)
"""What RoboTwin's planner code hard-codes (`envs/robot/planner.py`): its collision world, one
"table" cuboid 0.7 x 2 x 0.04 m whose top is at 0.74 m, and the aloha-agilex arms' turns after
`frame_bias`, the yaws its URDF mounts them with (configs/motion.example.yaml holds the same)."""

# Each field is documented on `roboscript.contracts.executor.ExecutorConfig`; the fields left out
# keep its defaults. Grouped by what they serve, with why RoboTwin takes these values.
EXECUTOR: Final = ExecutorConfig(
    # Paths: waypoints within 3 mm of the plan; straight moves solved every 2 cm / 5 deg; a 1 rad
    # jump between IK solutions or 5.5 rad of travel is a wrist spin or a wrapped joint, not motion
    # the TCP shows; an orientation lock may stray 2 cm / 5 deg from the line.
    waypoint_tolerance_m=0.003,
    straight_step_m=0.02,
    straight_step_deg=5.0,
    max_joint_step_rad=1.0,
    lock_tolerance_m=0.02,
    lock_tolerance_deg=5.0,
    max_joint_travel_rad=5.5,
    # cuRobo's refusals vary from seed to seed: a few fresh attempts before a move is refused.
    replans=3,
    # Gripper holds: a close settles within a few holds, the fingers open fully in four; a placed
    # object needs no wait before it is let go, and a place is sent as one move.
    settle_holds=3,
    open_holds=4,
    place_step_m=None,
    release_holds=0,
    # Outcomes: a deviated straight move is tried once more from where it stopped; a stop within
    # 1.5 cm / 8 deg of the target reached it; a touch 5 cm off its path never got there, one 1 cm
    # short was stopped by what it pushed.
    retry_deviated=True,
    stop_tolerance_m=0.015,
    stop_tolerance_deg=8.0,
    contact_off_path_m=0.05,
    press_m=0.01,
    # Bounds: a displacement at most 20 cm / 90 deg; two TCPs kept 3 cm apart; no probed
    # workspace (the planner refuses what the arms cannot reach).
    max_translation_m=0.2,
    max_rotation_deg=90.0,
    min_arm_gap_m=0.03,
    workspace=None,
    workspace_margin_m=0.08,
    # Clearance: perceived boxes grown 2 cm; after a release the TCP backs out 5-15 cm along its
    # approach; an empty gripper low near an obstacle first rises 3 cm, at most 15 cm.
    obstacle_margin_m=0.02,
    release_withdraw_m=0.05,
    release_withdraw_max_m=0.15,
    lift_clearance_m=0.03,
    max_lift_m=0.15,
    # Repeats are left to the Agent (not refused); the tolerances only name what a repeat is.
    reject_repeats=False,
    repeat_tolerance_m=0.005,
    repeat_tolerance_deg=5.0,
    # Articulated parts: follow the joint 2.5 cm a step until less than 3 mm is left, a step under
    # a quarter of its length is a stall; regrasp once, a reported slip included; a path that ends
    # on a joint follows the joint, up to a quarter of its travel past the demonstration's end.
    follow_step_m=0.025,
    follow_tolerance_m=0.003,
    follow_stall=0.25,
    follow_regrasps=1,
    follow_regrasp_reported=True,
    follow_joint_paths=True,
    follow_past=0.25,
    # Symmetric objects: frames follow the hold, a named front fixes the turn, a tilted
    # placement tries the object's equivalent turns.
    hold_frames=True,
    honor_front=True,
    tilted_turns=True,
    # Receipts: a coarse-then-fine move reports the coarse outcome when the fine part had
    # nothing left to do.
    coarse_stops=True,
    # Planning in clutter: a start in collision drops only the objects the arm starts in, and
    # objects near the start or goal are kept at their true size before they are let go.
    drop_start_colliders=True,
    near_bare=True,
)
"""The executor's settings for RoboTwin's robot and physics
(`contracts.executor.ExecutorConfig`)."""

SUPPORT_SURFACE: Final = SupportSurface(
    top_z=PLANNER_PROFILE.table.top_z, max_drop_m=0.03, extent=(-0.6, 0.6, -0.35, 0.35)
)
"""RoboTwin's table as its support surface (`envs/_base_task.py` `create_table_and_wall`): 1.2 x
0.7 m centered on the world origin, its top at 0.74 m, which a randomized scene lowers by up to
`random_table_height` (0.03 m in our randomized task configs; the runs' `table_top_z` fallback is
0.71 there). The toolsets read their `table_top_z` config instead of this declaration."""

BINDINGS: Final = Bindings(
    planner_profile=PLANNER_PROFILE, executor=EXECUTOR, support_surface=SUPPORT_SURFACE
)
