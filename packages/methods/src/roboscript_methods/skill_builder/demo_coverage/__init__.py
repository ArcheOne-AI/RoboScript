"""Demonstration coverage: a task's one demonstration expressed as skill calls. `segments` cuts it
at each arm's gripper events, `fitting` fits each segment to a skill with its free parameters, `run`
explains each segment in the demonstration's scene and runs the ceiling planner, `report` writes
`coverage.md`. The benchmark's glue gives the recording and the env. No eager imports: the
modules import the core's manipulation stack."""
