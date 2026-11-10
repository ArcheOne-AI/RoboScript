"""Perception (docs/concepts/perception.md): geometry first, semantics second.

Depth geometry gives positions, heights, sizes and orientations; grounding backends only say
which object or part is which. Every module is a library usable without a model; import from the
submodules:

- `frames`: `Frame` (an observation whose images load on use) and `Frames` (references into the
  episode's frame store)
- `views`: a frame's `FrameGeometry` (world point per pixel, robot pixels, table), `analyze`
- `robot`: the robot's own geometry at its measured joints: `SelfFilter` (URDF + planner FK)
- `table`, `lift`, `proposals`, `fit`, `overlay`: the default pipeline's steps and Set-of-Mark
- `registry`: the per-episode scene registry (identify once, then grasp relations only; its
  decisions logged as steps) and the `ObjectGeometry` manipulation consumes
- `relations`, `predicates`: read-only checks over a `Scene`
- `grounding`: the backend interface (`Grounder`, `Segmenter`) and the `geometric` grounder
- `locate`: composites (`locate`, `refresh`, `scene`) that log every inner step
- `config`: the pipeline's knobs (`configs/perception.example.yaml`)
"""
