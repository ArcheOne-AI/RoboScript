# Perception

`roboscript.perception` measures the scene from RGB-D. Geometry comes first: positions, sizes and orientations from depth. VLMs and detectors only say which object or part is which. Accuracy depends on the depth the env supplies.

Functions take an explicit, immutable frame (`live`, `frame_<n>` for demonstration frames, or `log:<try>:<turn>` for earlier tries), so they cannot race an action. Depth is stored as Z in millimetres, with 0 where there is no return.

## Primitives

Primitives are tools. Results carry `conf`, a `reason` (`ok`, `not_visible`, `ambiguous`, `occluded`), the backend and latency.

| Tool | What it does |
|---|---|
| `ground` | Boxes or points from text, with scores and a numbered overlay |
| `segment` | A mask from text, a box, a point or a region |
| `lift` | Regions to world points, with robot self-filtering |
| `fit` | Fit a shape (`box`, `cylinder`, `plane`, `line`): center, frame, axis, symmetry |
| `table` | Table plane, extent, edges and landmarks |
| `project`, `measure_pixel` | Between world points and pixels |

Composites chain primitives: `locate` runs ground, segment, lift, multi-view merge, fit and register. `locate_part` registers a part candidate. `relation` gives displacement, distance and spatial predicates (`on`, `inside`, `above`, etc.).

## The default pipeline

The table plane comes from a height histogram. Proposals come from height-map connected components. Views fuse anchored on the binding's `anchor` camera and checked in wrist cameras. Centers are shape-aware (box top face, container rim, cylinder footprint). VLM prompts name cameras in the binding's words (`anchor_words`, `camera_words`).

## Backends

Backends register in `roboscript.perception`: VLM grounding through the gateway, geometric proposals with Set-of-Mark, GroundingDINO and SAM 3. Torch backends run in the vision service (`roboscript_vision`) on one GPU. Model scores rank candidates and are never thresholded.

## Identification

The system-only `identify` tool runs when an episode opens. It asks each backend to find each of Memory's OBJECTS roles, producing a small vote per role (`ok`, `unconfirmed`, `ambiguous`, `not_named`). Where words cannot distinguish look-alikes, roles with an `order:` line sort by distance to the base or left-to-right. Identity is decided here and not revisited; later measurements only update poses.

## The scene registry

The registry lives in the episode's frame store: objects and parts with frames, sizes, boxes, symmetry, confidence and grasp relations. Every change goes through `frames.edit_registry()` under a lock. In an Actor's episode the registry changes only at identification, on grasp-relation updates, part registration, remeasurement, or refinding a lost object. A held object's pose comes from kinematics, keeping it usable while the hand hides it.

## Done criteria

`wicl`'s done criteria (`DoneSpec`) describe the scene at task completion: a metric on a registered role with a value and tolerance. Metrics are generic (`distance_xy`, `gap_z`, `overlap`, `inside`, `yaw_diff`, `held`, `surface_change`, `press`). `check_done` reports them as relative targets, and the Actor corrects small misses with `nudge`.
