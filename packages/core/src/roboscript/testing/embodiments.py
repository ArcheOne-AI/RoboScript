"""Conformance checks for embodiment profiles (`roboscript.embodiments`,
docs/concepts/embodied-core.md).

A profile conforms when it loads by the name it is registered under (or from its YAML path), is
the same profile each time and survives a JSON round trip, lays its arms' joints and grippers and
its joint groups out over the joint vector exactly once, declares the kinds of action it takes
consistently (at least one; ee-delta control on arms it has), normalizes its grippers (0 closed,
1 open; a closed gripper on nothing reads below the holding threshold), converts between ee and
TCP poses both ways for every arm, starts every arm inside its stops, hangs its wrist cameras on
arms it has, frames its grasps and assets with an orthonormal `poses`, and keeps its planner
assets relative to `assets_root`.

Use them from pytest by subclassing `EmbodimentConformance` as a `Test*` class with an
`embodiment_target` fixture, or without pytest via `roboscript.testing.precheck`:

```python
@pytest.fixture(params=registered("roboscript.embodiments", "roboscript_myrobot"))
def embodiment_target(request):
    return EmbodimentTarget(request.param)

class TestMyRobot(EmbodimentConformance): ...
```
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roboscript.contracts.geometry import Pose
from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.testing import unit


@dataclass(frozen=True)
class EmbodimentTarget:
    """`ref`: the profile's name in `roboscript.embodiments`, or the path of its YAML file. A
    registered name is also the profile's own `name`."""

    ref: str | Path


class EmbodimentConformance:
    """The checks, as `test_*` methods taking an `EmbodimentTarget`."""

    @unit
    async def test_loads_and_round_trips(self, embodiment_target: EmbodimentTarget) -> None:
        ref = embodiment_target.ref
        profile = load_embodiment(ref)
        assert isinstance(profile, EmbodimentProfile)
        if isinstance(ref, str) and not ref.endswith((".yaml", ".yml")):
            assert profile.name == ref, f"registered as {ref!r}, named {profile.name!r}"
        assert load_embodiment(ref) == profile
        assert EmbodimentProfile.model_validate_json(profile.model_dump_json()) == profile

    @unit
    async def test_the_action_vector_is_laid_out_once(
        self, embodiment_target: EmbodimentTarget
    ) -> None:
        profile = load_embodiment(embodiment_target.ref)
        used: list[int] = []
        for arm, spec in profile.arms.items():
            start, stop = spec.action
            assert stop - start == len(spec.joints), arm
            assert not start <= spec.gripper_index < stop, f"{arm}: the gripper is among the joints"
            used += [*range(start, stop), spec.gripper_index]
        for name, group in profile.joint_groups.items():
            start, stop = group.action
            assert stop - start == len(group.joints), name
            used += range(start, stop)
        assert sorted(used) == list(range(profile.action_dim))

    @unit
    async def test_action_kinds_are_declared(self, embodiment_target: EmbodimentTarget) -> None:
        profile = load_embodiment(embodiment_target.ref)
        actions = profile.actions
        assert actions.kinds(), "a profile takes at least one kind of action"
        if actions.ee_delta is not None:
            assert set(actions.ee_delta.arms) <= set(profile.arms), actions.ee_delta.arms
        again = EmbodimentProfile.model_validate_json(profile.model_dump_json())
        assert again.actions.kinds() == actions.kinds()

    @unit
    async def test_grippers_are_normalized(self, embodiment_target: EmbodimentTarget) -> None:
        profile = load_embodiment(embodiment_target.ref)
        for arm, spec in profile.arms.items():
            gripper = spec.gripper
            low, high = gripper.scale
            assert low != high, arm
            assert abs(gripper.opening(low)) < 1e-9 and abs(gripper.opening(high) - 1) < 1e-9, arm
            assert 0.0 <= gripper.empty < gripper.holding_threshold < gripper.open, arm
            assert gripper.closed < gripper.open, arm

    @unit
    async def test_ee_and_tcp_convert_both_ways(self, embodiment_target: EmbodimentTarget) -> None:
        profile = load_embodiment(embodiment_target.ref)
        probe = Pose(p=(0.2, -0.1, 0.9), q=(0.5, 0.5, 0.5, 0.5))
        for arm in profile.arms:
            back = profile.ee_pose(arm, profile.tcp_pose(arm, probe))
            assert np.allclose(back.as_list(), probe.as_list(), atol=1e-9), arm

    @unit
    async def test_arms_start_inside_their_stops(self, embodiment_target: EmbodimentTarget) -> None:
        profile = load_embodiment(embodiment_target.ref)
        for arm, spec in profile.arms.items():
            if spec.home is not None:
                assert spec.past_stops([spec.home]) is None, f"{arm}: home is past a stop"

    @unit
    async def test_cameras_hang_on_arms_it_has(self, embodiment_target: EmbodimentTarget) -> None:
        profile = load_embodiment(embodiment_target.ref)
        assert profile.cameras, "a profile names its cameras"
        for name, camera in profile.cameras.items():
            if camera.kind == "wrist":
                assert camera.arm in profile.arms, f"{name}: wrist camera on {camera.arm!r}"

    @unit
    async def test_pose_conventions_are_frames(self, embodiment_target: EmbodimentTarget) -> None:
        poses = load_embodiment(embodiment_target.ref).poses
        grasp = np.asarray(poses.contact_to_grasp, dtype=float)
        assert grasp.shape == (4, 4) and np.allclose(grasp[3], (0, 0, 0, 1))
        rotations = {
            "contact_to_grasp": grasp[:3, :3],
            "placement_rotation": np.asarray(poses.placement_rotation, dtype=float),
        }
        for name, rotation in rotations.items():
            assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6), name
            assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6), name
        for name in ("tcp_closing_axis", "object_up"):
            assert np.isclose(np.linalg.norm(getattr(poses, name)), 1.0, atol=1e-6), name

    @unit
    async def test_planner_assets_are_relative(self, embodiment_target: EmbodimentTarget) -> None:
        profile = load_embodiment(embodiment_target.ref)
        for arm, spec in profile.arms.items():
            if spec.planner is None:
                continue
            assert profile.assets_root is not None, f"{arm} plans, so the profile has assets"
            for path in (spec.planner.curobo, spec.planner.urdf, spec.planner.srdf):
                assert path is None or not Path(path).is_absolute(), (arm, path)
