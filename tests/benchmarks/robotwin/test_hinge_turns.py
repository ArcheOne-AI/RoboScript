"""RoboTwin's `rotate_about` skill on the core manipulation toolset, over the Hub, MockEnv and
MockPlanner (the scene of `tests/core/toolsets/test_manipulation_objects.py`): a part is grasped
and turned about a hinge on its box, in chunks, seen after each."""

from typing import Final

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.envs.mock import TABLE_Z, MockEnv
from roboscript.manipulation import HingeSpec, hinge_line
from roboscript.records import EventWriter
from tests.core.toolsets.conftest import events, model
from tests.core.toolsets.test_manipulation_objects import (
    TOP,
    Model,
    _angle,
    _proxy,
    _radius,
    env,
    scene,
)

__all__ = ["env", "events", "model"]  # the fixtures these tests take

WITH_HINGE_TURN: Final = ("robotwin",)
"""RoboTwin's skillset, which has `rotate_about`."""


async def test_a_part_is_grasped_and_turned_about_a_hinge_on_its_box(
    env: tuple[MockEnv, LocalBlobStore], events: EventWriter, model: Model
) -> None:
    hub, opened = await scene(env, events, model, skillsets=WITH_HINGE_TURN)
    async with hub:
        s = await opened()
        arm, red = await s.side("red"), s.obj["red"]
        async with hub.store.get(s.ep).frames.edit_registry() as registry:
            points = registry.geometry(s.ids["red"]).points
            top = points[points[:, 2] > TABLE_Z + 0.03]  # the box's upper 2 cm
            registry.register_part(s.ids["red"], "lid", top)
        measured = await _proxy(s, "red")
        spec = HingeSpec(part="lid", side="left", at="upright")  # a door's
        point, direction = hinge_line(spec, measured)  # on the box as measured
        lid = measured.pose @ measured.part("lid").frame
        grasped = await s.act("grasp_object", arm=arm, obj=red, grasp={**TOP, "part": "lid"})
        assert grasped.reason == "ok" and await s.tcp(arm) == pytest.approx(lid[:3, 3])
        assert grasped.grasp is not None and grasped.grasp.n_candidates == 2  # x either way
        # The registry carries the box along; the hinge stays where the part was measured.
        await s.act("move_by", arm=arm, x=0.03)
        start = await s.tcp(arm)
        turned = await s.act(
            "rotate_about", arm=arm, obj=red, hinge=spec.model_dump(), angle_deg=20, steps=2
        )
        # The closed gripper holds the lid: the API follows the turn on its joint.
        assert turned.reason == "ok" and turned.joint is not None
        assert turned.joint.kind == "revolute" and turned.joint.travel == pytest.approx(20, abs=1)
        end = await s.tcp(arm)
        assert _radius(end, point, direction) == pytest.approx(
            _radius(start, point, direction), abs=1e-4
        )
        # Right-handed about up: counterclockwise seen from above.
        assert _angle(start, end, point, direction) == pytest.approx(20.0, abs=0.1)
