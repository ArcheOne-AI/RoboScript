"""Done criteria: their form, the metrics measure takes from geometry, the finish gate without its
fact, targets only for what the demonstration moved, and S2 only tightening a criterion."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.geometry import OrientedBox
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.ids import BlobId
from roboscript.contracts.phase import PhaseRules
from roboscript.memory import (
    Editor,
    MemoryDoc,
    MemoryRecipe,
    open_cache,
)
from roboscript.perception.criteria import DoneSpec, EndTarget
from roboscript.perception.frames import CameraImage
from roboscript.perception.registry import ObjectGeometry
from roboscript.perception.views import View
from roboscript_methods.wicl.finish import (
    CHECK_DONE,
    done_gate,
    require_done_fact,
)
from roboscript_methods.wicl.memory import validators
from roboscript_methods.wicl.perception.done import (
    measure,
)
from roboscript_methods.wicl.perception.endstate import end_targets, place_offset


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"metric": "held", "a": "cup", "b": "plate"}, "held takes no b"),
        ({"metric": "gap_z", "a": "cup"}, "gap_z needs b"),
        ({"metric": "overlap", "a": "cup", "b": "table"}, "only gap_z is measured against"),
        ({"metric": "offset_x", "a": "cup", "b": "cup"}, "different roles"),
        ({"metric": "gap_z", "a": "cup", "b": "table", "part": "rim"}, "only surface_change"),
        ({"metric": "held", "a": "cup", "part": "rim"}, "only surface_change and press"),
        ({"metric": "press", "a": "cup", "b": "plate"}, "press takes no b"),
        ({"metric": "gap_z", "a": "table", "b": "cup"}, "the table is only gap_z's b"),
        ({"metric": "overlap", "a": "cup", "b": "plate", "value": 1.5}, "outside [0, 1]"),
        ({"metric": "inside", "a": "cup", "b": "plate", "tol": 2.0}, "wider than its span"),
        ({"metric": "held", "a": "Cup"}, "should match pattern"),
    ],
)
def test_a_criterion_s_form_is_checked(spec: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message.replace("[", r"\[")):
        DoneSpec.model_validate({"value": 0.5, "tol": 0.1, **spec})


def geometry(center: tuple[float, float, float], yaw: float, **rest: Any) -> Any:
    half = np.radians(yaw) / 2
    pose = np.array([*center, np.cos(half), 0.0, 0.0, np.sin(half)])
    fields = {"held_by": (), "symmetry": "none", "pose": pose, "yaw_deg": yaw, **rest}
    return SimpleNamespace(**fields)


def test_measure_unary_and_yaw(monkeypatch: pytest.MonkeyPatch) -> None:
    table = SimpleNamespace(height=lambda p: float(p[2] - 0.74))
    cup = geometry((0.0, 0.0, 0.8), 30.0, held_by=("left",), bottom_center=np.array([0, 0, 0.76]))
    plate = geometry((0.1, 0.0, 0.75), -140.0)
    assert measure("held", cup, None, table) == (1.0, "")  # type: ignore[arg-type]
    value, _ = measure("gap_z", cup, None, table)  # type: ignore[arg-type]
    assert value == pytest.approx(0.02)
    assert measure("yaw_diff", cup, plate, table)[0] == pytest.approx(10.0)  # type: ignore[arg-type]
    half = geometry((0.1, 0.0, 0.75), -140.0, symmetry="half_pi")
    assert measure("yaw_diff", cup, half, table)[0] == pytest.approx(10.0)  # type: ignore[arg-type]
    round_ = geometry((0.1, 0.0, 0.75), 0.0, symmetry="continuous")
    assert measure("yaw_diff", cup, round_, table) == (None, "a round object has no yaw")  # type: ignore[arg-type]


def view(depth: float, cam2world: np.ndarray, robot: bool = False) -> View:
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    image = np.full((64, 64), depth, dtype=np.float32)
    xyz = np.zeros((64, 64, 3))
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    return View(CameraImage("head", K, cam2world, rgb, image), image, xyz, np.full((64, 64), robot))


SPEC = DoneSpec(metric="gap_z", a="tool", b="table", value=0.0, tol=0.01)


def test_finish_gated_without_the_fact_is_refused(memory: MemoryDoc) -> None:
    """The finish gate and the `DONE` fact go together: a gate without the fact is an error."""
    gated = PhaseRules(done_check=CHECK_DONE)
    with pytest.raises(ValueError, match="no done fact"):
        require_done_fact(gated, None)
    with pytest.raises(ValueError, match="no done fact"):
        require_done_fact(gated, EpisodeSeed(facts={"other": {}}))
    require_done_fact(None, None)
    require_done_fact(PhaseRules(fail_fast=True), None)
    view = BlobRef(id=BlobId("b" * 64), media_type="image/png", size=1)
    rules, seed = done_gate(memory, None, None, goal=({"overhead": view}, "Lift it."))
    require_done_fact(rules, seed)


TABLE_Z = 0.70


TABLE = SimpleNamespace(height=lambda p: np.asarray(p)[:, 2] - TABLE_Z)


def box(
    xy: tuple[float, float],
    size: tuple[float, float, float] = (0.06, 0.04, 0.04),
    yaw: float = 0.0,
    bottom: float = TABLE_Z,
    held_by: tuple[str, ...] = (),
    symmetry: str = "none",
) -> ObjectGeometry:
    center = np.array([xy[0], xy[1], bottom + size[2] / 2])
    half = np.radians(yaw) / 2
    q = (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    grid = np.stack(np.meshgrid(np.linspace(-0.5, 0.5, 7), np.linspace(-0.5, 0.5, 7)), -1)
    local = grid.reshape(-1, 2) * np.asarray(size[:2])
    xy_points = local @ np.array([[c, s], [-s, c]]) + center[:2]
    points = np.concatenate([xy_points, np.full((len(xy_points), 1), center[2])], axis=1)
    return ObjectGeometry(
        id="o",
        name="o",
        shape="box",
        upright=True,
        symmetry=symmetry,  # type: ignore[arg-type]
        pose=np.array([*center, *q]),
        obb=OrientedBox(center=tuple(center), size=size, q=q),
        center=center,
        top_center=np.array([xy[0], xy[1], bottom + size[2]]),
        bottom_center=np.array([xy[0], xy[1], bottom]),
        points=points,
        parts={},
        held_by=held_by,  # type: ignore[arg-type]
        stale_age=0,
        partially_visible=False,
    )


def kinds(targets: list[EndTarget]) -> set[tuple[str, str, str | None]]:
    return {(t.kind, t.a, t.b) for t in targets}


def test_only_what_the_demonstration_moved_gets_targets() -> None:
    start = {"block": box((0.0, -0.1)), "pad": box((0.2, -0.1), (0.15, 0.15, 0.005))}
    end = {"block": box((0.2, -0.1), bottom=TABLE_Z + 0.005), "pad": start["pad"]}
    targets = end_targets(end, start, TABLE)
    assert kinds(targets) == {
        ("holds", "block", None),
        ("yaw", "block", None),  # set down: its end pose fixes its yaw, turned or not
        ("place", "block", "pad"),
        ("overlap", "block", "pad"),
    }
    place = next(t for t in targets if t.kind == "place")
    assert place.offset == pytest.approx((0.0, 0.0))
    # room: 0.075 - 0.03 = 0.045, half of it, within [0.02, 0.03]
    assert place.tol == pytest.approx(0.0225)


BASE = np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, -0.6], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])


"""A base frame: x ahead (world +y), y to the robot's left (world -x)."""


WHY = {"why": "the try's end says so"}


DOWN = {"metric": "gap_z", "a": "tool", "b": "base", "value": 0.017, "tol": 0.0107}


def with_criterion(memory: MemoryDoc, recipe: MemoryRecipe) -> MemoryDoc:
    editor = Editor(
        memory,
        rules=recipe.rules("s1"),
        lint=recipe.lint,
        seed=0,
        source="S1",
        validators=validators(),
    )
    param = {"kind": "done_spec", "value": DOWN}
    record = editor.edit({"op": "define_param", "name": "done_strike", "param": param, **WHY})
    assert record.accepted, record.reason
    return editor.doc


def test_s2_only_tightens_a_done_criterion(memory: MemoryDoc, recipe: MemoryRecipe) -> None:
    """A revision raises the set_down gap's value 0.017 -> 0.027 and drops the step that closed it:
    under `strict_done` that is refused, as
    are a wider band, another metric and the criterion's removal. Narrowing passes."""
    doc = with_criterion(memory, recipe)
    cache = open_cache(
        doc,
        budget=6,
        seed=10004,
        rules=recipe.rules("s2"),
        lint=recipe.lint,
        validators=validators(),
    )

    def edit(**op: Any) -> Any:
        return cache.edit({"param": "done_strike", **op, **WHY})

    raised = edit(op="set_number", field="value", value=0.027)
    assert not raised.accepted and "only tightened, never loosened" in raised.reason
    assert "0.0063 to 0.0277" in raised.reason
    assert not edit(op="set_number", field="tol", value=0.015).accepted
    assert not edit(op="set_category", field="metric", value="distance_xy").accepted
    gone = cache.edit({"op": "remove_param", "name": "done_strike", **WHY})
    assert not gone.accepted and "removing it" in gone.reason
    noise = edit(op="set_number", field="tol", value=0.006)  # below perception's error
    assert not noise.accepted and "below perception's error" in noise.reason
    assert edit(op="set_number", field="tol", value=0.01).accepted
    assert edit(op="set_number", field="value", value=0.0175).accepted  # 0.0075-0.0275: inside
    assert not edit(op="set_number", field="value", value=0.024).accepted  # 0.014-0.034: not
    spec = cache.doc.params["done_strike"].value
    assert (spec.value, spec.tol) == (pytest.approx(0.0175), pytest.approx(0.01))
    added = {"kind": "done_spec", "value": {**DOWN, "metric": "distance_xy", "value": 0.0}}
    more = cache.edit({"op": "define_param", "name": "done_over", "param": added, **WHY})
    assert more.accepted, more.reason  # a new criterion only adds a condition


def test_a_place_target_mirrors_for_the_other_arm() -> None:
    target = EndTarget(kind="place", a="tool", b="base", offset=(-0.12, 0.03), tol=0.02)
    target = target.model_copy(update={"mirror": True, "arm": "left"})
    base = np.eye(4)  # the base frame's y is the robot's lateral axis
    base[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    by = {arm: SimpleNamespace(handled_by=arm) for arm in ("left", "right")}
    np.testing.assert_allclose(place_offset(target, by["left"], base), [-0.12, 0.03])
    np.testing.assert_allclose(place_offset(target, by["right"], base), [0.12, 0.03])
    np.testing.assert_allclose(place_offset(target, by["right"], None), [-0.12, 0.03])
    fixed = target.model_copy(update={"mirror": False})
    np.testing.assert_allclose(place_offset(fixed, by["right"], base), [-0.12, 0.03])
