"""Sides relative to the acting arm: TASK's sides stay the robot's, a criterion's side taken from
the demonstration's arm follows the acting arm (`perception.sides`, `memory.sided`, the linter's
`side_bound` / `side_order`, `criteria.for_arm`, `beside`'s arm sides, `end_for`)."""

from roboscript.memory import MemoryDoc, MemoryRecipe, lint
from roboscript.memory.params import DoneSpecParam
from roboscript.memory.sided import demo_arms, side_rule
from roboscript.perception.criteria import DoneSpec, EndTarget, for_arm
from roboscript.perception.predicates import Scene, check
from roboscript.perception.registry import SceneRegistry
from roboscript.perception.sides import names_side, resolve, task_sides
from roboscript.testing.scenes import TABLE_Z, Box, robot_state, surface, table_model
from roboscript.tools.done import end_for


def sided(memory: MemoryDoc, statement: str = "Put the tool beside the base.") -> MemoryDoc:
    """The sample Memory ending with the tool on the demonstration arm's side of the base."""
    stage = memory.workflow[1].model_copy(
        update={
            "done": "beside(tool, base, left, {place_near}, 0.2) and near(A, base, {place_near})"
        }
    )
    offset = DoneSpecParam(
        value=DoneSpec(metric="offset_x", a="tool", b="base", value=-0.12, tol=0.01)
    )
    return memory.replace(
        statement=statement,
        workflow=[memory.workflow[0], stage],
        params={**memory.params, "tool_x": offset},
    )


def arm_relative(doc: MemoryDoc) -> MemoryDoc:
    """`sided`'s Memory with both sides written relative to the arm that took the tool."""
    offset = doc.params["tool_x"]
    assert isinstance(offset, DoneSpecParam)
    stage = doc.workflow[1]
    done = stage.done.replace("beside(tool, base, left,", "beside(tool, base, arm_side,")
    return doc.replace(
        workflow=[doc.workflow[0], stage.model_copy(update={"done": done})],
        params={
            **doc.params,
            "tool_x": offset.model_copy(
                update={"value": offset.value.model_copy(update={"arm": "left"})}
            ),
        },
    )


def test_sides_a_text_names() -> None:
    assert names_side("Put the blocks in a row from left to right.")
    assert not names_side("Use the left arm to lift the cup; the right hand waits.")
    assert task_sides("Move the soap to the left of the box.")
    assert not task_sides("") and not task_sides("Put the tool on the base.")
    assert (resolve("arm_side", "left"), resolve("other_side", "left")) == ("left", "right")
    assert resolve("arm_side", None) is None and resolve("front", None) == "front"


def test_a_criterion_follows_the_acting_arm() -> None:
    spec = DoneSpec(metric="offset_x", a="tool", b="base", value=-0.12, tol=0.01, arm="left")
    assert for_arm(spec, "left").value == -0.12 and for_arm(spec, None).value == -0.12
    assert for_arm(spec, "right").value == 0.12
    fixed = spec.model_copy(update={"arm": None})
    assert for_arm(fixed, "right").value == -0.12


def test_beside_takes_its_side_from_the_arm_that_handled_it() -> None:
    registry = SceneRegistry()
    shapes = {
        "base": Box((0.0, 0.0, TABLE_Z + 0.025), (0.08, 0.08, 0.05)),
        "tool": Box((-0.15, 0.0, TABLE_Z + 0.02), (0.04, 0.04, 0.04)),
    }
    for name, shape in shapes.items():
        registry.register(name, surface(shape), frame="frame-1", step=0, table_z=TABLE_Z)
    robot = robot_state(step=0)

    def scene() -> Scene:
        objects = {i: registry.geometry(i, robot) for i in registry.objects}
        return Scene(robot=robot, table=table_model(), objects=objects, records=registry.objects)

    base, tool = "o1", "o2"
    unknown = check(scene(), "beside", a=tool, b=base, direction="arm_side", dmin=0.1, dmax=0.2)
    assert unknown.value is None and "no arm" in unknown.reason
    for arm, side in (("left", True), ("right", False)):
        registry.attach(tool, arm, robot)
        registry.detach(tool, arm, robot)  # handled stays after the release
        at = check(scene(), "beside", a=tool, b=base, direction="arm_side", dmin=0.1, dmax=0.2)
        other = check(scene(), "beside", a=tool, b=base, direction="other_side", dmin=0.1, dmax=0.2)
        assert (at.value, other.value) == (side, not side), (arm, at.reason)


def test_the_demonstration_arm_is_its_fitted_grasp_s(memory: MemoryDoc) -> None:
    assert demo_arms(sided(memory)) == {"tool": "left"}  # grasp_handle's source, `demo left K2`


def test_the_rule_that_binds_a_side(memory: MemoryDoc) -> None:
    doc = sided(memory)
    anchor = side_rule(doc, "tool", "base")
    assert (anchor.bind, anchor.arm) == (True, "left") and "never moves" in anchor.why
    into = doc.replace(
        params={
            **doc.params,
            "in": DoneSpecParam(
                value=DoneSpec(metric="inside", a="tool", b="base", value=1.0, tol=0.5)
            ),
        }
    )
    assert not side_rule(into, "tool", "base").bind  # the offset is the base's geometry
    stages = [s.model_copy() for s in doc.workflow]
    grasp_base = stages[0].model_copy(
        update={"name": "grasp_base", "call": "grasp_object(arm=B, obj=base, grasp={grasp_base})"}
    )
    right = doc.params["grasp_handle"].model_copy(update={"source": "demo right K3"})
    held = doc.replace(
        workflow=[stages[0], grasp_base, stages[1]], params={**doc.params, "grasp_base": right}
    )
    rule = side_rule(held, "tool", "base")
    assert rule.bind and "held at once" in rule.why  # still held: grasping does not release
    place_base = stages[1].model_copy(
        update={
            "name": "place_base",
            "call": "place_object(arm=B, obj=base, target={place_on_base})",
        }
    )
    row = doc.replace(
        workflow=[grasp_base, place_base, *stages], params={**doc.params, "grasp_base": right}
    )
    assert not side_rule(row, "tool", "base").bind  # both placed in turn: the task's order


def test_the_linter_flags_robot_sides_task_does_not_name(
    memory: MemoryDoc, recipe: MemoryRecipe
) -> None:
    assert lint(memory, recipe.lint) == []
    rules = [(f.rule, f.where) for f in lint(sided(memory), recipe.lint)]
    assert ("side_bound", "params.tool_x") in rules
    assert ("side_bound", "workflow.set_down.done") in rules
    bound = arm_relative(sided(memory))
    assert not [f for f in lint(bound, recipe.lint) if f.rule == "side_bound"]
    named = sided(memory, "Put the tool to the left of the base.")
    assert not [f for f in lint(named, recipe.lint) if f.rule == "side_bound"]
    empty = [f for f in lint(memory.replace(statement=""), recipe.lint) if f.rule == "no_statement"]
    assert [f.level for f in empty] == ["warning"]


def test_fixed_sides_keep_the_demonstration_s_end(memory: MemoryDoc) -> None:
    target = EndTarget(kind="place", a="tool", b="base", offset=(-0.12, 0.0), tol=0.02)
    target = target.model_copy(update={"mirror": True, "arm": "left"})
    (kept,) = end_for(memory, [target])
    assert (kept.mirror, kept.arm) == (True, "left")
    (fixed,) = end_for(memory.replace(statement="Put the tool left of the base."), [target])
    assert (fixed.mirror, fixed.arm) == (False, None)
