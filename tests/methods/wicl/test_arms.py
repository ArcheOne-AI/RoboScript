"""`unbind_arms`: a Memory that binds an arm comes out lint-clean, rewritten by role; the holds
`plan_arms` plans over (`memory.chains`)."""

from roboscript.memory import CategoryParam, MemoryDoc, MemoryRecipe, WorkflowStage, lint
from roboscript.memory.chains import arm_chains
from roboscript_methods.wicl.memory.arms import rule, unbind_arms


def _bound(memory: MemoryDoc) -> MemoryDoc:
    """The sample Memory bound to arms: the demonstration's right arm hands to the left."""
    first, second = memory.workflow
    return memory.replace(
        notes="Close the right gripper firmly; use {active_arm} for the set_down.",
        workflow=[
            first.model_copy(
                update={
                    "arm": "right, because the instruction names the right arm",
                    "call": "grasp_object(arm=right, obj=tool, grasp={grasp_handle})",
                    "done": "holding(right) and lifted(tool, {lift_min})",
                }
            ),
            second.model_copy(
                update={
                    "arm": "left arm, after the right-to-left hand-over",
                    "call": "place_object(arm={active_arm}, obj=tool, target={place_on_base})",
                    "done": "near(left, base, {place_near})",
                }
            ),
        ],
        params={**memory.params, "active_arm": CategoryParam(value="left")},
    )


def test_a_bound_memory_comes_out_lint_clean_by_role(
    memory: MemoryDoc, recipe: MemoryRecipe
) -> None:
    old = _bound(memory)
    assert any(f.rule == "fixed_arm" for f in lint(old, recipe.lint))
    rewrite = unbind_arms(old)
    doc = rewrite.doc
    assert [f for f in lint(doc, recipe.lint) if f.level == "error"] == []
    first, second = doc.workflow
    assert (first.arm, second.arm) == (rule(first.name), rule(second.name))
    assert first.call == "grasp_object(arm=<arm>, obj=tool, grasp={grasp_handle})"
    assert first.done == "holding(<arm>) and lifted(tool, {lift_min})"
    assert second.call == "place_object(arm=<other arm>, obj=tool, target={place_on_base})"
    assert second.done == "near(<other arm>, base, {place_near})"
    assert doc.notes == "Close the chosen gripper firmly; use <other arm> for the set_down."
    assert "active_arm" not in doc.params and doc.params.keys() < old.params.keys()
    assert ("params.active_arm", "left", "removed") in rewrite.changes
    assert doc.instruction == old.instruction and doc.lessons == old.lessons
    assert unbind_arms(doc).changes == []


def _stage(name: str, call: str, arm: str = "", done: str = "") -> WorkflowStage:
    return WorkflowStage(name=name, goal=name, arm=arm, call=call, done=done)


def _handover(memory: MemoryDoc) -> MemoryDoc:
    """A hand-over: one arm picks the block, the other takes it while the first holds still."""
    return memory.replace(
        workflow=[
            _stage(
                "pick",
                "grasp_object(arm=<arm>, obj=block, grasp={grasp_handle})",
                arm=rule("pick"),
            ),
            _stage(
                "take",
                "together(moves=[{arm: <other arm>, action: grasp_object(obj=block, "
                "grasp={grasp_handle})}, {arm: <arm>, action: move_by(z=0.0)}])",
                arm="the arms plan_arms gives for take",
            ),
            _stage("release", "open_gripper(arm=<arm>)", arm="the giving arm"),
            _stage(
                "put",
                "place_object(arm=<other arm>, obj=block, place={place_on_base})",
                arm="the receiving arm",
            ),
        ],
    )


def test_holds_span_a_hand_over_and_the_stages_around_it(memory: MemoryDoc) -> None:
    chains = arm_chains(_handover(memory))
    giver, receiver = chains.holds
    assert (giver.role, giver.opens, giver.closes) == ("block", "pick", "release")
    assert (receiver.receives_from, receiver.opens, receiver.closes) == (0, "take", "put")
    assert receiver.place == "place_on_base" and chains.together == [(0, 1)]
    assert chains.stages["put"] == [1]


def test_a_co_grasp_is_two_holds_of_one_role_not_a_hand_over(memory: MemoryDoc) -> None:
    doc = memory.replace(
        workflow=[
            _stage(
                "lift",
                "together(moves=[{arm: <arm>, action: grasp_object(obj=pot)}, "
                "{arm: <other arm>, action: grasp_object(obj=pot)}])",
            )
        ]
    )
    first, second = arm_chains(doc).holds
    assert (first.role, second.role) == ("pot", "pot") and second.receives_from is None


def test_holds_read_a_positional_object_and_path(memory: MemoryDoc) -> None:
    doc = memory.replace(
        workflow=[
            _stage("take", "grasp_object(<arm>, cup, grasp={grasp_handle})"),
            _stage("carry", "follow_path(<arm>, {lift_path})"),
            _stage("put", "place_object(<arm>, cup, place={place_on_base})"),
        ]
    )
    (hold,) = arm_chains(doc).holds
    assert (hold.role, hold.path, hold.place, hold.closes) == (
        "cup",
        "lift_path",
        "place_on_base",
        "put",
    )


def test_a_stage_that_calls_plan_arms_lets_the_actor_assign_its_own_arms(
    memory: MemoryDoc,
) -> None:
    """S1's named-argument rule: `plan_arms()` names no argument, yet `assign` (the Actor's own
    choice, logged with its reason) is not refused."""
    from roboscript_methods.wicl.imitate import stage_arguments

    doc = memory.replace(workflow=[_stage("arms", "plan_arms()"), *memory.workflow])
    named = stage_arguments(doc)
    assert named["arms"] == {"plan_arms": ["assign"]}
    assert all("assign" not in args for args in named[memory.workflow[0].name].values())
