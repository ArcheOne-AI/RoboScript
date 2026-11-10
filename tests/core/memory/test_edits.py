"""Typed edits: every operation, every rejection, INSTRUCTION never edited, the budget and
the revision's trust radius."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from roboscript.memory import (
    Change,
    EditCosts,
    Editor,
    EditRecord,
    Lesson,
    MemoryDoc,
    MemoryRecipe,
    WorkflowStage,
    open_cache,
)
from roboscript.testing.memory import sample_memory

WHY = {"why": "the grasp stopped short of the handle"}
NUMBER = {"kind": "number", "value": 0.5, "unit": "s", "bounds": [0, 2], "trust": 0.1}


def s1_editor(doc: MemoryDoc, recipe: MemoryRecipe) -> Editor:
    return Editor(doc, rules=recipe.rules("s1"), lint=recipe.lint, seed=0, source="S1 revision 1")


def cache(doc: MemoryDoc, recipe: MemoryRecipe, budget: int = 10) -> Editor:
    return open_cache(doc, budget=budget, seed=7, rules=recipe.rules("s2"), lint=recipe.lint)


def edit(editor: Editor, **op: Any) -> EditRecord:
    return editor.edit({**op, **WHY} if op["op"] != "scratch" else op)


def accepted(editor: Editor, **op: Any) -> MemoryDoc:
    record = edit(editor, **op)
    assert record.accepted, record.reason
    return editor.doc


def test_every_operation(memory: MemoryDoc, recipe: MemoryRecipe) -> None:
    editor = s1_editor(memory, recipe)
    doc = accepted(editor, op="set_number", param="pre_grasp", value=0.12)
    assert doc.params["pre_grasp"].value == 0.12
    assert doc.params["pre_grasp"].history == [Change(how="edit", before=0.1, after=0.12, seed=0)]
    offset = [-0.02, 0, -0.01]
    doc = accepted(editor, op="set_number", param="grasp_handle", field="offset_m", value=offset)
    assert doc.params["grasp_handle"].value.offset_m == (-0.02, 0.0, -0.01)
    doc = accepted(editor, op="set_number", param="grasp_handle", field="yaw_deg", value=10)
    doc = accepted(editor, op="set_category", param="arm_rule", value="holding")
    doc = accepted(editor, op="set_category", param="grasp_handle", field="approach", value="side")
    doc = accepted(editor, op="set_category", param="grasp_handle", field="x_toward", value=None)
    grasp = doc.params["grasp_handle"].value
    assert (grasp.yaw_deg, grasp.approach, grasp.x_toward) == (10.0, "side", None)
    target = {"kind": "landmark", "landmark": "table_center"}
    doc = accepted(editor, op="set_category", param="place_on_base", field="target", value=target)
    assert doc.params["place_on_base"].value.target.landmark == "table_center"
    doc = accepted(editor, op="add_lesson", when="the base is near the edge", do="set_down slower")
    assert doc.lessons[-1] == Lesson(
        id="L2", when="the base is near the edge", do="set_down slower", seed=0
    )
    doc = accepted(editor, op="edit_lesson", lesson="L1", do="approach from the top")
    assert (doc.lessons[0].when, doc.lessons[0].do) == (
        memory.lessons[0].when,
        "approach from the top",
    )
    assert (doc.lessons[0].seed, doc.lessons[0].merges) == (10004, 1)
    doc = accepted(editor, op="remove_lesson", lesson="L2")
    assert [lesson.id for lesson in doc.lessons] == ["L1"]
    doc = accepted(editor, op="edit_stage", stage="set_down", field="goal", text="hit the base")
    assert doc.workflow[1].goal == "hit the base"
    retreat = {"name": "retreat", "goal": "lift the tool clear", "call": "return_home(arm=A)"}
    doc = accepted(editor, op="add_stage", stage=retreat)
    doc = accepted(editor, op="add_stage", stage={"name": "look"}, before="grasp_tool")
    assert [s.name for s in doc.workflow] == ["look", "grasp_tool", "set_down", "retreat"]
    doc = accepted(editor, op="move_stage", stage="look")
    assert [s.name for s in doc.workflow] == ["grasp_tool", "set_down", "retreat", "look"]
    doc = accepted(editor, op="move_stage", stage="retreat", before="grasp_tool")
    assert [s.name for s in doc.workflow] == ["retreat", "grasp_tool", "set_down", "look"]
    doc = accepted(editor, op="move_stage", stage="grasp_tool", before="look")
    assert [s.name for s in doc.workflow] == ["retreat", "set_down", "grasp_tool", "look"]
    doc = accepted(editor, op="remove_stage", stage="look")
    doc = accepted(editor, op="set_done", stage="set_down", done="moved(base, {place_near})")
    assert doc.workflow[1] == WorkflowStage(
        name="set_down",
        goal="hit the base",
        arm="the arm holding the tool",
        call="place_object(arm=A, obj=tool, target={place_on_base})",
        done="moved(base, {place_near})",
    )
    doc = accepted(editor, op="define_param", name="settle_s", param=NUMBER)
    settle = doc.params["settle_s"]
    assert (settle.value, settle.source) == (0.5, "S1 revision 1")
    assert settle.history == [Change(how="define", after=0.5, seed=0)]
    doc = accepted(editor, op="remove_param", name="settle_s")
    doc = accepted(editor, op="set_text", section="notes", text="Measure first.\n\n- then act ")
    assert doc.notes == "Measure first.\n\n- then act"
    doc = accepted(editor, op="set_text", section="statement", text="Beat the base.")
    assert doc.statement == "Beat the base." and doc.instruction == memory.instruction
    assert memory == sample_memory()  # edits make new documents
    assert all(record.accepted for record in editor.records)
    assert editor.used == sum(EditCosts().of(record.op) for record in editor.records)


@pytest.mark.parametrize(
    ("op", "reason"),
    [
        ({"op": "set_number", "param": "nope", "value": 1}, "no parameter nope; parameters: pre_"),
        (
            {"op": "set_number", "param": "arm_rule", "value": 1},
            "arm_rule (category) has no numeric field null; its numeric fields: none (use set_ca",
        ),
        (
            {"op": "set_number", "param": "grasp_handle", "value": 1},
            "has no numeric field null; its numeric fields: offset_m, offset_frac, yaw_deg",
        ),
        (
            {"op": "set_number", "param": "grasp_handle", "field": "offset_m", "value": 0.1},
            "grasp_handle.offset_m takes [x, y, z]",
        ),
        (
            {"op": "set_number", "param": "grasp_handle", "field": "yaw_deg", "value": [1, 2, 3]},
            "grasp_handle.yaw_deg takes a number",
        ),
        (
            {"op": "set_number", "param": "pre_grasp", "value": 0.2},
            "pre_grasp: value 0.2 is outside its bounds [0.05, 0.15]",
        ),
        ({"op": "set_number", "param": "pre_grasp", "value": 0.1}, "pre_grasp is 0.1 already"),
        ({"op": "set_category", "param": "arm_rule", "value": "up"}, "not one of the choices"),
        (
            {"op": "set_category", "param": "pre_grasp", "value": "up"},
            "pre_grasp (number) has no categorical field null",
        ),
        (
            {"op": "set_category", "param": "place_on_base", "field": "target", "value": "base"},
            "place_on_base.target: Input should be",
        ),
        (
            {"op": "set_category", "param": "grasp_handle", "field": "part", "value": "Handle!"},
            "grasp_handle.part: String should match",
        ),
        ({"op": "edit_lesson", "lesson": "L9", "do": "x"}, "no lesson L9; lessons: L1"),
        (
            {"op": "edit_lesson", "lesson": "L1", "when": "the handle points away from the robot"},
            "lesson L1 says that already",
        ),
        (
            {"op": "edit_stage", "stage": "nope", "field": "goal", "text": "x"},
            "no workflow stage nope; stages: grasp_tool, set_down",
        ),
        (
            {
                "op": "edit_stage",
                "stage": "set_down",
                "field": "arm",
                "text": "the arm holding the tool",
            },
            "the arm of stage set_down says that already",
        ),
        ({"op": "add_stage", "stage": {"name": "set_down"}}, "there is a stage set_down already"),
        ({"op": "add_stage", "stage": {"name": "x"}, "before": "nope"}, "no workflow stage nope"),
        ({"op": "move_stage", "stage": "set_down", "before": "set_down"}, "cannot move before it"),
        ({"op": "move_stage", "stage": "set_down"}, "stage set_down is there already"),
        ({"op": "remove_stage", "stage": "nope"}, "no workflow stage nope"),
        (
            {
                "op": "define_param",
                "name": "pre_grasp",
                "param": {"kind": "category", "value": "a"},
            },
            "parameter pre_grasp exists",
        ),
        ({"op": "remove_param", "name": "nope"}, "no parameter nope"),
        ({"op": "remove_param", "name": "pre_grasp"}, "the Memory would fail the linter"),
        ({"op": "scratch", "text": "a thought"}, "SCRATCH exists only in a cache"),
        (
            {
                "op": "add_lesson",
                "when": "on seed 10004 the tool lay left",
                "do": "use the left arm",
            },
            "the Memory would fail the linter",
        ),
        (
            {
                "op": "define_param",
                "name": "acting_arm",
                "param": {"kind": "category", "value": "left"},
            },
            "the Memory would fail the linter",
        ),
        (
            {"op": "add_stage", "stage": {"name": "x", "call": "move_to_pose(arm=A, pose={nope})"}},
            "the Memory would fail the linter",
        ),
    ],
)
def test_rejections(
    memory: MemoryDoc, recipe: MemoryRecipe, op: dict[str, Any], reason: str
) -> None:
    editor = s1_editor(memory, recipe)
    record = edit(editor, **op)
    assert not record.accepted and reason in record.reason
    assert editor.doc == memory and editor.used == 0 and editor.rejected == [record]


def test_instruction_is_never_edited(memory: MemoryDoc, recipe: MemoryRecipe) -> None:
    instruction = {"op": "set_text", "section": "instruction", "text": "Beat the base."}
    for editor in (cache(memory, recipe), s1_editor(memory, recipe)):
        with pytest.raises(ValidationError):
            edit(editor, **instruction)
    editor = s1_editor(memory, recipe)
    task = {"op": "set_text", "section": "statement", "text": "Beat the base."}
    assert edit(editor, **task).accepted
    again = edit(editor, **task)
    assert not again.accepted and again.reason == "TASK says that already"
    assert edit(
        cache(memory, recipe), op="set_text", section="notes", text="Measure twice."
    ).accepted


def test_the_budget(memory: MemoryDoc, recipe: MemoryRecipe) -> None:
    editor = open_cache(
        memory.replace(version=3), budget=2, seed=10004, rules=recipe.rules("s2"), lint=recipe.lint
    )
    records = [
        edit(editor, op="set_category", param="arm_rule", value="holding"),
        edit(editor, op="add_stage", stage={"name": "retreat"}),
        edit(editor, op="set_number", param="pre_grasp", value=0.11),
        edit(editor, op="add_lesson", when="the tool lies diagonal", do="grasp nearer the head"),
        edit(editor, op="scratch", text="the handle looked shorter than measured"),
        edit(editor, op="edit_lesson", lesson="L1", do="approach from the top"),
    ]
    assert [r.accepted for r in records] == [True, False, True, True, True, False]
    assert [(r.cost, r.used, r.left) for r in records] == [
        (1, 1, 1),
        (2, 1, 1),
        (0, 1, 1),
        (1, 2, 0),
        (0, 2, 0),
        (1, 2, 0),
    ]
    assert records[1].reason == "add_stage costs 2; 1 of 2 units are left"
    assert [r.seq for r in editor.records] == list(range(1, len(records) + 1))
    assert editor.rejected == [records[1], records[5]] and (editor.used, editor.left) == (2, 0)
    assert editor.doc.scratch == ["the handle looked shorter than measured"]
    assert editor.doc.lessons[-1].seed == 10004
    assert EditRecord.model_validate_json(records[3].model_dump_json()) == records[3]


def test_an_edit_beyond_its_trust_radius_is_clamped(
    memory: MemoryDoc, recipe: MemoryRecipe
) -> None:
    """S2 moves an edit asking for more than its radius the radius toward it, along the straight
    line from the revision's start; the next revision may go one radius further; every other check
    sees the value applied."""
    editor = cache(memory, recipe)
    start = memory.params["grasp_handle"].value.offset_m
    radius = memory.params["grasp_handle"].trust.offset_m
    far = tuple(s + d for s, d in zip(start, (0.03, 0.0, -0.04), strict=True))  # 0.05 away
    record = edit(editor, op="set_number", param="grasp_handle", field="offset_m", value=list(far))
    assert record.accepted, record.reason
    applied = editor.doc.params["grasp_handle"].value.offset_m
    assert math.dist(start, applied) == pytest.approx(radius)
    step = [(a - s) / radius for s, a in zip(start, applied, strict=True)]
    assert step == pytest.approx([0.6, 0.0, -0.8])  # toward the value asked for
    assert record.op.value == pytest.approx(applied)  # type: ignore[union-attr]
    assert "asked for" in record.clamped and "its trust radius" in record.clamped
    # within the radius, or exactly at it: unchanged, no note
    exact = edit(editor, op="set_number", param="pre_grasp", value=0.13)  # 0.03 from 0.1
    assert exact.accepted and not exact.clamped
    assert editor.doc.params["pre_grasp"].value == 0.13
    # a scalar beyond: clamped from the revision's start, not from its value now
    back = edit(editor, op="set_number", param="pre_grasp", value=0.05)
    assert back.accepted and back.op.value == pytest.approx(0.07)  # type: ignore[union-attr]
    assert "from 0.1 (its value when this revision started)" in back.clamped
    editor.revise()
    again = edit(editor, op="set_number", param="grasp_handle", field="offset_m", value=list(far))
    assert again.accepted
    assert math.dist(start, editor.doc.params["grasp_handle"].value.offset_m) == pytest.approx(
        2 * radius
    )
    # a value out of its bounds is rejected as asked, not moved toward
    lift = edit(editor, op="set_number", param="lift_min", value=0.3)
    assert not lift.accepted and "outside its bounds" in lift.reason
    lift = edit(editor, op="set_number", param="lift_min", value=0.12)
    assert lift.accepted and editor.doc.params["lift_min"].value == pytest.approx(0.07)


POSE = {
    "anchor": "pose",
    "approach_axis": [0.0, 0.0, -1.0],
    "closing_axis": [1.0, 0.0, 0.0],
    "reference_q": [0.0, 1.0, 0.0, 0.0],
    "arm": "left",
}
