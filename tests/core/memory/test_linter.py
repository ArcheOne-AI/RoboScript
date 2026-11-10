"""The linter: the recipe's vocabularies, the config check, fixed arms as errors, and privileged
references as warnings."""

from typing import Any

import pytest
from pydantic import ValidationError

from roboscript.memory import (
    GraspSpec,
    LintConfig,
    MemoryDoc,
    MemoryRecipe,
    NumberParam,
    ToolSig,
    lint,
)
from roboscript.memory.linter import errors


def put(doc: MemoryDoc, where: str, text: str) -> MemoryDoc:
    """`doc` with `text` in one place: an OBJECTS role's description, the notes, a stage's call,
    a lesson's action, a part name, or the name of an extra parameter that the notes name."""
    if where == "objects":
        block = (
            "- base: what the tool strikes; a small cube, symmetric every 90 degrees; the "
            "tool's handle and head are its parts."
        )
        return doc.replace(objects=f"- tool: {text.removeprefix('- ')}\n{block}")
    if where == "notes":
        return doc.replace(notes=text)
    if where == "call":
        stage = doc.workflow[0].model_copy(update={"call": text})
        return doc.replace(workflow=[stage, *doc.workflow[1:]])
    if where == "lesson":
        return doc.replace(lessons=[doc.lessons[0].model_copy(update={"do": text})])
    if where == "name":
        param = NumberParam(value=0.0, unit="m", bounds=(-0.1, 0.1), trust=0.01)
        return doc.replace(params={**doc.params, text: param}, notes=f"Use {{{text}}}.")
    grasp = doc.params["grasp_handle"]
    spec = GraspSpec.model_validate({**dict(grasp.value), "part": text})
    return doc.replace(
        params={**doc.params, "grasp_handle": grasp.model_copy(update={"value": spec})}
    )


WHERE = {
    "objects": "objects:1",
    "notes": "notes:1",
    "call": "workflow.grasp_tool.call",
    "lesson": "lessons.L1.do",
    "part": "params.grasp_handle.part",
}


@pytest.mark.parametrize(
    ("where", "text", "rule", "match"),
    [
        (
            "objects",
            "- grasp at anchor_point 2 of the tool",
            "asset_annotation",
            "anchor_point",
        ),
        ("lesson", "use marker_point_id 1", "asset_annotation", "marker_point"),
        ("part", "ap_3", "asset_annotation", "ap_3"),
        ("part", "ap1_handle", "asset_annotation", "ap1"),
        ("part", "simx_handle", "simulator_name", "simx"),
        ("call", "grasp_object(arm=A, obj=id020)", "simulator_name", "id020"),
        ("objects", "- the task code calls it self.tool", "simulator_name", "self.tool"),
        ("lesson", "on seed 10004 use the nearer arm", "seed", "seed 10004"),
        ("lesson", "on seed10003 the tool lay left", "seed", "seed10003"),
        ("objects", "- it lay left in 10017", "seed", "10017"),
        ("lesson", "stop once success_probe holds", "evaluator", "success_probe"),
        ("objects", "- pass L-K2 before closing", "evaluator", "L-K2"),
        ("lesson", "the ground-truth pose put it 3 cm left", "evaluator", "ground-truth"),
        ("lesson", "groundtruth showed the base had rotated", "evaluator", "groundtruth"),
        ("lesson", "the true pose of the tool was off", "evaluator", "true pose"),
        ("lesson", "the gt yaw was 20 degrees off", "evaluator", "gt"),
        ("lesson", "the red block is A", "episode_fact", "red block is A"),
        ("lesson", "A is the red block", "episode_fact", "A is the red"),
        ("call", "grasp_object(arm=A, obj=obj_3)", "episode_fact", "obj_3"),
        ("part", "part_2", "episode_fact", "part_2"),
        ("lesson", "in this try the grasp slipped", "episode_fact", "this try"),
        ("lesson", "in try 2 the grasp slipped", "episode_fact", "in try 2"),
        ("lesson", "place it at (0.12, -0.05, 0.76)", "absolute_position", "0.12, -0.05"),
        ("lesson", "place it at 0.12,-0.05 on the table", "absolute_position", "0.12,-0.05"),
        ("objects", "- the bin centre is at x 0.31", "absolute_position", "x 0"),
        ("objects", "- the base sits near y=-0.1", "absolute_position", "y=-0"),
        (
            "lesson",
            "the tool lies 0.31 m right of the robot",
            "robot_distance",
            "1 m right of the robot",
        ),
    ],
)
def test_vocabularies(
    memory: MemoryDoc, recipe: MemoryRecipe, where: str, text: str, rule: str, match: str
) -> None:
    findings = errors(lint(put(memory, where, text), recipe.lint))
    assert [(f.rule, f.where, f.match) for f in findings] == [(rule, WHERE[where], match)]
    assert findings[0].message == recipe.lint.rules[rule].message


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"rules": {"x": {"message": "m", "patterns": ["("]}}}, "pattern '\\('"),
        ({"rules": {"words": {"message": "m", "words": ["a"]}}}, "built in"),
        ({"rules": {"x": {"message": "m"}}}, "words or patterns"),
        ({"rules": {"x": {"message": "m", "words": ["a"], "fixed_targets": "world"}}}, "fixed"),
        ({"fixed_targets": "world"}, "fixed_targets"),
        ({"max_lessons": 0}, "max_lessons"),
    ],
)
def test_the_config_is_checked(data: dict[str, Any], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        LintConfig.model_validate(data)


def test_every_fixed_arm_is_an_error(memory: MemoryDoc, recipe: MemoryRecipe) -> None:
    """Anywhere in the text: an edit never lands on a Memory that binds an arm."""
    found = lint(put(memory, "notes", "pick the cup with the left arm"), recipe.lint)
    assert [(f.match, f.level) for f in found if f.rule == "fixed_arm"] == [("left arm", "error")]


@pytest.mark.parametrize(
    ("where", "text", "match"),
    [
        ("call", "move_to_place(tool, {place_on_base}) at K4", "K4"),
        ("notes", "the carry target is the K3 pose", "K3"),
        ("lesson", "stop when the official success check holds", "official success"),
        ("call", "the scene-level success check fires", "success check"),
        ("notes", "the release latched the success", "latched"),
        ("call", "if a finish attempt is refused, repeat", "finish attempt is refused"),
        ("name", "press_k3", "k3"),
    ],
)
def test_privileged_references_are_warnings(
    memory: MemoryDoc, recipe: MemoryRecipe, where: str, text: str, match: str
) -> None:
    """Keypoint ids with or without an arm, and the success machinery, are advice, not blocks."""
    found = [f for f in lint(put(memory, where, text), recipe.lint) if f.rule == "privileged_ref"]
    assert match in [f.match for f in found] and {f.level for f in found} == {"warning"}
    assert errors(lint(put(memory, where, text), recipe.lint)) == []


TOOLS = {
    "grasp_object": ToolSig(params=("arm", "obj", "grasp", "approach_m")),
    "place_object": ToolSig(params=("arm", "obj", "target")),
    "locate_part": ToolSig(params=("obj", "part", "x_toward")),
}
