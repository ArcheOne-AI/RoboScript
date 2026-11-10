"""RoboTwin's prompt words: with the perception config every RoboTwin run path binds (S1 and S2
through `run_s2.scene_config`, which the pipeline's jobs mount; P3's Agent host; P1's demo
coverage) and RoboTwin's embodiment, every VLM prompt and the Actor's first line read exactly as
they did when the words were hard-coded (aabff27e: "tabletop", "a robot's head camera", "a
two-armed robot at a table"). The box prompts go through `BoxGrounder` itself, as the Perception
toolset builds it; the others are rendered as their call sites render them."""

from typing import Any

import numpy as np
import pytest

from roboscript.perception.config import PerceptionConfig
from roboscript.perception.vlm import (
    BoxGrounder,
    VlmConfig,
    axis_prompt,
    camera_words,
    compare_prompt,
    goal_prompt,
    part_prompt,
    roles_prompt,
    som_prompt,
    som_roles_prompt,
    words_prompt,
)
from roboscript.settings import repo_root
from roboscript.testing.perception import ScriptedModel, asked_for, prompt_of
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.config import with_perception_binding
from roboscript_robotwin.eval.launch import P3Config
from tests.benchmarks.robotwin.test_run_s2 import load_script

INSTRUCTION = "Lift the red block and keep the blue one where it is."
CARD = "1: red, 5 x 5 x 5 cm, at (-0.15, 0.05)\n2: blue, 5 x 5 x 5 cm, at (0.02, 0.15)"
QUERIES = {"r1": "red block", "r2": "blue block"}
ROLES = {"r1": ("lifted", "what the robot lifts; a small block"), "r2": ("kept", "what stays")}
CANDIDATES = {"r1": ("lifted", "what the robot lifts; a small block", "red block", ["A", "B"])}
WIDTH, HEIGHT = 64, 48

EXPECTED: dict[str, str] = {
    "camera.head": "a robot's head camera",
    "camera.wrist": "a camera on the robot's left wrist",
    "som": "Two images of the same tabletop from a robot's head camera: the plain image, then the "
    "same image with candidate objects outlined and numbered (the marks); a mark's number "
    "sits on its object or is joined to it by a line. Measured attributes of each mark "
    "(color is the median pixel color; sizes and positions are from depth):\n"
    "1: red, 5 x 5 x 5 cm, at (-0.15, 0.05)\n"
    "2: blue, 5 x 5 x 5 cm, at (0.02, 0.15)\n"
    "\n"
    "For each object description below, give the number of the mark that shows that "
    "object, or null if no mark shows it.\n"
    '- "r1": red block\n'
    '- "r2": blue block\n'
    "\n"
    'Answer with JSON: {"r1": <mark number or null>, "r2": <mark number or null>}',
    "som.wrist": "Two images of the same tabletop from a camera on the robot's left wrist: the "
    "plain image, then the same image with candidate objects outlined and numbered "
    "(the marks); a mark's number sits on its object or is joined to it by a line. "
    "Measured attributes of each mark (color is the median pixel color; sizes and "
    "positions are from depth):\n"
    "1: red, 5 x 5 x 5 cm, at (-0.15, 0.05)\n"
    "2: blue, 5 x 5 x 5 cm, at (0.02, 0.15)\n"
    "\n"
    "For each object description below, give the number of the mark that shows that "
    "object, or null if no mark shows it.\n"
    '- "r1": red block\n'
    '- "r2": blue block\n'
    "\n"
    'Answer with JSON: {"r1": <mark number or null>, "r2": <mark number or null>}',
    "som_roles": "Two images of the same tabletop from a robot's head camera: the plain image, "
    "then the same image with candidate objects outlined and numbered (the marks); a "
    "mark's number sits on its object or is joined to it by a line. Measured "
    "attributes of each mark (color is the median pixel color; sizes and positions "
    "are from depth):\n"
    "1: red, 5 x 5 x 5 cm, at (-0.15, 0.05)\n"
    "2: blue, 5 x 5 x 5 cm, at (0.02, 0.15)\n"
    "\n"
    "The robot's instruction: \"Lift the red block and keep the blue one where it "
    'is."\n'
    "The task's objects, by role: what each does in the task and what kind of object "
    "it is. Which object on the table plays a role is decided by the instruction's "
    "words for it (such as a color, a size, a shape or where it is). A role's "
    "description is only a hint at its shape and use, written for another episode of "
    "the task: where it names a kind of object the instruction does not, the "
    "instruction's object plays the role, never one named only by the description.\n"
    '- "r1" lifted: what the robot lifts; a small block\n'
    '- "r2" kept: what stays\n'
    "\n"
    'For each role, give "phrase", the words of the instruction that name its '
    "object, copied from the instruction (the role's name if the instruction does "
    'not name it; never words of the description), and "mark", the number of the '
    "mark that shows that object, or null if no mark shows it. Different roles are "
    "different objects.\n"
    'Answer with JSON: {"r1": {"phrase": <text>, "mark": <number or null>}, "r2": '
    '{"phrase": <text>, "mark": <number or null>}}',
    "roles": 'A two-armed robot at a table is told: "Lift the red block and keep the blue one '
    'where it is."\n'
    "List the objects on the table that this instruction is about (not the robot, its "
    "arms or grippers, and not the table). For each give a role, a short lowercase name "
    "of one or two words joined by _ (such as <noun> or <noun>_<qualifier>), and a "
    "description to find it by in an image: the instruction's own words for it, with the "
    'color, shape or material the instruction gives. Answer with JSON: {"objects": '
    '[{"role": ..., "description": ...}]}',
    "goal": "Image pairs from a robot's cameras (pair 1: a robot's head camera; pair 2: a camera "
    "on the robot's left wrist): in each pair the first image is the scene now, the "
    "second the end state of a demonstration of the same task on another scene. The task: "
    '"Lift the red block and keep the blue one where it is.". Objects may sit elsewhere '
    "on the table and look different in detail; compare only how the task's objects sit "
    "relative to each other and to the robot's grippers. Does the scene now show the "
    'task\'s end state? Answer with JSON: {"reached": <true, false, or null if the views '
    'cannot tell>, "differs": "<what still differs from the end state, empty if '
    'nothing>"}',
    "words": 'The robot\'s instruction: "Lift the red block and keep the blue one where it is."\n'
    "The task's objects, by role: what each does in the task and what kind of object it "
    "is. Which object on the table plays a role is decided by the instruction's words "
    "for it (such as a color, a size, a shape or where it is). A role's description is "
    "only a hint at its shape and use, written for another episode of the task: where it "
    "names a kind of object the instruction does not, the instruction's object plays the "
    "role, never one named only by the description.\n"
    '- "r1" lifted: what the robot lifts; a small block\n'
    '- "r2" kept: what stays\n'
    "\n"
    'For each role give: "phrase", the instruction\'s words for its object, copied from '
    "the instruction (the role's name if the instruction does not name it; never words "
    'of the description); "noun", the kind of object the instruction names in one '
    "lowercase word; and only from words that describe which object it is (never from "
    'where it goes): "color", a color word the instruction gives it, else null; "size", '
    '"largest", "middle" or "smallest" when the words single it out by size among '
    'objects of its kind, else null; "place", "leftmost", "rightmost", "nearest", '
    '"farthest" or "middle" when the words single it out by where it is now, as the '
    'robot sees it, else null; "named", false only when the instruction asks for no '
    "object in this role (it names fewer objects of this kind than there are roles of "
    "it: the roles it leaves out), else true; words that refer to several objects at "
    "once (a number such as two, both, all, each, the rest, the others, a plural) name "
    'every role of the kind they cover; "arm", "left" or "right" when the instruction '
    "says which robot arm (hand, gripper) acts on this role's object (read lists in "
    "parallel: the k-th arm it lists goes with the k-th object it lists), else null "
    "(left of or right of an object is where something goes, not an arm).\n"
    'Answer with JSON: {"r1": {...}, "r2": {...}}',
    "compare": "The first image shows candidate objects from a robot's head camera, cut out and "
    "enlarged, each under its letter; below a candidate, the same object seen closer "
    "by a wrist camera; the second image is the whole view with the same candidates "
    "outlined and numbered in letter order (1 = A, 2 = B, ...).\n"
    "The robot's instruction: \"Lift the red block and keep the blue one where it "
    'is."\n'
    "The task's objects, by role: what each does in the task and what kind of object "
    "it is. Which object on the table plays a role is decided by the instruction's "
    "words for it (such as a color, a size, a shape or where it is). A role's "
    "description is only a hint at its shape and use, written for another episode of "
    "the task: where it names a kind of object the instruction does not, the "
    "instruction's object plays the role, never one named only by the description.\n"
    '- "r1" lifted: what the robot lifts; a small block; the instruction\'s words for '
    'it: "red block"; candidates A, B\n'
    "\n"
    "For each role, give the letter of the candidate that is the object the "
    "instruction means for it, or null if none of its candidates is. Different roles "
    "are different objects.\n"
    'Answer with JSON: {"r1": <letter or null>}',
    "compare.no_wrists": "The first image shows candidate objects from a robot's head camera, cut "
    "out and enlarged, each under its letter; the second image is the whole "
    "view with the same candidates outlined and numbered in letter order (1 "
    "= A, 2 = B, ...).\n"
    "The robot's instruction: \"Lift the red block and keep the blue one "
    'where it is."\n'
    "The task's objects, by role: what each does in the task and what kind "
    "of object it is. Which object on the table plays a role is decided by "
    "the instruction's words for it (such as a color, a size, a shape or "
    "where it is). A role's description is only a hint at its shape and use, "
    "written for another episode of the task: where it names a kind of "
    "object the instruction does not, the instruction's object plays the "
    "role, never one named only by the description.\n"
    '- "r1" lifted: what the robot lifts; a small block; the instruction\'s '
    'words for it: "red block"; candidates A, B\n'
    "\n"
    "For each role, give the letter of the candidate that is the object the "
    "instruction means for it, or null if none of its candidates is. "
    "Different roles are different objects.\n"
    'Answer with JSON: {"r1": <letter or null>}',
    "part": "Two images from a robot's head camera: the plain image, then the same image with "
    "regions of one object outlined and numbered (the marks); a mark's number sits on its "
    'region or is joined to it by a line. Which mark shows "the top of the red block"? '
    'Left and right are as the robot sees them. Answer with JSON: {"mark": <number, or '
    "null if no mark shows it>}",
    "axis": "Three images from a robot's head camera: the plain image, the same image with arrows "
    "drawn from one object, lifted: its candidate x axes in red, labeled a, b, and its z "
    "(up) in blue, then a close-up of that object with the same arrows. Common sense: "
    'this kind of object\'s front is "the side with a logo". Which labeled arrow points '
    'toward "the far side" of that object? Answer with JSON: {"arrow": <label, or null if '
    "no arrow does or the object cannot be seen>}",
    "actor": "You are the Actor. You carry out a manipulation task with a two-armed robot at a "
    "table, only by calling tools: you look with the sensing and perception tools, act "
    "with physical actions, and say when you are done with `finish`.",
    "box.head_camera": "The image is 64x48 pixels (width x height), a tabletop seen by a robot's "
    'head camera. Locate: "red block". Answer with JSON: {"found": true|false, '
    '"box": [x_min, y_min, x_max, y_max], "point": [x, y]} in pixel '
    "coordinates of this image (origin top-left, x to the right, y down). The "
    "box must tightly enclose the whole object; the point must lie on the "
    "object itself. If the object is not visible, answer found=false.",
    "box.left_camera": "The image is 64x48 pixels (width x height), a tabletop seen by a camera "
    'on the robot\'s left wrist. Locate: "red block". Answer with JSON: '
    '{"found": true|false, "box": [x_min, y_min, x_max, y_max], "point": [x, '
    "y]} in pixel coordinates of this image (origin top-left, x to the right, "
    "y down). The box must tightly enclose the whole object; the point must "
    "lie on the object itself. If the object is not visible, answer "
    "found=false.",
    "boxes.descriptions": "The image is 64x48 pixels (width x height), a tabletop seen by a "
    "robot's head camera. Locate each object described below.\n"
    '- "r1": red block\n'
    "\n"
    "For each, locate the object: the box must tightly enclose the whole "
    "object and the point must lie on the object itself, in pixel "
    "coordinates of this image (origin top-left, x to the right, y down). "
    "If the object is not visible, answer found=false.\n"
    'Answer with JSON: {"r1": {"found": true|false, "box": [x_min, y_min, '
    'x_max, y_max], "point": [x, y]}}',
    "boxes.instruction": "The image is 64x48 pixels (width x height), a tabletop seen by a "
    "robot's head camera. The robot's instruction: \"Lift the red block and "
    'keep the blue one where it is."\n'
    "The task's objects, by role: what each does in the task and what kind "
    "of object it is. Which object on the table plays a role is decided by "
    "the instruction's words for it (such as a color, a size, a shape or "
    "where it is). A role's description is only a hint at its shape and use, "
    "written for another episode of the task: where it names a kind of "
    "object the instruction does not, the instruction's object plays the "
    "role, never one named only by the description.\n"
    '- "r1" r1: what the robot lifts; a small block\n'
    "\n"
    "For each, locate the object the instruction means: the box must tightly "
    "enclose the whole object and the point must lie on the object itself, "
    "in pixel coordinates of this image (origin top-left, x to the right, y "
    "down). If the object is not visible, answer found=false.\n"
    'Answer with JSON: {"r1": {"found": true|false, "box": [x_min, y_min, '
    'x_max, y_max], "point": [x, y]}}',
}
"""The prompts at aabff27e with these inputs (its own `vlm.py` and `actor.md` rendered them)."""


def bound_pipelines(monkeypatch: pytest.MonkeyPatch) -> list[PerceptionConfig]:
    """The scene pipeline each RoboTwin run path perceives with, no config of its own given."""
    script = load_script()

    def missing(*_: Any, **__: Any) -> Any:
        raise FileNotFoundError("no configs/scene.yaml")

    monkeypatch.setattr(script, "load_config", missing)
    s2 = script.scene_config().pipeline  # S1, S2 and the pipeline's jobs (`run_s2.toolsets`)
    host = {"host": {"toolsets": [{"name": "perception"}, {"name": "done"}]}}
    p3 = [
        PerceptionConfig.model_validate(spec["config"]["pipeline"])
        for spec in P3Config._perception_binding(host)["host"]["toolsets"]
    ]
    p1 = PerceptionConfig.model_validate(with_perception_binding({}))  # p1/cover_demos.py
    return [s2, *p3, p1]


async def grounder_prompts(pipeline: PerceptionConfig) -> dict[str, str]:
    """The box prompts `BoxGrounder` sends, built as the Perception toolset builds it."""
    found = {"found": False, "box": [0, 0, 1, 1], "point": [0, 0]}
    model = ScriptedModel(lambda body: found if asked_for(body) == "box" else {"r1": found})
    grounder = BoxGrounder(model.client(), VlmConfig(retry_waits_s=[]), pipeline=pipeline)
    rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    anchor = pipeline.anchor
    out = {}
    for camera in ("head_camera", "left_camera"):
        await grounder.locate(rgb, "red block", camera, anchor)
        out[f"box.{camera}"] = prompt_of(model.requests[-1])
    await grounder.locate_roles(rgb, {"r1": "red block"}, "head_camera", anchor)
    out["boxes.descriptions"] = prompt_of(model.requests[-1])
    roles = {"r1": "what the robot lifts; a small block"}
    await grounder.locate_roles(rgb, roles, "head_camera", anchor, INSTRUCTION)
    out["boxes.instruction"] = prompt_of(model.requests[-1])
    return out


def rendered(pipeline: PerceptionConfig, robot: str) -> dict[str, str]:
    """The other prompts, with the words their call sites pass (`SomGrounder.choose`,
    `identify._compare`, `Perception._roles`, `Done`'s `goal_match`, the part and axis tools)."""
    head = camera_words("head_camera", pipeline, pipeline.anchor)
    wrist = camera_words("left_camera", pipeline, pipeline.anchor)
    scene = pipeline.scene_words
    actor = (repo_root() / "recipes/wicl/prompts/actor.md").read_text("utf-8").splitlines()[0]
    return {
        "camera.head": head,
        "camera.wrist": wrist,
        "som": som_prompt(CARD, QUERIES, head, scene),
        "som.wrist": som_prompt(CARD, QUERIES, wrist, scene),
        "som_roles": som_roles_prompt(CARD, ROLES, INSTRUCTION, head, scene),
        "roles": roles_prompt(INSTRUCTION, robot),
        "goal": goal_prompt(["head_camera", "left_camera"], INSTRUCTION, pipeline),
        "words": words_prompt(ROLES, INSTRUCTION),
        "compare": compare_prompt(CANDIDATES, INSTRUCTION, True, pipeline.anchor_words),
        "compare.no_wrists": compare_prompt(CANDIDATES, INSTRUCTION, False, pipeline.anchor_words),
        "part": part_prompt("the top of the red block", head),
        "axis": axis_prompt("lifted", "the far side", ["a", "b"], head, "the side with a logo"),
        "actor": actor.replace("{{robot}}", robot),
    }


async def test_robotwin_prompts_read_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    pipelines = bound_pipelines(monkeypatch)
    assert all(p == pipelines[0] for p in pipelines)
    robot = ROBOTWIN.profile().description
    for pipeline in pipelines:
        texts = rendered(pipeline, robot) | await grounder_prompts(pipeline)
        assert set(texts) == set(EXPECTED)
        for name, text in texts.items():
            assert text == EXPECTED[name], name
