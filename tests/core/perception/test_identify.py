"""perception.identify: the identify-once chain on a synthetic scene with a scripted model and a
fake vision service: abstract roles with the live instruction, the three-way vote in one box
call, and refusals per call against an absent model."""

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import InfraError, ToolError
from roboscript.contracts.ids import FrameId
from roboscript.contracts.vision import Detection
from roboscript.perception.grounding import mask_box
from roboscript.perception.identify import Identified, identify_scene
from roboscript.perception.proposals import propose
from roboscript.perception.registry import SceneRegistry
from roboscript.perception.views import FrameGeometry, frame_geometry
from roboscript.perception.vision import VisionDetector
from roboscript.perception.vlm import (
    BoxGrounder,
    SomGrounder,
    VlmConfig,
)
from roboscript.testing.perception import (
    FakeVision,
    ScriptedModel,
    asked_for,
    box_query,
    instruction_of,
    prompt_of,
    query_roles,
    som_queries,
)
from roboscript.testing.scenes import (
    HEAD,
    PERCEPTION,
    TABLE_Z,
    Box,
    Cylinder,
    capture,
    look_at,
    perception,
    pinhole,
)

WORDS = perception(scene_words="tabletop", anchor_words="a robot's head camera")
SHAPES = {
    "red cube": Box((-0.15, 0.05, TABLE_Z + 0.025), (0.05, 0.05, 0.05), 20, (200, 30, 30)),
    "green bar": Box((0.12, -0.05, TABLE_Z + 0.02), (0.12, 0.04, 0.04), -35, (40, 170, 60)),
    "blue can": Cylinder((0.02, 0.15, TABLE_Z + 0.04), radius=0.03, length=0.08),
}
ROLES = {"cube": "red cube", "bar": "green bar", "can": "blue can"}
"""Roles with descriptions (the Demo Reader's `locate`)."""
ABSTRACT = {
    "cube": "what the robot picks up; a small block",
    "bar": "what the cube is put beside; a long block",
    "can": "what stays put; a round container",
}
"""Abstract roles, as a Memory's OBJECTS has them."""
INSTRUCTION = "Put the red cube beside the green bar, away from the blue can."
PHRASES = {"cube": "red cube", "bar": "green bar", "can": "blue can"}


class World:
    """The scene, its analysis, which proposal shows each shape and each shape's head box."""

    def __init__(self) -> None:
        cameras = [
            pinhole("overhead", HEAD),
            pinhole("wrist_left", look_at([-0.45, 0.0, 0.95], [0.0, 0.05, TABLE_Z]), 320, 240),
        ]
        images, index = capture(list(SHAPES.values()), cameras)
        self.index = index
        self.geometry: FrameGeometry = frame_geometry(FrameId("frame-1"), 0, images, PERCEPTION)
        head = index["overhead"]
        proposals = propose(self.geometry, PERCEPTION)
        self.marks: dict[str, int] = {}
        self.boxes: dict[str, tuple[float, float, float, float]] = {}
        for k, name in enumerate(SHAPES):
            shape = head == k
            overlap = [(p.masks["overhead"] & shape).sum() for p in proposals]
            self.marks[name] = proposals[int(np.argmax(overlap))].id
            self.boxes[name] = mask_box(shape)
        self.truth = {name: np.asarray(shape.center) for name, shape in SHAPES.items()}


@pytest.fixture(scope="module")
def world() -> World:
    return World()


def _box(box: Any) -> dict[str, Any]:
    return {"found": box is not None, "box": None if box is None else list(box), "point": None}


def model(
    marks: dict[str, int | None],
    boxes: dict[str, Any] | None = None,
    refuse: tuple[str, ...] = (),
    words: dict[str, dict[str, Any]] | None = None,
    letters: dict[str, str | None] | None = None,
) -> ScriptedModel:
    """Set-of-Mark answers `marks` by role (abstract roles, with the phrase `PHRASES` gives) or
    description; box requests get `boxes` the same way (a box, or None: not found); the words
    call `words` by role (by default the phrase, the role as the noun and nothing that singles
    it out); the look-alike call `letters` by role (None: no answer). A request whose kind is in
    `refuse` gets a 400."""

    def key(body: dict[str, Any], answer_key: str) -> str:
        return query_roles(body).get(answer_key) or som_queries(body)[answer_key]

    def said(role: str) -> dict[str, Any]:
        plain = {"phrase": PHRASES.get(role, role), "noun": role}
        return {**plain, "color": None, "size": None, "place": None, **(words or {}).get(role, {})}

    def answer(body: dict[str, Any]) -> Any:
        kind = asked_for(body)
        if kind in refuse:
            return httpx.Response(400, json={"error": {"message": "refused"}})
        if kind == "role_words":
            return {k: said(role) for k, role in query_roles(body).items()}
        if kind == "compare":
            return {k: (letters or {}).get(role) for k, role in query_roles(body).items()}
        if kind == "set_of_mark" and query_roles(body):
            return {
                k: {"phrase": PHRASES.get(role, role), "mark": marks.get(role)}
                for k, role in query_roles(body).items()
            }
        if kind == "set_of_mark":
            return {k: marks.get(text) for k, text in som_queries(body).items()}
        if kind == "boxes":
            return {k: _box((boxes or {}).get(key(body, k))) for k in som_queries(body)}
        return _box((boxes or {}).get(box_query(body)))

    return ScriptedModel(answer)


def vision(blobs: LocalBlobStore, boxes: dict[str, Any]) -> FakeVision:
    def detect(query: str) -> list[Detection]:
        box = boxes.get(query)
        return [] if box is None else [Detection(box=box, score=0.4, label=query)]

    return FakeVision(blobs, boxes=detect)


async def run(
    world: World,
    scripted: ScriptedModel,
    detector: FakeVision | None,
    roles: dict[str, str] = ROLES,
    blobs: LocalBlobStore | None = None,
    registry: SceneRegistry | None = None,
    instruction: str | None = None,
    config: VlmConfig | None = None,
) -> tuple[dict[str, Identified], SceneRegistry, list[tuple[str, dict[str, Any]]]]:
    steps: list[tuple[str, dict[str, Any]]] = []
    registry = SceneRegistry() if registry is None else registry
    client = scripted.client()
    found = await identify_scene(
        world.geometry,
        roles,
        registry,
        instruction=instruction,
        som=SomGrounder(client, config, blobs=blobs, pipeline=WORDS),
        box=BoxGrounder(client, config, blobs=blobs, pipeline=WORDS),
        detector=None if detector is None else VisionDetector(detector, detector.blobs),
        config=WORDS,
        log=lambda step, **data: steps.append((step, data)),
    )
    return found, registry, steps


def center_error(registry: SceneRegistry, found: Identified, truth: np.ndarray) -> float:
    assert found.object is not None
    return float(np.linalg.norm(np.asarray(registry.get(found.object).fit.center) - truth))


def by_role(world: World) -> dict[str, int | None]:
    return {role: world.marks[text] for role, text in ROLES.items()}


async def test_abstract_roles_with_the_live_instruction(world: World, tmp_path: Path) -> None:
    """The prompt carries the instruction and the abstract roles; the model's phrases are the
    cross-check's queries; confirmed marks are registered, one call for every role."""
    blobs = LocalBlobStore(tmp_path)
    scripted = model(by_role(world))
    detector = vision(blobs, world.boxes)
    found, registry, steps = await run(
        world, scripted, detector, ABSTRACT, blobs, instruction=INSTRUCTION
    )
    assert [f.reason for f in found.values()] == ["ok", "ok", "ok"]
    assert all(f.confidence == 1.0 and not f.ambiguous for f in found.values())
    assert [found[role].object for role in ABSTRACT] == ["o1", "o2", "o3"]  # in role order
    for role, text in ROLES.items():
        assert center_error(registry, found[role], world.truth[text]) < 0.005
        assert registry.get(found[role].object or "").name == role
        assert found[role].phrase == PHRASES[role]
    kinds = sorted(asked_for(r) for r in scripted.requests)
    assert kinds == ["role_words", "set_of_mark"]  # one call for every role, none for boxes
    [request] = [r for r in scripted.requests if asked_for(r) == "set_of_mark"]
    [words] = [r for r in scripted.requests if asked_for(r) == "role_words"]
    assert instruction_of(words) == INSTRUCTION and query_roles(words) == query_roles(request)
    assert not [p for p in words["input"][0]["content"] if p["type"] == "input_image"]
    assert instruction_of(request) == INSTRUCTION
    assert "Different roles are different objects." in prompt_of(request)
    assert "is joined to it by a line" in prompt_of(request)
    assert query_roles(request) == {"r1": "cube", "r2": "bar", "r3": "can"}
    assert som_queries(request)["r1"] == ABSTRACT["cube"]
    assert detector.calls == [("detect", list(PHRASES.values()))]  # the phrases are the queries
    schema = request["text"]["format"]
    assert (schema["name"], schema["strict"]) == ("set_of_mark", True)
    mark = schema["schema"]["properties"]["r1"]["properties"]["mark"]
    assert sorted(world.marks.values()) == sorted(mark["anyOf"][0]["enum"])
    assert request["reasoning"] == {"effort": "medium"} and request["model"] == "grounder"
    images = [part for part in request["input"][0]["content"] if part["type"] == "input_image"]
    assert len(images) == 2 and images[0]["image_url"].startswith("data:image/jpeg;base64,")
    names = [step for step, _ in steps]
    assert names[:4] == ["proposals", "som", "words", "detect"]
    assert names[4:] == ["decide"] * 3 + ["lift", "fit", "register"] * 3
    som = dict(steps)["som"]
    assert som["asked"]["prompt"].startswith("Two images of the same tabletop")
    assert all(blobs.exists(BlobRef.model_validate(ref)) for ref in som["asked"]["images"])
    assert som["asked"]["usage"]["input_tokens"] == 1000 and som["asked"]["deployment"]


async def test_the_box_breaks_a_tie_two_of_three_decide(world: World, tmp_path: Path) -> None:
    """GroundingDINO contradicts three marks; one box call for the three: the box siding with
    the mark keeps it (dissent: GroundingDINO), with GroundingDINO takes its object (dissent:
    Set-of-Mark), and with neither leaves the role ambiguous."""
    blobs = LocalBlobStore(tmp_path)
    marks = by_role(world)
    marks["bar"] = world.marks["blue can"]  # Set-of-Mark is wrong for the bar
    check = {
        "red cube": world.boxes["green bar"],  # GroundingDINO is wrong for the cube
        "green bar": world.boxes["green bar"],
        "blue can": world.boxes["red cube"],  # and for the can
    }
    boxes = {
        "cube": world.boxes["red cube"],  # sides with Set-of-Mark
        "bar": world.boxes["green bar"],  # sides with GroundingDINO
        "can": None,  # finds nothing: no majority
    }
    scripted = model(marks, boxes)
    found, registry, steps = await run(
        world, scripted, vision(blobs, check), ABSTRACT, blobs, instruction=INSTRUCTION
    )
    asked = [asked_for(r) for r in scripted.requests]
    assert sorted(asked[:2]) == ["role_words", "set_of_mark"] and asked[2:] == ["boxes"]
    assert query_roles(scripted.requests[2]) == {"r1": "cube", "r2": "bar", "r3": "can"}
    [box] = [data for step, data in steps if step == "box"]
    assert (box["fallback"], box["tie_break"]) == ([], ["cube", "bar", "can"])
    cube, bar, can = found["cube"], found["bar"], found["can"]
    assert (cube.reason, cube.confidence, cube.dissent) == ("ok", 1.0, ["grounding_dino"])
    assert center_error(registry, cube, world.truth["red cube"]) < 0.005
    assert (bar.reason, bar.confidence, bar.dissent) == ("ok", 1.0, ["som"])
    assert center_error(registry, bar, world.truth["green bar"]) < 0.005  # GroundingDINO's
    assert set(bar.evidence) == {"som", "grounding_dino", "vlm"}  # the dissent stays recorded
    assert (can.reason, can.ambiguous, can.confidence) == ("ambiguous", True, 0.5)
    assert can.object is not None and registry.get(can.object).ambiguous
    shown = [(c.backend, c.proposal) for c in can.candidates]
    assert shown == [("som", world.marks["blue can"]), ("grounding_dino", world.marks["red cube"])]
    record = registry.get(can.object)
    assert [c.backend for c in record.candidates] == ["som", "grounding_dino"]


QUICK = VlmConfig(retry_waits_s=[0.0, 0.01])


@pytest.mark.parametrize(
    ("status", "error"), [(400, ToolError), (429, InfraError), (503, InfraError)]
)
async def test_a_refusal_is_the_tool_s_an_absent_model_the_infrastructure_s(
    world: World, status: int, error: type[Exception]
) -> None:
    body = {"error": {"message": "no"}}
    scripted = ScriptedModel(lambda _: httpx.Response(status, json=body))
    rgb = world.geometry.view("overhead").image.rgb
    with pytest.raises(error):
        await BoxGrounder(scripted.client(), QUICK, pipeline=WORDS).locate(rgb, "a thing")
    assert len(scripted.requests) == (1 if status == 400 else 3)  # a refusal is not asked again
