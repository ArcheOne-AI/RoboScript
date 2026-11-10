"""@tool, ToolModel and Toolset: models from signatures, strict params, definition-time checks."""

from dataclasses import dataclass
from typing import Annotated, Any, Literal, NamedTuple

import pytest
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import CallContext
from roboscript.tools import (
    FrameRef,
    FrameStore,
    ObjectRef,
    PartRef,
    Rationale,
    ToolContext,
    ToolModel,
    Toolset,
    tool,
    tool_signatures,
)
from roboscript.toolsets.manipulation import Manipulation

RATIONALE = {"stage": "grasp", "scene": "a box", "progress": "at the box", "plan": "lift it"}


class Grip(ToolModel):
    """How to hold it."""

    part: Annotated[str, Field(description="Which part to hold.")]
    depth: Annotated[float, Field(ge=0, le=0.1)] = 0.02


class Moved(BaseModel):
    obj: ObjectRef
    depth: float
    frames: list[FrameRef] = []


class ProbeConfig(BaseModel):
    speed: float = 1.0


class Probe(Toolset):
    name = "probe"
    Config = ProbeConfig

    @tool(kind="execution", tier="D", timeout_s=5)
    async def move(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        obj: ObjectRef,
        grip: Grip,
        arm: Literal["left", "right"] = "left",
        frame: FrameRef | None = None,
    ) -> Moved:
        """Move an object
        to where it belongs.

        Implementation notes that models never read.
        """
        return Moved(obj=obj, depth=grip.depth, frames=[frame] if frame else [])

    @tool(kind="pure", name="peek", tier="P")
    async def look(self, ctx: ToolContext) -> Moved:
        """Peek."""
        return Moved(obj="o", depth=0.0)


def context() -> ToolContext:
    async def service(ctx: ToolContext, tool: str, args: Any) -> BaseModel:
        raise AssertionError("not called")

    return ToolContext(
        context=CallContext(episode_ref=EpisodeId("ep-1"), stage="dev", tier="D"),
        toolset="probe",
        call_id="c1",
        env_client=None,
        blobs=None,  # type: ignore[arg-type]
        state={},
        frames=FrameStore(None, alias=str, emit=lambda kind, data: None),  # type: ignore[arg-type]
        service_call=service,
        event_sink=lambda ctx, kind, data: None,
    )


def test_specs_come_from_the_signature_and_docstring() -> None:
    move, peek = Probe.specs()
    assert (move.name, peek.name) == ("move", "peek")
    assert move.description == "Move an object to where it belongs."
    assert list(move.params.model_fields) == ["rationale", "obj", "grip", "arm", "frame"]
    assert move.result is Moved
    assert move.meta.model_dump() == {
        "toolset": "probe",
        "kind": "execution",
        "tier": "D",
        "system_only": False,
        "after_end": False,
        "uses_steps": True,
        "needs_rationale": True,
        "timeout_s": 5,
    }
    assert (peek.meta.kind, peek.meta.tier) == (
        "pure",
        "P",
    )  # the method's phases decide who sees it
    assert not peek.meta.uses_steps and not peek.meta.needs_rationale
    assert move.refs_in and move.refs_out and not peek.refs_in and peek.refs_out


def test_params_refuse_unknown_keys_and_read_null_as_the_default() -> None:
    params = Probe.specs()[0].params
    full = {"rationale": RATIONALE, "obj": "obj_1", "grip": {"part": "handle", "depth": None}}
    parsed = params.model_validate({**full, "arm": None, "frame": None})
    assert (parsed.arm, parsed.frame, parsed.grip.depth) == ("left", None, 0.02)
    assert params.model_validate(full) == parsed  # left out and null alike
    for bad in (
        {**full, "extra": 1},
        {**full, "grip": {"part": "handle", "extra": 1}},
        {**full, "rationale": None},  # required: null is not a default
        {**full, "grip": {"part": "handle", "depth": 0.5}},  # le=0.1
        {**full, "rationale": {k: v for k, v in RATIONALE.items() if k != "stage"}},
        {**full, "rationale": {**RATIONALE, "stage": "x" * 41}},  # a stage name is short
    ):
        with pytest.raises(ValidationError):
            params.model_validate(bad)


class Plain(BaseModel):
    x: int = 0


class Recursive(ToolModel):
    children: list["Recursive"] = Field(default_factory=list)


@dataclass
class Point:
    x: float


class Pair(TypedDict):
    x: float


class Coords(NamedTuple):
    x: float


def _define(signature: str, *, kind: str = "pure", options: str = "", doc: str = '"""Doc."""'):
    namespace: dict[str, Any] = {
        "Toolset": Toolset,
        "tool": tool,
        "ToolContext": ToolContext,
        "Moved": Moved,
        "Plain": Plain,
        "Recursive": Recursive,
        "Rationale": Rationale,
        "Any": Any,
        "Point": Point,
        "Pair": Pair,
        "Coords": Coords,
        "ObjectRef": ObjectRef,
        "PartRef": PartRef,
    }
    source = (
        "class Bad(Toolset):\n"
        "    name = 'bad'\n"
        f"    @tool(kind={kind!r}, tier='D'{options})\n"
        f"    async def act({signature}) -> Moved:\n"
        f"        {doc}\n"
        "        return Moved(obj='o', depth=0)\n"
    )
    exec(source, namespace)


def test_a_tool_declares_its_tier() -> None:
    """No tool is tier D by default: a decorator without a tier is refused (fail closed)."""
    with pytest.raises(TypeError, match="tier"):
        tool(kind="pure")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("signature", "extra", "message"),
    [
        ("self, ctx", {"doc": "pass"}, "write a docstring"),
        ("self, ctx", {"options": ", name='Bad-Name'"}, "a tool name matches"),
        ("ctx, self", {}, r"starts with \(self, ctx\)"),
        ("self, ctx, x", {}, "'x' must be named and annotated"),
        ("self, ctx, *xs: int", {}, "'xs' must be named and annotated"),
        ("self, ctx, table: dict[str, float]", {}, "mappings with free keys"),
        ("self, ctx, anything: Any", {}, "needs exactly one JSON type"),
        ("self, ctx, pair: tuple[int, str]", {}, "tuples of mixed item types"),
        ("self, ctx, plain: Plain", {}, r"\['Plain'\] must derive from ToolModel"),
        ("self, ctx, point: Point", {}, r"\['Point'\] must derive from ToolModel"),
        ("self, ctx, pair: list[Pair]", {}, r"\['Pair'\] must derive from ToolModel"),
        ("self, ctx, xy: Coords | None = None", {}, r"\['Coords'\] must derive from ToolModel"),
        ("self, ctx, target: ObjectRef | PartRef", {}, "mixes reference kinds"),
        ("self, ctx, tree: Recursive", {}, "cannot be recursive"),
        ("self, ctx, n: int | None = 3", {}, "null would mean both None and the default"),
        ("self, ctx", {"kind": "execution"}, "takes a `rationale: Rationale` parameter"),
        ("self, ctx, rationale: Rationale = None", {"kind": "execution"}, "takes no default"),
    ],
)
def test_bad_definitions_fail_when_the_class_is_created(
    signature: str, extra: dict[str, str], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        _define(signature, **extra)


def test_a_spec_argument_says_which_kind_of_parameter_it_takes() -> None:
    signatures = tool_signatures(Manipulation.specs())
    assert signatures["grasp_object"].takes == {"grasp": "grasp_spec"}
    assert signatures["place_object"].takes == {"place": "place_spec"}
    assert signatures["move_to_pose"].takes == {}
    line = signatures["grasp_object"].line("grasp_object")
    assert line.startswith("- grasp_object(arm, obj, grasp:grasp_spec, [approach_m]")
