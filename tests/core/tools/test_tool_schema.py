"""Schema export: the conservative strict dialect and descriptor hashes."""

import hashlib
from enum import Enum
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.geometry import Vec3
from roboscript.tools import (
    ObjectRef,
    PartRef,
    Rationale,
    ToolContext,
    ToolModel,
    Toolset,
    check_strict,
    descriptor,
    tool,
)

# What an OpenAI strict function schema may contain here (anything else goes to descriptions).
ALLOWED = {"type", "properties", "required", "additionalProperties", "items", "enum", "anyOf"}


class Axis(str, Enum):
    x = "x"
    z = "z"


class Offset(ToolModel):
    """Where to go, relative to the anchor."""

    shift: Vec3 = (0.0, 0.0, 0.0)
    axis: Axis = Axis.z


class Spec(ToolModel):
    """A grasp specification."""

    part: PartRef
    approach: Annotated[str, Field(max_length=40, description="Approach direction.")] = "top"
    offsets: list[Offset] = Field(default_factory=list)
    width: Annotated[float, Field(gt=0, le=0.1, description="Opening in meters.")] | None = None


class Result(BaseModel):
    obj: ObjectRef


class Rich(Toolset):
    name = "rich"

    @tool(kind="execution", tier="D")
    async def grasp(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        obj: ObjectRef,
        spec: Annotated[Spec, Field(description="How to grasp it.")],
        arm: Literal["left", "right"],
        speed: Annotated[int, Field(ge=1, le=5)] = 3,
        mode: Literal["fast"] = "fast",
        notes: list[str] | None = None,
        axis: Axis = Axis.x,
    ) -> Result:
        """Grasp an object with one arm."""
        return Result(obj=obj)


GRASP = Rich.specs()[0]


def walk(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    """Every schema object in `node` with its path."""
    found = [(path, node)]
    for key, value in node.get("properties", {}).items():
        found += walk(value, f"{path}.{key}")
    for i, branch in enumerate(node.get("anyOf", [])):
        found += walk(branch, f"{path}|{i}")
    if "items" in node:
        found += walk(node["items"], f"{path}[]")
    return found


def test_parameters_follow_openai_strict_rules() -> None:
    schema = GRASP.input_schema
    assert schema["type"] == "object"
    for path, node in walk(schema):
        assert set(node) <= ALLOWED | {"description"}, path
        assert "$ref" not in str(node) and "$defs" not in str(node), path
        if node.get("type") == "object":
            assert node["additionalProperties"] is False, path
            assert node["required"] == list(node["properties"]), path
        if "anyOf" in node:
            assert set(node) <= {"anyOf", "description"}, path
        if node.get("type") == "array":
            assert "items" in node, path
    check_strict(schema)


def test_descriptor_hash_covers_what_models_see() -> None:
    d = GRASP.descriptor
    identity = {"name": d.name, "description": d.description, "input_schema": d.input_schema}
    assert d.hash == hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    assert d.output_schema == Result.model_json_schema(mode="serialization")
    other_meta = GRASP.meta.model_copy(update={"stages": ["s1"], "tier": "P"})
    assert descriptor(GRASP, other_meta).hash == d.hash
    assert descriptor(GRASP, other_meta).meta.stages == ["s1"]


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"anyOf": [{"type": "object"}]}, "the root must be an object"),
        (
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            "additionalProperties to false",
        ),
        (
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            "every property must be required",
        ),
        (
            {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            r"\['\$ref'\]",
        ),
        (
            {
                "type": "object",
                "properties": {"a": {"type": "number", "minimum": 0}},
                "required": ["a"],
                "additionalProperties": False,
            },
            r"\['minimum'\]",
        ),
        (
            {
                "type": "object",
                "properties": {"a": {"anyOf": [{"type": "string"}], "type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            "no siblings",
        ),
        (
            {
                "type": "object",
                "properties": {"a": {"type": "array"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            "arrays need items",
        ),
    ],
)
def test_check_strict_refuses(schema: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_strict(schema)
