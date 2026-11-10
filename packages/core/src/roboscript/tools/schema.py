"""Tool schemas in the most conservative JSON Schema dialect, and the tool formats of clients.

Parameters export as OpenAI strict-mode function schemas that other providers accept too: `$ref`
inlined, every object closed with all properties required, optional ones nullable (null means
the default), and only `type`, `properties`, `required`, `additionalProperties`, `items`, `enum`,
`anyOf` and `description`. Ranges, lengths, patterns, defaults and alias hints move into the
descriptions. Anything the dialect cannot say faithfully is a TypeError at definition time.
"""

import copy
import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.ids import ToolName
from roboscript.contracts.tools import ToolDescriptor, ToolMeta
from roboscript.tools.refs import REF_KEY, ROLE

if TYPE_CHECKING:
    from roboscript.tools.spec import ToolSpec

STRICT_KEYWORDS: Final = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "anyOf",
        "description",
    }
)
MAX_DEPTH: Final = 10  # OpenAI strict mode allows 10 levels of nesting
_TYPES: Final = frozenset({"string", "number", "integer", "boolean", "null", "object", "array"})
_NULL: Final = {"type": "null"}
_IGNORED: Final = ("title", "examples", "readOnly", "writeOnly", "deprecated", "discriminator")
_BOUNDS: Final = (
    ("minimum", ">="),
    ("exclusiveMinimum", ">"),
    ("maximum", "<="),
    ("exclusiveMaximum", "<"),
)
_MISSING: Final = object()


def strict_schema(model: type[BaseModel], terse: Sequence[str] = ()) -> dict[str, Any]:
    """The parameters `model` validates, as a strict-mode schema (the parameters `terse` names:
    without the descriptions nested inside them)."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    schema = _convert(raw, defs, "$", ())
    schema.pop("description", None)  # the tool's description covers the whole object
    for name in terse:
        prop = schema["properties"][name]
        schema["properties"][name] = {**_undescribed(prop), **_own(prop)}
    check_strict(schema)
    return schema


def _own(node: dict[str, Any]) -> dict[str, Any]:
    return {"description": node["description"]} if "description" in node else {}


def _undescribed(node: Any) -> Any:
    """`node` without any description in it."""
    if isinstance(node, dict):
        return {
            k: _undescribed(v)
            for k, v in node.items()
            if not (k == "description" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_undescribed(v) for v in node]
    return node


def has_refs(schema: Any) -> bool:
    """Whether a pydantic JSON schema contains a reference field anywhere."""
    if isinstance(schema, dict):
        return REF_KEY in schema or any(has_refs(value) for value in schema.values())
    if isinstance(schema, list):
        return any(has_refs(value) for value in schema)
    return False


def check_strict(schema: dict[str, Any]) -> None:
    """Raise ValueError, naming the path, unless `schema` fits the strict-mode subset."""
    if schema.get("type") != "object":
        raise ValueError("$: the root must be an object")
    _check(schema, "$", 1)


def descriptor(spec: "ToolSpec", meta: ToolMeta | None = None) -> ToolDescriptor:
    """`spec` as clients see it, with `meta` in place of the spec's own if given.

    The hash covers name, description and input schema: what a model is shown.
    """
    identity = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }
    return ToolDescriptor(
        name=ToolName(spec.name),
        description=spec.description,
        input_schema=spec.input_schema,
        output_schema=spec.output_schema,
        meta=spec.meta if meta is None else meta,
        hash=hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest(),
    )


def to_responses_tool(tool: "ToolSpec | ToolDescriptor") -> dict[str, Any]:
    """A strict function tool for the Responses API."""
    d = _descriptor(tool)
    parameters = copy.deepcopy(d.input_schema)
    return {
        "type": "function",
        "name": d.name,
        "description": d.description,
        "parameters": parameters,
        "strict": True,
    }


def to_chat_tool(tool: "ToolSpec | ToolDescriptor") -> dict[str, Any]:
    """A strict function tool for Chat Completions."""
    d = _descriptor(tool)
    function = {
        "name": d.name,
        "description": d.description,
        "parameters": copy.deepcopy(d.input_schema),
        "strict": True,
    }
    return {"type": "function", "function": function}


def to_mcp_tool(tool: "ToolSpec | ToolDescriptor") -> dict[str, Any]:
    """An MCP tool listing entry."""
    d = _descriptor(tool)
    return {
        "name": d.name,
        "description": d.description,
        "inputSchema": copy.deepcopy(d.input_schema),
    }


def _descriptor(tool: "ToolSpec | ToolDescriptor") -> ToolDescriptor:
    return tool if isinstance(tool, ToolDescriptor) else tool.descriptor


def _convert(
    node: dict[str, Any], defs: dict[str, Any], path: str, seen: tuple[str, ...]
) -> dict[str, Any]:
    node = dict(node)
    if "$ref" in node:
        name = node.pop("$ref").rsplit("/", 1)[-1]
        if name in seen:
            raise TypeError(f"{path}: {name} contains itself; tool parameters cannot be recursive")
        # Siblings such as a field's description override the referenced model's.
        return _convert({**defs[name], **node}, defs, path, (*seen, name))
    if "allOf" in node:
        parts = node.pop("allOf")
        if len(parts) != 1:
            raise TypeError(f"{path}: allOf with several schemas is not supported")
        return _convert({**parts[0], **node}, defs, path, seen)
    description = node.pop("description", None)
    node.pop("default", None)  # the enclosing object puts it into the description
    notes = _notes(node)
    for key in _IGNORED:
        node.pop(key, None)
    if "const" in node:
        node.setdefault("enum", [node["const"]])
        del node["const"]
    if "oneOf" in node:
        node["anyOf"] = node.pop("oneOf")
    if "anyOf" in node:
        branches: list[dict[str, Any]] = []
        for i, branch in enumerate(node.pop("anyOf")):
            out = _convert(branch, defs, f"{path}.anyOf[{i}]", seen)
            branches += out["anyOf"] if list(out) == ["anyOf"] else [out]
        schema: dict[str, Any] = {"anyOf": branches}
    else:
        schema = _typed(node, defs, path, seen, notes)
    if node:
        raise TypeError(f"{path}: JSON Schema keywords {sorted(node)} are not supported")
    if text := _describe(description, notes):
        schema["description"] = text
    return schema


def _typed(
    node: dict[str, Any], defs: dict[str, Any], path: str, seen: tuple[str, ...], notes: list[str]
) -> dict[str, Any]:
    kind = node.pop("type", None)
    if kind not in _TYPES:
        raise TypeError(f"{path}: each value needs exactly one JSON type, got {kind!r}")
    schema: dict[str, Any] = {"type": kind}
    if "enum" in node:
        schema["enum"] = node.pop("enum")
    if kind == "object":
        schema |= _object(node, defs, path, seen)
    elif kind == "array":
        schema |= _array(node, defs, path, seen, notes)
    return schema


def _object(
    node: dict[str, Any], defs: dict[str, Any], path: str, seen: tuple[str, ...]
) -> dict[str, Any]:
    if node.pop("additionalProperties", False) is not False:
        raise TypeError(f"{path}: mappings with free keys are not supported; use a model")
    properties = node.pop("properties", None)
    if properties is None:
        raise TypeError(f"{path}: objects need declared properties; use a model")
    required = set(node.pop("required", ()))
    out = {}
    for name, prop in properties.items():
        schema = _convert(prop, defs, f"{path}.{name}", seen)
        out[name] = schema if name in required else _nullable(schema, prop.get("default", _MISSING))
    return {"properties": out, "required": list(out), "additionalProperties": False}


def _nullable(schema: dict[str, Any], default: Any) -> dict[str, Any]:
    schema = dict(schema)
    description = schema.pop("description", None)
    branches = schema["anyOf"] if list(schema) == ["anyOf"] else [schema]
    out: dict[str, Any] = {"anyOf": branches if _NULL in branches else [*branches, _NULL]}
    note = (
        [] if default is _MISSING or default is None else [f"Default when null: {_json(default)}"]
    )
    if text := _describe(description, note):
        out["description"] = text
    return out


def _array(
    node: dict[str, Any], defs: dict[str, Any], path: str, seen: tuple[str, ...], notes: list[str]
) -> dict[str, Any]:
    low, high = node.pop("minItems", None), node.pop("maxItems", None)
    if "prefixItems" in node:
        items = [
            _convert(v, defs, f"{path}[{i}]", seen) for i, v in enumerate(node.pop("prefixItems"))
        ]
        if not items or "items" in node or any(item != items[0] for item in items):
            raise TypeError(f"{path}: tuples of mixed item types are not supported; use a model")
        item = items[0]
    elif "items" in node:
        item = _convert(node.pop("items"), defs, f"{path}[]", seen)
    else:
        raise TypeError(f"{path}: arrays need an item type")
    if low is not None and low == high:
        notes.append(f"Exactly {low} items")
    else:
        if low:
            notes.append(f"At least {low} items")
        if high is not None:
            notes.append(f"At most {high} items")
    if node.pop("uniqueItems", False):
        notes.append("Items must be unique")
    return {"items": item}


def _notes(node: dict[str, Any]) -> list[str]:
    notes = []
    bounds = [f"{op} {_json(node.pop(key))}" for key, op in _BOUNDS if key in node]
    if bounds:
        notes.append("Must be " + " and ".join(bounds))
    if "multipleOf" in node:
        notes.append(f"A multiple of {_json(node.pop('multipleOf'))}")
    if "minLength" in node:
        notes.append(f"At least {node.pop('minLength')} characters")
    if "maxLength" in node:
        notes.append(f"At most {node.pop('maxLength')} characters")
    if "pattern" in node:
        notes.append(f"Must match the pattern {node.pop('pattern')}")
    if "format" in node:
        notes.append(f"Format: {node.pop('format')}")
    if REF_KEY in node and (kind := node.pop(REF_KEY)) != ROLE:  # a role's field says so
        notes.append(f"An alias such as {kind}_1")
    return notes


def _describe(description: str | None, notes: list[str]) -> str:
    parts = [description.strip()] if description else []
    return " ".join(p if p.endswith((".", "!", "?")) else f"{p}." for p in [*parts, *notes] if p)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _check(node: Any, path: str, depth: int) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"{path}: a schema must be an object")
    if unknown := set(node) - STRICT_KEYWORDS:
        raise ValueError(f"{path}: keywords outside the strict subset: {sorted(unknown)}")
    if not isinstance(node.get("description", ""), str):
        raise ValueError(f"{path}: a description must be a string")
    if "anyOf" in node:
        if set(node) - {"anyOf", "description"} or not node["anyOf"]:
            raise ValueError(f"{path}: anyOf needs branches and takes no siblings but description")
        for i, branch in enumerate(node["anyOf"]):
            _check(branch, f"{path}.anyOf[{i}]", depth)
        return
    kind = node.get("type")
    if kind not in _TYPES:
        raise ValueError(f"{path}: type must be one of {sorted(_TYPES)}, got {kind!r}")
    if "enum" in node and not (isinstance(node["enum"], list) and node["enum"]):
        raise ValueError(f"{path}: enum must be a non-empty list")
    if kind == "object":
        if depth > MAX_DEPTH:
            raise ValueError(f"{path}: nested deeper than {MAX_DEPTH} levels")
        properties = node.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"{path}: objects must list their properties")
        if node.get("additionalProperties") is not False:
            raise ValueError(f"{path}: objects must set additionalProperties to false")
        required = node.get("required")
        if not isinstance(required, list) or sorted(required) != sorted(properties):
            raise ValueError(f"{path}: every property must be required")
        for name, prop in properties.items():
            _check(prop, f"{path}.{name}", depth + 1)
    elif {"properties", "required", "additionalProperties"} & set(node):
        raise ValueError(f"{path}: only objects have properties")
    if kind == "array":
        if "items" not in node:
            raise ValueError(f"{path}: arrays need items")
        _check(node["items"], f"{path}[]", depth + 1)
    elif "items" in node:
        raise ValueError(f"{path}: only arrays have items")
