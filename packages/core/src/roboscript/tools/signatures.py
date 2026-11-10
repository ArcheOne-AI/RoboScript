"""What a Memory's call templates may call: the mounted tools' parameter names, their required
ones and a line of purpose, for the Memory linter (`roboscript.memory.LintConfig.tools`) and the
roles that write templates (a method's prompts, such as wicl's `Prompts`). Generated from the
tool specs; the memory package sits below the tools and cannot import them."""

import re
from collections.abc import Iterable
from typing import Any, get_args

from pydantic.fields import FieldInfo

from roboscript.manipulation.specs import GraspSpec, HingeSpec, PathSpec, PlaceSpec
from roboscript.memory.calls import ToolSig
from roboscript.tools.spec import MemoryKind, ToolSpec

_PURPOSE_CHARS = 140
_SPECS = {
    GraspSpec: "grasp_spec",
    PlaceSpec: "place_spec",
    HingeSpec: "hinge_spec",
    PathSpec: "path_spec",
}
_SENTENCE = re.compile(r"(?<=[.;])\s")


def purpose(spec: ToolSpec) -> str:
    """The first sentence of a tool's description, cut to a line."""
    first = _SENTENCE.split(spec.description, maxsplit=1)[0].rstrip(".;")
    return first if len(first) <= _PURPOSE_CHARS else first[: _PURPOSE_CHARS - 3] + "..."


def _spec_kind(annotation: Any) -> str | None:
    """The kind of Memory parameter a field's type is (a spec), looking through Annotated, unions
    and None."""
    if annotation in _SPECS:
        return _SPECS[annotation]
    return next((k for a in get_args(annotation) if (k := _spec_kind(a))), None)


def _marked_kind(field: FieldInfo) -> str | None:
    """The kind of Memory parameter a field is marked as taking by name (`MemoryKind`)."""
    return next((m.kind for m in field.metadata if isinstance(m, MemoryKind)), None)


def tool_signatures(specs: Iterable[ToolSpec]) -> dict[str, ToolSig]:
    """Tool name -> its parameters (`rationale` is the harness's, not a template's)."""
    found: dict[str, ToolSig] = {}
    for spec in specs:
        fields = {n: f for n, f in spec.params.model_fields.items() if n != "rationale"}
        found[spec.name] = ToolSig(
            params=tuple(fields),
            required=tuple(n for n, f in fields.items() if f.is_required()),
            purpose=purpose(spec),
            takes={
                n: k
                for n, f in fields.items()
                if (k := _spec_kind(f.annotation) or _marked_kind(f))
            },
        )
    return found
