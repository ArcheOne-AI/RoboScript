"""Typed references: opaque ids inside the system, aliases such as `obj_3` for models.

A reference field is a `str` annotated with `Ref(kind)`. Validating a model with a translator in
the context under `REFS` maps every reference through it, however deeply it is nested: the Hub
passes alias -> id for arguments and id -> alias for results. Without one, values pass unchanged.

A `RoleRef` names an object by its role (as the Memory's specs do); it is not an id, so results
keep it as it is, and in arguments the Hub also takes an object's alias for it (its role).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Final

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, ValidationInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

REFS: Final = "rs_refs"
"""Validation-context key of a `(kind, value) -> value` translator."""

REF_KEY: Final = "x-rs-ref"
"""JSON Schema key marking a reference; the conservative export turns it into a hint."""

Translate = Callable[[str, str], str]


@dataclass(frozen=True)
class Ref:
    """Marks a `str` as a reference of `kind`; models see it as `<kind>_<n>`."""

    kind: str

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.with_info_after_validator_function(self._translate, handler(source))

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        json_schema[REF_KEY] = self.kind
        return json_schema

    def _translate(self, value: str, info: ValidationInfo) -> str:
        translate: Translate | None = (info.context or {}).get(REFS)
        return value if translate is None else translate(self.kind, value)


FrameRef = Annotated[str, Ref("frame")]
ObjectRef = Annotated[str, Ref("obj")]
RegionRef = Annotated[str, Ref("region")]
CloudRef = Annotated[str, Ref("cloud")]
PartRef = Annotated[str, Ref("part")]
ROLE: Final = "role"
"""The kind of a `RoleRef`."""
RoleRef = Annotated[str, Ref(ROLE)]
