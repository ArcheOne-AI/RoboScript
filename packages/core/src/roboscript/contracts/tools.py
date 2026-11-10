"""The tool plane: descriptors, calls, results and the context toolsets receive, and
`ToolModel`, the base of every model in tool parameters (here, below the libraries that define
such models, e.g. Memory's edit operations)."""

import types
from typing import Annotated, Any, Final, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.ids import EpisodeId, PhaseName, RoleName, ToolName, WireSecret
from roboscript.contracts.trace import TraceContext

Tier = Literal["D", "P"]
ToolKind = Literal["pure", "perception", "execution", "memory", "control"]

TURNS: Final[frozenset[ToolKind]] = frozenset({"execution", "control"})
"""Tool kinds that end a turn: physical actions and `finish` (the Hub counts them as turns)."""


class ToolModel(BaseModel):
    """Base of every model in tool parameters: unknown keys are refused, and null for a field
    with a default means that default (strict schemas make every field required). Fields keep
    their names on the wire: aliases are refused."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for name, info in cls.model_fields.items():
            if info.alias or info.validation_alias or info.serialization_alias:
                raise TypeError(f"{cls.__name__}.{name}: tool parameters take no field aliases")
            if (
                not info.is_required()
                and info.default is not None
                and _admits_none(info.annotation)
            ):
                raise TypeError(
                    f"{cls.__name__}.{name}: with a default other than None, null would mean "
                    "both None and the default; drop None from the type or default to None"
                )

    @model_validator(mode="before")
    @classmethod
    def _null_means_default(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fields = cls.model_fields
        return {
            key: value
            for key, value in data.items()
            if value is not None or key not in fields or fields[key].is_required()
        }


def _admits_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _admits_none(get_args(annotation)[0])
    if origin is Union or origin is types.UnionType:
        return any(_admits_none(arg) for arg in get_args(annotation))
    return annotation is None or annotation is type(None) or annotation is Any


class ToolMeta(BaseModel):
    """How the Hub treats a tool, as its toolset declares it (`roboscript.tools.tool`); which
    phases and roles see it is the method's to declare (`contracts.phase.PhaseSpec`). A
    `system_only` tool is the runtime's alone: no model sees or calls it; one also `after_end`
    the runtime may still call once the episode has ended, until it closes (a reading of the
    scene the episode ended in)."""

    model_config = ConfigDict(frozen=True)

    toolset: str
    kind: ToolKind
    tier: Tier
    system_only: bool = False
    after_end: bool = False
    uses_steps: bool
    needs_rationale: bool
    timeout_s: PositiveFloat


class ToolDescriptor(BaseModel):
    """A tool as exported to clients: JSON schemas plus the hash permits refer to."""

    model_config = ConfigDict(frozen=True)

    name: Annotated[ToolName, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    meta: ToolMeta
    hash: str
    """sha256 hex of `canonical_json({"name": ..., "description": ..., "input_schema": ...})`."""


class TextPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["image"] = "image"
    blob: BlobRef
    label: str | None = None


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class ToolResult(BaseModel):
    """Model-facing content plus optional structured data; `is_error` results carry a code."""

    model_config = ConfigDict(frozen=True)

    content: list[ContentPart]
    structured: dict[str, Any] | None = None
    is_error: bool = False
    error_code: str | None = None


class ToolCall(BaseModel):
    """One call as the Hub received it; `origin` says who issued it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    name: ToolName
    arguments: dict[str, Any]
    origin: Literal["model", "system", "service"]


class EnvBinding(BaseModel):
    """How a toolset reaches the episode's env: agent-plane URL, env episode id and token."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    episode_id: EpisodeId
    token: WireSecret


class CallContext(BaseModel):
    """Episode context the Hub attaches to every toolset call (MCP `_meta`); `stage` (the
    method's phase) and `role` are the episode's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_ref: EpisodeId
    stage: PhaseName
    tier: Tier
    role: RoleName | None = None
    env: EnvBinding | None = None
    embodiment: str | None = None
    trace: TraceContext | None = None
