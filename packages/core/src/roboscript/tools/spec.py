"""Tools: async methods of a `Toolset`, declared once with `@tool`
(docs/concepts/agent-runtime.md).

```python
class Geometry(Toolset):
    name = "geometry"

    @tool(kind="pure", tier="D")
    async def table(self, ctx: ToolContext, margin: Annotated[float, Field(ge=0)] = 0.0) -> Table:
        \"\"\"Model-facing description: the docstring's first paragraph.\"\"\"
```

The parameters after `ctx` form the params model (a `ToolModel`), the return annotation is the
result model. Everything is built and checked when the class is created. The toolset declares a
tool's kind, tier and whether only the runtime calls it; which phases and roles see it is the
method's declaration (`contracts.phase.PhaseSpec`).
"""

import dataclasses
import inspect
import re
import types
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any, ClassVar, Final, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model
from typing_extensions import is_typeddict

from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.skills import BindingName, Bindings
from roboscript.contracts.tools import (
    ContentPart,
    Tier,
    ToolDescriptor,
    ToolKind,
    ToolMeta,
    ToolModel,
)
from roboscript.kernel import NoConfig
from roboscript.tools.context import ToolsetRuntime
from roboscript.tools.refs import Ref
from roboscript.tools.schema import descriptor, has_refs, strict_schema

TOOL_NAME: Final = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_TOOLSET_NAME: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_MARK: Final = "__rs_tool__"

Render = Callable[[Any], list[ContentPart]]
"""Turns a result (ids already aliased) into model content; the Hub still resizes images."""


class Rationale(ToolModel):
    """Why this physical action."""

    stage: Annotated[
        str,
        Field(
            max_length=40,
            description="The workflow stage this action belongs to, named as in the Memory.",
        ),
    ]
    scene: Annotated[str, Field(description="What you see that matters for this action.")]
    progress: Annotated[
        str,
        Field(description="Where you are in the workflow and whether its conditions hold."),
    ]
    plan: Annotated[str, Field(description="Why this action now and what it should achieve.")]


@dataclass(frozen=True)
class _Options:
    kind: ToolKind
    tier: Tier
    name: str | None
    timeout_s: float
    uses_steps: bool | None
    needs_rationale: bool | None
    render: Render | None
    terse: tuple[str, ...] = ()
    needs: tuple[BindingName, ...] = ()
    system_only: bool = False
    after_end: bool = False


def tool(
    kind: ToolKind,
    *,
    tier: Tier,
    name: str | None = None,
    timeout_s: float = 60.0,
    uses_steps: bool | None = None,
    needs_rationale: bool | None = None,
    render: Render | None = None,
    terse: tuple[str, ...] = (),
    needs: tuple[BindingName, ...] = (),
    system_only: bool = False,
    after_end: bool = False,
) -> Callable[[Callable[..., Awaitable[BaseModel]]], Callable[..., Awaitable[BaseModel]]]:
    """Declare a toolset method `async (self, ctx, <params>) -> <result model>` as a tool.

    `tier` (required: no tool is D by default) is the tool's tier: a tier-D episode never sees
    or calls a tier-P tool, and no phase of kind `eval` lists one. A
    `system_only` tool is the runtime's alone (no model sees or calls it: the one-time
    identification as an episode opens); `after_end` lets the runtime call it after the
    episode ended, until it closes (a reading of the scene the episode ended in).
    `uses_steps` and `needs_rationale` default to `kind == "execution"`; a tool needing a
    rationale takes a required `rationale: Rationale` parameter. `name` defaults to the method's.
    `terse` names parameters exported with their own description only, none nested in them: for
    parameters that repeat other tools' (the description says which), so a model reads them once.
    `needs` names the bindings the tool's implementation asks of the env (`contracts.skills`):
    without them it is not mounted (`tools.skillsets`).
    """
    if after_end and not system_only:
        raise TypeError("after_end is for a system_only tool: no model calls after the end")
    options = _Options(
        kind,
        tier,
        name,
        timeout_s,
        uses_steps,
        needs_rationale,
        render,
        terse,
        needs,
        system_only,
        after_end,
    )

    def mark(fn: Callable[..., Awaitable[BaseModel]]) -> Callable[..., Awaitable[BaseModel]]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"{fn.__qualname__}: a tool is an async method")
        setattr(fn, _MARK, options)
        return fn

    return mark


@dataclass(frozen=True)
class MemoryKind:
    """A tool parameter's mark (`Annotated[str, MemoryKind("skill_spec"), ...]`): it takes a
    Memory parameter of this kind by name, as a spec-typed parameter takes a spec. The Memory
    linter checks a call template's `{name}` there against it (`tools.signatures`)."""

    kind: str


@dataclass(frozen=True)
class ToolSpec:
    """A tool, defined once: its model-facing name and description, params and result models,
    how the Hub treats it (`meta`), the handler (unbound), an optional renderer, the exported
    schemas and whether its params or result hold references."""

    name: str
    description: str
    params: type[ToolModel]
    result: type[BaseModel]
    meta: ToolMeta
    handler: Callable[..., Awaitable[BaseModel]]
    render: Render | None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    refs_in: bool
    refs_out: bool
    needs: tuple[BindingName, ...] = ()

    @cached_property
    def descriptor(self) -> ToolDescriptor:
        return descriptor(self)


class SharedConfig(BaseModel):
    """The base of a config that several toolsets work on as one: a Hub mounts toolsets whose
    configs are the same such class only with equal values (the scene toolsets, whose region ids
    must resolve alike in each). A Hub sees the configs of toolsets in its process only: the
    servers of remote ones must be started from the same config file."""


class Toolset:
    """Tools that share resources. Subclass it, set `name` and declare tools with `@tool`; a
    subclass keeps the tools it inherits unless it redefines the attribute. `config` (a
    `Config`) and `runtime` are set before `start` is awaited."""

    name: ClassVar[str]
    Config: ClassVar[type[BaseModel]] = NoConfig
    config: BaseModel
    runtime: ToolsetRuntime
    bindings: Bindings = Bindings()
    """What the env's skillsets bind for the core entries this toolset has (set by its provider
    before `start`, `roboscript.tools.skillsets`)."""
    _specs: ClassVar[tuple[ToolSpec, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._specs = _collect(cls, "name")

    @classmethod
    def specs(cls) -> list[ToolSpec]:
        return list(cls._specs)

    async def start(self, runtime: ToolsetRuntime) -> None:
        """Acquire resources; called once before any call."""

    async def stop(self) -> None:
        """Release them."""

    async def close_episode(self, episode_id: EpisodeId) -> None:
        """Free what the toolset keeps for a closed episode beyond `ctx.state`, if anything."""


class ToolsetExtension:
    """Tools a skillset adds to a core toolset, `extends` (docs/how-to/add-a-skillset.md): they run
    in that toolset's provider as its own tools do, with its config, its per-episode `ctx.state`
    and its event prefix, on `host`, the toolset instance (whose handles they build on). Declare
    tools with `@tool` as on a `Toolset`; their toolset is `extends`."""

    extends: ClassVar[str]
    _specs: ClassVar[tuple[ToolSpec, ...]] = ()

    def __init__(self, host: Toolset) -> None:
        if host.name != self.extends:
            raise TypeError(f"{type(self).__qualname__} extends {self.extends}, not {host.name}")
        self.host = host

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._specs = _collect(cls, "extends")

    @classmethod
    def specs(cls) -> list[ToolSpec]:
        return list(cls._specs)


def _collect(cls: type, name_attr: str) -> tuple[ToolSpec, ...]:
    """The specs of the `@tool` methods of `cls` and its bases, as toolset `cls.<name_attr>`."""
    marked: dict[str, tuple[Any, _Options]] = {}
    for klass in reversed(cls.__mro__):
        for attr, value in vars(klass).items():
            if hasattr(value, _MARK):
                marked[attr] = (value, getattr(value, _MARK))
            else:
                marked.pop(attr, None)
    if not marked:
        return ()
    name = getattr(cls, name_attr, None)
    if not isinstance(name, str) or not _TOOLSET_NAME.fullmatch(name):
        raise TypeError(f"{cls.__qualname__}: set `{name_attr}` to a lowercase identifier")
    specs = tuple(_build(fn, name, options) for fn, options in marked.values())
    names = [spec.name for spec in specs]
    if duplicates := sorted({n for n in names if names.count(n) > 1}):
        raise TypeError(f"{cls.__qualname__}: tool names used twice: {duplicates}")
    return specs


def _build(fn: Callable[..., Awaitable[BaseModel]], toolset: str, options: _Options) -> ToolSpec:
    name = options.name or fn.__name__
    where = f"{toolset}.{name}"
    if not TOOL_NAME.fullmatch(name):
        raise TypeError(f"{where}: a tool name matches {TOOL_NAME.pattern}")
    description = " ".join(inspect.cleandoc(fn.__doc__ or "").split("\n\n")[0].split())
    if not description:
        raise TypeError(f"{where}: write a docstring; its first paragraph is what models read")
    parameters = list(inspect.signature(fn).parameters.values())
    if [p.name for p in parameters[:2]] != ["self", "ctx"]:
        raise TypeError(f"{where}: a tool's signature starts with (self, ctx)")
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for p in parameters[2:]:
        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) or p.name not in hints:
            raise TypeError(f"{where}: parameter {p.name!r} must be named and annotated")
        fields[p.name] = (hints[p.name], ... if p.default is p.empty else p.default)
    result = hints.get("return")
    if not (inspect.isclass(result) and issubclass(result, BaseModel)):
        raise TypeError(f"{where}: annotate the return type with a pydantic model")
    kind = options.kind
    needs_rationale = (
        kind == "execution" if options.needs_rationale is None else options.needs_rationale
    )
    rationale = fields.get("rationale")
    if rationale is not None and rationale[1] is not ...:  # before the type: 3.10 reads a None
        raise TypeError(f"{where}: `rationale` takes no default")  # default as Optional
    if needs_rationale and (rationale is None or not _is_rationale(rationale[0])):
        raise TypeError(f"{where}: a {kind} tool takes a `rationale: Rationale` parameter")
    try:
        params = create_model(
            _model_name(where), __base__=ToolModel, __module__=fn.__module__, **fields
        )
        _check_types(params, result)
        input_schema = strict_schema(params, terse=options.terse)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{where}: {error}") from None
    output_schema = result.model_json_schema(mode="serialization")
    meta = ToolMeta(
        toolset=toolset,
        kind=kind,
        tier=options.tier,
        system_only=options.system_only,
        after_end=options.after_end,
        uses_steps=kind == "execution" if options.uses_steps is None else options.uses_steps,
        needs_rationale=needs_rationale,
        timeout_s=options.timeout_s,
    )
    return ToolSpec(
        name=name,
        description=description,
        params=params,
        result=result,
        meta=meta,
        handler=fn,
        render=options.render,
        input_schema=input_schema,
        output_schema=output_schema,
        refs_in=has_refs(params.model_json_schema()),
        refs_out=has_refs(output_schema),
        needs=options.needs,
    )


def _check_types(params: type[BaseModel], result: type[BaseModel]) -> None:
    """Parameter types a strict schema says faithfully, and references a translator can map."""
    if foreign := sorted(t.__name__ for t in _structured(params) if not issubclass(t, ToolModel)):
        raise TypeError(f"structured parameter types {foreign} must derive from ToolModel")
    for model in (params, *_structured(params), result, *_structured(result)):
        if not issubclass(model, BaseModel):
            continue
        for field, info in model.model_fields.items():
            for union in _unions(info.annotation):
                if len(kinds := _ref_kinds(union)) > 1:
                    raise TypeError(
                        f"{model.__name__}.{field} mixes reference kinds {sorted(kinds)}; "
                        "keep one kind per field"
                    )


def _model_name(where: str) -> str:
    return "".join(part.title() for part in re.split(r"[._]", where)) + "Params"


def _is_rationale(annotation: Any) -> bool:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return inspect.isclass(annotation) and issubclass(annotation, Rationale)


def _structured(model: type[BaseModel]) -> Iterator[type]:
    """The classes in `model`'s field types, at any depth, that travel as JSON objects or
    tuples of named fields: models, dataclasses, TypedDicts and named tuples."""
    seen: set[type] = {model}
    pending: list[Any] = [info.annotation for info in model.model_fields.values()]
    while pending:
        annotation = pending.pop()
        if get_origin(annotation) is not None or not inspect.isclass(annotation):
            pending += get_args(annotation)
        elif annotation not in seen and _is_structured(annotation):
            seen.add(annotation)
            yield annotation
            if issubclass(annotation, BaseModel):
                pending += [info.annotation for info in annotation.model_fields.values()]


def _is_structured(cls: type) -> bool:
    return (
        issubclass(cls, BaseModel)
        or dataclasses.is_dataclass(cls)
        or is_typeddict(cls)
        or (issubclass(cls, tuple) and hasattr(cls, "_fields"))
    )


def _unions(annotation: Any) -> Iterator[Any]:
    """The unions within `annotation` (not looking into models)."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        yield annotation
    for arg in get_args(annotation):
        yield from _unions(arg)


def _ref_kinds(annotation: Any) -> set[str]:
    """The reference kinds within `annotation` (not looking into models)."""
    kinds: set[str] = set()
    if get_origin(annotation) is Annotated:
        kinds |= {meta.kind for meta in annotation.__metadata__ if isinstance(meta, Ref)}
    for arg in get_args(annotation):
        kinds |= _ref_kinds(arg)
    return kinds
