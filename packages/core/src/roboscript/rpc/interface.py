"""RPC interfaces: `typing.Protocol` classes with `PLANE` and `VERSION` class variables.

Every public async method is an RPC method. Its request body is a JSON object of parameter
names to values (parameters with defaults may be left out); its response body is the dumped
return value.
"""

import inspect
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, TypedDict

from roboscript.contracts import versions

_NAMED = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


@dataclass(frozen=True)
class Method:
    name: str
    signature: inspect.Signature  # without `self`
    params: TypeAdapter[Any]  # the request body
    result: TypeAdapter[Any]


@dataclass(frozen=True)
class Interface:
    name: str
    contract: versions.Contract
    methods: dict[str, Method]


_described: dict[type, Interface] = {}


def describe(interface: type) -> Interface:
    """The methods of an interface with their request and response adapters, built once."""
    if interface not in _described:
        _described[interface] = _describe(interface)
    return _described[interface]


def _describe(interface: type) -> Interface:
    contract = versions.parse(f"{interface.PLANE}/{interface.VERSION}")  # type: ignore[attr-defined]
    methods = {
        name: _method(interface, name, fn)
        for name, fn in inspect.getmembers(interface, inspect.iscoroutinefunction)
        if not name.startswith("_")
    }
    if not methods:
        raise TypeError(f"{interface.__name__} has no public async methods")
    return Interface(interface.__name__, contract, methods)


def _method(interface: type, name: str, fn: Any) -> Method:
    where = f"{interface.__name__}.{name}"
    hints = get_type_hints(fn, include_extras=True)  # extras keep WireSecret's serializer
    params = list(inspect.signature(fn).parameters.values())[1:]
    fields: dict[str, Any] = {}
    for param in params:
        if param.kind not in _NAMED or param.name not in hints:
            raise TypeError(f"{where}: parameter {param.name!r} must be named and annotated")
        hint = hints[param.name]
        fields[param.name] = hint if param.default is param.empty else NotRequired[hint]
    if "return" not in hints:
        raise TypeError(f"{where}: annotate the return type")
    body = with_config(ConfigDict(extra="forbid"))(TypedDict(where, fields))  # type: ignore[operator]
    return Method(name, inspect.Signature(params), TypeAdapter(body), TypeAdapter(hints["return"]))
