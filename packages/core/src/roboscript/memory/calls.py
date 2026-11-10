"""Reading the calls and numbers a Memory's text names (the linter's executability checks).

A workflow stage's `call` and a lesson are templates the Actor turns into real tool calls, so a
call must name a tool the Actor has, with parameters it takes; a number or vector the text refers
to must be a parameter (`{name}`), not prose or a literal; and a parameter named without its
braces (`grasp=seal_grasp`) is still a reference. The tools come from the caller
(`ToolSig`: the toolsets' signatures, which the memory package cannot import); without them the
call checks stay off. Pure functions over text.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict

_CALL: Final = re.compile(r"(?<![\w.{<])([A-Za-z][A-Za-z0-9_]*)\(")
_KEYWORD: Final = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.*?)\s*$", re.DOTALL)
_PLURAL: Final = frozenset({"s", "es"})
_BRACKETS: Final = {"(": ")", "[": "]", "{": "}"}
_SEEN: Final = r"demonstrat\w*|demo|recorded|original|observed|measured"
_QUANTITY: Final = (
    r"vectors?|offsets?|displacements?|distances?|heights?|angles?|clearances?|translations?"
    r"|values?"
)
_DEMO_QUANTITY: Final = re.compile(
    rf"(?<![^\W_])(?:{_SEEN})(?![^\W_])[^.;(){{}}\n]{{0,40}}?(?<![^\W_])(?:{_QUANTITY})(?![^\W_])",
    re.IGNORECASE,
)
_LITERAL: Final = re.compile(
    r"(?<![\w.])-?\d+(?:\.\d+)?\s*(?:m|cm|mm|deg|degrees?)(?![^\W_])", re.IGNORECASE
)
_DECIMAL_ARG: Final = re.compile(r"^-?\d*\.\d+$")


class ToolSig(BaseModel):
    """What a template may call: a tool's parameter names (`rationale` aside), the ones it
    requires and a line on what it is for (what the roles that write templates read)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    params: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    purpose: str = ""
    takes: dict[str, str] = {}
    """The parameters that take a Memory spec: keyword -> its kind (`grasp_spec`, ...)."""

    def line(self, name: str) -> str:
        """`name(a, b:grasp_spec, [c]): purpose`: required parameters plain, the others in
        brackets, a spec parameter with the kind of Memory parameter it takes."""
        shown = ", ".join(
            (f"{p}:{self.takes[p]}" if p in self.takes else p) if p in self.required else f"[{p}]"
            for p in self.params
        )
        return f"- {name}({shown})" + (f": {self.purpose}" if self.purpose else "")


@dataclass(frozen=True)
class Call:
    """A call written in text: its name, how many positional arguments, its keyword arguments
    (name -> text) and its position."""

    name: str
    positional: int
    keywords: Mapping[str, str]
    start: int
    values: tuple[str, ...] = ()
    """The positional arguments' text."""


def parse_calls(text: str) -> list[Call]:
    """Every `name(args)` in `text`, nested ones included, in order. A call with no closing
    parenthesis reads to the end of the text."""
    calls: list[Call] = []
    for match in _CALL.finditer(text):
        args = _arguments(text, match.end())
        if len(args) == 1 and args[0].strip() in _PLURAL:
            continue  # "arm(s)"
        keywords = {m[1]: m[2] for a in args if (m := _KEYWORD.match(a))}
        positional = sum(1 for a in args if a.strip() and not _KEYWORD.match(a))
        values = tuple(a.strip() for a in args if a.strip() and not _KEYWORD.match(a))
        calls.append(Call(match[1], positional, keywords, match.start(), values))
    return calls


def _arguments(text: str, start: int) -> list[str]:
    """The top-level comma-separated arguments from `start` (just after a `(`) to its match."""
    args: list[str] = []
    stack: list[str] = [")"]
    quote = ""
    current: list[str] = []
    for ch in text[start:]:
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch in _BRACKETS:
            stack.append(_BRACKETS[ch])
        elif ch == stack[-1]:
            stack.pop()
            if not stack:
                break
        elif ch == "," and len(stack) == 1:
            args.append("".join(current))
            current = []
            continue
        current.append(ch)
    args.append("".join(current))
    return [a for a in args if a.strip()]


def unknown_calls(
    text: str, tools: Mapping[str, ToolSig], ignore: Collection[str] = ()
) -> list[tuple[str, str]]:
    """What in `text`'s calls the tools do not allow, as (match, why): a name that is no tool
    (and none of `ignore`: the predicates a check takes), too many positional arguments, a
    keyword the tool does not take."""
    found: list[tuple[str, str]] = []
    for call in parse_calls(text):
        if call.name in ignore:
            continue
        signature = tools.get(call.name)
        if signature is None:
            found.append((call.name, "names no tool the Actor has: write the real tool's name"))
            continue
        listed = ", ".join(signature.params) or "none"
        if extra := sorted(set(call.keywords) - set(signature.params) - {"rationale"}):
            why = f"{call.name} takes no parameter {extra[0]!r}; it takes: {listed}"
            found.append((f"{call.name}({extra[0]}=...)", why))
        elif call.positional + len(call.keywords) > len(signature.params):
            why = f"{call.name} takes at most {len(signature.params)} arguments: {listed}"
            found.append((call.name, why))
    return found


_REF_VALUE: Final = re.compile(r"\{([a-z][a-z0-9_]*)(?:\.[a-z][a-z0-9_]*)?\}|([a-z][a-z0-9_]*)")


def spec_mismatches(
    text: str,
    tools: Mapping[str, ToolSig],
    kinds: Mapping[str, str],
    ignore: Collection[str] = (),
) -> list[tuple[str, str]]:
    """Where a call passes a parameter to an argument that does not take its kind, as
    (match, why): a spec to an argument that takes none, or another kind of spec, or a number or
    category to one that takes a spec. `kinds`: each parameter's kind. Arguments given
    positionally are the tool's parameters in order."""
    found: list[tuple[str, str]] = []
    for call in parse_calls(text):
        signature = tools.get(call.name)
        if signature is None or call.name in ignore:
            continue
        for key, value in _values(text, call, signature):
            ref = _REF_VALUE.fullmatch(value.strip())
            name = ref and (ref[1] or ref[2])
            if name not in kinds:
                continue
            kind, takes = kinds[name], signature.takes.get(key)
            if kind.endswith("_spec") and takes != kind:
                what = f"takes a {takes}" if takes else "takes no spec"
                found.append(
                    (
                        f"{call.name}({key}={name})",
                        f"{key} of {call.name} {what}, not {name} (a {kind})",
                    )
                )
            elif takes is not None and not kind.endswith("_spec"):
                found.append(
                    (
                        f"{call.name}({key}={name})",
                        f"{key} of {call.name} takes a {takes}, not {name} (a {kind})",
                    )
                )
    return found


def _values(text: str, call: Call, signature: ToolSig) -> list[tuple[str, str]]:
    """A call's arguments as (keyword, value): the keyword ones, and the positional ones named
    by the tool's parameter order."""
    start = call.start + len(call.name) + 1
    args = _arguments(text, start)
    values: list[tuple[str, str]] = []
    position = 0
    for arg in args:
        if m := _KEYWORD.match(arg):
            values.append((m[1], m[2]))
        else:
            if position < len(signature.params):
                values.append((signature.params[position], arg))
            position += 1
    return values


_NAMED: Final = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=](?!=)")


def call_arguments(text: str) -> dict[str, list[str]]:
    """The arguments a call template names (`name=value` or `name:value`), by the tool it calls;
    a tool called twice names the union. Positional arguments name nothing."""
    found: dict[str, list[str]] = {}
    for match in _CALL.finditer(text):
        named = found.setdefault(match[1], [])
        for arg in _arguments(text, match.end()):
            if (key := _NAMED.match(arg)) is not None and key[1] not in named:
                named.append(key[1])
    return found


def template_refs(text: str, params: Collection[str]) -> list[str]:
    """The parameters `text` names, braced (`{name}`, `{name.field}`) or bare (`key=name`, a
    positional `name`, as a call template writes it), in order, each once."""
    braced = re.findall(r"\{([a-z][a-z0-9_]*)(?:\.[a-z][a-z0-9_]*)?\}", text)
    return list(
        dict.fromkeys(name for name in [*braced, *unbraced(text, params)] if name in params)
    )


def keyed_refs(text: str, params: Collection[str]) -> dict[str, str]:
    """The parameters a template's calls pass as a keyword's value, braced or bare
    (`height={lift}`, `grasp=seal_grasp`), as parameter -> keyword."""
    keyed: dict[str, str] = {}
    for call in parse_calls(text):
        for key, value in call.keywords.items():
            ref = re.fullmatch(r"\{?([a-z][a-z0-9_]*)(?:\.[a-z][a-z0-9_]*)?\}?", value.strip())
            if ref is not None and ref[1] in params:
                keyed[ref[1]] = key
    return keyed


def unbraced(text: str, params: Collection[str]) -> list[str]:
    """The parameters `text` names without braces: a value (`grasp=seal_grasp`, `f(seal_grasp)`),
    not a keyword (`seal_grasp=...`), not inside `{}` or `<>`."""
    found: list[str] = []
    for name in sorted(params, key=len, reverse=True):
        pattern = rf"(?<![\w.{{<])({re.escape(name)})(?![\w}}>.]|\s*=(?!=))"
        if re.search(pattern, text):
            found.append(name)
    return found


def demo_quantities(text: str) -> list[str]:
    """Phrases that point at a quantity the demonstration showed (`the demonstrated retreat
    displacement`), as text: it must be a parameter, so `text` should name one."""
    return [m.group(0) for m in _DEMO_QUANTITY.finditer(text)]


def literal_numbers(text: str, *, calls: bool) -> list[str]:
    """Lengths and angles written as numbers (`0.15 m`, `10 deg`), and with `calls` decimal
    keyword arguments (`height=0.10`): values that belong to parameters."""
    found = [m.group(0) for m in _LITERAL.finditer(text)]
    if calls:
        for call in parse_calls(text):
            found += [f"{k}={v}" for k, v in call.keywords.items() if _DECIMAL_ARG.match(v)]
    return found


def part_names(calls: Iterable[Call], tools: Mapping[str, ToolSig] | None = None) -> set[str]:
    """The parts a template registers: `part=<name>` of its calls, and where `tools` say which
    argument of a tool is its `part`, that positional argument too."""
    found: set[str] = set()
    for call in calls:
        if "part" in call.keywords:
            found.add(call.keywords["part"].strip("\"'"))
        elif (signature := (tools or {}).get(call.name)) is not None and "part" in signature.params:
            at = signature.params.index("part")
            if at < len(call.values):
                found.add(call.values[at].strip("\"'"))
    return found
