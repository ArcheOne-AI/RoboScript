"""Which tools an episode has (docs/architecture.md,
docs/concepts/agent-runtime.md): the mounted toolsets' own declarations (kind, tier,
system-only) and the method's phases (`contracts.phase.PhaseSpec`), which list, per role and in
order, what each role sees.

The order a model is shown its tools is the order of its role's entries, a toolset entry
(`toolset.*`) expanding to its tools in the order the toolset declares them and a skillset entry
(`toolset.+`) to the mounted tools registered skillsets add to that toolset, in mount order
(`tools.skillsets.mounted`; none where no skillset adds any): never the order toolsets happen to
be mounted or imported in. Entries naming a toolset that is not mounted are
left out (a stack may mount fewer toolsets than the method knows), and so is one naming a tool a
registered skillset adds to its toolset that this mount leaves out (`absent`: an env without the
skillset) when the phase marks it optional (`PhaseSpec.optional`); unmarked, it is an error at
start, so that a tool never drops out of a stack silently (a skillset a stack names mounts all its
tools or fails, `tools.skillsets.mounted`). An entry naming any other tool its mounted toolset
lacks is an error at start, as is a role of a phase that cannot `finish`.

The information boundary (docs/architecture.md) is checked here and in the Hub: a
phase of kind `eval` gives every role tier D (`PhaseSpec` refuses otherwise when the method loads),
an episode opens at most at its role's tier, and a tier-D episode never sees a tier-P tool.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.ids import PhaseName, RoleName, ToolName
from roboscript.contracts.phase import PhaseSpec, eval_tier_violations
from roboscript.contracts.tools import Tier, ToolDescriptor, ToolMeta
from roboscript.hub.builtins import FINISH
from roboscript.kernel import KernelError
from roboscript.tools import SharedConfig, ToolProvider, ToolSpec
from roboscript.tools.skillsets import extension_tools


@dataclass(frozen=True)
class Tool:
    """A mounted tool: spec, provider, `meta` (its toolset's declaration) and descriptor."""

    spec: ToolSpec
    provider: ToolProvider
    meta: ToolMeta
    descriptor: ToolDescriptor

    @property
    def qualified(self) -> str:
        return f"{self.meta.toolset}.{self.spec.name}"


def mount(providers: Sequence[ToolProvider]) -> dict[ToolName, Tool]:
    """Every provider's tools by name, in provider then declaration order. A toolset or tool
    name mounted twice, or toolsets of one `SharedConfig` class with different values (checked
    for toolsets in this process; a remote one's config is its server's) is an error."""
    names = [provider.name for provider in providers]
    if twice := sorted({name for name in names if names.count(name) > 1}):
        raise KernelError(f"toolsets mounted twice: {twice}")
    shared: dict[type[BaseModel], ToolProvider] = {}
    for provider in providers:
        if isinstance(provider.config, SharedConfig):
            first = shared.setdefault(type(provider.config), provider)
            if first.config != provider.config:
                kind = type(provider.config).__name__
                raise KernelError(
                    f"{first.name} and {provider.name} work on one {kind}, but their configs "
                    "differ: mount them with the same config"
                )
    tools: dict[ToolName, Tool] = {}
    for provider in providers:
        for spec in provider.specs():
            name = ToolName(spec.name)
            if name in tools:
                raise KernelError(
                    f"tool {name!r} is provided by both {tools[name].meta.toolset!r} "
                    f"and {provider.name!r}"
                )
            tools[name] = Tool(spec, provider, spec.meta, spec.descriptor)
    return tools


def absent(providers: Sequence[ToolProvider]) -> dict[str, set[str]]:
    """By toolset, the tools registered skillsets add to it that this mount does not have: those
    a provider leaves out for lack of a binding, and those of skillsets it does not mount."""
    found: dict[str, set[str]] = {}
    for provider in providers:
        have = {spec.name for spec in provider.specs()}
        missing = set(provider.unmounted) | extension_tools(provider.name, provider.skillsets)
        found[provider.name] = missing - have
    return found


def added_tools(
    toolset: str, mounted: Sequence[ToolName], skillsets: Sequence[str] = ()
) -> list[ToolName]:
    """What a `toolset.+` entry stands for: of `mounted` (the toolset's tools, in mount order),
    those registered skillsets, or the `skillsets` its provider mounts, add to it."""
    added = extension_tools(toolset, skillsets)
    return [tool for tool in mounted if tool in added]


def mounted_skillsets(tools: Mapping[ToolName, Tool]) -> dict[str, tuple[str, ...]]:
    """By toolset, the skillsets its provider mounts (as `ToolProvider.skillsets` names them)."""
    return {tool.meta.toolset: tuple(tool.provider.skillsets) for tool in tools.values()}


class Phases:
    """A method's phases over the mounted `tools`: each role's tools resolved once, in its
    entries' order. Raises `KernelError` for a phase named twice, an entry naming a tool its
    mounted toolset lacks, a tool listed twice for a role, an eval or D-only phase listing (or
    reaching through `toolset.+`) a tier-P tool, or a role without `finish` (where the
    Hub's own tools are mounted: not for the toolsets alone, as `learning.runner.actor_tools`
    reads them)."""

    def __init__(
        self,
        phases: Sequence[PhaseSpec],
        tools: Mapping[ToolName, Tool],
        absent: Mapping[str, set[str]] | None = None,
    ) -> None:
        names = [phase.name for phase in phases]
        if twice := sorted({name for name in names if names.count(name) > 1}):
            raise KernelError(f"phases declared twice: {twice}")
        self.specs: dict[PhaseName, PhaseSpec] = {phase.name: phase for phase in phases}
        by_toolset: dict[str, list[ToolName]] = {}
        for name, tool in tools.items():
            by_toolset.setdefault(tool.meta.toolset, []).append(name)
        skillsets = mounted_skillsets(tools)
        hub, _, finish = FINISH.partition(".")
        self._tools: dict[tuple[PhaseName, RoleName], tuple[ToolName, ...]] = {}
        for phase in phases:
            for role, entries in phase.visibility.items():
                where = f"phase {phase.name!r}, role {role!r}"
                optional = set(phase.optional)
                resolved = _resolve(entries, by_toolset, skillsets, absent or {}, optional, where)
                if hub in by_toolset and finish not in resolved:
                    raise KernelError(f"{where}: lists no {FINISH}, so no episode could end")
                privileged = [n for n in resolved if tools[n].meta.tier != "D"]
                if phase.holds_d and privileged:
                    raise KernelError(
                        f"{where}: {_article(phase.held)} lists tier-P tools {privileged} (D only)"
                    )
                self._tools[phase.name, role] = resolved

    @property
    def names(self) -> list[PhaseName]:
        return list(self.specs)

    def phase(self, name: str) -> PhaseSpec:
        """The phase named `name`; an unknown one is a request the Hub refuses."""
        found = self.specs.get(name)
        if found is None:
            raise ProtocolError(
                "invalid_request", f"unknown phase {name!r}; the method declares {self.names}"
            )
        return found

    def check_open(self, phase: PhaseSpec, role: str, tier: Tier) -> None:
        """Refuse an episode of `role` at `tier` that `phase` does not allow: a role it does not
        declare, a tier above the role's, and any tier but D in an eval or D-only phase."""
        if role not in phase.visibility:
            raise ProtocolError(
                "invalid_request",
                f"phase {phase.name!r} has no role {role!r}: {sorted(phase.visibility)}",
            )
        if wrong := eval_tier_violations(phase.kind, {role: tier}):
            raise ProtocolError(
                "invalid_request", f"eval phase {phase.name!r} runs at tier D only: {wrong[0]}"
            )
        if tier == "P" and (phase.d_only or phase.tier(role) != "P"):
            raise ProtocolError(
                "invalid_request", f"phase {phase.name!r} runs at tier D only for role {role!r}"
            )

    def tools(self, phase: PhaseName, role: RoleName) -> tuple[ToolName, ...]:
        """`role`'s tools in `phase`, in order, of every tier (the episode filters by its own)."""
        return self._tools[phase, role]


def _article(held: str) -> str:
    return f"an {held}" if held.startswith("eval") else f"a {held}"


def _resolve(
    entries: Sequence[str],
    by_toolset: Mapping[str, Sequence[ToolName]],
    skillsets: Mapping[str, Sequence[str]],
    absent: Mapping[str, set[str]],
    optional: set[str],
    where: str,
) -> tuple[ToolName, ...]:
    resolved: list[ToolName] = []
    for entry in entries:
        toolset, _, name = entry.partition(".")
        mounted = by_toolset.get(toolset)
        if mounted is None:
            continue
        if name == "*":
            found = list(mounted)
        elif name == "+":
            found = added_tools(toolset, mounted, skillsets.get(toolset, ()))
        elif name in mounted:
            found = [ToolName(name)]
        elif name in absent.get(toolset, ()):
            if entry in optional:
                continue
            raise KernelError(
                f"{where}: {entry} is a skillset's tool this stack does not mount (mount the "
                "skillset that adds it, or mark the entry optional in the phase)"
            )
        else:
            raise KernelError(f"{where}: {entry} names a tool the {toolset} toolset lacks")
        if again := sorted(set(found) & set(resolved)):
            raise KernelError(f"{where}: {entry} lists {again} again")
        resolved += found
    return tuple(resolved)
