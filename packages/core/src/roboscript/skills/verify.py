"""Skillset verification, the interface step of validation (docs/how-to/add-a-skillset.md).

`check_interface` says why a skillset does not fit an embodiment; `check_mounts` is the precheck of
a stack's mounts: what refuses it (a named skillset that does not fit, a toolset that mounts none
of its tools) and which entries it goes without (a toolset's own tool that does not fit the robot
or lacks a binding: not mounted, and the check says why). The other steps (the information
boundary, the reachable ceiling, the freeze) and the report are the validation tool's,
`roboscript.testing.validation`.
"""

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass

from roboscript.contracts.skills import BindingName, Bindings, SkillSet
from roboscript.embodiments import EmbodimentProfile
from roboscript.skills.guide import render_guide
from roboscript.tools import Toolset
from roboscript.tools.skillsets import core_skillsets, load_skillset, misfits, mounted, unfit


def check_interface(
    skillset: SkillSet, profile: EmbodimentProfile, provided: Set[BindingName]
) -> list[str]:
    """Why a skillset, or skills of it, cannot mount on `profile` with the bindings `provided`,
    one line each (empty: everything mounts): an embodiment it does not fit, too few arms, arms
    named otherwise than its parameters take, a kind of action the profile does not take, a
    binding no skillset gives, a guide placeholder the robot has no value for. Schemas are
    checked by the manifest model."""
    lines = _whole(skillset, profile)
    for skill in skillset.skills:
        lines += misfits(skill, profile)
        if lacking := sorted(set(skill.bindings) - set(provided)):
            lines.append(f"{skill.name}: no binding for {lacking}")
    return lines


def _whole(skillset: SkillSet, profile: EmbodimentProfile) -> list[str]:
    """Why `skillset` as a whole cannot mount on `profile`: an embodiment it does not fit, a guide
    placeholder the robot has no value for."""
    lines = [] if (why := unfit(skillset, profile)) is None else [why]
    if skillset.guide is not None:
        try:
            render_guide(skillset.guide, profile)
        except ValueError as error:
            lines.append(f"{skillset.name}: its guide: {error}")
    return lines


@dataclass(frozen=True)
class MountCheck:
    """A stack's mounts checked (`check_mounts`): `problems`, why it cannot mount as named, one
    line each (the run is refused); `unmounted`, the entries it goes without on this robot, each
    `<toolset>.<tool>: not mounted, <why>` (the run goes on without them)."""

    problems: list[str]
    unmounted: list[str]


def check_mounts(
    skillsets: Sequence[str], toolsets: Mapping[str, type[Toolset]], profile: EmbodimentProfile
) -> MountCheck:
    """The skillsets named in `skillsets` (an env's, `Benchmark.skillsets`, and a run's) mounted
    on the core `toolsets` for `profile`, as the providers mount them (`tools.skillsets.mounted`).
    Problems: a named skillset that does not fit (`check_interface`: the embodiment, a skill it
    needs more arms or bindings for), a core skillset that does not fit as a whole, a toolset that
    mounts none of its tools (a provider refuses it). Unmounted: each toolset's own tool that does
    not fit the robot or that no skillset binds, as the core skillsets describing the toolsets'
    tools (`core_skillsets`, e.g. `manipulation`) and the named ones declare them: such an entry
    is not mounted, and that is no reason to refuse the run (docs/how-to/add-a-skillset.md:
    `together` mounts only on a robot with two arms)."""
    core = [ref for toolset in toolsets.values() for ref in core_skillsets(toolset)]
    named = list(dict.fromkeys(skillsets))
    plugins = {ref: load_skillset(ref) for ref in dict.fromkeys([*named, *core])}
    bindings = Bindings()
    for plugin in plugins.values():
        bindings = bindings.merged(plugin.manifest.bindings)
    problems: list[str] = []
    fitting = []  # the named skillsets that mount, as the providers get them
    for ref, plugin in plugins.items():
        manifest = plugin.manifest
        if manifest.layer == "core":  # its entries mount where they fit: the rest is unmounted
            lines = _whole(manifest, profile)
        else:
            lines = check_interface(manifest, profile, bindings.provided())
        problems += lines
        if ref in named and not lines:
            fitting.append(ref)
    unmounted: list[str] = []
    for name, toolset in toolsets.items():
        try:
            found = mounted(toolset, fitting, profile)
        except ValueError as error:
            problems.append(f"{name}: {error}")
            continue
        if toolset.specs() and not found.specs:
            problems.append(f"{name}: mounts none of its tools")
        unmounted += [f"{name}.{tool}: not mounted, {why}" for tool, why in found.unmounted.items()]
    return MountCheck(problems, unmounted)
