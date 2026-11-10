"""Skillsets as a provider mounts them (docs/how-to/add-a-skillset.md).

A skillset is registered in the `roboscript.skillsets` entry-point group (or named by import
path, `package.module:attribute`, where its package is only on `PYTHONPATH`, as in the simulator
env) as a `SkillSetPlugin`: its manifest, data that holds its bindings, and its extensions, the
tools it adds to core toolsets (`ToolsetExtension`). A provider of a core toolset is mounted with
skillset names (data, so a toolset server mounts them as an in-process provider does): it adds
the extensions of that toolset, binds the toolset with the skillsets' bindings and leaves out
each tool whose bindings none of them provides (`Mounted.unmounted` says why). Given the
embodiment it mounts for, it also leaves out each of the toolset's own tools that does not fit the
robot (its arms by count and name, the kinds of action it takes, as the registered core skillset
describing the toolset declares them: `misfits`), and refuses a named skillset that does not fit
it. Which roles see the tools, and in which order, stays the Hub's business: a skillset decides
where code lives and when it mounts, never a tool's place in the list. A mounted skillset's
`guide` reaches the roles' prompts through `guides`.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from roboscript.contracts.action import ActionKind
from roboscript.contracts.robot import ArmName
from roboscript.contracts.skills import Bindings, ChainStep, SkillNeeds, SkillSet, SkillSpec
from roboscript.embodiments import EmbodimentProfile
from roboscript.kernel import entrypoints
from roboscript.tools.skill_fit import WrittenFit
from roboscript.tools.spec import Toolset, ToolsetExtension, ToolSpec

GROUP: Final = "roboscript.skillsets"
CORE_PACKAGE: Final = "roboscript"
"""The package whose registered skillsets may be core ones (`describing`)."""


@dataclass(frozen=True)
class SkillSetPlugin:
    """A registered skillset: `manifest` (with its bindings), `extensions`, the tools it adds,
    each to the core toolset it `extends`, and `fitters`: by skill name, how each of its written
    skills' parameters are fitted on a segment of a demonstration, with the skill's `act` that S1
    rehearses to measure the fit (`tools.skill_fit.WrittenFit`); S1 stores the parameters in the
    Memory for the skill's runs."""

    manifest: SkillSet
    extensions: tuple[type[ToolsetExtension], ...] = ()
    fitters: Mapping[str, WrittenFit] = field(default_factory=dict)

    def __post_init__(self) -> None:
        declared = {skill.name for skill in self.manifest.skills}
        added = [spec.name for ext in self.extensions for spec in ext.specs()]
        if missing := sorted(set(added) - declared):
            raise TypeError(f"skillset {self.manifest.name}: its manifest lacks {missing}")


def skill(
    spec: ToolSpec,
    chain: Sequence[ChainStep],
    *,
    min_arms: int = 1,
    arms: Sequence[ArmName] = (),
    actions: Sequence[ActionKind] | None = None,
    perception: Sequence[str] = (),
) -> SkillSpec:
    """A manifest's entry for the tool `spec`: its name, description and parameter schema as the
    model sees them (taken from the spec, so the two never drift), its bindings (`spec.needs`)
    and what the caller declares: its implementation chain, the arms it needs (how many, and the
    names its parameters take), the kinds of action it executes and the perception inputs it
    reads. `actions` defaults to joint positions for a skill whose chain reaches the env's
    `execute` (`env_execute`), which carries only those, and to none otherwise."""
    if actions is None:
        actions = ["joint_position"] if "env_execute" in chain else []
    return SkillSpec(
        name=spec.name,
        description=spec.description,
        parameters=spec.input_schema,
        needs=SkillNeeds(min_arms=min_arms, arms=list(arms), actions=list(actions)),
        perception=list(perception),
        chain=list(chain),
        bindings=list(spec.needs),
    )


def load_skillset(ref: str) -> SkillSetPlugin:
    """A skillset by registered name, or by import path `package.module:attribute`."""
    found = entrypoints.load_ref(ref) if ":" in ref else entrypoints.load(GROUP, ref)
    if not isinstance(found, SkillSetPlugin):
        raise TypeError(f"{ref} is a {type(found).__name__}, not a SkillSetPlugin")
    return found


def extension_tools(toolset: str, mounted: Collection[str] = ()) -> set[str]:
    """The tools the registered skillsets add to `toolset`, whether mounted or not, and those
    the skillsets named in `mounted` add (a provider's: a benchmark example's skillset is
    mounted by import path, not registered)."""
    refs = {*entrypoints.names(GROUP), *mounted}
    return {
        spec.name
        for ref in sorted(refs)
        for ext in load_skillset(ref).extensions
        if ext.extends == toolset
        for spec in ext.specs()
    }


def fits_skills(skillsets: Collection[Sequence[str]]) -> bool:
    """Whether any of the skillsets mounted (each provider's, as `ToolProvider.skillsets` names
    them) fits written skills (`SkillSetPlugin.fitters`): a stack where the Memory may keep a
    skill's fitted parameters (`toolsets.memory.SkillMemoryTools`)."""
    return any(load_skillset(ref).fitters for refs in skillsets for ref in refs)


def misfits(skill: SkillSpec, profile: EmbodimentProfile) -> list[str]:
    """Why `skill` does not fit `profile`'s robot, one line each (empty: it fits): too few arms,
    arms named otherwise than its parameters take, a kind of action the robot does not take."""
    lines = []
    needs, arms, kinds = skill.needs, len(profile.arms), profile.actions.kinds()
    if needs.min_arms > arms:
        lines.append(f"{skill.name}: needs {needs.min_arms} arms, {profile.name} has {arms}")
    if needs.arms and (other := sorted(set(profile.arms) - set(needs.arms))):
        lines.append(
            f"{skill.name}: takes arms named {sorted(needs.arms)}, {profile.name} has {other}"
        )
    if absent := sorted(set(needs.actions) - kinds):
        lines.append(
            f"{skill.name}: executes {absent} actions, {profile.name} takes {sorted(kinds)}"
        )
    return lines


def unfit(skillset: SkillSet, profile: EmbodimentProfile) -> str | None:
    """Why `skillset` does not fit `profile`'s embodiment at all (None: it may)."""
    if skillset.embodiments and profile.name not in skillset.embodiments:
        return f"{skillset.name}: fits {skillset.embodiments}, not {profile.name!r}"
    return None


def core_skillsets(toolset: type[Toolset]) -> list[str]:
    """The registered core skillsets (`layer="core"`) that describe tools of `toolset`'s own,
    e.g. `manipulation` for the `manipulation` toolset: whatever mounts the toolset mounts them."""
    return describing({spec.name for spec in toolset.specs()})


def describing(tools: Collection[str]) -> list[str]:
    """The registered core skillsets whose manifests describe any of `tools`: those the core
    package registers (`CORE_PACKAGE`; the core's skillsets live in it,
    docs/how-to/add-a-skillset.md) with `layer="core"`. Only those are imported, so another
    package's skillset, loadable or not, never enters a stack that does not name it."""
    own = set(tools)
    found = []
    for name in entrypoints.names_in(GROUP, CORE_PACKAGE):
        manifest = load_skillset(name).manifest
        if manifest.layer == "core" and own & {skill.name for skill in manifest.skills}:
            found.append(name)
    return found


def fit_problems(
    mounts: Sequence[tuple[str, Collection[str], Sequence[str]]], profile: EmbodimentProfile
) -> list[str]:
    """Mounted tools that do not fit `profile`'s robot, one line each (empty: all fit): `mounts`
    gives each toolset's name, the tools it mounted and the skillsets mounted on it; a tool is
    judged as its skillset's manifest, or a core skillset's, describes it (`misfits`)."""
    lines = []
    for name, tools, skillsets in mounts:
        described: dict[str, SkillSpec] = {}
        for ref in [*describing(tools), *skillsets]:
            described |= {skill.name: skill for skill in load_skillset(ref).manifest.skills}
        lines += [
            f"{name}.{line}"
            for tool in tools
            if tool in described
            for line in misfits(described[tool], profile)
        ]
    return lines


def guides(skillsets: Sequence[str]) -> list[str]:
    """The guides (`SkillSet.guide`, Markdown templates) of the skillsets named in `skillsets`,
    each skillset once, in the order named; those without one give none."""
    found = [load_skillset(ref).manifest for ref in dict.fromkeys(skillsets)]
    return [manifest.guide for manifest in found if manifest.guide is not None]


@dataclass(frozen=True)
class Mounted:
    """What a toolset mounted with skillsets has: its tools (its own, then each extension's, in
    declaration order), each tool's owner (the toolset, or an extension class to make on it),
    the bindings, and the tools left out with why."""

    specs: list[ToolSpec]
    owners: dict[str, type[ToolsetExtension] | None]
    bindings: Bindings
    unmounted: dict[str, str] = field(default_factory=dict)


def mounted(
    toolset: type[Toolset], skillsets: Sequence[str], profile: EmbodimentProfile | None = None
) -> Mounted:
    """`toolset`'s tools with the skillsets named in `skillsets`, for `profile`'s robot (None: no
    fit check; see the module)."""
    plugins = [load_skillset(ref) for ref in skillsets]
    names = [plugin.manifest.name for plugin in plugins]
    if twice := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"skillsets named twice: {twice}")
    described: dict[str, SkillSpec] = {}  # each tool's skill, where a manifest describes it
    if profile is not None:
        for plugin in plugins:
            if (why := unfit(plugin.manifest, profile)) is not None:
                raise ValueError(f"skillset {why}")
        for ref in core_skillsets(toolset):
            described |= {skill.name: skill for skill in load_skillset(ref).manifest.skills}
        for plugin in plugins:
            described |= {skill.name: skill for skill in plugin.manifest.skills}
    bindings = Bindings()
    for plugin in plugins:
        bindings = bindings.merged(plugin.manifest.bindings)
    owners: dict[str, type[ToolsetExtension] | None] = {}
    specs: list[ToolSpec] = []
    unmounted: dict[str, str] = {}
    candidates: list[tuple[ToolSpec, type[ToolsetExtension] | None]] = [
        (spec, None) for spec in toolset.specs()
    ]
    for plugin in plugins:
        for ext in plugin.extensions:
            if ext.extends == toolset.name:
                candidates += [(spec, ext) for spec in ext.specs()]
    for spec, owner in candidates:
        if spec.name in owners or spec.name in unmounted:
            raise TypeError(f"tool {spec.name!r} of {toolset.name} is added twice")
        if lacking := sorted(set(spec.needs) - bindings.provided()):
            if owner is not None:  # a skillset named here mounts all its tools, or none
                raise ValueError(
                    f"skillset tool {spec.name!r} of {toolset.name}: no mounted skillset binds "
                    f"{lacking}"
                )
            unmounted[spec.name] = f"no mounted skillset binds {lacking}"
            continue
        skill = described.get(spec.name)
        if profile is not None and skill is not None and (lines := misfits(skill, profile)):
            if owner is not None:
                raise ValueError(f"skillset tool of {toolset.name} does not fit: {lines}")
            unmounted[spec.name] = "; ".join(lines)
            continue
        owners[spec.name] = owner
        specs.append(spec)
    return Mounted(specs, owners, bindings, unmounted)
