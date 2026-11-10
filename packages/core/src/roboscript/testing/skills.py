"""Conformance checks for skillsets (docs/how-to/add-a-skillset.md).

`MOCK_SKILLS` (registered as `mock`) is the Mock benchmark's skillset: no tools of its own, its
bindings for the core's entries on MockEnv's robot (the executor's tuning: the Mock benchmark's
values; MockEnv's table as its support surface) and its guide (`mock_guide.md`).
`SkillSetConformance` checks a manifest: it round trips and every skill mounts on the target
embodiment with the bindings the env provides (step 1 of validation, the interface check).
`SkillSetPluginConformance` checks a registered skillset (`roboscript.skillsets`) as providers
mount it: its manifest describes exactly the tools it adds, those tools mount on the core toolsets
they extend with the bindings of the skillsets mounted with it, and a remote toolset's provider
(the Hub's side of a toolset server, `integrations.mcp.remote`) lists the same tools as an
in-process one, so the skillset mounts alike served or not. Use from pytest by subclassing with a
`skillset_target` / `plugin_target` fixture, or via `roboscript.testing.precheck`. All four
validation steps, run and frozen as one, are `roboscript.testing.validation.validate`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from roboscript.contracts.executor import ExecutorConfig
from roboscript.contracts.skills import Bindings, SkillSet, SupportSurface
from roboscript.embodiments import EmbodimentProfile
from roboscript.envs.mock import TABLE_Z
from roboscript.skills import check_interface
from roboscript.testing import unit
from roboscript.tools import Toolset
from roboscript.tools.skillsets import SkillSetPlugin, load_skillset, mounted

MOCK_EXECUTOR: Final = ExecutorConfig(
    waypoint_tolerance_m=0.003,
    straight_step_m=0.02,
    straight_step_deg=5.0,
    max_joint_step_rad=1.0,
    lock_tolerance_m=0.02,
    lock_tolerance_deg=5.0,
    max_joint_travel_rad=5.5,
    replans=3,
    settle_holds=3,
    open_holds=4,
    place_step_m=None,
    release_holds=0,
    retry_deviated=True,
    stop_tolerance_m=0.015,
    stop_tolerance_deg=8.0,
    contact_off_path_m=0.05,
    press_m=0.01,
    max_translation_m=0.2,
    max_rotation_deg=90.0,
    min_arm_gap_m=0.03,
    workspace=None,
    workspace_margin_m=0.08,
    obstacle_margin_m=0.02,
    release_withdraw_m=0.05,
    release_withdraw_max_m=0.15,
    lift_clearance_m=0.03,
    max_lift_m=0.15,
    reject_repeats=False,
    repeat_tolerance_m=0.005,
    repeat_tolerance_deg=5.0,
    follow_step_m=0.025,
    follow_tolerance_m=0.003,
    follow_stall=0.25,
    follow_regrasps=1,
    follow_regrasp_reported=True,
    follow_joint_paths=True,
    follow_past=0.25,
    hold_frames=True,
    honor_front=True,
    tilted_turns=True,
    coarse_stops=True,
    drop_start_colliders=True,
    near_bare=True,
)
MOCK_SKILLS: Final = SkillSetPlugin(
    manifest=SkillSet(
        name="mock",
        version="1",
        layer="benchmark",
        envs=["mock"],
        embodiments=["mock_bimanual"],
        bindings=Bindings(executor=MOCK_EXECUTOR, support_surface=SupportSurface(top_z=TABLE_Z)),
        guide=Path(__file__).with_name("mock_guide.md").read_text(encoding="utf-8"),
    )
)


@dataclass(frozen=True)
class SkillSetTarget:
    skillset: SkillSet
    profile: EmbodimentProfile
    bindings: Bindings


class SkillSetConformance:
    """The checks, as `test_*` methods taking a `SkillSetTarget`."""

    @unit
    async def test_manifest_round_trips(self, skillset_target: SkillSetTarget) -> None:
        manifest = skillset_target.skillset
        assert SkillSet.model_validate_json(manifest.model_dump_json()) == manifest

    @unit
    async def test_interface(self, skillset_target: SkillSetTarget) -> None:
        target = skillset_target
        assert check_interface(target.skillset, target.profile, target.bindings.provided()) == []


@dataclass(frozen=True)
class PluginTarget:
    """`ref`, a skillset's registered name or import path; the embodiment it is to mount on;
    `hosts`, the core toolsets by name (those it extends, and those its manifest's skills are,
    for a core skillset); `also`, the skillsets an env mounts with it (their bindings count)."""

    ref: str
    profile: EmbodimentProfile
    hosts: Mapping[str, type[Toolset]]
    also: Sequence[str] = field(default_factory=tuple)

    @property
    def skillsets(self) -> list[str]:
        return [*self.also, self.ref]


class SkillSetPluginConformance:
    """The checks, as `test_*` methods taking a `PluginTarget`."""

    @unit
    async def test_manifest_round_trips(self, plugin_target: PluginTarget) -> None:
        manifest = load_skillset(plugin_target.ref).manifest
        assert SkillSet.model_validate_json(manifest.model_dump_json()) == manifest

    @unit
    async def test_interface(self, plugin_target: PluginTarget) -> None:
        bindings = Bindings()
        for ref in plugin_target.skillsets:
            bindings = bindings.merged(load_skillset(ref).manifest.bindings)
        manifest = load_skillset(plugin_target.ref).manifest
        assert check_interface(manifest, plugin_target.profile, bindings.provided()) == []

    @unit
    async def test_manifest_describes_the_tools_it_mounts(
        self, plugin_target: PluginTarget
    ) -> None:
        plugin = load_skillset(plugin_target.ref)
        found = {}
        for host in plugin_target.hosts.values():
            on = mounted(host, plugin_target.skillsets)
            assert not on.unmounted, f"{host.name} leaves out {on.unmounted}"
            found |= {spec.name: spec for spec in on.specs}
        for skill in plugin.manifest.skills:
            assert skill.name in found, (
                f"{skill.name} mounts on none of {sorted(plugin_target.hosts)}"
            )
            spec = found[skill.name]
            assert (skill.description, skill.parameters) == (spec.description, spec.input_schema)
            assert sorted(skill.bindings) == sorted(spec.needs)
        for ext in plugin.extensions:
            assert ext.extends in plugin_target.hosts, f"it extends {ext.extends}, not a host"

    @unit
    async def test_mounts_alike_served_or_in_process(self, plugin_target: PluginTarget) -> None:
        from roboscript.integrations.mcp.remote import RemoteProvider  # the `mcp` extra

        for host in plugin_target.hosts.values():
            local = mounted(host, plugin_target.skillsets)  # what a LocalProvider mounts
            served = RemoteProvider(host, "http://127.0.0.1:1/mcp", plugin_target.skillsets)
            assert [s.name for s in served.specs()] == [s.name for s in local.specs]
            assert served.unmounted == local.unmounted
