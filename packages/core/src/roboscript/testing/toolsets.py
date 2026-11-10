"""Conformance checks for toolsets (`roboscript.toolsets`,
docs/concepts/agent-runtime.md).

A toolset conforms when the class registered under its name is a `Toolset` of that name that
needs no constructor arguments; its config validates (and refuses keys it does not know); every
tool it declares is well formed, that is, a strict schema, a descriptor that survives JSON, the
three client exports and a name unique within the toolset; a provider mounts it, with the
skillsets the target names, leaving out a tool only with a reason; a remote provider (the Hub's
side of a toolset server, `integrations.mcp.remote`) lists the same tools as an in-process one, so
the toolset is the same served or not; and a Hub starts it, lists its tools to a role that sees
them all (every tool at tier P, the tier-D ones at tier D), opens and closes episodes, and stops
it.

A toolset that needs a config or skillsets to mount names them in its `ToolsetTarget`. Use the
checks from pytest by subclassing `ToolsetConformance` as a `Test*` class with a
`toolset_target` fixture, or without pytest via `roboscript.testing.precheck`:

```python
@pytest.fixture(params=registered("roboscript.toolsets", "roboscript_myrobot"))
def toolset_target(request):
    return ToolsetTarget.registered(request.param)

class TestMyToolsets(ToolsetConformance): ...
```
"""

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.tools import ToolDescriptor
from roboscript.hub import Hub
from roboscript.kernel import entrypoints
from roboscript.testing import unit
from roboscript.tools import (
    LocalProvider,
    Toolset,
    check_strict,
    to_chat_tool,
    to_mcp_tool,
    to_responses_tool,
)

GROUP = "roboscript.toolsets"


@dataclass(frozen=True)
class ToolsetTarget:
    """`toolset`: the class registered as `name`; `config`: a config it mounts with (the toolset's
    `Config` validates it); `skillsets`: the skillsets (names or import paths) mounted on it, for
    a toolset whose tools need bindings or are added by them."""

    name: str
    toolset: type[Toolset]
    config: Mapping[str, Any] = field(default_factory=dict)
    skillsets: Sequence[str] = ()

    @classmethod
    def registered(
        cls, name: str, config: Mapping[str, Any] | None = None, skillsets: Sequence[str] = ()
    ) -> "ToolsetTarget":
        """The toolset registered as `name` in `roboscript.toolsets`, loaded through its entry
        point."""
        return cls(name, entrypoints.load(GROUP, name), config or {}, tuple(skillsets))

    def provider(self) -> LocalProvider:
        return LocalProvider(self.toolset(), self.config, self.skillsets)


class ToolsetConformance:
    """The checks, as `test_*` methods taking a `ToolsetTarget`."""

    @unit
    async def test_is_a_toolset_of_its_name(self, toolset_target: ToolsetTarget) -> None:
        target = toolset_target
        assert isinstance(target.toolset, type) and issubclass(target.toolset, Toolset)
        assert target.toolset.name == target.name, (
            f"registered as {target.name!r}, named {target.toolset.name!r}"
        )
        assert isinstance(target.toolset(), Toolset)  # the Agent host and servers build it so

    @unit
    async def test_config_validates_and_refuses_unknown_keys(
        self, toolset_target: ToolsetTarget
    ) -> None:
        target = toolset_target
        config = target.toolset.Config.model_validate(dict(target.config))
        assert target.toolset.Config.model_validate_json(config.model_dump_json()) == config
        try:
            target.toolset.Config.model_validate({**target.config, "no_such_option": 1})
        except ValidationError:
            return
        raise AssertionError("a config with a key the toolset does not know must be refused")

    @unit
    async def test_tools_are_well_formed(self, toolset_target: ToolsetTarget) -> None:
        target = toolset_target
        specs = target.toolset.specs()
        names = [spec.name for spec in specs]
        assert len(names) == len(set(names)), f"tool names used twice in {target.name}"
        for spec in specs:
            where = f"{target.name}.{spec.name}"
            assert spec.meta.toolset == target.name, where
            assert spec.description, where
            check_strict(spec.input_schema)
            described = spec.descriptor
            assert ToolDescriptor.model_validate_json(described.model_dump_json()) == described
            assert to_responses_tool(spec)["name"] == spec.name, where
            assert to_chat_tool(spec)["function"]["name"] == spec.name, where
            assert to_mcp_tool(spec)["name"] == spec.name, where

    @unit
    async def test_a_provider_mounts_it(self, toolset_target: ToolsetTarget) -> None:
        target = toolset_target
        provider = target.provider()
        mounted = {spec.name for spec in provider.specs()}
        declared = {spec.name for spec in target.toolset.specs()}
        if declared:
            assert mounted, f"none of {target.name}'s tools mounts: {dict(provider.unmounted)}"
        assert not set(provider.unmounted) & mounted
        assert all(provider.unmounted.values()), "a tool left out says why"
        assert declared <= mounted | set(provider.unmounted), (
            "a tool is neither mounted nor left out"
        )

    @unit
    async def test_mounts_alike_served_or_in_process(self, toolset_target: ToolsetTarget) -> None:
        from roboscript.integrations.mcp.remote import RemoteProvider  # the `mcp` extra

        target = toolset_target
        local = target.provider()
        served = RemoteProvider(target.toolset, "http://127.0.0.1:1/mcp", target.skillsets)
        assert [(s.name, s.description, s.input_schema) for s in served.specs()] == [
            (s.name, s.description, s.input_schema) for s in local.specs()
        ]
        assert served.unmounted == local.unmounted

    async def test_a_hub_starts_it_and_lists_its_tools(self, toolset_target: ToolsetTarget) -> None:
        target = toolset_target
        provider = target.provider()
        shown = [s for s in provider.specs() if not s.meta.system_only]
        # an episode at tier P lists every tool; one at tier D only the tier-D ones
        visible = {
            "P": [s.name for s in shown],
            "D": [s.name for s in shown if s.meta.tier == "D"],
        }
        phase = PhaseSpec.model_validate(
            {
                "name": "dev",
                "kind": "learn",
                "visibility": {"actor": [f"{target.name}.*", "hub.finish"]},
                "tiers": {"actor": "P"},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            blobs = LocalBlobStore(Path(directory))
            async with Hub([provider], blobs=blobs, phases=[phase]) as hub:
                # a second episode meets what the first left, if anything
                for tier in ("P", "D", "P"):
                    opened = OpenEpisode(stage="dev", role="actor", tier=tier)
                    session = await hub.open_episode(opened)
                    listed = [tool.name for tool in session.tools if tool.name != "finish"]
                    assert listed == visible[tier], (
                        f"{target.name}: the Hub lists {listed} at tier {tier}"
                    )
                    await hub.close_episode(session.episode_id)
