"""The method's phases decide who sees which tool, in which order, at which tier, under which
rules; the toolset's code decides a tool's tier; mounting refuses clashes and typos; the eval
invariant holds at the Hub."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import ToolName
from roboscript.contracts.phase import PhaseRules, PhaseSpec
from roboscript.hub import Hub
from roboscript.kernel import KernelError
from roboscript.testing.mock import Mock
from roboscript.testing.phases import TEST_PHASES
from roboscript.tools import LocalProvider, ToolContext, Toolset, tool

CONFIGS = Path(__file__).parents[3] / "configs"


class Out(BaseModel):
    ok: bool = True


class Sensors(Toolset):
    name = "sensors"

    @tool(kind="perception", tier="P")
    async def truth(self, ctx: ToolContext) -> Out:
        """Ground truth: tier P, by its code."""
        return Out()

    @tool(kind="pure", tier="D")
    async def draft(self, ctx: ToolContext) -> Out:
        """In no phase's table."""
        return Out()

    @tool(kind="memory", tier="D")
    async def edit(self, ctx: ToolContext) -> Out:
        """For the Reviser only."""
        return Out()


class Clashing(Sensors):
    name = "clashing"

    @tool(kind="perception", tier="D")
    async def look(self, ctx: ToolContext) -> Out:
        """Clashes with the Mock's look."""
        return Out()


class Named(Toolset):
    name = "hub"

    @tool(kind="pure", tier="D")
    async def ping(self, ctx: ToolContext) -> Out:
        """Ping."""
        return Out()


ENDS = {"actor": ["hub.finish", "hub.scratchpad"], "reviser": ["hub.finish"]}
PHASES = [
    PhaseSpec(
        name="learn",
        kind="learn",
        visibility={
            "actor": ["hub.finish", "sensors.truth", "hub.scratchpad"],  # the table's order
            "reviser": ["sensors.edit", "hub.finish"],
        },
        tiers={"actor": "P"},
    ),
    PhaseSpec(
        name="practice",
        kind="learn",
        rules=PhaseRules(fail_fast=True),
        visibility={"actor": ["sensors.truth", *ENDS["actor"]], "reviser": ["sensors.*", "hub.*"]},
    ),
    PhaseSpec(
        name="exam",
        kind="eval",
        rules=PhaseRules(recover=True),
        visibility={"actor": ["sensors.draft", *ENDS["actor"]]},
    ),
]


async def test_the_method_not_the_code_decides_who_sees_a_tool_and_in_what_order(
    tmp_path: Path,
) -> None:
    hub = Hub([LocalProvider(Sensors())], blobs=LocalBlobStore(tmp_path), phases=PHASES)
    cases = [
        ("learn", "P", "actor"),
        ("learn", "D", "actor"),
        ("learn", "D", "reviser"),
        ("practice", "D", "actor"),
        ("practice", "D", "reviser"),
        ("exam", "D", "actor"),
    ]
    seen = {}
    async with hub:
        for stage, tier, role in cases:
            spec = OpenEpisode(stage=stage, tier=tier, role=role)  # type: ignore[arg-type]
            session = await hub.open_episode(spec)
            assert session.role == role
            seen[stage, tier, role] = [t.name for t in session.tools]
    assert seen == {
        ("learn", "P", "actor"): ["finish", "truth", "scratchpad"],  # tier P sees tier-P tools
        ("learn", "D", "actor"): ["finish", "scratchpad"],
        ("learn", "D", "reviser"): ["edit", "finish"],
        ("practice", "D", "actor"): ["finish", "scratchpad"],
        ("practice", "D", "reviser"): ["draft", "edit", "finish", "scratchpad"],  # truth: tier P
        ("exam", "D", "actor"): ["draft", "finish", "scratchpad"],
    }


async def test_episodes_a_phase_does_not_allow_are_refused(tmp_path: Path) -> None:
    async with Hub(
        [LocalProvider(Sensors())], blobs=LocalBlobStore(tmp_path), phases=PHASES
    ) as hub:
        with pytest.raises(ProtocolError, match="unknown phase 'dev'"):
            await hub.open_episode(OpenEpisode(stage="dev"))
        with pytest.raises(ProtocolError, match="has no role 'demo_reader'"):
            await hub.open_episode(OpenEpisode(stage="learn", role="demo_reader"))
        with pytest.raises(ProtocolError, match="runs at tier D only for role 'actor'"):
            await hub.open_episode(OpenEpisode(stage="practice", tier="P"))
        with pytest.raises(ProtocolError, match="eval phase 'exam' runs at tier D only"):
            await hub.open_episode(OpenEpisode(stage="exam", tier="P"))
        session = await hub.open_episode(OpenEpisode(stage="exam"))
        with pytest.raises(PermissionError, match="eval phase"):
            await hub.service_call(session.episode_id, "truth", {}, parent="c1")


def test_an_eval_phase_listing_a_privileged_tool_fails_at_start(tmp_path: Path) -> None:
    """An eval phase's table may not name a tier-P tool (docs/architecture.md): the
    Hub refuses the method's phases before any episode, whatever tier its episodes would open at."""
    listed = PhaseSpec(
        name="exam", kind="eval", visibility={"actor": ["sensors.truth", "hub.finish"]}
    )
    with pytest.raises(KernelError, match=r"an eval phase lists tier-P tools \['truth'\]"):
        Hub([LocalProvider(Sensors())], blobs=LocalBlobStore(tmp_path), phases=[listed])
    whole = listed.model_copy(update={"visibility": {"actor": ["sensors.*", "hub.finish"]}})
    with pytest.raises(KernelError, match="an eval phase lists tier-P tools"):
        Hub([LocalProvider(Sensors())], blobs=LocalBlobStore(tmp_path), phases=[whole])


def test_clashes_and_typos_fail_at_start(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path)
    with pytest.raises(KernelError, match="'look' is provided by both 'mock' and 'clashing'"):
        Hub([LocalProvider(Mock()), LocalProvider(Clashing())], blobs=blobs, phases=TEST_PHASES)
    with pytest.raises(KernelError, match=r"toolsets mounted twice: \['mock'\]"):
        Hub([LocalProvider(Mock()), LocalProvider(Mock())], blobs=blobs, phases=TEST_PHASES)
    with pytest.raises(KernelError, match=r"toolsets mounted twice: \['hub'\]"):
        Hub([LocalProvider(Named())], blobs=blobs, phases=TEST_PHASES)
    typo = PhaseSpec(name="x", kind="learn", visibility={"actor": ["sensors.turth", "hub.finish"]})
    with pytest.raises(KernelError, match=r"sensors\.turth names a tool the sensors toolset lacks"):
        Hub([LocalProvider(Sensors())], blobs=blobs, phases=[typo])
    Hub([LocalProvider(Mock())], blobs=blobs, phases=[typo])  # sensors is not mounted here
    endless = PhaseSpec(name="x", kind="learn", visibility={"actor": ["sensors.*"]})
    with pytest.raises(KernelError, match=r"lists no hub\.finish"):
        Hub([LocalProvider(Sensors())], blobs=blobs, phases=[endless])
    twice = PhaseSpec(
        name="x", kind="learn", visibility={"actor": ["sensors.*", "sensors.edit", "hub.finish"]}
    )
    with pytest.raises(KernelError, match=r"sensors.edit lists \['edit'\] again"):
        Hub([LocalProvider(Sensors())], blobs=blobs, phases=[twice])
    with pytest.raises(KernelError, match=r"phases declared twice: \['x'\]"):
        Hub([LocalProvider(Sensors())], blobs=blobs, phases=[typo, typo])


def test_a_skillset_entry_stands_for_the_tools_skillsets_add_in_mount_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`toolset.+` lists, in mount order, the mounted tools registered skillsets add to that
    toolset, so a method names a category, never a benchmark's skill; where no skillset adds
    any, it lists none. The catalog lock names the tools it stands for."""
    from roboscript.hub import lock, visibility
    from roboscript.hub.visibility import Phases, mount

    added = {"sensors": {"draft", "truth"}}
    monkeypatch.setattr(
        visibility, "extension_tools", lambda toolset, mounted=(): added.get(toolset, set())
    )
    tools = mount([LocalProvider(Sensors()), LocalProvider(Mock())])
    phase = PhaseSpec(
        name="s1",
        kind="learn",
        visibility={"actor": ["mock.look", "sensors.+", "mock.+"]},
        tiers={"actor": "P"},
    )
    assert Phases([phase], tools).tools("s1", "actor") == ("look", "truth", "draft")
    locked = lock.catalog_lock(tools, {"s1": phase})["phases"][0]["visibility"]
    assert locked == {"actor": ["mock.look", "sensors.truth", "sensors.draft"]}
    with pytest.raises(KernelError, match=r"sensors\.\+ lists \['truth'\] again"):
        twice = phase.model_copy(update={"visibility": {"actor": ["sensors.truth", "sensors.+"]}})
        Phases([twice], tools)


def test_a_d_only_phase_refuses_a_privileged_tool_even_through_a_skillset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A learn phase the method marks D only is held as an eval phase is: no role at tier P, and
    a table resolving to a tier-P tool fails at start, whether it names the tool or reaches it
    through `toolset.+`, the tools a registered skillset adds."""
    from roboscript.contracts.skills import SkillSet
    from roboscript.hub.visibility import Phases, mount
    from roboscript.kernel import entrypoints
    from roboscript.tools.skillsets import GROUP, SkillSetPlugin, skill
    from roboscript.tools.spec import ToolsetExtension

    class Peeking(ToolsetExtension):
        extends = "mock"

        @tool(kind="pure", tier="P")
        async def peek_truth(self, ctx: ToolContext) -> Out:
            """Ground truth, added by a skillset: tier P."""
            return Out()

    plugin = SkillSetPlugin(
        manifest=SkillSet(
            name="peeking",
            version="1",
            layer="benchmark",
            skills=[skill(spec, []) for spec in Peeking.specs()],
        ),
        extensions=(Peeking,),
    )
    names, load = entrypoints.names, entrypoints.load
    monkeypatch.setattr(
        entrypoints,
        "names",
        lambda group: [*names(group), "peeking"] if group == GROUP else names(group),
    )
    monkeypatch.setattr(
        entrypoints,
        "load",
        lambda group, name: plugin if (group, name) == (GROUP, "peeking") else load(group, name),
    )
    tools = mount([LocalProvider(Mock(), None, ["peeking"])])
    assert tools[ToolName("peek_truth")].meta.tier == "P"
    learn = PhaseSpec(name="s2", kind="learn", visibility={"actor": ["mock.look", "mock.+"]})
    assert Phases([learn], tools).tools("s2", "actor") == ("look", "peek_truth")
    for entries in (["mock.look", "mock.+"], ["mock.look", "mock.peek_truth"]):
        held = learn.model_copy(update={"d_only": True, "visibility": {"actor": entries}})
        refused = r"a D-only phase lists tier-P tools \['peek_truth'\]"
        with pytest.raises(KernelError, match=refused):
            Phases([held], tools)
    with pytest.raises(ValidationError, match="D-only phase 's2' uses D only"):
        PhaseSpec(name="s2", kind="learn", d_only=True, visibility={}, tiers={"actor": "P"})
