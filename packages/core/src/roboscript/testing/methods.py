"""Conformance checks for methods' declarations, and the information boundary's negative tests
(docs/how-to/add-a-method.md, docs/architecture.md).

A method conforms when its declarations load (`learning.method.check_method`), its phases
configure a Hub, no eval or D-only phase (`PhaseSpec.d_only`) gives any role a P tier, and a
method that composes jobs (`Method.jobs`) gives a body for each phase of them (`Method.kinds`,
`orchestrate.worker.check_kinds`). The negative half
of the information boundary's checks runs for every method:

- a phase declaring an eval or D-only phase with privilege must fail to load;
- a table of an eval or D-only phase listing a tier-P tool fails at start;
- in an episode of an eval or D-only phase, a tier-P tool is refused, even where a table lists
  it: a model's call (the episode does not have it), another tool's call on the episode's
  behalf, and opening the episode at tier P.

Use the checks from pytest by subclassing `MethodConformance` as a `Test*` class with a
`method_target` fixture (the method class), or without pytest via
`roboscript.testing.precheck`. Running a task on MockEnv comes with the method skeletons.
"""

import tempfile
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import ProtocolError, ToolError
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import ToolName
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.tools import ToolCall
from roboscript.hub import Hub
from roboscript.kernel import KernelError
from roboscript.learning.method import check_method
from roboscript.orchestrate.worker import check_kinds
from roboscript.testing import raises, unit
from roboscript.testing.phases import extended_phases
from roboscript.tools import LocalProvider, ToolContext, Toolset, tool

PROBE: Final = "privilege_probe"


class Peeked(BaseModel):
    peeked: bool = True


class PrivilegeProbe(Toolset):
    """A tier-P tool, and a tier-D tool that calls it for its episode (a tool-to-tool call)."""

    name = PROBE

    @tool(kind="pure", tier="P")
    async def peek(self, ctx: ToolContext) -> Peeked:
        """Read the simulator's ground truth (a probe: tier P)."""
        return Peeked()

    @tool(kind="pure", tier="D")
    async def relay(self, ctx: ToolContext) -> Peeked:
        """Peek on the episode's behalf (a probe: tier D, calls a tier-P tool)."""
        try:
            result = await ctx.call("peek")
        except PermissionError as error:  # what the Hub said, for the model to read
            raise ToolError(f"refused: {error}") from None
        assert isinstance(result, Peeked)
        return result


def probed_phases(phases: list[PhaseSpec]) -> list[PhaseSpec]:
    """`phases` with every role of every eval or D-only phase listing both probe tools first: the
    table a method could wrongly declare, which the Hub refuses before any episode."""
    roles = {role for phase in phases for role in phase.visibility}
    add = {role: [f"{PROBE}.peek", f"{PROBE}.relay"] for role in roles}
    evals = [phase.name for phase in phases if phase.holds_d]
    return extended_phases(phases, add, after=PROBE, only=evals)


def relayed_phases(phases: list[PhaseSpec]) -> list[PhaseSpec]:
    """`phases` with every role of every eval or D-only phase listing the tier-D `relay` alone,
    and each such phase rebuilt without validation (`model_construct`) with every role at tier P:
    a declaration that slipped past `PhaseSpec`'s own check, so that the Hub's must hold."""
    roles = {role for phase in phases for role in phase.visibility}
    evals = [phase.name for phase in phases if phase.holds_d]
    relayed = extended_phases(
        phases, {role: [f"{PROBE}.relay"] for role in roles}, after=PROBE, only=evals
    )
    return [
        PhaseSpec.model_construct(**{**dict(p), "tiers": dict.fromkeys(p.visibility, "P")})
        if p.holds_d
        else p
        for p in relayed
    ]


class _NoEvaluation:
    """An evaluation of no phase: a method's jobs alone."""

    phases: tuple[str, ...] = ()

    def add(self, graph: Any, key: Any, learned: Any, begun: Any) -> None:
        pass


class MethodConformance:
    """The checks, as `test_*` methods taking the method class."""

    @unit
    async def test_declarations_load(self, method_target: Any) -> None:
        assert check_method(method_target) == []

    @unit
    async def test_jobs_have_bodies(self, method_target: Any) -> None:
        """A method that composes jobs gives a body for each phase of them, so that its workers
        run them."""
        if not hasattr(method_target, "jobs"):
            return
        stages = method_target().jobs(["unit"], _NoEvaluation()).stages
        assert check_kinds(getattr(method_target, "kinds", {}), stages) == []

    @unit
    async def test_eval_phases_use_d_only(self, method_target: Any) -> None:
        for phase in method_target.phases:
            if phase.holds_d:
                assert set(phase.tiers.values()) <= {"D"}, phase.name

    @unit
    async def test_privileged_eval_is_refused(self, method_target: Any) -> None:
        for phase in method_target.phases:
            roles = list(phase.visibility) or ["actor"]
            data = phase.model_dump() | {"kind": "eval", "tiers": dict.fromkeys(roles, "P")}
            try:
                PhaseSpec.model_validate(data)
            except ValidationError:
                pass
            else:
                raise AssertionError(f"{phase.name}: an eval phase with P tiers loaded")
            held = data | {"kind": "learn", "d_only": True}
            with raises(ValidationError, "D-only phase"):
                PhaseSpec.model_validate(held)

    @unit
    async def test_an_eval_table_listing_a_privileged_tool_is_refused(
        self, method_target: Any
    ) -> None:
        if not any(phase.holds_d for phase in method_target.phases):
            return
        phases = probed_phases(list(method_target.phases))
        provider = LocalProvider(PrivilegeProbe())
        with (
            tempfile.TemporaryDirectory() as directory,
            raises(KernelError, r"(an eval|a D-only) phase lists tier-P tools"),
        ):
            Hub([provider], blobs=LocalBlobStore(Path(directory)), phases=phases)

    @unit
    async def test_privileged_tools_are_refused_in_eval_episodes(self, method_target: Any) -> None:
        phases = relayed_phases(list(method_target.phases))
        evals = [phase for phase in phases if phase.holds_d]
        provider = LocalProvider(PrivilegeProbe())
        with tempfile.TemporaryDirectory() as directory:
            blobs = LocalBlobStore(Path(directory))
            async with Hub([provider], blobs=blobs, phases=phases) as hub:
                for phase in evals:
                    for role in phase.visibility:
                        await _check_role(hub, phase, role)


async def _check_role(hub: Hub, phase: PhaseSpec, role: str) -> None:
    where = (phase.name, role)
    # the Hub's own check, though the declaration gives the role tier P
    with raises(ProtocolError, rf"phase '{phase.name}' runs at tier D only"):
        await hub.open_episode(OpenEpisode(stage=phase.name, role=role, tier="P"))
    session = await hub.open_episode(OpenEpisode(stage=phase.name, role=role))
    episode = session.episode_id
    names = [descriptor.name for descriptor in session.tools]
    assert "peek" not in names and "relay" in names, (*where, names)
    peek = ToolCall(call_id="c1", name=ToolName("peek"), arguments={}, origin="model")
    called = await hub.call(episode, peek)
    assert (called.is_error, called.error_code) == (True, "unknown_tool"), where
    relay = peek.model_copy(update={"call_id": "c2", "name": ToolName("relay")})
    relayed = await hub.call(episode, relay)
    assert relayed.error_code == "tool_error", where
    assert "is a tier-P tool" in _text(relayed.content), where
    # an episode at tier P in such a phase, had anything opened one: refused as an eval or
    # D-only phase, the tool-to-tool call and a direct one alike
    hub.store.get(episode).tier = "P"
    relayed = await hub.call(episode, relay.model_copy(update={"call_id": "c3"}))
    assert relayed.error_code == "tool_error", where
    held = "an eval" if phase.kind == "eval" else "a D-only"
    assert f"phase {phase.name!r} is {held} phase" in _text(relayed.content)
    with raises(PermissionError, f"is {held} phase"):
        await hub.service_call(episode, "peek", {}, parent="c4")
    await hub.close_episode(episode)


def _text(parts: Any) -> str:
    return "\n".join(part.text for part in parts if getattr(part, "text", None))
