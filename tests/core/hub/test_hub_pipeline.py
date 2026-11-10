"""The guard pipeline: the order of its steps, and what each built-in step does to a call."""

import logging
from collections.abc import Callable
from typing import Any, Literal

import anyio
import pytest
from pydantic import BaseModel

from roboscript.contracts.errors import EpisodeEnded, InfraError, ProtocolError, ToolError
from roboscript.contracts.events import EventKind
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.hub import Budgets, CallOutcome, DoneReport, OpenEpisode
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.phase import PhaseRules
from roboscript.contracts.receipt import ActionReceipt, GripperReport, ReceiptReason
from roboscript.contracts.tools import ContentPart, ToolCall
from roboscript.hub import (
    DEFAULT_PIPELINE,
    Call,
    Hub,
    HubConfig,
    Pipeline,
    Reject,
    RenderConfig,
)
from roboscript.hub.pipeline import Proceed
from roboscript.records import EventWriter, read_events
from roboscript.testing.mock import Mock
from roboscript.testing.phases import ROLES, TEST_PHASES, dev_phases, extended_phases
from roboscript.tools import (
    LocalProvider,
    ObjectRef,
    PartRef,
    Rationale,
    ToolContext,
    ToolModel,
    Toolset,
    tool,
)

Model = Callable[..., ToolCall]


def rationale(stage: str = "grasp") -> dict[str, str]:
    return {"stage": stage, "scene": "a box", "progress": "at it", "plan": "act"}


class Found(BaseModel):
    obj: ObjectRef
    parts: list[PartRef] = []


class Ok(BaseModel):
    ok: bool = True


class GraspSpec(ToolModel):
    """Where and how to grasp, relative to a part."""

    part: PartRef
    approach: Literal["top", "side", "auto"] = "auto"
    offset_m: Vec3 = (0.0, 0.0, 0.0)
    x_toward: PartRef | None = None


class BothArms(BaseModel):
    receipts: dict[str, ActionReceipt]
    parts: list[PartRef]


def broken_render(result: BaseModel) -> list[ContentPart]:
    raise RuntimeError("the renderer broke")


class Probe(Toolset):
    """Tools for poking at the pipeline; `PROBED` shows some of them beyond phase `dev`."""

    name = "probe"

    def __init__(self) -> None:
        self.grabbed: list[str] = []
        self.grasped: list[tuple[str, str, str | None]] = []

    @tool(kind="perception", tier="D")
    async def find(self, ctx: ToolContext, name: str) -> Found:
        """Find an object."""
        return Found(obj=f"object-{name}", parts=[f"part-{name}-handle"])

    @tool(kind="pure", tier="D")
    async def grab(self, ctx: ToolContext, obj: ObjectRef, times: int = 2) -> Found:
        """Grab it."""
        self.grabbed += [obj] * times
        return Found(obj=obj)

    @tool(kind="execution", tier="D")
    async def grasp(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        obj: ObjectRef,
        grasp: GraspSpec,
        outcome: ReceiptReason = ReceiptReason.ok,
    ) -> BothArms:
        """Grasp with both arms."""
        self.grasped.append((obj, grasp.part, grasp.x_toward))
        receipts = {"left": ActionReceipt(reason="ok"), "right": ActionReceipt(reason=outcome)}
        return BothArms(receipts=receipts, parts=[grasp.part])

    @tool(kind="execution", tier="D")
    async def push(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        outcome: ReceiptReason = ReceiptReason.ok,
        holding: bool | None = None,
    ) -> ActionReceipt:
        """Push something."""
        if holding is None:
            return ActionReceipt(reason=outcome, steps_used=1)
        gripper = GripperReport(commanded=0.0, measured=0.5, holding=holding)
        return ActionReceipt(reason=outcome, steps_used=1, arm="left", gripper=gripper)

    @tool(kind="execution", tier="D")
    async def refuse(
        self, ctx: ToolContext, rationale: Rationale, outcome: ReceiptReason = ReceiptReason.ok
    ) -> ActionReceipt:
        """An action the planner refuses before anything moves."""
        return ActionReceipt(reason=outcome, note="no IK solution reaches the target")

    @tool(kind="execution", tier="D", timeout_s=0.05)
    async def jam(self, ctx: ToolContext, rationale: Rationale) -> ActionReceipt:
        """An action that hangs."""
        await anyio.sleep(5)
        return ActionReceipt(reason="ok")

    @tool(kind="execution", tier="D", render=broken_render)
    async def ugly(
        self, ctx: ToolContext, rationale: Rationale, outcome: ReceiptReason
    ) -> ActionReceipt:
        """An action whose result cannot be rendered."""
        return ActionReceipt(reason=outcome)

    @tool(kind="pure", tier="D", timeout_s=0.05)
    async def slow(self, ctx: ToolContext) -> Ok:
        """Too slow."""
        await anyio.sleep(5)
        return Ok()

    @tool(kind="pure", tier="D")
    async def boom(
        self,
        ctx: ToolContext,
        how: Literal["infra", "grouped", "mixed", "ended", "bug", "long"],
    ) -> Ok:
        """Fail somehow."""
        if how in ("grouped", "mixed"):  # out of a task group: an exception group

            async def fail(bug: bool) -> None:
                if bug:
                    raise RuntimeError("bug")
                raise InfraError("not_ready", "the model did not answer")

            async with anyio.create_task_group() as group:
                group.start_soon(fail, False)
                group.start_soon(fail, how == "mixed")
        if how == "infra":
            raise InfraError("busy", "the env at http://10.1.2.3:18100 is busy")
        if how == "ended":
            raise EpisodeEnded("the env closed")
        if how == "long":
            raise ToolError("no " * 100)
        raise RuntimeError("bug near /home/someone/secret")

    @tool(kind="pure", tier="D")
    async def wrong(self, ctx: ToolContext) -> Ok:
        """Returns the wrong model."""
        return Found(obj="x")  # type: ignore[return-value]

    @tool(kind="pure", tier="D")
    async def delegate(self, ctx: ToolContext, name: str, private: bool = False) -> Found:
        """Find through another tool."""
        found = await ctx.call("secret" if private else "find", name=name)
        assert isinstance(found, Found)
        return found

    @tool(kind="pure", tier="P")
    async def secret(self, ctx: ToolContext, name: str) -> Found:
        """Privileged."""
        return Found(obj=f"truth-{name}")

    @tool(kind="perception", tier="D")
    async def judge(
        self,
        ctx: ToolContext,
        fail: list[str] = [],  # noqa: B006
        wait: float = 0.0,
        gave_up: bool = False,
    ) -> DoneReport:
        """Check whether the task looks done."""
        await anyio.sleep(wait)
        return DoneReport(passed=not fail, failing=fail, gave_up=gave_up)

    @tool(kind="execution", tier="D")
    async def trip(self, ctx: ToolContext, rationale: Rationale) -> ActionReceipt:
        """An action that moves the robot, then fails."""
        raise InfraError("busy", "the env stopped answering after the move")

    @tool(kind="pure", tier="D")
    async def pose(self, ctx: ToolContext) -> Ok:
        """Ask the env."""
        robot = await ctx.env.robot(ctx.env_episode)
        ctx.state["step"] = robot.step
        ctx.emit("asked", {"step": robot.step})
        return Ok()


SHOWN = ("find", "grasp", "push", "refuse", "jam", "ugly", "judge", "trip")
PROBED = extended_phases(
    dev_phases(["probe"]),
    {role: [f"probe.{name}" for name in SHOWN] for role in ROLES},
    after="mock",
    only=["s1", "s2", "p3"],
)
"""`TEST_PHASES` with every probe tool in phase `dev` and `SHOWN` in the others."""


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


async def open_ep(hub: Hub, stage: str = "dev", **options: Any) -> EpisodeId:
    return (await hub.open_episode(OpenEpisode(stage=stage, **options))).episode_id  # type: ignore[arg-type]


def error(outcome: CallOutcome) -> tuple[str | None, str]:
    assert outcome.is_error, outcome
    return outcome.error_code, outcome.content[0].text  # type: ignore[union-attr]


def text(outcome: CallOutcome) -> str:
    assert not outcome.is_error, outcome
    return " ".join(part.text for part in outcome.content if part.type == "text")


async def ended(hub: Hub, ep: EpisodeId) -> str | None:
    end = (await hub.state(ep)).ended
    return None if end is None else end.reason


@pytest.fixture
def probe() -> Probe:
    return Probe()


@pytest.fixture
async def probed(make_hub: Callable[..., Hub], probe: Probe) -> Any:
    async with make_hub(LocalProvider(probe), phases=PROBED) as hub:
        yield hub


async def test_steps_run_in_order_and_a_rejection_stops_the_call(
    make_hub: Callable[..., Hub], model: Model
) -> None:
    seen: list[str] = []

    def spy(step: Any) -> Any:
        async def wrapped(call: Call, *rest: Proceed) -> None:
            seen.append(step.__name__)
            await step(call, *rest)

        wrapped.__name__ = step.__name__
        return wrapped

    async def quota(call: Call) -> None:  # a guard added in front, as permits will be
        seen.append("quota")
        if call.request.name == "look":
            raise Reject("no_quota", "No quota left.")

    pipeline = Pipeline(
        pre=tuple(map(spy, DEFAULT_PIPELINE.pre)),
        guards=(quota, *map(spy, DEFAULT_PIPELINE.guards)),
        around=tuple(map(spy, DEFAULT_PIPELINE.around)),
        post=tuple(map(spy, DEFAULT_PIPELINE.post)),
    )
    async with make_hub(pipeline=pipeline) as hub:
        ep = await open_ep(hub, "s2")
        assert not (await hub.call(ep, model("act", rationale=rationale()))).is_error
        assert seen == [
            *("episode_open", "visibility", "validate", "resolve_aliases"),
            *("quota", "action_lock", "budget", "stage_rules", "retry_guard", "count_actions"),
            *("timeout", "metering"),
            *("check_output", "count_steps", "unmoved", "fail_fast", "recovery"),
            "give_up",
            "retry_record",
            *("done_state", "finish"),
            "sanitize",
            *("render", "action_notes"),
        ]
        seen.clear()
        assert error(await hub.call(ep, model("look"))) == ("no_quota", "No quota left.")
        assert seen == ["episode_open", "visibility", "validate", "resolve_aliases", "quota"]


async def test_ids_reach_models_as_aliases_and_come_back_as_ids(
    probed: Hub, probe: Probe, model: Model
) -> None:
    ep = await open_ep(probed)
    found = await probed.call(ep, model("find", name="mug"))
    assert text(found) == 'find: obj="obj_1", parts=["part_1"]'
    assert text(await probed.call(ep, model("find", name="mug"))) == text(found)  # stable
    cup = await probed.call(ep, model("find", name="cup"))
    assert text(cup).startswith('find: obj="obj_2"')
    grabbed = await probed.call(ep, model("grab", obj="obj_1", times=None))
    assert text(grabbed) == 'grab: obj="obj_1"'
    assert probe.grabbed == ["object-mug", "object-mug"]  # the tool saw the id
    for bad, why in [
        ("obj_9", "unknown object 'obj_9': name it by its alias (obj_1) or its role (none)"),
        ("part_1", "expected an alias like obj_1, got 'part_1'"),
    ]:
        code, message = error(await probed.call(ep, model("grab", obj=bad)))
        assert code == "invalid_arguments" and why in message
    status = await probed.state(ep)
    assert status.counters.rejected == 2 and status.counters.info_calls == 4


async def test_failures_become_error_outcomes_with_stable_codes(
    make_hub: Callable[..., Hub], probe: Probe, model: Model, caplog: pytest.LogCaptureFixture
) -> None:
    config = HubConfig(render=RenderConfig(max_text_chars=60))
    async with make_hub(LocalProvider(probe), config=config) as hub:
        ep = await open_ep(hub)
        with anyio.fail_after(2):
            assert error(await hub.call(ep, model("slow"))) == (
                "timeout",
                "The tool did not finish within 0.05 s.",
            )
        assert error(await hub.call(ep, model("boom", how="infra"))) == (
            "infra_error",
            "The tool is unavailable right now (busy); try again.",
        )
        assert error(await hub.call(ep, model("boom", how="grouped"))) == (
            "infra_error",
            "The tool is unavailable right now (not_ready); try again.",
        )
        assert error(await hub.call(ep, model("boom", how="mixed")))[0] == "internal_error"
        with caplog.at_level(logging.ERROR, "roboscript.hub"):
            assert error(await hub.call(ep, model("boom", how="bug"))) == (
                "internal_error",
                "The tool failed unexpectedly.",
            )
        assert "bug near /home/someone/secret" in caplog.text  # the details stay in the log
        assert error(await hub.call(ep, model("boom", how="long"))) == (
            "tool_error",
            ("no " * 100)[:60] + " ... (cut)",  # error text obeys max_text_chars too
        )
        assert error(await hub.call(ep, model("wrong"))) == (
            "internal_error",
            "The tool returned an invalid result.",
        )
        assert await ended(hub, ep) is None  # neither failures nor timeouts of tools end it
        outcome = await hub.call(ep, model("boom", how="ended"))  # the env says it is over
        assert error(outcome)[0] == "episode_ended" and outcome.episode_ended
        assert await ended(hub, ep) == "env_ended"
        counters = (await hub.state(ep)).counters
        assert (counters.errors, counters.rejected) == (8, 0)


class Gated(Toolset):
    """A tool that answers, or fails for infrastructure reasons, once its gate opens."""

    name = "gated"

    def __init__(self) -> None:
        self.gates = {"first": anyio.Event(), "second": anyio.Event()}

    @tool(kind="pure", tier="D")
    async def wait(self, ctx: ToolContext, gate: str, infra: bool) -> Ok:
        """Wait for a gate."""
        await self.gates[gate].wait()
        if infra:
            raise InfraError("not_ready", "the planner did not answer")
        return Ok()


@pytest.mark.parametrize(
    ("between", "cleared"),
    [
        ({"how": "long"}, False),  # the tool refused: a tool error
        ({"how": "nope"}, False),  # refused before the tool: invalid arguments
        ({"name": "cup"}, True),  # answered
        ({"rationale": rationale(), "outcome": "empty_grasp"}, True),  # a hard failure: answered
    ],
    ids=["tool error", "refused", "answered", "hard failure"],
)
async def test_only_a_call_answered_without_an_error_clears_an_infrastructure_error(
    make_hub: Callable[..., Hub], probe: Probe, model: Model, between: Any, cleared: bool
) -> None:
    """The strict rule: after a call failed as infrastructure, only a later
    call forwarded and answered without an error clears it (a hard-failure receipt is an
    answer); a tool error, a refusal and `finish` do not."""
    name = "boom" if "how" in between else "find" if "name" in between else "push"
    async with make_hub(LocalProvider(probe)) as hub:
        ep = await open_ep(hub)
        assert error(await hub.call(ep, model("boom", how="infra")))[0] == "infra_error"
        await hub.call(ep, model(name, **between))
        assert (await hub.call(ep, model("finish", note="done"))).episode_ended
        said = "boom: The tool is unavailable right now (busy); try again."
        assert hub.unanswered_infra(ep) == (None if cleared else said)


@pytest.mark.parametrize(
    ("stage", "rules", "ends"),
    [
        ("s2", None, True),
        ("s1", None, False),
        ("s1", PhaseRules(fail_fast=True), True),
        ("p3", None, False),  # P3 recovers
        ("p3", PhaseRules(fail_fast=True), True),  # with recover, fail-fast wins
    ],
)
async def test_fail_fast_ends_the_episode_on_a_hard_failure(
    hub: Hub, model: Model, stage: str, rules: PhaseRules | None, ends: bool
) -> None:
    ep = await open_ep(hub, stage, rules=rules)
    short = await hub.call(ep, model("act", rationale=rationale(), outcome="stopped_short"))
    assert not short.episode_ended  # not a hard failure
    failed = await hub.call(ep, model("act", rationale=rationale(), outcome="empty_grasp"))
    assert not failed.is_error and failed.episode_ended is ends
    status = await hub.state(ep)
    if ends:
        assert text(failed).endswith("Hard failure (empty_grasp): the episode has ended.")
        assert status.ended is not None and status.ended.reason == "fail_fast:empty_grasp@grasp"
        assert status.ended.note.startswith("act (call-")
    else:
        assert status.ended is None


async def test_an_unmoved_rejection_goes_on_under_continue_unmoved(
    probed: Hub, model: Model, events: EventWriter
) -> None:
    """Unhinted S1: a rejection that moved nothing does not end the episode, the
    first `unmoved_events` add turns, the next action that runs is logged as its fix, and a hard
    failure after anything moved still ends it."""
    rules = PhaseRules(fail_fast=True, continue_unmoved=True, unmoved_turns=2, unmoved_events=1)
    ep = await open_ep(probed, "s1", rules=rules, budgets=Budgets(max_turns=1))
    first = await probed.call(ep, model("refuse", rationale=rationale(), outcome="unreachable"))
    assert not first.episode_ended  # the last turn, but two more came with it
    assert "Rejected before anything moved (unreachable): the attempt goes on (2 more turns)" in (
        text(first)
    )
    second = await probed.call(ep, model("refuse", rationale=rationale(), outcome="plan_failed"))
    assert not second.episode_ended and "more turns" not in text(second)
    assert (await probed.call(ep, model("push", rationale=rationale("lift")))).episode_ended
    assert await ended(probed, ep) == "max_turns"  # 3 turns: 1 + 2 for the first event only
    logged = [e.data for e in read_events(events.path) if e.kind == EventKind.HUB_UNMOVED]
    assert [(d["phase"], d["name"]) for d in logged] == [
        ("rejected", "refuse"),
        ("rejected", "refuse"),
        ("fixed", "push"),
    ]
    assert logged[2]["after"] == logged[1]["call_id"] and logged[2]["stage"] == "lift"
    moved = await open_ep(probed, "s1", rules=rules)
    call = model("push", rationale=rationale(), outcome="unreachable")  # one step used
    assert (await probed.call(moved, call)).episode_ended
    assert await ended(probed, moved) == "fail_fast:unreachable@grasp"
    plain = await open_ep(probed, "s1", rules=PhaseRules(fail_fast=True))
    call = model("refuse", rationale=rationale(), outcome="unreachable")
    assert (await probed.call(plain, call)).episode_ended  # S2's rule: it ends


async def test_failure_turns_cap_an_attempt_after_its_first_moving_hard_failure(
    probed: Hub, model: Model
) -> None:
    """Hinted S1 (no fail-fast): a rejection that moved nothing leaves the budget; the first hard
    failure that moved anything leaves `failure_turns` more turns, and a later one does not
    extend them."""
    ep = await open_ep(probed, "s1", rules=PhaseRules(failure_turns=2))
    unmoved = await probed.call(ep, model("refuse", rationale=rationale(), outcome="unreachable"))
    assert not unmoved.episode_ended and "more turns" not in text(unmoved)
    failed = await probed.call(ep, model("act", rationale=rationale(), outcome="empty_grasp"))
    assert not failed.episode_ended
    assert text(failed).endswith(
        "Hard failure (empty_grasp): this attempt ends after at most 2 more turns."
    )
    again = await probed.call(ep, model("act", rationale=rationale(), outcome="lost_hold"))
    assert not again.episode_ended and "more turns" not in text(again)
    assert (await probed.call(ep, model("act", rationale=rationale()))).episode_ended
    assert await ended(probed, ep) == "max_turns"
    sooner = await open_ep(
        probed, "s1", rules=PhaseRules(failure_turns=5), budgets=Budgets(max_turns=2)
    )
    first = await probed.call(sooner, model("act", rationale=rationale(), outcome="empty_grasp"))
    assert "more turns" not in text(first)  # the budget already ends it sooner
    assert (await probed.call(sooner, model("act", rationale=rationale()))).episode_ended


async def test_p3_recovers_from_every_hard_failure(probed: Hub, model: Model) -> None:
    """P3 is error recovery, not a retry budget: the attempt goes on after any hard failure, as
    often as it happens; only the benchmark's step limit (or `finish`) ends it. (The retry guard,
    off here, only redirects identical retries:
    `test_p3_retry_guard_redirects_identical_retries`.)"""
    ep = await open_ep(probed, "p3", rules=PhaseRules(retry_guard=False))

    async def fail(tool: str, stage: str, reason: str = "plan_failed") -> str:
        outcome = await probed.call(ep, model(tool, rationale=rationale(stage), outcome=reason))
        assert not outcome.episode_ended
        return text(outcome)

    assert await fail("act", "grasp") == (
        "act: ok=false, reason=plan_failed, steps_used=1, look_steps=0, "
        'note="action 1 of this episode" Hard failure (plan_failed): the attempt goes on. Move to '
        "a safe pose (lift, retreat or back to the origin), look again and try again."
    )
    for reason in ("empty_grasp", "lost_hold", "unreachable", "empty_grasp"):
        assert "the attempt goes on" in await fail("push", "grasp", reason)  # no count, no cap
    assert (await probed.state(ep)).ended is None
    assert text(await probed.call(ep, model("act", rationale=rationale()))).startswith("act: ok")


async def test_under_a_done_check_finish_needs_it_to_pass_after_the_last_action(
    probed: Hub, model: Model
) -> None:
    ep = await open_ep(probed, "s2", rules=PhaseRules(done_check="judge"))
    assert error(await probed.call(ep, model("finish", note="done"))) == (
        "finish_refused",
        "Call judge after the last action first: finish needs it to pass.",
    )
    assert not (await probed.call(ep, model("judge", fail=["gap_z cup coaster"]))).is_error
    assert error(await probed.call(ep, model("finish", note="done"))) == (
        "finish_refused",
        "The last judge did not pass (gap_z cup coaster): act on what fails, check again, "
        "then finish.",
    )
    assert not (await probed.call(ep, model("judge"))).is_error
    assert not (await probed.call(ep, model("find", name="cup"))).is_error  # not an action
    assert not (await probed.call(ep, model("push", rationale=rationale()))).is_error
    refused = await probed.call(ep, model("finish", note="done"))
    assert error(refused)[1].startswith("Call judge after the last action first")
    assert not (await probed.call(ep, model("judge"))).is_error
    accepted = await probed.call(ep, model("finish", note="done"))
    assert not accepted.is_error and accepted.episode_ended
    assert await ended(probed, ep) == "finish"


async def test_p3_retry_guard_redirects_identical_retries(probed: Hub, model: Model) -> None:
    """P3's retry guard: an identical retry of a hard-failed call (the rationale aside) is
    refused while nothing changed; after `retry_after` hard failures in a stage it also needs a
    measurement since the last one; another call always goes through; the refusals are capped."""
    ep = await open_ep(probed, "p3")
    await probed.call(ep, model("find", name="mug"))  # obj_1, a measurement before any failure

    def push(stage: str = "grasp", plan: str = "act", **arguments: Any) -> ToolCall:
        why = {**rationale(stage), "plan": plan}
        return model("push", rationale=why, outcome="plan_failed", **arguments)

    first = await probed.call(ep, push())
    assert not first.is_error and "the attempt goes on" in text(first)
    again = await probed.call(ep, push(plan="try the same once more"))
    assert error(again)[0] == "retry_refused" and "nothing has changed since" in error(again)[1]
    assert not (await probed.call(ep, push("lift"))).is_error  # another stage: its own record
    moved = model("push", rationale=rationale("retreat"), outcome="ok")
    assert not (await probed.call(ep, moved)).is_error  # an action that worked: a change
    assert not (await probed.call(ep, push())).is_error  # the 2nd hard failure in grasp
    other = model("refuse", rationale=rationale(), outcome="unreachable")
    assert not (await probed.call(ep, other)).is_error  # another call: the 3rd
    assert not (await probed.call(ep, moved)).is_error  # a change, but no measurement
    late = await probed.call(ep, push())
    assert error(late)[0] == "retry_refused" and "measure the scene again" in error(late)[1]
    await probed.call(ep, model("find", name="mug"))  # a measurement
    assert not (await probed.call(ep, push())).is_error
    # Capped: after `retry_refusals` refusals in a stage, calls go through.
    capped = await open_ep(
        probed, "p3", rules=PhaseRules(recover=True, retry_guard=True, retry_refusals=1)
    )
    await probed.call(capped, model("find", name="mug"))
    await probed.call(capped, push())
    assert (await probed.call(capped, push())).error_code == "retry_refused"
    assert not (await probed.call(capped, push())).is_error
    off = await open_ep(probed, "p3", rules=PhaseRules(retry_guard=False))
    await probed.call(off, push())
    assert not (await probed.call(off, push())).is_error
    assert (await probed.state(ep)).ended is None


async def test_p3_gives_up_after_a_run_of_like_hard_failures(probed: Hub, model: Model) -> None:
    """`give_up_failures`: hard failures in a row with the same reason and stage, nothing taken
    or let go since, let `finish` end a P3 episode without a passing done check; another reason
    or a take starts over (and holds finish again); off without `recover`."""
    rules = PhaseRules(recover=True, retry_guard=False, done_check="judge", give_up_failures=3)
    ep = await open_ep(probed, "p3", rules=rules)

    def fail(outcome: str = "unreachable", stage: str = "grasp") -> ToolCall:
        return model("refuse", rationale=rationale(stage), outcome=outcome)

    for _ in range(2):
        assert "GIVE UP" not in text(await probed.call(ep, fail()))
    assert "GIVE UP" not in text(await probed.call(ep, fail("plan_failed")))  # another reason
    assert "GIVE UP" not in text(await probed.call(ep, fail(stage="lift")))  # another stage
    assert "GIVE UP" not in text(await probed.call(ep, fail(stage="lift")))
    assert error(await probed.call(ep, model("finish", note="x")))[0] == "finish_refused"
    third = await probed.call(ep, fail(stage="lift"))
    assert "GIVE UP ALLOWED: 3 actions in a row in stage lift" in text(third)
    assert (await probed.state(ep)).ended is None  # the attempt goes on
    took = model("push", rationale=rationale("lift"), holding=True)
    assert not (await probed.call(ep, took)).is_error  # a take: starts over, holds finish again
    assert error(await probed.call(ep, model("finish", note="x")))[0] == "finish_refused"
    for _ in range(3):
        await probed.call(ep, fail(stage="lift"))
    accepted = await probed.call(ep, model("finish", note="out of reach"))
    assert not accepted.is_error and await ended(probed, ep) == "finish"
    off = await open_ep(probed, "dev", rules=PhaseRules(give_up_failures=3))
    for _ in range(4):
        assert "GIVE UP" not in text(await probed.call(off, fail()))


async def test_service_calls_skip_the_pipeline_but_keep_the_tier(
    make_hub: Callable[..., Hub], probe: Probe, model: Model
) -> None:
    async with make_hub(LocalProvider(probe)) as hub:
        dev = await open_ep(hub)
        delegated = await hub.call(dev, model("delegate", name="mug"))
        assert text(delegated).startswith('delegate: obj="obj_1"')
        refused = await hub.call(dev, model("delegate", name="mug", private=True))
        assert error(refused)[0] == "internal_error"  # a tier-D episode reaches no tier-P tool
        privileged = await open_ep(hub, tier="P")
        assert "secret" in (await hub.state(privileged)).tools
        outcome = await hub.call(privileged, model("delegate", name="mug", private=True))
        assert text(outcome) == 'delegate: obj="obj_1"'  # each episode aliases on its own
        status = await hub.state(dev)  # the inner calls are not the model's
        assert (status.counters.info_calls, status.counters.errors) == (2, 1)


async def test_episode_api(hub: Hub, model: Model) -> None:
    session = await hub.open_episode(OpenEpisode(stage="p3", budgets=Budgets(max_turns=60)))
    assert session.rules == PhaseRules(recover=True, retry_guard=True, give_up_failures=20)
    assert session.budgets.max_turns == 60
    assert session.role == "actor"
    ep = session.episode_id
    await hub.call(ep, model("look"))
    await hub.call(ep, model("fail"))
    status = await hub.end_episode(ep, "success", "latched at step 57")
    assert status.ended is not None
    assert (status.ended.reason, status.ended.note) == ("success", "latched at step 57")
    again = await hub.end_episode(ep, "other")
    assert again.ended == status.ended  # the first end stands
    summary = await hub.close_episode(ep)
    assert summary.end == status.ended
    assert (summary.counters.info_calls, summary.counters.errors) == (2, 1)
    with pytest.raises(ProtocolError, match="closed"):
        await hub.close_episode(ep)
    with pytest.raises(ProtocolError, match="phase 's2' runs at tier D only"):
        await hub.open_episode(OpenEpisode(stage="s2", tier="P"))
    unstarted = Hub([LocalProvider(Mock())], blobs=hub.blobs, phases=TEST_PHASES)
    with pytest.raises(RuntimeError, match="not started"):
        await unstarted.open_episode(OpenEpisode(stage="s2"))
