"""The built-in pipeline steps, their guards (`GUARDS`) and `DEFAULT_PIPELINE`
(docs/concepts/agent-runtime.md).

Each step reads what the ones before it filled in: after `episode_open` a call has its episode,
after `visibility` its tool, after `validate` its params, after `check_output` its result.
"""

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Final, cast

from pydantic import BaseModel, ValidationError

from roboscript.contracts.events import EventKind
from roboscript.contracts.gateway import MAX_DEPTH, UNHASHABLE, args_hash
from roboscript.contracts.hub import ENDED_MESSAGE, Budgets, DoneReport
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.refs import ROLE
from roboscript.contracts.tools import TURNS, TextPart
from roboscript.hub.builtins import FINISH, Finished
from roboscript.hub.pipeline import (
    Around,
    Call,
    Guard,
    Phase,
    Proceed,
    Reject,
    Step,
    assemble,
    fixed,
)
from roboscript.hub.render import render_result
from roboscript.hub.store import EpisodeState
from roboscript.stack import stall
from roboscript.tools import IDENTIFIED, REFS, Rationale

INFO: Final = frozenset({"pure", "perception"})  # information calls
NOT_MEASURED: Final = frozenset({"observe", "get_robot_state"})
"""Perception calls that are no measurement of the scene for `PhaseRules.retry_guard`."""
NOT_PLAIN: Final = (
    f"Invalid arguments: use finite numbers, valid Unicode text and at most {MAX_DEPTH} levels "
    "of nesting."
)

log = logging.getLogger(__name__)


# pre


async def episode_open(call: Call) -> None:
    """The episode exists and has not ended; after the end every call gets one fixed error,
    except the runtime's calls of the system-only tools marked `after_end` until the episode
    closes (the reading of the scene an attempt ended in)."""
    store = call.hub.store
    if store.is_closed(call.episode_id):
        raise Reject("episode_ended", ENDED_MESSAGE)
    episode = store.find(call.episode_id)
    if episode is None:
        raise Reject("unknown_episode", "Unknown episode.")
    call.episode = episode
    name = call.request.name
    reading = call.request.origin == "system" and name in episode.system_tools
    if episode.ended is not None and not (reading and call.hub.tool(name).meta.after_end):
        raise Reject("episode_ended", ENDED_MESSAGE)


async def visibility(call: Call) -> None:
    """Only the episode's tools exist, and its system-only tools for system calls: a hidden tool
    looks exactly like a missing one."""
    name, episode = call.request.name, call.episode
    system = call.request.origin == "system" and name in episode.system_tools
    if name not in episode.tools and not system:
        raise Reject("unknown_tool", f"There is no tool named {name[:64]!r}.")
    call.tool = call.hub.tool(name)


async def validate(call: Call) -> None:
    """Arguments must be plain JSON (`plain_json`) and fit the params model: unknown keys are
    refused, and null for a field with a default means that default."""
    if not plain_json(call.request.arguments):
        raise Reject("invalid_arguments", NOT_PLAIN)
    try:
        call.params = call.tool.spec.params.model_validate(call.request.arguments)
    except ValidationError as error:
        raise Reject("invalid_arguments", _invalid(error)) from None


async def resolve_aliases(call: Call) -> None:
    """Aliases in the arguments become the ids they stand for, and so do the registered
    objects' roles where an object is meant; where a role is meant (`RoleRef`), an object's alias
    becomes its role. Unknown ones are refused."""
    spec = call.tool.spec
    if not spec.refs_in:
        return
    aliases = call.episode.aliases
    registry = await call.episode.frames.registry()
    roles: dict[str, list[str]] = {}
    for record in registry.objects.values():
        roles.setdefault(record.name, []).append(record.id)

    def resolve(kind: str, value: str) -> str:
        """An alias's id; an object's role stands for its alias, and an alias for a role."""
        if kind == ROLE:
            try:
                return registry.objects[aliases.resolve("obj", value)].name
            except (ValueError, KeyError):
                return value
        try:
            return aliases.resolve(kind, value)
        except ValueError as error:
            if kind != "obj" or not str(error).startswith("unknown alias"):
                raise
            if value in roles and len(roles[value]) == 1:
                return roles[value][0]
            if value in roles:  # a plural role (`order: rest`): which of its objects is meant
                ids = _in_order(call, value, roles[value])
                listed = ", ".join(aliases.alias("obj", obj) for obj in ids)
                raise ValueError(
                    f"{value!r} names {len(roles[value])} objects ({listed}): name the one meant "
                    "by its alias"
                ) from None
            known = ", ".join(sorted(roles)) or "none"
            raise ValueError(
                f"unknown object {value!r}: name it by its alias (obj_1) or its role ({known})"
            ) from None

    context = {REFS: resolve}
    try:
        params = call.params.model_dump(exclude_computed_fields=True)
        call.params = spec.params.model_validate(params, context=context)
    except ValidationError as error:
        raise Reject("invalid_arguments", _invalid(error)) from None


def _in_order(call: Call, role: str, registered: list[str]) -> list[str]:
    """A plural role's registered objects in its identification's order
    (`IdentifiedRole.instances`), then any registered under its name since."""
    fact = call.episode.frames.fact(IDENTIFIED) or {}
    listed = (fact.get("roles", {}).get(role) or {}).get("instances") or []
    first = [obj for obj in listed if obj in registered]
    return first + [obj for obj in registered if obj not in first]


# guards


async def action_lock(call: Call) -> None:
    """One turn-ending call at a time per episode: while an action runs, another action or
    `finish` is refused, not queued, since it would act on or end what is still changing."""
    if call.tool.meta.kind not in TURNS:
        return
    episode = call.episode
    if episode.action is not None:
        raise Reject("action_in_progress", "An action is still running; wait for its result.")
    episode.action = call
    call.defer(lambda: _release(episode, call))


async def budget(call: Call) -> None:
    """The wall clock bounds every call and ends the episode when it runs out; turns and
    information calls bound model calls."""
    episode, budgets = call.episode, call.episode.budgets
    remaining = call.hub.remaining(episode)
    if remaining is not None and remaining < 0:
        call.hub.end(episode, "wall_clock")
        raise Reject("budget_exhausted", _wall_clock(budgets))
    if call.request.origin != "model":
        return
    kind, counters = call.tool.meta.kind, episode.counters
    if kind in TURNS and budgets.max_turns is not None and counters.turns >= budgets.max_turns:
        raise Reject("budget_exhausted", f"All {budgets.max_turns} turns are used up.")
    info = budgets.max_info_calls
    if kind in INFO and info is not None and counters.info_calls >= info:
        message = f"All {info} information calls are used up; act or finish."
        raise Reject("budget_exhausted", message)


async def count_actions(call: Call) -> None:
    """The last guard: note the episode's action count as the call is admitted, then count an
    admitted action, whether or not it returns (the robot may have moved), for `done_state`."""
    episode = call.episode
    call.actions = episode.actions
    if call.tool.meta.kind == "execution":
        episode.actions += 1


async def stage_rules(call: Call) -> None:
    """Caps on model calls per tool (`PhaseRules.max_calls`), and arguments the stage's Memory
    call does not name (`PhaseRules.call_arguments`)."""
    name, episode = call.request.name, call.episode
    cap = episode.rules.max_calls.get(name)
    if cap is not None and call.request.origin == "model" and episode.tool_calls[name] >= cap:
        raise Reject("stage_rule", f"{name} may be called at most {cap} times in this episode.")
    stage = _stage(call)
    named = episode.rules.call_arguments.get(stage or "", {}).get(name)
    if named is None or call.request.origin != "model":
        return
    extra = invented_arguments(call.request.arguments, call.params, named)
    if (stage, name) in episode.unplanned:
        extra = [key for key in extra if key not in episode.rules.freed_arguments]
    if extra:
        raise Reject(
            "stage_rule",
            f"The Memory's call for stage {stage} passes {name} no {', '.join(extra)}: call it as "
            "the Memory writes it (an argument it does not name keeps its default).",
        )


def invented_arguments(given: dict[str, Any], params: BaseModel, named: list[str]) -> list[str]:
    """The optional arguments a call passed at other than their default (`params`: the call's
    validated model) that `named` (a Memory call's arguments) does not name."""
    fields = type(params).model_fields
    return [
        key
        for key, value in given.items()
        if value is not None
        and key not in named
        and key in fields
        and not fields[key].is_required()
        and getattr(params, key) != fields[key].get_default(call_default_factory=True)
    ]


async def retry_guard(call: Call) -> None:
    """Under `PhaseRules.retry_guard`, a model's physical call that repeats a hard-failed one is
    refused while nothing changed since (no information call, no action without a hard
    failure), and, once `retry_after` hard failures happened in its workflow stage, while
    nothing was measured since the stage's last one; at most `retry_refusals` times per stage."""
    episode, rules = call.episode, call.episode.rules
    if not rules.retry_guard or call.request.origin != "model":
        return
    if call.tool.meta.kind != "execution":
        return
    state, stage = episode.retry, _stage(call)
    failed = state.failed.get(stage, {})
    at = failed.get(_retry_key(call))
    if at is None or state.refused[stage] >= rules.retry_refusals:
        return
    if at == state.epoch:
        state.refused[stage] += 1
        raise Reject(
            "retry_refused",
            "This exact call failed hard and nothing has changed since: look again or measure "
            "first, change the call (another candidate, grasp, placement or arm), or move to a "
            "safe pose.",
        )
    if state.failures[stage] >= rules.retry_after and state.measured <= max(failed.values()):
        state.refused[stage] += 1
        where = "" if stage is None else f" in stage {stage}"
        raise Reject(
            "retry_refused",
            f"{state.failures[stage]} calls have failed hard{where}, this one among them: "
            "measure the scene again (re-measure or locate the object) before repeating it, or "
            "make another call (another candidate, grasp, placement or arm).",
        )


# around


async def timeout(call: Call, proceed: Proceed) -> None:
    """The tool's `timeout_s`, or what is left of the wall clock if that is less. The tool's own
    timeout is put off by the time the host stood still meanwhile (stalled on I/O or memory, at
    most `timeout_s` more: `roboscript.stack.stall`), since a frozen host is no failure of the
    tool; the wall clock is not. An action cut off midway (by either, or because the caller went
    away) ends the episode as `action_interrupted` and keeps the action lock: what the robot is
    doing is unknown."""
    episode, meta = call.episode, call.tool.meta
    remaining = call.hub.remaining(episode)
    deadline = meta.timeout_s if remaining is None else min(meta.timeout_s, remaining)
    by_clock = deadline < meta.timeout_s
    try:
        with stall.stretched(deadline, most=0.0 if by_clock else None) as scope:
            await proceed()
        if scope.cancelled_caught:
            raise TimeoutError
    except BaseException as error:
        if call.returned:
            raise
        clock_ran_out = by_clock and isinstance(error, TimeoutError)
        if clock_ran_out:  # a policy failure, named before the interruption it causes
            call.hub.end(episode, "wall_clock")
        if meta.kind == "execution":
            call.interrupted = True
            call.hub.end(episode, "action_interrupted", note=_origin(call))
        if clock_ran_out:
            raise Reject("budget_exhausted", _wall_clock(episode.budgets)) from None
        if call.interrupted and isinstance(error, TimeoutError):
            message = f"The action did not finish within {meta.timeout_s:g} s; the episode is over."
            raise Reject("timeout", message) from None
        raise


async def metering(call: Call, proceed: Proceed) -> None:
    """Count the call where budgets and stage rules look, end the episode after its last turn,
    and time the tool."""
    episode, kind = call.episode, call.tool.meta.kind
    counters = episode.counters
    if call.request.origin == "model":
        episode.tool_calls[call.request.name] += 1
        if kind in TURNS:
            counters.turns += 1
            if (
                episode.budgets.max_turns is not None
                and counters.turns >= episode.budgets.max_turns
            ):
                # Deferred past the post steps, so that a transition names the end first and a
                # rejection that moved nothing may add turns first (`unmoved`).
                call.defer(lambda: _out_of_turns(call))
        elif kind in INFO:
            counters.info_calls += 1
        else:
            counters.other_calls += 1
    else:
        counters.system_calls += 1
    started = call.hub.clock()
    try:
        await proceed()
    finally:
        call.duration_s = call.hub.clock() - started


# post


async def check_output(call: Call) -> None:
    """The result must be the tool's result model (remote toolsets send JSON)."""
    expected = call.tool.spec.result
    if isinstance(call.raw, expected):
        call.result = call.raw
        return
    try:
        call.result = expected.model_validate(call.raw)
    except ValidationError as error:
        log.warning("%s returned an invalid result: %s", call.tool.qualified, error)
        raise Reject("internal_error", "The tool returned an invalid result.") from None


async def count_steps(call: Call) -> None:
    """Add the env steps an action's receipts report used to the episode's count."""
    if call.tool.meta.kind == "execution":
        call.episode.counters.steps += sum(r.steps_used for r in _receipts(call.result))


async def unmoved(call: Call) -> None:
    """Under `PhaseRules.continue_unmoved` (with `fail_fast`), a rejection that moved nothing
    (`ActionReceipt.unmoved`) goes on: `fail_fast` passes it over, its result tells the model to
    change the call, and each of the first `unmoved_events` adds `unmoved_turns` turns. Each is
    logged (`hub.unmoved`, `rejected`), and so is the next action that runs without a hard
    failure (`fixed`): what changed the call."""
    episode, rules = call.episode, call.episode.rules
    if not (rules.fail_fast and rules.continue_unmoved) or call.tool.meta.kind != "execution":
        return
    receipt = _hard_failure(call.result)
    data = {"call_id": call.request.call_id, "name": call.request.name, "stage": _stage(call)}
    if receipt is None:
        if episode.unmoved_call is not None:
            reasons = [str(r.reason) for r in _receipts(call.result)]
            fixed = {**data, "phase": "fixed", "after": episode.unmoved_call, "reasons": reasons}
            fixed["arguments"] = (
                call.request.arguments if plain_json(call.request.arguments) else None
            )
            call.hub.note(episode.id, EventKind.HUB_UNMOVED, fixed)
            episode.unmoved_call = None
        return
    if not receipt.unmoved:
        return
    call.unmoved = True
    if receipt.reason in _UNPLANNED:
        episode.unplanned.add((_stage(call), call.request.name))
    episode.unmoved += 1
    episode.unmoved_call = call.request.call_id
    extra = rules.unmoved_turns if episode.unmoved <= rules.unmoved_events else 0
    budgets = episode.budgets
    if extra and budgets.max_turns is not None:
        episode.budgets = budgets.model_copy(update={"max_turns": budgets.max_turns + extra})
    rejected = {**data, "phase": "rejected", "reason": str(receipt.reason), "note": receipt.note}
    rejected |= {"n": episode.unmoved, "extra_turns": extra}
    if receipt.commanded is not None:  # where it could not go, for a reviewing role
        rejected |= {"arm": receipt.arm, "commanded": list(receipt.commanded.tcp.p)}
    rejected["arguments"] = call.request.arguments if plain_json(call.request.arguments) else None
    call.hub.note(episode.id, EventKind.HUB_UNMOVED, rejected)
    turns = f" ({extra} more turns)" if extra and budgets.max_turns is not None else ""
    text = (
        f"Rejected before anything moved ({receipt.reason}): the attempt goes on{turns}. Change "
        "the call before acting again: another approach, grasp or placement, a target the arm "
        "reaches, the other arm, or another tool. A hard failure after anything moved still ends "
        "the attempt."
    )
    call.content.append(TextPart(text=text))


_UNPLANNED: Final = frozenset({ReceiptReason.unreachable, ReceiptReason.plan_failed})


async def fail_fast(call: Call) -> None:
    """Under `PhaseRules.fail_fast`, a hard-failure receipt ends the episode as
    `fail_fast:<reason>@<stage>`, unless it moved nothing and goes on (`unmoved`). Without it,
    under `PhaseRules.failure_turns`, a hard failure that moved anything leaves at most that many
    more turns (the turn budget lowered, never raised)."""
    receipt = _hard_failure(call.result)
    rules, episode = call.episode.rules, call.episode
    if receipt is None or call.unmoved:
        return
    if rules.fail_fast:
        call.hub.end(episode, f"fail_fast:{receipt.reason}{_at(call)}", note=_origin(call))
        ended = f"Hard failure ({receipt.reason}): the episode has ended."
        call.content.append(TextPart(text=ended))
        return
    if rules.failure_turns is None or receipt.unmoved or episode.ended is not None:
        return
    budgets, limit = episode.budgets, episode.counters.turns + rules.failure_turns
    if budgets.max_turns is not None and budgets.max_turns <= limit:
        return  # an earlier failure (or the budget) already ends it sooner
    episode.budgets = budgets.model_copy(update={"max_turns": limit})
    text = (
        f"Hard failure ({receipt.reason}): this attempt ends after at most "
        f"{rules.failure_turns} more turns."
    )
    call.content.append(TextPart(text=text))


async def recovery(call: Call) -> None:
    """Under `PhaseRules.recover` (an evaluation phase's), a hard failure leaves the episode
    running, and its result says so: the attempt goes on from a safe pose. No count and no cap: the
    benchmark's step limit ends the episode."""
    receipt = _hard_failure(call.result)
    if receipt is None or not call.episode.rules.recover or call.episode.ended is not None:
        return
    text = (
        f"Hard failure ({receipt.reason}): the attempt goes on. Move to a safe pose (lift, "
        "retreat or back to the origin), look again and try again."
    )
    call.content.append(TextPart(text=text))


async def give_up(call: Call) -> None:
    """Under `PhaseRules.give_up_failures` (with `recover`): the model's physical calls that fail
    hard for the same reason in the same workflow stage, in a row and with no gripper taking or
    letting go of anything since (a receipt's `holding` changing), are counted; at that many,
    `finish` no longer waits for the done check (`EpisodeState.failures.gave_up`) and the result
    says so. A take or release starts over, and holds `finish` again."""
    episode, rules = call.episode, call.episode.rules
    cap = rules.give_up_failures
    if not (cap and rules.recover) or call.request.origin != "model":
        return
    if call.tool.meta.kind != "execution" or episode.ended is not None:
        return
    run = episode.failures
    changed = False
    for receipt in _receipts(call.result):
        held = None if receipt.gripper is None else receipt.gripper.holding
        if receipt.arm is not None and held is not None:
            changed |= run.holding.get(receipt.arm, False) != held
            run.holding[receipt.arm] = held
    if changed:
        run.key, run.count, run.gave_up = None, 0, False
    receipt = _hard_failure(call.result)
    if receipt is None:
        return
    key = (_stage(call), str(receipt.reason))
    run.count = run.count + 1 if key == run.key else 1
    run.key = key
    if run.count < cap or run.gave_up:
        return
    run.gave_up = True
    where = "" if key[0] is None else f" in stage {key[0]}"
    text = (
        f"GIVE UP ALLOWED: {run.count} actions in a row{where} have failed hard ({receipt.reason}) "
        "with nothing taken or let go since: what is being tried does not get there. Change "
        "what you try (another arm, approach or object) if the receipts show a way; if they show "
        "none (e.g. the target beyond either arm's reach), finish now: finish no longer waits for "
        "the done check, and ends the episode uncredited (only the official check credits it)."
    )
    call.content.append(TextPart(text=text))


async def retry_record(call: Call) -> None:
    """For `PhaseRules.retry_guard`: a model's hard-failed physical call is remembered under its
    workflow stage at the current epoch; a model's information call or action without a hard
    failure starts a new epoch, and a perception call other than `NOT_MEASURED` is a
    measurement."""
    episode = call.episode
    if not episode.rules.retry_guard or call.request.origin != "model":
        return  # the loop's own observation each turn is routine, not a look the model took
    state, kind = episode.retry, call.tool.meta.kind
    if kind == "execution":
        if _hard_failure(call.result) is not None:
            stage = _stage(call)
            state.failed.setdefault(stage, {})[_retry_key(call)] = state.epoch
            state.failures[stage] += 1
        else:
            state.epoch += 1
    elif kind in INFO:
        state.epoch += 1
        if kind == "perception" and call.request.name not in NOT_MEASURED:
            state.measured = state.epoch


async def done_state(call: Call) -> None:
    """Under `PhaseRules.done_check`, keep what its last call reported, with the action count
    when that call was admitted: a later action (even one that raised, or ran while the check
    did) makes it stale."""
    episode = call.episode
    check = episode.rules.done_check
    if check is not None and call.request.name == check:
        episode.done = cast(DoneReport, call.result)
        episode.done_actions = call.actions


async def finish(call: Call) -> None:
    """The built-in `finish` ends the episode with its note, unless the done check has not
    passed since the last action (`PhaseRules.done_check`; refused every time, or at most
    `PhaseRules.done_refusals` times when set, and under `PhaseRules.recover` worded as a failure
    to recover from: the attempt goes on, within the step limit; there a check that gave up,
    `DoneReport.gave_up`, or a run of like failures that did, `give_up`, no longer holds it) or
    the runner's finish guard
    refuses it (as often as it was set to, once by default)."""
    if call.tool.qualified != FINISH:
        return
    episode, note = call.episode, cast(Finished, call.result).note
    check, cap = episode.rules.done_check, episode.rules.done_refusals
    # A run of like failures gave up (`give_up`): the done check no longer holds finish.
    failed = episode.failures.gave_up and episode.rules.recover
    if (
        check is not None
        and call.request.origin == "model"
        and (cap is None or episode.done_refused < cap)
        and not failed
    ):
        if episode.done is None or episode.done_actions != episode.actions:
            episode.done_refused += 1
            raise Reject(
                "finish_refused",
                f"Call {check} after the last action first: finish needs it to pass.",
            )
        # under recovery only: a learning phase's reviewing role reads an unconfirmed finish as
        # loose criteria
        given_up = episode.done.gave_up and episode.rules.recover
        if not episode.done.passed and not given_up:
            failing = "; ".join(episode.done.failing)
            if episode.rules.recover:
                then = (
                    "the task is not done and the attempt goes on. Recover: move to a safe pose "
                    "(lift, retreat or back to the origin), look again, redo the step that "
                    f"failed, then {check} again before finish."
                )
            else:
                then = "act on what fails, check again, then finish."
            episode.done_refused += 1
            raise Reject("finish_refused", f"The last {check} did not pass ({failing}): {then}")
    if episode.finish_guard is not None and episode.finish_refusals < episode.finish_guard_times:
        refusal = await episode.finish_guard(note)
        if refusal is not None:
            episode.finish_refusals += 1
            raise Reject("finish_refused", refusal)
    call.hub.end(episode, "finish", note=note)


async def sanitize(call: Call) -> None:
    """Ids in the result become aliases; sessions that opted in, and the runtime's own calls,
    also get the result as JSON."""
    spec, result = call.tool.spec, call.result
    if spec.refs_out:
        context = {REFS: call.episode.aliases.alias}
        dump = result.model_dump(exclude_computed_fields=True)
        call.shown = spec.result.model_validate(dump, context=context)
    else:
        call.shown = result
    if call.episode.structured or call.request.origin == "system":
        call.view = call.shown.model_dump(mode="json")


async def render(call: Call) -> None:
    """Text and images for the model (`roboscript.hub.render`, within the episode's limits),
    then the transitions' notes."""
    hub = call.hub
    rendered = await render_result(call.tool.spec, call.shown, hub.blobs, call.episode.render)
    call.content = [*rendered, *call.content]


async def action_notes(call: Call) -> None:
    """The runner's notes after a physical action (`Hub.set_action_notes`), after its result."""
    notes = call.episode.action_notes
    if notes is not None and call.tool.meta.kind == "execution":
        call.content = [*call.content, *await notes()]


def _guards(phase: Phase, *steps: Step | Around) -> list[Guard]:
    return [Guard(name=step.__name__, phase=phase, make=fixed(step)) for step in steps]


GUARDS: Final = SimpleNamespace(
    **{
        guard.name: guard
        for guard in [
            *_guards("pre", episode_open, visibility, validate, resolve_aliases),
            *_guards("guards", action_lock, budget, stage_rules, retry_guard, count_actions),
            *_guards("around", timeout, metering),
            *_guards(
                "post",
                check_output,
                count_steps,
                unmoved,
                fail_fast,
                recovery,
                give_up,
                retry_record,
                done_state,
                finish,
                sanitize,
                render,
                action_notes,
            ),
        ]
    }
)
"""The built-in guards in pipeline order, by name: what `roboscript.guards` registers (entry
point `roboscript.hub.steps:GUARDS.<name>`)."""
DEFAULT_GUARDS: Final[tuple[str, ...]] = tuple(vars(GUARDS))
"""The built-in guards' names in pipeline order."""
DEFAULT_PIPELINE: Final = assemble(list(vars(GUARDS).values()))


def plain_json(arguments: dict[str, Any]) -> bool:
    """Whether tool-call arguments are plain JSON: finite numbers, valid Unicode, nested at most
    `MAX_DEPTH` levels. The event log and permits need them to be."""
    try:
        args_hash(arguments)
    except UNHASHABLE:
        return False
    return True


def _invalid(error: ValidationError) -> str:
    # Locations and messages; inputs are left out.
    problems = error.errors(include_url=False, include_context=False, include_input=False)
    text = "; ".join(f"{'.'.join(map(str, p['loc'])) or 'arguments'}: {p['msg']}" for p in problems)
    return f"Invalid arguments: {text}"


def _wall_clock(budgets: Budgets) -> str:
    return f"The {budgets.wall_clock_s:g} s wall-clock budget is used up; the episode has ended."


def _release(episode: EpisodeState, call: Call) -> None:
    if episode.action is call and not call.interrupted:
        episode.action = None


def _receipts(result: BaseModel) -> Iterator[ActionReceipt]:
    """The result if it is a receipt, else the receipts in its fields (alone, listed or
    mapped)."""
    if isinstance(result, ActionReceipt):
        yield result
        return
    for _, value in result:
        if isinstance(value, dict):
            value = list(value.values())
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(item, ActionReceipt):
                yield item


def _out_of_turns(call: Call) -> None:
    """End the episode once its turns are used up (the budget as it stands after the call)."""
    episode = call.episode
    limit = episode.budgets.max_turns
    if limit is not None and episode.counters.turns >= limit:
        call.hub.end(episode, "max_turns")


def _hard_failure(result: BaseModel) -> ActionReceipt | None:
    return next((receipt for receipt in _receipts(result) if receipt.hard_failure), None)


def _retry_key(call: Call) -> str:
    """A physical call without its rationale: what makes two calls the same."""
    params = call.params.model_dump(
        mode="json", exclude={"rationale"}, exclude_computed_fields=True
    )
    return f"{call.request.name}:{args_hash(params)}"


def _stage(call: Call) -> str | None:
    """The workflow stage the call's rationale names, if it has one."""
    rationale = getattr(call.params, "rationale", None)
    return rationale.stage if isinstance(rationale, Rationale) else None


def _at(call: Call) -> str:
    stage = _stage(call)
    return "" if stage is None else f"@{stage}"


def _origin(call: Call) -> str:
    return f"{call.request.name} ({call.request.call_id})"
