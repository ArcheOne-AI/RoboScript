"""wicl's done gate (`learning.runtime.FinishGate`; docs/concepts/learning.md): an Actor episode
whose Memory holds done criteria (`done_spec` parameters), or that knows the demonstration's
final state (`goal`), starts with them as the `DONE` fact and runs under `PhaseRules.done_check`,
so that its `finish` waits until `check_done` passes after the last action. With neither nothing
changes.

`goal` is the demonstration's final-state views (camera -> RGB blob: its last keyframe, head and
wrist cameras) and the live instruction; `check_done` compares the scene with them camera by
camera (`worldstate.DoneGoal`), and only a confident match passes. S2, `evaluate` and P3 pass
one whenever they have the task's demo package, whatever their `demo_context` shows the Actor:
the comparison is perception on demonstration data, no simulator privilege. A finish is never a
success by itself (only the simulator's latch is); the gate makes a false claim cost a refusal,
not the try. `end`, the demonstration's end as targets measured against each object it moved
(read on its seed at an S1 attempt's end, `END_TARGETS`), adds conditions with the correction of
each gap. S1 attempts are gated too (beside the official refusals, with the Memory's criteria
left out, since S1 writes them): by the views from the first attempt on, by the targets once an
attempt has read them. Wherever `rules.done_check` is set, the `DONE` fact is seeded
(`require_done_fact`)."""

import logging
from collections.abc import Collection
from typing import TYPE_CHECKING

from roboscript.agent.host import DemoView
from roboscript.contracts.errors import ProtocolError

if TYPE_CHECKING:
    from roboscript.learning.episode import Episode


from collections.abc import Mapping, Sequence
from typing import Final

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.env import EnvControl
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.ids import EpisodeId, PhaseName, TaskName, ToolName
from roboscript.contracts.phase import PhaseRules
from roboscript.contracts.robot import Arm
from roboscript.hub import FinishGuard
from roboscript.memory import (
    DoneSpecParam,
    MemoryDoc,
)
from roboscript.perception.criteria import DoneSpec, EndTarget
from roboscript.tools.done import end_for
from roboscript.tools.frames import DONE
from roboscript_methods.wicl.extension import UNCONFIRMED, finish_guard, settings
from roboscript_methods.wicl.goal import demo_goal

log = logging.getLogger(__name__)

CHECK_DONE: Final = ToolName("check_done")


def done_criteria(doc: MemoryDoc) -> list[DoneSpec]:
    return [param.value for param in doc.params.values() if isinstance(param, DoneSpecParam)]


def done_gate(
    doc: MemoryDoc,
    rules: PhaseRules | None,
    seed: EpisodeSeed | None,
    goal: tuple[Mapping[str, BlobRef], str] | None = None,
    press: bool = False,
    end: Sequence[EndTarget] = (),
    reference: BlobRef | None = None,
    latched: Sequence[str] = (),
    criteria_read: bool = True,
    latch: Mapping[Arm, Vec3] | None = None,
) -> tuple[PhaseRules | None, EpisodeSeed | None]:
    """`rules` and `seed` with the done gate added when `doc` has criteria or there is a `goal`;
    only the fields the runner gave stay set, so the stage's other defaults still apply.
    `press`: the demonstration pushed something with empty fingers
    (`goal.ends_pressed`), so `check_done` also wants the episode's own press.
    `end`: the demonstration's end as targets (`perception.endstate`), which also gate it.
    `reference`: the demonstration's final frame (its `Observation`), whose table a lift is
    measured against in the robot's base frame (`worldstate.DoneFact.reference`).
    `latched`: the roles whose lift latched the demonstration's success
    (`goal.latched_lifts`), so their lifts are minimums (`worldstate.DoneFact.latched`).
    `criteria_read` false: the episode reads no Memory criteria (S1 strips them, `doc` has
    none), so a line about a missing criterion names what does gate it
    (`worldstate.DoneFact.criteria_read`). `latch`: the arms still moving what they held into
    the demonstration's success latch, with that motion (`goal.latch_motion`), what
    `check_done` names to an Actor that makes no progress (`worldstate.DoneFact.latch`)."""
    criteria = done_criteria(doc)
    if not criteria and goal is None and not end:
        return rules, seed
    given = {} if rules is None else {name: getattr(rules, name) for name in rules.model_fields_set}
    fact: dict[str, object] = {"criteria": [spec.model_dump(mode="json") for spec in criteria]}
    if goal is not None:
        views, instruction = goal
        fact["goal"] = {
            "views": {c: ref.model_dump(mode="json") for c, ref in views.items()},
            "instruction": instruction,
        }
    if press:
        fact["press"] = True
    if end:
        fact["end"] = [target.model_dump(mode="json") for target in end]
    if reference is not None:
        fact["reference"] = reference.model_dump(mode="json")
    if latched:
        fact["latched"] = sorted(latched)
    if not criteria_read:
        fact["criteria_read"] = False
    if latch:
        fact["latch"] = {arm: list(d) for arm, d in sorted(latch.items())}
    seed = seed or EpisodeSeed()
    seeded = seed.model_copy(update={"facts": {**seed.facts, DONE: fact}})
    return PhaseRules(**{**given, "done_check": CHECK_DONE}), seeded


def require_done_fact(rules: PhaseRules | None, seed: EpisodeSeed | None) -> None:
    """Raise unless the finish gate and the `DONE` fact go together: rules that gate `finish` on
    `check_done` need the fact it checks against (else every check would fail with nothing to
    compare, and no finish could pass)."""
    gated = rules is not None and rules.done_check == CHECK_DONE
    if gated and (seed is None or DONE not in seed.facts):
        raise ValueError("finish is gated on check_done, but the episode has no done fact")


class DoneGate:
    """wicl's finish gate (`learning.runtime.FinishGate`): the functions above, applied as
    S1, S2 and P3 open their Actor episodes. An Actor episode whose Memory has done criteria but
    that knows no demonstration end, on a stack without `check_done`, runs ungated, its criteria
    never checked, the official check alone deciding. The gate warns of it once per task and
    phase, as the first such episode opens."""

    def __init__(self) -> None:
        self._unchecked: set[tuple[TaskName, PhaseName]] = set()  # warned of, by task and phase

    def episode(
        self,
        episode: "Episode",
        rules: PhaseRules | None,
        seed: EpisodeSeed | None,
        *,
        instruction: str | None,
        actor: bool,
        mounted: Collection[ToolName],
    ) -> tuple[PhaseRules | None, EpisodeSeed | None, bool]:
        checkable = CHECK_DONE in mounted
        ending = demo_goal(episode.goal)  # wicl's own (`goal.DemoGoal`): the core passes it
        known = ending.views is not None or bool(ending.end)
        if actor and episode.memory is not None and known and not checkable:
            raise ValueError(
                f"{episode.directory}: the demonstration's end is known but {CHECK_DONE} is not "
                "mounted: mount the done toolset, or finish would not be gated"
            )
        if actor and episode.memory is not None and not checkable:
            self._warn_unchecked(episode, episode.memory)
        gated = False
        if actor and episode.memory is not None and checkable:
            shown = ending.views
            goal = None if shown is None or instruction is None else (shown, instruction)
            s1 = episode.stage == "s1"
            # S1 writes the criteria: it is gated by the demonstration's end alone (its
            # last keyframe's views, and its end targets once an attempt has read them)
            doc = episode.memory.model_copy(update={"params": {}}) if s1 else episode.memory
            rules, seed = done_gate(
                doc,
                rules,
                seed,
                goal,
                ending.press,
                ending.end,
                ending.reference,
                ending.latched,
                criteria_read=not s1,
                latch=ending.latch,
            )
            gated = rules is not None and rules.done_check is not None
            if s1 and gated:  # capped as the official refusals are
                assert rules is not None
                given = {name: getattr(rules, name) for name in rules.model_fields_set}
                rules = PhaseRules(**{**given, "done_refusals": episode.finish_refusals})
            demo = goal is not None or bool(ending.end)
            if demo and not gated:
                raise ValueError("a demonstration's end is known, but finish is not gated")
            require_done_fact(rules, seed)
        return rules, seed, gated

    def _warn_unchecked(self, episode: "Episode", memory: MemoryDoc) -> None:
        """Warn, once for the episode's task and phase, that its Memory's done criteria go
        unchecked: S1 reads none (it writes them), so only a later phase's can."""
        key = (episode.task, episode.stage)
        if episode.stage == "s1" or key in self._unchecked or not done_criteria(memory):
            return
        self._unchecked.add(key)
        log.warning(
            "%s, phase %s: the Memory has done criteria but no demonstration end state is known "
            "and %s is not mounted, so finish is not gated and the criteria are never checked "
            "(the official check alone judges completion); mount the done toolset to check them",
            episode.task,
            episode.stage,
            CHECK_DONE,
        )

    def guard(
        self, episode: "Episode", control: EnvControl, env_episode: EpisodeId
    ) -> FinishGuard | None:
        # S1, and S2 under its latch refusals: the privileged guard (`extension.not_done`)
        return finish_guard(episode, control, env_episode)

    def unconfirmed(self, episode: "Episode", gated: bool) -> str | None:
        # S2 under done criteria; S1 past its official refusals: a finish the simulator does not
        # confirm ends the attempt as a failure, never as a finish let through (S2's latch
        # refusals alone leave an ungated try's finish as it was)
        s1 = episode.stage == "s1"
        refused = (gated and not s1) or (settings(episode.method).official_finish and s1)
        return UNCONFIRMED if refused else None

    def host(
        self,
        doc: MemoryDoc,
        rules: PhaseRules | None,
        demo: DemoView | None,
        instruction: str,
        positional: bool,
    ) -> tuple[PhaseRules | None, EpisodeSeed | None]:
        known = demo_goal(None if demo is None else demo.goal)  # nothing known without a demo
        goal = None if known.views is None else (known.views, instruction)
        seed = None if demo is None else demo.seed
        end = [] if demo is None else end_for(doc, known.end, positional)
        rules, seed = done_gate(
            doc,
            rules,
            seed,
            goal,
            known.press,
            end,
            known.reference,
            known.latched,
            latch=known.latch,
        )
        require_done_fact(rules, seed)
        if goal is not None and (rules is None or rules.done_check is None):
            detail = "a demonstration's end is known, but finish is not gated"
            raise ProtocolError("invalid_request", detail)
        return rules, seed
