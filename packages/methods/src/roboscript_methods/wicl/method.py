"""`wicl`'s declaration (docs/how-to/add-a-method.md): its phases, roles, the tools each role
sees in each phase and at which tier, and each phase's default rules.

- `s1` (learn, `imitate`): the Actor (hinted and unhinted tries), the Demo Reader and the Reviser,
  every role up to tier P (privileged tools such as `true_objects`);
- `s2` (learn, `practice`): the Actor and the Reviser, D only as an eval phase is (`d_only`;
  the Reviser's ground-truth diagnosis is protocol input, not a tool);
- `p3` (eval, the Agent host): the Actor, tier D only;
- `dev` (learn): development and tests, every role up to tier P, the Actor with `check_reach`.

Each role's list is the order its model is shown the tools
(docs/concepts/agent-runtime.md). Tools a benchmark's skillset adds are named by category
(`manipulation.+`, `perception.+`), never by name: the method knows no benchmark's skills.
"""

import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from roboscript.contracts.artifact import Artifact
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.jobs import Evaluation, JobKind
from roboscript.contracts.phase import PhaseRules, PhaseSpec, RoleSpec
from roboscript.learning.runtime import MethodRuntime
from roboscript.orchestrate.jobs import RunKey
from roboscript_methods.wicl.diagnosis import explain_try
from roboscript_methods.wicl.extension import read_ends
from roboscript_methods.wicl.finish import CHECK_DONE, DoneGate
from roboscript_methods.wicl.goal import read_goal
from roboscript_methods.wicl.jobs import LearnJobs
from roboscript_methods.wicl.kinds import ImitateJob, PracticeJob
from roboscript_methods.wicl.memory.arms import unbound
from roboscript_methods.wicl.opening import Identification
from roboscript_methods.wicl.roles import Briefs, Prompts
from roboscript_methods.wicl.toolsets.demo import check_fitted

SENSING_ACTOR = [
    "sensing.observe",
    "sensing.get_image",
    "sensing.show_frames",
    "sensing.get_robot_state",
]
SENSING_ALL = [
    "sensing.observe",
    "sensing.get_image",
    "sensing.show_frames",
    "sensing.get_camera",
    "sensing.get_robot_state",
]
SENSING_FRAMES = ["sensing.get_image", "sensing.show_frames", "sensing.get_camera"]
ACTIONS_HEAD = [
    "manipulation.move_to_pose",
    "manipulation.move_by",
    "manipulation.open_gripper",
    "manipulation.close_gripper",
    "manipulation.return_home",
    "manipulation.grasp_object",
    "manipulation.place_object",
    "manipulation.+",  # the tools the benchmark's skillset adds to manipulation
    "manipulation.follow_path",
    "manipulation.together",
    "manipulation.nudge",
]
ACTIONS = [*ACTIONS_HEAD, "manipulation.plan_arms"]
ACTIONS_DEV = [*ACTIONS_HEAD, "manipulation.check_reach", "manipulation.plan_arms"]
"""`check_reach` (a plan without the action) is for development only: a refused action moves
nothing either, and its schema is among the largest a model reads."""
DEMO = ["demo.demo_relation", "demo.+"]  # +: fit_skill, where a stack mounts skill_fitting
PERCEPTION_ACTOR = ["perception.identify", "perception.ground", "perception.locate_part"]
PERCEPTION_ALL = [
    "perception.identify",
    "perception.locate",
    "perception.ground",
    "perception.segment",
    "perception.locate_part",
    "perception.remeasure",
    "perception.refine",
    "perception.look_pose",
    "perception.+",  # and to perception
]
PERCEPTION_REVIEW = ["perception.ground", "perception.segment"]
GEOMETRY_ALL = [
    "geometry.lift",
    "geometry.fit",
    "geometry.table",
    "geometry.project",
    "geometry.measure_pixel",
]
WORLD_ACTOR = [
    "worldstate.register",
    "worldstate.register_part",
    "worldstate.orient",
    "worldstate.check",
    "done.check_done",
]
TRY_END = ["done.read_end_targets", "done.read_relations"]
"""System-only: the demonstration's end as targets, and the record of an S1 or S2 try's end."""
WORLD_READER = [
    "worldstate.register",
    "worldstate.register_part",
    "worldstate.refresh",
    "worldstate.orient",
    "worldstate.relation",
    "worldstate.check",
]
WORLD_REVIEW = ["worldstate.relation", "worldstate.check"]
TRUTH = ["privileged.true_objects"]
MEMORY = ["memory.memory_read", "memory.memory_edit"]
FINISH = ["hub.finish"]
ACTOR_END = ["hub.finish", "hub.scratchpad"]


def _actor(actions: list[str], *, demo: bool, try_end: bool, truth: bool) -> list[str]:
    return [
        *SENSING_ACTOR,
        *actions,
        *(DEMO if demo else []),
        *PERCEPTION_ACTOR,
        "geometry.table",
        *WORLD_ACTOR,
        *(TRY_END if try_end else []),
        *(TRUTH if truth else []),
        *ACTOR_END,
    ]


def _reader(*, demo: bool, truth: bool, memory: bool) -> list[str]:
    return [
        *SENSING_ALL,
        *(DEMO if demo else []),
        *PERCEPTION_ALL,
        *GEOMETRY_ALL,
        *WORLD_READER,
        *(TRUTH if truth else []),
        *(MEMORY if memory else []),
        *FINISH,
    ]


def _reviser(*, memory: bool) -> list[str]:
    return [
        *SENSING_FRAMES,
        *PERCEPTION_REVIEW,
        *GEOMETRY_ALL,
        *WORLD_REVIEW,
        *(MEMORY if memory else []),
        *FINISH,
    ]


EVERY_ROLE_P = {"actor": "P", "demo_reader": "P", "reviser": "P"}

_S1_SEES = {
    "actor": _actor(ACTIONS, demo=True, try_end=True, truth=True),
    "demo_reader": _reader(demo=True, truth=True, memory=True),
    "reviser": _reviser(memory=True),
}
S1 = PhaseSpec(
    name="s1",
    kind="learn",
    # hinted tries go on after a hard failure; unhinted ones pass fail_fast
    rules=PhaseRules(),
    visibility=_S1_SEES,
    tiers=EVERY_ROLE_P,  # type: ignore[arg-type]
)
_S2_SEES = {
    "actor": _actor(ACTIONS, demo=False, try_end=True, truth=False),
    "demo_reader": _reader(demo=False, truth=False, memory=True),
    "reviser": _reviser(memory=True),
}
S2 = PhaseSpec(
    name="s2",
    kind="learn",
    rules=PhaseRules(fail_fast=True),
    visibility=_S2_SEES,
    d_only=True,  # what S2 learns must not need privilege (P3 runs on the same Memory)
)
P3 = PhaseSpec(
    name="p3",
    kind="eval",
    # error recovery: a hard failure does not end the episode; identical retries are redirected;
    # a run of like failures with nothing taken or let go lets finish end it uncredited
    rules=PhaseRules(recover=True, retry_guard=True, give_up_failures=20),
    visibility={"actor": _actor(ACTIONS, demo=False, try_end=False, truth=False)},
)
_DEV_SEES = {
    "actor": _actor(ACTIONS_DEV, demo=True, try_end=True, truth=True),
    "demo_reader": _reader(demo=True, truth=True, memory=False),
    "reviser": _reviser(memory=False),
}
DEV = PhaseSpec(
    name="dev",
    kind="learn",
    rules=PhaseRules(),
    visibility=_DEV_SEES,
    tiers=EVERY_ROLE_P,  # type: ignore[arg-type]
)


class Wicl:
    """Learn a task from one demonstration in context: S1 imitates it, S2 practises on new
    seeds, P3 evaluates the frozen Memory. Registered as `wicl` in `roboscript.methods`.
    The roles' prompts are read from the recipe (`roles.Prompts`, repository
    paths here), and nothing reads `RoleSpec.prompt`."""

    name: ClassVar[str] = "wicl"
    phases: ClassVar[list[PhaseSpec]] = [S1, S2, P3, DEV]
    roles: ClassVar[list[RoleSpec]] = [
        RoleSpec(name="actor", prompt="recipes/wicl/prompts/actor.md"),
        RoleSpec(name="demo_reader", prompt="recipes/wicl/prompts/demo_reader.md"),
        RoleSpec(name="reviser", prompt="recipes/wicl/prompts/reviser.md"),
    ]
    requires: ClassVar[EnvCapabilities] = EnvCapabilities(
        success=frozenset({"latch"}), reset=frozenset({"seed"}), truth=True, expert=True
    )
    """What it needs of an env (docs/concepts/embodied-core.md), which the `env_capabilities`
    precheck checks before a run starts: episodes reset by seed and judged by a success latch, the
    env's setup check that decides whether a seed counts (`expert`: an optional capability the env
    declares, such as an expert pre-run), and the ground-truth diagnosis S2's Reviser reads by
    default. The demonstration is a package file (the pipeline's `demo`), not an env face,
    so `demos` is not required."""
    artifact: ClassVar[type[Artifact]] = Artifact
    kinds: ClassVar[Mapping[str, JobKind]] = {"imitate": ImitateJob(), "practice": PracticeJob()}
    """Its jobs' bodies by phase (`kinds`; `jobs.PHASES`)."""

    def jobs(
        self,
        units: Sequence[str],
        evaluation: Evaluation,
        *,
        s1_only: bool = False,
        evaluate_only: bool = False,
        practice_again: Callable[[RunKey], bool] | None = None,
    ) -> LearnJobs:
        """A launch's jobs (`wicl.jobs`): each task's S1, then each unit's S2 and the
        evaluation of the Memory it froze; only S1s (`s1_only`), or only evaluations
        (`evaluate_only`); a unit that froze practises again when `practice_again` says so."""
        return LearnJobs(
            units,
            evaluation,
            s1_only=s1_only,
            evaluate_only=evaluate_only,
            practice_again=practice_again,
        )

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        """The one-time identification as each episode opens (`opening.Identification`), the
        done gate on its finish (`finish.DoneGate`), the roles' prompts from the recipe's
        templates (`roles.Prompts`, `<recipe>/prompts/`), the Actor's views of the
        demonstration (`roles.Briefs`), the Reviser's reading of a failed try's ground truth
        (`diagnosis.explain_try`), the reading of a judged try's end (`extension.read_ends`), a
        frozen Memory with its arm bindings rewritten by role (`memory.arms.unbound`), the
        demonstration's end (`goal.read_goal`) and the check that a spec edit keeps what
        `demo_relation` fitted (`toolsets.demo.check_fitted`); P3 needs `check_done` mounted
        (`mounts`; S1 and S2 refuse a gated episode without it as it opens, `finish.DoneGate`).
        The core's `evaluate` runs under S2's rules, the Agent host in P3."""
        return MethodRuntime(
            opening=Identification(),
            finish=DoneGate(),
            prompts=functools.partial(Prompts, recipe / "prompts"),
            demo=Briefs(),
            explain=explain_try,
            judged=read_ends,
            frozen=unbound,
            goal=read_goal,
            mounts={P3.name: (CHECK_DONE,)},  # every P3 episode with a demonstration is gated
            evaluate_phase=S2.name,  # `evaluate` tries a Memory as S2 does, without editing it
            host_phase=P3.name,
            edit_checks=(check_fitted,),
        )
