"""Phases for tests, and a way to show a method's roles the tools of a test or benchmark toolset.

- `TestMethod` (`TEST_PHASES`): a method declaration over the core's toolsets and its test
  toolsets (`mock`, `hold`, the protocols' `lifter`), for Hubs and runners in tests that are
  about the core rather than a method: phases `s1`, `s2` (learn), `p3` (eval) and `dev` (learn),
  roles `actor`, `demo_reader` and `reviser`, each tool shown to the roles of `TOOLS` in the
  order `TOOLS` lists them.
- `extend(method, add, after=...)`: `method` with more entries for some roles, placed after a
  toolset's entries: how a test stack shows a method's roles the Mock benchmark's toolsets;
  `dev_phases(toolsets)`: `TEST_PHASES` with every role of `dev` seeing all of `toolsets`' tools
  (a test's own toolsets, in development only).
"""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Final

from roboscript.contracts.artifact import Artifact
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.ids import PhaseName, RoleName
from roboscript.contracts.phase import PhaseRules, PhaseSpec, RoleSpec
from roboscript.embodiments import EmbodimentProfile
from roboscript.learning.method import Declaration
from roboscript.learning.runtime import MethodRuntime
from roboscript.memory import MemoryDoc, ToolSig, view

ROLES: Final = ("actor", "demo_reader", "reviser")
LEARN: Final = ("s1", "s2", "dev")
EVERY: Final = ("s1", "s2", "p3", "dev")
LIVE: Final = ("actor", "demo_reader")  # the roles on a live env
REVIEW: Final = ("demo_reader", "reviser")  # the roles that read material

TOOLS: Final[tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]] = (
    ("sensing.observe", EVERY, LIVE),
    ("sensing.get_image", EVERY, ROLES),
    ("sensing.show_frames", EVERY, ROLES),
    ("sensing.get_camera", EVERY, REVIEW),
    ("sensing.get_robot_state", EVERY, LIVE),
    ("mock.look", EVERY, ROLES),
    ("mock.act", EVERY, ROLES),
    ("mock.fail", EVERY, ROLES),
    ("hold.hold", EVERY, ("actor",)),
    # `lift` finds its box through the env's ground truth: a test oracle no real stack mounts. It
    # stays in the eval phase because the Memory linter's vocabulary is the tools every D-only
    # phase's Actor has (`learning.runner.actor_tools`), and the test Memories call `lift`.
    ("lifter.lift", ("s1", "s2", "p3"), ("actor",)),
    ("lifter.fault", ("s1", "s2", "p3"), ("actor",)),
    *(
        (f"manipulation.{name}", EVERY, ("actor",))
        for name in (
            "move_to_pose",
            "move_by",
            "open_gripper",
            "close_gripper",
            "return_home",
            "grasp_object",
            "place_object",
            "rotate_about",
            "follow_path",
            "together",
            "nudge",
        )
    ),
    ("manipulation.check_reach", ("dev",), ("actor",)),
    ("manipulation.plan_arms", EVERY, ("actor",)),
    ("demo.demo_relation", ("s1", "dev"), LIVE),
    ("perception.identify", EVERY, LIVE),
    ("perception.locate", EVERY, ("demo_reader",)),
    ("perception.ground", EVERY, ROLES),
    ("perception.segment", EVERY, REVIEW),
    ("perception.locate_part", EVERY, LIVE),
    *(
        (f"perception.{name}", EVERY, ("demo_reader",))
        for name in ("remeasure", "refine", "look_pose", "estimate_hinge", "measure_turn")
    ),
    ("geometry.lift", EVERY, REVIEW),
    ("geometry.fit", EVERY, REVIEW),
    ("geometry.table", EVERY, ROLES),
    ("geometry.project", EVERY, REVIEW),
    ("geometry.measure_pixel", EVERY, REVIEW),
    ("worldstate.register", EVERY, LIVE),
    ("worldstate.register_part", EVERY, LIVE),
    ("worldstate.refresh", EVERY, ("demo_reader",)),
    ("worldstate.orient", EVERY, LIVE),
    ("worldstate.relation", EVERY, REVIEW),
    ("worldstate.check", EVERY, ROLES),
    ("done.check_done", EVERY, ("actor",)),
    ("done.read_end_targets", LEARN, ("actor",)),
    ("done.read_relations", LEARN, ("actor",)),
    ("privileged.true_objects", ("s1", "dev"), LIVE),
    ("memory.memory_read", ("s1", "s2"), REVIEW),
    ("memory.memory_edit", ("s1", "s2"), REVIEW),
    ("hub.finish", EVERY, ROLES),
    ("hub.scratchpad", EVERY, ("actor",)),
)
"""Each tool, the phases and the roles that see it, in the order the roles are shown them."""
SKILLSET_TOOLS: Final = (
    "manipulation.rotate_about",
    "perception.estimate_hinge",
    "perception.measure_turn",
)
"""The entries naming a benchmark skillset's tools: optional, so that a test stack without it (the
Mock benchmark's) leaves them out."""


def _phase(name: str, kind: str, rules: PhaseRules, roles: Sequence[str], tier: str) -> PhaseSpec:
    visibility = {
        role: [tool for tool, phases, seen in TOOLS if name in phases and role in seen]
        for role in roles
    }
    return PhaseSpec.model_validate(
        {
            "name": name,
            "kind": kind,
            "rules": rules,
            "visibility": visibility,
            "tiers": dict.fromkeys(roles, tier),
            "optional": sorted(
                {t for entries in visibility.values() for t in entries} & set(SKILLSET_TOOLS)
            ),
        }
    )


TEST_PHASES: Final = [
    _phase("s1", "learn", PhaseRules(), ROLES, "P"),
    _phase("s2", "learn", PhaseRules(fail_fast=True), ROLES, "D"),
    _phase(
        "p3",
        "eval",
        PhaseRules(recover=True, retry_guard=True, give_up_failures=20),
        ROLES,
        "D",
    ),
    _phase("dev", "learn", PhaseRules(), ROLES, "P"),
]


class TestMethod:
    """The declaration of `TEST_PHASES`: what a Hub or runner in a core test runs."""

    __test__ = False  # not a pytest class
    name: ClassVar[str] = "test"
    phases: ClassVar[list[PhaseSpec]] = TEST_PHASES
    roles: ClassVar[list[RoleSpec]] = [RoleSpec(name=role, prompt=f"{role}.md") for role in ROLES]
    requires: ClassVar[EnvCapabilities] = EnvCapabilities()
    artifact: ClassVar[type[Artifact]] = Artifact

    @classmethod
    def runtime(cls, recipe: Path) -> MethodRuntime:
        """No open-time hook (nothing is identified as an episode opens), no finish gate, plain
        prompts (`PlainPrompts`), `evaluate` under `s2`'s rules and the Agent host in `p3`."""
        return MethodRuntime(prompts=PlainPrompts, evaluate_phase="s2", host_phase="p3")


class PlainPrompts:
    """A test method's prompts: the Actor's names the stage's rules and the robot's world axes,
    then the instruction under `# Task` and the Memory as the Actor reads it (what the test
    brains read, `testing.protocols`), in no template and without the mounted skillsets' guides;
    the digest is of the tools a Memory may call."""

    def __init__(self, tools: Mapping[str, ToolSig], guides: Sequence[str] = ()) -> None:
        self.tools = tool_listing(tools)
        self.digest = hashlib.sha256(self.tools.encode()).hexdigest()

    def actor(
        self, *, instruction: str, memory: MemoryDoc, rules: str, profile: EmbodimentProfile
    ) -> str:
        return "\n\n".join(
            [
                f"You are the Actor ({rules}). World axes: {profile.world_axes}.",
                f"# Task\n{instruction}",
                f"# Memory\n{view(memory, detail=False).strip()}",
            ]
        )

    library_digest = ""

    def library_actor(self, *, instruction: str, rules: str, profile: EmbodimentProfile) -> str:
        """In P3's library mode: the Memory's place says a library is offered instead."""
        return "\n\n".join(
            [
                f"You are the Actor ({rules}). World axes: {profile.world_axes}.",
                f"# Task\n{instruction}",
                "# Memory\nNone given: read and use the library's (library_index, library_use).",
            ]
        )


def tool_listing(tools: Mapping[str, ToolSig]) -> str:
    return "\n".join(f"{name}({', '.join(sig.params)})" for name, sig in sorted(tools.items()))


def extended_phases(
    phases: Sequence[PhaseSpec],
    add: Mapping[RoleName, Sequence[str]],
    *,
    after: str,
    only: Sequence[PhaseName] | None = None,
) -> list[PhaseSpec]:
    """`phases` with `add`'s entries for each of its roles a phase declares (of the phases in
    `only`, None: every one), placed after the role's last entry of toolset `after` (first,
    without one)."""
    out = []
    for phase in phases:
        if only is not None and phase.name not in only:
            out.append(phase)
            continue
        visibility = {}
        for role, entries in phase.visibility.items():
            extra = list(add.get(role, ()))
            at = max(
                (i + 1 for i, entry in enumerate(entries) if entry.startswith(f"{after}.")),
                default=0,
            )
            visibility[role] = [*entries[:at], *extra, *entries[at:]]
        out.append(phase.model_copy(update={"visibility": visibility}))
    return [PhaseSpec.model_validate(phase.model_dump()) for phase in out]


def extend(
    method: type[Declaration],
    add: Mapping[RoleName, Sequence[str]],
    *,
    after: str,
    only: Sequence[PhaseName] | None = None,
) -> type[Declaration]:
    """`method` with `extended_phases` in place of its phases."""
    phases = extended_phases(method.phases, add, after=after, only=only)
    return type(f"{method.__name__}Extended", (method,), {"phases": phases})


def dev_phases(
    toolsets: Sequence[str], *, base: Sequence[PhaseSpec] = TEST_PHASES, after: str = "mock"
) -> list[PhaseSpec]:
    """`base` with every role of phase `dev` seeing all of `toolsets`' tools, after `after`'s;
    toolsets `base` already names keep their entries."""
    named = {
        e.partition(".")[0] for p in base for entries in p.visibility.values() for e in entries
    }
    entries = [f"{name}.*" for name in toolsets if name not in named]
    return extended_phases(base, dict.fromkeys(ROLES, entries), after=after, only=["dev"])
