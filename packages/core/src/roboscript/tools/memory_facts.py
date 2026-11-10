"""The Memory's specs and numbers as an Actor episode's facts (`MEMORY_SPECS`, `ARM_SIDES`;
docs/concepts/perception.md): what lets an Actor's call name a spec instead of copying its
fields, and `plan_arms` give the arms its workflow's holds and a role's direction call for.
The episode driver and the P3 host seed every Actor episode with a Memory with them
(`spec_seed`), whatever its method."""

import json
import re
from typing import Final

from roboscript.contracts.hub import EpisodeSeed
from roboscript.memory import (
    GraspSpecParam,
    HingeSpecParam,
    MemoryDoc,
    PathSpecParam,
    PlaceSpecParam,
    SkillSpecParam,
)
from roboscript.memory.calls import parse_calls
from roboscript.memory.chains import arm_chains
from roboscript.memory.objects import object_arm_sides
from roboscript.memory.params import FITTED_SOURCE, NumberParam
from roboscript.tools.frames import ARM_SIDES, MEMORY_SPECS

_FITTED_AT: Final = re.compile(rf"{FITTED_SOURCE} (left|right) K[1-9][0-9]*$")
"""A spec fitted at a demonstration keypoint (`demo <arm> K<n>`)."""


def _spec_value(
    param: GraspSpecParam | PlaceSpecParam | HingeSpecParam | PathSpecParam,
) -> dict[str, object]:
    """A spec as an Actor's call names it: its value; a held path that names no arm and was fitted
    at a demonstration keypoint gets that keypoint's arm (so the other arm follows its mirror
    image; the Memory itself is unchanged)."""
    value: dict[str, object] = param.value.model_dump(mode="json")
    path = isinstance(param, PathSpecParam) and param.value.arm is None
    if path and (fitted := _FITTED_AT.match(param.source)):
        value["arm"] = fitted[1]
    return value


def spec_seed(doc: MemoryDoc | None, seed: EpisodeSeed | None) -> EpisodeSeed | None:
    """`seed` with the `MEMORY_SPECS` fact of `doc`'s grasp, place, hinge, path and skill specs (and
    its numbers, which `move_by` takes by name), and
    the `ARM_SIDES` fact of its OBJECTS' `arm side:` lines (unchanged without any): what lets an
    Actor's call name a spec instead of copying its fields, and `plan_arms` plan the workflow's
    holds (`memory.chains.arm_chains`) and give the arm a role's direction calls for."""
    params = {} if doc is None else doc.params
    specs: dict[str, dict[str, object]] = {
        name: {"kind": param.kind, "value": _spec_value(param)}
        for name, param in params.items()
        if isinstance(param, GraspSpecParam | PlaceSpecParam | HingeSpecParam | PathSpecParam)
    }
    specs |= {  # a written skill's parameters, which `run_skill` takes by name
        name: {"kind": param.kind, "skill": param.skill, "value": json.loads(param.value)}
        for name, param in params.items()
        if isinstance(param, SkillSpecParam)
    }
    numbers = {
        name: param.value
        for name, param in ({} if doc is None else doc.params).items()
        if isinstance(param, NumberParam)
    }
    sides = {} if doc is None else object_arm_sides(doc)
    facts: dict[str, object] = {}
    if specs or numbers:
        fact: dict[str, object] = {"specs": specs}
        if numbers:
            fact["numbers"] = numbers
        staged = stage_arms(doc)
        if staged:
            fact["stage_arms"] = staged
        chains = arm_chains(doc)
        if chains.holds:
            fact["chains"] = chains.record()
        facts[MEMORY_SPECS] = fact
    if sides:
        facts[ARM_SIDES] = {"roles": sides}
    if not facts:
        return seed
    seed = seed or EpisodeSeed()
    return seed.model_copy(update={"facts": {**seed.facts, **facts}})


def stage_arms(doc: MemoryDoc | None) -> dict[str, list[str]]:
    """The roles a workflow grasps with both arms in turn (a set-down and a regrasp by the other
    arm, a hand-over): role -> the arms of its grasps in workflow order, each grasp's arm the one
    its grasp spec was written for (`grasp_object(obj=<role>, grasp={spec})`); a role with one arm
    is left out. `plan_arms` keeps these per-stage arms against an instruction naming one arm."""
    found: dict[str, list[str]] = {}
    for stage in [] if doc is None else doc.workflow:
        for call in parse_calls(stage.call):
            role, name = call.keywords.get("obj"), call.keywords.get("grasp")
            if call.name != "grasp_object" or role is None or name is None:
                continue
            param = doc.params.get(name.strip().strip("{}"))  # type: ignore[union-attr]
            arm = param.value.arm if isinstance(param, GraspSpecParam) else None
            if arm is not None and arm not in found.setdefault(role.strip(), []):
                found[role.strip()].append(arm)
    return {role: arms for role, arms in found.items() if len(arms) > 1}
