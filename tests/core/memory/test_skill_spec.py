"""A written skill's parameters in the Memory (`skill_spec`): stored with their skill, shown in the
Memory's text, merged whole (never interpolated), and given to an Actor's episode with the specs
(`MEMORY_SPECS`), where `run_skill` takes them by name."""

import pytest
from pydantic import TypeAdapter, ValidationError

from roboscript.memory import MemoryDoc, SkillSpecParam, merge
from roboscript.memory.doc import render_params
from roboscript.memory.params import ParamDef, SkillParamDef, categories, numbers
from roboscript.tools import MEMORY_SPECS
from roboscript.tools.memory_facts import spec_seed

FITTED = SkillSpecParam(
    skill="raised_contact_close",
    value='{"lift": 0.03, "approach": [0.0, 0.0, -1.0]}',
    source="demo left K3",
)


def with_skill(memory: MemoryDoc, param: SkillSpecParam) -> MemoryDoc:
    return memory.replace(params={**memory.params, "press_params": param})


def test_a_skill_spec_is_its_skill_and_params_taken_whole(memory: MemoryDoc) -> None:
    given = {"kind": "skill_spec", "skill": "raised_contact_close", "value": '{"lift": 0.03}'}
    defined = TypeAdapter(SkillParamDef).validate_python(given)
    assert defined.kind == "skill_spec"
    with pytest.raises(ValidationError):  # a stack that fits no skill is not offered the kind
        TypeAdapter(ParamDef).validate_python(given)
    for wrong in ({"kind": "skill_spec", "value": "{}"}, {**FITTED.model_dump(), "value": "[1]"}):
        with pytest.raises(ValidationError):
            TypeAdapter(SkillParamDef).validate_python(wrong)
    assert numbers(FITTED) == {} and categories(FITTED) == {None: FITTED.value}
    doc = with_skill(memory, FITTED)
    assert "skill_spec(raised_contact_close, params" in render_params(doc)
    tuned = FITTED.model_copy(update={"value": '{"lift": 0.05, "approach": [0.0, 0.0, -1.0]}'})
    merged = merge(doc, with_skill(memory, tuned), 0.5, seed=10001)
    assert merged.doc is not None
    assert merged.doc.params["press_params"].value == tuned.value  # replaced, not averaged


def test_an_actor_s_episode_gets_it_by_name_with_its_skill(memory: MemoryDoc) -> None:
    seed = spec_seed(with_skill(memory, FITTED), None)
    assert seed is not None
    specs = seed.facts[MEMORY_SPECS]["specs"]  # type: ignore[index]
    assert specs["press_params"] == {
        "kind": "skill_spec",
        "skill": "raised_contact_close",
        "value": {"lift": 0.03, "approach": [0.0, 0.0, -1.0]},
    }
