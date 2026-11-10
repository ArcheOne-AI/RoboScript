"""roboscript.skills: bindings and the skillset interface check (docs/how-to/add-a-skillset.md)."""

import pytest

from roboscript.contracts.skills import Bindings, SkillSet, SkillSpec
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.skills import MountCheck, check_interface, check_mounts
from roboscript.testing.skills import (
    MOCK_EXECUTOR,
    PluginTarget,
    SkillSetConformance,
    SkillSetPluginConformance,
    SkillSetTarget,
)
from roboscript.tools.skillsets import mounted
from roboscript.toolsets.manipulation import Manipulation

CHAIN = ["parameters", "planning", "executor", "env_execute", "receipt"]


def _skill(name: str, **extra: object) -> SkillSpec:
    return SkillSpec.model_validate(
        {"name": name, "description": "d", "parameters": {"type": "object"}, "chain": CHAIN} | extra
    )


def _skillset(**extra: object) -> SkillSet:
    return SkillSet.model_validate(
        {
            "name": "manipulation",
            "version": "1",
            "layer": "core",
            "skills": [
                _skill("move_by"),
                _skill("together", needs={"min_arms": 2}),
                _skill("place_object", bindings=["planner_profile", "executor"]),
            ],
        }
        | extra
    )


def test_bindings_list_what_they_provide() -> None:
    assert Bindings().provided() == frozenset()
    assert Bindings(executor=MOCK_EXECUTOR).provided() == {"executor"}


def test_interface_check_names_what_does_not_mount() -> None:
    profile = load_embodiment(MOCK_BIMANUAL)
    assert check_interface(_skillset(), profile, {"executor"}) == [
        "place_object: no binding for ['planner_profile']"
    ]
    single = _skillset(skills=[_skill("together", needs={"min_arms": 3})], embodiments=["other"])
    assert check_interface(single, profile, set()) == [
        f"manipulation: fits ['other'], not {profile.name!r}",
        f"together: needs 3 arms, {profile.name} has 2",
    ]


class TestSkillSetConformance(SkillSetConformance):
    @pytest.fixture
    def skillset_target(self) -> SkillSetTarget:
        skillset = _skillset(skills=[_skill("together", needs={"min_arms": 2})])
        return SkillSetTarget(skillset, load_embodiment(MOCK_BIMANUAL), Bindings())


class TestCoreSkillSet(SkillSetPluginConformance):
    @pytest.fixture
    def plugin_target(self) -> PluginTarget:
        hosts = {"manipulation": Manipulation}
        # the Mock benchmark's skillset binds what the core's entries need
        return PluginTarget("manipulation", load_embodiment(MOCK_BIMANUAL), hosts, ["mock"])


class TestMockSkillSet(SkillSetPluginConformance):
    @pytest.fixture
    def plugin_target(self) -> PluginTarget:
        return PluginTarget("mock", load_embodiment(MOCK_BIMANUAL), {"manipulation": Manipulation})


def test_an_entry_without_its_binding_is_not_mounted_and_the_precheck_says_why() -> None:
    bare = mounted(Manipulation, [])
    assert not bare.specs and set(bare.unmounted) == {s.name for s in Manipulation.specs()}
    assert set(bare.unmounted.values()) == {"no mounted skillset binds ['executor']"}
    profile = load_embodiment(MOCK_BIMANUAL)
    toolsets = {"manipulation": Manipulation}
    assert check_mounts(["mock"], toolsets, profile) == MountCheck([], [])
    found = check_mounts([], toolsets, profile)
    line = "manipulation.grasp_object: not mounted, no mounted skillset binds ['executor']"
    assert line in found.unmounted
    assert found.problems == ["manipulation: mounts none of its tools"]  # as a provider refuses
    with pytest.raises(ValueError, match="two skillsets bind"):
        Bindings(executor=MOCK_EXECUTOR).merged(Bindings(executor=MOCK_EXECUTOR))
