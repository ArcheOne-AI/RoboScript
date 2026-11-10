"""`skill_builder` registers as a method and loads as a P1 builder (`check_builder`)."""

import pytest

from roboscript.contracts.skills import SkillSet
from roboscript.learning.builder import DERIVE, VERIFY, check_builder, load_builder

pytestmark = pytest.mark.unit


async def test_it_registers_as_a_builder_that_is_not_wired_yet() -> None:
    method = load_builder("skill_builder")
    assert check_builder(method) == []
    assert method.artifact is SkillSet
    assert [p.name for p in method.phases] == [DERIVE, VERIFY]
    assert [p.kind for p in method.phases] == ["learn", "learn"]
    assert [r.name for r in method.roles] == ["builder"]
    assert all(p.visibility == {"builder": []} for p in method.phases)
    builder = method()  # type: ignore[call-arg]
    with pytest.raises(NotImplementedError, match=r"p1/build\.py"):
        builder.jobs(["a"], object())  # type: ignore[attr-defined]
    with pytest.raises(NotImplementedError, match=r"p1/build\.py"):
        await builder.learn(None, None)  # type: ignore[attr-defined]
