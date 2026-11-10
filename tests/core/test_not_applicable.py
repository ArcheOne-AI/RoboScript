"""`not_applicable`: a conformance check that does not apply to its target is skipped under pytest
and reported as not applicable by `precheck` (also when `precheck` itself runs under pytest)."""

import pytest

from roboscript.testing import NOT_APPLICABLE, not_applicable, precheck


class Suite:
    async def test_applies(self, target: bool) -> None:
        assert target

    async def test_does_not_apply(self, target: bool) -> None:
        if target:
            not_applicable("the target declares no such face")

    async def test_fails(self, target: bool) -> None:
        raise AssertionError("broken")


async def test_precheck_reports_it_apart_from_passes_and_failures() -> None:
    assert await precheck(Suite(), True) == {
        "test_applies": None,
        "test_does_not_apply": f"{NOT_APPLICABLE}the target declares no such face",
        "test_fails": "AssertionError: broken",
    }


async def test_pytest_skips_it() -> None:
    with pytest.raises(pytest.skip.Exception, match="no such face"):
        await Suite().test_does_not_apply(True)
