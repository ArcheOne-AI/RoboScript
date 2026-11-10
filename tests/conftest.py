"""Repository-wide test setup."""

import pytest

_NOT_UNIT = ("sim", "gpu", "model")


@pytest.hookimpl(tryfirst=True)  # before `-m` deselects anything
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if not any(item.get_closest_marker(name) for name in _NOT_UNIT):
            item.add_marker(pytest.mark.unit)
