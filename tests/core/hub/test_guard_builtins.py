"""The guard conformance suite (`roboscript.testing.guards`) on every guard the core registers in
`roboscript.guards`, loaded through the entry point as a third-party package's would be: a name
per guard, no list to keep up to date."""

import pytest

from roboscript.testing import registered
from roboscript.testing.guards import GuardConformance

BUILTIN = registered("roboscript.guards")

pytestmark = pytest.mark.skipif(
    not BUILTIN, reason="roboscript is not installed here (the simulator env uses PYTHONPATH)"
)


def _suite(name: str) -> type[GuardConformance]:
    return type(f"Test_{name}", (GuardConformance,), {"guard_name": name, "__module__": __name__})


for _name in BUILTIN:
    globals()[f"Test_{_name}"] = _suite(_name)
