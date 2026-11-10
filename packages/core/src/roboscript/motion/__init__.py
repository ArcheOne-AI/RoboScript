"""Motion planning (docs/concepts/embodied-core.md): providers of the `Planner` interface.

`MockPlanner` is pure numpy, the planner of the `mock_bimanual` robot (MockEnv), for unit tests.
`CuroboPlanner` is a cuRobo adapter driven by an embodiment profile and runs as a
service, one per GPU, in the simulator env: `python -m roboscript.motion --config motion`
(`MotionConfig`). Callers reach the service with
`rpc.client(Planner, url, token=None, timeout=10, timeouts=contracts.motion.TIMEOUTS)`.
"""

from roboscript.motion.config import ArmConfig, ArmSetup, CuroboConfig
from roboscript.motion.curobo import CuroboPlanner
from roboscript.motion.mock import MockPlanner

__all__ = ["ArmConfig", "ArmSetup", "CuroboConfig", "CuroboPlanner", "MockPlanner"]
