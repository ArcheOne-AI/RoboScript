"""The env contract's server side and the benchmark extension point
(docs/concepts/embodied-core.md).

`Benchmark` describes an integration (`load_benchmark("my_package.benchmark:BENCHMARK")`),
`serve_env` mounts an env implementation's planes on two ports, `Episodes` and `Batches` hold the
episode bookkeeping every env shares, `reset_options` reads a benchmark's own reset options from a
spec, `BridgeEnv` serves the agent plane of an episode that a benchmark's own client drives, and
`roboscript.envs.mock.MockEnv` is the reference implementation that tests of the layers above
use. `check_capabilities` is the startup check of a method's `requires` against what the env
offers (`offered_by`).
"""

from roboscript.envs.benchmark import Benchmark, EvalSeeds, TaskInfo, load_benchmark
from roboscript.envs.bridge import BridgeConfig, BridgeEnv, Bridges
from roboscript.envs.capabilities import (
    MissingCapabilities,
    check_capabilities,
    discovered,
    offered_by,
)
from roboscript.envs.episodes import Batches, Episodes, reset_options, unsolvable
from roboscript.envs.server import ServerConfig, control_token, env_apps, serve_env

__all__ = [
    "Batches",
    "Benchmark",
    "BridgeConfig",
    "BridgeEnv",
    "Bridges",
    "Episodes",
    "EvalSeeds",
    "MissingCapabilities",
    "ServerConfig",
    "TaskInfo",
    "check_capabilities",
    "control_token",
    "discovered",
    "env_apps",
    "load_benchmark",
    "offered_by",
    "reset_options",
    "serve_env",
    "unsolvable",
]
