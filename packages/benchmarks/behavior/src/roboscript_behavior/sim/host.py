"""The BEHAVIOR env server: `python -m roboscript_behavior.sim.host --config <file> --ready-file
<path>` in the interpreter that has OmniGibson. The planes are the skeleton's
(`roboscript.envs.skeleton.SkeletonEnv`); this module only says what BEHAVIOR senses and builds
the env."""

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.capabilities import SensingCapabilities
from roboscript.envs.skeleton import SkeletonEnv, run_server
from roboscript_behavior.benchmark import BEHAVIOR
from roboscript_behavior.config import BehaviorConfig
from roboscript_behavior.sim.simulator import BehaviorSimulator


def make_env(config: BehaviorConfig, blobs: LocalBlobStore) -> SkeletonEnv:
    profile = BEHAVIOR.profile()
    sensing = SensingCapabilities(
        scene_identity=True,  # an instance index reproduces its scene
        measured_proprio=True,
        depth=True,
        cameras=list(profile.cameras),
        step_unit="action",
    )
    return SkeletonEnv(BEHAVIOR, config, BehaviorSimulator(config, profile), blobs, sensing)


def main() -> None:
    run_server(
        "Serve a BEHAVIOR activity on a control and an agent port.", BehaviorConfig, make_env
    )


if __name__ == "__main__":
    main()
