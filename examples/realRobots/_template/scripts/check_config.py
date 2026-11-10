"""Check a robot's configuration offline: `python check_config.py configs/robot.yaml` (needs
`examples/realRobots/_template` on PYTHONPATH). It loads the configuration as the robot server
does (`robot_template.MyRobotConfig`, the robots package's schema), and prints what the server
would describe and the embodiment profile it derives. It touches no hardware."""

import argparse
from pathlib import Path

from robot_template import load_deployment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = load_deployment(parser.parse_args(argv).config)
    profile = config.profile()
    print(config.description().model_dump_json(indent=2))
    print(f"embodiment {profile.name}: {len(profile.arms)} arm(s), action_dim {profile.action_dim}")
    print("hardware settings (the backend's):", config.address, sorted(config.camera_devices))


if __name__ == "__main__":
    main()
