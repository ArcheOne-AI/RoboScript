"""Serve the robot: `python serve_robot.py configs/server.yaml [--ready-file <path>]` (needs
`examples/realRobots/_template` on PYTHONPATH and `RS_ENV_TOKEN` in `.env`).

The robot server is `packages/robots`' (`roboscript_robots.base`: the env contract's required
planes plus the real-robot extensions, emergency stop, limits, heartbeat, manual reset). This
script runs it over `robot_template.MY_ROBOT` with the server config (`RobotServerConfig`): with
`backend: fake` it serves the robot in software; with `backend: hardware` it stops at the first
hardware point of `MyRobotBackend` that is not implemented yet."""

import argparse
import logging
from pathlib import Path

import anyio
from robot_template import MY_ROBOT

from roboscript.envs import control_token
from roboscript.settings import load_config_file
from roboscript_robots.base import RobotServerConfig, build_server, serve_robot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="a RobotServerConfig YAML file")
    parser.add_argument("--ready-file", type=Path, help="where to write the planes' URLs")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config_file(args.config, RobotServerConfig)
    token = control_token()
    anyio.run(serve_robot, build_server(config, MY_ROBOT), config.server, args.ready_file, token)


if __name__ == "__main__":
    main()
