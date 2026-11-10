# Real-robot template

Copy this to `examples/realRobots/<model>/` and rename the `robot_template` package. Built on
`packages/robots` (`roboscript_robots`).

## Layout

```
configs/robot.example.yaml        Robot configuration (arms, cameras, limits)
configs/server.example.yaml       Server config (backend, operator)
embodiments/my_robot.yaml         Embodiment profile (generated from config)
robot_template/
├── deployment.py                 Config, backend, RobotSpec
└── skills.py                     Skillset skeleton
scripts/check_config.py           Validate config offline
scripts/serve_robot.py            Run the robot server
```

## Steps

1. **Configure.** Pick a base config (`SingleArmConfig`, `DualArmConfig` or
   `MobileManipulatorConfig`). Copy `configs/robot.example.yaml` to `configs/robot.yaml` and
   fill in your robot's values. Validate: `python scripts/check_config.py configs/robot.yaml`.

2. **Embodiment.** Write the profile YAML from the config
   (`EMBODIMENT.write_text(CONFIG.profile_yaml())`).

3. **Backend.** Implement `MyRobotBackend`'s hardware methods. Test first with `backend: fake`.

4. **Serve.** `python scripts/serve_robot.py configs/server.yaml`. Safety rules (e-stop, limits,
   heartbeat) come from the robots package.

5. **Skills.** Write operation APIs in `skills.py`.

6. **Test.** `uv run pytest tests/examples/real_robots`.
