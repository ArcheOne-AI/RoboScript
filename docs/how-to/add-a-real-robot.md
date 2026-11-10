# How to add a real robot

A real robot is an env server built on `roboscript-robots`. It serves the env contract's required
planes plus real-robot extensions (e-stop, homing, limits, heartbeat). Copy
`examples/realRobots/_template/` to `examples/realRobots/<model>/` and rename the package.

## 1. Pick a configuration

Choose the shape that fits your robot:

| Shape | Config base | Arms |
|---|---|---|
| `SingleArmConfig` | one arm | `arm` |
| `DualArmConfig` | two arms | `left`, `right` |
| `MobileManipulatorConfig` | arms + base | any names |

Subclass it as `MyRobotConfig`, adding hardware-specific fields (controller address, camera
devices). Copy `configs/robot.example.yaml`, fill in values, and validate:

```bash
python scripts/check_config.py configs/robot.yaml
```

## 2. Write the embodiment profile

Generate it from the config (`CONFIG.profile_yaml()`) and save as `embodiments/<model>.yaml`.
A test keeps the config and the YAML equal.

## 3. Implement the backend

Subclass the config's backend skeleton (`SingleArmBackend`, etc.) as `MyRobotBackend`. Implement:

- `connect()` / `disconnect()` -- open and close hardware
- `measure()` -- current joint positions
- `capture(cameras)` -- RGB-D frames
- `forward_kinematics(joints)` -- joint positions to poses
- `command_joints(target)` -- send one target per control tick
- `halt()` -- immediate stop

Start with `backend: fake` in the server config to test without hardware.

## 4. Register

```toml
[project.entry-points."roboscript.robots"]
my_arm = "my_arm:MY_ARM"

[project.entry-points."roboscript.embodiments"]
my_arm = "my_arm:EMBODIMENT"

[project.entry-points."roboscript.benchmarks"]
my_arm = "my_arm:BENCHMARK"
```

## 5. Serve

```bash
python scripts/serve_robot.py configs/server.yaml
```

Safety rules (e-stop, limits, heartbeat watchdog) come from `roboscript-robots`.

## 6. Write skills (optional)

Add operation APIs in `skills.py`. The learning runner refuses a real-robot env, so an agent
drives the robot only through what you build on the server.

## 7. Test

```bash
uv run pytest tests/examples/real_robots
```

Run `RobotConformance` from `roboscript_robots.testing` over the fake backend.
