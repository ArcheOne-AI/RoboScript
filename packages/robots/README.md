# roboscript-robots

Interfaces for running RoboScript on real robots. A real robot is its own env server, serving the
env contract plus real-robot extensions (e-stop, homing, limits, heartbeat). The package contains
the server, configuration schemas, safety rules and a `FakeBackend`. No hardware backend ships.

Three configurations: `single_arm`, `dual_arm`, `mobile_manipulator`. Each has a schema, a backend
skeleton and a fake robot.

```bash
python -m roboscript_robots.base.server --config <server.yaml>
```

To add a robot, start from [examples/realRobots/_template](../../examples/realRobots/_template/README.md).
