"""Real-robot interfaces for RoboScript (docs/how-to/add-a-real-robot.md): a robot server per
robot, implementing the env contract's required planes plus the real-robot extensions, by
configuration. No hardware backend is included.

- `base`: the robot server skeleton, the hardware seam, the fake backend, the safety conventions.
- `single_arm`, `dual_arm`, `mobile_manipulator`: a configuration schema, a hardware backend
  skeleton and a fake robot each. A robot package subclasses a skeleton for its model and
  registers a `RobotSpec` in `roboscript.robots`.
- `testing`: `RobotConformance`, the checks every registered robot passes over its fake backend.
"""

__version__ = "0.1.0.dev0"
