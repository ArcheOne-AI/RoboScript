"""The safety conventions of a real-robot server, as code (docs/how-to/add-a-real-robot.md).

`estop` takes effect before anything else the server is doing. It latches, halts the hardware
without waiting for a command in flight, and refuses every motion that is running or asked for
afterwards. It holds until the server is restarted, and nothing in the protocol clears it.

Before any motion of a batch, the server checks every step against the joint stops, the gripper
range and the workspace box on the TCP, and refuses the whole batch if one step would break one. A
step that would exceed the speed caps is stretched over more control ticks. Each tick the server
checks the measured TCP against the box again and halts the robot if it left.

While an episode is open, the client must call `heartbeat` more often than
`RobotLimits.heartbeat_timeout_s`. Silence halts the robot and ends the episode, as a fault that
the next `open` clears. Other calls do not count, because a client stuck in a loop can keep calling
`observe`.

`Safety` holds the two latches and the heartbeat clock, and `RobotServer` acts on what it says.
Time comes from an injected clock so the rules are tested without waiting.
"""

import time
from collections.abc import Callable
from typing import Final

from roboscript.contracts.robot_server import RobotLimits

MEASURED_MARGIN_M: Final = 0.02
"""How far past the workspace box a measured TCP may be before the robot is halted: tracking error
on real hardware. Planned waypoints get no margin."""
MAX_STEP_S: Final = 60.0
"""The longest one action may take at the speed caps; a step that would take longer is refused."""


class Safety:
    """The e-stop latch, the episode fault and the heartbeat clock of one robot server."""

    def __init__(self, limits: RobotLimits, now: Callable[[], float] = time.monotonic) -> None:
        self._timeout = limits.heartbeat_timeout_s
        self._now = now
        self._estop: str | None = None
        self._fault: str | None = None
        self._last_beat: float | None = None  # None: no episode is armed

    @property
    def estopped(self) -> str | None:
        """Why the e-stop latched (None: it has not)."""
        return self._estop

    def estop(self, reason: str) -> None:
        """Latch the e-stop; the first reason stays."""
        if self._estop is None:
            self._estop = reason

    def fault(self, reason: str) -> None:
        """Stop the open episode (a limit broken in motion); the first reason stays until the
        next `arm`."""
        if self._fault is None:
            self._fault = reason

    def arm(self) -> None:
        """An episode opens: clear the episode's fault and start the heartbeat clock."""
        self._fault = None
        self._last_beat = self._now()

    def disarm(self) -> None:
        """The episode is over: nobody has to send heartbeats, and its fault ends with it (the
        e-stop does not)."""
        self._last_beat = None
        self._fault = None

    def beat(self) -> None:
        """A heartbeat arrived. It keeps an armed episode alive; it does not revive one that
        already lapsed."""
        if self._last_beat is not None and self._fault is None:
            self._last_beat = self._now()

    def stopped(self) -> str | None:
        """Why the robot may not move now, in words (None: it may). Looks at the heartbeat
        clock, so a lapse is noticed whoever asks first."""
        if self._estop is not None:
            return f"emergency stop: {self._estop}"
        if (
            self._fault is None
            and self._last_beat is not None
            and self._now() - self._last_beat > self._timeout
        ):
            self._fault = f"no heartbeat for more than {self._timeout:g} s"
        return self._fault
