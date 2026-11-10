"""Waypoints: straight lines as pose samples, and the sparse points of a joint path to send.

`take_action(qpos)` moves an arm along the joint-space line from its drive target, so a planned
path is sent as a few of its points: the fewest, greedily, such that the TCP along each joint-space
line between consecutive ones stays within a tolerance of the planned TCP path. The TCP along a
joint line comes from forward kinematics, asked for in batches: one request per chosen waypoint.
"""

import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.geometry import Pose
from roboscript.spatial.pose import pose_interpolate
from roboscript.spatial.rotation import quat_angle_deg

TcpFK = Callable[[NDArray[np.float64]], Awaitable[NDArray[np.float64]]]
"""Joint vectors (m, dof) -> TCP positions (m, 3)."""

_SAMPLES_PER_RAD: Final = 50  # TCP samples along a joint line: one per 0.02 rad
_MIN_SAMPLES: Final = 3
_MAX_SAMPLES: Final = 40
_CANDIDATES: Final = 100  # ends tried per FK request: 100 x 40 samples fit one request


def line(start: Pose, goal: Pose, step_m: float, step_deg: float, max_points: int) -> list[Pose]:
    """Poses from `start` (excluded) to `goal` (included), at most `step_m` and `step_deg` apart
    (fewer if that would take more than `max_points`); position linear, orientation slerped."""
    distance = float(np.linalg.norm(np.subtract(goal.p, start.p)))
    angle = float(quat_angle_deg(start.q, goal.q))
    n = min(max_points, max(1, math.ceil(distance / step_m), math.ceil(angle / step_deg)))
    t = np.arange(1, n + 1) / n
    return [Pose.from_list(p) for p in pose_interpolate(start.as_list(), goal.as_list(), t)]


def polyline_distance(
    points: NDArray[np.float64], polyline: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Distance of each point (m, 3) to the polyline through the points (k, 3)."""
    if len(polyline) == 1:
        distance: NDArray[np.float64] = np.linalg.norm(points - polyline[0], axis=1)
        return distance
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    t = np.einsum("mkd,kd->mk", points[:, None] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12)
    nearest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    distances: NDArray[np.float64] = np.linalg.norm(points[:, None] - nearest, axis=2).min(axis=1)
    return distances


async def select(
    path: NDArray[np.float64], reference: NDArray[np.float64], fk: TcpFK, tolerance: float
) -> list[int]:
    """Indices of the waypoints of a joint `path` (n, dof) whose planned TCP positions are
    `reference` (n, 3): the last point always, the others greedily as far along as the joint line
    from the previous waypoint keeps its TCP within `tolerance` of the reference. Ends are tried
    on a grid of at most 100 points per request, so a waypoint may sit a little early."""
    n = len(path)

    async def deviations(i: int, ends: Sequence[int]) -> NDArray[np.float64]:
        """The largest distance of the TCP on the joint line i -> j to the reference, per j."""
        lines, owner = [], []
        for k, j in enumerate(ends):
            if j - i > 1:
                m = int(np.abs(path[j] - path[i]).max() * _SAMPLES_PER_RAD)
                f = np.linspace(0.0, 1.0, min(max(m, _MIN_SAMPLES), _MAX_SAMPLES) + 2)[1:-1]
                lines.append(path[i] + f[:, None] * (path[j] - path[i]))
                owner += [k] * len(f)
        worst = np.zeros(len(ends))
        if lines:
            tcp, of = await fk(np.concatenate(lines)), np.asarray(owner)
            for k, j in enumerate(ends):
                if j - i > 1:
                    worst[k] = polyline_distance(tcp[of == k], reference[i : j + 1]).max()
        return worst

    chosen: list[int] = []
    i = 0
    while i < n - 1:
        if (await deviations(i, [n - 1]))[0] <= tolerance:
            chosen.append(n - 1)
            break
        stride = math.ceil((n - 2 - i) / _CANDIDATES)
        ends = sorted({i + 1, *range(i + stride, n - 1, stride)})
        failing = np.flatnonzero(await deviations(i, ends) > tolerance)
        i = ends[failing[0] - 1] if len(failing) else ends[-1]  # ends[0] = i + 1 always passes
        chosen.append(i)
    return chosen
