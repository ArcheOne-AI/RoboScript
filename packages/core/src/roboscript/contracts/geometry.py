"""Poses and boxes on the wire: meters, quaternions (w, x, y, z), world frame unless a field says
otherwise. The math lives in `roboscript.spatial`."""

import math
from typing import Annotated

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, FiniteFloat
from typing_extensions import Self

from roboscript.spatial.pose import pose_from_matrix, pose_to_matrix


def unit_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """`q` (w, x, y, z) normalized if its norm is within 1e-2 of 1, else a ValueError: the check
    of `Quat`, for validators of other shapes (a 7-D pose)."""
    norm = math.hypot(*q)
    if abs(norm - 1.0) > 1e-2:
        raise ValueError(f"a quaternion (w, x, y, z) needs unit norm, got norm {norm:.4g}")
    if abs(norm - 1.0) < 1e-12:  # already unit: keep it bit for bit, so JSON round trips are exact
        return q
    w, x, y, z = (v / norm for v in q)
    return w, x, y, z


Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Quat = Annotated[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat], AfterValidator(unit_quaternion)
]
"""A unit quaternion (w, x, y, z): norms within 1e-2 of 1 are normalized, others refused."""
_Length = Annotated[FiniteFloat, Field(ge=0)]


class Pose(BaseModel):
    """A rigid transform from a child frame into its parent: x_parent = R(q) x_child + p."""

    model_config = ConfigDict(frozen=True)

    p: Vec3
    q: Quat

    def as_list(self) -> list[float]:
        """The 7-vector [x, y, z, qw, qx, qy, qz]."""
        return [*self.p, *self.q]

    @classmethod
    def from_list(cls, values: ArrayLike) -> Self:
        """From a 7-vector [x, y, z, qw, qx, qy, qz] (list or array)."""
        x, y, z, qw, qx, qy, qz = np.asarray(values, dtype=float).tolist()
        return cls(p=(x, y, z), q=(qw, qx, qy, qz))

    def matrix(self) -> NDArray[np.float64]:
        """The 4x4 homogeneous transform."""
        return pose_to_matrix(self.as_list())

    @classmethod
    def from_matrix(cls, m: ArrayLike) -> Self:
        return cls.from_list(pose_from_matrix(m))


class OrientedBox(BaseModel):
    """A box with edge lengths `size` along its axes; q rotates box axes into the parent frame."""

    model_config = ConfigDict(frozen=True)

    center: Vec3
    size: tuple[_Length, _Length, _Length]
    q: Quat
