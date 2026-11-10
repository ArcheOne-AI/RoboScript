"""Joint estimation from achieved TCP positions: a circle is fitted with its axis turning the way
the points go."""

import numpy as np
import pytest

from roboscript.manipulation.articulation import fit_circle

UP = np.array([0.0, 0.0, 1.0])


def arc(center: np.ndarray, radius: float, degrees: list[float], z: float = 0.0) -> np.ndarray:
    t = np.radians(degrees)
    return center + np.column_stack([radius * np.cos(t), radius * np.sin(t), np.full(len(t), z)])


def test_a_circle_is_fitted_with_its_axis_turning_the_way_the_points_go() -> None:
    center = np.array([0.1, -0.2, 0.8])
    fitted = fit_circle(arc(center, 0.3, [0, 8, 16, 24]), UP)
    assert fitted is not None
    point, direction, residual = fitted
    assert point == pytest.approx(center, abs=1e-6)
    assert direction == pytest.approx(UP, abs=1e-6) and residual < 1e-9
    _, backwards, _ = fit_circle(arc(center, 0.3, [24, 16, 8, 0]), UP)  # type: ignore[misc]
    assert backwards == pytest.approx(-UP, abs=1e-6)
