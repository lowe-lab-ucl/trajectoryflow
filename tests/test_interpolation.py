import numpy as np
import pytest

from trajectoryflow.interpolation import (
    _calculate_distance_matrix,
    _calculate_shepard_weights,
)


def test_distance_matrix(grid_xy):
    """Test calculation of Euclidean distances."""
    x = np.array([[0.5, 0.5]])
    d = _calculate_distance_matrix(x, grid_xy)
    d_expected = [np.sqrt(0.5**2 + 0.5**2)] * grid_xy.shape[0]
    np.testing.assert_equal(d, d_expected)


@pytest.mark.parametrize(
    "power, expected", [(0, 1.0), (1, 0.41421356), (2, 0.17157288)]
)
def test_shephard_weights(grid_xy, power, expected):
    """Test calculation of shepard weights."""
    x = np.array([[0.5, 0.5]])
    w = _calculate_shepard_weights(x, grid_xy, 1.0, power)
    np.testing.assert_almost_equal(w, [expected] * grid_xy.shape[0])
