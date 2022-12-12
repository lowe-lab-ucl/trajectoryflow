import numpy as np
import pytest


@pytest.fixture
def grid_xy():
    grid_x, grid_y = np.meshgrid([0, 1], [0, 1])
    xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    return xy
