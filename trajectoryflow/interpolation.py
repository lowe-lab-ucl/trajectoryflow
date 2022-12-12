from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


SPATIAL_DIMS = frozenset({"x", "y", "z"})


def vectors_from_tracks(
    tracks: list, *, spatial_dims: str = "xy"
) -> np.ndarray:
    """Make an array of vectors from a list of tracks.

    Parameters
    ----------
    tracks : list
        A list of tracks.
    spatial_dims : str, default = "xy"
        The spatial dimensions to use.

    Returns
    -------
    vectors : array
        An array of vectors (NxD), where D is an even number. Data are stored
        as [xyuv] for 2D data, or [xyzuvw] for 3D, and so on.
    """

    if not all(d in SPATIAL_DIMS for d in spatial_dims):
        raise ValueError(
            f"Spatial dimensions ``{spatial_dims}`` not recognised."
        )

    vectors = []
    for track in tracks:
        track_data = [
            track.t,
        ] + [getattr(track, d) for d in spatial_dims]
        track_arr = np.stack(track_data, axis=-1)
        d = np.diff(track_arr, n=1, axis=0)

        # scale the vector by dt
        d[:, 1:] = d[:, 1:] * (1.0 / d[:, 0:1])

        # make the vector as [x, y, u, v]
        vec = np.concatenate(
            [
                track_arr[:-1, 1:],
                d[:, 1:],
            ],
            axis=-1,
        )
        vectors.append(vec)
    return np.concatenate(vectors, axis=0)


def _calculate_shepard_weights(
    x: np.ndarray, xy: np.ndarray, max_radius: float, power: int
) -> np.ndarray:
    """Calculate Shepard weights."""
    d_xy = xy - np.broadcast_to(x, xy.shape)
    d = np.linalg.norm(d_xy, axis=-1)
    weights = np.power(
        np.clip(max_radius - d, 0, np.inf) / (max_radius * d), power
    )
    return weights


def shepard_interp(
    vectors: np.ndarray,
    grid: np.ndarray,
    *,
    max_radius: float = 100.0,
    power: int = 2,
) -> np.ndarray:
    """Modified Shepard's method of interpolating vectors.

    Parameters
    ----------
    vectors : array
        An array of vectors (NxD), where D is an even number. Data are stored
        as [xyuv] for 2D data, or [xyzuvw] for 3D, and so on.
    grid : array
        An array of points (NxD), where D is the number of spatial dimenisons,
        at which to interpolate the vector field.
    max_radius : float
        The maximum radius over which to interpolate. In unites of the data.
    power : int
        An exponent used to scale the Shepard weights.

    Returns
    -------
    interpolated : array
        An array of interpolated vectors.
    """

    xy, uv = np.split(vectors, 2, axis=1)
    tree = KDTree(xy)
    queries = tree.query_ball_point(grid, max_radius)
    interpolated = np.zeros(grid.shape)
    nq = grid.shape[0]

    for idx, query in tqdm(enumerate(queries), desc="Interpolation", total=nq):
        if not query:
            continue

        # grid point to interpolate to
        x = grid[idx, ...]

        # nearest real locations
        x_xy = xy[query, ...]

        # vectors at the real locations
        x_uv = uv[query, ...]

        # calculate the weights and interpolate the vectors
        weights = _calculate_shepard_weights(x, x_xy, max_radius, power)
        sum_x = np.sum(weights)
        sum_u = np.sum(weights[:, np.newaxis] * x_uv, axis=0)
        interpolated[idx, ...] = sum_u / sum_x

    return interpolated
