from __future__ import annotations

import numpy as np

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
