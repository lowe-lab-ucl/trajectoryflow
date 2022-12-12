import numpy as np


def vectors_to_napari(
    grid: np.ndarray, interpolated: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert vectors to napari compatible vectors.

    Parameters
    ----------
    grid : array
    interpolated : array

    Returns
    -------
    vectors : array
    colors : array

    Usage
    -----
    >>> v, c = vectors_to_napari(grid, interp)
    >>> viewer = napari.Viewer()
    >>> viewer.add_vectors(
    ...     v, properties={"mag": c}, edge_color="mag", edge_colormap="turbo",
    ... )

    Notes
    -----
    From https://napari.org/stable/howtos/layers/vectors.html
    'The input data to the vectors layer must either be a Nx2xD numpy array
    representing N vectors with start position and projection values in D
    dimensions.'

    """
    # NOTE(arl): this might need to be changed - assumes tracks_to_vectors is
    # xyz, therefore flip lr to reverse xyz -> zyx for napari
    vectors = np.stack([np.fliplr(grid), np.fliplr(interpolated)], axis=1)
    colors = np.linalg.norm(interpolated, axis=-1)

    # TODO(arl): remove these debugging assertions
    assert vectors.ndim == 3
    assert vectors.shape[1] == 2
    assert colors.shape[0] == vectors.shape[0]

    return vectors, colors
