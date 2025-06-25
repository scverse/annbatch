from __future__ import annotations

import numpy as np


def sample_rows(
    x_list: list[np.ndarray], obs_list: list[np.ndarray], shuffle: bool = True
):
    """Samples rows from multiple arrays and their corresponding observation arrays.

    Args:
        x_list: A list of numpy arrays containing the data to sample from.
        obs_list: A list of numpy arrays containing the corresponding observations.
        shuffle: Whether to shuffle the rows before sampling. Defaults to True.

    Yields:
        tuple: A tuple containing a row from `x_list` and the corresponding row from `obs_list`.
    """
    lengths = np.fromiter((x.shape[0] for x in x_list), dtype=int)
    cum = np.concatenate(([0], np.cumsum(lengths)))
    total = cum[-1]
    idxs = np.arange(total)
    if shuffle:
        np.random.default_rng().shuffle(idxs)
    arr_idxs = np.searchsorted(cum, idxs, side="right") - 1
    row_idxs = idxs - cum[arr_idxs]
    for ai, ri in zip(arr_idxs, row_idxs):
        yield x_list[ai][ri], obs_list[ai][ri]
