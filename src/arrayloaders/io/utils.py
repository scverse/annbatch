from __future__ import annotations

import numpy as np
from torch.utils.data import get_worker_info


def sample_rows(
    x_list: list[np.ndarray], obs_list: list[np.ndarray] | None, shuffle: bool = True
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
        yield x_list[ai][ri], obs_list[ai][ri] if obs_list is not None else None


class WorkerHandle:
    def __init__(self):
        self._worker_info = get_worker_info()  # TODO: typing
        if self._worker_info is None:
            self._rng = np.random.default_rng()
        else:
            # This is used for the _get_chunks function
            # Use the same seed for all workers that the resulting splits are the same across workers
            # torch default seed is `base_seed + worker_id`. Hence, subtract worker_id to get the base seed
            self._rng = np.random.default_rng(
                self._worker_info.seed - self._worker_info.id
            )

    def shuffle(self, obj: np.typing.ArrayLike) -> None:
        """Perform in-place shuffle.

        Args:
            obj: The object to be shuffled
        """
        self._rng.shuffle(obj)

    def get_part_for_worker(self, obj: np.ndarray) -> np.ndarray:
        if self._worker_info is None:
            return obj
        num_workers, worker_id = self._worker_info.num_workers, self._worker_info.id
        chunks_per_worker = len(obj) // num_workers
        start = worker_id * chunks_per_worker
        end = (
            (worker_id + 1) * chunks_per_worker
            if worker_id != num_workers - 1
            else None
        )
        return obj[start:end]
