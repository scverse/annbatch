from __future__ import annotations

from functools import cached_property
from typing import Protocol

import numpy as np
from torch.utils.data import get_worker_info


def sample_rows(
    x_list: list[np.ndarray],
    obs_list: list[np.ndarray] | None,
    indices: list[np.ndarray] | None = None,
    *,
    shuffle: bool = True,
):
    """Samples rows from multiple arrays and their corresponding observation arrays.

    Args:
        x_list: A list of numpy arrays containing the data to sample from.
        obs_list: A list of numpy arrays containing the corresponding observations.
        indices: the list of indexes for each element in x_list/
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
    for ai, ri in zip(arr_idxs, row_idxs, strict=False):
        res = [
            x_list[ai][ri],
            obs_list[ai][ri] if obs_list is not None else None,
        ]
        if indices is not None:
            yield (*res, indices[ai][ri])
        else:
            yield tuple(res)


class WorkerHandle:
    @cached_property
    def _worker_info(self):
        return get_worker_info()

    @cached_property
    def _rng(self):
        if self._worker_info is None:
            return np.random.default_rng()
        else:
            # This is used for the _get_chunks function
            # Use the same seed for all workers that the resulting splits are the same across workers
            # torch default seed is `base_seed + worker_id`. Hence, subtract worker_id to get the base seed
            return np.random.default_rng(self._worker_info.seed - self._worker_info.id)

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
        chunks_split = np.array_split(obj, num_workers)
        return chunks_split[worker_id]


def check_lt_1(vals: list[int], labels: list[str]):
    """Raise a ValueError if any of the values are less than one.

    The format of the error is "{labels[i]} must be greater than 1, got {values[i]}"
    and is raised based on the first found less than one value.

    Args:
        vals: The values to check < 1
        labels: The label for the value in the error if the value is less than one.

    Raises:
        ValueError: _description_
    """
    if any(is_lt_1 := [v < 1 for v in vals]):
        label, value = next(
            (label, value)
            for label, value, check in zip(
                labels,
                vals,
                is_lt_1,
                strict=False,
            )
            if check
        )
        raise ValueError(f"{label} must be greater than 1, got {value}")


class SupportsShape(Protocol):
    @property
    def shape(self) -> tuple[int, int] | list[int]: ...


def check_var_shapes(objs: list[SupportsShape]):
    if not all(objs[0].shape[1] == d.shape[1] for d in objs):
        raise ValueError("TODO: All datasets must have same shape along the var axis.")
