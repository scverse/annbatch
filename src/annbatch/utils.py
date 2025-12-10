from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from importlib.util import find_spec
from itertools import islice
from typing import TYPE_CHECKING, Protocol

import numpy as np
import scipy as sp
import zarr

try:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix  # pragma: no cover
except ImportError:
    CupyArray = None
    CupyCSRMatrix = None

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from torch import Tensor

    from annbatch.types import OutputInMemoryArray


def split_given_size(a: np.ndarray, size: int) -> list[np.ndarray]:
    """Wrapper around `np.split` to split up an array into `size` chunks"""
    return np.split(a, np.arange(size, len(a), size))


@dataclass
class CSRContainer:
    """A low-cost container for moving around the buffers of a CSR object"""

    elems: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int]
    dtype: np.dtype


def _batched[T](iterable: Iterable[T], n: int) -> Generator[list[T], None, None]:
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


# TODO: make this part of the public zarr or zarrs-python API.
# We can do chunk coalescing in zarrs based on integer arrays, so I think
# there would make sense with ezclump or similar.
# Another "solution" would be for zarrs to support integer indexing properly, if that pipeline works,
# or make this an "experimental setting" and to use integer indexing for the zarr-python pipeline.
# See: https://github.com/zarr-developers/zarr-python/issues/3175 for why this is better than simpler alternatives.
class MultiBasicIndexer(zarr.core.indexing.Indexer):
    """Custom indexer to enable joint fetching of disparate slices"""

    def __init__(self, indexers: list[zarr.core.indexing.Indexer]):
        self.shape = (sum(i.shape[0] for i in indexers), *indexers[0].shape[1:])
        self.drop_axes = indexers[0].drop_axes  # maybe?
        self.indexers = indexers

    def __iter__(self):
        total = 0
        for i in self.indexers:
            for c in i:
                out_selection = c[2]
                gap = out_selection[0].stop - out_selection[0].start
                yield type(c)(c[0], c[1], (slice(total, total + gap), *out_selection[1:]), c[3])
                total += gap


def sample_rows(
    x_list: list[np.ndarray],
    obs_list: list[np.ndarray] | None,
    indices: list[np.ndarray] | None = None,
    *,
    shuffle: bool = True,
) -> Generator[tuple[np.ndarray, np.ndarray | None], None, None]:
    """Samples rows from multiple arrays and their corresponding observation arrays.

    Parameters
    ----------
        x_list
            A list of numpy arrays containing the data to sample from.
        obs_list
            A list of numpy arrays containing the corresponding observations.
        indices
            the list of indexes for each element in `x_list/`
        shuffle
            Whether to shuffle the rows before sampling.

    Yields
    ------
        tuple
            A tuple containing a row from `x_list` and the corresponding row from `obs_list`.
    """
    lengths = np.fromiter((x.shape[0] for x in x_list), dtype=int)
    cum = np.concatenate(([0], np.cumsum(lengths)))
    total = cum[-1]
    idxs = np.arange(total)
    if shuffle:
        np.random.default_rng().shuffle(idxs)
    arr_idxs = np.searchsorted(cum, idxs, side="right") - 1
    row_idxs = idxs - cum[arr_idxs]
    for ai, ri in zip(arr_idxs, row_idxs, strict=True):
        res = [
            x_list[ai][ri],
            obs_list[ai][ri] if obs_list is not None else None,
        ]
        if indices is not None:
            yield (*res, indices[ai][ri])
        else:
            yield tuple(res)


class WorkerHandle:  # noqa: D101
    @cached_property
    def _worker_info(self):
        if find_spec("torch"):
            from torch.utils.data import get_worker_info

            return get_worker_info()
        return None

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

        Parameters
        ----------
            obj
                The object to be shuffled
        """
        self._rng.shuffle(obj)

    def get_part_for_worker(self, obj: np.ndarray) -> np.ndarray:
        """Get a chunk of an incoming array accordnig to the current worker id.

        Parameters
        ----------
            obj
                Incoming array

        Returns
        -------
            A evenly split part of the ray corresponding to how many workers there are.
        """
        if self._worker_info is None:
            return obj
        num_workers, worker_id = self._worker_info.num_workers, self._worker_info.id
        chunks_split = np.array_split(obj, num_workers)
        return chunks_split[worker_id]


def check_lt_1(vals: list[int], labels: list[str]) -> None:
    """Raise a ValueError if any of the values are less than one.

    The format of the error is "{labels[i]} must be greater than 1, got {values[i]}"
    and is raised based on the first found less than one value.

    Parameters
    ----------
        vals
            The values to check < 1
        labels
            The label for the value in the error if the value is less than one.

    Raises
    ------
        ValueError: _description_
    """
    if any(is_lt_1 := [v < 1 for v in vals]):
        label, value = next(
            (label, value)
            for label, value, check in zip(
                labels,
                vals,
                is_lt_1,
                strict=True,
            )
            if check
        )
        raise ValueError(f"{label} must be greater than 1, got {value}")


class SupportsShape(Protocol):  # noqa: D101
    @property
    def shape(self) -> tuple[int, int] | list[int]: ...  # noqa: D102


def check_var_shapes(objs: list[SupportsShape]) -> None:
    """Small utility function to check that all objects have the same shape along the second axis"""
    if not all(objs[0].shape[1] == d.shape[1] for d in objs):
        raise ValueError("TODO: All datasets must have same shape along the var axis.")


def to_torch(input: OutputInMemoryArray, preload_to_gpu: bool) -> Tensor:
    """Send the input data to a torch.Tensor"""
    import torch

    if isinstance(input, torch.Tensor):
        return input
    if isinstance(input, sp.sparse.csr_matrix):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state", UserWarning)
            tensor = torch.sparse_csr_tensor(
                torch.from_numpy(input.indptr),
                torch.from_numpy(input.indices),
                torch.from_numpy(input.data),
                input.shape,
            )
        if preload_to_gpu:
            return tensor.cuda(non_blocking=True)
        return tensor
    if isinstance(input, np.ndarray):
        tensor = torch.from_numpy(input)
        if preload_to_gpu:
            return tensor.cuda(non_blocking=True)
        return tensor
    if isinstance(input, CupyArray):
        return torch.from_dlpack(input)
    if isinstance(input, CupyCSRMatrix):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state", UserWarning)
            return torch.sparse_csr_tensor(
                torch.from_dlpack(input.indptr),
                torch.from_dlpack(input.indices),
                torch.from_dlpack(input.data),
                input.shape,
            )
    raise TypeError(f"Cannot convert {type(input)} to torch.Tensor")
