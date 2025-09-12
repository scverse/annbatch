from __future__ import annotations

import asyncio
import inspect
import platform
from dataclasses import dataclass
from functools import cached_property
from importlib.util import find_spec
from itertools import islice
from typing import TYPE_CHECKING, Protocol

import numpy as np
import scipy as sp
import torch
import zarr

try:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix  # pragma: no cover
except ImportError:
    CupyArray = None
    CupyCSRMatrix = None

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Awaitable, Callable

    from arrayloaders.types import InputInMemoryArray, OutputInMemoryArray


def split_given_size(a: np.ndarray, size: int) -> list[np.ndarray]:
    """Wrapper around `np.split` to split up an array into `size` chunks"""
    return np.split(a, np.arange(size, len(a), size))


@dataclass
class CSRContainer:
    """A low-cost container for moving around the buffers of a CSR object"""

    elems: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int]


def _batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


async def index_datasets(
    dataset_index_to_slices: OrderedDict[int, list[slice]],
    fetch_data: Callable[[list[slice], int], Awaitable[CSRContainer | np.ndarray]],
) -> list[InputInMemoryArray]:
    """Helper function meant to encapsulate asynchronous calls so that we can use the same event loop as zarr.

    Parameters
    ----------
        dataset_index_to_slices
            A lookup of the list-placement index of a dataset to the request slices.
        fetch_data
            The function to do the fetching for a given slice-dataset index pair.
    """
    tasks = []
    for dataset_idx in dataset_index_to_slices.keys():
        tasks.append(
            fetch_data(
                dataset_index_to_slices[dataset_idx],
                dataset_idx,
            )
        )
    return await asyncio.gather(*tasks)


add_datasets_docstring = """\
Append datasets to this dataset.

Parameters
----------
    datasets
        List of :class:`{on_disk_array_type}` objects, generally from :attr:`anndata.AnnData.X`.
    obs
        List of :class:`numpy.ndarray` labels, generally from :attr:`anndata.AnnData.obs`.
"""

add_dataset_docstring = """\
Append a dataset to this dataset.

Parameters
----------
    dataset
        :class:`{on_disk_array_type}` object, generally from :attr:`anndata.AnnData.X`.
    obs
        :class:`numpy.ndarray` labels for the anndata, generally from :attr:`anndata.AnnData.obs`.
"""


add_anndatas_docstring = """\
Append anndatas to this dataset.

Parameters
----------
    anndatas
        List of :class:`anndata.AnnData` objects, with :class:`{on_disk_array_type}` as the data matrix
    obs_keys
        List of :attr:`anndata.AnnData.obs` column labels
    layer_keys
        List of :attr:`anndata.AnnData.layers` keys, and if None, :attr:`anndata.AnnData.X` will be used
"""

add_anndata_docstring = """\
Append a anndata to this dataset.

Parameters
----------
    anndata
        :class:`anndata.AnnData` object, with :class:`{on_disk_array_type}` as the data matrix
    obs_key
        :attr:`anndata.AnnData.obs` column labels
    layer_key
        :attr:`anndata.AnnData.layers` key, and if None, :attr:`anndata.AnnData.X` will be used
"""


__init_docstring__ = """A loader for on-disk {array_type} data.

This loader batches together slice requests to the underlying {array_type} stores to acheive higher performance.
This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
The loader is agnostic to the on-disk chunking/sharding, but it may be advisable to align with the in-memory chunk size for dense.

The dataset class on its own is quite performant for "chunked loading" i.e., `chunk_size > 1`.
When `chunk_size == 1`, a :class:`torch.utils.data.DataLoader` should wrap the dataset object.
In this case, do not use the `add_anndata` or `add_anndatas` option due to https://github.com/scverse/anndata/issues/2021.
Instead use :func:`anndata.io.sparse_dataset` or :func:`zarr.open` to only get the array you need.


Parameters
----------
    chunk_size
        The obs size (i.e., axis 0) of contiguous array data to fetch, by default 512
    preload_nchunks
        The number of chunks of contiguous array data to fetch, by default 32
    shuffle
        Whether or not to shuffle the data, by default True
    return_index
        Whether or not to return the index on each iteration, by default False
    preload_to_gpu
        Whether or not to use cupy for non-io array operations like vstack and indexing.
        This option entails greater GPU memory usage.
        Setting this to `False` is advisable when using the :class:`torch.utils.data.DataLoader` wrapper or potentially with dense data.
    drop_last
        Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    to_torch
        Whether to return `torch.Tensor` as the output

Examples
--------
    >>> from arrayloaders import {child_class}
    >>> ds = {child_class}(
            batch_size=4096,
            chunk_size=32,
            preload_nchunks=512,
        ).add_anndata(my_anndata)
    >>> for batch in ds:
            do_fit(batch)
"""


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
):
    """Samples rows from multiple arrays and their corresponding observation arrays.

    Parameters
    ----------
        x_list
            A list of numpy arrays containing the data to sample from.
        obs_list
            A list of numpy arrays containing the corresponding observations.
        indices
            the list of indexes for each element in x_list/
        shuffle
            Whether to shuffle the rows before sampling. Defaults to True.

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


def check_lt_1(vals: list[int], labels: list[str]):
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


def check_var_shapes(objs: list[SupportsShape]):
    """Small utility function to check that all objects have the same shape along the second axis"""
    if not all(objs[0].shape[1] == d.shape[1] for d in objs):
        raise ValueError("TODO: All datasets must have same shape along the var axis.")


def is_in_torch_dataloader_on_linux():
    """Check if the caller of this function is inside a torch DataLoader"""
    stack = inspect.stack()
    for frame_info in stack:
        local_vars = frame_info.frame.f_locals
        if "self" in local_vars:
            instance = local_vars["self"]
            if find_spec("torch"):
                # TODO: Not sure how else to detect we are in a torch dataloader
                from torch.utils.data._utils.fetch import _IterableDatasetFetcher

                if isinstance(instance, _IterableDatasetFetcher) and platform.system() == "Linux":
                    return True
    return False


def to_torch(input: OutputInMemoryArray, preload_to_gpu: bool):
    """Send the input data to a torch.Tensor"""
    if isinstance(input, torch.Tensor):
        return input
    if isinstance(input, sp.sparse.csr_matrix):
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
        return torch.sparse_csr_tensor(
            torch.from_dlpack(input.indptr),
            torch.from_dlpack(input.indices),
            torch.from_dlpack(input.data),
            input.shape,
        )
    raise TypeError(f"Cannot convert {type(input)} to torch.Tensor")
