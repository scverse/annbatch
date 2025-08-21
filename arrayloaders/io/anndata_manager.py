from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, defaultdict
from itertools import islice
from typing import TYPE_CHECKING, Generic

import numpy as np
import zarr.core.sync as zsync
from scipy import sparse as sp

from .types import BackingArray, InMemoryArray
from .utils import check_lt_1, check_var_shapes

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    import anndata as ad

    from .utils import WorkerHandle


def split_given_size(a: np.ndarray, size: int) -> list[np.ndarray]:
    return np.split(a, np.arange(size, len(a), size))


accepted_backing_types = BackingArray.__constraints__


def _batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


async def index_datasets(
    dataset_index_to_slices: OrderedDict[int, list[slice]],
    fetch_data: Callable[[list[slice], int], Awaitable[InMemoryArray]],
) -> list[InMemoryArray]:
    """Helper function meant to encapsulate asynchronous calls so that we can use the same event loop as zarr.

    Args:
        dataset_index_to_slices: A lookup of the list-placement index of a dataset to the request slices.
        fetch_data: The function to do the fetching for a given slice-dataset index pair.
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


add_dataset_docstring = """\
Append datasets to this loader.

Args:
    datasets: List of :class:`anndata.abc.CSRDataset` or :class:`zarr.Array` objects, generally from :attr:`anndata.AnnData.X`.
    obs: List of `numpy.ndarray` labels, generally from :attr:`anndata.AnnData.obs`.
"""

add_dataset_docstring = """\
Append a dataset to this loader.

Args:
    dataset: :class:`anndata.abc.CSRDataset` or :class:`zarr.Array` object, generally from :attr:`anndata.AnnData.X`.
    obs: `numpy.ndarray` labels for the dataset, generally from :attr:`anndata.AnnData.obs`.
"""


class AnnDataManager(Generic[BackingArray, InMemoryArray]):
    train_datasets: list[BackingArray] = []
    labels: list[np.ndarray] | None = None
    _return_index: bool = False
    _on_add: Callable | None = None
    _batch_size: int = 1

    def __init__(
        self,
        *,
        on_add: Callable | None = None,
        return_index: bool = False,
        batch_size: int = 1,
    ):
        self._on_add = on_add
        self._return_index = return_index
        self._batch_size = batch_size

    @property
    def dataset_type(self) -> type[BackingArray]:
        return type(self.train_datasets[0])

    @property
    def n_obs(self) -> int:
        return sum(ds.shape[0] for ds in self.train_datasets)

    @property
    def n_var(self) -> int:
        return self.train_datasets[0].shape[1]

    def add_anndatas(
        self,
        adatas: list[ad.AnnData],
        layer_keys: list[str | None] | str | None = None,
        obs_keys: list[str] | str | None = None,
    ) -> None:
        raise NotImplementedError("See https://github.com/scverse/anndata/issues/2021")

    def add_anndata(
        self,
        adata: ad.AnnData,
        layer_key: str | None = None,
        obs_key: str | None = None,
    ) -> None:
        raise NotImplementedError("See https://github.com/scverse/anndata/issues/2021")

    def add_datasets(
        self, datasets: list[BackingArray], obs: list[np.ndarray] | None = None
    ) -> None:
        if obs is None:
            obs = [None] * len(datasets)
        for ds, o in zip(datasets, obs, strict=True):
            self.add_dataset(ds, o)

    def add_dataset(self, dataset: BackingArray, obs: np.ndarray | None = None) -> None:
        if len(self.train_datasets) > 0:
            if self.labels is None and obs is not None:
                raise ValueError(
                    f"Cannot add a dataset with obs label {obs} when training datasets have already been added without labels"
                )
            if self.labels is not None and obs is None:
                raise ValueError(
                    "Cannot add a dataset with no obs label when training datasets have already been added without labels"
                )
        if not isinstance(dataset, accepted_types := accepted_backing_types):
            raise TypeError(
                f"Cannot add a dataset of type {type(dataset)}, only {accepted_types} are allowed"
            )
        if len(self.train_datasets) > 0 and not isinstance(dataset, self.dataset_type):
            raise TypeError(
                f"Cannot add a dataset whose data of type {type(dataset)} was not an instance of expected type {self.dataset_type}"
            )
        datasets = self.train_datasets + [dataset]
        check_var_shapes(datasets)
        self._var_size = datasets[0].shape[1]  # TODO: joins
        self.train_datasets = datasets
        if self.labels is not None:  # labels exist
            self.labels += [obs]
        elif (
            obs is not None
        ):  # labels dont exist yet, but are being added for the first time
            self.labels = [obs]
        if self._on_add is not None:
            self._on_add()

    def _get_relative_obs_indices(
        self, index: slice, *, use_original_space: bool = False
    ) -> list[tuple[slice, int]]:
        """Generate a slice relative to a dataset given a global slice index over all datasets.

        For a given slice indexer of axis 0, return a new slice relative to the on-disk
        data it represents given the number of total observations as well as the index of
        the underlying data on disk from the argument `sparse_datasets` to the initializer.

        For example, given slice index (10, 15), for 4 datasets each with size 5 on axis zero,
        this function returns ((0,5), 2) representing slice (0,5) along axis zero of sparse dataset 2.

        Args:
            index: The queried slice.
            use_original_space: Whether or not the slices should be reindexed against the anndata objects.

        Returns:
            A slice relative to the dataset it represents as well as the index of said dataset in `sparse_datasets`.
        """
        min_idx = index.start
        max_idx = index.stop
        curr_pos = 0
        slices = []
        for idx, array in enumerate(self.train_datasets):
            array_start = curr_pos
            n_obs = array.shape[0]
            array_end = curr_pos + n_obs

            start = max(min_idx, array_start)
            stop = min(max_idx, array_end)
            if start < stop:
                if use_original_space:
                    slices.append((slice(start, stop), idx))
                else:
                    relative_start = start - array_start
                    relative_stop = stop - array_start
                    slices.append((slice(relative_start, relative_stop), idx))
            curr_pos += n_obs
        return slices

    def _slices_to_slices_with_array_index(
        self, slices: list[slice], *, use_original_space: bool = False
    ) -> OrderedDict[int, list[slice]]:
        """Given a list of slices, give the lookup between on-disk datasets and slices relative to that dataset.

        Args:
            slices: Slices to relative to the on-disk datasets.
            use_original_space: Whether or not the slices should be reindexed against the anndata objects.

        Returns:
            A lookup between the dataset and its indexing slices, ordered by keys.
        """
        dataset_index_to_slices: defaultdict[int, list[slice]] = defaultdict(list)
        for slice in slices:
            for relative_obs_indices in self._get_relative_obs_indices(
                slice, use_original_space=use_original_space
            ):
                dataset_index_to_slices[relative_obs_indices[1]] += [
                    relative_obs_indices[0]
                ]
        keys = sorted(dataset_index_to_slices.keys())
        dataset_index_to_slices_sorted = OrderedDict()
        for k in keys:
            dataset_index_to_slices_sorted[k] = dataset_index_to_slices[k]
        return dataset_index_to_slices_sorted

    def _get_chunks(
        self, chunk_size: int, worker_handle: WorkerHandle, shuffle: bool
    ) -> np.ndarray:
        """Get a potentially shuffled list of chunk ids, accounting for the fact that this dataset might be inside a worker.

        Returns:
            A :class:`numpy.ndarray` of chunk ids.
        """
        chunks = np.array(list(range(math.ceil(self.n_obs / chunk_size))))
        if shuffle:
            worker_handle.shuffle(chunks)

        return worker_handle.get_part_for_worker(chunks)

    def iter(
        self,
        chunk_size: int,
        worker_handle: WorkerHandle,
        preload_nchunks: int,
        shuffle: bool,
        fetch_data: Callable[[list[slice], int], Awaitable[InMemoryArray]],
    ) -> Iterator[
        tuple[InMemoryArray, None | np.ndarray]
        | tuple[InMemoryArray, None | np.ndarray, np.ndarray]
    ]:
        """Iterate over the on-disk csr datasets.

        Yields:
            A one-row sparse matrix.
        """
        check_lt_1(
            [len(self.train_datasets), self.n_obs],
            ["Number of datasets", "Number of observations"],
        )
        # In order to handle data returned where (chunk_size * preload_nchunks) mod batch_size != 0
        # we must keep track of the leftover data.
        in_memory_data = None
        in_memory_labels = None
        in_memory_indices = None
        for chunk_indices in _batched(
            self._get_chunks(chunk_size, worker_handle, shuffle), preload_nchunks
        ):
            slices = [
                slice(
                    index * chunk_size,
                    min(self.n_obs, (index + 1) * chunk_size),
                )
                for index in chunk_indices
            ]
            dataset_index_to_slices = self._slices_to_slices_with_array_index(slices)
            # Fetch the data over slices
            chunks: list[InMemoryArray] = zsync.sync(
                index_datasets(dataset_index_to_slices, fetch_data)
            )
            # Accumulate labels
            labels: None | list[np.ndarray] = None
            if self.labels is not None:
                labels = []
                for dataset_idx in dataset_index_to_slices.keys():
                    labels += [
                        self.labels[dataset_idx][
                            np.concatenate(
                                [
                                    np.arange(s.start, s.stop)
                                    for s in dataset_index_to_slices[dataset_idx]
                                ]
                            )
                        ]
                    ]
            # Accumulate indices if necessary
            indices: None | list[np.ndarray] = None
            if self._return_index:
                dataset_index_to_slices = self._slices_to_slices_with_array_index(
                    slices, use_original_space=True
                )
                dataset_indices = dataset_index_to_slices.keys()
                indices = [
                    np.concatenate(
                        [
                            np.arange(
                                s.start,
                                s.stop,
                            )
                            for s in dataset_index_to_slices[index]
                        ]
                    )
                    for index in dataset_indices
                ]
            # Do batch returns, handling leftover data as necessary
            mod = sp if isinstance(chunks[0], sp.csr_matrix) else np
            in_memory_data = (
                mod.vstack(chunks)
                if in_memory_data is None
                else mod.vstack([in_memory_data, *chunks])
            )
            if self.labels is not None:
                in_memory_labels = (
                    np.concatenate(labels)
                    if in_memory_labels is None
                    else np.concatenate([in_memory_labels, *labels])
                )
            if self._return_index:
                in_memory_indices = (
                    np.concatenate(indices)
                    if in_memory_indices is None
                    else np.concatenate([in_memory_indices, *indices])
                )
            # Create random indices into in_memory_data and then index into it
            # If there is "leftover" at the end (see the modulo op),
            # save it for the next iteration.
            batch_indices = np.arange(in_memory_data.shape[0])
            if shuffle:
                np.random.default_rng().shuffle(batch_indices)
            splits = split_given_size(batch_indices, self._batch_size)
            for i, s in enumerate(splits):
                if s.shape[0] == self._batch_size:
                    res = [
                        in_memory_data[s],
                        in_memory_labels[s] if self.labels is not None else None,
                    ]
                    if self._return_index:
                        res += [in_memory_indices[s]]
                    yield tuple(res)
                if i == (
                    len(splits) - 1
                ):  # end of iteration, leftover data needs be kept
                    if (s.shape[0] % self._batch_size) != 0:
                        in_memory_data = in_memory_data[s]
                        if in_memory_labels is not None:
                            in_memory_labels = in_memory_labels[s]
                        if in_memory_indices is not None:
                            in_memory_indices = in_memory_indices[s]
                    else:
                        in_memory_data = None
                        in_memory_labels = None
                        in_memory_indices = None
        if in_memory_data is not None:  # handle any leftover data
            res = [
                in_memory_data,
                in_memory_labels if self.labels is not None else None,
            ]
            if self._return_index:
                res += [in_memory_indices[s]]
            yield tuple(res)


AnnDataManager.add_datasets.__doc__ = add_dataset_docstring
AnnDataManager.add_dataset.__doc__ = add_dataset_docstring
