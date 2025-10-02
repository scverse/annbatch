from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from types import NoneType
from typing import TYPE_CHECKING, Generic, cast

import anndata as ad
import numpy as np
import zarr.core.sync as zsync
from scipy import sparse as sp

from arrayloaders.types import InputInMemoryArray, OnDiskArray, OutputInMemoryArray
from arrayloaders.utils import (
    CSRContainer,
    WorkerHandle,
    _batched,
    add_dataset_docstring,
    check_lt_1,
    check_var_shapes,
    index_datasets,
    split_given_size,
)

try:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix  # pragma: no cover
except ImportError:
    CupyCSRMatrix = NoneType
    CupyArray = NoneType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from types import ModuleType

accepted_on_disk_types = OnDiskArray.__constraints__


class AnnDataManager(Generic[OnDiskArray, InputInMemoryArray, OutputInMemoryArray]):  # noqa: D101
    train_datasets: list[OnDiskArray] = []
    labels: list[np.ndarray] | None = None
    _return_index: bool = False
    _on_add: Callable | None = None
    _batch_size: int = 1
    _shapes: list[tuple[int, int]] = []
    _preload_to_gpu: bool = False

    def __init__(
        self,
        *,
        on_add: Callable | None = None,
        return_index: bool = False,
        batch_size: int = 1,
        preload_to_gpu: bool = False,
    ):
        self._on_add = on_add
        self._return_index = return_index
        self._batch_size = batch_size
        self._preload_to_gpu = preload_to_gpu

    @property
    def _sp_module(self) -> ModuleType:
        if self._preload_to_gpu:
            try:
                import cupyx.scipy.sparse as cpx  # pragma: no cover

                return cpx
            except ImportError:
                raise ImportError(
                    "Cannot find cupy module even though `preload_to_gpu` argument was set to `True`"
                ) from None
        return sp

    @property
    def _np_module(self) -> ModuleType:
        if self._preload_to_gpu:
            try:
                import cupy as cp

                return cp
            except ImportError:
                raise ImportError(
                    "Cannot find cupy module even though `preload_to_gpu` argument was set to `True`"
                ) from None

        return np

    @property
    def dataset_type(self) -> type[OnDiskArray]:  # noqa: D102
        return type(self.train_datasets[0])

    @property
    def n_obs(self) -> int:  # noqa: D102
        return sum(shape[0] for shape in self._shapes)

    @property
    def n_var(self) -> int:  # noqa: D102
        return self._shapes[0][1]

    def add_anndatas(  # noqa: D102
        self,
        adatas: list[ad.AnnData],
        layer_keys: list[str | None] | str | None = None,
        obs_keys: list[str] | str | None = None,
    ) -> None:
        if isinstance(layer_keys, str | None):
            layer_keys = [layer_keys] * len(adatas)
        if isinstance(obs_keys, str | None):
            obs_keys = [obs_keys] * len(adatas)
        elem_to_keys = dict(zip(["layer", "obs"], [layer_keys, obs_keys], strict=True))
        check_lt_1(
            [len(adatas)] + sum((([len(k)] if k is not None else []) for k in elem_to_keys.values()), []),
            ["Number of anndatas"]
            + sum(
                ([f"Number of {label} keys"] if keys is not None else [] for keys, label in elem_to_keys.items()),
                [],
            ),
        )
        for adata, obs_key, layer_key in zip(adatas, obs_keys, layer_keys, strict=True):
            kwargs = {"obs_key": obs_key, "layer_key": layer_key}
            self.add_anndata(adata, **kwargs)

    def add_anndata(  # noqa: D102
        self,
        adata: ad.AnnData,
        layer_key: str | None = None,
        obs_key: str | None = None,
    ) -> None:
        dataset = adata.X if layer_key is None else adata.layers[layer_key]
        if not isinstance(dataset, accepted_on_disk_types):
            raise TypeError(f"Found {type(dataset)} but only {accepted_on_disk_types} are usable")
        obs = adata.obs[obs_key].to_numpy() if obs_key is not None else None
        self.add_dataset(cast("OnDiskArray", dataset), obs)

    def add_datasets(self, datasets: list[OnDiskArray], obs: list[np.ndarray] | None = None) -> None:  # noqa: D102
        if obs is None:
            obs = [None] * len(datasets)
        for ds, o in zip(datasets, obs, strict=True):
            self.add_dataset(ds, o)

    def add_dataset(self, dataset: OnDiskArray, obs: np.ndarray | None = None) -> None:  # noqa: D102
        if len(self.train_datasets) > 0:
            if self.labels is None and obs is not None:
                raise ValueError(
                    f"Cannot add a dataset with obs label {obs} when training datasets have already been added without labels"
                )
            if self.labels is not None and obs is None:
                raise ValueError(
                    "Cannot add a dataset with no obs label when training datasets have already been added without labels"
                )
        if not isinstance(dataset, accepted_types := accepted_on_disk_types):
            raise TypeError(f"Cannot add a dataset of type {type(dataset)}, only {accepted_types} are allowed")
        if len(self.train_datasets) > 0 and not isinstance(dataset, self.dataset_type):
            raise TypeError(
                f"Cannot add a dataset whose data of type {type(dataset)} was not an instance of expected type {self.dataset_type}"
            )
        datasets = self.train_datasets + [dataset]
        check_var_shapes(datasets)
        self._shapes = self._shapes + [dataset.shape]
        self.train_datasets = datasets
        if self.labels is not None:  # labels exist
            self.labels += [obs]
        elif obs is not None:  # labels dont exist yet, but are being added for the first time
            self.labels = [obs]
        if self._on_add is not None:
            self._on_add()

    def _get_relative_obs_indices(self, index: slice, *, use_original_space: bool = False) -> list[tuple[slice, int]]:
        """Generate a slice relative to a dataset given a global slice index over all datasets.

        For a given slice indexer of axis 0, return a new slice relative to the on-disk
        data it represents given the number of total observations as well as the index of
        the underlying data on disk from the argument `sparse_datasets` to the initializer.

        For example, given slice index (10, 15), for 4 datasets each with size 5 on axis zero,
        this function returns ((0,5), 2) representing slice (0,5) along axis zero of sparse dataset 2.

        Args:
            index: The queried slice.
            use_original_space: Whether or not the slices should be reindexed against the anndata objects.

        Returns
        -------
            A slice relative to the dataset it represents as well as the index of said dataset in `sparse_datasets`.
        """
        min_idx = index.start
        max_idx = index.stop
        curr_pos = 0
        slices = []
        for idx, (n_obs, _) in enumerate(self._shapes):
            array_start = curr_pos
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

        Returns
        -------
            A lookup between the dataset and its indexing slices, ordered by keys.
        """
        dataset_index_to_slices: defaultdict[int, list[slice]] = defaultdict(list)
        for slice in slices:
            for relative_obs_indices in self._get_relative_obs_indices(slice, use_original_space=use_original_space):
                dataset_index_to_slices[relative_obs_indices[1]] += [relative_obs_indices[0]]
        keys = sorted(dataset_index_to_slices.keys())
        dataset_index_to_slices_sorted = OrderedDict()
        for k in keys:
            dataset_index_to_slices_sorted[k] = dataset_index_to_slices[k]
        return dataset_index_to_slices_sorted

    def _get_chunks(self, chunk_size: int, worker_handle: WorkerHandle, shuffle: bool) -> np.ndarray:
        """Get a potentially shuffled list of chunk ids, accounting for the fact that this dataset might be inside a worker.

        Returns
        -------
            A :class:`numpy.ndarray` of chunk ids.
        """
        chunks = np.arange(math.ceil(self.n_obs / chunk_size))
        if shuffle:
            worker_handle.shuffle(chunks)

        return worker_handle.get_part_for_worker(chunks)

    def iter(
        self,
        chunk_size: int,
        worker_handle: WorkerHandle,
        preload_nchunks: int,
        shuffle: bool,
        fetch_data: Callable[[list[slice], int], Awaitable[np.ndarray | CSRContainer]],
    ) -> Iterator[
        tuple[InputInMemoryArray, None | np.ndarray] | tuple[InputInMemoryArray, None | np.ndarray, np.ndarray]
    ]:
        """Iterate over the on-disk csr datasets.

        Yields
        ------
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
        mod = self._sp_module if issubclass(self.dataset_type, ad.abc.CSRDataset) else np
        for chunk_indices in _batched(self._get_chunks(chunk_size, worker_handle, shuffle), preload_nchunks):
            slices = [
                slice(
                    index * chunk_size,
                    min(self.n_obs, (index + 1) * chunk_size),
                )
                for index in chunk_indices
            ]
            dataset_index_to_slices = self._slices_to_slices_with_array_index(slices)
            # Fetch the data over slices
            chunks: list[InputInMemoryArray] = zsync.sync(index_datasets(dataset_index_to_slices, fetch_data))
            if any(isinstance(c, CSRContainer) for c in chunks):
                chunks_converted: list[OutputInMemoryArray] = [
                    self._sp_module.csr_matrix(tuple(self._np_module.asarray(e) for e in c.elems), shape=c.shape)
                    for c in chunks
                ]
            else:
                chunks_converted = [self._np_module.asarray(c) for c in chunks]
            # Accumulate labels
            labels: None | list[np.ndarray] = None
            if self.labels is not None:
                labels = []
                for dataset_idx in dataset_index_to_slices.keys():
                    labels += [
                        self.labels[dataset_idx][
                            np.concatenate([np.arange(s.start, s.stop) for s in dataset_index_to_slices[dataset_idx]])
                        ]
                    ]
            # Accumulate indices if necessary
            indices: None | list[np.ndarray] = None
            if self._return_index:
                dataset_index_to_slices = self._slices_to_slices_with_array_index(slices, use_original_space=True)
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
            in_memory_data = (
                mod.vstack(chunks_converted)
                if in_memory_data is None
                else mod.vstack([in_memory_data, *chunks_converted])
            )
            if self.labels is not None:
                in_memory_labels = (
                    np.concatenate(labels) if in_memory_labels is None else np.concatenate([in_memory_labels, *labels])
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
                if i == (len(splits) - 1):  # end of iteration, leftover data needs be kept
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
                res += [in_memory_indices]
            yield tuple(res)


AnnDataManager.add_datasets.__doc__ = add_dataset_docstring
AnnDataManager.add_dataset.__doc__ = add_dataset_docstring
