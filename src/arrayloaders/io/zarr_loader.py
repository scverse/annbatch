from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from itertools import accumulate, chain, islice, pairwise
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

import anndata as ad
import numpy as np
import zarr
import zarr.core.sync as zsync
from scipy import sparse as sp
from torch.utils.data import IterableDataset

from .utils import WorkerHandle, check_lt_1, check_var_shapes, sample_rows

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator
    from typing import Callable

    import pandas as pd


def _batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


ArrayType = TypeVar("ArrayType", ad.abc.CSRDataset, zarr.Array)


class DatasetManager(Generic[ArrayType]):
    train_datasets: list[ArrayType] = []

    @property
    def array_type(self) -> type[ArrayType]:
        return type(self.train_datasets[0])

    @property
    def n_obs(self) -> int:
        return sum(ds.shape[0] for ds in self.train_datasets)

    @property
    def n_var(self) -> int:
        return self.train_datasets[0].shape[1]

    def add_anndatas(
        self, adatas: list[ad.AnnData], layer_keys: list[str | None] | None = None
    ) -> None:
        """Append anndata datasets to this loader.

        Args:
            adatas: List of :class:`anndata.AnnData` objects.
            layer_keys: Keys for getting the underlying data out of the anndata object.
                None within the list of keys means using :attr:`~anndata.AnnData.X` while a string value gets from :attr:`~anndata.AnnData.layers`.
                If not provided, all :class:`~anndata.AnnData` objects will have their data taken from :attr:`~anndata.AnnData.X`.
                Defaults to None.
        """
        check_lt_1(
            [len(adatas)] + ([len(layer_keys)] if layer_keys is not None else []),
            ["Number of anndatas"]
            + (["Number of layer keys"] if layer_keys is not None else []),
        )
        if layer_keys is not None:
            if len(adatas) != len(layer_keys):
                raise ValueError(
                    f"Number of anndatas {len(adatas)} must match number of layer keys {len(layer_keys)}"
                )
        else:
            layer_keys = [None] * len(adatas)
        for adata, layer_key in zip(adatas, layer_keys):
            self.add_anndata(adata, layer_key)

    def add_anndata(self, adata: ad.AnnData, layer_key: str | None = None) -> None:
        """Append an anndata dataset to this loader.

        Args:
            adata: :class:`anndata.AnnData` object.
            layer_key: Keys for getting the underlying data out of the anndata object.
                None means using :attr:`~anndata.AnnData.X` while a string value gets from :attr:`~anndata.AnnData.layers`.
                Defaults to None.
        """
        check_lt_1([adata.shape[0]], ["Anndata obs axis size"])
        dataset = adata.X if layer_key is None else adata.layers[layer_key]
        if len(self.train_datasets) > 0 and not isinstance(dataset, self.array_type):
            raise ValueError(
                f"Anndata dataset was not an instance of expected type {self.array_type}"
            )
        datasets = self.train_datasets + [dataset]
        check_var_shapes(datasets)
        self._var_size = datasets[0].shape[1]  # TODO: joins
        self.train_datasets = datasets

    def _get_relative_obs_indices(self, index: slice) -> list[tuple[slice, int]]:
        """Generate a slice relative to a dataset given a global slice index over all datasets.

        For a given slice indexer of axis 0, return a new slice relative to the on-disk
        data it represents given the number of total observations as well as the index of
        the underlying data on disk from the argument `sparse_datasets` to the initializer.

        For example, given slice index (10, 15), for 4 datasets each with size 5 on axis zero,
        this function returns ((0,5), 2) representing slice (0,5) along axis zero of sparse dataset 2.

        Args:
            index: The queried slice.

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
                relative_start = start - array_start
                relative_stop = stop - array_start
                slices.append((slice(relative_start, relative_stop), idx))
            curr_pos += n_obs
        return slices

    def _slices_to_slices_with_array_index(
        self, slices: list[slice]
    ) -> defaultdict[int, list[slice]]:
        """Given a list of slices, give the lookup between on-disk datasets and slices relative to that dataset.

        Args:
            slices: Slices to relative to the on-disk datasets.

        Returns:
            A lookup between the dataset and its indexing slices.
        """
        dataset_index_to_slices: defaultdict[int, list[slice]] = defaultdict(list)
        for slice in slices:
            for relative_obs_indices in self._get_relative_obs_indices(slice):
                dataset_index_to_slices[relative_obs_indices[1]] += [
                    relative_obs_indices[0]
                ]
        return dataset_index_to_slices

    def _get_chunks(self, chunk_size: int, worker_handle, shuffle: bool) -> np.ndarray:
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
        worker_handle,
        preload_nchunks: int,
        shuffle: bool,
        fetch_data: Callable[[list[slice], int], Awaitable[ArrayType]],
    ) -> Iterator[sp.csr_matrix]:
        """Iterate over the on-disk csr datasets.

        Yields:
            A one-row sparse matrix.
        """
        check_lt_1(
            [len(self.train_datasets), self.n_obs],
            ["Number of datasets", "Number of observations"],
        )
        for chunk_indices in _batched(
            self._get_chunks(chunk_size, worker_handle, shuffle), preload_nchunks
        ):

            async def get(chunk_indices: np.ndarray) -> list[sp.csr_matrix]:
                slices = [
                    slice(
                        index * chunk_size,
                        min(self.n_obs, (index + 1) * chunk_size),
                    )
                    for index in chunk_indices
                ]
                tasks = []
                dataset_index_to_slices = self._slices_to_slices_with_array_index(
                    slices
                )
                for dataset_idx in dataset_index_to_slices:
                    tasks.append(
                        fetch_data(
                            dataset_index_to_slices[dataset_idx],
                            dataset_idx,
                        )
                    )
                return await asyncio.gather(*tasks)

            chunks = zsync.sync(get(chunk_indices))
            yield from sample_rows(
                chunks,
                None,
                shuffle,
            )


class ZarrDenseDataset(IterableDataset):
    def __init__(
        self,
        x_list: list[zarr.Array],
        obs_list: list[
            pd.DataFrame
        ],  # TODO: split obs off into its own dataloader so it can be reused.
        obs_column: str,
        shuffle: bool = True,
        preload_nchunks: int = 8,
    ):
        check_lt_1(
            [len(x_list), len(obs_list), preload_nchunks],
            ["Number of arrays", "Number of obs labels", "Preload chunks"],
        )
        check_var_shapes(x_list)
        self._arrays = x_list
        self._obs = obs_list
        self._obs_column = obs_column
        self._shuffle = shuffle
        self._preload_nchunks = preload_nchunks
        self._worker_handle = WorkerHandle()

        self.n_obs_list: list[int] = []  # number of observations for each array
        self.chunks_lengths: list[int] = []  # chunk length for each array
        arrays_chunks: list[list[int]] = []  # list of chunk indices for each array
        arrays_nchunks: list[int] = []  # number of chunks for each array
        for array in x_list:
            self.n_obs_list.append(array.shape[0])
            self.chunks_lengths.append(array.chunks[0])
            array_nchunks = array.nchunks
            arrays_nchunks.append(array_nchunks)
            arrays_chunks.append(np.arange(array_nchunks))

        self._n_obs = sum(self.n_obs_list)
        # assumes the same for all arrays
        array0 = x_list[0]
        self.n_vars = array0.shape[1]
        self.dtype = array0.dtype
        self.order = array0.order

        self.chunks = np.hstack(arrays_chunks)
        self.array_idxs = np.repeat(np.arange(len(self._arrays)), arrays_nchunks)
        # pre-compute chunk slices
        # slices are needed because we want to iterate over (logical) chunks, not (physical) shards
        # but in zarr array.blocks[i] returns whole shards unlike dask
        self.chunks_slices: list[slice] = []
        for i, chunk_idx in enumerate(self.chunks):
            self.chunks_slices.append(self._chunk_slice(chunk_idx, self.array_idxs[i]))

    def _chunk_slice(self, chunk_idx: int, array_idx: int):
        chunk_length = self.chunks_lengths[array_idx]
        array_n_obs = self.n_obs_list[array_idx]
        start = chunk_length * chunk_idx
        stop = min(chunk_length * (chunk_idx + 1), array_n_obs)
        return slice(start, stop)

    async def _fetch_chunks_x(self, chunk_idxs: list[int]):
        tasks = []
        for i in chunk_idxs:
            array_idx = self.array_idxs[i]
            array = self._arrays[array_idx]
            tasks.append(array._async_array.getitem(self.chunks_slices[i]))
        return await asyncio.gather(*tasks)

    def _fetch_chunks_obs(self, chunk_idxs: list[int]):
        obs = []
        for i in chunk_idxs:
            array_idx = self.array_idxs[i]
            obs.append(
                self._obs[array_idx][self._obs_column]
                .iloc[self.chunks_slices[i]]
                .to_numpy()
            )
        return obs

    def __iter__(self):
        chunks = np.arange(len(self.chunks))
        if self._shuffle:
            self._worker_handle.shuffle(chunks)
        chunks = self._worker_handle.get_part_for_worker(chunks)

        for batch in _batched(chunks, self._preload_nchunks):
            yield from sample_rows(
                list(zsync.sync(self._fetch_chunks_x(batch))),
                self._fetch_chunks_obs(batch),
                self._shuffle,
            )

    def __len__(self):
        return self._n_obs


# TODO: make this part of the public zarr or zarrs-python API.
# We can do chunk coalescing in zarrs based on integer arrays, so I think
# there would make sense with ezclump or similar.
# Another "solution" would be for zarrs to support integer indexing properly, if that pipeline works,
# or make this an "experimental setting" and to use integer indexing for the zarr-python pipeline.
# See: https://github.com/zarr-developers/zarr-python/issues/3175 for why this is better than simpler alternatives.
class MultiBasicIndexer(zarr.core.indexing.Indexer):
    def __init__(self, indexers: list[zarr.core.indexing.Indexer]):
        self.shape = tuple(
            sum(i.shape[k] for i in indexers) for k in range(len(indexers[0].shape))
        )
        self.drop_axes = indexers[0].drop_axes  # maybe?
        self.indexers = indexers

    def __iter__(self):
        total = 0
        for i in self.indexers:
            for c in i:
                gap = c[2][0].stop - c[2][0].start
                yield type(c)(c[0], c[1], (slice(total, total + gap)), c[3])
                total += gap


class CSRDatasetElems(NamedTuple):
    indptr: np.ndarray
    indices: zarr.AsyncArray
    data: zarr.AsyncArray


class ZarrSparseDataset(IterableDataset):
    def __init__(
        self,
        *,
        chunk_size: int = 512,
        preload_nchunks: int = 32,
        shuffle: bool = True,
    ):
        """A loader for on-disk sparse data.

        This loader batches together slice requests to the underlying sparse stores to acheive higher performance.
        This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
        At the moment, the loader is agnostic to the on-disk chunking/sharding, but initial tests show excellent performance for
        sharded data/indices where the shards are extremely small (only ~30,000 elements).

        Args:
            chunk_size: The obs size (i.e., axis 0) of contiguous array data to fetch, by default 512
            preload_nchunks: The number of chunks of contiguous array data to fetch, by default 32
            shuffle: Whether or not to shuffle the data, by default True
        """
        check_lt_1(
            [chunk_size, preload_nchunks],
            ["Chunk size", "Preload chunks"],
        )
        self._anndata_manager: DatasetManager[ad.abc.CSRDataset] = DatasetManager()
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._shuffle = shuffle
        self._dataset_elem_cache: dict[int, CSRDatasetElems] = {}
        self._worker_handle = WorkerHandle()

    def add_anndatas(
        self, adatas: list[ad.AnnData], layer_keys: list[str | None] | None = None
    ) -> ZarrSparseDataset:
        """Append anndata datasets to this loader.

        Args:
            adatas: List of :class:`anndata.AnnData` objects.
            layer_keys: Keys for getting the underlying data out of the anndata object.
                None within the list of keys means using :attr:`~anndata.AnnData.X` while a string value gets from :attr:`~anndata.AnnData.layers`.
                If not provided, all :class:`~anndata.AnnData` objects will have their data taken from :attr:`~anndata.AnnData.X`.
                Defaults to None.

        Returns:
            The iterator with the anndatas added.
        """
        self._anndata_manager.add_anndatas(adatas, layer_keys)
        return self

    def add_anndata(
        self, adata: ad.AnnData, layer_key: str | None = None
    ) -> ZarrSparseDataset:
        """Append an anndata dataset to this loader.

        Args:
            adata: :class:`anndata.AnnData` object.
            layer_key: Keys for getting the underlying data out of the anndata object.
                None means using :attr:`~anndata.AnnData.X` while a string value gets from :attr:`~anndata.AnnData.layers`.
                Defaults to None.

        Returns:
            The iterator with the anndata added.
        """
        self._anndata_manager.add_anndata(adata, layer_key)
        return self

    async def _create_sparse_elems(self, idx: int) -> CSRDatasetElems:
        """Fetch the in-memory indptr, and backed indices and data for a given dataset index.

        Args:
            idx: The index

        Returns:
            The constituent elems of the CSR dataset.
        """
        indptr = await self._anndata_manager.train_datasets[
            idx
        ].group._async_group.getitem("indptr")
        return CSRDatasetElems(
            *(
                await asyncio.gather(
                    indptr.getitem(Ellipsis),
                    self._anndata_manager.train_datasets[
                        idx
                    ].group._async_group.getitem("indices"),
                    self._anndata_manager.train_datasets[
                        idx
                    ].group._async_group.getitem("data"),
                )
            )
        )

    async def _ensure_cache(self):
        """Build up the cache of datasets i.e., in-memory indptr, and backed indices and data."""
        arr_idxs = [
            idx
            for idx in range(len(self._anndata_manager.train_datasets))
            if idx not in self._dataset_elem_cache
        ]
        all_elems = await asyncio.gather(
            *(
                self._create_sparse_elems(idx)
                for idx in range(len(self._anndata_manager.train_datasets))
                if idx not in self._dataset_elem_cache
            )
        )
        for idx, elems in zip(arr_idxs, all_elems):
            self._dataset_elem_cache[idx] = elems

    async def _get_sparse_elems(self, dataset_idx: int) -> CSRDatasetElems:
        """Return the arrays (zarr or otherwise) needed to represent on-disk data at a given index.

        Args:
            dataset_idx: The index of the dataset whose arrays are sought.

        Returns:
            The arrays representing the sparse data.
        """
        if dataset_idx not in self._dataset_elem_cache:
            self._ensure_cache()
        return self._dataset_elem_cache[dataset_idx]

    async def _fetch_data(
        self,
        slices: list[slice],
        dataset_idx: int,
    ) -> sp.csr_matrix:
        """Fetch the data for given slices and the arrays representing a sparse dataset on-disk.

        See https://github.com/scverse/anndata/blob/361325fc621887bf4f381e9412b150fcff599ff7/src/anndata/_core/sparse_dataset.py#L272-L295
        for the inspiration of this function.

        Args:
            slices: The indexing slices to fetch.
            dataset_idx: The index of the dataset to fetch from.

        Returns:
            The in-memory csr data.
        """
        indptr, indices, data = await self._get_sparse_elems(dataset_idx)
        indptr_indices = [indptr[slice(s.start, s.stop + 1)] for s in slices]
        indptr_limits = [slice(i[0], i[-1]) for i in indptr_indices]
        indexer = MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer(
                    (l,), shape=data.metadata.shape, chunk_grid=data.metadata.chunk_grid
                )
                for l in indptr_limits
            ]
        )
        data_np, indices_np = await asyncio.gather(
            data._get_selection(
                indexer, prototype=zarr.core.buffer.default_buffer_prototype()
            ),
            indices._get_selection(
                indexer, prototype=zarr.core.buffer.default_buffer_prototype()
            ),
        )
        gaps = (s1.start - s0.stop for s0, s1 in pairwise(indptr_limits))
        offsets = accumulate(chain([indptr_limits[0].start], gaps))
        start_indptr = indptr_indices[0] - next(offsets)
        if len(slices) < 2:  # there is only one slice so no need to concatenate
            return sp.csr_matrix(
                (data_np, indices_np, start_indptr),
                shape=(start_indptr.shape[0] - 1, self._anndata_manager.n_var),
            )
        end_indptr = np.concatenate(
            [s[1:] - o for s, o in zip(indptr_indices[1:], offsets, strict=True)]
        )
        indptr_np = np.concatenate([start_indptr, end_indptr])
        return sp.csr_matrix(
            (data_np, indices_np, indptr_np),
            shape=(indptr_np.shape[0] - 1, self._anndata_manager.n_var),
        )

    def __iter__(self) -> Iterator[sp.csr_matrix]:
        """Iterate over the on-disk csr datasets.

        Yields:
            A one-row sparse matrix.
        """
        zsync.sync(self._ensure_cache())
        yield from self._anndata_manager.iter(
            self._chunk_size,
            self._worker_handle,
            self._preload_nchunks,
            self._shuffle,
            self._fetch_data,
        )

    def __len__(self) -> int:
        return self._anndata_manager.n_obs
