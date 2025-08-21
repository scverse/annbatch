from __future__ import annotations

import asyncio
from itertools import accumulate, chain, pairwise
from typing import NamedTuple, cast

import numpy as np
import zarr
import zarr.core.sync as zsync
from scipy import sparse as sp
from torch.utils.data import IterableDataset

from arrayloaders.io.abstract_dataset import AbstractIterableDataset

__init_docstring__ = """A loader for on-disk {array_type} data.

This loader batches together slice requests to the underlying {array_type} stores to acheive higher performance.
This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
The loader is agnostic to the on-disk chunking/sharding, but it may be advisable to align with the in-memory chunk size.

Args:
    chunk_size: The obs size (i.e., axis 0) of contiguous array data to fetch, by default 512
    preload_nchunks: The number of chunks of contiguous array data to fetch, by default 32
    shuffle: Whether or not to shuffle the data, by default True
    return_index: Whether or not to return the index on each iteration, by default False
"""


# TODO: make this part of the public zarr or zarrs-python API.
# We can do chunk coalescing in zarrs based on integer arrays, so I think
# there would make sense with ezclump or similar.
# Another "solution" would be for zarrs to support integer indexing properly, if that pipeline works,
# or make this an "experimental setting" and to use integer indexing for the zarr-python pipeline.
# See: https://github.com/zarr-developers/zarr-python/issues/3175 for why this is better than simpler alternatives.
class MultiBasicIndexer(zarr.core.indexing.Indexer):
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
                yield type(c)(
                    c[0], c[1], (slice(total, total + gap), *out_selection[1:]), c[3]
                )
                total += gap


class ZarrDenseDataset(AbstractIterableDataset, IterableDataset):
    async def _fetch_data(self, slices: list[slice], dataset_idx: int) -> np.ndarray:
        dataset = self._dataset_manager.train_datasets[dataset_idx]
        indexer = MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer(
                    (s, Ellipsis),
                    shape=dataset.metadata.shape,
                    chunk_grid=dataset.metadata.chunk_grid,
                )
                for s in slices
            ]
        )
        res = cast(
            "np.ndarray",
            await dataset._async_array._get_selection(
                indexer, prototype=zarr.core.buffer.default_buffer_prototype()
            ),
        )
        return res


ZarrDenseDataset.__init__.__doc__ = __init_docstring__.format(array_type="dense")


class CSRDatasetElems(NamedTuple):
    indptr: np.ndarray
    indices: zarr.AsyncArray
    data: zarr.AsyncArray


class ZarrSparseDataset(AbstractIterableDataset, IterableDataset):
    _dataset_elem_cache: dict[int, CSRDatasetElems] = {}

    def _cache_update_callback(self):
        """Callback for when datasets are added to ensure the cache is updated."""
        return zsync.sync(self._ensure_cache())

    async def _create_sparse_elems(self, idx: int) -> CSRDatasetElems:
        """Fetch the in-memory indptr, and backed indices and data for a given dataset index.

        Args:
            idx: The index

        Returns:
            The constituent elems of the CSR dataset.
        """
        indptr = await self._dataset_manager.train_datasets[
            idx
        ].group._async_group.getitem("indptr")
        return CSRDatasetElems(
            *(
                await asyncio.gather(
                    indptr.getitem(Ellipsis),
                    self._dataset_manager.train_datasets[
                        idx
                    ].group._async_group.getitem("indices"),
                    self._dataset_manager.train_datasets[
                        idx
                    ].group._async_group.getitem("data"),
                )
            )
        )

    async def _ensure_cache(self):
        """Build up the cache of datasets i.e., in-memory indptr, and backed indices and data."""
        arr_idxs = [
            idx
            for idx in range(len(self._dataset_manager.train_datasets))
            if idx not in self._dataset_elem_cache
        ]
        all_elems = await asyncio.gather(
            *(
                self._create_sparse_elems(idx)
                for idx in range(len(self._dataset_manager.train_datasets))
                if idx not in self._dataset_elem_cache
            )
        )
        for idx, elems in zip(arr_idxs, all_elems, strict=True):
            self._dataset_elem_cache[idx] = elems

    async def _get_sparse_elems(self, dataset_idx: int) -> CSRDatasetElems:
        """Return the arrays (zarr or otherwise) needed to represent on-disk data at a given index.

        Args:
            dataset_idx: The index of the dataset whose arrays are sought.

        Returns:
            The arrays representing the sparse data.
        """
        if dataset_idx not in self._dataset_elem_cache:
            await self._ensure_cache()
        return self._dataset_elem_cache[dataset_idx]

    async def _fetch_data(
        self,
        slices: list[slice],
        dataset_idx: int,
    ) -> sp.csr_matrix:
        # See https://github.com/scverse/anndata/blob/361325fc621887bf4f381e9412b150fcff599ff7/src/anndata/_core/sparse_dataset.py#L272-L295
        # for the inspiration of this function.
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
                shape=(start_indptr.shape[0] - 1, self._dataset_manager.n_var),
            )
        end_indptr = np.concatenate(
            [s[1:] - o for s, o in zip(indptr_indices[1:], offsets, strict=True)]
        )
        indptr_np = np.concatenate([start_indptr, end_indptr])
        return sp.csr_matrix(
            (data_np, indices_np, indptr_np),
            shape=(indptr_np.shape[0] - 1, self._dataset_manager.n_var),
        )


ZarrSparseDataset.__init__.__doc__ = __init_docstring__.format(array_type="sparse")
