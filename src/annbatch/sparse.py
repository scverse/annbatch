from __future__ import annotations

import asyncio
from itertools import accumulate, chain, pairwise
from typing import NamedTuple

import anndata as ad
import numpy as np
import zarr
import zarr.core.sync as zsync
from torch.utils.data import IterableDataset

from annbatch.abc import AbstractIterableDataset
from annbatch.utils import CSRContainer, MultiBasicIndexer, __init_docstring__


class CSRDatasetElems(NamedTuple):
    """Container for cached objects that will be indexed into to generate CSR matrices"""

    indptr: np.ndarray
    indices: zarr.AsyncArray
    data: zarr.AsyncArray


class ZarrSparseDataset(AbstractIterableDataset, IterableDataset):  # noqa: D101
    _dataset_elem_cache: dict[int, CSRDatasetElems] = {}

    def _cache_update_callback(self):
        """Callback for when datasets are added to ensure the cache is updated."""
        return zsync.sync(self._ensure_cache())

    def _validate(self, datasets: list[ad.abc.CSRDataset]):
        if not all(isinstance(d, ad.abc.CSRDataset) for d in datasets):
            raise TypeError("Cannot create sparse dataset using CSRDataset data")

    async def _create_sparse_elems(self, idx: int) -> CSRDatasetElems:
        """Fetch the in-memory indptr, and backed indices and data for a given dataset index.

        Args:
            idx: The index

        Returns
        -------
            The constituent elems of the CSR dataset.
        """
        indptr = await self._dataset_manager.train_datasets[idx].group._async_group.getitem("indptr")
        return CSRDatasetElems(
            *(
                await asyncio.gather(
                    indptr.getitem(Ellipsis),
                    self._dataset_manager.train_datasets[idx].group._async_group.getitem("indices"),
                    self._dataset_manager.train_datasets[idx].group._async_group.getitem("data"),
                )
            )
        )

    async def _ensure_cache(self):
        """Build up the cache of datasets i.e., in-memory indptr, and backed indices and data."""
        arr_idxs = [
            idx for idx in range(len(self._dataset_manager.train_datasets)) if idx not in self._dataset_elem_cache
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

        Returns
        -------
            The arrays representing the sparse data.
        """
        if dataset_idx not in self._dataset_elem_cache:
            await self._ensure_cache()
        return self._dataset_elem_cache[dataset_idx]

    async def _fetch_data(
        self,
        slices: list[slice],
        dataset_idx: int,
    ) -> CSRContainer:
        # See https://github.com/scverse/anndata/blob/361325fc621887bf4f381e9412b150fcff599ff7/src/anndata/_core/sparse_dataset.py#L272-L295
        # for the inspiration of this function.
        indptr, indices, data = await self._get_sparse_elems(dataset_idx)
        indptr_indices = [indptr[slice(s.start, s.stop + 1)] for s in slices]
        indptr_limits = [slice(i[0], i[-1]) for i in indptr_indices]
        indexer = MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer((l,), shape=data.metadata.shape, chunk_grid=data.metadata.chunk_grid)
                for l in indptr_limits
            ]
        )
        data_np, indices_np = await asyncio.gather(
            data._get_selection(indexer, prototype=zarr.core.buffer.default_buffer_prototype()),
            indices._get_selection(indexer, prototype=zarr.core.buffer.default_buffer_prototype()),
        )
        gaps = (s1.start - s0.stop for s0, s1 in pairwise(indptr_limits))
        offsets = accumulate(chain([indptr_limits[0].start], gaps))
        start_indptr = indptr_indices[0] - next(offsets)
        if len(slices) < 2:  # there is only one slice so no need to concatenate
            return CSRContainer(
                elems=(data_np, indices_np, start_indptr),
                shape=(start_indptr.shape[0] - 1, self._dataset_manager.n_var),
            )
        end_indptr = np.concatenate([s[1:] - o for s, o in zip(indptr_indices[1:], offsets, strict=True)])
        indptr_np = np.concatenate([start_indptr, end_indptr])
        return CSRContainer(
            elems=(data_np, indices_np, indptr_np),
            shape=(indptr_np.shape[0] - 1, self._dataset_manager.n_var),
        )


ZarrSparseDataset.__init__.__doc__ = __init_docstring__.format(array_type="sparse")
