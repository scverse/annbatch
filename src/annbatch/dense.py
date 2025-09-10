from __future__ import annotations

from typing import TYPE_CHECKING, cast

import zarr
from torch.utils.data import IterableDataset

from annbatch.abc import AbstractIterableDataset
from annbatch.utils import MultiBasicIndexer

if TYPE_CHECKING:
    import numpy as np


class ZarrDenseDataset(AbstractIterableDataset, IterableDataset):  # noqa: D101
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
            await dataset._async_array._get_selection(indexer, prototype=zarr.core.buffer.default_buffer_prototype()),
        )
        return res

    def _validate(self, datasets: list[zarr.Array]):
        if not all(isinstance(d, zarr.Array) for d in datasets):
            raise TypeError("Cannot create dense dataset without using a zarr.Array")
