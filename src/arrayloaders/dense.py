from __future__ import annotations

from types import NoneType
from typing import cast

import numpy as np
import zarr
from torch.utils.data import IterableDataset

from arrayloaders.abc import AbstractIterableDataset, _assign_add_methods
from arrayloaders.utils import (
    MultiBasicIndexer,
    __init_docstring__,
    add_anndata_docstring,
    add_anndatas_docstring,
    add_dataset_docstring,
    add_datasets_docstring,
)

try:
    from cupy import ndarray as CupyArray
except ImportError:
    CupyArray = NoneType


class ZarrDenseDataset(AbstractIterableDataset[zarr.Array, np.ndarray, CupyArray | np.ndarray], IterableDataset):  # noqa: D101
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


_assign_add_methods(ZarrDenseDataset)


ZarrDenseDataset.__doc__ = __init_docstring__.format(array_type="dense")
ZarrDenseDataset.add_datasets.__doc__ = add_datasets_docstring.format(on_disk_array_type="zarr.Array")
ZarrDenseDataset.add_dataset.__doc__ = add_dataset_docstring.format(on_disk_array_type="zarr.Array")
ZarrDenseDataset.add_anndatas.__doc__ = add_anndatas_docstring.format(on_disk_array_type="zarr.Array")
ZarrDenseDataset.add_anndata.__doc__ = add_anndata_docstring.format(on_disk_array_type="zarr.Array")
