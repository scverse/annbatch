from __future__ import annotations

from importlib.util import find_spec
from typing import cast

import numpy as np
import zarr

if find_spec("torch"):
    from torch.utils.data import IterableDataset as _IterableDataset
else:

    class _IterableDataset:
        pass


from annbatch.abc import AbstractIterableDataset, _assign_methods_to_ensure_unique_docstrings
from annbatch.utils import (
    MultiBasicIndexer,
    add_anndata_docstring,
    add_anndatas_docstring,
    add_dataset_docstring,
    add_datasets_docstring,
)


class ZarrDenseDataset(AbstractIterableDataset[zarr.Array, np.ndarray], _IterableDataset):  # noqa: D101
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

    def _cache_update_callback(self) -> None:
        pass


_assign_methods_to_ensure_unique_docstrings(ZarrDenseDataset)


ZarrDenseDataset.__doc__ = AbstractIterableDataset.__init__.__doc__.format(
    array_type="dense", child_class="ZarrDenseDataset"
)
ZarrDenseDataset.add_datasets.__doc__ = add_datasets_docstring.format(on_disk_array_type="zarr.Array")
ZarrDenseDataset.add_dataset.__doc__ = add_dataset_docstring.format(on_disk_array_type="zarr.Array")
ZarrDenseDataset.add_anndatas.__doc__ = add_anndatas_docstring.format(on_disk_array_type="zarr.Array")
ZarrDenseDataset.add_anndata.__doc__ = add_anndata_docstring.format(on_disk_array_type="zarr.Array")
ZarrDenseDataset.__iter__.__doc__ = AbstractIterableDataset.__iter__.__doc__.format(
    gpu_array="cupy.ndarray", cpu_array="numpy.ndarray"
)
