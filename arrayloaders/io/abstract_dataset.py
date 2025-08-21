from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic

from .anndata_manager import AnnDataManager, add_dataset_docstring
from .types import BackingArray, InMemoryArray
from .utils import WorkerHandle, check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    import anndata as ad
    import numpy as np
    from scipy import sparse as sp


class AbstractIterableDataset(Generic[BackingArray, InMemoryArray], metaclass=ABCMeta):
    _shuffle: bool
    _preload_nchunks: int
    _worker_handle: WorkerHandle
    _chunk_size: int
    _dataset_manager: AnnDataManager[BackingArray, InMemoryArray]

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        preload_nchunks: int = 32,
        shuffle: bool = True,
        return_index: bool = False,
        batch_size: int = 1,
    ):
        check_lt_1(
            [
                chunk_size,
                preload_nchunks,
            ],
            ["Chunk size", "Preload chunks"],
        )
        if batch_size > (chunk_size * preload_nchunks):
            raise NotImplementedError(
                "If you need batch loading that is bigger than the iterated in-memory size, please open an issue."
            )
        self._dataset_manager: AnnDataManager[ad.abc.CSRDataset, sp.csr_matrix] = (
            AnnDataManager(
                # TODO: https://github.com/scverse/anndata/issues/2021
                # on_add=self._cache_update_callback,
                return_index=return_index,
                batch_size=batch_size,
            )
        )
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._shuffle = shuffle
        self._worker_handle = WorkerHandle()

    async def _cache_update_callback(self):
        pass

    @abstractmethod
    async def _fetch_data(self, slices: list[slice], dataset_idx: int) -> InMemoryArray:
        """Fetch the data for given slices and the arrays representing a dataset on-disk.

        Args:
            slices: The indexing slices to fetch.
            dataset_idx: The index of the dataset to fetch from.

        Returns:
            The in-memory array data.
        """
        ...

    def add_anndatas(
        self,
        adatas: list[ad.AnnData],
        layer_keys: list[str | None] | str | None = None,
        obs_keys: list[str] | str | None = None,
    ) -> Self:
        raise NotImplementedError("See https://github.com/scverse/anndata/issues/2021")

    def add_anndata(
        self,
        adata: ad.AnnData,
        layer_key: str | None = None,
        obs_key: str | None = None,
    ) -> Self:
        raise NotImplementedError("See https://github.com/scverse/anndata/issues/2021")

    def add_datasets(
        self, datasets: list[BackingArray], obs: list[np.ndarray] | None = None
    ) -> Self:
        self._dataset_manager.add_datasets(datasets, obs)
        return self

    def add_dataset(self, dataset: BackingArray, obs: np.ndarray | None = None) -> Self:
        self._dataset_manager.add_dataset(dataset, obs)
        return self

    def __len__(self) -> int:
        return self._dataset_manager.n_obs

    def __iter__(
        self,
    ) -> Iterator[
        tuple[InMemoryArray, None | np.ndarray]
        | tuple[InMemoryArray, None | np.ndarray, np.ndarray]
    ]:
        """Iterate over the on-disk datasets.

        Yields:
            A one-row in-memory array optionally with its label.
        """
        yield from self._dataset_manager.iter(
            self._chunk_size,
            self._worker_handle,
            self._preload_nchunks,
            self._shuffle,
            self._fetch_data,
        )


AbstractIterableDataset.add_dataset.__doc__ = add_dataset_docstring
AbstractIterableDataset.add_datasets.__doc__ = add_dataset_docstring
