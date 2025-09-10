from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic

from arrayloaders.anndata_manager import AnnDataManager
from arrayloaders.types import InputInMemoryArray, OnDiskArray, OutputInMemoryArray
from arrayloaders.utils import WorkerHandle, add_dataset_docstring, check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    import anndata as ad
    import numpy as np


class AbstractIterableDataset(Generic[OnDiskArray, InputInMemoryArray, OutputInMemoryArray], metaclass=ABCMeta):  # noqa: D101
    _shuffle: bool
    _preload_nchunks: int
    _worker_handle: WorkerHandle
    _chunk_size: int
    _dataset_manager: AnnDataManager[OnDiskArray, InputInMemoryArray, OutputInMemoryArray]

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        preload_nchunks: int = 32,
        shuffle: bool = True,
        return_index: bool = False,
        batch_size: int = 1,
        preload_to_gpu: bool = False,
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
        self._dataset_manager = AnnDataManager(
            # TODO: https://github.com/scverse/anndata/issues/2021
            # on_add=self._cache_update_callback,
            return_index=return_index,
            batch_size=batch_size,
            preload_to_gpu=preload_to_gpu,
        )
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._shuffle = shuffle
        self._worker_handle = WorkerHandle()

    async def _cache_update_callback(self):
        pass

    @abstractmethod
    async def _fetch_data(self, slices: list[slice], dataset_idx: int) -> InputInMemoryArray:
        """Fetch the data for given slices and the arrays representing a dataset on-disk.

        Args:
            slices: The indexing slices to fetch.
            dataset_idx: The index of the dataset to fetch from.

        Returns
        -------
            The in-memory array data.
        """
        ...

    def add_anndatas(  # noqa: D102
        self,
        adatas: list[ad.AnnData],
        layer_keys: list[str | None] | str | None = None,
        obs_keys: list[str] | str | None = None,
    ) -> Self:
        raise NotImplementedError("See https://github.com/scverse/anndata/issues/2021")

    def add_anndata(  # noqa: D102
        self,
        adata: ad.AnnData,
        layer_key: str | None = None,
        obs_key: str | None = None,
    ) -> Self:
        raise NotImplementedError("See https://github.com/scverse/anndata/issues/2021")

    @abstractmethod
    def _validate(self, datasets: list[OnDiskArray]) -> None: ...

    def add_datasets(self, datasets: list[OnDiskArray], obs: list[np.ndarray] | None = None) -> Self:  # noqa: D102
        self._validate(datasets)
        self._dataset_manager.add_datasets(datasets, obs)
        return self

    def add_dataset(self, dataset: OnDiskArray, obs: np.ndarray | None = None) -> Self:  # noqa: D102
        self._validate([dataset])
        self._dataset_manager.add_dataset(dataset, obs)
        return self

    def __len__(self) -> int:
        return self._dataset_manager.n_obs

    def __iter__(
        self,
    ) -> Iterator[
        tuple[InputInMemoryArray, None | np.ndarray] | tuple[InputInMemoryArray, None | np.ndarray, np.ndarray]
    ]:
        """Iterate over the on-disk datasets.

        Yields
        ------
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
