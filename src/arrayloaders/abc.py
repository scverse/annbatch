from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Generic

from arrayloaders.anndata_manager import AnnDataManager
from arrayloaders.types import InputInMemoryArray, OnDiskArray, OutputInMemoryArray
from arrayloaders.utils import (
    WorkerHandle,
    add_anndata_docstring,
    add_anndatas_docstring,
    add_dataset_docstring,
    add_datasets_docstring,
    check_lt_1,
)

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
        preload_to_gpu: bool = True,
    ):
        """A loader for on-disk {array_type} data.

        This loader batches together slice requests to the underlying {array_type} stores to achieve higher performance.
        This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
        The loader is agnostic to the on-disk chunking/sharding, but it may be advisable to align with the in-memory chunk size for dense.

        The dataset class on its own is quite performant for "chunked loading" i.e., `chunk_size > 1`.
        When `chunk_size == 1`, a :class:`torch.utils.data.DataLoader` should wrap the dataset object.
        In this case, do not use the `add_anndata` or `add_anndatas` option due to https://github.com/scverse/anndata/issues/2021.
        Instead use :func:`anndata.io.sparse_dataset` or :func:`zarr.open` to only get the array you need.


        Parameters
        ----------
            chunk_size
                The obs size (i.e., axis 0) of contiguous array data to fetch.
            preload_nchunks
                The number of chunks of contiguous array data to fetch.
            shuffle
                Whether or not to shuffle the data.
            return_index
                Whether or not to yield the index on each iteration.
            batch_size
                Batch size to yield from the dataset.
            preload_to_gpu
                Whether or not to use cupy for non-io array operations like vstack and indexing.
                This option entails greater GPU memory usage.
                Setting this to `False` is advisable when using the :class:`torch.utils.data.DataLoader` wrapper or potentially with dense data.

        Examples
        --------
            >>> from arrayloaders import {child_class}
            >>> ds = {child_class}(
                    batch_size=4096,
                    chunk_size=32,
                    preload_nchunks=512,
                ).add_anndata(my_anndata)
            >>> for batch in ds:
                    do_fit(batch)
        """
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

        Parameters
        ----------
            slices: The indexing slices to fetch.
            dataset_idx: The index of the dataset to fetch from.

        Returns
        -------
            The in-memory array data.
        """
        ...

    # TODO: validations once the sparse and dense are merged with the AnnDataManager
    def add_anndatas(  # noqa: D102
        self,
        adatas: list[ad.AnnData],
        layer_keys: list[str | None] | str | None = None,
        obs_keys: list[str] | str | None = None,
    ) -> Self:
        self._dataset_manager.add_anndatas(adatas, layer_keys=layer_keys, obs_keys=obs_keys)
        return self

    def add_anndata(  # noqa: D102
        self,
        adata: ad.AnnData,
        layer_key: str | None = None,
        obs_key: str | None = None,
    ) -> Self:
        self._dataset_manager.add_anndata(adata, layer_key=layer_key, obs_key=obs_key)
        return self

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
        tuple[OutputInMemoryArray, None | np.ndarray] | tuple[OutputInMemoryArray, None | np.ndarray, np.ndarray]
    ]:
        """Iterate over the on-disk datasets, returning :class:`{gpu_array}` or :class:`{cpu_array}` depending on whether or not `preload_to_gpu` is set.

        Yields
        ------
            An in-memory array optionally with its label and location in the global store.
        """
        yield from self._dataset_manager.iter(
            self._chunk_size,
            self._worker_handle,
            self._preload_nchunks,
            self._shuffle,
            self._fetch_data,
        )


AbstractIterableDataset.add_dataset.__doc__ = add_dataset_docstring
AbstractIterableDataset.add_datasets.__doc__ = add_datasets_docstring
AbstractIterableDataset.add_anndata.__doc__ = add_anndata_docstring
AbstractIterableDataset.add_anndatas.__doc__ = add_anndatas_docstring


def _assign_methods_to_ensure_unique_docstrings(typ):
    """Because both children AbstractIterableDataset inherit but do not override the methods listed, they need to be copied to ensure unique docstrings"""
    for name in ["add_datasets", "add_dataset", "add_anndatas", "add_anndata", "__init__", "__iter__"]:

        @wraps(getattr(AbstractIterableDataset, name))
        def func(self, *args, name=name, **kwargs):
            return getattr(super(typ, self), name)(*args, **kwargs)

        setattr(typ, name, func)
