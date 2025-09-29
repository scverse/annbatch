from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import wraps
from importlib.util import find_spec
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
    import torch


class AbstractIterableDataset(Generic[OnDiskArray, InputInMemoryArray], metaclass=ABCMeta):  # noqa: D101
    _shuffle: bool
    _preload_nchunks: int
    _worker_handle: WorkerHandle
    _chunk_size: int
    _dataset_manager: AnnDataManager[OnDiskArray, InputInMemoryArray]

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        preload_nchunks: int = 32,
        shuffle: bool = True,
        return_index: bool = False,
        batch_size: int = 1,
        preload_to_gpu: bool = True,
        drop_last: bool = False,
        to_torch: bool = find_spec("torch") is not None,
    ):
        """A loader for on-disk {array_type} data.

        This loader batches together slice requests to the underlying {array_type} stores to achieve higher performance.
        This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
        The loader is agnostic to the on-disk chunking/sharding, but it may be advisable to align with the in-memory chunk size for dense.

        The dataset class on its own is quite performant for "chunked loading" i.e., `chunk_size > 1`.
        When `chunk_size == 1`, a :class:`torch.utils.data.DataLoader` should wrap the dataset object.
        In this case, do not use the `add_anndata` or `add_anndatas` option due to https://github.com/scverse/anndata/issues/2021.
        Instead use :func:`anndata.io.sparse_dataset` or :func:`zarr.open` to only get the array you need.

        If `preload_to_gpu` to True and `to_torch` is False, the yielded type is a `cupy` matrix.
        If `to_torch` is True, the yielded type is a :class:`torch.Tensor`.
        If both `preload_to_gpu` and `to_torch` are False, then the return type is the CPU class for {array_type}.

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
                Whether or not to use cupy for non-io array operations like vstack and indexing once the data is in memory internally.
                This option entails greater GPU memory usage, but is faster at least for sparse operations.
                :func:`torch.vstack` does not support CSR sparse matrices, hence the current use of cupy internally.
                Setting this to `False` is advisable when using the :class:`torch.utils.data.DataLoader` wrapper or potentially with dense data.
                For top performance, this should be used in conjuction with `to_torch` and then :meth:`torch.Tensor.to_dense` if you wish to denseify.
            drop_last
                Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
                If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
                Leave as False when using in conjunction with a :class:`torch.utils.data.DataLoader`.
            to_torch
                Whether to return `torch.Tensor` as the output.
                Data transferred should be 0-copy independent of source, and transfer to cuda when applicable is non-blocking.
                Defaults to True if `torch` is installed.

        Examples
        --------
            >>> from arrayloaders import {child_class}
            >>> ds = {child_class}(
                    batch_size=4096,
                    chunk_size=32,
                    preload_nchunks=512,
                ).add_anndata(my_anndata)
            >>> for batch in ds:
                    # optionally convert to dense
                    # batch = batch.to_dense()
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
                "Cannot yield batches bigger than the iterated in-memory size i.e., batch_size > (chunk_size * preload_nchunks)."
            )

        for package, arg, arg_name in [
            ("torch", to_torch, f"{to_torch=}"),
            ("cupy", preload_to_gpu, f"{preload_to_gpu=}"),
        ]:
            if arg and not find_spec(package):
                raise ImportError(
                    f"Could not find {package} dependency even though {arg_name}.  Try `uv pip install {package}`"
                )
        self._dataset_manager = AnnDataManager(
            # TODO: https://github.com/scverse/anndata/issues/2021
            # on_add=self._cache_update_callback,
            return_index=return_index,
            batch_size=batch_size,
            preload_to_gpu=preload_to_gpu,
            drop_last=drop_last,
            to_torch=to_torch,
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
        tuple[OutputInMemoryArray, None | np.ndarray]
        | tuple[OutputInMemoryArray | torch.Tensor, None | np.ndarray, np.ndarray]
    ]:
        """
        Iterate over the on-disk datasets, returning :class:`{gpu_array}` or :class:`{cpu_array}` depending on whether or not `preload_to_gpu` is set.

        Will convert to a :class:`torch.Tensor` if `to_torch` is True.

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
