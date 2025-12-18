from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, defaultdict
from functools import singledispatchmethod
from importlib.util import find_spec
from itertools import accumulate, chain, pairwise
from typing import TYPE_CHECKING, NamedTuple, Self, cast

import anndata as ad
import numpy as np
import pandas as pd
import zarr
import zarr.core.sync as zsync
from scipy import sparse as sp
from zarr import Array as ZarrArray

from annbatch.types import BackingArray_T, InputInMemoryArray_T, LoaderOutput, OutputInMemoryArray_T
from annbatch.utils import (
    CSRContainer,
    MultiBasicIndexer,
    WorkerHandle,
    _batched,
    check_lt_1,
    check_var_shapes,
    split_given_size,
    to_torch,
)

from .compat import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    # TODO: remove after sphinx 9 - myst compat
    BackingArray = BackingArray_T
    OutputInMemoryArray = OutputInMemoryArray_T
    InputInMemoryArray = InputInMemoryArray_T


class CSRDatasetElems(NamedTuple):
    """Container for cached objects that will be indexed into to generate CSR matrices"""

    indptr: np.ndarray
    indices: zarr.AsyncArray
    data: zarr.AsyncArray


class Loader[
    BackingArray: BackingArray_T,
    InputInMemoryArray: InputInMemoryArray_T,
    OutputInMemoryArray: OutputInMemoryArray_T,
](IterableDataset):
    """A loader for on-disk data anndata stores.

    This loader batches together slice requests to the underlying stores to achieve higher performance.
    This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
    The loader is agnostic to the on-disk chunking/sharding, but it may be advisable to align with the in-memory chunk size for dense.

    The dataset class on its own is quite performant for "chunked loading" i.e., `chunk_size > 1`.
    When `chunk_size == 1`, a :class:`torch.utils.data.DataLoader` should wrap the dataset object.
    In this case, be sure to use `spawn` multiprocessing int he wrapping loader.

    If `preload_to_gpu` to True and `to_torch` is False, the yielded type is a `cupy` matrix.
    If `to_torch` is True, the yielded type is a :class:`torch.Tensor`.
    If both `preload_to_gpu` and `to_torch` are False, then the return type is the CPU class for the fiven data type.

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
        >>> from annbatch import Loader
        >>> ds = Loader(
                batch_size=4096,
                chunk_size=32,
                preload_nchunks=512,
            ).add_anndata(my_anndata)
        >>> for batch in ds:
                # optionally convert to dense
                # batch = batch.to_dense()
                do_fit(batch)
    """

    _train_datasets: list[BackingArray]
    _obs: list[pd.DataFrame] | None = None
    _return_index: bool = False
    _batch_size: int = 1
    _shapes: list[tuple[int, int]]
    _preload_to_gpu: bool = True
    _drop_last: bool = False
    _to_torch: bool = True
    _shuffle: bool
    _preload_nchunks: int
    _worker_handle: WorkerHandle
    _chunk_size: int
    _dataset_elem_cache: dict[int, CSRDatasetElems]

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        preload_nchunks: int = 32,
        shuffle: bool = True,
        return_index: bool = False,
        batch_size: int = 1,
        preload_to_gpu: bool = find_spec("cupy") is not None,
        drop_last: bool = False,
        to_torch: bool = find_spec("torch") is not None,
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
                "Cannot yield batches bigger than the iterated in-memory size i.e., batch_size > (chunk_size * preload_nchunks)."
            )
        if to_torch and not find_spec("torch"):
            raise ImportError("Could not find torch dependeny. Try `pip install torch`.")
        if preload_to_gpu and not find_spec("cupy"):
            raise ImportError("Follow the directions at https://docs.cupy.dev/en/stable/install.html to install cupy.")
        self._return_index = return_index
        self._batch_size = batch_size
        self._preload_to_gpu = preload_to_gpu
        self._to_torch = to_torch
        self._drop_last = drop_last
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._shuffle = shuffle
        self._worker_handle = WorkerHandle()
        self._train_datasets = []
        self._shapes = []
        self._dataset_elem_cache = {}

    def __len__(self) -> int:
        return self.n_obs

    @property
    def _sp_module(self) -> ModuleType:
        if self._preload_to_gpu:
            try:
                import cupyx.scipy.sparse as cpx  # pragma: no cover

                return cpx
            except ImportError:
                raise ImportError(
                    "Cannot find cupy module even though `preload_to_gpu` argument was set to `True`"
                ) from None
        return sp

    @property
    def _np_module(self) -> ModuleType:
        if self._preload_to_gpu:
            try:
                import cupy as cp

                return cp
            except ImportError:
                raise ImportError(
                    "Cannot find cupy module even though `preload_to_gpu` argument was set to `True`"
                ) from None

        return np

    @property
    def dataset_type(self) -> type[BackingArray]:
        """The type of on-disk data used in this loader.

        Returns
        -------
            The type used.
        """
        return type(self._train_datasets[0])

    @property
    def n_obs(self) -> int:
        """The total number of observations in this instance i.e., the sum of the first axis of all added datasets.

        Returns
        -------
            The number of observations.
        """
        return sum(shape[0] for shape in self._shapes)

    @property
    def n_var(self) -> int:
        """The total number of variables in this instance i.e., the second axis (which is the same) across all datasets.

        Returns
        -------
            The number of variables.
        """
        return self._shapes[0][1]

    def add_anndatas(
        self,
        adatas: list[ad.AnnData],
    ) -> Self:
        """Append anndatas to this dataset.

        Parameters
        ----------
            adatas
                List of :class:`anndata.AnnData` objects, with :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` as the data matrix.
        """
        check_lt_1([len(adatas)], ["Number of anndatas"])
        for adata in adatas:
            self.add_anndata(adata)
        return self

    def add_anndata(self, adata: ad.AnnData) -> Self:
        """Append an anndata to this dataset.

        Parameters
        ----------
            adata
                A :class:`anndata.AnnData` object, with :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` as the data matrix.
        """
        dataset = adata.X
        labels = adata.obs
        if not isinstance(dataset, BackingArray_T.__value__):
            raise TypeError(f"Found {type(dataset)} but only {BackingArray_T.__value__} are usable")
        self.add_dataset(cast("BackingArray", dataset), labels)
        return self

    def add_datasets(self, datasets: list[BackingArray], obs: list[pd.DataFrame] | None = None) -> Self:
        """Append datasets to this dataset.

        Parameters
        ----------
            datasets
                List of :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` objects, generally from :attr:`anndata.AnnData.X`.
                They must all be of the same type and match that of any already added datasets.
            obs
                List of :class:`numpy.ndarray` labels, generally from :attr:`anndata.AnnData.obs`.
        """
        if obs is None:
            obs = [None] * len(datasets)
        for ds, o in zip(datasets, obs, strict=True):
            self.add_dataset(ds, o)
        return self

    def add_dataset(self, dataset: BackingArray, obs: pd.DataFrame | None = None) -> Self:
        """Append a dataset to this dataset.

        Parameters
        ----------
            dataset
                A :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` object, generally from :attr:`anndata.AnnData.X`.
            obs
                :class:`numpy.ndarray` labels, generally from :attr:`anndata.AnnData.obs`.
        """
        if len(self._train_datasets) > 0:
            if self._obs is None and obs is not None:
                raise ValueError(
                    f"Cannot add a dataset with obs label {obs} when training datasets have already been added without labels"
                )
            if self._obs is not None and obs is None:
                raise ValueError(
                    "Cannot add a dataset with no obs label when training datasets have already been added without labels"
                )
            if not isinstance(dataset, self.dataset_type):
                raise ValueError(
                    f"All datasets on a given loader must be of the same type {self.dataset_type} but got {type(dataset)}"
                )
        if not isinstance(dataset, BackingArray_T.__value__):
            raise TypeError(f"Cannot add dataset of type {type(dataset)}")
        if isinstance(dataset, ad.abc.CSRDataset) and not dataset.backend == "zarr":
            raise TypeError(
                "Cannot add CSRDataset backed by h5ad at the moment: see https://github.com/zarr-developers/VirtualiZarr/pull/790"
            )
        if not isinstance(obs, pd.DataFrame):
            raise TypeError("obs must be a pandas DataFrame")
        datasets = self._train_datasets + [dataset]
        check_var_shapes(datasets)
        self._shapes = self._shapes + [dataset.shape]
        self._train_datasets = datasets
        if self._obs is not None:  # labels exist
            self._obs += [obs]
        elif obs is not None:  # labels dont exist yet, but are being added for the first time
            self._obs = [obs]
        return self

    def _get_relative_obs_indices(self, index: slice, *, use_original_space: bool = False) -> list[tuple[slice, int]]:
        """Generate a slice relative to a dataset given a global slice index over all datasets.

        For a given slice indexer of axis 0, return a new slice relative to the on-disk
        data it represents given the number of total observations as well as the index of
        the underlying data on disk from the argument `sparse_datasets` to the initializer.

        For example, given slice index (10, 15), for 4 datasets each with size 5 on axis zero,
        this function returns ((0,5), 2) representing slice (0,5) along axis zero of sparse dataset 2.

        Parameters
        ----------
            index
                The queried slice.
            use_original_space
                Whether the slices should be reindexed against the anndata objects.

        Returns
        -------
            A slice relative to the dataset it represents as well as the index of said dataset in `sparse_datasets`.
        """
        min_idx = index.start
        max_idx = index.stop
        curr_pos = 0
        slices = []
        for idx, (n_obs, _) in enumerate(self._shapes):
            array_start = curr_pos
            array_end = curr_pos + n_obs

            start = max(min_idx, array_start)
            stop = min(max_idx, array_end)
            if start < stop:
                if use_original_space:
                    slices.append((slice(start, stop), idx))
                else:
                    relative_start = start - array_start
                    relative_stop = stop - array_start
                    slices.append((slice(relative_start, relative_stop), idx))
            curr_pos += n_obs
        return slices

    def _slices_to_slices_with_array_index(
        self, slices: list[slice], *, use_original_space: bool = False
    ) -> OrderedDict[int, list[slice]]:
        """Given a list of slices, give the lookup between on-disk datasets and slices relative to that dataset.

        Parameters
        ----------
            slices
                Slices to relative to the on-disk datasets.
            use_original_space
                Whether the slices should be reindexed against the anndata objects.

        Returns
        -------
            A lookup between the dataset and its indexing slices, ordered by keys.
        """
        dataset_index_to_slices: defaultdict[int, list[slice]] = defaultdict(list)
        for slice in slices:
            for relative_obs_indices in self._get_relative_obs_indices(slice, use_original_space=use_original_space):
                dataset_index_to_slices[relative_obs_indices[1]] += [relative_obs_indices[0]]
        keys = sorted(dataset_index_to_slices.keys())
        dataset_index_to_slices_sorted = OrderedDict()
        for k in keys:
            dataset_index_to_slices_sorted[k] = dataset_index_to_slices[k]
        return dataset_index_to_slices_sorted

    def _get_chunks(self, chunk_size: int) -> np.ndarray:
        """Get a potentially shuffled list of chunk ids, accounting for the fact that this dataset might be inside a worker.

        Returns
        -------
            A :class:`numpy.ndarray` of chunk ids.
        """
        chunks = np.arange(math.ceil(self.n_obs / chunk_size))
        if self._shuffle:
            self._worker_handle.shuffle(chunks)

        return self._worker_handle.get_part_for_worker(chunks)

    @singledispatchmethod
    async def _fetch_data(self, dataset: ZarrArray | CSRDatasetElems, slices: list[slice]) -> InputInMemoryArray:
        """Fetch data from an on-disk store.

        Parameters
        ----------
        dataset
            The underlying store.
        slices
            The slices to fetch

        Returns
        -------
            The sparse or dense fetched data.

        Raises
        ------
        NotImplementedError
            If the dataset is not recognized.
        """
        raise NotImplementedError(f"Cannot fetch data for type {type(dataset)}")

    @_fetch_data.register
    async def _fetch_data_dense(self, dataset: ZarrArray, slices: list[slice]) -> np.ndarray:
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

    async def _create_sparse_elems(self, idx: int) -> CSRDatasetElems:
        """Fetch the in-memory indptr, and backed indices and data for a given dataset index.

        Parameters
        ----------
            idx
                The index

        Returns
        -------
            The constituent elems of the CSR dataset.
        """
        if isinstance(ds := self._train_datasets[idx], ZarrArray):
            raise ValueError(f"Requested sparse dataset at idx {idx} of {self._train_datasets} but found dense array")
        indptr = await ds.group._async_group.getitem("indptr")
        return CSRDatasetElems(
            *(
                await asyncio.gather(
                    indptr.getitem(Ellipsis),
                    ds.group._async_group.getitem("indices"),
                    ds.group._async_group.getitem("data"),
                )
            )
        )

    async def _ensure_sparse_cache(self) -> None:
        """Build up the cache of datasets i.e., in-memory indptr, and backed indices and data."""
        arr_idxs = [idx for idx in range(len(self._train_datasets)) if idx not in self._dataset_elem_cache]
        all_elems: list[CSRDatasetElems] = await asyncio.gather(
            *(
                self._create_sparse_elems(idx)
                for idx in range(len(self._train_datasets))
                if idx not in self._dataset_elem_cache
            )
        )
        for idx, elems in zip(arr_idxs, all_elems, strict=True):
            self._dataset_elem_cache[idx] = elems

    def _get_elem_from_cache(self, dataset_idx: int) -> CSRDatasetElems | ZarrArray:
        """Return the arrays (zarr or otherwise) needed to represent on-disk data at a given index.

        Parameters
        ----------
            dataset_idx
                The index of the dataset whose arrays are sought.

        Returns
        -------
            The arrays representing the sparse data.
        """
        if dataset_idx not in self._dataset_elem_cache:
            raise ValueError("Cache not prepared")
        return self._dataset_elem_cache[dataset_idx]

    @_fetch_data.register
    async def _fetch_data_sparse(
        self,
        dataset: CSRDatasetElems,
        slices: list[slice],
    ) -> CSRContainer:
        # See https://github.com/scverse/anndata/blob/361325fc621887bf4f381e9412b150fcff599ff7/src/anndata/_core/sparse_dataset.py#L272-L295
        # for the inspiration of this function.
        indptr, indices, data = dataset
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
                shape=(start_indptr.shape[0] - 1, self.n_var),
                dtype=data_np.dtype,
            )
        end_indptr = np.concatenate([s[1:] - o for s, o in zip(indptr_indices[1:], offsets, strict=True)])
        indptr_np = np.concatenate([start_indptr, end_indptr])
        return CSRContainer(
            elems=(data_np, indices_np, indptr_np),
            shape=(indptr_np.shape[0] - 1, self.n_var),
            dtype=data_np.dtype,
        )

    async def _index_datasets(
        self,
        dataset_index_to_slices: OrderedDict[int, list[slice]],
    ) -> list[InputInMemoryArray]:
        """Helper function meant to encapsulate asynchronous calls so that we can use the same event loop as zarr.

        Parameters
        ----------
            dataset_index_to_slices
                A lookup of the list-placement index of a dataset to the request slices.
            fetch_data
                The function to do the fetching for a given slice-dataset index pair.
        """
        tasks = []
        if is_sparse := issubclass(self.dataset_type, ad.abc.CSRDataset):
            await self._ensure_sparse_cache()
        for dataset_idx in dataset_index_to_slices.keys():
            tasks.append(
                self._fetch_data(
                    self._get_elem_from_cache(dataset_idx) if is_sparse else self._train_datasets[dataset_idx],
                    dataset_index_to_slices[dataset_idx],
                )
            )
        return await asyncio.gather(*tasks)

    def __iter__(
        self,
    ) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
        """Iterate over the on-disk datasets.

        Yields
        ------
            A batch of data along with its labels and index (both optional).
        """
        check_lt_1(
            [len(self._train_datasets), self.n_obs],
            ["Number of datasets", "Number of observations"],
        )
        # In order to handle data returned where (chunk_size * preload_nchunks) mod batch_size != 0
        # we must keep track of the leftover data.
        in_memory_data = None
        concatenated_obs = None
        in_memory_indices = None
        mod = self._sp_module if issubclass(self.dataset_type, ad.abc.CSRDataset) else np
        for chunk_indices in _batched(self._get_chunks(self._chunk_size), self._preload_nchunks):
            slices = [
                slice(
                    index * self._chunk_size,
                    min(self.n_obs, (index + 1) * self._chunk_size),
                )
                for index in chunk_indices
            ]
            dataset_index_to_slices = self._slices_to_slices_with_array_index(slices)
            # Fetch the data over slices
            chunks: list[InputInMemoryArray] = zsync.sync(self._index_datasets(dataset_index_to_slices))
            if any(isinstance(c, CSRContainer) for c in chunks):
                chunks_converted: list[OutputInMemoryArray] = [
                    self._sp_module.csr_matrix(
                        tuple(self._np_module.asarray(e) for e in c.elems),
                        shape=c.shape,
                        dtype="float64" if self._preload_to_gpu else c.dtype,
                    )
                    for c in chunks
                ]
            else:
                chunks_converted = [self._np_module.asarray(c) for c in chunks]
            # Accumulate labels
            obs: None | list[pd.DataFrame] = None
            if self._obs is not None:
                obs = []
                for dataset_idx in dataset_index_to_slices.keys():
                    obs += [
                        self._obs[dataset_idx].iloc[
                            np.concatenate([np.arange(s.start, s.stop) for s in dataset_index_to_slices[dataset_idx]])
                        ]
                    ]
            # Accumulate indices if necessary
            indices: None | list[np.ndarray] = None
            if self._return_index:
                dataset_index_to_slices = self._slices_to_slices_with_array_index(slices, use_original_space=True)
                dataset_indices = dataset_index_to_slices.keys()
                indices = [
                    np.concatenate(
                        [
                            np.arange(
                                s.start,
                                s.stop,
                            )
                            for s in dataset_index_to_slices[index]
                        ]
                    )
                    for index in dataset_indices
                ]
            # Do batch returns, handling leftover data as necessary
            in_memory_data = (
                mod.vstack(chunks_converted)
                if in_memory_data is None
                else mod.vstack([in_memory_data, *chunks_converted])
            )
            if self._obs is not None:
                concatenated_obs = pd.concat(obs) if concatenated_obs is None else pd.concat([concatenated_obs, *obs])
            if self._return_index:
                in_memory_indices = (
                    np.concatenate(indices)
                    if in_memory_indices is None
                    else np.concatenate([in_memory_indices, *indices])
                )
            # Create random indices into in_memory_data and then index into it
            # If there is "leftover" at the end (see the modulo op),
            # save it for the next iteration.
            batch_indices = np.arange(in_memory_data.shape[0])
            if self._shuffle:
                np.random.default_rng().shuffle(batch_indices)
            splits = split_given_size(batch_indices, self._batch_size)
            for i, s in enumerate(splits):
                if s.shape[0] == self._batch_size:
                    output: LoaderOutput = {
                        "data": to_torch(in_memory_data[s], self._preload_to_gpu)
                        if self._to_torch
                        else in_memory_data[s],
                        "labels": concatenated_obs.iloc[s] if self._obs is not None else None,
                        "index": in_memory_indices[s] if self._return_index else None,
                    }
                    yield output
                if i == (len(splits) - 1):  # end of iteration, leftover data needs be kept
                    if (s.shape[0] % self._batch_size) != 0:
                        in_memory_data = in_memory_data[s]
                        if concatenated_obs is not None:
                            concatenated_obs = concatenated_obs.iloc[s]
                        if in_memory_indices is not None:
                            in_memory_indices = in_memory_indices[s]
                    else:
                        in_memory_data = None
                        concatenated_obs = None
                        in_memory_indices = None
        if in_memory_data is not None and not self._drop_last:  # handle any leftover data
            output: LoaderOutput = {
                "data": to_torch(in_memory_data, self._preload_to_gpu) if self._to_torch else in_memory_data,
                "labels": concatenated_obs if self._obs is not None else None,
                "index": in_memory_indices if self._return_index else None,
            }
            yield output
