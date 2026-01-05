from __future__ import annotations

import asyncio
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

from annbatch.sampler import Sampler, SliceSampler
from annbatch.types import BackingArray_T, InputInMemoryArray_T, LoaderOutput, OutputInMemoryArray_T
from annbatch.utils import CSRContainer, MultiBasicIndexer, check_lt_1, check_var_shapes, to_torch

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
    In this case, be sure to use `spawn` multiprocessing in the wrapping loader.

    If `preload_to_gpu` to True and `to_torch` is False, the yielded type is a `cupy` matrix.
    If `to_torch` is True, the yielded type is a :class:`torch.Tensor`.
    If both `preload_to_gpu` and `to_torch` are False, then the return type is the CPU class for the given data type.
    When providing a custom sampler, `chunk_size`, `preload_nchunks`, `batch_size`,
    `shuffle`, and `drop_last` must not be set (they are controlled by the sampler).

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
    _shapes: list[tuple[int, int]]
    _preload_to_gpu: bool = True
    _to_torch: bool = True
    _dataset_elem_cache: dict[int, CSRDatasetElems]
    _batch_sampler: Sampler[list[slice]] | None = None

    # Default sampler args (used when no custom sampler provided)
    _batch_size: int = 1
    _drop_last: bool = False
    _shuffle: bool = False
    _preload_nchunks: int = 32
    _chunk_size: int = 512

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        preload_nchunks: int | None = None,
        shuffle: bool | None = None,
        return_index: bool = False,
        batch_size: int | None = None,
        preload_to_gpu: bool = find_spec("cupy") is not None,
        drop_last: bool | None = None,
        to_torch: bool = find_spec("torch") is not None,
    ):
        sampler_args = {
            "chunk_size": chunk_size,
            "preload_nchunks": preload_nchunks,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }
        # if any is not None then set_sampler can't be called later.
        self._cannot_set_sampler_msg = None
        self._cannot_set_sampler = any(v is not None for v in sampler_args.values())
        if self._cannot_set_sampler:
            provided_args = [name for name, val in sampler_args.items() if val is not None]
            self._cannot_set_sampler_msg = (
                f"Cannot specify {', '.join(provided_args)} when providing a custom sampler. "
                "These parameters are controlled by the sampler."
            )
            sampler_args_filtered = {k: v for k, v in sampler_args.items() if v is not None}
            self._chunk_size = sampler_args_filtered.get("chunk_size", self._chunk_size)
            self._preload_nchunks = sampler_args_filtered.get("preload_nchunks", self._preload_nchunks)
            self._batch_size = sampler_args_filtered.get("batch_size", self._batch_size)
            self._shuffle = sampler_args_filtered.get("shuffle", self._shuffle)
            self._drop_last = sampler_args_filtered.get("drop_last", self._drop_last)

        check_lt_1(
            [
                self._chunk_size,
                self._preload_nchunks,
            ],
            ["Chunk size", "Preload chunks"],
        )
        if self._batch_size > (self._chunk_size * self._preload_nchunks):
            raise NotImplementedError(
                "Cannot yield batches bigger than the iterated in-memory size i.e., batch_size > (chunk_size * preload_nchunks)."
            )
        if to_torch and not find_spec("torch"):
            raise ImportError("Could not find torch dependency. Try `pip install torch`.")
        if preload_to_gpu and not find_spec("cupy"):
            raise ImportError("Follow the directions at https://docs.cupy.dev/en/stable/install.html to install cupy.")

        self._return_index = return_index
        self._preload_to_gpu = preload_to_gpu
        self._to_torch = to_torch
        self._train_datasets = []
        self._shapes = []
        self._dataset_elem_cache = {}
        self._batch_sampler = None

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
        if len(self._shapes) == 0:
            raise ValueError("No datasets added yet")
        return self._shapes[0][1]

    def add_anndatas(
        self,
        adatas: list[ad.AnnData],
    ) -> Self:
        """Append anndatas to this dataset.

        Parameters
        ----------
            adatas
                List of :class:`anndata.AnnData` objects, with :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing labels to yield in a :class:`pandas.DataFrame`.
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
                A :class:`anndata.AnnData` object, with :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing labels to yield in a :class:`pandas.DataFrame`.
        """
        dataset = adata.X
        obs = adata.obs
        if not isinstance(dataset, BackingArray_T.__value__):
            raise TypeError(f"Found {type(dataset)} but only {BackingArray_T.__value__} are usable")
        self.add_dataset(cast("BackingArray", dataset), obs)
        return self

    def add_datasets(self, datasets: list[BackingArray], obs: list[pd.DataFrame] | None = None) -> Self:
        """Append datasets to this dataset.

        Parameters
        ----------
            datasets
                List of :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` objects, generally from :attr:`anndata.AnnData.X`.
                They must all be of the same type and match that of any already added datasets.
            obs
                List of :class:`~pandas.DataFrame` labels, generally from :attr:`anndata.AnnData.obs`.
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
                :class:`~pandas.DataFrame` labels, generally from :attr:`anndata.AnnData.obs`.

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
        if self._batch_sampler is not None:
            self._batch_sampler.validate(self.n_obs)
        return self

    def set_sampler(self, sampler: Sampler[list[slice]]) -> Self:
        """Set the batch sampler for this loader.

        Once a sampler is set, no more datasets can be added via :meth:`add_dataset`.

        Parameters
        ----------
        sampler
            A sampler that controls how data is batched and loaded.

        """
        if self._cannot_set_sampler:
            raise ValueError(self._cannot_set_sampler_msg)
        if len(self._train_datasets) == 0:
            raise ValueError("Cannot set sampler before adding datasets. Add datasets first.")
        sampler.validate(self.n_obs)
        self._batch_sampler = sampler
        # TODO: it must be ensured that the sampler args of Loader should not be used after this point.
        # if not,this can lead to bugs in the future.
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
        for slice_ in slices:
            for relative_obs_indices in self._get_relative_obs_indices(slice_, use_original_space=use_original_space):
                dataset_index_to_slices[relative_obs_indices[1]] += [relative_obs_indices[0]]
        keys = sorted(dataset_index_to_slices.keys())
        dataset_index_to_slices_sorted = OrderedDict()
        for k in keys:
            dataset_index_to_slices_sorted[k] = dataset_index_to_slices[k]
        return dataset_index_to_slices_sorted

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
        # Create the sampler now that n_obs is known
        batch_sampler: Sampler[list[slice]] = (
            self._batch_sampler
            if self._batch_sampler is not None
            else SliceSampler(
                mask=slice(0, None),  # stop will be resolved in sample(n_obs)
                batch_size=self._batch_size,
                # Note: Loader uses chunk_size/preload_nchunks, Sampler uses slice_size/preload_nslices
                preload_nslices=self._preload_nchunks,
                slice_size=self._chunk_size,
                shuffle=self._shuffle,
                drop_last=self._drop_last,
            )
        )

        # In order to handle data returned where (chunk_size * preload_nchunks) mod batch_size != 0
        # we must keep track of the leftover data.
        in_memory_data = None
        concatenated_obs = None
        in_memory_indices = None
        batch_size = batch_sampler.batch_size

        # Detect if we're in a PyTorch worker context and configure the sampler
        # this has to happen here because the workers are spawned here
        if find_spec("torch"):
            from torch.utils.data import get_worker_info

            from annbatch.utils import WorkerHandle

            if get_worker_info() is not None:
                worker_handle = WorkerHandle()
                batch_sampler.set_worker_handle(worker_handle)

        mod = self._sp_module if issubclass(self.dataset_type, ad.abc.CSRDataset) else np

        for load_request in batch_sampler.sample(self.n_obs):
            slices = load_request.slices
            splits = load_request.splits
            # Sampler yields a list of slices that sum to batch_size
            dataset_index_to_slices = self._slices_to_slices_with_array_index(slices, use_original_space=False)
            # Fetch the data over slices
            chunks: list[InputInMemoryArray] = zsync.sync(self._index_datasets(dataset_index_to_slices))
            chunks_converted = self._accumulate_chunks(chunks)
            # Accumulate labels and indices if possible
            obs: None | list[pd.DataFrame] = self._maybe_accumulate_labels(dataset_index_to_slices)
            indices: None | list[np.ndarray] = self._maybe_accumulate_indices(slices)

            # Do batch returns, handling leftover data as necessary
            in_memory_data = (
                mod.vstack(chunks_converted)
                if in_memory_data is None
                else mod.vstack([in_memory_data, *chunks_converted])
            )
            if self._obs is not None and obs is not None:
                concatenated_obs = pd.concat(obs) if concatenated_obs is None else pd.concat([concatenated_obs, *obs])
            if self._return_index and indices is not None:
                in_memory_indices = (
                    np.concatenate(indices)
                    if in_memory_indices is None
                    else np.concatenate([in_memory_indices, *indices])
                )

            # yield all full batches (all but possibly the last)
            for split in splits[:-1]:
                yield self._prepare_output(
                    in_memory_data=in_memory_data,
                    concatenated_obs=concatenated_obs,
                    in_memory_indices=in_memory_indices,
                    split=split,
                )

            # handle the last split
            last_split = splits[-1]
            if len(last_split) == batch_size:
                # full batch, yield and reset
                yield self._prepare_output(
                    in_memory_data=in_memory_data,
                    concatenated_obs=concatenated_obs,
                    in_memory_indices=in_memory_indices,
                    split=last_split,
                )
                in_memory_data = None
                concatenated_obs = None
                in_memory_indices = None
            else:
                # partial batch, carry over for next iteration
                in_memory_data, concatenated_obs, in_memory_indices = self._prepare_leftover_data(
                    in_memory_data=in_memory_data,
                    concatenated_obs=concatenated_obs,
                    in_memory_indices=in_memory_indices,
                    leftover_split=last_split,
                )

        # after sampler is exhausted, yield any remaining leftover
        # only true when drop_last=False
        if in_memory_data is not None:
            yield self._prepare_output(
                in_memory_data=in_memory_data,
                concatenated_obs=concatenated_obs,
                in_memory_indices=in_memory_indices,
                split=np.arange(in_memory_data.shape[0]),
            )

    def _accumulate_chunks(self, chunks: list[InputInMemoryArray]) -> list[OutputInMemoryArray_T]:
        """Convert fetched chunks to output array format (CSR or ndarray)."""
        result: list[OutputInMemoryArray_T] = []
        for chunk in chunks:
            if isinstance(chunk, CSRContainer):
                result.append(
                    self._sp_module.csr_matrix(
                        tuple(self._np_module.asarray(e) for e in chunk.elems),
                        shape=chunk.shape,
                        dtype="float64" if self._preload_to_gpu else chunk.dtype,
                    )
                )
            else:
                result.append(self._np_module.asarray(chunk))
        return result

    def _maybe_accumulate_labels(
        self, dataset_index_to_slices: OrderedDict[int, list[slice]]
    ) -> list[pd.DataFrame] | None:
        """Gather obs labels for the loaded slices if possible."""
        if self._obs is None:
            return None
        return [
            self._obs[idx].iloc[np.concatenate([np.arange(s.start, s.stop) for s in slices])]
            for idx, slices in dataset_index_to_slices.items()
        ]

    def _maybe_accumulate_indices(self, slices: list[slice]) -> list[np.ndarray] | None:
        """Gather original indices for the loaded slices if possible."""
        if self._return_index is False:
            return None
        dataset_index_to_slices = self._slices_to_slices_with_array_index(slices, use_original_space=True)
        return [
            np.concatenate([np.arange(s.start, s.stop) for s in dataset_index_to_slices[idx]])
            for idx in dataset_index_to_slices
        ]

    def _prepare_output(
        self,
        *,
        in_memory_data: OutputInMemoryArray_T,
        concatenated_obs: pd.DataFrame | None,
        in_memory_indices: np.ndarray | None,
        split: np.ndarray,
    ) -> LoaderOutput:
        """Prepare the final output dict for a single batch."""
        index = None
        labels = None
        if self._obs is not None and concatenated_obs is not None:
            labels = concatenated_obs.iloc[split]
        if self._return_index and in_memory_indices is not None:
            index = in_memory_indices[split]
        data = in_memory_data[split]
        if self._to_torch:
            data = to_torch(data, self._preload_to_gpu)
        return {"data": data, "labels": labels, "index": index}

    def _prepare_leftover_data(
        self,
        *,
        in_memory_data: OutputInMemoryArray_T,
        concatenated_obs: pd.DataFrame | None,
        in_memory_indices: np.ndarray | None,
        leftover_split: np.ndarray,
    ) -> tuple[OutputInMemoryArray_T, pd.DataFrame | None, np.ndarray | None]:
        """Subset data/labels/indices to keep only leftover rows for next iter."""
        in_memory_data = in_memory_data[leftover_split]
        if concatenated_obs is not None:
            concatenated_obs = concatenated_obs.iloc[leftover_split]
        if in_memory_indices is not None:
            in_memory_indices = in_memory_indices[leftover_split]
        return in_memory_data, concatenated_obs, in_memory_indices
