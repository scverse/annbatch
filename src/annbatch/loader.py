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

from annbatch.samplers import ChunkSampler
from annbatch.types import BackingArray_T, InputInMemoryArray_T, LoaderOutput, OutputInMemoryArray_T
from annbatch.utils import (
    CSRContainer,
    MultiBasicIndexer,
    check_lt_1,
    check_var_shapes,
    interval_indexer_from_slices,
    load_x_and_obs,
    to_torch,
    validate_sampler,
)

from .compat import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

    from annbatch.abc import Sampler
    from annbatch.io import DatasetCollection

    # TODO: remove after sphinx 9 - myst compat
    BackingArray = BackingArray_T
    OutputInMemoryArray = OutputInMemoryArray_T
    InputInMemoryArray = InputInMemoryArray_T


class CSRDatasetElems(NamedTuple):
    """Container for cached objects that will be indexed into to generate CSR matrices"""

    indptr: np.ndarray
    indices: zarr.AsyncArray
    data: zarr.AsyncArray


def _cupy_dtype(dtype: np.dtype) -> np.dtype:
    if dtype in {np.dtype("float32"), np.dtype("float64"), np.dtype("bool")}:
        return dtype
    if dtype.itemsize < 4:
        return np.dtype("float32")
    return np.dtype("float64")


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
    `shuffle`, and `drop_last` must not be set (they are controlled by the `batch_sampler` instead).
    When providing these arguments and no `batch_sampler`, they are used to construct a :class:`annbatch.ChunkSampler`.

    Parameters
    ----------
        batch_sampler
            If not provided, a default :class:`annbatch.ChunkSampler` will be used with the same defaults below.
        chunk_size
            The obs size (i.e., axis 0) of contiguous array data to fetch. Mutually exclusive with `batch_sampler`. Defaults to 512.
        preload_nchunks
            The number of chunks of contiguous array data to fetch. Mutually exclusive with `batch_sampler`. Defaults to 32.
        shuffle
            Whether or not to shuffle the data. Mutually exclusive with `batch_sampler`. Defaults to False.
        batch_size
            Batch size to yield from the dataset. Mutually exclusive with `batch_sampler`. Defaults to 1.
        drop_last
            Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            Leave as False when using in conjunction with a :class:`torch.utils.data.DataLoader`.
            Mutually exclusive with `batch_sampler`. Defaults to False.
        return_index
            Whether or not to yield the index on each iteration.
        preload_to_gpu
            Whether or not to use cupy for non-io array operations like vstack and indexing once the data is in memory internally.
            This option entails greater GPU memory usage, but is faster at least for sparse operations.
            :func:`torch.vstack` does not support CSR sparse matrices, hence the current use of cupy internally.
            Setting this to `False` is advisable when using the :class:`torch.utils.data.DataLoader` wrapper or potentially with dense data.
            For top performance, this should be used in conjuction with `to_torch` and then :meth:`torch.Tensor.to_dense` if you wish to densify.
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

    _COMMON_SAMPLER_ARGS = {
        "chunk_size": 512,
        "preload_nchunks": 32,
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
    }
    # TODO(selmanozleyen): these should be also presented in the documentation
    # but this is not ideal since they are hardcoded into the docstrings
    # maybe we should make _COMMON_SAMPLER_ARGS a public class field?

    _train_datasets: list[BackingArray]
    _obs: list[pd.DataFrame] | None = None
    _return_index: bool = False
    _shapes: list[tuple[int, int]]
    _preload_to_gpu: bool = True
    _to_torch: bool = True
    _dataset_elem_cache: dict[int, CSRDatasetElems]
    _batch_sampler: Sampler

    def __init__(
        self,
        *,
        batch_sampler: Sampler | None = None,
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
        if batch_sampler is not None:
            if any(v is not None for v in sampler_args.values()):
                provided_args = [name for name, val in sampler_args.items() if val is not None]
                raise ValueError(
                    f"Cannot specify {', '.join(provided_args)} when providing a custom sampler. "
                    "These parameters are controlled by the sampler."
                )
            self._batch_sampler = batch_sampler
        else:
            sampler_args_processed = {
                k: (v if v is not None else Loader._COMMON_SAMPLER_ARGS[k]) for k, v in sampler_args.items()
            }
            self._batch_sampler = ChunkSampler(**sampler_args_processed)

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

    @property
    def batch_sampler(self) -> Sampler:
        """The sampler used to generate batches.

        Returns
        -------
            The sampler.
        """
        return self._batch_sampler

    def use_collection(
        self, collection: DatasetCollection, *, load_adata: Callable[[zarr.Group], ad.AnnData] = load_x_and_obs
    ) -> Self:
        """Load from an existing :class:`annbatch.DatasetCollection`.

        This function can only be called once. If you want to manually add more data, use :meth:`Loader.add_anndatas` or open an issue.

        Parameters
        ----------
        collection
            The collection who on-disk datasets should be used in this loader.
        load_adata
            A custom load function - recall that whatever is found in :attr:`~anndata.AnnData.X` and :attr:`~anndata.AnnData.obs` will be yielded in batches.
            Default is to just load `X` and all of `obs`.
            This default behavior can degrade performance if you don't need all columns in `obs` - it is recommended to use the `load_adata` argument.
        """
        if collection.is_empty:
            raise ValueError("DatasetCollection is empty")
        if getattr(self, "_collection_added", False):
            raise RuntimeError(
                "You should not add multiple collections, independently shuffled - please preshuffle multiple collections, use `add_anndatas` manually if you know what you are doing, or open an issue if you believe that this should be supported at an API level higher than `add_anndatas`."
            )
        adatas = [load_adata(g) for g in collection]
        self.add_anndatas(adatas)
        self._collection_added = True
        return self

    @validate_sampler
    def add_anndatas(
        self,
        adatas: list[ad.AnnData],
    ) -> Self:
        """Append anndatas to this dataset.

        Parameters
        ----------
            adatas
                List of :class:`anndata.AnnData` objects, with :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing annotations to yield in a :class:`pandas.DataFrame`.
        """
        check_lt_1([len(adatas)], ["Number of anndatas"])
        for adata in adatas:
            dataset, obs = self._prepare_dataset_and_obs(adata)
            self._add_dataset_unchecked(dataset, obs)
        return self

    def add_anndata(self, adata: ad.AnnData) -> Self:
        """Append an anndata to this dataset.

        Parameters
        ----------
            adata
                A :class:`anndata.AnnData` object, with :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing annotations to yield in a :class:`pandas.DataFrame`.
        """
        dataset, obs = self._prepare_dataset_and_obs(adata)
        self.add_dataset(dataset, obs)
        return self

    def _prepare_dataset_and_obs(self, adata: ad.AnnData) -> tuple[BackingArray, pd.DataFrame | None]:
        dataset = adata.X
        obs = adata.obs
        if len(obs.columns) == 0:
            obs = None
        if not isinstance(dataset, BackingArray_T.__value__):
            raise TypeError(f"Found {type(dataset)} but only {BackingArray_T.__value__} are usable")
        return cast("BackingArray", dataset), obs

    @validate_sampler
    def add_datasets(self, datasets: list[BackingArray], obs: list[pd.DataFrame] | None = None) -> Self:
        """Append datasets to this dataset.

        Parameters
        ----------
            datasets
                List of :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` objects, generally from :attr:`anndata.AnnData.X`.
                They must all be of the same type and match that of any already added datasets.
            obs
                List of :class:`~pandas.DataFrame` obs, generally from :attr:`anndata.AnnData.obs`.
        """
        if obs is None:
            obs = [None] * len(datasets)
        for ds, o in zip(datasets, obs, strict=True):
            self._add_dataset_unchecked(ds, o)
        return self

    @validate_sampler
    def add_dataset(self, dataset: BackingArray, obs: pd.DataFrame | None = None) -> Self:
        """Append a dataset to this dataset.

        Parameters
        ----------
            dataset
                A :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` object, generally from :attr:`anndata.AnnData.X`.
            obs
                :class:`~pandas.DataFrame` obs, generally from :attr:`anndata.AnnData.obs`.
        """
        self._add_dataset_unchecked(dataset, obs)
        return self

    def _add_dataset_unchecked(self, dataset: BackingArray, obs: pd.DataFrame | None = None) -> Self:
        if len(self._train_datasets) > 0:
            if self._obs is None and obs is not None:
                raise ValueError(
                    f"Cannot add a dataset with obs label {obs} when training datasets have already been added without obs"
                )
            if self._obs is not None and obs is None:
                raise ValueError(
                    "Cannot add a dataset with no obs label when training datasets have already been added without obs"
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
        if not isinstance(obs, pd.DataFrame) and obs is not None:
            raise TypeError("obs must be a pandas DataFrame")
        datasets = self._train_datasets + [dataset]
        check_var_shapes(datasets)
        self._shapes = self._shapes + [dataset.shape]
        self._train_datasets = datasets
        if self._obs is not None:  # obs exist
            self._obs += [obs]
        elif obs is not None:  # obs dont exist yet, but are being added for the first time
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

        In the codebase we use slice and chunk interchangeably. Not to be confused with the zarr chunking/sharding terminology.

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
            A batch of data along with its obs and index (both optional).
        """
        check_lt_1(
            [len(self._train_datasets), self.n_obs],
            ["Number of datasets", "Number of observations"],
        )
        mod = self._sp_module if issubclass(self.dataset_type, ad.abc.CSRDataset) else np
        for load_request in self._batch_sampler.sample(self.n_obs):
            chunks_to_load = load_request["chunks"]
            splits = load_request["splits"]
            # Sampler yields a list of slices that sum to batch_size
            dataset_index_to_slices = self._slices_to_slices_with_array_index(chunks_to_load, use_original_space=False)
            # Fetch the data over slices
            chunks: list[InputInMemoryArray] = zsync.sync(self._index_datasets(dataset_index_to_slices))
            in_memory_data = self._accumulate_chunks(chunks)
            # Accumulate labels and indices if possible
            concatenated_obs: None | pd.DataFrame = self._maybe_accumulate_obs(dataset_index_to_slices)
            in_memory_indices: None | np.ndarray = self._maybe_accumulate_indices(chunks_to_load)
            # An IntervalIndexer with start-stop bounds of each chunk's dataset
            dataset_interval_indexer = interval_indexer_from_slices(dataset_index_to_slices.values())
            for split in splits:
                sorted_split = np.sort(split)
                # Get the index of the dataset for the given split relative to the in-memory data
                dataset_locs = dataset_interval_indexer.get_indexer_for(sorted_split)
                # Get the left bound of that dataset relative to the in-memory data
                offsets = dataset_interval_indexer.left[dataset_locs]
                # Stack the chunks in dataset order, offseting each split by its dataset's leftmost in-memory bound
                data = mod.vstack(
                    [
                        chunk[sorted_split[dataset_locs == i] - offsets[dataset_locs == i]]
                        for i, chunk in enumerate(in_memory_data)
                    ]
                )
                yield {
                    "X": data if not self._to_torch else to_torch(data, self._preload_to_gpu),
                    "obs": concatenated_obs.iloc[sorted_split] if concatenated_obs is not None else None,
                    "index": in_memory_indices[sorted_split] if in_memory_indices is not None else None,
                }

    def _accumulate_chunks(self, chunks: list[InputInMemoryArray]) -> list[OutputInMemoryArray_T]:
        """Convert fetched chunks to output array format (CSR or ndarray)."""
        result: list[OutputInMemoryArray_T] = []
        for chunk in chunks:
            if isinstance(chunk, CSRContainer):
                result.append(
                    self._sp_module.csr_matrix(
                        tuple(self._np_module.asarray(e) for e in chunk.elems),
                        shape=chunk.shape,
                        dtype=_cupy_dtype(chunk.dtype) if self._preload_to_gpu else chunk.dtype,
                    )
                )
            else:
                result.append(self._np_module.asarray(chunk))
        return result

    def _maybe_accumulate_obs(self, dataset_index_to_slices: OrderedDict[int, list[slice]]) -> pd.DataFrame | None:
        """Gather obs labels for the loaded slices if possible."""
        if self._obs is None:
            return None
        return pd.concat(
            [
                self._obs[idx].iloc[np.concatenate([np.arange(s.start, s.stop) for s in slices])]
                for idx, slices in dataset_index_to_slices.items()
            ]
        )

    def _maybe_accumulate_indices(self, slices: list[slice]) -> np.ndarray | None:
        """Gather original indices for the loaded slices if possible."""
        if self._return_index is False:
            return None
        dataset_index_to_slices = self._slices_to_slices_with_array_index(slices, use_original_space=True)
        return np.concatenate(
            [
                np.concatenate([np.arange(s.start, s.stop) for s in dataset_index_to_slices[idx]])
                for idx in dataset_index_to_slices
            ]
        )
