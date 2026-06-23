from __future__ import annotations

import asyncio
from collections import OrderedDict
from functools import singledispatchmethod
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, cast
from warnings import warn

import anndata as ad
import numpy as np
import pandas as pd
import zarr
import zarr.core.sync as zsync
from packaging.version import Version
from scipy import sparse as sp
from zarr import Array as ZarrArray

from annbatch.samplers import RandomSampler, SequentialSampler
from annbatch.types import BackingArray_T, LoaderOutput, OutputInMemoryArray_T
from annbatch.utils import (
    CSRContainer,
    MultiBasicIndexer,
    check_lt_1,
    check_var_shapes,
    load_x_and_obs_and_var,
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
type concat_strategies = Literal["concat-shuffle", "shuffle-concat"]
zarr_version = Version(version("zarr"))


class CSRDatasetElems(NamedTuple):
    """Container for cached objects that will be indexed into to generate CSR matrices"""

    indptr: np.ndarray
    indices: zarr.AsyncArray
    data: zarr.AsyncArray


if find_spec("numba"):
    import numba

    @numba.njit(parallel=True, cache=True, nogil=True)
    def _csr_subset_rows(src_data, src_indices, src_indptr, rows, out_data, out_indices):  # type: ignore
        n_rows = rows.shape[0]
        row_nnz = np.empty(n_rows, dtype=np.int64)
        for i in range(n_rows):
            r = rows[i]
            row_nnz[i] = src_indptr[r + 1] - src_indptr[r]
        out_offsets = np.empty(n_rows + 1, dtype=np.int64)
        out_offsets[0] = 0
        for i in range(n_rows):
            out_offsets[i + 1] = out_offsets[i] + row_nnz[i]
        for i in numba.prange(n_rows):
            r = rows[i]
            src_start = src_indptr[r]
            dst_start = out_offsets[i]
            n = row_nnz[i]
            for j in range(n):
                out_data[dst_start + j] = src_data[src_start + j]
                out_indices[dst_start + j] = src_indices[src_start + j]
else:  # pragma: no cover

    def _csr_subset_rows(src_data, src_indices, src_indptr, rows, out_data, out_indices):
        raise ImportError("numba must be installed for in-memory sparse data: `pip install annbatch[numba]`")


def _cupy_dtype(dtype: np.dtype) -> np.dtype:
    if dtype in {np.dtype("float32"), np.dtype("float64"), np.dtype("bool")}:
        return dtype
    if dtype.itemsize < 4:
        return np.dtype("float32")
    return np.dtype("float64")


class Loader[
    BackingArray: BackingArray_T,
    OutputInMemoryArray: OutputInMemoryArray_T,
](IterableDataset):
    """A loader for on-disk data anndata stores.

    This loader by default batches together slice requests (`chunk_size` parameter) to the underlying stores to achieve higher performance.
    You can also use `chunk_size==1` for perfect random sampling (for relevant samplers), although this comes at a performance penalty for on-disk (and likely in-memory) data as well.
    Custom samplers are supported via the `batch_sampler` argument.
    We thus recommend using :class:`~annbatch.DatasetCollection` to preshuffle your data (or pre-shuffling in-memory).
    The loader is agnostic to the on-disk chunking/sharding, but it may be advisable to align with the in-memory chunk size for dense.

    If `preload_to_gpu` to True and `to_torch` is False, the yielded type is a `cupy` matrix.
    If `to_torch` is True, the yielded type is a :class:`torch.Tensor`.
    If both `preload_to_gpu` and `to_torch` are False, then the return type is the CPU class for the given data type.
    When providing a custom sampler, `chunk_size`, `preload_nchunks`, `batch_size`,
    `shuffle`, `drop_last`, and `rng` must not be set (they are controlled by the `batch_sampler` instead).
    When providing these arguments and no `batch_sampler`, they are used to construct a :class:`~annbatch.samplers.RandomSampler` (if ``shuffle=True``) or :class:`~annbatch.samplers.SequentialSampler`.

    Parameters
    ----------
        batch_sampler
            If not provided, a default :class:`~annbatch.samplers.SequentialSampler` or :class:`~annbatch.samplers.RandomSampler` will be used with the same defaults below.
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
        rng
            Random number generator for shuffling. Mutually exclusive with `batch_sampler`. Defaults to `np.random.default_rng()` if not provided.
        return_index
            Whether or not to yield the index on each iteration.
        preload_to_gpu
            Whether or not to use cupy for non-io array operations like vstack and indexing once the data is in memory internally.
            This option entails greater GPU memory usage, but is faster at least for sparse operations.
            :func:`torch.vstack` does not support CSR sparse matrices, hence the current use of `cupy` internally (which also means `torch` is an optional dep).
            Setting this to `False` is advisable when using the :class:`torch.utils.data.DataLoader` wrapper or potentially with dense data due to memory pressure.
            For top performance, this should be used in conjunction with `to_torch` and then :meth:`torch.Tensor.to_dense` if you wish to densify.
            :meth:`cupy.cuda.MemoryPool.free_all_blocks` (i.e., the method of the pool of :func:`cupy.get_default_memory_pool()`) is called aggressively to keep memory usage low.
            If you are using your own memory pool or allocator, you may have to free blocks on your own.
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
            ).add_adata(my_anndata)
        >>> for batch in ds:
                # optionally convert to dense
                # batch = batch.to_dense()
                do_fit(batch)
    """

    _COMMON_SAMPLER_ARGS = {
        "chunk_size": 512,
        "preload_nchunks": 32,
        "batch_size": 1,
        "drop_last": False,
    }
    # TODO(selmanozleyen): these should be also presented in the documentation
    # but this is not ideal since they are hardcoded into the docstrings
    # maybe we should make _COMMON_SAMPLER_ARGS a public class field?

    _train_datasets: list[BackingArray]
    _obs: list[pd.DataFrame] | None = None
    _var: pd.DataFrame | None = None
    _return_index: bool = False
    _shapes: list[tuple[int, int]]
    _preload_to_gpu: bool = True
    _to_torch: bool = True
    _sparse_dataset_elem_cache: dict[int, CSRDatasetElems]
    _batch_sampler: Sampler
    _collection_added: bool = False
    _dtypes_homogeneous: bool = True

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
        rng: np.random.Generator | None = None,
    ):
        # args that are passed after resolving defaults
        core_sampler_args = {
            "chunk_size": chunk_size,
            "preload_nchunks": preload_nchunks,
            "batch_size": batch_size,
            "drop_last": drop_last,
        }
        sampler_args = {**core_sampler_args, "rng": rng, "shuffle": shuffle}
        if batch_sampler is not None:
            if any(v is not None for v in sampler_args.values()):
                provided_args = [name for name, val in sampler_args.items() if val is not None]
                raise ValueError(
                    f"Cannot specify {', '.join(provided_args)} when providing a custom sampler. "
                    "These parameters are controlled by the sampler."
                )
            self._batch_sampler = batch_sampler
        else:
            resolved_core_args = {
                k: Loader._COMMON_SAMPLER_ARGS[k] if v is None else v for k, v in core_sampler_args.items()
            }
            if shuffle is not None and shuffle:
                self._batch_sampler = RandomSampler(
                    **resolved_core_args,
                    rng=rng if rng is not None else np.random.default_rng(),
                )
            else:
                self._batch_sampler = SequentialSampler(**resolved_core_args)
        if to_torch and not find_spec("torch"):
            raise ImportError("Could not find torch dependency. Try `pip install torch`.")
        if preload_to_gpu and not find_spec("cupy"):
            raise ImportError("Follow the directions at https://docs.cupy.dev/en/stable/install.html to install cupy.")

        self._return_index = return_index
        self._preload_to_gpu = preload_to_gpu
        self._to_torch = to_torch
        self._train_datasets = []
        self._shapes = []
        self._sparse_dataset_elem_cache = {}

    def __len__(self) -> int:
        return self._batch_sampler.n_batches(self.n_obs)

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
    def var(self) -> pd.DataFrame | None:
        """The var annotations for the variables in this loader.

        Returns
        -------
            The var DataFrame or None if no var annotations were provided.
        """
        return self._var

    @property
    def batch_sampler(self) -> Sampler:
        """The sampler used to generate batches.

        Returns
        -------
            The sampler.
        """
        return self._batch_sampler

    def use_collection(
        self,
        collection: DatasetCollection,
        *,
        load_adata: Callable[[zarr.Group], ad.AnnData] = load_x_and_obs_and_var,
    ) -> Self:
        """Load from an existing :class:`annbatch.DatasetCollection`.

        This function can only be called once. If you want to manually add more data, use :meth:`Loader.add_adatas` or open an issue.

        Parameters
        ----------
        collection
            The collection whose on-disk datasets should be used in this loader.
        load_adata
            A custom load function - recall that whatever is found in :attr:`~anndata.AnnData.X` and :attr:`~anndata.AnnData.obs` will be yielded in batches.
            Default is to just load `X` and all of `obs`.
            This default behavior can degrade performance if you don't need all columns in `obs` - it is recommended to use the `load_adata` argument.
        """
        if collection.is_empty:
            raise ValueError("DatasetCollection is empty")
        if self._collection_added:
            raise RuntimeError(
                "You should not add multiple collections, independently shuffled - please preshuffle multiple collections, use `add_adatas` manually if you know what you are doing, or open an issue if you believe that this should be supported at an API level higher than `add_adatas`."
            )
        adatas = [load_adata(g) for g in collection]
        self.add_adatas(adatas)
        self._collection_added = True
        return self

    @validate_sampler
    def add_adatas(
        self,
        adatas: list[ad.AnnData],
    ) -> Self:
        """Append adatas to this dataset.

        Parameters
        ----------
            adatas
                List of :class:`anndata.AnnData` objects, with :class:`zarr.Array`, :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csr_array`, :class:`numpy.ndarray`, or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing annotations to yield in a :class:`pandas.DataFrame`.
        """
        check_lt_1([len(adatas)], ["Number of adatas"])
        for adata in adatas:
            dataset, obs, var = self._prepare_dataset_obs_and_var(adata)
            self._add_dataset_unchecked(dataset, obs, var)
        return self

    def add_adata(self, adata: ad.AnnData) -> Self:
        """Append an adata to this dataset.

        Parameters
        ----------
            adata
                A :class:`anndata.AnnData` object, with :class:`zarr.Array`, :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csr_array`, :class:`numpy.ndarray`, or :class:`anndata.abc.CSRDataset` as the data matrix in :attr:`~anndata.AnnData.X`, and :attr:`~anndata.AnnData.obs` containing annotations to yield in a :class:`pandas.DataFrame`.
                :attr:`~anndata.AnnData.var` must match the ``var`` of any previously added datasets.
        """
        dataset, obs, var = self._prepare_dataset_obs_and_var(adata)
        self.add_dataset(dataset, obs, var)
        return self

    def _prepare_dataset_obs_and_var(
        self, adata: ad.AnnData
    ) -> tuple[BackingArray, pd.DataFrame | None, pd.DataFrame | None]:
        dataset = adata.X
        obs = adata.obs
        var = adata.var
        if len(obs.columns) == 0:
            obs = None
        if not isinstance(dataset, BackingArray_T.__value__):
            raise TypeError(f"Found {type(dataset)} but only {BackingArray_T.__value__} are usable")

        return cast("BackingArray", dataset), obs, var

    @validate_sampler
    def add_datasets(
        self,
        datasets: list[BackingArray],
        obs: list[pd.DataFrame] | None = None,
        var: list[pd.DataFrame] | None = None,
    ) -> Self:
        """Append datasets to this dataset.

        Parameters
        ----------
            datasets
                List of :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` objects, generally from :attr:`anndata.AnnData.X`.
                They must all be of the same type and match that of any already added datasets.
            obs
                List of :class:`~pandas.DataFrame` for annotating observations (i.e., samples), generally from :attr:`anndata.AnnData.obs`.
            var
                List of :class:`~pandas.DataFrame` for annotating features, generally from :attr:`anndata.AnnData.var`.
                All var DataFrames must be identical.
        """
        if obs is None:
            obs = [None] * len(datasets)
        if var is None:
            var = [None] * len(datasets)
        for ds, o, v in zip(datasets, obs, var, strict=True):
            self._add_dataset_unchecked(ds, o, v)
        return self

    @validate_sampler
    def add_dataset(
        self,
        dataset: BackingArray,
        obs: pd.DataFrame | None = None,
        var: pd.DataFrame | None = None,
    ) -> Self:
        """Append a dataset to this dataset.

        Parameters
        ----------
            dataset
                A :class:`zarr.Array` or :class:`anndata.abc.CSRDataset` object, generally from :attr:`anndata.AnnData.X`.
            obs
                :class:`~pandas.DataFrame` obs, generally from :attr:`anndata.AnnData.obs`.
            var
                :class:`~pandas.DataFrame` var, generally from :attr:`anndata.AnnData.var`.
                :attr:`~anndata.AnnData.var` must match the ``var`` of any previously added datasets.
        """
        self._add_dataset_unchecked(dataset, obs, var)
        return self

    def _add_dataset_unchecked(
        self,
        dataset: BackingArray,
        obs: pd.DataFrame | None = None,
        var: pd.DataFrame | None = None,
    ) -> Self:
        if len(self._train_datasets) > 0:
            if self._obs is None and obs is not None:
                raise ValueError(
                    f"Cannot add a dataset with obs label {obs} when training datasets have already been added without obs"
                )
            if self._obs is not None and obs is None:
                raise ValueError(
                    "Cannot add a dataset with no obs label when training datasets have already been added without obs"
                )
            if self._var is None and var is not None:
                raise ValueError(
                    "Cannot add a dataset with var when training datasets have already been added without var"
                )
            if self._var is not None and var is None:
                raise ValueError(
                    "Cannot add a dataset without var when training datasets have already been added with var"
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
        if isinstance(dataset, sp.csr_matrix | sp.csr_array) and not find_spec("numba"):
            raise ImportError("numba must be installed for in-memory sparse data: `pip install annbatch[numba]`")
        if not isinstance(obs, pd.DataFrame) and obs is not None:
            raise TypeError("obs must be a pandas DataFrame")
        if not isinstance(var, pd.DataFrame) and var is not None:
            raise TypeError("var must be a pandas DataFrame")
        datasets = self._train_datasets + [dataset]
        check_var_shapes(datasets)
        self._dtypes_homogeneous = self._datasets_share_dtype(datasets)
        if self._train_datasets and not self._dtypes_homogeneous:
            warn(
                f"Adding dataset with dtype {dataset.dtype!r} that differs from the existing dataset dtype(s) "
                f"(first dataset: {self._train_datasets[0].dtype!r}). Heterogeneous dtypes incur extra per-batch "
                "allocation and dtype promotion in the loader; consider casting all datasets to a common dtype.",
                stacklevel=2,
            )
        self._shapes = self._shapes + [dataset.shape]
        self._train_datasets = datasets
        if self._obs is not None:  # obs exist
            self._obs += [obs]
        elif obs is not None:  # obs dont exist yet, but are being added for the first time
            self._obs = [obs]
        # var is the same across all datasets (describes variables/features)
        if self._var is None and var is not None:
            self._var = var
        elif self._var is not None and var is not None and not self._var.equals(var):
            raise ValueError(
                "All datasets must have identical var DataFrames. "
                "The var of the new dataset does not match the existing var."
            )
        return self

    def _requests_to_dataset_rows(self, requests: list[slice] | np.ndarray) -> OrderedDict[int, np.ndarray]:
        """Given a ndarray or list of slices, give the lookup between on-disk datasets and row indices relative to that dataset.

        Parameters
        ----------
            requests
                Slices or array of integers relative to the on-disk datasets.

        Returns
        -------
            A lookup between the dataset and its row indices, ordered by keys, and the permutation
            ``order`` mapping each in-memory buffer position to its index in the original chunk order
            (the buffer is filled in dataset order, so ``order`` is what undoes that reordering).
        """
        if isinstance(requests, np.ndarray) and np.issubdtype(requests.dtype, np.integer):
            global_index = requests
        else:
            global_index = np.concatenate([np.arange(s.start, s.stop) for s in requests])

        # Locate each requested row in its dataset by binary-searching the dataset boundaries,
        sizes = np.fromiter(
            (shape[0] for shape in self._shapes),
            dtype=np.int64,
            count=len(self._shapes),
        )
        ends = np.cumsum(sizes)
        starts = ends - sizes
        dataset_of_row = np.searchsorted(ends, global_index, side="right")

        # Group rows by dataset: a stable sort keeps the within-dataset order
        order = np.argsort(dataset_of_row, kind="stable")
        grouped = dataset_of_row[order]
        group_start = np.concatenate([[0], np.flatnonzero(np.diff(grouped)) + 1])
        group_end = np.append(group_start[1:], grouped.size)

        result: OrderedDict[int, np.ndarray] = OrderedDict()
        for gs, ge in zip(group_start, group_end, strict=True):
            ds = int(grouped[gs])
            result[ds] = global_index[order[gs:ge]] - starts[ds]
        return result, order

    def _alloc(self, shape: tuple[int, ...], dtype: np.dtype, *, use_pinned: bool) -> np.ndarray:
        if use_pinned:
            import cupyx as cpx

            return cpx.empty_pinned(shape, dtype)
        return np.empty(shape, dtype)

    def _allocate_out(self, dataset_index_to_rows: OrderedDict[int, np.ndarray]) -> CSRContainer | np.ndarray:
        """Preallocate a single contiguous output buffer covering all datasets and rows.

        For sparse data the buffer is a :class:`~annbatch.utils.CSRContainer` whose ``data``
        and ``indices`` arrays span the total number of non-zeros (derived from the cached
        ``indptr``) and whose ``indptr`` array spans the total number of rows + 1.
        For dense data it is a plain :class:`numpy.ndarray` of shape
        ``(total_rows, n_var)``.

        Must be called after :meth:`_ensure_sparse_cache` for sparse datasets.
        """
        total_rows = sum(len(rows) for rows in dataset_index_to_rows.values())

        if (is_backed := issubclass(self.dataset_type, ad.abc.CSRDataset)) or issubclass(
            self.dataset_type, sp.csr_array | sp.csr_matrix
        ):
            datasets = self._sparse_dataset_elem_cache if is_backed else self._train_datasets
            total_nnz = sum(
                int((datasets[idx].indptr[rows + 1] - datasets[idx].indptr[rows]).sum())
                for idx, rows in dataset_index_to_rows.items()
            )
            first_idx = next(iter(dataset_index_to_rows))
            data_dtype = datasets[first_idx].data.dtype
            indices_dtype = datasets[first_idx].indices.dtype
            indptr_dtype = datasets[first_idx].indptr.dtype
            return CSRContainer(
                elems=(
                    self._alloc((total_nnz,), data_dtype, use_pinned=self._preload_to_gpu),
                    self._alloc((total_nnz,), indices_dtype, use_pinned=self._preload_to_gpu),
                    np.empty(total_rows + 1, dtype=indptr_dtype),
                ),
                shape=(total_rows, self.n_var),
                dtype=data_dtype,
            )
        else:
            first_idx = next(iter(dataset_index_to_rows))
            dtype = self._train_datasets[first_idx].dtype
            shape_res = self._train_datasets[first_idx].shape[1:]
            return self._alloc((total_rows, *shape_res), dtype, use_pinned=self._preload_to_gpu)

    @staticmethod
    def _datasets_share_dtype(datasets: list[BackingArray]) -> bool:
        """Whether all given dataset-like objects share the same dtype(s)."""
        if len(datasets) <= 1:
            return True

        def dtypes_of(d):
            if isinstance(d, ad.abc.CSRDataset):
                return (d.group["data"].dtype, d.group["indices"].dtype)
            if hasattr(d, "data") and hasattr(d, "indices"):
                return (d.data.dtype, d.indices.dtype)
            return (d.dtype,)

        first = dtypes_of(datasets[0])
        return all(dtypes_of(d) == first for d in datasets[1:])

    def _allocate_per_dataset_outs(
        self, dataset_index_to_rows: OrderedDict[int, np.ndarray]
    ) -> OrderedDict[int, CSRContainer | np.ndarray]:
        """Allocate one output buffer per dataset, each using that dataset's native dtype(s).

        Used when datasets have differing dtypes — the per-dataset buffers are concatenated
        into a final buffer of the promoted dtype by :meth:`_concatenate_outs`.
        Must be called after :meth:`_ensure_sparse_cache` for backed-sparse datasets.
        """
        is_backed_sparse = issubclass(self.dataset_type, ad.abc.CSRDataset)
        is_sparse = is_backed_sparse or issubclass(self.dataset_type, sp.csr_array | sp.csr_matrix)
        outs: OrderedDict[int, CSRContainer | np.ndarray] = OrderedDict()
        if is_sparse:
            datasets = self._sparse_dataset_elem_cache if is_backed_sparse else self._train_datasets
            for idx, rows in dataset_index_to_rows.items():
                ds = datasets[idx]
                nnz = int((ds.indptr[rows + 1] - ds.indptr[rows]).sum())
                outs[idx] = CSRContainer(
                    elems=(
                        self._alloc((nnz,), ds.data.dtype, use_pinned=False),
                        self._alloc((nnz,), ds.indices.dtype, use_pinned=False),
                        self._alloc((len(rows) + 1,), np.min_scalar_type(nnz), use_pinned=False),
                    ),
                    shape=(len(rows), self.n_var),
                    dtype=ds.data.dtype,
                )
        else:
            for idx, rows in dataset_index_to_rows.items():
                ds = self._train_datasets[idx]
                outs[idx] = self._alloc((len(rows), *ds.shape[1:]), ds.dtype, use_pinned=False)
        return outs

    def _concatenate_outs(self, outs: OrderedDict[int, CSRContainer | np.ndarray]) -> CSRContainer | np.ndarray:
        """Concatenate per-dataset buffers into a single buffer with promoted dtype(s)."""
        values = list(outs.values())
        if isinstance(values[0], CSRContainer):
            data_dtype = np.result_type(*[o.elems[0].dtype for o in values])
            indices_dtype = np.result_type(*[o.elems[1].dtype for o in values])
            total_nnz = sum(o.elems[0].size for o in values)
            total_rows = sum(o.shape[0] for o in values)
            data = self._alloc((total_nnz,), data_dtype, use_pinned=self._preload_to_gpu)
            indices = self._alloc((total_nnz,), indices_dtype, use_pinned=self._preload_to_gpu)
            indptr = self._alloc((total_rows + 1,), np.min_scalar_type(total_nnz), use_pinned=self._preload_to_gpu)
            indptr[0] = 0
            nnz_offset = 0
            row_offset = 0
            for o in values:
                n = o.elems[0].size
                r = o.shape[0]
                data[nnz_offset : nnz_offset + n] = o.elems[0]
                indices[nnz_offset : nnz_offset + n] = o.elems[1]
                indptr[row_offset + 1 : row_offset + r + 1] = o.elems[2][1:] + nnz_offset
                nnz_offset += n
                row_offset += r
            return CSRContainer(
                elems=(data, indices, indptr),
                shape=(total_rows, self.n_var),
                dtype=data_dtype,
            )
        dtype = np.result_type(*[o.dtype for o in values])
        total_rows = sum(o.shape[0] for o in values)
        out = self._alloc((total_rows, *values[0].shape[1:]), dtype, use_pinned=self._preload_to_gpu)
        offset = 0
        for o in values:
            out[offset : offset + o.shape[0]] = o
            offset += o.shape[0]
        return out

    @singledispatchmethod
    async def _fetch_data(
        self,
        dataset: ZarrArray | CSRDatasetElems,
        rows: np.ndarray,
        out: CSRContainer | np.ndarray,
    ) -> None:
        """Fetch data from an on-disk store into a preallocated buffer.

        Parameters
        ----------
        dataset
            The underlying store.
        rows
            Array of integer row indices within this dataset to fetch.
        out
            Preallocated buffer to write into — a contiguous view of the full
            output buffer allocated by :meth:`_allocate_out`.

        Raises
        ------
        NotImplementedError
            If the dataset type is not recognised.
        """
        raise NotImplementedError(f"Cannot fetch data for type {type(dataset)}")

    @_fetch_data.register
    async def _fetch_data_dense(self, dataset: ZarrArray, rows: np.ndarray, out: np.ndarray) -> None:
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        row_runs = np.split(rows, breaks)
        indexer = MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer(
                    (slice(int(r[0]), int(r[-1]) + 1), Ellipsis),
                    shape=dataset.metadata.shape,
                    chunk_grid=dataset.metadata.chunk_grid if zarr_version <= Version("3.1.6") else dataset._chunk_grid,
                )
                for r in row_runs
            ]
        )
        buffer_prototype = zarr.core.buffer.default_buffer_prototype()
        await dataset._async_array._get_selection(
            indexer,
            prototype=buffer_prototype,
            out=buffer_prototype.nd_buffer(out),
        )

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
        arr_idxs = [idx for idx in range(len(self._train_datasets)) if idx not in self._sparse_dataset_elem_cache]
        all_elems: list[CSRDatasetElems] = await asyncio.gather(
            *(
                self._create_sparse_elems(idx)
                for idx in range(len(self._train_datasets))
                if idx not in self._sparse_dataset_elem_cache
            )
        )
        for idx, elems in zip(arr_idxs, all_elems, strict=True):
            self._sparse_dataset_elem_cache[idx] = elems

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
        if dataset_idx not in self._sparse_dataset_elem_cache:
            raise ValueError("Cache not prepared")
        return self._sparse_dataset_elem_cache[dataset_idx]

    @_fetch_data.register
    async def _fetch_data_numpy_matrix(
        self,
        dataset: np.ndarray,
        rows: np.ndarray,
        out: np.ndarray,
    ) -> None:
        out[:] = dataset[rows]

    @_fetch_data.register
    async def _fetch_data_csr_matrix(
        self,
        dataset: sp.csr_matrix | sp.csr_array,
        rows: np.ndarray,
        out: CSRContainer,
    ) -> None:
        _csr_subset_rows(
            dataset.data,
            dataset.indices,
            dataset.indptr,
            np.ascontiguousarray(rows),
            out.elems[0],
            out.elems[1],
        )

    @_fetch_data.register
    async def _fetch_data_sparse(
        self,
        dataset: CSRDatasetElems,
        rows: np.ndarray,
        out: CSRContainer,
    ) -> None:
        # See https://github.com/scverse/anndata/blob/361325fc621887bf4f381e9412b150fcff599ff7/src/anndata/_core/sparse_dataset.py#L272-L295
        # for the inspiration of this function.
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        row_runs = np.split(rows, breaks)
        indptr, indices, data = dataset
        indptr_indices = [indptr[slice(s[0], s[-1] + 2)] for s in row_runs]
        indptr_limits = [slice(i[0].item(), i[-1].item()) for i in indptr_indices]
        indexer_data, indexer_indices = (
            MultiBasicIndexer(
                [
                    zarr.core.indexing.BasicIndexer(
                        (l,),
                        shape=arr.metadata.shape,
                        chunk_grid=arr.metadata.chunk_grid if zarr_version <= Version("3.1.6") else arr._chunk_grid,
                    )
                    for l in indptr_limits
                ]
            )
            for arr in [data, indices]
        )

        buffer_prototype = zarr.core.buffer.default_buffer_prototype()
        await asyncio.gather(
            data._get_selection(
                indexer_data,
                prototype=buffer_prototype,
                out=buffer_prototype.nd_buffer(out.elems[0]),
            ),
            indices._get_selection(
                indexer_indices,
                prototype=buffer_prototype,
                out=buffer_prototype.nd_buffer(out.elems[1]),
            ),
        )

    async def _index_datasets(
        self,
        dataset_index_to_rows: OrderedDict[int, np.ndarray],
    ) -> CSRContainer | np.ndarray:
        """Preallocate one output buffer, dispatch concurrent fetches into per-dataset views, then return the buffer.

        Parameters
        ----------
            dataset_index_to_rows
                A lookup of the list-placement index of a dataset to the sorted row indices to fetch.
        """
        is_backed_sparse = issubclass(self.dataset_type, ad.abc.CSRDataset)
        is_sparse = is_backed_sparse or issubclass(self.dataset_type, sp.csr_array | sp.csr_matrix)
        if is_backed_sparse:
            await self._ensure_sparse_cache()

        if not self._dtypes_homogeneous:
            per_dataset_outs = self._allocate_per_dataset_outs(dataset_index_to_rows)
            tasks = [
                self._fetch_data(
                    self._get_elem_from_cache(dataset_idx) if is_backed_sparse else self._train_datasets[dataset_idx],
                    rows,
                    per_dataset_outs[dataset_idx],
                )
                for dataset_idx, rows in dataset_index_to_rows.items()
            ]
            await asyncio.gather(*tasks)
            if is_sparse:
                datasets = self._sparse_dataset_elem_cache if is_backed_sparse else self._train_datasets
                for dataset_idx, rows in dataset_index_to_rows.items():
                    sub_out = per_dataset_outs[dataset_idx]
                    cached_indptr = datasets[dataset_idx].indptr
                    per_row_nnz = cached_indptr[rows + 1] - cached_indptr[rows]
                    sub_out.elems[2][0] = 0
                    np.cumsum(per_row_nnz, out=sub_out.elems[2][1:])
            return self._concatenate_outs(per_dataset_outs)

        out = self._allocate_out(dataset_index_to_rows)

        tasks = []
        row_offset = 0
        nnz_offset = 0

        for dataset_idx, rows in dataset_index_to_rows.items():
            nrows = len(rows)
            if is_sparse:
                datasets = self._sparse_dataset_elem_cache if is_backed_sparse else self._train_datasets
                cached_indptr = datasets[dataset_idx].indptr
                nnz = int((cached_indptr[rows + 1] - cached_indptr[rows]).sum())
                out_view: CSRContainer | np.ndarray = CSRContainer(
                    elems=(
                        out.elems[0][nnz_offset : nnz_offset + nnz],
                        out.elems[1][nnz_offset : nnz_offset + nnz],
                        out.elems[2][row_offset : row_offset + nrows + 1],
                    ),
                    shape=(nrows, self.n_var),
                    dtype=out.dtype,
                )
                nnz_offset += nnz
            else:
                out_view = out[row_offset : row_offset + nrows]

            tasks.append(
                self._fetch_data(
                    self._get_elem_from_cache(dataset_idx) if is_backed_sparse else self._train_datasets[dataset_idx],
                    rows,
                    out_view,
                )
            )
            row_offset += nrows

        await asyncio.gather(*tasks)

        if is_sparse:
            datasets = self._sparse_dataset_elem_cache if is_backed_sparse else self._train_datasets
            running_nnz = 0
            row_pos = 0
            out.elems[2][0] = 0
            for dataset_idx, rows in dataset_index_to_rows.items():
                cached_indptr = datasets[dataset_idx].indptr
                per_row_nnz = cached_indptr[rows + 1] - cached_indptr[rows]
                dest = out.elems[2][row_pos + 1 : row_pos + len(rows) + 1]
                np.cumsum(per_row_nnz, out=dest)
                dest += running_nnz
                running_nnz = dest[-1]
                row_pos += len(rows)

        return out

    def __iter__(
        self,
    ) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
        """Iterate over the on-disk datasets.

        Data for all requested datasets is fetched concurrently into a single preallocated
        buffer, converted to the output format once, and then yielded as direct row-index
        subsets — no vstack or intermediate concatenation is required.

        Yields
        ------
            A batch of data along with its obs and index (both optional).
        """
        check_lt_1(
            [len(self._train_datasets), self.n_obs],
            ["Number of datasets", "Number of observations"],
        )
        is_sparse = issubclass(self.dataset_type, ad.abc.CSRDataset | sp.csr_matrix | sp.csr_array)
        # Create `positions` variable so we don't need to run `np.arange` (O(n)) every time
        positions = np.empty(0, dtype=np.intp)
        for load_request in self._batch_sampler.sample(self.n_obs):
            requests_to_load = load_request.get("requests", None)
            if requests_to_load is None:
                requests_to_load = load_request.get("chunks", None)
                if requests_to_load is not None:
                    # this is for backwards compat.
                    warn(
                        "The `chunks` key in the load request is deprecated and will be removed in a future version. Please use `requests` instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    raise KeyError("load_request must contain either 'requests' or 'chunks'.")
            splits = load_request["splits"]

            dataset_index_to_rows, order = self._requests_to_dataset_rows(requests_to_load)

            # The buffer below is filled in dataset order, but ``splits`` are expressed in the
            # sampler's `LoadRequest.request` order. ``inv`` maps a request-order position to its buffer position so
            # the split semantics are independent of how chunks were regrouped across datasets.
            # ``order`` is a permutation of ``range(n)``, so every used slot is overwritten -- the
            # reused buffer never carries stale values from a previous request.
            n = order.size
            inv_buffer = np.empty(n, dtype=np.intp)
            if n > positions.size:
                positions = np.arange(n, dtype=np.intp)
            inv = inv_buffer[:n]
            inv = positions[order]

            raw_out: CSRContainer | np.ndarray = zsync.sync(self._index_datasets(dataset_index_to_rows))

            if is_sparse:
                in_memory_data = self._sp_module.csr_matrix(
                    tuple(self._np_module.asarray(e) for e in raw_out.elems),
                    shape=raw_out.shape,
                    dtype=_cupy_dtype(raw_out.dtype) if self._preload_to_gpu else raw_out.dtype,
                )
            else:
                in_memory_data = self._np_module.asarray(raw_out)

            concatenated_obs: None | pd.DataFrame = self._maybe_accumulate_obs(dataset_index_to_rows)
            in_memory_indices: None | np.ndarray = self._maybe_accumulate_indices(dataset_index_to_rows)
            for split in splits:
                sel = inv[split]
                data = in_memory_data[sel]
                yield {
                    "X": data if not self._to_torch else to_torch(data, self._preload_to_gpu),
                    "obs": concatenated_obs.iloc[sel] if concatenated_obs is not None else None,
                    "var": self._var,
                    "index": in_memory_indices[sel] if in_memory_indices is not None else None,
                }

            # https://github.com/cupy/cupy/issues/9625
            if self._preload_to_gpu and is_sparse:
                self._np_module.get_default_memory_pool().free_all_blocks()

    def _maybe_accumulate_obs(self, dataset_index_to_rows: OrderedDict[int, np.ndarray]) -> pd.DataFrame | None:
        """Gather obs labels for the loaded rows if possible."""
        if self._obs is None:
            return None
        return pd.concat([self._obs[idx].iloc[rows] for idx, rows in dataset_index_to_rows.items()])

    def _maybe_accumulate_indices(self, dataset_index_to_rows: OrderedDict[int, np.ndarray]) -> np.ndarray | None:
        """Gather original indices for the loaded rows if possible."""
        if self._return_index is False:
            return None
        dataset_offsets = np.concatenate(([0], np.cumsum([shape[0] for shape in self._shapes])))
        return np.concatenate([rows + dataset_offsets[idx] for idx, rows in dataset_index_to_rows.items()])
