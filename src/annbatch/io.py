from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Self

import anndata as ad
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata._core.sparse_dataset import BaseCompressedSparseDataset
from anndata.experimental.backed import Dataset2D
from dask.array.core import Array as DaskArray
from humanfriendly import parse_size
from tqdm.auto import tqdm
from zarr.codecs import BloscCodec, BloscShuffle

from annbatch.utils import split_given_size

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping
    from os import PathLike
    from typing import Any, Literal

    from zarr.abc.codec import BytesBytesCodec

V1_ENCODING = {"encoding-type": "annbatch-preshuffled", "encoding-version": "0.1.0"}
GROUPBY_ATTR_KEY = "annbatch-groupby"


def _default_load_adata[T: zarr.Group | h5py.Group | PathLike[str] | str](x: T) -> ad.AnnData:
    adata = ad.experimental.read_lazy(x, load_annotation_index=False)
    if not isinstance(x, zarr.Group | h5py.Group):
        group = (
            h5py.File(adata.file.filename, mode="r")
            if adata.file.filename is not None
            else zarr.open_group(x, mode="r")
        )
    else:
        group = x
    # -1 indicates that all of each `obs` column should just be loaded, but this is probably fine since it goes column by column and discards.
    # TODO: Bug with empty columns: https://github.com/scverse/anndata/pull/2307
    for attr in ["obs", "var"]:
        # Only one column at a time will be loaded so we will hopefully pick up the benefit of loading into memory by the cache without having memory pressure.
        if len(getattr(adata, attr).columns) > 0:
            setattr(adata, attr, ad.experimental.read_elem_lazy(group[attr], chunks=(-1,), use_range_index=True))
            for col in getattr(adata, attr).columns:
                # Nullables / categoricals have bad perforamnce characteristics when concatenating using dask
                if pd.api.types.is_extension_array_dtype(getattr(adata, attr)[col].dtype):
                    getattr(adata, attr)[col] = getattr(adata, attr)[col].data
    return adata


def _round_down(num: int, divisor: int):
    return num - (num % divisor)


def _shard_size_param_to_n_obs(shard_size: int | str, elem) -> int:
    """Convert `shard_size` to a number of observations given the size of an element from the anndata object.

    If *shard_size* is already an int, it is interpreted as `n_obs`.  When it is a
    size string the target byte budget is divided by the element's
    uncompressed bytes-per-observation-row.
    """
    if isinstance(shard_size, int):
        return shard_size
    target_bytes = parse_size(shard_size, binary=True)

    def _cs_bytes(x) -> int:
        return int(x.data.nbytes + x.indptr.nbytes + x.indices.nbytes)

    n_obs = elem.shape[0] if hasattr(elem, "shape") else len(elem)
    if n_obs == 0:
        return 1

    if isinstance(elem, h5py.Dataset):
        total_bytes = int(np.array(elem.shape).prod() * elem.dtype.itemsize)
    elif isinstance(elem, BaseCompressedSparseDataset):
        total_bytes = _cs_bytes(elem._to_backed())
    elif sp.issparse(elem):
        total_bytes = _cs_bytes(elem)
    else:
        total_bytes = elem.__sizeof__()

    bytes_per_row = total_bytes / n_obs
    return max(1, int(target_bytes / bytes_per_row)) if bytes_per_row > 0 else 1


def write_sharded(
    group: zarr.Group,
    adata: ad.AnnData,
    *,
    n_obs_per_chunk: int = 64,
    shard_size: int | str = "1GB",
    compressors: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
    key: str | None = None,
):
    """Write a sharded zarr store from a single AnnData object.

    Parameters
    ----------
        group
            The destination group, must be zarr v3
        adata
            The source anndata object
        n_obs_per_chunk
            Number of observations per chunk. For dense arrays this directly sets the first-axis chunk size.
            For sparse arrays it is converted to element counts using the average non-zero elements per row of the matrix being written.
        shard_size
            Number of observations per shard, or a size string (e.g. ``'1GB'``, ``'512MB'``).
            If a size string is provided, the observation count is derived independently for each array element from its uncompressed bytes-per-row so that every shard stays close to the target size.
            For dense arrays the resolved count directly sets the first-axis shard size.
            For sparse arrays it is converted to element counts using the average non-zero elements per row of the matrix being written.
        compressors
            The compressors to pass to `zarr`.
        key
            The key to which this object should be written - by default the root, in which case the *entire* store (not just the group) is cleared first.
    """
    with ad.settings.override(zarr_write_format=3, write_csr_csc_indices_with_min_possible_dtype=True):

        def callback(
            write_func: ad.experimental.Write,
            store: zarr.Group,
            elem_name: str,
            elem: ad.typing.RWAble,
            dataset_kwargs: Mapping[str, Any],
            *,
            iospec: ad.experimental.IOSpec,
        ):
            # Ensure we're not overriding anything here
            dataset_kwargs = dict(dataset_kwargs)
            if iospec.encoding_type in {"array"} and (
                any(n in store.name for n in {"obsm", "layers", "obsp"}) or "X" == elem_name
            ):
                obs_per_shard = _shard_size_param_to_n_obs(shard_size, elem)
                # Clamp chunk/shard to the element size for small datasets
                dense_chunk = min(n_obs_per_chunk, elem.shape[0])
                if dense_chunk == 0:
                    raise ValueError(f"Cannot write sharded array {elem_name!r} with 0 observations.")
                dense_shard = min(obs_per_shard, elem.shape[0])
                dense_shard = max(dense_chunk, _round_down(dense_shard, dense_chunk))
                dataset_kwargs = {
                    **dataset_kwargs,
                    "shards": (dense_shard,) + elem.shape[1:],  # only shard over 1st dim
                    "chunks": (dense_chunk,) + elem.shape[1:],  # only chunk over 1st dim
                    "compressors": compressors,
                }
            elif iospec.encoding_type in {"csr_matrix", "csc_matrix"}:
                obs_per_shard = _shard_size_param_to_n_obs(shard_size, elem)
                nnz = elem.nnz
                if elem.shape[0] == 0:
                    raise ValueError(f"Cannot write sharded sparse matrix {elem_name!r} with 0 observations.")
                avg_nnz_per_obs = nnz / elem.shape[0]
                sparse_chunk = max(1, int(n_obs_per_chunk * avg_nnz_per_obs))
                sparse_chunk = min(sparse_chunk, nnz) if nnz > 0 else sparse_chunk
                sparse_shard = max(1, int(obs_per_shard * avg_nnz_per_obs))
                sparse_shard = min(sparse_shard, nnz) if nnz > 0 else sparse_shard
                sparse_shard = max(sparse_chunk, _round_down(sparse_shard, sparse_chunk))
                dataset_kwargs = {
                    **dataset_kwargs,
                    "shards": (sparse_shard,),
                    "chunks": (sparse_chunk,),
                    "compressors": compressors,
                }
            write_func(store, elem_name, elem, dataset_kwargs=dataset_kwargs)

        ad.experimental.write_dispatched(group, "/" if key is None else key, adata, callback=callback)
        zarr.consolidate_metadata(group.store)


def _estimate_bytes_per_obs_row(
    adata: ad.AnnData,
    backing: zarr.Group | h5py.Group,
) -> float:
    """Estimate uncompressed bytes per observation row from on-disk metadata.

    Uses the lazy-loaded *adata* to determine which array keys are present, then
    reads shapes and dtypes from *backing* (the on-disk h5py/zarr group) to
    compute the per-row byte budget without materialising any data.
    """
    n_obs = adata.shape[0]
    if n_obs == 0:
        return 0.0

    elem_paths: list[str] = []
    if adata.X is not None:
        elem_paths.append("X")
    for k in adata.layers.keys():
        elem_paths.append(f"layers/{k}")
    for k in adata.obsm.keys():
        elem_paths.append(f"obsm/{k}")
    elem_paths.append("obs")

    mean_bytes_per_row = 0.0
    for elem_path in elem_paths:
        if elem_path not in backing:
            raise KeyError(f"Could not find {elem_path} on AnnData object in backing store")
        node = backing[elem_path]
        encoding = dict(node.attrs).get("encoding-type", "")
        if encoding in {"csr_matrix", "csc_matrix"}:
            data, indices, indptr = node["data"], node["indices"], node["indptr"]
            mean_bytes_per_row += (
                data.shape[0] * (data.dtype.itemsize + indices.dtype.itemsize) + indptr.shape[0] * indptr.dtype.itemsize
            ) / n_obs
        elif encoding in {"array", ""}:
            mean_bytes_per_row += int(np.prod(node.shape[1:])) * node.dtype.itemsize
        elif encoding == "dataframe":
            for col_key in node:
                if col_key == "_index":
                    continue
                col_node = node[col_key]
                col_encoding = dict(col_node.attrs).get("encoding-type", "")
                if col_encoding == "categorical":
                    col_node = col_node["codes"]
                if hasattr(col_node, "shape") and hasattr(col_node, "dtype"):
                    mean_bytes_per_row += col_node.shape[0] * col_node.dtype.itemsize / n_obs
        elif encoding == "awkward-array":
            for buf_key in node:
                buf = node[buf_key]
                if hasattr(buf, "shape") and hasattr(buf, "dtype"):
                    mean_bytes_per_row += buf.shape[0] * buf.dtype.itemsize / n_obs
        else:
            raise ValueError(
                f"Unsupported encoding-type {encoding!r} for element {elem_path!r}. Cannot estimate per-row byte size."
            )

    return mean_bytes_per_row


def _validate_anndatas_and_maybe_get_bytes_per_row[T: zarr.Group | h5py.Group | PathLike[str] | str](
    paths_or_anndatas: Iterable[T | ad.AnnData],
    *,
    load_adata: Callable[[T], ad.AnnData] = lambda x: ad.experimental.read_lazy(x, load_annotation_index=False),
    estimate_bytes_per_obs_row: bool = False,
) -> float | None:
    """Validate that all datasets share the same keys and optionally estimate bytes per observation row.

    Parameters
    ----------
    paths_or_anndatas
        Paths or AnnData objects to validate.
    load_adata
        Function to lazy-load an AnnData from a path.
    estimate_bytes_per_obs_row
        If ``True``, estimate the average uncompressed bytes per observation row from the on-disk data.
        All entries must be paths or groups (not AnnData objects) in this case.

    Returns
    -------
    The average bytes per observation row when *estimate_bytes_per_obs_row* is ``True``, otherwise ``None``.
    """
    num_raw_in_adata = 0
    found_keys: dict[str, defaultdict[str, int]] = {
        "layers": defaultdict(lambda: 0),
        "obsm": defaultdict(lambda: 0),
        "obs": defaultdict(lambda: 0),
    }
    bytes_per_obs_samples: list[float] = []
    for path_or_anndata in tqdm(paths_or_anndatas, desc="Validating anndatas"):
        if not isinstance(path_or_anndata, ad.AnnData):
            adata = load_adata(path_or_anndata)
            if estimate_bytes_per_obs_row:
                if isinstance(path_or_anndata, zarr.Group | h5py.Group):
                    backing = path_or_anndata
                else:
                    p = Path(str(path_or_anndata))
                    backing = h5py.File(str(p), "r") if p.is_file() else zarr.open_group(str(p), mode="r")
                bytes_per_obs_samples.append(_estimate_bytes_per_obs_row(adata, backing=backing))
        else:
            if estimate_bytes_per_obs_row:
                raise NotImplementedError(
                    "Cannot estimate bytes per observation row from an AnnData object. "
                    "Provide file paths or groups instead, or pass an integer for dataset_size."
                )
            adata = path_or_anndata
        for elem_name, key_count in found_keys.items():
            curr_keys = set(getattr(adata, elem_name).keys())
            for key in curr_keys:
                if not (elem_name in {"var", "obs"} and key == "_index"):
                    key_count[key] += 1
        if adata.raw is not None:
            num_raw_in_adata += 1
    if num_raw_in_adata != (num_anndatas := len(list(paths_or_anndatas))) and num_raw_in_adata != 0:
        warnings.warn(
            f"Found raw keys not present in all anndatas {paths_or_anndatas}, consider deleting raw or moving it to a shared layer/X location via `load_adata`",
            stacklevel=2,
        )
    for elem_name, key_count in found_keys.items():
        elem_keys_mismatched = [key for key, count in key_count.items() if (count != num_anndatas and count != 0)]
        if len(elem_keys_mismatched) > 0:
            warnings.warn(
                f"Found {elem_name} keys {elem_keys_mismatched} not present in all anndatas {paths_or_anndatas}, consider stopping and using the `load_adata` argument to alter {elem_name} accordingly.",
                stacklevel=2,
            )
    return float(np.mean(bytes_per_obs_samples)) if bytes_per_obs_samples else None


def _lazy_load_adata[T: zarr.Group | h5py.Group | PathLike[str] | str](
    paths: Iterable[T],
    load_adata: Callable[[T], ad.AnnData] = _default_load_adata,
):
    adatas = []
    categoricals_in_all_adatas: dict[str, pd.Index] = {}
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Lazy loading anndatas"):
        adata = load_adata(path)
        # Track the source file for this given anndata object
        adata.obs["src_path"] = pd.Categorical.from_codes(
            np.ones((adata.shape[0],), dtype="int") * i, categories=pd.Index([str(p) for p in paths])
        )
        # Concatenating Dataset2D drops categoricals so we need to track them
        if isinstance(adata.obs, Dataset2D):
            categorical_cols_in_this_adata = {
                col: adata.obs[col].dtype.categories for col in adata.obs.columns if adata.obs[col].dtype == "category"
            }
            if not categoricals_in_all_adatas:
                categoricals_in_all_adatas = {
                    **categorical_cols_in_this_adata,
                    "src_path": adata.obs["src_path"].dtype.categories,
                }
            else:
                for k in categoricals_in_all_adatas.keys() & categorical_cols_in_this_adata.keys():
                    categoricals_in_all_adatas[k] = categoricals_in_all_adatas[k].union(
                        categorical_cols_in_this_adata[k]
                    )
        # TODO: Probably bug in anndata, need the true index for proper outer joins (can't skirt this with fake indexes, at least not in the mixed-type regime).
        # See: https://github.com/scverse/anndata/pull/2299
        if isinstance(adata.var, Dataset2D):
            adata.var.index = adata.var.true_index
        if adata.raw is not None and isinstance(adata.raw.var, Dataset2D):
            adata.raw.var.index = adata.raw.var.true_index
        adatas.append(adata)
    if len(adatas) == 1:
        return adatas[0]
    adata = ad.concat(adatas, join="outer")
    if len(categoricals_in_all_adatas) > 0:
        adata.uns["dataset2d_categoricals_to_convert"] = categoricals_in_all_adatas
    return adata


def _create_chunks_for_shuffling(
    n_obs: int,
    rng: np.random.Generator,
    shuffle_chunk_size: int = 1000,
    shuffle: bool = True,
    *,
    shuffle_n_obs_per_dataset: int | None = None,
    n_chunkings: int | None = None,
) -> list[np.ndarray]:
    # this splits the array up into `shuffle_chunk_size` contiguous runs
    idxs = split_given_size(np.arange(n_obs), shuffle_chunk_size)
    if shuffle:
        rng.shuffle(idxs)
    match shuffle_n_obs_per_dataset is not None, n_chunkings is not None:
        case True, False:
            n_slices_per_dataset = int(shuffle_n_obs_per_dataset // shuffle_chunk_size)
            use_single_chunking = n_obs <= shuffle_n_obs_per_dataset or n_slices_per_dataset <= 1
        case False, True:
            n_slices_per_dataset = (n_obs // n_chunkings) // shuffle_chunk_size
            use_single_chunking = n_chunkings == 1
        case _, _:
            raise ValueError("Cannot provide both shuffle_n_obs_per_dataset and n_chunkings or neither")
    # In this case `shuffle_n_obs_per_dataset` is bigger than the size of the dataset or the slice size is probably too big.
    if use_single_chunking:
        return [np.concatenate(idxs)]
    # unfortunately, this is the only way to prevent numpy.split from trying to np.array the idxs list, which can have uneven elements.
    idxs_as_slices = np.array([slice(int(idx[0]), int(idx[-1] + 1)) for idx in idxs])
    return [
        np.concatenate([np.arange(s.start, s.stop) for s in idx])
        for idx in (
            split_given_size(idxs_as_slices, n_slices_per_dataset)
            if n_chunkings is None
            else np.array_split(idxs_as_slices, n_chunkings)
        )
    ]


def _normalize_groupby(groupby: str | Iterable[str]) -> list[str]:
    groupby_cols = [groupby] if isinstance(groupby, str) else list(groupby)
    if len(groupby_cols) == 0:
        raise ValueError("`groupby` must contain at least one `obs` column.")
    if len(set(groupby_cols)) != len(groupby_cols):
        raise ValueError("`groupby` columns must be unique.")
    return groupby_cols


def _groupby_from_attrs(attrs: Mapping[str, object]) -> list[str] | None:
    if GROUPBY_ATTR_KEY not in attrs:
        return None
    groupby_meta = attrs[GROUPBY_ATTR_KEY]
    if not isinstance(groupby_meta, dict):
        raise ValueError(f"Expected `{GROUPBY_ATTR_KEY}` attrs to be a mapping.")
    if "obs_columns" in groupby_meta:
        return _normalize_groupby(groupby_meta["obs_columns"])
    raise ValueError(f"Could not find `obs_columns` in `{GROUPBY_ATTR_KEY}` attrs.")


def _groupby_adata(
    adata: ad.AnnData, *, groupby: str | Iterable[str]
) -> tuple[ad.AnnData, dict[str, dict[str, list[str] | list[int]]]]:
    groupby_cols = _normalize_groupby(groupby)
    missing_cols = [col for col in groupby_cols if col not in adata.obs]
    if len(missing_cols) > 0:
        raise ValueError(f"Could not find groupby columns {missing_cols!r} in `obs`.")
    group_values = adata.obs[groupby_cols].reset_index(drop=True)

    sorted_values = group_values.sort_values(by=groupby_cols, kind="stable")
    order = sorted_values.index.to_numpy(dtype=int, copy=False)

    boundaries = np.flatnonzero(~sorted_values.duplicated()).tolist()
    boundaries.append(len(sorted_values))  # add end boundary

    return adata[order], {GROUPBY_ATTR_KEY: {"obs_columns": groupby_cols, "boundaries": boundaries}}


def _compute_blockwise(x: DaskArray) -> sp.spmatrix:
    """.compute() for large datasets is bad: https://github.com/scverse/annbatch/pull/75"""
    if isinstance(x._meta, sp.csr_matrix | sp.csr_array):
        return sp.vstack(da.compute(*list(x.blocks)))
    return x.compute()


def _to_categorical_obs(adata: ad.AnnData) -> ad.AnnData:
    """Convert columns marked as categorical in `uns` to categories, accounting for `concat` on `Dataset2D` lost dtypes"""
    if "dataset2d_categoricals_to_convert" in adata.uns:
        for col, categories in adata.uns["dataset2d_categoricals_to_convert"].items():
            adata.obs[col] = pd.Categorical(np.array(adata.obs[col]), categories=categories)
        del adata.uns["dataset2d_categoricals_to_convert"]
    return adata


def _persist_adata_in_memory(adata: ad.AnnData) -> ad.AnnData:
    if isinstance(adata.X, DaskArray):
        adata.X = _compute_blockwise(adata.X)
    if isinstance(adata.obs, Dataset2D):
        adata.obs = adata.obs.to_memory()
        # TODO: This is a bug in anndata?
        if "_index" in adata.obs.columns:
            adata.obs.index = adata.obs["_index"]
            del adata.obs["_index"]
    adata = _to_categorical_obs(adata)
    if isinstance(adata.var, Dataset2D):
        adata.var = adata.var.to_memory()
        if "_index" in adata.var.columns:
            del adata.var["_index"]

    if adata.raw is not None:
        adata_raw = adata.raw.to_adata()
        if isinstance(adata_raw.X, DaskArray):
            adata_raw.X = _compute_blockwise(adata_raw.X)
        if isinstance(adata_raw.var, Dataset2D):
            adata_raw.var = adata_raw.var.to_memory()
            if "_index" in adata_raw.var.columns:
                del adata_raw.var["_index"]
        if isinstance(adata_raw.obs, Dataset2D):
            adata_raw.obs = adata_raw.obs.to_memory()
        del adata.raw
        adata.raw = adata_raw

    for axis_name in ["layers", "obsm", "varm", "obsp", "varp"]:
        for k, elem in getattr(adata, axis_name).items():
            # TODO: handle `Dataset2D` in `obsm` and `varm` that are
            if isinstance(elem, DaskArray):
                getattr(adata, axis_name)[k] = _compute_blockwise(elem)
            if isinstance(elem, Dataset2D):
                elem = elem.to_memory()
                if "_index" in elem.columns:
                    del elem["_index"]
                # TODO: Bug in anndata
                if "obs" in axis_name:
                    elem.index = adata.obs_names
                getattr(adata, axis_name)[k] = elem

    return adata.to_memory()


DATASET_PREFIX = "dataset"


def _with_settings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with ad.settings.override(zarr_write_format=3, remove_unused_categories=False):
            return func(*args, **kwargs)

    return wrapper


class DatasetCollection:
    """A preshuffled collection object including functionality for creating, adding to, and loading collections shuffled by `annbatch`."""

    _group: zarr.Group | Path
    _groupby: list[str] | None

    def __init__(
        self,
        group: zarr.Group | str | Path,
        *,
        mode: Literal["a", "r", "r+"] = "a",
        is_collection_h5ad: bool = False,
        groupby: str | Iterable[str] | None = None,
    ):
        """Initialization of the object at a given location.

        Note that if the group is a h5py/zarr object, it must have the correct permissions for any subsequent operations you plan to do.
        Otherwise, the store will be opened according to the mode argument.


        Parameters
        ----------
            group
                The base location for a preshuffled collection.
                A :class:`zarr.Group` or path ending in `.zarr` indicates zarr as the shuffled format and otherwise a directory of `h5ad` files will be created.
            groupby
                Optional collection-level grouping configuration.
                When provided, new datasets are grouped within each dataset by these `obs` columns.
                Group boundaries are stored as jsonable dataset attrs rather than inside the AnnData payload.
                If opening an existing grouped zarr collection, any provided value must match the stored metadata.
        """
        configured_groupby = _normalize_groupby(groupby) if groupby is not None else None
        if not isinstance(group, zarr.Group):
            if isinstance(group, str | Path):
                if not is_collection_h5ad:
                    if not str(group).endswith(".zarr"):
                        warnings.warn(
                            f"It is highly recommended to make your collections have the `.zarr` suffix, got: {group}.",
                            stacklevel=2,
                        )
                    self._group = zarr.open_group(group, mode=mode)
                else:
                    warnings.warn(
                        "Loading h5ad is currently not supported and thus we cannot guarantee the funcionality of the ecosystem with h5ad files."
                        "DatasetCollection should be able to handle shuffling but we guarantee little else."
                        "Proceed with caution.",
                        stacklevel=2,
                    )
                    self._group = Path(group)
                    self._group.mkdir(exist_ok=True)
                    if configured_groupby is not None:
                        raise ValueError("`groupby` metadata is only supported for zarr collections.")
            else:
                raise TypeError("Group must either be a zarr group or a path")
        else:
            if is_collection_h5ad:
                raise ValueError("Do not set `is_collection_h5ad` to True when also passing in a zarr Group.")
            self._group = group
        stored_groupby = _groupby_from_attrs(dict(self._group.attrs)) if isinstance(self._group, zarr.Group) else None
        if configured_groupby is not None and stored_groupby is not None and configured_groupby != stored_groupby:
            raise ValueError(
                f"`groupby` {configured_groupby!r} does not match existing collection metadata {stored_groupby!r}."
            )
        self._groupby = stored_groupby if stored_groupby is not None else configured_groupby

    @property
    def _dataset_keys(self) -> list[str]:
        if isinstance(self._group, zarr.Group):
            return sorted(
                [k for k in self._group.keys() if re.match(rf"{DATASET_PREFIX}_([0-9]*)", k) is not None],
                key=lambda x: int(x.split("_")[1]),
            )
        else:
            raise ValueError("Cannot iterate through folder of h5ad files")

    def __iter__(self) -> Generator[zarr.Group]:
        if isinstance(self._group, zarr.Group):
            for k in self._dataset_keys:
                yield self._group[k]
        else:
            raise ValueError("Cannot iterate through folder of h5ad files")

    @property
    def is_empty(self) -> bool:
        """Whether or not there is an existing store at the group location."""
        return (
            (not (V1_ENCODING.items() <= self._group.attrs.items()) or len(self._dataset_keys) == 0)
            if isinstance(self._group, zarr.Group)
            else (len(list(self._group.iterdir())) == 0)
        )

    @_with_settings
    def add_adatas(
        self,
        adata_paths: Iterable[zarr.Group | h5py.Group | PathLike[str] | str],
        *,
        load_adata: Callable[[zarr.Group | h5py.Group | PathLike[str] | str], ad.AnnData] = _default_load_adata,
        var_subset: Iterable[str] | None = None,
        n_obs_per_chunk: int = 64,
        shard_size: int | str = "1GB",
        zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
        h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
        dataset_size: int | str = "20GB",
        shuffle_chunk_size: int = 1000,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Take AnnData paths and create or add to an on-disk set of AnnData datasets with uniform var spaces at the desired path (with `dataset_size` rows per dataset if running for the first time).

        The set of AnnData datasets is collectively referred to as a "collection" where each dataset is called `dataset_i{.h5ad}`.
        The main purpose of this function is to create shuffled sharded zarr datasets, which is the default behavior of this function.
        However, this function can also output h5 datasets and also unshuffled datasets as well.
        The var space is by default outer-joined initially, and then subsequently added datasets (i.e., on second calls to this function) are subsetted, but this behavior can be controlled by `var_subset`.
        A key `src_path` is added to `obs` to indicate where individual row came from.
        We highly recommend making your indexes unique across files, and this function will call `AnnData.obs_names_make_unique`.
        Memory usage should be controlled by `dataset_size` + `shuffle_chunk_size` as so many rows will be read into memory before writing to disk.
        After the dataset completes, a marker is added to the group's `attrs` to note that this dataset has been shuffled by `annbatch`.
        This is only for internal purposes at the moment so that we can recognize datasets that have been shuffled by an instance of this class.

        Parameters
        ----------
            adata_paths
                Paths to the AnnData files used to create the zarr store.
            load_adata
                Function to customize (lazy-)loading the invidiual input anndata files. By default, :func:`anndata.experimental.read_lazy` is used with categoricals/nullables read into memory and `(-1)` chunks for `obs`.
                If you only need a subset of the input anndata files' elems (e.g., only `X` and certain `obs` columns), you can provide a custom function here to speed up loading and harmonize your data.
                Beware that concatenating nullables/categoricals (i.e., what happens if `len(adata_paths) > 1` internally in this function) from {class}`anndata.experimental.backed.Dataset2D` `obs` is very time consuming - consider loading these into memory if you use this argument.
            var_subset
                Subset of gene names to include in the store. If None, all genes are included.
                Genes are subset based on the `var_names` attribute of the concatenated AnnData object.
            n_obs_per_chunk
                Number of observations per zarr chunk. For dense arrays this is used directly as the first-axis chunk size.
                For sparse arrays it is converted to element counts using the average number of non-zero elements per row of the matrix being written.
            shard_size
                Number of observations per zarr shard, or a size string (e.g. ``'1GB'``).
                If a size string is provided, the number of obersevations per zarr shard is estimated automatically.
                String sizes get parsed using the humanfriendly package.
                For sparse arrays the number of observations is converted to element counts using the average number of non-zero elements per row of the matrix being written
            zarr_compressor
                Compressors to use to compress the data in the zarr store.
            h5ad_compressor
                Compressors to use to compress the data in the h5ad store. See anndata.write_h5ad.
            dataset_size
                Number of observations to load into memory at once for shuffling / pre-processing, or a size string (e.g. ``'2GB'``, ``'512MB'``).
                When a size string is provided, the observation count is derived from the estimated uncompressed bytes per row of the input data.
                String sizes get parsed using the humanfriendly package.
                The higher this number, the more memory is used, but the better the shuffling.
                This corresponds to the size of the dataset level shards created.
                Only applicable when adding datasets for the first time, otherwise ignored.
            shuffle
                Whether to shuffle the data before writing it to the store.
                Ignored once the store is non-empty.
            shuffle_chunk_size
                How many contiguous rows to load into memory before shuffling at once.
                `(shuffle_chunk_size // dataset_size)` slices will be loaded of size `shuffle_chunk_size`.
            rng
                Random number generator for shuffling.

        Examples
        --------
            >>> import anndata as ad
            >>> from annbatch import DatasetCollection
            # create a custom load function to only keep `.X`, `.obs` and `.var` in the output store
            >>> def read_lazy_x_and_obs_only(path):
            ...     adata = ad.experimental.read_lazy(path)
            ...     return ad.AnnData(
            ...         X=adata.X,
            ...         obs=adata.obs.to_memory(),
            ...         var=adata.var.to_memory(),
            ...)
            >>> datasets = [
            ...     "path/to/first_adata.h5ad",
            ...     "path/to/second_adata.h5ad",
            ...     "path/to/third_adata.h5ad",
            ... ]
            >>> DatasetCollection("path/to/output/zarr_store.zarr").add_adatas(
            ...    datasets,
            ...    load_adata=read_lazy_x_and_obs_only,
            ...)
        """
        if rng is None:
            rng = np.random.default_rng()
        resolved_groupby = self._groupby
        shared_kwargs = {
            "adata_paths": adata_paths,
            "load_adata": load_adata,
            "n_obs_per_chunk": n_obs_per_chunk,
            "shard_size": shard_size,
            "zarr_compressor": zarr_compressor,
            "h5ad_compressor": h5ad_compressor,
            "shuffle_chunk_size": shuffle_chunk_size,
            "shuffle": shuffle,
            "rng": rng,
        }
        if self.is_empty:
            self._create_collection(**shared_kwargs, dataset_size=dataset_size, var_subset=var_subset)
            self._groupby = resolved_groupby
        else:
            self._add_to_collection(**shared_kwargs)
        return self

    def _create_collection(
        self,
        *,
        adata_paths: Iterable[PathLike[str]] | Iterable[str],
        load_adata: Callable[[PathLike[str] | str], ad.AnnData] = _default_load_adata,
        var_subset: Iterable[str] | None = None,
        n_obs_per_chunk: int = 64,
        shard_size: int | str = "1GB",
        zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
        h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
        dataset_size: int | str = "20GB",
        shuffle_chunk_size: int = 1000,
        shuffle: bool = True,
        rng: np.random.Generator,
    ) -> None:
        """Take AnnData paths, create an on-disk set of AnnData datasets with uniform var spaces at the desired path with `dataset_size` rows per dataset.

        The set of AnnData datasets is collectively referred to as a "collection" where each dataset is called `dataset_i.{zarr,h5ad}`.
        The main purpose of this function is to create shuffled sharded zarr datasets, which is the default behavior of this function.
        However, this function can also output h5 datasets and also unshuffled datasets as well.
        The var space is by default outer-joined, but can be subsetted by `var_subset`.
        A key `src_path` is added to `obs` to indicate where individual row came from.
        We highly recommend making your indexes unique across files, and this function will call `AnnData.obs_names_make_unique`.
        Memory usage should be controlled by `dataset_size` as so many rows will be read into memory before writing to disk.

        Parameters
        ----------
            adata_paths
                Paths to the AnnData files used to create the zarr store.
            load_adata
                Function to customize lazy-loading the invidiual input anndata files. By default, :func:`anndata.experimental.read_lazy` is used.
                If you only need a subset of the input anndata files' elems (e.g., only `X` and `obs`), you can provide a custom function here to speed up loading and harmonize your data.
                The input to the function is a path to an anndata file, and the output is an anndata object which has `X` as a :class:`dask.array.Array`.
            var_subset
                Subset of gene names to include in the store. If None, all genes are included.
                Genes are subset based on the `var_names` attribute of the concatenated AnnData object.
                Only applicable when adding datasets for the first time, otherwise ignored and the incoming data's var space is subsetted to that of the existing collection.
            n_obs_per_chunk
                Number of observations per zarr chunk. For dense arrays this is used directly as the first-axis chunk size.
                For sparse arrays it is converted to element counts using the average number of non-zero elements per row of the matrix being written.
            shard_size
                Number of observations per zarr shard, or a size string (e.g. ``'1GB'``).
                If a size string is provided, the number of obersevations per zarr shard is estimated automatically.
                For sparse arrays the number of observations is converted to element counts using the average number of non-zero elements per row of the matrix being written
            zarr_compressor
                Compressors to use to compress the data in the zarr store.
            h5ad_compressor
                Compressors to use to compress the data in the h5ad store. See anndata.write_h5ad.
            dataset_size
                Number of observations to load into memory at once for shuffling / pre-processing, or a size string (e.g. ``'2GB'``, ``'512MB'``).
                When a size string is provided, the observation count is derived from the estimated uncompressed bytes per row of the input data.
                The higher this number, the more memory is used, but the better the shuffling.
                This corresponds to the size of the shards created.
                Only applicable when adding datasets for the first time, otherwise ignored.
            shuffle
                Whether to shuffle the data before writing it to the store.
            shuffle_chunk_size
                How many contiguous rows to load into memory before shuffling at once.
                `(shuffle_chunk_size // dataset_size)` slices will be loaded of size `shuffle_chunk_size`.
            rng
                Random number generator for shuffling.
        """
        if not self.is_empty:
            raise RuntimeError("Cannot create a collection at a location that already has a shuffled collection")
        needs_estimate = isinstance(dataset_size, str)
        estimated_bytes_per_row = _validate_anndatas_and_maybe_get_bytes_per_row(
            adata_paths, load_adata=load_adata, estimate_bytes_per_obs_row=needs_estimate
        )

        if needs_estimate:
            target_bytes = parse_size(dataset_size, binary=True)
            dataset_size = max(1, int(target_bytes / estimated_bytes_per_row))

        if shuffle_chunk_size > dataset_size:
            raise ValueError(
                "Cannot have a larger slice size than observations per dataset. Reduce `shuffle_chunk_size` or increase `dataset_size`."
            )

        adata_concat = _lazy_load_adata(adata_paths, load_adata=load_adata)
        adata_concat.obs_names_make_unique()
        dataset_size = min(adata_concat.shape[0], dataset_size)
        chunks = _create_chunks_for_shuffling(
            adata_concat.shape[0],
            rng=rng,
            shuffle_chunk_size=shuffle_chunk_size,
            shuffle=shuffle,
            shuffle_n_obs_per_dataset=dataset_size,
        )
        groupby = self._groupby

        if var_subset is None:
            var_subset = adata_concat.var_names
        for i, chunk in enumerate(tqdm(chunks, desc="Creating dataset collection")):
            var_mask = adata_concat.var_names.isin(var_subset)
            # np.sort: It's more efficient to access elements sequentially from dask arrays
            # The data will be shuffled later on, we just want the elements at this point
            adata_chunk = adata_concat[np.sort(chunk), :][:, var_mask].copy()
            adata_chunk = _persist_adata_in_memory(adata_chunk)
            if shuffle:
                # shuffle adata in memory to break up individual chunks
                idxs = rng.permutation(np.arange(len(adata_chunk)))
                adata_chunk = adata_chunk[idxs]
            dataset_attrs = None
            if groupby is not None:
                adata_chunk, dataset_attrs = _groupby_adata(adata_chunk, groupby=groupby)
            if isinstance(self._group, zarr.Group):
                write_sharded(
                    self._group,
                    adata_chunk,
                    n_obs_per_chunk=min(n_obs_per_chunk, adata_chunk.shape[0]),
                    shard_size=shard_size,
                    compressors=zarr_compressor,
                    key=f"{DATASET_PREFIX}_{i}",
                )
                if dataset_attrs is not None:
                    dataset_group = self._group[f"{DATASET_PREFIX}_{i}"]
                    dataset_group.update_attributes(dataset_attrs)
                    zarr.consolidate_metadata(self._group.store)
            else:
                ad.io.write_h5ad(
                    self._group / f"{DATASET_PREFIX}_{i}.h5ad",
                    adata_chunk,
                    dataset_kwargs={"compression": h5ad_compressor},
                )
        if isinstance(self._group, zarr.Group):
            attrs = dict(V1_ENCODING)
            if groupby is not None:
                attrs[GROUPBY_ATTR_KEY] = {"obs_columns": _normalize_groupby(groupby)}
            self._group.update_attributes(attrs)
            zarr.consolidate_metadata(self._group.store)

    def _add_to_collection(
        self,
        *,
        adata_paths: Iterable[PathLike[str]] | Iterable[str],
        load_adata: Callable[[PathLike[str] | str], ad.AnnData] = ad.read_h5ad,
        n_obs_per_chunk: int = 64,
        shard_size: int | str = "1GB",
        zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
        h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
        shuffle_chunk_size: int = 1000,
        shuffle: bool = True,
        rng: np.random.Generator,
    ) -> None:
        """Add anndata files to an existing collection of sharded anndata zarr datasets.

        The var space of the source anndata files will be adapted to the target store.

        Parameters
        ----------
            adata_paths
                Paths to the anndata files to be appended to the collection of output chunks.
            rng
                Random number generator for shuffling.
            load_adata
                Function to customize loading the invidiual input anndata files. By default, :func:`anndata.read_h5ad` is used.
                If you only need a subset of the input anndata files' elems (e.g., only `X` and `obs`), you can provide a custom function here to speed up loading and harmonize your data.
                The input to the function is a path to an anndata file, and the output is an anndata object.
                If the input data is too large to fit into memory, you should use :func:`annndata.experimental.read_lazy` instead.
            n_obs_per_chunk
                Number of observations per zarr chunk. For dense arrays this is used directly as the first-axis chunk size.
                For sparse arrays it is converted to element counts using the average number of non-zero elements per row of the matrix being written.
            shard_size
                Number of observations per zarr shard, or a size string (e.g. ``'1GB'``).
                If a size string is provided, the number of obersevations per zarr shard is estimated automatically.
                For sparse arrays the number of observations is converted to element counts using the average number of non-zero elements per row of the matrix being written
            zarr_compressor
                Compressors to use to compress the data in the zarr store.
            should_sparsify_output_in_memory
                This option is for testing only appending sparse files to dense stores.
                To save memory, the blocks of a dense on-disk store can be sparsified for in-memory processing.
            shuffle_chunk_size
                How many contiguous rows to load into memory of the input data for pseudo-blockshuffling into the existing datasets.
            shuffle
                Whether or not to shuffle when adding.  Otherwise, the incoming data will just be split up and appended.
        """
        if self.is_empty:
            raise ValueError("Store is empty. Please run `DatasetCollection.add_adatas` first.")
        groupby = self._groupby
        # Check for mismatched keys among the inputs.
        adata_concat = _lazy_load_adata(adata_paths, load_adata=load_adata)
        if math.ceil(adata_concat.shape[0] / shuffle_chunk_size) < len(self._dataset_keys):
            raise ValueError(
                f"Use a shuffle size small enough to distribute the input data with {adata_concat.shape[0]} obs across {len(self._dataset_keys)} anndata stores."
                "Open an issue if the incoming anndata is so small it cannot be distributed across the on-disk data"
            )
        # Check for mismatched keys between datasets and the inputs.
        _validate_anndatas_and_maybe_get_bytes_per_row([adata_concat] + [self._group[k] for k in self._dataset_keys])
        chunks = _create_chunks_for_shuffling(
            adata_concat.shape[0],
            rng=rng,
            shuffle_chunk_size=shuffle_chunk_size,
            shuffle=shuffle,
            n_chunkings=len(self._dataset_keys),
        )

        adata_concat.obs_names_make_unique()
        for dataset, chunk in tqdm(
            zip(self._dataset_keys, chunks, strict=True),
            total=len(self._dataset_keys),
            desc="Extending dataset collection",
        ):
            adata_dataset = ad.io.read_elem(self._group[dataset])
            subset_adata = _to_categorical_obs(
                adata_concat[chunk, :][:, adata_concat.var.index.isin(adata_dataset.var.index)]
            )
            adata = ad.concat([adata_dataset, subset_adata], join="outer")
            if shuffle:
                idxs = rng.permutation(adata.shape[0])
            else:
                idxs = np.arange(adata.shape[0])
            adata = _persist_adata_in_memory(adata[idxs, :].copy())
            dataset_attrs = None
            if groupby is not None:
                adata, dataset_attrs = _groupby_adata(adata, groupby=groupby)
            if isinstance(self._group, zarr.Group):
                write_sharded(
                    self._group,
                    adata,
                    n_obs_per_chunk=min(n_obs_per_chunk, adata.shape[0]),
                    shard_size=shard_size,
                    compressors=zarr_compressor,
                    key=dataset,
                )
                if dataset_attrs is not None:
                    dataset_group = self._group[dataset]
                    dataset_group.update_attributes(dataset_attrs)
                    zarr.consolidate_metadata(self._group.store)
            else:
                ad.io.write_h5ad(
                    self._group / f"{dataset}.h5ad",
                    adata,
                    dataset_kwargs={"compression": h5ad_compressor},
                )
        if isinstance(self._group, zarr.Group):
            zarr.consolidate_metadata(self._group.store)
