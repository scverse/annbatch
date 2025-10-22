from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import scipy.sparse as sp
import zarr
from anndata.experimental.backed import Dataset2D
from dask.array.core import Array as DaskArray
from tqdm import tqdm
from zarr.codecs import BloscCodec, BloscShuffle

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from os import PathLike
    from typing import Any, Literal

    from zarr.abc.codec import BytesBytesCodec


def write_sharded(
    group: zarr.Group,
    adata: ad.AnnData,
    *,
    sparse_chunk_size: int = 32768,
    sparse_shard_size: int = 134_217_728,
    dense_chunk_size: int = 1024,
    dense_shard_size: int = 4194304,
    compressors: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
):
    """Write a sharded zarr store from a single AnnData object.

    Parameters
    ----------
        group
            The destination group, must be zarr v3
        adata
            The source anndata object
        sparse_chunk_size
            Chunk size of `indices` and `data` inside a shard.
        sparse_shard_size
            Shard size i.e., number of elements in a single sparse `data` or `indices` file.
        dense_chunk_size
            Number of obs elements per dense chunk along the first axis
        dense_shard_size
            Number of obs elements per dense shard along the first axis
        compressors
            The compressors to pass to `zarr`.
    """
    ad.settings.zarr_write_format = 3

    def callback(
        write_func: ad.experimental.Write,
        store: zarr.Group,
        elem_name: str,
        elem: ad.typing.RWAble,
        dataset_kwargs: Mapping[str, Any],
        *,
        iospec: ad.experimental.IOSpec,
    ):
        if iospec.encoding_type in {"array"} and (
            any(n in store.name for n in {"obsm", "layers", "obsp"}) or "X" == elem_name
        ):
            dataset_kwargs = {
                "shards": (min(dense_shard_size, elem.shape[0]),) + (elem.shape[1:]),  # only shard over 1st dim
                "chunks": (min(dense_chunk_size, elem.shape[0]),) + (elem.shape[1:]),  # only chunk over 1st dim
                "compressors": compressors,
                **dataset_kwargs,
            }
        elif iospec.encoding_type in {"csr_matrix", "csc_matrix"}:
            dataset_kwargs = {
                "shards": (sparse_shard_size,),
                "chunks": (sparse_chunk_size,),
                "compressors": compressors,
                **dataset_kwargs,
            }
        write_func(store, elem_name, elem, dataset_kwargs=dataset_kwargs)

    ad.experimental.write_dispatched(group, "/", adata, callback=callback)
    zarr.consolidate_metadata(group.store)


def _lazy_load_anndatas(
    paths: Iterable[PathLike[str]] | Iterable[str],
    load_function: Callable[[PathLike[str] | str], ad.AnnData] = ad.experimental.read_lazy,
):
    adatas = []
    for path in paths:
        adata = load_function(path)
        adatas.append(adata)
    if len(adatas) == 1:
        return adatas[0]
    return ad.concat(adatas, join="outer")


def _read_into_memory(paths: Iterable[PathLike[str]] | Iterable[str]):
    adatas = []
    for path in paths:
        print(path)
        adata = getattr(ad, f"read_{Path(path).suffix.split('.')[-1]}")(path)
        adatas.append(adata)

    return ad.concat(adatas, join="outer")


def _create_chunks_for_shuffling(adata: ad.AnnData, shuffle_n_obs_per_dataset: int = 1_048_576, shuffle: bool = True):
    chunk_boundaries = np.cumsum([0] + list(adata.X.chunks[0]))
    slices = [
        slice(int(start), int(end)) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:], strict=True)
    ]
    if shuffle:
        random.shuffle(slices)
    idxs = np.concatenate([np.arange(s.start, s.stop) for s in slices])
    idxs = np.array_split(idxs, np.ceil(len(idxs) / shuffle_n_obs_per_dataset))

    return idxs


def _persist_adata_in_memory(adata: ad.AnnData) -> ad.AnnData:
    adata.X = adata.X.persist()

    if isinstance(adata.obs, Dataset2D):
        adata.obs = adata.obs.to_memory()
    if isinstance(adata.var, Dataset2D):
        adata.var = adata.var.to_memory()
    if adata.raw is not None:
        adata_raw = adata.raw.to_adata()
        if isinstance(adata_raw.X, DaskArray):
            adata_raw.X = adata_raw.X.persist()
        if isinstance(adata_raw.var, Dataset2D):
            adata_raw.var = adata_raw.var.to_memory()
        if isinstance(adata_raw.obs, Dataset2D):
            adata_raw.obs = adata_raw.obs.to_memory()
        del adata.raw
        adata.raw = adata_raw

    for k, elem in adata.obsm.items():
        # TODO: handle `Dataset2D` in `obsm` and `varm` that are
        if isinstance(elem, DaskArray):
            adata.obsm[k] = elem.persist()

    for k, elem in adata.layers.items():
        if isinstance(elem, DaskArray):
            adata.layers[k] = elem.persist()

    return adata


DATASET_PREFIX = "dataset"


def create_anndata_collection(
    adata_paths: Iterable[PathLike[str]] | Iterable[str],
    output_path: PathLike[str] | str,
    *,
    load_function: Callable[[PathLike[str] | str], ad.AnnData] = ad.experimental.read_lazy,
    var_subset: Iterable[str] | None = None,
    zarr_sparse_chunk_size: int = 32768,
    zarr_sparse_shard_size: int = 134_217_728,
    zarr_dense_chunk_size: int = 1024,
    zarr_dense_shard_size: int = 4_194_304,
    zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
    h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
    n_obs_per_dataset: int = 2_097_152,
    shuffle: bool = True,
    should_denseify: bool = False,
    output_format: Literal["h5ad", "zarr"] = "zarr",
):
    """Take AnnData paths, create an on-disk set of AnnData datasets with uniform var spaces at the desired path with `n_obs_per_dataset` rows per store.

    The set of AnnData datasets is collectively referred to as a "collection" where each dataset is called `dataset_i.{zarr,h5ad}`.
    The main purpose of this function is to create shuffled sharded zarr datasets, which is the default behavior of this function.
    However, this function can also output h5 datasets and also unshuffled datasets as well.
    The var space is by default outer-joined, but can be subsetted by `var_subset`.

    Parameters
    ----------
        adata_paths
            Paths to the AnnData files used to create the zarr store.
        output_path
            Path to the output zarr store.
        load_function
            Function to lazy-load anndata files. By default, {func}`anndata.experimental.read_lazy` is used.
            If you only need a subset of the input anndata files (e.g., only `X` and `obs`), you can provide a custom function here to speed up loading.
            The input to the function is a path to an anndata file, and the output is an anndata object which has `X` as a {class}`dask.array.Array`.
        var_subset
            Subset of gene names to include in the store. If None, all genes are included.
            Genes are subset based on the `var_names` attribute of the concatenated AnnData object.
        zarr_sparse_chunk_size
            Size of the chunks to use for the `indices` and `data` of a sparse matrix in the zarr store.
        zarr_sparse_shard_size
            Size of the shards to use for the `indices` and `data` of a sparse matrix in the zarr store.
        zarr_dense_chunk_size
            Number of observations per dense zarr chunk i.e., sharding is only done along the first axis of the array.
        zarr_dense_shard_size
            Number of observations per dense zarr shard i.e., chunking is only done along the first axis of the array.
        zarr_compressor
            Compressors to use to compress the data in the zarr store.
        h5ad_compressor
            Compressors to use to compress the data in the h5ad store. See anndata.write_h5ad.
        n_obs_per_dataset
            Number of observations to load into memory at once for shuffling / pre-processing.
            The higher this number, the more memory is used, but the better the shuffling.
            This corresponds to the size of the shards created.
        shuffle
            Whether to shuffle the data before writing it to the store.
        should_denseify
            Whether to write as dense on disk.
        output_format
            Format of the output store. Can be either "zarr" or "h5ad".

    Examples
    --------
        >>> import anndata as ad
        >>> from annbatch import create_anndata_collection
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
        >>> create_anndata_collection(
        ...    datasets,
        ...    "path/to/output/zarr_store",
        ...    load_function=read_lazy_x_and_obs_only,
        ...)
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ad.settings.zarr_write_format = 3
    adata_concat = _lazy_load_anndatas(adata_paths, load_function=load_function)
    adata_concat.obs_names_make_unique()
    chunks = _create_chunks_for_shuffling(adata_concat, n_obs_per_dataset, shuffle=shuffle)

    if var_subset is None:
        var_subset = adata_concat.var_names

    for i, chunk in enumerate(tqdm(chunks)):
        var_mask = adata_concat.var_names.isin(var_subset)
        adata_chunk = adata_concat[chunk, :][:, var_mask].copy()
        adata_chunk = _persist_adata_in_memory(adata_chunk)
        if shuffle:
            # shuffle adata in memory to break up individual chunks
            idxs = np.random.default_rng().permutation(np.arange(len(adata_chunk)))
            adata_chunk = adata_chunk[idxs].copy()
        # convert to dense format before writing to disk
        if should_denseify:
            adata_chunk.X = adata_chunk.X.map_blocks(lambda xx: xx.toarray(), dtype=adata_chunk.X.dtype)

        if output_format == "zarr":
            f = zarr.open_group(Path(output_path) / f"{DATASET_PREFIX}_{i}.zarr", mode="w")
            write_sharded(
                f,
                adata_chunk,
                sparse_chunk_size=zarr_sparse_chunk_size,
                sparse_shard_size=zarr_sparse_shard_size,
                dense_chunk_size=zarr_dense_chunk_size,
                dense_shard_size=zarr_dense_shard_size,
                compressors=zarr_compressor,
            )
        elif output_format == "h5ad":
            adata_chunk.write_h5ad(Path(output_path) / f"{DATASET_PREFIX}_{i}.h5ad", compression=h5ad_compressor)
        else:
            raise ValueError(f"Unrecognized output_format: {output_format}. Only 'zarr' and 'h5ad' are supported.")


def _get_array_encoding_type(path: PathLike[str] | str) -> str:
    shards = list(Path(path).glob(f"{DATASET_PREFIX}_*.zarr"))
    with open(shards[0] / "X" / "zarr.json") as f:
        encoding = json.load(f)
    return encoding["attributes"]["encoding-type"]


def add_to_collection(
    adata_paths: Iterable[PathLike[str]] | Iterable[str],
    output_path: PathLike[str] | str,
    zarr_sparse_chunk_size: int = 32768,
    zarr_sparse_shard_size: int = 134_217_728,
    zarr_dense_chunk_size: int = 1024,
    zarr_dense_shard_size: int = 4_194_304,
    zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
    read_full_anndatas: bool = True,
    should_sparsify_output_in_memory: bool = False,
) -> None:
    """Add anndata files to an existing collection of sharded anndata zarr datasets.

    The var space of the source anndata files will be adapted to the target store.

    Parameters
    ----------
        adata_paths
            Paths to the anndata files to be appended to the collection of output chunks.
        output_path
            Path to the output zarr store.
        zarr_sparse_chunk_size
            Size of the chunks to use for the `indices` and `data` of a sparse matrix in the zarr store.
        zarr_sparse_shard_size
            Size of the shards to use for the `indices` and `data` of a sparse matrix in the zarr store.
        zarr_dense_chunk_size
            Number of observations per dense zarr chunk i.e., sharding is only done along the first axis of the array.
        zarr_dense_shard_size
            Number of observations per dense zarr shard i.e., chunking is only done along the first axis of the array.
        zarr_compressor
            Compressors to use to compress the data in the zarr store.
        read_full_anndatas
            Whether to read the full input anndata files into memory before writing them to the store.
            Otherwise, reading will be done lazily.
        should_sparsify_output_in_memory
            This option is for testing only appending sparse files to dense stores.
            To save memory, the blocks of a dense on-disk store can be sparsified for in-memory processing.

    Examples
    --------
        >>> from annbatch import add_to_collection
        >>> datasets = [
        ...     "path/to/first_adata.h5ad",
        ...     "path/to/second_adata.h5ad",
        ...     "path/to/third_adata.h5ad",
        ... ]
        >>> add_to_collection(datasets, "path/to/output/zarr_store")
    """
    shards = list(Path(output_path).glob(f"{DATASET_PREFIX}_*.zarr"))
    if len(shards) == 0:
        raise ValueError(
            "Store at `output_path` does not exist or is empty. Please run `create_anndata_collection` first."
        )
    encoding = _get_array_encoding_type(output_path)
    if encoding == "array":
        print("Detected array encoding type. Will convert to dense format before writing.")

    if read_full_anndatas:
        adata_concat = _read_into_memory(adata_paths)
        chunks = np.array_split(np.random.default_rng().permutation(len(adata_concat)), len(shards))
    else:
        adata_concat = _lazy_load_anndatas(adata_paths)
        chunks = _create_chunks_for_shuffling(adata_concat, np.ceil(len(adata_concat) / len(shards)), shuffle=True)
    adata_concat.obs_names_make_unique()
    if encoding == "array":
        if not should_sparsify_output_in_memory:
            if isinstance(adata_concat.X, sp.spmatrix):
                adata_concat.X = adata_concat.X.toarray()
            elif isinstance(adata_concat.X, DaskArray) and isinstance(adata_concat.X._meta, sp.spmatrix):
                adata_concat.X = adata_concat.X.map_blocks(
                    lambda x: x.toarray(), meta=np.ndarray, dtype=adata_concat.X.dtype
                )
    elif encoding == "csr_matrix":
        if isinstance(adata_concat.X, np.ndarray):
            adata_concat.X = sp.csr_matrix(adata_concat.X)
        elif isinstance(adata_concat.X, DaskArray) and isinstance(adata_concat.X._meta, np.ndarray):
            adata_concat.X = adata_concat.X.map_blocks(
                sp.csr_matrix, meta=sp.csr_matrix(np.array([0], dtype=adata_concat.X.dtype))
            )

    for shard, chunk in tqdm(zip(shards, chunks, strict=False), total=len(shards)):
        if should_sparsify_output_in_memory and encoding == "array":
            adata_shard = _lazy_load_anndatas([shard])
            adata_shard.X = adata_shard.X.map_blocks(sp.csr_matrix).compute()
        else:
            adata_shard = ad.read_zarr(shard)

        adata = ad.concat(
            [adata_shard, adata_concat[chunk, :][:, adata_concat.var.index.isin(adata_shard.var.index)]], join="outer"
        )
        idxs_shuffled = np.random.default_rng().permutation(len(adata))
        adata = adata[idxs_shuffled, :].copy()  # this significantly speeds up writing to disk
        if should_sparsify_output_in_memory and encoding == "array":
            adata.X = adata.X.map_blocks(lambda x: x.toarray(), meta=np.array([0], dtype=adata.X.dtype)).compute()

        f = zarr.open_group(shard, mode="w")
        write_sharded(
            f,
            adata,
            sparse_chunk_size=zarr_sparse_chunk_size,
            sparse_shard_size=zarr_sparse_shard_size,
            dense_chunk_size=zarr_dense_chunk_size,
            dense_shard_size=zarr_dense_shard_size,
            compressors=zarr_compressor,
        )
