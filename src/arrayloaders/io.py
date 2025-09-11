from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import dask.array as da
import numpy as np
import scipy.sparse as sp
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec, BloscShuffle

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from os import PathLike
    from typing import Any, Literal

    from zarr.abc.codec import BytesBytesCodec


def write_sharded(
    group: zarr.Group,
    adata: ad.AnnData,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    compressors: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
):
    """Write a sharded zarr store from a single anndata object

    Parameters
    ----------
        group
            The destination group, must be zarr v3
        adata
            The source anndata object
        chunk_size
            Chunk size inside a shard. Defaults to 4096.
        shard_size
            Shard size i.e., number of elements in a single file. Defaults to 65536.
        compressors
            The compressors to pass to `zarr`. Defaults to (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),).
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
        if iospec.encoding_type in {"array"}:
            dataset_kwargs = {
                "shards": (shard_size,) + (elem.shape[1:]),  # only shard over 1st dim
                "chunks": (chunk_size,) + (elem.shape[1:]),  # only chunk over 1st dim
                "compressors": compressors,
                **dataset_kwargs,
            }
        elif iospec.encoding_type in {"csr_matrix", "csc_matrix"}:
            dataset_kwargs = {
                "shards": (shard_size,),
                "chunks": (chunk_size,),
                "compressors": compressors,
                **dataset_kwargs,
            }

        write_func(store, elem_name, elem, dataset_kwargs=dataset_kwargs)

    ad.experimental.write_dispatched(group, "/", adata, callback=callback)
    zarr.consolidate_metadata(group.store)


def _lazy_load_with_obs_var_in_memory(paths: Iterable[PathLike[str]] | Iterable[str], chunk_size: int = 4096):
    adatas = []
    for path in paths:
        adata = ad.experimental.read_lazy(path)
        adata.obs = adata.obs.to_memory()
        adata.var = adata.var.to_memory()
        adatas.append(adata)

    return ad.concat(adatas, join="outer")


def _read_into_memory(paths: Iterable[PathLike[str]] | Iterable[str]):
    adatas = []
    for path in paths:
        adata = getattr(ad, f"read_{Path(path).suffix.split('.')[-1]}")(path)
        adatas.append(adata)

    return ad.concat(adatas, join="outer")


def _create_chunks_for_shuffling(
    adata: ad.AnnData, shuffle_n_obs_per_output_anndata: int = 1_048_576, shuffle: bool = True
):
    chunk_boundaries = np.cumsum([0] + list(adata.X.chunks[0]))
    slices = [
        slice(int(start), int(end)) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:], strict=True)
    ]
    if shuffle:
        random.shuffle(slices)
    idxs = np.concatenate([np.arange(s.start, s.stop) for s in slices])
    idxs = np.array_split(idxs, np.ceil(len(idxs) / shuffle_n_obs_per_output_anndata))

    return idxs


def create_anndata_chunks_directory(
    adata_paths: Iterable[PathLike[str]] | Iterable[str],
    output_path: PathLike[str] | str,
    *,
    var_subset: Iterable[str] | None = None,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
    h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
    n_obs_per_output_anndata: int = 1_048_576,
    shuffle: bool = True,
    should_denseify: bool = True,
    output_format: Literal["h5ad", "zarr"] = "zarr",
):
    """Take a list of anndata paths, create a list of anndata chunks with uniform var spaces at the desired path with `n_obs_per_output_anndata` rows per store.

    The main purpose of this function is to create shuffled sharded zarr stores, which is the default behavior.
    However, this function can also output h5 stores and also unshuffled stores as well.
    The var space is by default outer-joined, but can be subsetted by `var_subset`.

    Parameters
    ----------
        adata_paths
            Paths to the anndata files used to create the zarr store.
        output_path
            Path to the output zarr store.
        var_subset
            Subset of gene names to include in the store. If None, all genes are included.
            Genes are subset based on the `var_names` attribute of the concatenated AnnData object.
        chunk_size
            Size of the chunks to use for the data in the zarr store.
        shard_size
            Size of the shards to use for the data in the zarr store.
        zarr_compressor
            Compressors to use to compress the data in the zarr store.
        h5ad_compressor
            Compressors to use to compress the data in the h5ad store. See anndata.write_h5ad.
        n_obs_per_output_anndata
            Number of observations to load into memory at once for shuffling / pre-processing.
            The higher this number, the more memory is used, but the better the shuffling.
            This corresponds to the size of the shards created.
        shuffle
            Whether to shuffle the data before writing it to the store.
        should_denseify
            Whether or not to write as dense on disk.
        output_format
            Format of the output store. Can be either "zarr" or "h5ad".

    Examples
    --------
        >>> from arrayloaders.io.store_creation import create_anndata_chunks_directory
        >>> datasets = [
        ...     "path/to/first_adata.h5ad",
        ...     "path/to/second_adata.h5ad",
        ...     "path/to/third_adata.h5ad",
        ... ]
        >>> create_anndata_chunks_directory(datasets, "path/to/output/zarr_store")
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ad.settings.zarr_write_format = 3
    adata_concat = _lazy_load_with_obs_var_in_memory(adata_paths, chunk_size=chunk_size)
    adata_concat.obs_names_make_unique()
    chunks = _create_chunks_for_shuffling(adata_concat, n_obs_per_output_anndata, shuffle=shuffle)

    if var_subset is None:
        var_subset = adata_concat.var_names

    for i, chunk in enumerate(tqdm(chunks)):
        var_mask = adata_concat.var_names.isin(var_subset)
        adata_chunk = adata_concat[chunk, :][:, var_mask].copy()
        adata_chunk.X = adata_chunk.X.persist()
        if shuffle:
            # shuffle adata in memory to break up individual chunks
            idxs = np.random.default_rng().permutation(np.arange(len(adata_chunk)))
            adata_chunk.X = adata_chunk.X[idxs, :]
            adata_chunk.obs = adata_chunk.obs.iloc[idxs]
        # convert to dense format before writing to disk
        if should_denseify:
            adata_chunk.X = adata_chunk.X.map_blocks(lambda xx: xx.toarray(), dtype=adata_chunk.X.dtype)

        if output_format == "zarr":
            f = zarr.open_group(Path(output_path) / f"chunk_{i}.zarr", mode="w")
            write_sharded(
                f,
                adata_chunk,
                chunk_size=chunk_size,
                shard_size=shard_size,
                compressors=zarr_compressor,
            )
        elif output_format == "h5ad":
            adata_chunk.write_h5ad(Path(output_path) / f"chunk_{i}.h5ad", compression=h5ad_compressor)
        else:
            raise ValueError(f"Unrecognized output_format: {output_format}. Only 'zarr' and 'h5ad' are supported.")


def _get_array_encoding_type(path: PathLike[str] | str):
    shards = list(Path(path).glob("chunk_*.zarr"))
    with open(shards[0] / "X" / "zarr.json") as f:
        encoding = json.load(f)
    return encoding["attributes"]["encoding-type"]


def add_anndata_to_chunks_directory(
    adata_paths: Iterable[PathLike[str]] | Iterable[str],
    output_path: PathLike[str] | str,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    zarr_compressor: Iterable[BytesBytesCodec] = (BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),),
    cache_h5ads: bool = True,
):
    """Add anndata files to an existing Zarr store.

    Parameters
    ----------
        adata_paths
            Paths to the anndata files used to create the zarr store.
        output_path
            Path to the output zarr store.
        chunk_size
            Size of the chunks to use for the data in the zarr store.
        shard_size
            Size of the shards to use for the data in the zarr store.
        zarr_compressor
            Compressors to use to compress the data in the zarr store.
        cache_h5ads
            Whether to cache the h5ad files into memory before writing them to the store.

    Examples
    --------
        >>> from arrayloaders.io.store_creation import add_anndata_to_chunks_directory
        >>> datasets = [
        ...     "path/to/first_adata.h5ad",
        ...     "path/to/second_adata.h5ad",
        ...     "path/to/third_adata.h5ad",
        ... ]
        >>> add_anndata_to_chunks_directory(datasets, "path/to/output/zarr_store")
    """
    shards = list(Path(output_path).glob("chunk_*.zarr"))
    if len(shards) == 0:
        raise ValueError(
            "Store at `output_path` does not exist or is empty. Please run `create_anndata_chunks_directory` first."
        )
    encoding = _get_array_encoding_type(output_path)
    if encoding == "array":
        print("Detected array encoding type. Will convert to dense format before writing.")

    if cache_h5ads:
        adata_concat = _read_into_memory(adata_paths)
        chunks = np.array_split(np.random.default_rng().permutation(len(adata_concat)), len(shards))
    else:
        adata_concat = _lazy_load_with_obs_var_in_memory(adata_paths, chunk_size=chunk_size)
        chunks = _create_chunks_for_shuffling(adata_concat, np.ceil(len(adata_concat) / len(shards)), shuffle=True)
    var_mask = adata_concat.var_names.isin(adata_concat.var_names)
    adata_concat.obs_names_make_unique()

    for shard, chunk in tqdm(zip(shards, chunks, strict=False), total=len(shards)):
        if encoding == "array":
            f = zarr.open_group(shard)
            adata_shard = ad.AnnData(
                X=ad.experimental.read_elem_lazy(f["X"]).map_blocks(sp.csr_matrix).compute(),
                obs=ad.io.read_elem(f["obs"]),
                var=ad.io.read_elem(f["var"]),
            )
        else:
            adata_shard = ad.read_zarr(shard)
        adata = ad.concat(
            [
                adata_shard,
                ad.AnnData(
                    X=adata_concat.X[chunk, :][:, var_mask],
                    obs=adata_concat.obs.iloc[chunk],
                    var=adata_concat.var.loc[var_mask],
                ),
            ]
        )
        idxs_shuffled = np.random.default_rng().permutation(len(adata))
        adata = adata[idxs_shuffled, :].copy()  # this significantly speeds up writing to disk

        if encoding == "array":
            adata.X = da.from_array(adata.X, chunks=(chunk_size, -1)).map_blocks(
                lambda xx: xx.toarray().astype("f4"), dtype="f4"
            )

        f = zarr.open_group(shard, mode="w")
        write_sharded(
            f,
            adata,
            chunk_size=chunk_size,
            shard_size=shard_size,
            compressors=zarr_compressor,
        )
