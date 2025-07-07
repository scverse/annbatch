from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import dask.array as da
import h5py
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


def _write_sharded(
    group: zarr.Group,
    adata: ad.AnnData,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    compressors: Iterable[BytesBytesCodec] = (
        BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),
    ),
):
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr

    def callback(
        func: ad.experimental.Write,
        g: zarr.Group,
        k: str,
        elem: ad.typing.RWAble,
        dataset_kwargs: Mapping[str, Any],
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

        func(g, k, elem, dataset_kwargs=dataset_kwargs)

    ad.experimental.write_dispatched(group, "/", adata, callback=callback)
    zarr.consolidate_metadata(group.store)


def _lazy_load_h5ads(
    paths: Iterable[PathLike[str]] | Iterable[str], chunk_size: int = 4096
):
    adatas = []
    for path in paths:
        with h5py.File(path) as f:
            adata = ad.AnnData(
                X=ad.experimental.read_elem_lazy(f["X"], chunks=(chunk_size, -1)),
                obs=ad.io.read_elem(f["obs"]),
                var=ad.io.read_elem(f["var"]),
            )
            adatas.append(adata)

    return ad.concat(adatas, join="outer")


def _create_chunks_for_shuffling(
    adata: ad.AnnData, shuffle_buffer_size: int = 1_048_576, shuffle: bool = True
):
    chunk_boundaries = np.cumsum([0] + list(adata.X.chunks[0]))
    slices = [
        slice(int(start), int(end))
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:], strict=True)
    ]
    if shuffle:
        random.shuffle(slices)
    idxs = np.concatenate([np.arange(s.start, s.stop) for s in slices])
    idxs = np.array_split(idxs, np.ceil(len(idxs) / shuffle_buffer_size))

    return idxs


def create_store_from_h5ads(
    adata_paths: Iterable[PathLike[str]] | Iterable[str],
    output_path: PathLike[str] | str,
    var_subset: Iterable[str] | None = None,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    zarr_compressor: Iterable[BytesBytesCodec] = (
        BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),
    ),
    h5ad_compressor: Literal["gzip", "lzf"] | None = "gzip",
    buffer_size: int = 1_048_576,
    shuffle: bool = True,
    *,
    should_denseify: bool = True,
    output_format: Literal["h5ad", "zarr"] = "zarr",
):
    """Create a Zarr store from multiple h5ad files.

    Args:
        adata_paths: Paths to the h5ad files used to create the zarr store.
        output_path: Path to the output zarr store.
        var_subset: Subset of gene names to include in the store. If None, all genes are included.
            Genes are subset based on the `var_names` attribute of the concatenated AnnData object.
        chunk_size: Size of the chunks to use for the data in the zarr store.
        shard_size: Size of the shards to use for the data in the zarr store.
        zarr_compressor: Compressors to use to compress the data in the zarr store.
        h5ad_compressor: Compressors to use to compress the data in the h5ad store. See anndata.write_h5ad.
        buffer_size: Number of observations to load into memory at once for shuffling / pre-processing.
            The higher this number, the more memory is used, but the better the shuffling.
            This corresponds to the size of the shards created.
        shuffle: Whether to shuffle the data before writing it to the store.
        should_denseify: Whether or not to write as dense on disk.
        output_format: Format of the output store. Can be either "zarr" or "h5ad".

    Examples:
        >>> from arrayloaders.io.store_creation import create_store_from_h5ads
        >>> datasets = [
        ...     "path/to/first_adata.h5ad",
        ...     "path/to/second_adata.h5ad",
        ...     "path/to/third_adata.h5ad",
        ... ]
        >>> create_store_from_h5ads(datasets, "path/to/output/zarr_store")
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr
    print("setting ad.settings.zarr_write_format to 3")
    adata_concat = _lazy_load_h5ads(adata_paths, chunk_size=chunk_size)
    adata_concat.obs_names_make_unique()
    chunks = _create_chunks_for_shuffling(adata_concat, buffer_size, shuffle=shuffle)

    if var_subset is None:
        var_subset = adata_concat.var_names

    for i, chunk in enumerate(tqdm(chunks)):
        var_mask = adata_concat.var_names.isin(var_subset)
        adata_chunk = ad.AnnData(
            X=adata_concat.X[chunk, :][:, var_mask].persist(),
            obs=adata_concat.obs.iloc[chunk],
            var=adata_concat.var.loc[var_mask],
        )
        if shuffle:
            # shuffle adata in memory to break up individual chunks
            idxs = np.random.default_rng().permutation(np.arange(len(adata_chunk)))
            adata_chunk.X = adata_chunk.X[idxs, :]
            adata_chunk.obs = adata_chunk.obs.iloc[idxs]
        # convert to dense format before writing to disk
        if should_denseify:
            adata_chunk.X = adata_chunk.X.map_blocks(
                lambda xx: xx.toarray().astype("f4"), dtype="f4"
            )

        if output_format == "zarr":
            f = zarr.open_group(Path(output_path) / f"chunk_{i}.zarr", mode="w")
            _write_sharded(
                f,
                adata_chunk,
                chunk_size=chunk_size,
                shard_size=shard_size,
                compressors=zarr_compressor,
            )
        elif output_format == "h5ad":
            adata_chunk.write_h5ad(
                Path(output_path) / f"chunk_{i}.h5ad", compression=h5ad_compressor
            )
        else:
            raise ValueError(
                f"Unrecognized output_format: {output_format}. Only 'zarr' and 'h5ad' are supported."
            )


def _get_array_encoding_type(path: PathLike[str] | str):
    shards = list(Path(path).glob("chunk_*.zarr"))
    with open(shards[0] / "X" / "zarr.json") as f:
        encoding = json.load(f)
    return encoding["attributes"]["encoding-type"]


def add_h5ads_to_store(
    adata_paths: Iterable[PathLike[str]] | Iterable[str],
    output_path: PathLike[str] | str,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    zarr_compressor: Iterable[BytesBytesCodec] = (
        BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),
    ),
):
    """Add files to an existing Zarr store."""
    shards = list(Path(output_path).glob("chunk_*.zarr"))
    if len(shards) == 0:
        raise ValueError(
            "Store at `output_path` does not exist or is empty. Please run `create_store_from_h5ads` first."
        )
    encoding = _get_array_encoding_type(output_path)
    if encoding == "array":
        print(
            "Detected array encoding type. Will convert to dense format before writing."
        )

    adata_concat = _lazy_load_h5ads(adata_paths, chunk_size=chunk_size)
    var_mask = adata_concat.var_names.isin(adata_concat.var_names)
    adata_concat.obs_names_make_unique()
    chunks = _create_chunks_for_shuffling(
        adata_concat, np.ceil(len(adata_concat) / len(shards)), shuffle=True
    )

    for shard, chunk in tqdm(zip(shards, chunks, strict=False)):
        if encoding == "array":
            f = zarr.open_group(shard)
            adata_shard = ad.AnnData(
                X=ad.experimental.read_elem_lazy(f["X"])
                .map_blocks(sp.csr_matrix)
                .compute(),
                obs=ad.io.read_elem(f["obs"]),
                var=ad.io.read_elem(f["var"]),
            )
        else:
            adata_shard = ad.read_zarr(shard)
        adata = ad.concat(
            [
                adata_shard,
                ad.AnnData(
                    X=adata_concat.X[chunk, :][:, var_mask].compute(),
                    obs=adata_concat.obs.iloc[chunk],
                    var=adata_concat.var.loc[var_mask],
                ),
            ]
        )
        idxs_shuffled = np.random.default_rng().permutation(np.arange(len(adata)))
        f = zarr.open_group(shard, mode="w")
        if encoding == "array":
            adata.X = da.from_array(adata.X, chunks=(chunk_size, -1)).map_blocks(
                lambda xx: xx.toarray().astype("f4"), dtype="f4"
            )
        _write_sharded(
            f,
            adata[idxs_shuffled, :]
            if encoding == "csr_matrix"
            else adata[idxs_shuffled, :].copy(),
            chunk_size=chunk_size,
            shard_size=shard_size,
            compressors=zarr_compressor,
        )
