import random
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import zarr
from tqdm import tqdm
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec, BloscShuffle


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
    adata: ad.AnnData, shuffle_buffer_size: int = 1_048_576
):
    chunk_boundaries = np.cumsum([0] + list(adata.X.chunks[0]))
    slices = [
        slice(int(start), int(end))
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    ]
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
    compressors: Iterable[BytesBytesCodec] = (
        BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle),
    ),
    shuffle_buffer_size: int = 1_048_576,
):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr
    print("setting ad.settings.zarr_write_format to 3")
    adata_concat = _lazy_load_h5ads(adata_paths, chunk_size=chunk_size)
    adata_concat.obs_names_make_unique()
    shuffle_chunks = _create_chunks_for_shuffling(adata_concat, shuffle_buffer_size)

    if var_subset is None:
        var_subset = adata_concat.var_names.tolist()

    for i, chunk in enumerate(tqdm(shuffle_chunks)):
        var_mask = adata_concat.var_names.isin(var_subset)
        adata_chunk = ad.AnnData(
            X=adata_concat.X[chunk, :][:, var_mask].persist(),
            obs=adata_concat.obs.iloc[chunk],
            var=adata_concat.var.loc[var_mask],
        )
        # shuffle adata in memory to break up individual chunks
        idxs = np.random.default_rng().permutation(np.arange(len(adata_chunk)))
        adata_chunk.X = adata_chunk.X[idxs, :]
        adata_chunk.obs = adata_chunk.obs.iloc[idxs]
        # convert to dense format before writing to disk
        adata_chunk.X = adata_chunk.X.map_blocks(
            lambda xx: xx.toarray().astype("f4"), dtype="f4"
        )
        f = zarr.open(Path(output_path) / f"chunk_{i}.zarr", mode="w")
        _write_sharded(
            f,
            adata_chunk,
            chunk_size=chunk_size,
            shard_size=shard_size,
            compressors=compressors,
        )
