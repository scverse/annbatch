from __future__ import annotations

import asyncio
from collections import defaultdict
from functools import cache
from itertools import accumulate, chain, islice, pairwise
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import zarr
import zarr.core.sync as zsync
from scipy import sparse as sp
from torch.utils.data import IterableDataset
from upath import UPath

from .utils import sample_rows

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike


def _encode_str_to_int(obs_list: list[pd.DataFrame]):
    """Encodes string and categorical columns in a list of DataFrames to integer codes, modifying the DataFrames in place.

    Args:
        obs_list (list[pd.DataFrame]): A list of pandas DataFrames containing the data to encode.

    Returns:
        dict: A mapping of column names to dictionaries, where each dictionary maps integer codes
              to their corresponding unique string or category values.
    """
    categorical_mapping = {}
    for col in obs_list[0].select_dtypes(include=["object", "category"]).columns:
        uniques = set().union(*(df[col].unique() for df in obs_list))
        for df in obs_list:
            df[col] = pd.Categorical(
                df[col], categories=uniques, ordered=True
            ).codes.astype("i8")
        categorical_mapping[col] = dict(enumerate(uniques))
    return categorical_mapping


def load_store(
    path: PathLike, obs_columns: Iterable[str] = None
) -> tuple[list[zarr.Array], list[pd.DataFrame], dict[str, dict[int, str]]]:
    upath = UPath(path)
    arrays, obs_dfs = [], []
    for p in upath.iterdir():
        if p.suffix != ".zarr":
            continue
        p_x = p / "X"
        if p_x.protocol == "":
            store = p_x.as_posix()
        else:
            store = zarr.storage.FsspecStore.from_upath(UPath(p_x, asynchronous=True))
        arrays.append(zarr.open(store, mode="r"))

        g = zarr.open(p, mode="r")
        if obs_columns is None:
            obs = ad.io.read_elem(g["obs"])
        else:
            obs = pd.DataFrame(
                {col: ad.io.read_elem(g[f"obs/{col}"]) for col in obs_columns}
            )
        obs_dfs.append(obs)

    categorical_mapping = _encode_str_to_int(obs_dfs)

    return arrays, obs_dfs, categorical_mapping


def _batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class ZarrDenseDataset(IterableDataset):
    def __init__(
        self,
        x_list: list[zarr.Array],
        obs_list: list[pd.DataFrame],
        obs_column: str,
        shuffle: bool = True,
        preload_nchunks: int = 8,
    ):
        self.arrays = x_list
        self.obs = obs_list
        self.obs_column = obs_column
        self.shuffle = shuffle
        self.preload_chunks = preload_nchunks

        self.n_obs_list: list[int] = []  # number of observations for each array
        self.chunks_lengths: list[int] = []  # chunk length for each array
        arrays_chunks: list[list[int]] = []  # list of chunk indices for each array
        arrays_nchunks: list[int] = []  # number of chunks for each array
        for array in x_list:
            self.n_obs_list.append(array.shape[0])
            self.chunks_lengths.append(array.chunks[0])
            array_nchunks = array.nchunks
            arrays_nchunks.append(array_nchunks)
            arrays_chunks.append(np.arange(array_nchunks))

        self.n_obs = sum(self.n_obs_list)
        # assumes the same for all arrays
        array0 = x_list[0]
        self.n_vars = array0.shape[1]
        self.dtype = array0.dtype
        self.order = array0.order

        self.chunks = np.hstack(arrays_chunks)
        self.array_idxs = np.repeat(np.arange(len(self.arrays)), arrays_nchunks)
        # pre-compute chunk slices
        # slices are needed because we want to iterate over (logical) chunks, not (physical) shards
        # but in zarr array.blocks[i] returns whole shards unlike dask
        self.chunks_slices: list[slice] = []
        for i, chunk_idx in enumerate(self.chunks):
            self.chunks_slices.append(self._chunk_slice(chunk_idx, self.array_idxs[i]))

    def _chunk_slice(self, chunk_idx: int, array_idx: int):
        chunk_length = self.chunks_lengths[array_idx]
        array_n_obs = self.n_obs_list[array_idx]
        start = chunk_length * chunk_idx
        stop = min(chunk_length * (chunk_idx + 1), array_n_obs)
        return slice(start, stop)

    async def _fetch_chunks_x(self, chunk_idxs: list[int]):
        tasks = []
        for i in chunk_idxs:
            array_idx = self.array_idxs[i]
            array = self.arrays[array_idx]
            tasks.append(array._async_array.getitem(self.chunks_slices[i]))
        return await asyncio.gather(*tasks)

    def _fetch_chunks_obs(self, chunk_idxs: list[int]):
        obs = []
        for i in chunk_idxs:
            array_idx = self.array_idxs[i]
            obs.append(
                self.obs[array_idx][self.obs_column]
                .iloc[self.chunks_slices[i]]
                .to_numpy()
            )
        return obs

    def __iter__(self):
        chunks_global = np.arange(len(self.chunks))
        if self.shuffle:
            np.random.shuffle(chunks_global)  # noqa: NPY002

        for batch in _batched(chunks_global, self.preload_chunks):
            yield from sample_rows(
                list(zsync.sync(self._fetch_chunks_x(batch))),
                self._fetch_chunks_obs(batch),
                self.shuffle,
            )

    def __len__(self):
        return self.n_obs


class MultiBasicIndexer:
    def __init__(self, indexers: list):
        self.shape = [
            sum(i.shape[k] for i in indexers) for k in range(len(indexers[0].shape))
        ]
        self.drop_axes = indexers[0].drop_axes  # maybe?
        self.indexers = indexers

    def __iter__(self):
        for i in self.indexers:
            yield from i


def chunked(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


class SparseDataset(IterableDataset):
    """A loader for on-disk sparse data.

    This loader batches together slice requests to the underlying sparse stores to acheive higher performance.
    This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
    At the moment, the loader is agnostic to the on-disk chunking/sharding, but initial tests show excellent performance for
    sharded data/indices where the shards are extremely small (only ~30,000 elements).
    """

    def __init__(
        self,
        sparse_datasets: list[ad.abc.CSRDataset],
        *,
        chunk_size: int = 512,
        preload_nchunks: int = 32,
        shuffle: bool = True,
    ):
        self.arrays = sparse_datasets
        self.n_obs = sum(a.shape[0] for a in self.arrays)
        self.chunk_size = chunk_size
        self.preload_nchunks = preload_nchunks
        self.shuffle = shuffle
        self._var_size = self.arrays[0].shape[1]

    def _get_relative_obs_indices(self, index: slice) -> list[tuple[slice, int]]:
        min_idx = 0
        max_idx = 0
        res = []
        for anndata_idx, array in enumerate(self.arrays):
            max_idx += array.shape[0]
            if (index.start >= min_idx) and (index.stop < max_idx):
                return [
                    (slice(index.start - min_idx, index.stop - min_idx), anndata_idx)
                ]
            if (index.start >= min_idx) and (index.stop >= max_idx):
                res += [(slice(index.start - min_idx, max_idx), anndata_idx)]
            if (index.start < min_idx) and (index.stop < max_idx):
                return res + [(slice(min_idx, index.stop), anndata_idx)]
            min_idx += array.shape[0]
        raise StopIteration()

    @cache  # noqa: B019
    def get_groups(self, anndata_idx: int):
        indptr = self.arrays[anndata_idx].group["indptr"][...]
        indices = self.arrays[anndata_idx].group["indices"]
        data = self.arrays[anndata_idx].group["data"]
        return indptr, indices, data

    async def _fetch_data(self, anndata_index, slices, indptr, indices, data):
        indptr_indices = [indptr[s] for s in slices]
        indptr_limits = [slice(i[0], i[-1]) for i in indptr_indices]
        indexer_data = MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer(
                    l, shape=data.metadata.shape, chunk_grid=data.metadata.chunk_grid
                )
                for l in indptr_limits
            ]
        )
        indexer_indices = MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer(
                    l,
                    shape=indices.metadata.shape,
                    chunk_grid=indices.metadata.chunk_grid,
                )
                for l in indptr_limits
            ]
        )
        data_np, indices_np = await asyncio.gather(
            data._async_array._get_selection(
                indexer_data, prototype=zarr.core.buffer.default_buffer_prototype()
            ),
            indices._async_array._get_selection(
                indexer_indices, prototype=zarr.core.buffer.default_buffer_prototype()
            ),
        )
        gaps = (s1.start - s0.stop for s0, s1 in pairwise(indptr_limits))
        offsets = accumulate(chain([indptr_limits[0].start], gaps))
        start_indptr = indptr_indices[0] - next(offsets)
        if len(slices) < 2:  # there is only one slice so no need to concatenate
            return sp.csr_matrix(
                (data_np, indices_np, start_indptr),
                shape=(start_indptr.shape[0] - 1, self._var_size),
            )
        end_indptr = np.concatenate(
            [s[1:] - o for s, o in zip(indptr_indices[1:], offsets, strict=True)]
        )
        indptr_np = np.concatenate([start_indptr, end_indptr])
        return sp.csr_matrix(
            (data_np, indices_np, indptr_np),
            shape=(indptr_np.shape[0] - 1, self._var_size),
        )

    def _slices_to_slices_with_array_index(
        self, slices: list[slice]
    ) -> defaultdict[int, list[slice]]:
        anndata_index_to_slices: defaultdict[int, list[slice]] = defaultdict(list)
        for slice in slices:
            for relative_obs_indices in self._get_relative_obs_indices(slice):
                anndata_index_to_slices[relative_obs_indices[1]] += [
                    relative_obs_indices[0]
                ]
        return anndata_index_to_slices

    def __iter__(self):
        shuffled_chunk_indices = np.array(list(range(self.n_obs // self.chunk_size)))
        if self.shuffle:
            np.random.shuffle(shuffled_chunk_indices)
        for i, _ in enumerate(self.arrays):
            self.get_groups(i)  # TODO: asyncify

        for chunks in chunked(shuffled_chunk_indices, self.preload_nchunks):

            async def get():
                slices = [
                    slice(index, min(self.n_obs, index + self.chunk_size) + 1)
                    for index in chunks
                ]
                tasks = []
                anndata_index_to_slices = self._slices_to_slices_with_array_index(
                    slices
                )
                for anndata_idx in anndata_index_to_slices:
                    tasks.append(
                        self._fetch_data(
                            anndata_idx,
                            anndata_index_to_slices[anndata_idx],
                            *self.get_groups(anndata_idx),
                        )
                    )
                return await asyncio.gather(*tasks)

            chunks = zsync.sync(get())
            yield from sample_rows(
                list(chunks),
                None,
                self.shuffle,
            )

    def __len__(self):
        return self.n_obs
