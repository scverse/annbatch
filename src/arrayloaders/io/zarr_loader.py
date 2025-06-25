from __future__ import annotations

import asyncio
import math
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
    from collections.abc import Iterable, Iterator
    from os import PathLike


def _encode_str_to_int(obs_list: list[pd.DataFrame]):
    """Encodes string and categorical columns in a list of DataFrames to integer codes, modifying the DataFrames in place.

    Args:
        obs_list: A list of pandas DataFrames containing the data to encode.

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
        self._arrays = x_list
        self._obs = obs_list
        self._obs_column = obs_column
        self._shuffle = shuffle
        self._preload_nchunks = preload_nchunks
        self._rng = np.random.default_rng()

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

        self._n_obs = sum(self.n_obs_list)
        # assumes the same for all arrays
        array0 = x_list[0]
        self.n_vars = array0.shape[1]
        self.dtype = array0.dtype
        self.order = array0.order

        self.chunks = np.hstack(arrays_chunks)
        self.array_idxs = np.repeat(np.arange(len(self._arrays)), arrays_nchunks)
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
            array = self._arrays[array_idx]
            tasks.append(array._async_array.getitem(self.chunks_slices[i]))
        return await asyncio.gather(*tasks)

    def _fetch_chunks_obs(self, chunk_idxs: list[int]):
        obs = []
        for i in chunk_idxs:
            array_idx = self.array_idxs[i]
            obs.append(
                self._obs[array_idx][self._obs_column]
                .iloc[self.chunks_slices[i]]
                .to_numpy()
            )
        return obs

    def __iter__(self):
        chunks_global = np.arange(len(self.chunks))
        if self._shuffle:
            np.random.shuffle(chunks_global)  # noqa: NPY002

        for batch in _batched(chunks_global, self._preload_nchunks):
            yield from sample_rows(
                list(zsync.sync(self._fetch_chunks_x(batch))),
                self._fetch_chunks_obs(batch),
                self._shuffle,
            )

    def __len__(self):
        return self._n_obs


# TODO: make this part of the public zarr or zarrs-python API.
# We can do chunk coalescing in zarrs based on integer arrays, so I think
# there would make sense with ezclump or similar.
class MultiBasicIndexer:
    def __init__(self, indexers: list):
        self.shape = [
            sum(i.shape[k] for i in indexers) for k in range(len(indexers[0].shape))
        ]
        self.drop_axes = indexers[0].drop_axes  # maybe?
        self.indexers = indexers

    def __iter__(self):
        total = 0
        for i in self.indexers:
            for c in i:
                gap = c[2][0].stop - c[2][0].start
                yield type(c)(c[0], c[1], (slice(total, total + gap)), c[3])
                total += gap


def chunked(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


class ZarrSparseDataset(IterableDataset):
    def __init__(
        self,
        sparse_datasets: list[ad.abc.CSRDataset],
        *,
        chunk_size: int = 512,
        preload_nchunks: int = 32,
        shuffle: bool = True,
    ):
        """A loader for on-disk sparse data.

        This loader batches together slice requests to the underlying sparse stores to acheive higher performance.
        This custom code to do this task will be upstreamed into anndata at some point and no longer rely on private zarr apis.
        At the moment, the loader is agnostic to the on-disk chunking/sharding, but initial tests show excellent performance for
        sharded data/indices where the shards are extremely small (only ~30,000 elements).

        Args:
            sparse_datasets: Disk-backed sparse data.  For now, must all be of the same var (i.e., axis 1) size.
            chunk_size: The obs size (i.e., axis 0) of contiguous array data to fetch, by default 512
            preload_nchunks: The number of chunks of contiguous array data to fetch, by default 32
            shuffle: Whether or not to shuffle the data, by default True
        """
        if not all(s.shape[1] == sparse_datasets[0].shape[1] for s in sparse_datasets):
            raise ValueError(
                "TODO: Implement join indexing for sparse datasets with different axis=1 shapes"
            )
        self._sparse_datasets = sparse_datasets
        self._n_obs = sum(a.shape[0] for a in self._sparse_datasets)
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._shuffle = shuffle
        self._var_size = self._sparse_datasets[0].shape[1]
        self._groups_cache: dict[
            int, tuple[np.ndarray, zarr.AsyncArray, zarr.AsyncArray]
        ] = {}
        self._rng = np.random.default_rng()

    def _get_relative_obs_indices(self, index: slice) -> list[tuple[slice, int]]:
        """Generate a slice relative to a dataset given a global slice index over all datasets.

        For a given slice indexer of axis 0, return a new slice relative to the on-disk
        data it represents given the number of total observations as well as the index of
        the underlying data on disk from the argument `sparse_datasets` to the initializer.

        For example, given slice index (10, 15), for 4 datasets each with size 5 on axis zero,
        this function returns ((0,5), 2) representing slice (0,5) along axis zero of sparse dataset 2.

        Args:
            index: The queried slice.

        Returns:
            A slice relative to the dataset it represents as well as the index of said dataset in `sparse_datasets`.
        """
        min_idx = index.start
        max_idx = index.stop
        curr_pos = 0
        slices = []
        for anndata_idx, array in enumerate(self._sparse_datasets):
            array_start = curr_pos
            n_obs = array.shape[0]
            array_end = curr_pos + n_obs

            start = max(min_idx, array_start)
            stop = min(max_idx, array_end)
            if start < stop:
                relative_start = start - array_start
                relative_stop = stop - array_start
                slices.append((slice(relative_start, relative_stop), anndata_idx))
            curr_pos += n_obs
        return slices

    async def _get_sparse_elems(
        self, anndata_idx: int
    ) -> tuple[np.ndarray, zarr.AsyncArray, zarr.AsyncArray]:
        """Return the arrays (zarr or otherwise) needed to represent on-disk data at a given index.

        Args:
            anndata_idx: The index of the dataset whose arrays are sought.

        Returns:
            The arrays representing the sparse data.
        """
        if anndata_idx not in self._groups_cache:

            async def get_arr(idx):
                indptr = await self._sparse_datasets[idx].group._async_group.getitem(
                    "indptr"
                )
                return await asyncio.gather(
                    indptr.getitem(Ellipsis),
                    self._sparse_datasets[idx].group._async_group.getitem("indices"),
                    self._sparse_datasets[idx].group._async_group.getitem("data"),
                )

            arrs = await asyncio.gather(
                *(get_arr(idx) for idx in range(len(self._sparse_datasets)))
            )
            for idx, arr in enumerate(arrs):
                self._groups_cache[idx] = tuple(arr)
        return self._groups_cache[anndata_idx]

    async def _fetch_data(
        self,
        slices: list[slice],
        indptr: np.ndarray,
        indices: zarr.AsyncArray,
        data: zarr.AsyncArray,
    ) -> sp.csr_matrix:
        """Fetch the data for given slices and the arrays representing a sparse dataset on-disk.

        Args:
            slices: The indexing slices to fetch.
            indptr: The indptr of a csr matrix.
            indices: The indices of a csr matrix.
            data: The data of a csr matrix.

        Returns:
            The in-memory csr data.
        """
        indptr_indices = [indptr[slice(s.start, s.stop + 1)] for s in slices]
        indptr_limits = [slice(i[0], i[-1]) for i in indptr_indices]
        indexer_data = MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer(
                    (l,), shape=data.metadata.shape, chunk_grid=data.metadata.chunk_grid
                )
                for l in indptr_limits
            ]
        )
        indexer_indices = MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer(
                    (l,),
                    shape=indices.metadata.shape,
                    chunk_grid=indices.metadata.chunk_grid,
                )
                for l in indptr_limits
            ]
        )
        data_np, indices_np = await asyncio.gather(
            data._get_selection(
                indexer_data, prototype=zarr.core.buffer.default_buffer_prototype()
            ),
            indices._get_selection(
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
        """Given a list of slices, give the lookup between on-disk datasets and slices relative to that dataset.

        Args:
            slices: Slices to relative to the on-disk datasets.

        Returns:
            A lookup between the dataset and its indexing slices.
        """
        anndata_index_to_slices: defaultdict[int, list[slice]] = defaultdict(list)
        for slice in slices:
            for relative_obs_indices in self._get_relative_obs_indices(slice):
                anndata_index_to_slices[relative_obs_indices[1]] += [
                    relative_obs_indices[0]
                ]
        return anndata_index_to_slices

    def __iter__(self) -> Iterator[sp.csr_matrix]:
        maybe_shuffled_chunk_indices = np.array(
            list(range(math.ceil(self._n_obs / self._chunk_size)))
        )
        if self._shuffle:
            self._rng.shuffle(maybe_shuffled_chunk_indices)
        zsync.sync(
            self._get_sparse_elems(0)
        )  # activate cache, TODO: better way of handling this?
        for chunks in chunked(maybe_shuffled_chunk_indices, self._preload_nchunks):

            async def get(chunks):
                slices = [
                    slice(
                        index * self._chunk_size,
                        min(self._n_obs, (index + 1) * self._chunk_size),
                    )
                    for index in chunks
                ]
                tasks = []
                anndata_index_to_slices = self._slices_to_slices_with_array_index(
                    slices
                )
                for anndata_idx in anndata_index_to_slices:
                    tasks.append(
                        self._fetch_data(
                            anndata_index_to_slices[anndata_idx],
                            *(await self._get_sparse_elems(anndata_idx)),
                        )
                    )
                return await asyncio.gather(*tasks)

            chunks = zsync.sync(get(chunks))
            yield from sample_rows(
                list(chunks),
                None,
                self._shuffle,
            )

    def __len__(self):
        return self._n_obs
