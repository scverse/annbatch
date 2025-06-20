import asyncio
from collections.abc import Iterable
from itertools import islice
from os import PathLike

import anndata as ad
import numpy as np
import pandas as pd
import zarr
import zarr.core.sync as zsync
from torch.utils.data import IterableDataset
from upath import UPath

from .utils import sample_rows


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
