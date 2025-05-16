import random
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from torch.utils.data import IterableDataset, get_worker_info


def read_lazy(path, obs_columns: list[str] = None, read_obs_lazy: bool = False):
    g = zarr.open(path, mode="r")
    if read_obs_lazy:
        obs = ad.experimental.read_elem_lazy(g["obs"])
    else:
        if obs_columns is None:
            obs = ad.io.read_elem(g["obs"])
        else:
            obs = pd.DataFrame(
                {col: ad.io.read_elem(g[f"obs/{col}"]) for col in obs_columns}
            )

    adata = ad.AnnData(
        X=ad.experimental.read_elem_lazy(g["X"]),
        obs=obs,
    )

    return adata


def _combine_chunks(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _yield_samples(x, y, shuffle=True):
    num_samples = len(x)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.default_rng().shuffle(indices)

    for i in indices:
        yield x[i], y[i]


class ZarrDataset(IterableDataset):
    def __init__(
        self,
        adata: ad.AnnData,
        label_column: str,
        n_chunks: int = 8,
        shuffle: bool = True,
        dask_scheduler: Literal["synchronous", "threads"] = "threads",
        n_workers: int = None,
    ):
        self.adata = adata
        self.label_column = label_column
        self.n_chunks = n_chunks
        self.shuffle = shuffle
        self.dask_scheduler = dask_scheduler
        self.n_workers = n_workers

        self.worker_info = get_worker_info()
        if self.worker_info is None:
            self.rng_split = random.Random()
        else:
            # This is used for the _get_chunks function
            # Use the same seed for all workers that the resulting splits are the same across workers
            # torch default seed is `base_seed + worker_id`. Hence, subtract worker_id to get the base seed
            self.rng_split = random.Random(self.worker_info.seed - self.worker_info.id)

    def _get_chunks(self):
        chunk_boundaries = np.cumsum([0] + list(self.adata.X.chunks[0]))
        slices = [
            slice(int(start), int(end))
            for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
        ]
        blocks_idxs = np.arange(len(self.adata.X.chunks[0]))
        chunks = list(zip(blocks_idxs, slices))

        if self.shuffle:
            self.rng_split.shuffle(chunks)

        if self.worker_info is None:
            return chunks
        else:
            num_workers, worker_id = self.worker_info.num_workers, self.worker_info.id
            chunks_per_worker = len(chunks) // num_workers
            start = worker_id * chunks_per_worker
            end = (
                (worker_id + 1) * chunks_per_worker
                if worker_id != num_workers - 1
                else None
            )
            return chunks[start:end]

    def __iter__(self):
        for chunks in _combine_chunks(self._get_chunks(), self.n_chunks):
            block_idxs, slices = zip(*chunks)
            x = self.adata.X.blocks[list(block_idxs)].compute(
                scheduler=self.dask_scheduler
            )
            obs = self.adata.obs[self.label_column].iloc[np.r_[slices]].to_numpy()
            yield from _yield_samples(x, obs, self.shuffle)

    def __len__(self):
        return len(self.adata)
