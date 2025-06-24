from __future__ import annotations

import pathlib
import random
import warnings
from typing import TYPE_CHECKING

import anndata as ad
import dask
import numpy as np
import pandas as pd
import zarr
from torch.utils.data import IterableDataset, get_worker_info

from .utils import sample_rows

if TYPE_CHECKING:
    from typing import Literal


def read_lazy(path, obs_columns: list[str] = None, read_obs_lazy: bool = False):
    """Read an individual shard of a Zarr store into an AnnData object.

    Parameters
    ----------
    path : pathlib.Path | str
        Path to individual Zarr based AnnData shard.
    obs_columns : list[str] | None
        List of observation columns to read. If None, all columns are read.
    read_obs_lazy : bool
        If True, read the obs DataFrame lazily. This is useful for large obs DataFrames.
    """
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
        var=ad.io.read_elem(g["var"]),
    )

    return adata


def read_lazy_store(
    path, obs_columns: list[str] | None = None, read_obs_lazy: bool = False
):
    """Read a Zarr store containing multiple shards into a single AnnData object.

    Parameters
    ----------
    path : pathlib.Path | str
        Path to the Zarr store containing multiple shards.
    obs_columns : list[str] | None
        List of observation columns to read. If None, all columns are read.
    read_obs_lazy : bool
        If True, read the obs DataFrame lazily. This is useful for large obs DataqFrames.
    """
    path = pathlib.Path(path)

    with warnings.catch_warnings():
        # Ignore zarr v3 warnings
        warnings.simplefilter("ignore")
        adata = ad.concat(
            [
                read_lazy(path / shard, obs_columns, read_obs_lazy)
                for shard in path.iterdir()
                if str(shard).endswith(".zarr")
            ]
        )

    return adata


def _combine_chunks(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


class DaskDataset(IterableDataset):
    """Dask-based IterableDataset for loading AnnData objects in chunks.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to yield samples from.
    label_column : str
        The name of the column in `adata.obs` that contains the labels.
    n_chunks : int = 8
        Number of chunks of the underlying dask.array to load at a time.
        Loading more chunks at a time can improve performance and will also improve randomness.
        However, it increases memory usage.
    shuffle : bool = True
        Whether to yield samples in a random order.
    dask_scheduler : Literal["synchronous", "threads"] = "threads"
        The Dask scheduler to use for parallel computation.
        "synchronous" for single-threaded execution, "threads" for multithreaded execution
    n_workers : int | None = None
        Number of Dask workers to use. If None, the number of workers is determined by Dask.

    Examples:
    --------
    >>> from arrayloaders.io.dask_loader import DaskDataset, read_lazy_store
    >>> from torch.utils.data import DataLoader
    >>> label_column = "y"
    >>> adata = read_lazy_store("path/to/zarr/store", obs_columns=[label_column])
    >>> dataset = DaskDataset(adata, label_column=label_column, n_chunks=8, shuffle=True)
    >>> dataloader = DataLoader(dataset, batch_size=2048, num_workers=4, drop_last=True)
    >>> for batch in dataloader:
    ...     x, y = batch
    ...     # Process the batch
    """

    def __init__(
        self,
        adata: ad.AnnData,
        label_column: str,
        n_chunks: int = 8,
        shuffle: bool = True,
        dask_scheduler: Literal["synchronous", "threads"] = "threads",
        n_workers: int | None = None,
    ):
        self.adata = adata
        self.label_column = label_column
        self.n_chunks = n_chunks
        self.shuffle = shuffle
        self.dask_scheduler = dask_scheduler
        self.n_workers = n_workers

        self.worker_info = get_worker_info()
        if self.worker_info is None:
            self.rng_split = random.Random()  # noqa: S311
        else:
            # This is used for the _get_chunks function
            # Use the same seed for all workers that the resulting splits are the same across workers
            # torch default seed is `base_seed + worker_id`. Hence, subtract worker_id to get the base seed
            # fmt: off
            self.rng_split = random.Random(self.worker_info.seed - self.worker_info.id)  # noqa: S311
            # fmt: on

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
            x_list = dask.compute(
                [self.adata.X.blocks[i] for i in block_idxs],
                scheduler=self.dask_scheduler,
            )[0]
            obs_list = [
                self.adata.obs[self.label_column].iloc[s].to_numpy() for s in slices
            ]
            yield from sample_rows(x_list, obs_list, self.shuffle)

    def __len__(self):
        return len(self.adata)
