from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

import anndata as ad
import dask
import numpy as np
import pandas as pd
import zarr
from torch.utils.data import IterableDataset

from .utils import WorkerHandle, check_lt_1, sample_rows

if TYPE_CHECKING:
    from typing import Literal


# TODO: refactor to read full lazy and then simply pick out the needed columns into memory instead of having `read_obs_lazy` as a separate arg
def read_lazy(path, obs_columns: list[str] | None = None, read_obs_lazy: bool = False):
    """Reads an individual shard of a Zarr store into an AnnData object.

    Args:
        path: Path to individual Zarr-based AnnData shard.
        obs_columns: List of observation columns to read. If None, all columns are read.
        read_obs_lazy: If True, reads the obs DataFrame lazily. Useful for large obs DataFrames.

    Returns:
        AnnData object loaded from the specified shard.
    """
    g = zarr.open(path, mode="r")

    adata = ad.experimental.read_lazy(g)
    # TODO: Adapt dask code below to just handle an in-memory xarray data array
    if not read_obs_lazy:
        if obs_columns is None:
            adata.obs = ad.io.read_elem(g["obs"])
        else:
            adata.obs = pd.DataFrame(
                {col: ad.io.read_elem(g[f"obs/{col}"]) for col in obs_columns}
            )

    return adata


def read_lazy_store(
    path, obs_columns: list[str] | None = None, read_obs_lazy: bool = False
):
    """Reads a Zarr store containing multiple shards into a single AnnData object.

    Args:
        path: Path to the Zarr store containing multiple shards.
        obs_columns: List of observation columns to read. If None, all columns are read.
        read_obs_lazy: If True, reads the obs DataFrame lazily. Useful for large obs DataFrames.

    Returns:
        AnnData: The concatenated AnnData object loaded from all shards.
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

    Args:
        adata: The AnnData object to yield samples from.
        label_column: The name of the column in `adata.obs` that contains the labels.
        n_chunks: Number of chunks of the underlying dask.array to load at a time.
            Loading more chunks at a time can improve performance and randomness, but increases memory usage.
            Defaults to 8.
        shuffle: Whether to yield samples in a random order. Defaults to True.
        dask_scheduler: The Dask scheduler to use for parallel computation.
            "synchronous" for single-threaded execution, "threads" for multithreaded execution. Defaults to "threads".
        n_workers: Number of Dask workers to use. If None, the number of workers is determined by Dask.

    Examples:
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
        check_lt_1(
            [adata.shape[0], n_chunks],
            ["Size of anndata obs dimension", "Number of chunks"],
        )
        self.adata = adata
        self.label_column = label_column
        self.n_chunks = n_chunks
        self.shuffle = shuffle
        self.dask_scheduler = dask_scheduler
        self.n_workers = n_workers

        self.worker_handle = WorkerHandle()

    def _get_chunks(self):
        chunk_boundaries = np.cumsum([0] + list(self.adata.X.chunks[0]))
        slices = [
            slice(int(start), int(end))
            for start, end in zip(
                chunk_boundaries[:-1], chunk_boundaries[1:], strict=False
            )
        ]
        blocks_idxs = np.arange(len(self.adata.X.chunks[0]))
        chunks = list(zip(blocks_idxs, slices, strict=False))

        if self.shuffle:
            self.worker_handle.shuffle(chunks)

        return self.worker_handle.get_part_for_worker(chunks)

    def __iter__(self):
        for chunks in _combine_chunks(self._get_chunks(), self.n_chunks):
            block_idxs, slices = zip(*chunks, strict=False)
            x_list = dask.compute(
                [self.adata.X.blocks[i] for i in block_idxs],
                scheduler=self.dask_scheduler,
            )[0]
            obs_list = [
                self.adata.obs[self.label_column].iloc[s].to_numpy() for s in slices
            ]
            yield from sample_rows(x_list, obs_list, shuffle=self.shuffle)

    def __len__(self):
        return len(self.adata)
