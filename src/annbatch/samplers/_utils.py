"""Utility functions for samplers."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from annbatch.utils import WorkerHandle


def is_in_worker() -> bool:
    """Check if currently running inside a torch DataLoader worker.

    Returns
    -------
    bool
        True if inside a DataLoader worker, False otherwise.
    """
    if find_spec("torch"):
        from torch.utils.data import get_worker_info

        return get_worker_info() is not None
    return False


def get_worker_handle(rng: np.random.Generator) -> WorkerHandle | None:
    """Get a WorkerHandle if running inside a torch DataLoader worker.

    Parameters
    ----------
    rng
        The RNG to spawn worker-specific RNGs from.

    Returns
    -------
    WorkerHandle | None
        A WorkerHandle if inside a DataLoader worker, None otherwise.
    """
    if is_in_worker():
        from annbatch.utils import WorkerHandle

        return WorkerHandle(rng)
    return None


def validate_batch_size(batch_size: int, chunk_size: int, preload_nchunks: int) -> None:
    """Validate batch_size against chunk_size and preload_nchunks constraints.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk.
    preload_nchunks
        Number of chunks to load per iteration.

    Raises
    ------
    ValueError
        If batch_size exceeds the total loaded size (chunk_size * preload_nchunks)
        or if the total loaded size is not divisible by batch_size.
    """
    preload_size = chunk_size * preload_nchunks

    if batch_size > preload_size:
        raise ValueError(
            "batch_size cannot exceed chunk_size * preload_nchunks. "
            f"Got batch_size={batch_size}, but max is {preload_size}."
        )
    if preload_size % batch_size != 0:
        raise ValueError(
            "chunk_size * preload_nchunks must be divisible by batch_size. "
            f"Got {preload_size} % {batch_size} = {preload_size % batch_size}."
        )
