"""RandomSampler -- shuffled chunk-based sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from annbatch.samplers._chunk_sampler import ChunkSampler

if TYPE_CHECKING:
    import numpy as np


class RandomSampler(ChunkSampler):
    """Shuffled chunk-based sampler for batched data access.

    TODO: docstring desc

    Parameters
    ----------
    chunk_size
        Size of each on-disk chunk range yielded.
    preload_nchunks
        Number of chunks to load per I/O request.
    batch_size
        Number of observations per batch.
    replacement
        If ``True``, draw random chunks with replacement.
    num_samples
        Total number of observations to draw.  Defaults to the
        effective observation count when ``None``.
    drop_last
        Whether to drop the last incomplete batch.
    mask
        A slice defining the observation range to sample from.
    rng
        Random number generator for shuffling.
    """

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        replacement: bool = False,
        num_samples: int | None = None,
        drop_last: bool = False,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            replacement=replacement,
            num_samples=num_samples,
            shuffle=True,
            drop_last=drop_last,
            mask=mask,
            rng=rng,
        )
