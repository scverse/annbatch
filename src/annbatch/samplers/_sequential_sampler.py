"""SequentialSampler -- ordered chunk-based sampler."""

from __future__ import annotations

import warnings

from annbatch.samplers._chunk_sampler import ChunkSampler


class SequentialSampler(ChunkSampler):
    """Ordered chunk-based sampler for batched data access.

    TODO: docstring desc

    Parameters
    ----------
    chunk_size
        Size of each on-disk chunk range yielded.
    preload_nchunks
        Number of chunks to load per I/O request.
    batch_size
        Number of observations per batch.
    drop_last
        Whether to drop the last incomplete batch.
    mask
        A slice defining the observation range to sample from.
    """

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        drop_last: bool = False,
        mask: slice | None = None,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ChunkSampler is deprecated", category=DeprecationWarning)
            super().__init__(
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=batch_size,
                replacement=False,
                num_samples=None,
                shuffle=False,
                drop_last=drop_last,
                mask=mask,
            )
