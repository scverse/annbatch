"""SequentialSampler -- ordered chunk-based sampler."""

from __future__ import annotations

import warnings

from annbatch.samplers._chunk_sampler import ChunkSampler


class SequentialSampler(ChunkSampler):
    """Ordered chunk-based sampler for batched data access.

    Chunks are emitted in sequential order and every observation in the
    range is visited exactly once.  Does not support multiple
    data-loading workers. Usually used for evaluation or inference.

    See :class:`~annbatch.RandomSampler` for a shuffled
    alternative.

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
