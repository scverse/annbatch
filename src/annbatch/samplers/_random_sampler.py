"""RandomSampler -- shuffled chunk-based sampler."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from annbatch.samplers._chunk_sampler import ChunkSampler

if TYPE_CHECKING:
    import numpy as np


class RandomSampler(ChunkSampler):
    """Shuffled chunk-based sampler for batched data access.

    Chunks are drawn in random order.  With ``replacement=False`` (the default),
    every observation in the range is visited exactly once per epoch.
    With ``replacement=True``, chunks are drawn independently at random
    and ``num_samples`` controls the total number of observations drawn.

    When the observation range is smaller than ``chunk_size``, sampling
    without replacement works normally (a single smaller chunk is
    yielded).  With replacement, this is only allowed when
    ``num_samples`` does not exceed the observation range.

    See :class:`~annbatch.samplers.SequentialSampler` for an ordered
    (non-shuffled) alternative.

    Parameters
    ----------
    rng
        A :class:`numpy.random.Generator` used for shuffling.  When
        ``None``, a new default generator is created.
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ChunkSampler is deprecated", category=DeprecationWarning)
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
