"""RandomSampler -- shuffled chunk-based sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from annbatch.samplers._chunk_sampler import _ChunkSampler

if TYPE_CHECKING:
    import numpy as np


class RandomSampler(_ChunkSampler):
    """Shuffled chunk-based sampler for batched data access.

    Chunks are drawn in random order.  With ``replacement=False`` (the default),
    every observation in the range is visited exactly once per epoch up to `drop_last`.
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
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    mask
        A slice defining the observation range to sample from (start:stop).
    preload_nchunks
        Number of chunks to load per iteration.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator for shuffling. Note that :func:`torch.manual_seed`
        has no effect on reproducibility here; pass a seeded
        :class:`numpy.random.Generator` to control randomness.
    replacement
        If ``True``, draw random chunks with replacement, allowing the
        same observations to appear more than once.
    num_samples
        Total number of observations to draw.  When ``None`` (the
        default), equals the effective observation range.  Must be
        positive when set and less than the number of observations to be
        yielded when ``replacement=False``.
    copy
        Whether each load request's ``splits`` are copied out of the buffer the sampler
        reshuffles for the next request. The :class:`~annbatch.Loader` consumes a request
        before asking for the next, so it does not need this; set it to ``True`` if you
        hold on to load requests, e.g. ``list(sampler.sample(n_obs))``.
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
        copy: bool = False,
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
            copy=copy,
        )
