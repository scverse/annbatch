"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng, check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class ChunkSampler(Sampler):
    """Chunk-based sampler for batched data access.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    mask
        A slice defining the observation range to sample from (start:stop).
    shuffle
        Whether to shuffle chunk and index order.
    preload_nchunks
        Number of chunks to load per iteration.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator for shuffling. Note that ``torch.manual_seed``
        has no effect on reproducibility here; pass a seeded
        :class:`numpy.random.Generator` to control randomness.
    """

    _batch_size: int
    _chunk_size: int
    _shuffle: bool
    _preload_nchunks: int
    _mask: slice
    _drop_last: bool
    _rng: np.random.Generator

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        mask: slice | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        if mask is None:
            mask = slice(0, None)
        if mask.step is not None and mask.step != 1:
            raise ValueError(f"mask.step must be 1, but got {mask.step}")
        start, stop = mask.start or 0, mask.stop
        if start < 0:
            raise ValueError("mask.start must be >= 0")
        if stop is not None and start >= stop:
            raise ValueError("mask.start must be < mask.stop when mask.stop is specified")

        check_lt_1([chunk_size, preload_nchunks], ["Chunk size", "Preloaded chunks"])
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
        self._rng = rng or np.random.default_rng()
        self._batch_size, self._chunk_size, self._shuffle = batch_size, chunk_size, shuffle
        self._preload_nchunks, self._mask, self._drop_last = (
            preload_nchunks,
            slice(start, stop),
            drop_last,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    def n_iters(self, n_obs: int) -> int:
        start, stop = self._mask.start or 0, self._mask.stop or n_obs
        total_obs = stop - start
        return total_obs // self._batch_size if self._drop_last else math.ceil(total_obs / self._batch_size)

    def validate(self, n_obs: int) -> None:
        """Validate the sampler configuration against the loader's n_obs.

        Parameters
        ----------
        n_obs
            The total number of observations in the loader.

        Raises
        ------
        ValueError
            If the sampler configuration is invalid for the given n_obs.
        """
        start, stop = self._mask.start or 0, self._mask.stop or n_obs
        if stop > n_obs:
            raise ValueError(
                f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
                "The sampler range must be within the loader's observations."
            )
        if start >= stop:
            raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        # Worker mode validation - only check when there are multiple workers

        if worker_info is not None and worker_info.num_workers > 1 and not self._drop_last and self._batch_size != 1:
            # With batch_size=1, every batch is exactly 1 item, so no partial batches exist
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")

        start, stop = self._mask.start or 0, self._mask.stop or n_obs
        # Compute chunks directly from resolved mask range
        # Create chunk indices for possible shuffling and worker sharding
        chunk_indices = np.arange(math.ceil((stop - start) / self._chunk_size))
        if self._shuffle:
            # Use sampler's RNG for chunk ordering - same across all workers
            self._rng.shuffle(chunk_indices)
        chunks = self._compute_chunks(chunk_indices, start, stop)
        if worker_info is not None:
            chunks = np.array_split(chunks, worker_info.num_workers)[worker_info.id]
        # Set up the iterator for chunks and the batch indices for splits
        in_memory_size = self._chunk_size * self._preload_nchunks
        chunks_per_request = split_given_size(chunks, self._preload_nchunks)
        batch_indices = np.arange(in_memory_size)
        split_batch_indices = split_given_size(batch_indices, self._batch_size)
        batch_rng = _spawn_worker_rng(self._rng, worker_info.id) if worker_info else self._rng
        for request_chunks in chunks_per_request[:-1]:
            if self._shuffle:
                # Avoid copies using in-place shuffling since `self._shuffle` should not change mid-training
                batch_rng.shuffle(batch_indices)
                split_batch_indices = split_given_size(batch_indices, self._batch_size)
            yield {"chunks": request_chunks, "splits": split_batch_indices}
        # On the last yield, drop the last uneven batch and create new batch_indices since the in-memory size of this last yield could be divisible by batch_size but smaller than preload_nslices * slice_size
        final_chunks = chunks_per_request[-1]
        total_obs_in_last_batch = int(sum(s.stop - s.start for s in final_chunks))
        if total_obs_in_last_batch == 0:  # pragma: no cover
            raise RuntimeError("Last batch was found to have no observations. Please open an issue.")
        if self._drop_last:
            if total_obs_in_last_batch < self._batch_size:
                return
            total_obs_in_last_batch -= total_obs_in_last_batch % self._batch_size
        indices = (
            batch_rng.permutation(total_obs_in_last_batch) if self._shuffle else np.arange(total_obs_in_last_batch)
        )
        batch_indices = split_given_size(indices, self._batch_size)
        yield {"chunks": final_chunks, "splits": batch_indices}

    def _compute_chunks(self, chunk_indices: np.ndarray, start: int, stop: int) -> list[slice]:
        """Compute chunks from start and stop indices.

        This function is used to compute the chunks for the data to load.
        The chunks are computed such that the last chunk is the incomplete chunk if the total number of observations is not divisible by the chunk size.
        Supposed to also work with shuffled chunk indices so that the last chunk computed isn't always the incomplete chunk.
        """
        n_chunks, pivot_index = len(chunk_indices), chunk_indices[-1]
        offsets = np.ones(n_chunks + 1, dtype=int) * self._chunk_size
        offsets[0] = start
        offsets[pivot_index + 1] = incomplete if (incomplete := (stop - start) % self._chunk_size) else self._chunk_size
        offsets = np.cumsum(offsets)
        starts, stops = offsets[:-1][chunk_indices], offsets[1:][chunk_indices]
        return [slice(int(s), int(e)) for s, e in zip(starts, stops, strict=True)]


def _get_dist_info_torch() -> tuple[int, int]:
    """Get rank and world_size from ``torch.distributed``."""
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Initialize it before creating a ChunkSamplerDistributed with backend='torch'."
        )
    return dist.get_rank(), dist.get_world_size()


def _get_dist_info_jax() -> tuple[int, int]:
    """Get rank and world_size from JAX multi-process API."""
    import jax

    if not jax.distributed.is_initialized():
        raise RuntimeError(
            "JAX distributed is not initialized. "
            "Call jax.distributed.initialize() before creating a ChunkSamplerDistributed with backend='jax'."
        )
    return jax.process_index(), jax.process_count()


DISTRIBUTED_BACKENDS: dict[str, callable] = {
    "torch": _get_dist_info_torch,
    "jax": _get_dist_info_jax,
}


class ChunkSamplerDistributed(ChunkSampler):
    """Distributed chunk-based sampler that shards data across distributed processes.

    Partitions the full observation range into ``world_size`` contiguous shards
    using the ``mask`` mechanism of :class:`ChunkSampler`.  Each rank receives a
    non-overlapping slice of the data.  The shard boundaries are computed lazily
    when ``n_obs`` becomes known.

    When ``enforce_equal_batches`` is *True* (the default), the per-rank observation
    count is rounded down to the nearest multiple of ``batch_size``,
    guaranteeing that every rank yields exactly the same number of complete
    batches.

    Rank and world size are obtained from the distributed framework specified by
    ``backend`` at construction time, so the framework must be initialized
    before creating an instance of this sampler.

    Parameters
    ----------
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch.
    backend
        Distributed backend to query for rank and world size.
        Supported values: ``"torch"`` (uses :mod:`torch.distributed`) and
        ``"jax"`` (uses :func:`jax.process_index` / :func:`jax.process_count`).
    shuffle
        Whether to shuffle chunk and index order.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator for shuffling.
    enforce_equal_batches
        If *True*, round each rank's observation count down to a multiple of
        ``batch_size`` so that all ranks yield the same number of batches.
        Set to *False* to use the raw ``n_obs // world_size`` split, which may
        result in an uneven number of batches per worker.
    """

    _rank: int
    _world_size: int
    _enforce_equal_batches: bool

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        backend: Literal["torch", "jax"],
        shuffle: bool = False,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
        enforce_equal_batches: bool = True,
    ):
        if backend not in DISTRIBUTED_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Supported backends: {sorted(DISTRIBUTED_BACKENDS)}")

        self._rank, self._world_size = DISTRIBUTED_BACKENDS[backend]()
        self._enforce_equal_batches = enforce_equal_batches

        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            rng=rng,
        )

    def _shard_mask(self, n_obs: int) -> slice:
        """Return the contiguous observation slice for this rank."""
        per_rank = n_obs // self._world_size
        if self._enforce_equal_batches:
            per_rank = per_rank // self._batch_size * self._batch_size
        rank_start = self._rank * per_rank
        rank_stop = rank_start + per_rank
        return slice(rank_start, rank_stop)

    def n_iters(self, n_obs: int) -> int:
        self._mask = self._shard_mask(n_obs)
        return super().n_iters(n_obs)

    def validate(self, n_obs: int) -> None:
        self._mask = self._shard_mask(n_obs)
        super().validate(n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        self._mask = self._shard_mask(n_obs)
        yield from super()._sample(n_obs)
