"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng, check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from annbatch.samplers._utils import WorkerInfo
    from annbatch.types import LoadRequest


class MaskableSampler(Sampler):
    """A sampler whose observation range can be restricted via a mask.

    Subclass this to create chunk-based samplers that can be wrapped
    by :class:`ChunkSamplerDistributed`.
    """

    _mask: slice
    _rng: np.random.Generator

    @property
    def mask(self) -> slice:
        """The observation range this sampler operates on."""
        return self._mask

    @mask.setter
    def mask(self, value: slice) -> None:
        self._mask = value

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator used by this sampler."""
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._rng = value


class ChunkSamplerBase(MaskableSampler):
    """Private base class for chunk-based sampling.

    Handles both epoch-based and with-replacement sampling modes.
    When ``n_iters`` is provided, operates in replacement mode with
    shuffle forced on and drop_last forced off.
    """

    _batch_size: int
    _chunk_size: int
    _shuffle: bool
    _preload_nchunks: int
    _drop_last: bool
    _in_memory_size: int
    _n_iters: int | None

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        n_iters: int | None = None,
        mask: slice | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        if n_iters is not None:
            check_lt_1([n_iters], ["n_iters"])
            shuffle = True
            drop_last = False

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
        self._n_iters = n_iters
        self._in_memory_size = chunk_size * preload_nchunks
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
        if self._n_iters is not None:
            return self._n_iters
        start, stop = self._resolve_start_stop(n_obs)
        total_obs = stop - start
        return total_obs // self.batch_size if self._drop_last else math.ceil(total_obs / self.batch_size)

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
        start, stop = self._resolve_start_stop(n_obs)
        if stop > n_obs:
            raise ValueError(
                f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
                "The sampler range must be within the loader's observations."
            )
        if start >= stop:
            raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")
        if self._n_iters is not None and stop - start < self._chunk_size:
            raise ValueError(
                f"Observation range ({stop - start}) is smaller than chunk_size ({self._chunk_size}). "
                "Reduce chunk_size or expand the mask range."
            )

    def _validate_worker_mode(self, worker_info: WorkerInfo | None) -> None:
        if self._n_iters is not None:
            if worker_info is not None and worker_info.num_workers > 1:
                raise ValueError("Multiple workers are not supported with this sampler.")
        elif worker_info is not None and worker_info.num_workers > 1 and not self._drop_last and self.batch_size != 1:
            # With batch_size=1, every batch is exactly 1 item, so no partial batches exist.
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        self._validate_worker_mode(worker_info)

        start, stop = self._resolve_start_stop(n_obs)
        worker_aware_rng = self._rng if worker_info is None else _spawn_worker_rng(self._rng, worker_info.id)

        chunks = self._compute_chunks(start, stop, rng=self._rng)
        yield from self._iter_from_chunks(chunks, batch_rng=worker_aware_rng, worker_info=worker_info)

    def _iter_from_chunks(
        self,
        chunks: list[slice],
        batch_rng: np.random.Generator,
        worker_info: WorkerInfo | None,
    ) -> Iterator[LoadRequest]:
        base = self._iter_from_chunks_base(chunks, batch_rng, worker_info)
        if self._n_iters is None:
            yield from base
            return
        batches_per_request = self._in_memory_size // self.batch_size
        n_full, tail = divmod(self._n_iters, batches_per_request)
        yield from itertools.islice(base, n_full)
        if tail > 0:
            load_request = next(base)
            yield {
                "chunks": load_request["chunks"],
                "splits": load_request["splits"][:tail],
            }

    def _iter_from_chunks_base(
        self,
        chunks: list[slice],
        batch_rng: np.random.Generator,
        worker_info: WorkerInfo | None,
    ) -> Iterator[LoadRequest]:
        # Worker sharding: each worker gets a disjoint subset of chunks
        if worker_info is not None:
            chunks = np.array_split(chunks, worker_info.num_workers)[worker_info.id]
        # Set up the iterator for chunks and the batch indices for splits
        chunks_per_request = split_given_size(chunks, self._preload_nchunks)
        batch_indices = np.arange(self._in_memory_size)
        split_batch_indices = split_given_size(batch_indices, self.batch_size)
        for request_chunks in chunks_per_request[:-1]:
            if self.shuffle:
                # Avoid copies using in-place shuffling since `self.shuffle` should not change mid-training
                batch_rng.shuffle(batch_indices)
                split_batch_indices = split_given_size(batch_indices, self.batch_size)
            yield {"chunks": request_chunks, "splits": split_batch_indices}
        # On the last yield, drop the last uneven batch and create new batch_indices since the in-memory size of this last yield could be divisible by batch_size but smaller than preload_nslices * slice_size
        final_chunks = chunks_per_request[-1]
        total_obs_in_last_batch = int(sum(s.stop - s.start for s in final_chunks))
        if total_obs_in_last_batch == 0:  # pragma: no cover
            raise RuntimeError("Last batch was found to have no observations. Please open an issue.")
        if self._drop_last:
            if total_obs_in_last_batch < self.batch_size:
                return
            total_obs_in_last_batch -= total_obs_in_last_batch % self.batch_size
        indices = batch_rng.permutation(total_obs_in_last_batch) if self.shuffle else np.arange(total_obs_in_last_batch)
        batch_indices = split_given_size(indices, self.batch_size)
        yield {"chunks": final_chunks, "splits": batch_indices}

    def _compute_chunks(self, start: int, stop: int, rng: np.random.Generator) -> list[slice]:
        """Compute chunks from start and stop indices.

        This function is used to compute the chunks for the data to load.
        The chunks are computed such that the last chunk is the incomplete chunk if the total number of observations is not divisible by the chunk size.
        Supposed to also work with shuffled chunk indices so that the last chunk computed isn't always the incomplete chunk.
        """
        if self._n_iters is not None:
            # stop - start >= chunk_size is guaranteed by validate()
            start_indices = rng.integers(
                start, stop - self._chunk_size + 1, size=math.ceil((self._n_iters * self.batch_size) / self._chunk_size)
            )
            return [slice(int(s), int(s + self._chunk_size)) for s in start_indices]
        # Compute chunks directly from resolved mask range
        # Create chunk indices for possible shuffling and worker sharding
        chunk_indices = np.arange(math.ceil((stop - start) / self._chunk_size))
        if self.shuffle:
            rng.shuffle(chunk_indices)
        n_chunks, pivot_index = len(chunk_indices), chunk_indices[-1]
        offsets = np.ones(n_chunks + 1, dtype=int) * self._chunk_size
        offsets[0] = start
        offsets[pivot_index + 1] = incomplete if (incomplete := (stop - start) % self._chunk_size) else self._chunk_size
        offsets = np.cumsum(offsets)
        starts, stops = offsets[:-1][chunk_indices], offsets[1:][chunk_indices]
        return [slice(int(s), int(e)) for s, e in zip(starts, stops, strict=True)]

    def _resolve_start_stop(self, n_obs: int) -> tuple[int, int]:
        return self._mask.start or 0, self._mask.stop or n_obs


class ChunkSampler(ChunkSamplerBase):
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
        Random number generator for shuffling. Note that :func:`torch.manual_seed`
        has no effect on reproducibility here; pass a seeded
        :class:`numpy.random.Generator` to control randomness.
    """

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
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            mask=mask,
            shuffle=shuffle,
            drop_last=drop_last,
            rng=rng,
        )


class ChunkSamplerWithReplacement(ChunkSamplerBase):
    """Chunk-based sampler that draws chunks with replacement.

    Unlike :class:`ChunkSampler`, this sampler draws random contiguous
    chunks from the observation range with replacement and is not limited
    to a single epoch. The number of batches to yield (``n_iters``) is required.

    Shuffle is always enabled and drop_last is always disabled,
    since the number of yielded batches is controlled exactly
    by ``n_iters``.

    See :class:`ChunkSampler` for the shared parameters.

    Parameters
    ----------
    n_iters
        Number of batches to yield.
    """

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        n_iters: int,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            n_iters=n_iters,
            mask=mask,
            rng=rng,
        )


def _get_dist_info_torch() -> tuple[int, int]:
    """Get rank and world_size from ``torch.distributed``."""
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Initialize it before creating a ChunkSamplerDistributed with dist_info='torch'."
        )
    return dist.get_rank(), dist.get_world_size()


def _get_dist_info_jax() -> tuple[int, int]:
    """Get rank and world_size from JAX multi-process API."""
    import jax

    if not jax.distributed.is_initialized():
        raise RuntimeError(
            "JAX distributed is not initialized. "
            "Call jax.distributed.initialize() before creating a ChunkSamplerDistributed with dist_info='jax'."
        )
    return jax.process_index(), jax.process_count()


DISTRIBUTED_BACKENDS: dict[str, Callable[[], tuple[int, int]]] = {
    "torch": _get_dist_info_torch,
    "jax": _get_dist_info_jax,
}


class ChunkSamplerDistributed(Sampler):
    """Distributed chunk-based sampler that shards data across distributed processes.

    Wraps any chunk-based sampler (e.g. :class:`ChunkSampler` or
    :class:`ChunkSamplerWithReplacement`) and partitions the observation
    range across ``world_size`` processes.  Each rank receives a
    non-overlapping slice of the data.

    When ``enforce_equal_batches`` is *True* (the default), the per-rank observation
    count is rounded down to the nearest multiple of ``batch_size``,
    guaranteeing that every rank yields exactly the same number of complete
    batches.

    Rank and world size are obtained from ``dist_info`` at construction time.
    The corresponding distributed framework must already be initialized.

    Parameters
    ----------
    sampler
        The base chunk sampler to distribute.
    dist_info
        How to obtain rank and world size.
        Either a string naming a distributed backend (``"torch"`` or ``"jax"``),
        or a callable that returns ``(rank, world_size)``.
    enforce_equal_batches
        If *True*, round each rank's observation count down to a multiple of ``batch_size`` so that all workers (ranks) yield the same number of batches.
        Set to *False* to use the raw ``n_obs // world_size`` split, which may result in an uneven number of batches per worker.
    """

    _rank: int
    _world_size: int
    _enforce_equal_batches: bool
    _sampler: MaskableSampler

    def __init__(
        self,
        sampler: MaskableSampler,
        *,
        dist_info: Literal["torch", "jax"] | Callable[[], tuple[int, int]],
        enforce_equal_batches: bool = True,
    ):
        if callable(dist_info):
            self._rank, self._world_size = dist_info()
        elif dist_info in DISTRIBUTED_BACKENDS:
            self._rank, self._world_size = DISTRIBUTED_BACKENDS[dist_info]()
        else:
            raise ValueError(f"Unknown dist_info {dist_info!r}. Supported backends: {sorted(DISTRIBUTED_BACKENDS)}")
        self._enforce_equal_batches = enforce_equal_batches
        self._sampler = sampler
        sampler.rng = _spawn_worker_rng(sampler.rng, self._rank)

    @property
    def batch_size(self) -> int:
        return self._sampler.batch_size

    @property
    def shuffle(self) -> bool:
        return self._sampler.shuffle

    def _shard_mask(self, n_obs: int) -> slice:
        """Return the contiguous observation slice for this rank."""
        per_rank = n_obs // self._world_size
        if self._enforce_equal_batches:
            per_rank = per_rank // self._sampler.batch_size * self._sampler.batch_size
        rank_start = self._rank * per_rank
        rank_stop = rank_start + per_rank
        return slice(rank_start, rank_stop)

    def n_iters(self, n_obs: int) -> int:
        self._sampler.mask = self._shard_mask(n_obs)
        return self._sampler.n_iters(n_obs)

    def validate(self, n_obs: int) -> None:
        self._sampler.mask = self._shard_mask(n_obs)
        self._sampler.validate(n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        self._sampler.mask = self._shard_mask(n_obs)
        yield from self._sampler._sample(n_obs)
