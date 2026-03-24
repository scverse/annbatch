"""Chunk-based sampler for efficient data access."""

from __future__ import annotations

import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from annbatch.samplers._utils import get_torch_worker_info
from annbatch.samplers.abc import Sampler
from annbatch.utils import _spawn_worker_rng, check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from annbatch.samplers._utils import WorkerInfo
    from annbatch.types import LoadRequest


class ChunkSampler(Sampler):
    """Chunk-based sampler for batched data access.

    .. deprecated:: 0.1.0
        Use :class:`~annbatch.samplers.RandomSampler` (for shuffled access) or
        :class:`~annbatch.samplers.SequentialSampler` (for ordered access) instead.

    This is the monolithic sampler that powers both :class:`~annbatch.samplers.RandomSampler` and :class:`~annbatch.samplers.SequentialSampler`.
    It supports epoch-based and with-replacement sampling, optional
    shuffling, and all combinations of ``replacement``, ``num_samples``,
    and ``drop_last``.

    """

    _batch_size: int
    _chunk_size: int
    _shuffle: bool
    _preload_nchunks: int
    _drop_last: bool
    _in_memory_size: int
    _replacement: bool
    _num_samples: int | None

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        replacement: bool = False,
        num_samples: int | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        warnings.warn(
            "ChunkSampler is deprecated, use RandomSampler or SequentialSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if num_samples is not None:
            check_lt_1([num_samples], ["num_samples"])

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
        self._replacement = replacement
        self._num_samples = num_samples
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

    def _resolve_num_samples(self, n_obs: int) -> int:
        """Return the effective number of samples to draw."""
        return self._num_samples or self._resolve_mask_size(n_obs)

    def _resolve_start_stop(self, n_obs: int) -> tuple[int, int]:
        return self._mask.start or 0, self._mask.stop or n_obs

    def _resolve_mask_size(self, n_obs: int) -> int:
        s, e = self._resolve_start_stop(n_obs)
        return e - s

    def n_iters(self, n_obs: int) -> int:
        total = self._resolve_num_samples(n_obs)
        return total // self.batch_size if self._drop_last else math.ceil(total / self.batch_size)

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
        if self._replacement and stop - start < self._chunk_size:
            raise ValueError(
                f"Observation range ({stop - start}) is smaller than chunk_size ({self._chunk_size}). "
                "Reduce chunk_size or expand the mask range."
            )

    def _validate_worker_mode(self, worker_info: WorkerInfo | None) -> None:
        if worker_info is None or worker_info.num_workers <= 1:
            return
        if not self._shuffle:
            raise ValueError("Multiple workers are not supported with non-shuffled sampling.")
        if self._replacement:
            raise ValueError("Multiple workers are not supported with replacement sampling.")
        if not self._drop_last and self.batch_size != 1:
            # With batch_size=1, every batch is exactly 1 item, so no partial batches exist.
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        self._validate_worker_mode(worker_info)

        worker_aware_rng = self._rng if worker_info is None else _spawn_worker_rng(self._rng, worker_info.id)
        chunks = self._compute_chunks(n_obs, rng=self._rng)
        yield from self._iter_from_chunks(n_obs, chunks, batch_rng=worker_aware_rng, worker_info=worker_info)

    def _iter_from_chunks(
        self,
        n_obs: int,
        chunks: list[slice],
        batch_rng: np.random.Generator,
        worker_info: WorkerInfo | None,
    ) -> Iterator[LoadRequest]:
        base = self._iter_from_chunks_base(chunks, batch_rng, worker_info)
        if not self._replacement and self._num_samples is None:
            yield from base
            return
        num_samples = self._resolve_num_samples(n_obs)
        n_batches = num_samples // self.batch_size if self._drop_last else math.ceil(num_samples / self.batch_size)
        batches_per_request = self._in_memory_size // self.batch_size
        n_full, tail = divmod(n_batches, batches_per_request)
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

    def _compute_chunks(self, n_obs: int, rng: np.random.Generator) -> list[slice]:
        """Compute chunks from start and stop indices.

        Chunks are computed such that the last chunk may be incomplete.
        """
        start, stop = self._resolve_start_stop(n_obs)
        if self._replacement:
            # stop - start >= chunk_size is guaranteed by validate()
            num_samples = self._resolve_num_samples(stop)
            n_chunks, remainder = divmod(num_samples, self._chunk_size)
            start_indices = rng.integers(start, stop - self._chunk_size + 1, size=n_chunks)
            res = [slice(int(s), int(s + self._chunk_size)) for s in start_indices]
            if remainder > 0 and not self._drop_last:
                start_index = rng.integers(start, stop - remainder + 1)
                res.append(slice(start_index, start_index + remainder))
            return res

        num_samples = self._resolve_num_samples(stop)
        epoch_size = stop - start
        incomplete = epoch_size % self._chunk_size

        if num_samples <= epoch_size:
            return self._compute_epoch_chunks(start, stop, rng)

        # Multi-epoch: all chunks except the very last must be exactly chunk_size,
        # because _iter_from_chunks_base indexes into in_memory_size for non-last
        # preload groups.  Middle epochs drop their incomplete tail; only the final
        # epoch keeps it.  We tile enough epochs to cover num_samples (the truncation
        # to exact n_batches happens in _iter_from_chunks).
        aligned_epoch_obs = epoch_size - incomplete
        # obs per full-chunks-only epoch vs full epoch (with incomplete tail)
        n_epochs = math.ceil(num_samples / aligned_epoch_obs) if incomplete else math.ceil(num_samples / epoch_size)
        all_chunks: list[slice] = []
        for i in range(n_epochs):
            is_last = i == n_epochs - 1
            all_chunks.extend(self._compute_epoch_chunks(start, stop, rng, keep_incomplete=is_last))
        return all_chunks

    def _compute_epoch_chunks(
        self, start: int, stop: int, rng: np.random.Generator, *, keep_incomplete: bool = True
    ) -> list[slice]:
        """Compute one epoch's worth of chunks.

        The incomplete chunk (when ``epoch_size`` is not divisible by
        ``chunk_size``) is always placed last in iteration order regardless
        of shuffling -- ensuring no observation is duplicated within an epoch.

        Parameters
        ----------
        keep_incomplete
            When ``False`` the trailing incomplete chunk is dropped.  Used
            for middle epochs in multi-epoch tiling so that every emitted
            chunk is exactly ``chunk_size``.
        """
        chunk_indices = np.arange(math.ceil((stop - start) / self._chunk_size))
        if self.shuffle:
            rng.shuffle(chunk_indices)
        n_chunks, pivot_index = len(chunk_indices), chunk_indices[-1]
        offsets = np.ones(n_chunks + 1, dtype=int) * self._chunk_size
        offsets[0] = start
        incomplete = (stop - start) % self._chunk_size
        offsets[pivot_index + 1] = incomplete if incomplete else self._chunk_size
        offsets = np.cumsum(offsets)
        starts, stops = offsets[:-1][chunk_indices], offsets[1:][chunk_indices]
        chunks = [slice(int(s), int(e)) for s, e in zip(starts, stops, strict=True)]
        if not keep_incomplete and incomplete:
            chunks.pop()
        return chunks


def _get_dist_info_torch() -> tuple[int, int]:
    """Get rank and world_size from ``torch.distributed``."""
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Initialize it before creating a DistributedRandomSampler with dist_info='torch'."
        )
    return dist.get_rank(), dist.get_world_size()


def _get_dist_info_jax() -> tuple[int, int]:
    """Get rank and world_size from JAX multi-process API."""
    import jax

    if not jax.distributed.is_initialized():
        raise RuntimeError(
            "JAX distributed is not initialized. "
            "Call jax.distributed.initialize() before creating a DistributedRandomSampler with dist_info='jax'."
        )
    return jax.process_index(), jax.process_count()


DISTRIBUTED_BACKENDS: dict[str, Callable[[], tuple[int, int]]] = {
    "torch": _get_dist_info_torch,
    "jax": _get_dist_info_jax,
}


class DistributedRandomSampler(Sampler):
    """Distributed sampler that shards data across distributed processes.

    Wraps any sampler and partitions the observation range across
    ``world_size`` processes.  Each rank receives a non-overlapping
    slice of the data.

    When ``enforce_equal_batches`` is *True* (the default), the per-rank observation
    count is rounded down to the nearest multiple of ``batch_size``,
    guaranteeing that every rank yields exactly the same number of complete
    batches.

    Rank and world size are obtained from ``dist_info`` at construction time.
    The corresponding distributed framework must already be initialized.

    Parameters
    ----------
    sampler
        The base sampler to distribute.
    dist_info
        How to obtain rank and world size.
        Either a string naming a distributed backend (``"torch"`` or
        ``"jax"``), or a callable that returns ``(rank, world_size)``.
    enforce_equal_batches
        If *True*, round each rank's observation count down to a multiple of ``batch_size`` so that all workers (ranks) yield the same number of batches.
        Set to *False* to use the raw ``n_obs // world_size`` split, which may result in an uneven number of batches per worker.
    """

    _rank: int
    _world_size: int
    _enforce_equal_batches: bool
    _sampler: Sampler

    def __init__(
        self,
        sampler: Sampler,
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
