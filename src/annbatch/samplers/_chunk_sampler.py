"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng, check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.samplers._utils import WorkerInfo
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
        Must be ``False`` when ``n_iters`` is set.
    n_iters
        If set, enables with-replacement sampling for exactly this many
        batches instead of epoch-based iteration.
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
    _n_iters: int | None
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
        n_iters: int | None = None,
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
        if n_iters is not None:
            check_lt_1([n_iters], ["n_iters"])
            if drop_last:
                raise ValueError("drop_last must be False when n_iters is set.")
        self._rng = rng or np.random.default_rng()
        self._batch_size, self._chunk_size, self._shuffle = batch_size, chunk_size, shuffle
        self._preload_nchunks, self._mask, self._drop_last = (
            preload_nchunks,
            slice(start, stop),
            drop_last,
        )
        self._n_iters = n_iters
        self._in_memory_size = self._chunk_size * self._preload_nchunks

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
        start, stop = self._resolve_start_stop(n_obs)
        if stop > n_obs:
            raise ValueError(
                f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
                "The sampler range must be within the loader's observations."
            )
        if start >= stop:
            raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")
        if self._n_iters is not None and (stop - start) < self._chunk_size:
            raise ValueError(
                f"With-replacement mode requires at least one full chunk: "
                f"(stop - start) = {stop - start} < chunk_size = {self._chunk_size}."
            )

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        # Worker mode validation - only check when there are multiple workers
        # With batch_size=1, every batch is exactly 1 item, so no partial batches exist
        if (
            worker_info is not None
            and worker_info.num_workers > 1
            and self._n_iters is None
            and not self._drop_last
            and self._batch_size != 1
        ):
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")

        start, stop = self._resolve_start_stop(n_obs)
        # Compute chunks directly from resolved mask range
        # Create chunk indices for possible shuffling and worker sharding
        chunk_indices = np.arange(math.ceil((stop - start) / self._chunk_size))
        if self._shuffle:
            # Use sampler's RNG for chunk ordering - same across all workers
            self._rng.shuffle(chunk_indices)
        chunks = self._compute_chunks(chunk_indices, start, stop)
        worker_aware_rng = self._rng if worker_info is None else _spawn_worker_rng(self._rng, worker_info.id)

        if self._n_iters is not None:
            yield from self._iter_with_replacement(chunks, stop, rng=worker_aware_rng, worker_info=worker_info)
        else:
            yield from self._iter_epoch(chunks, batch_rng=worker_aware_rng, worker_info=worker_info)

    def _iter_with_replacement(
        self, chunk_pool: list[slice], stop: int, rng: np.random.Generator, worker_info: WorkerInfo | None
    ) -> Iterator[LoadRequest]:
        # Fix up incomplete last chunk with overlapping full-size chunk
        # so that all obs are covered if n_iters is set large enough.
        last = chunk_pool[-1]
        if last.stop - last.start < self._chunk_size:
            new_stop = min(last.start + self._chunk_size, stop)
            new_start = new_stop - self._chunk_size
            chunk_pool[-1] = slice(new_start, new_stop)

        n_pool = len(chunk_pool)

        batches_per_request = self._in_memory_size // self._batch_size

        n_iters = self._n_iters
        # Worker sharding: each worker gets different number of iterations
        # but the chunks might overlap within workers.
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            base, remainder = divmod(n_iters, num_workers)
            n_iters = base + (1 if worker_id < remainder else 0)

        n_requests = math.ceil(n_iters / batches_per_request)
        last_n_batches = n_iters - (n_requests - 1) * batches_per_request

        batch_indices = np.arange(self._in_memory_size)

        for request_idx in range(n_requests):
            # Sample chunk indices with replacement from the pool
            # here we use worker_rng because each worker should be independent unlike other case
            sampled = rng.integers(0, n_pool, size=self._preload_nchunks)
            chunks = [chunk_pool[i] for i in sampled]

            # Shuffle in-memory indices (in-place, reuse array)
            if self._shuffle:
                rng.shuffle(batch_indices)
            splits = split_given_size(batch_indices, self._batch_size)
            if request_idx == n_requests - 1:
                splits = splits[:last_n_batches]

            yield {"chunks": chunks, "splits": splits}

    def _iter_epoch(
        self, chunks: list[slice], batch_rng: np.random.Generator, worker_info: WorkerInfo | None
    ) -> Iterator[LoadRequest]:
        # Worker sharding: each worker gets a disjoint subset of chunks
        if worker_info is not None:
            chunks = np.array_split(np.array(chunks), worker_info.num_workers)[worker_info.id]
        # Set up the iterator for chunks and the batch indices for splits
        chunks_per_request = split_given_size(chunks, self._preload_nchunks)
        batch_indices = np.arange(self._in_memory_size)
        split_batch_indices = split_given_size(batch_indices, self._batch_size)
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

    def _resolve_start_stop(self, n_obs: int) -> tuple[int, int]:
        return self._mask.start or 0, self._mask.stop or n_obs
