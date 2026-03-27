"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

import itertools
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng, check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.samplers._utils import WorkerInfo
    from annbatch.types import LoadRequest


class _ChunkSampler(Sampler):
    """Chunk-based sampler implementation for batched data access."""

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
        num_samples = self._resolve_num_samples(n_obs)
        mask_size = self._resolve_mask_size(n_obs)
        if not self._replacement and num_samples > mask_size:
            raise ValueError(f"num_samples ({num_samples}) cannot exceed the observation range ({mask_size}).")
        if self._replacement and mask_size < self._chunk_size and num_samples > mask_size:
            raise ValueError(
                f"Observation range ({mask_size}) is smaller than chunk_size ({self._chunk_size}) "
                f"with num_samples ({num_samples}) exceeding that range. "
                "Reduce chunk_size, expand the mask range, or set num_samples <= observation range."
            )

    def _validate_worker_mode(self, worker_info: WorkerInfo | None) -> None:
        if worker_info is None or worker_info.num_workers <= 1:
            return
        if not self._shuffle:
            raise ValueError("Multiple workers are not supported with non-shuffled sampling.")
        if self._replacement:
            raise NotImplementedError(
                "Multiple workers are not supported with replacement sampling. See https://github.com/scverse/annbatch/issues/173"
            )
        if not self._drop_last and self.batch_size > 1:
            # With batch_size=1, every batch is exactly 1 item, so no partial batches exist,
            # that's why we don't raise for that case.
            raise ValueError(
                "drop_last=False is not supported when using DataLoader with multiple workers and batch_size>1."
            )

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
        n_full, tail = divmod(self.n_iters(n_obs), self._in_memory_size // self.batch_size)
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
            return self._compute_chunks_with_replacement(start, stop, n_obs, rng)
        return self._compute_chunks_without_replacement(start, stop, rng)

    def _compute_chunks_with_replacement(
        self, start: int, stop: int, n_obs: int, rng: np.random.Generator
    ) -> list[slice]:
        """Draw random chunk positions with replacement."""
        num_samples = self._resolve_num_samples(n_obs)
        n_chunks, remainder = divmod(num_samples, self._chunk_size)
        start_indices = rng.integers(start, stop - self._chunk_size + 1, size=n_chunks)
        res = [slice(int(s), int(s + self._chunk_size)) for s in start_indices]
        if remainder > 0 and not self._drop_last:
            start_index = rng.integers(start, stop - remainder + 1)
            res.append(slice(start_index, start_index + remainder))
        return res

    def _compute_chunks_without_replacement(self, start: int, stop: int, rng: np.random.Generator) -> list[slice]:
        """Compute chunks covering the full range exactly once.

        The incomplete chunk (chunk that is less than chunk_size) is always placed last in iteration order regardless
        of shuffling -- ensuring no observation is duplicated.
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
        return [slice(int(s), int(e)) for s, e in zip(starts, stops, strict=True)]


class ChunkSampler(_ChunkSampler):
    """Chunk-based sampler for batched data access.

    .. deprecated:: 0.1.0
        Use :class:`~annbatch.samplers.RandomSampler` (for shuffled access) or
        :class:`~annbatch.samplers.SequentialSampler` (for ordered access) instead.

    This is the monolithic sampler that powers both :class:`~annbatch.samplers.RandomSampler` and
    :class:`~annbatch.samplers.SequentialSampler`. It supports with-replacement
    (shuffle/no-shuffle) and without-replacement sampling.

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
    replacement
        If ``True``, draw random chunks with replacement, allowing the
        same observations to appear more than once.
    num_samples
        Total number of observations to draw. When ``None`` (the
        default), equals the effective observation range. Must be
        positive when set and less than the number of observations to be
        yielded when ``replacement=False``.
    """

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
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            replacement=replacement,
            num_samples=num_samples,
            shuffle=shuffle,
            drop_last=drop_last,
            mask=mask,
            rng=rng,
        )
