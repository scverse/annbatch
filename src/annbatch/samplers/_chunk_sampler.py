"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

import math
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.utils import check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest
    from annbatch.utils import WorkerHandle


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
        Random number generator for shuffling.
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

    def _get_worker_handle(self) -> WorkerHandle | None:
        worker_handle = None
        if find_spec("torch"):
            from torch.utils.data import get_worker_info

            from annbatch.utils import WorkerHandle

            if get_worker_info() is not None:
                worker_handle = WorkerHandle()
        # Worker mode validation - only check when there are multiple workers
        # With batch_size=1, every batch is exactly 1 item, so no partial batches exist
        if (
            worker_handle is not None
            and worker_handle.num_workers > 1
            and not self._drop_last
            and self._batch_size != 1
        ):
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")
        return worker_handle

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_handle = self._get_worker_handle()
        start, stop = self._mask.start or 0, self._mask.stop or n_obs
        # Compute chunks directly from resolved mask range
        # Create chunk indices for possible shuffling and worker sharding
        chunk_indices = np.arange(math.ceil((stop - start) / self._chunk_size))
        if self._shuffle:
            if worker_handle is None:
                self._rng.shuffle(chunk_indices)
            else:
                worker_handle.shuffle(chunk_indices)
        chunks = self._compute_chunks(chunk_indices, start, stop)
        # Worker sharding: each worker gets a disjoint subset of chunks
        if worker_handle is not None:
            chunks = worker_handle.get_part_for_worker(chunks)
        # Set up the iterator for chunks and the batch indices for splits
        in_memory_size = self._chunk_size * self._preload_nchunks
        chunks_per_batch = split_given_size(chunks, self._preload_nchunks)
        batch_indices = np.arange(in_memory_size)
        split_batch_indices = split_given_size(batch_indices, self._batch_size)
        for batch_chunks in chunks_per_batch[:-1]:
            if self._shuffle:
                # Avoid copies using in-place shuffling since `self._shuffle` should not change mid-training
                np.random.default_rng().shuffle(batch_indices)
                split_batch_indices = split_given_size(batch_indices, self._batch_size)
            yield {"chunks": batch_chunks, "splits": split_batch_indices}
        # On the last yield, drop the last uneven batch and create new batch_indices since the in-memory size of this last yield could be divisible by batch_size but smaller than preload_nslices * slice_size
        final_chunks = chunks_per_batch[-1]
        total_obs_in_last_batch = int(sum(s.stop - s.start for s in final_chunks))
        if self._drop_last:
            total_obs_in_last_batch -= total_obs_in_last_batch % self._batch_size
        batch_indices = split_given_size(
            (np.random.default_rng().permutation if self._shuffle else np.arange)(total_obs_in_last_batch),
            self._batch_size,
        )
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
