"""Sampler classes for efficient chunk-aligned data access from Zarr stores."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from annbatch.utils import check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.utils import WorkerHandle


T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class LoadRequest[T_co]:
    """Load request from sampler."""

    # below the explanations are for when T_co = list[slice]
    # slices to load
    # a list of at most chunk_size ranged slices
    slices: T_co
    # how the concatenation of slices should be split into batches
    # a list of n_splits with batch_size arrays
    splits: list[np.ndarray]
    # indices that are not used in the last batch < batch_size
    # or None if there is none ( usually when drop_last is True)
    leftover_split: np.ndarray | None


class Sampler[T_co](ABC):
    """Base sampler class."""

    @abstractmethod
    def __iter__(self) -> Iterator[LoadRequest[T_co]]:
        """Iterator over load requests."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of iterations to exhaust the sampler."""

    def set_worker_handle(self, worker_handle: WorkerHandle) -> None:
        """Set the worker handle if desired. If the sampler doesn't support workers, this is a no-op."""
        # this is a separate method because we'd want the LoaderBuilder itself
        # passing this to the sampler, this way Loader doesn't need to know about
        # the sampler's worker handle and knows every Sampler supports this
        # but we don't want to force every sampler to implement this
        return None

    def supports_workers(self) -> bool:
        """Return whether the sampler supports workers."""
        return False


class SliceSampler(Sampler[list[slice]]):
    """Chunk-based slice sampler for batched data access.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk i.e. the range of each slice.
    start_index
        Starting observation index (inclusive).
    end_index
        Ending observation index (exclusive).
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
    _start_index: int
    _end_index: int
    _n_chunks: int
    _n_iters: int
    _drop_last: bool
    _rng: np.random.Generator

    def __init__(
        self,
        *,
        batch_size: int,
        chunk_size: int,
        start_index: int,
        end_index: int,
        shuffle: bool = False,
        preload_nchunks: int,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        if start_index < 0 or start_index >= end_index:
            raise ValueError("start_index must be >= 0 and < end_index")
        check_lt_1([chunk_size, preload_nchunks], ["Chunk size", "Preload chunks"])
        preload_size = chunk_size * preload_nchunks

        # TODO: asses this: Should we make this check mandatory in the base Sampler class?
        # Because this is actually a requirement for any sampler to run in a Loader class
        if batch_size > preload_size:
            raise ValueError(
                "batch_size cannot exceed chunk_size * preload_nchunks. "
                f"Got batch_size={batch_size}, but max is {preload_size}."
            )

        n_batches = (
            math.floor((end_index - start_index) / batch_size)
            if drop_last
            else math.ceil((end_index - start_index) / batch_size)
        )
        total_yielded_obs = n_batches * batch_size

        self._rng = np.random.default_rng() if rng is None else rng
        self._n_iters = math.ceil(total_yielded_obs / (chunk_size * preload_nchunks))
        self._n_chunks = math.ceil((end_index - start_index) / chunk_size)
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._preload_nchunks = preload_nchunks
        self._start_index = start_index
        self._end_index = end_index
        self._drop_last = drop_last
        self._worker_handle = None

    def __len__(self) -> int:
        return self._n_iters

    def __iter__(self) -> Iterator[LoadRequest[list[slice]]]:
        # Compute slices directly from index range
        slices = self._compute_slices()
        n_slices = len(slices)

        # Create slice indices for shuffling
        slice_indices = np.arange(n_slices)
        if self._shuffle:
            slice_indices = self._shuffle_integers(slice_indices)

        # Worker sharding: each worker gets a disjoint subset of slices
        if self._worker_handle is not None:
            slice_indices = self._worker_handle.get_part_for_worker(slice_indices)

        n_slices_for_worker = len(slice_indices)
        n_slice_iters = math.ceil(n_slices_for_worker / self._preload_nchunks) if n_slices_for_worker > 0 else 0

        n_leftover_loaded_indices = 0
        leftover_split = None

        for i in range(n_slice_iters):
            start = i * self._preload_nchunks
            end = min(start + self._preload_nchunks, n_slices_for_worker)
            indices_to_load = slice_indices[start:end]

            # Compute total observations to load from selected slices
            total_obs_to_load = sum(slices[idx].stop - slices[idx].start for idx in indices_to_load)

            # Generate loaded indices with leftover from previous iteration
            loaded_indices = np.arange(total_obs_to_load + n_leftover_loaded_indices)
            if self._shuffle:
                loaded_indices = self._shuffle_integers(loaded_indices)
            splits = np.split(loaded_indices, np.arange(self._batch_size, len(loaded_indices), self._batch_size))

            # if the last batch is full, there is no leftover data
            if splits[-1].shape[0] == self._batch_size:
                n_leftover_loaded_indices = 0
                leftover_split = None
            # handle leftover data
            else:
                is_last_iter = i == n_slice_iters - 1

                if is_last_iter and not self._drop_last:
                    # Yield the final partial batch
                    n_leftover_loaded_indices = 0
                    leftover_split = None
                else:
                    # Either save for next iteration, or drop on last iter
                    n_leftover_loaded_indices = splits[-1].shape[0]
                    leftover_split = None if (self._drop_last and is_last_iter) else splits[-1]
                    splits = splits[:-1]

            yield LoadRequest(
                slices=[slices[idx] for idx in indices_to_load],
                splits=splits,
                leftover_split=leftover_split,
            )

    def set_worker_handle(self, worker_handle: WorkerHandle) -> None:
        # Worker mode validation
        if not self._drop_last and self._preload_nchunks * self._chunk_size % self._batch_size != 0:
            raise ValueError(
                f"When using DataLoader workers with drop_last=False, "
                f"(chunk_size * preload_nchunks) must be divisible by batch_size. "
                f"Got {self._preload_nchunks * self._chunk_size} % {self._batch_size} = {self._preload_nchunks * self._chunk_size % self._batch_size}. "
                f"Set drop_last=True to allow non-divisible configs."
            )
        if self._drop_last:
            import warnings

            warnings.warn(
                "With drop_last=True and multiple workers, up to "
                "(batch_size - 1) * num_workers observations may be dropped "
                "(one partial batch per worker).",
                UserWarning,
                stacklevel=2,
            )
        # TODO: asses this: should we raise an error if worker handle is already set?
        self._worker_handle = worker_handle

    def supports_workers(self) -> bool:
        return True

    def _shuffle_integers(self, integers: np.ndarray) -> np.ndarray:
        if self._worker_handle is None:
            # TODO: deal with generators later
            self._rng.shuffle(integers)
        else:
            # should we always have a worker handle? even if its no-op?
            self._worker_handle.shuffle(integers)
        return integers

    def _compute_slices(self) -> list[slice]:
        """Compute slices directly from start/end indices."""
        starts = list(range(self._start_index, self._end_index, self._chunk_size))
        stops = starts[1:] + [self._end_index]
        return [slice(start, stop) for start, stop in zip(starts, stops, strict=False)]
