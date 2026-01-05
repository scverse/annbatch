"""Sampler classes for efficient slice-based data access."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from annbatch.utils import check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.utils import WorkerHandle




@dataclass(frozen=True)
class LoadRequest:
    """Load request from sampler."""

    # slices to load
    # a list of at most slice_size ranged slices
    slices: list[slice]
    # how the concatenation of slices should be split into batches
    # a list of splits, last one may be partial (< batch_size)
    # the loader carries over partial batches to the next iteration
    splits: list[np.ndarray]


class Sampler(ABC):
    """Base sampler class.

    Samplers control how data is batched and loaded from the underlying datasets.
    """

    @abstractmethod
    def sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Sample load requests given the total number of observations.

        Parameters
        ----------
        n_obs
            The total number of observations available.

        Yields
        ------
        LoadRequest
            Load requests for batching data.
        """

    @property
    @abstractmethod
    def batch_size(self) -> int | None:
        """The batch size of the sampler if valid."""

    @abstractmethod
    def validate(self, n_obs: int) -> None:
        """Validate the sampler configuration against the loader's state.

        This method is called when the sampler is set on a loader.
        Override this method to add custom validation for sampler parameters.

        Parameters
        ----------
        n_obs
            The total number of observations in the loader.

        Raises
        ------
        ValueError
            If the sampler configuration is invalid for the given n_obs.
        """

    def set_worker_handle(self, worker_handle: WorkerHandle) -> None:
        """Set the worker handle if desired. If the sampler doesn't support workers, this is a no-op."""
        del worker_handle  # to explicitly show that we don't use the worker handle
        return None

    def supports_workers(self) -> bool:
        """Return whether the sampler supports workers."""
        return False


class SliceSampler(Sampler):
    """Slice-based sampler for batched data access.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    slice_size
        Size of each slice i.e. the range of each slice yielded.
    mask
        A slice defining the observation range to sample from (start:stop).
    shuffle
        Whether to shuffle slice and index order.
    preload_nslices
        Number of slices to load per iteration.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator for shuffling.
    """

    _batch_size: int
    _slice_size: int
    _shuffle: bool
    _preload_nslices: int
    _mask: slice
    _n_slices: int
    _n_iters: int
    _drop_last: bool
    _rng: np.random.Generator

    def __init__(
        self,
        *,
        batch_size: int,
        slice_size: int,
        mask: slice,
        shuffle: bool = False,
        preload_nslices: int,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        start = mask.start if mask.start is not None else 0
        stop = mask.stop

        if start < 0:
            raise ValueError("mask.start must be >= 0")
        if stop is not None and start >= stop:
            raise ValueError("mask.start must be < mask.stop when mask.stop is specified")

        check_lt_1([slice_size, preload_nslices], ["Slice size", "Preload slices"])
        preload_size = slice_size * preload_nslices

        if batch_size > preload_size:
            raise ValueError(
                "batch_size cannot exceed slice_size * preload_nslices. "
                f"Got batch_size={batch_size}, but max is {preload_size}."
            )

        self._rng = np.random.default_rng() if rng is None else rng
        self._batch_size = batch_size
        self._slice_size = slice_size
        self._shuffle = shuffle
        self._preload_nslices = preload_nslices
        self._mask = slice(start, stop)  # stop can be None
        self._drop_last = drop_last
        self._worker_handle: WorkerHandle | None = None

    def _prepare_start_stop(self, n_obs: int) -> tuple[int, int]:
        """Prepare the start and stop indices for sampling."""
        start = self._mask.start if self._mask.start is not None else 0
        stop = self._mask.stop if self._mask.stop is not None else n_obs

        if stop > n_obs:
            raise ValueError(
                f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
                "The sampler range must be within the loader's observations."
            )
        if start >= stop:
            raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")

        return start, stop

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
        _ = self._prepare_start_stop(n_obs)  # ignore return only validate

    def sample(self, n_obs: int) -> Iterator[LoadRequest[list[slice]]]:
        """Sample load requests given the total number of observations.

        Parameters
        ----------
        n_obs
            The total number of observations available.

        Yields
        ------
        LoadRequest
            Load requests for batching data.
        """
        start, stop = self._prepare_start_stop(n_obs)

        # Compute slices directly from resolved mask range
        slices = self._compute_slices(start, stop)
        n_slices = len(slices)

        # Create slice indices for shuffling
        slice_indices = np.arange(n_slices)
        if self._shuffle:
            slice_indices = self._shuffle_integers(slice_indices)

        # Worker sharding: each worker gets a disjoint subset of slices
        if self._worker_handle is not None:
            slice_indices = self._worker_handle.get_part_for_worker(slice_indices)

        n_slices_for_worker = len(slice_indices)
        n_slice_iters = math.ceil(n_slices_for_worker / self._preload_nslices) if n_slices_for_worker > 0 else 0

        n_leftover_indices = 0

        for i in range(n_slice_iters):
            preload_start = i * self._preload_nslices
            preload_end = min(preload_start + self._preload_nslices, n_slices_for_worker)
            indices_to_load = slice_indices[preload_start:preload_end]

            # Compute total observations to load from selected slices
            total_obs_to_load = sum(slices[idx].stop - slices[idx].start for idx in indices_to_load)

            # Generate loaded indices with leftover from previous iteration
            loaded_indices = np.arange(total_obs_to_load + n_leftover_indices)
            if self._shuffle:
                loaded_indices = self._shuffle_integers(loaded_indices)
            splits = list(np.split(loaded_indices, np.arange(self._batch_size, len(loaded_indices), self._batch_size)))

            is_last_iter = i == n_slice_iters - 1
            last_is_partial = splits[-1].shape[0] < self._batch_size

            if last_is_partial:
                if is_last_iter and self._drop_last:
                    # Drop the final partial batch entirely
                    splits = splits[:-1]
                    n_leftover_indices = 0
                else:
                    # Track leftover count for next iteration's index generation
                    # splits[-1] is partial and will be carried over by Loader
                    n_leftover_indices = splits[-1].shape[0]
            else:
                n_leftover_indices = 0

            yield LoadRequest(
                slices=[slices[idx] for idx in indices_to_load],
                splits=splits,
            )

    def set_worker_handle(self, worker_handle: WorkerHandle) -> None:
        # Worker mode validation - only check when there are multiple workers
        if worker_handle.num_workers > 1:
            if not self._drop_last and self._preload_nslices * self._slice_size % self._batch_size != 0:
                raise ValueError(
                    f"When using DataLoader workers with drop_last=False, "
                    f"(slice_size * preload_nslices) must be divisible by batch_size. "
                    f"Got {self._preload_nslices * self._slice_size} % {self._batch_size} = "
                    f"{self._preload_nslices * self._slice_size % self._batch_size}. "
                    f"Set drop_last=True to allow non-divisible configs."
                )
            if self._drop_last:
                warnings.warn(
                    "With drop_last=True and multiple workers, up to "
                    f"(batch_size - 1) * num_workers = {(self._batch_size - 1) * worker_handle.num_workers} "
                    "observations may be dropped (one partial batch per worker).",
                    UserWarning,
                    stacklevel=2,
                )
        self._worker_handle = worker_handle

    def supports_workers(self) -> bool:
        return True

    def _shuffle_integers(self, integers: np.ndarray) -> np.ndarray:
        if self._worker_handle is None:
            self._rng.shuffle(integers)
        else:
            self._worker_handle.shuffle(integers)
        return integers

    def _compute_slices(self, start: int, stop: int) -> list[slice]:
        """Compute slices from start and stop indices."""
        starts = list(range(start, stop, self._slice_size))
        stops = starts[1:] + [stop]
        return [slice(s, e) for s, e in zip(starts, stops, strict=True)]

    @property
    def batch_size(self) -> int | None:
        return self._batch_size
