"""Sampler classes for efficient slice-based data access."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING

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
    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Implementation of the sample method.

        This method is called by the sample method to perform the actual sampling after
        the worker handle is set.

        Parameters
        ----------
        n_obs
            The total number of observations available.

        Yields
        ------
        LoadRequest
            Load requests for batching data.
        """

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
        worker_handle = self.worker_handle
        yield from self._sample(n_obs, worker_handle)

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

    @property
    @abstractmethod
    def worker_handle(self) -> WorkerHandle | None:
        """The worker handle if the sampler supports workers."""


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
        # TODO: These checks can be redundant since the Loader will also check them as well
        # but the problem is these assumptions are also made in the Loader
        # how do we handle this? (also holds for batch_size > preload_size)
        if preload_size % batch_size != 0:
            raise ValueError(
                "slice_size * preload_nslices must be divisible by batch_size. "
                f"Got {preload_size} % {batch_size} = {preload_size % batch_size}."
            )

        self._rng = np.random.default_rng() if rng is None else rng
        self._batch_size = batch_size
        self._slice_size = slice_size
        self._shuffle = shuffle
        self._preload_nslices = preload_nslices
        self._mask = slice(start, stop)  # stop can be None
        self._drop_last = drop_last

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

    def _process_iter(
        self, indices: np.ndarray, slices: list[slice], slice_indices_to_load: np.ndarray, n_split_per_iter: int
    ) -> LoadRequest:
        if self._shuffle:
            indices = self._shuffle_integers(indices.copy())
        splits = np.array_split(indices, n_split_per_iter)
        return LoadRequest(
            slices=[slices[idx] for idx in slice_indices_to_load],
            splits=splits,
        )

    def _sample(self, n_obs: int, worker_handle: WorkerHandle | None = None) -> Iterator[LoadRequest]:
        """Implementation of the sample method.

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

        # Create slice indices for shuffling
        n_slices = math.ceil((stop - start) / self._slice_size)
        slice_indices = np.arange(n_slices)
        if self._shuffle:
            slice_indices = self._shuffle_integers(slice_indices, worker_handle)

        slices = self._compute_slices(slice_indices, start, stop)

        # Worker sharding: each worker gets a disjoint subset of slices
        if worker_handle is not None:
            slice_indices = worker_handle.get_part_for_worker(slice_indices)

        n_slices_for_worker = len(slice_indices)
        n_slice_iters = math.ceil(n_slices_for_worker / self._preload_nslices) if n_slices_for_worker > 0 else 0

        # there is only one slice that isn't complete and that is the last slice
        # extract the iteration result that contains the last slice
        n_obs_per_iter = self._preload_nslices * self._slice_size
        n_split_per_iter = n_obs_per_iter // self._batch_size

        loaded_indices = np.arange(n_obs_per_iter)
        slices_per_iter = np.array_split(slice_indices, n_slice_iters)

        for slice_indices_to_load in slices_per_iter[:-1]:
            yield self._process_iter(loaded_indices, slices, slice_indices_to_load, n_split_per_iter)
        # don't want to check if last slice because python loops
        # are already expensive
        # no need to check each time, just do it once at the end
        last_n_obs = sum(slices[idx].stop - slices[idx].start for idx in slices_per_iter[-1])
        loaded_indices = np.arange(last_n_obs)
        if self._shuffle:
            loaded_indices = self._shuffle_integers(loaded_indices)
        splits = np.array_split(loaded_indices, n_split_per_iter)
        if (self._drop_last and splits[-1].shape[0] < self._batch_size) or splits[-1].shape[0] == 0:
            splits = splits[:-1]
        yield LoadRequest(
            slices=[slices[idx] for idx in slices_per_iter[-1]],
            splits=splits,
        )

    @property
    def worker_handle(self) -> WorkerHandle | None:
        # Worker mode validation - only check when there are multiple workers
        worker_handle = None
        if find_spec("torch"):
            from torch.utils.data import get_worker_info

            from annbatch.utils import WorkerHandle

            if get_worker_info() is not None:
                worker_handle = WorkerHandle()

        if worker_handle is not None and worker_handle.num_workers > 1 and not self._drop_last:
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")
        return worker_handle

    def _shuffle_integers(self, integers: np.ndarray, worker_handle: WorkerHandle | None = None) -> np.ndarray:
        if worker_handle is None:
            self._rng.shuffle(integers)
        else:
            worker_handle.shuffle(integers)
        return integers

    def _compute_slices(self, slice_indices: np.ndarray, start: int, stop: int) -> list[slice]:
        """Compute slices from start and stop indices.

        This function is used to compute the slices for the data to load.
        The slices are computed such that the last slice is the incomplete slice if the total number of observations is not divisible by the slice size.
        Supposed to also work with shuffled slice indices so that the last slice computed isn't always the incomplete slice.
        """
        n_slices = len(slice_indices)
        pivot_index = slice_indices[-1]

        offsets = np.ones(n_slices + 1, dtype=int) * self._slice_size
        offsets[0] = start
        if (incomplete_slice_size := (stop - start) % self._slice_size) == 0:
            incomplete_slice_size = self._slice_size
        offsets[pivot_index + 1] = incomplete_slice_size
        offsets = np.cumsum(offsets)

        starts = offsets[:-1][slice_indices]
        stops = offsets[1:][slice_indices]
        return [slice(s, e) for s, e in zip(starts, stops, strict=True)]

    @property
    def batch_size(self) -> int | None:
        return self._batch_size
