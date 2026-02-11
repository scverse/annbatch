"""Categorical sampler for group-stratified data access."""

from __future__ import annotations

from itertools import batched
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.abc import Sampler
from annbatch.samplers._chunk_sampler import ChunkSampler
from annbatch.samplers._utils import is_in_worker
from annbatch.utils import check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from annbatch.types import LoadRequest


class CategoricalSampler(Sampler):
    """Categorical sampler for group-stratified batched data access.

    This sampler ensures each batch contains observations from a single category/group.
    It iterates through all categories, yielding all batches exactly once per epoch.
    The batch order is shuffled across categories, but each individual batch contains
    observations from only one category.

    The sampler assumes data is sorted by category, with boundaries provided as slices.
    For convenience, use :meth:`from_pandas` to construct from a pandas Categorical.

    Parameters
    ----------
    category_boundaries
        A sequence of slices defining the boundaries for each category.
        Each slice represents a contiguous range of observations belonging to one category.
        Data must be sorted by category before using this sampler.
        Number of categories must be greater than 1 and all boundaries must be in increasing order.
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    shuffle
        Whether to shuffle chunk and index order within each category.
    preload_nchunks
        Number of chunks to load per iteration.
    rng
        Random number generator for shuffling.

    Notes
    -----
    This sampler does not support multiple workers. Using it with a DataLoader
    that has `num_workers > 0` will raise an error.

    Examples
    --------
    Using boundaries directly:

    >>> boundaries = [slice(0, 100), slice(100, 250), slice(250, 400)]
    >>> sampler = CategoricalSampler(
    ...     category_boundaries=boundaries,
    ...     batch_size=32,
    ...     chunk_size=64,
    ...     preload_nchunks=4,
    ... )

    Using from_pandas for convenience:

    >>> import pandas as pd
    >>> categories = pd.Categorical(["A", "A", "B", "B", "B", "C"])
    >>> sampler = CategoricalSampler.from_pandas(
    ...     categories,
    ...     batch_size=32,
    ...     chunk_size=64,
    ...     preload_nchunks=4,
    ... )
    """

    _category_samplers: list[ChunkSampler]
    _rng: np.random.Generator

    def __init__(
        self,
        category_boundaries: Sequence[slice],
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ):
        check_lt_1([len(category_boundaries)], ["Number of categories"])
        if batch_size < chunk_size:
            raise ValueError(
                f"batch_size ({batch_size}) cannot be less than chunk_size ({chunk_size}) because each batch must be from one category."
            )
        self._validate_boundaries(category_boundaries)
        self._rng = rng or np.random.default_rng()

        child_rngs = self._rng.spawn(len(category_boundaries))

        # Create a ChunkSampler for each category, using its boundary as the mask
        # Always use drop_last=True internally
        # also compute the number of batches for each category
        self._n_batches_per_category = [
            int((boundary.stop - boundary.start) // batch_size) for boundary in category_boundaries
        ]
        self._category_samplers = [
            ChunkSampler(
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=batch_size,
                mask=boundary,
                shuffle=shuffle,
                drop_last=True,
                rng=child_rng,
            )
            for boundary, child_rng in zip(category_boundaries, child_rngs, strict=True)
        ]

        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._shuffle = shuffle

    def _validate_boundaries(self, category_boundaries: Sequence[slice]) -> None:
        for i, boundary in enumerate(category_boundaries):
            if not isinstance(boundary, slice):
                raise TypeError(f"Expected slice for boundary {i}, got {type(boundary)}")
            if boundary.step is not None and boundary.step != 1:
                raise ValueError(f"Boundary {i} must have step=1 or None, got {boundary.step}")
            if boundary.start is None or boundary.stop is None:
                raise ValueError(f"Boundary {i} must have explicit start and stop")
            if boundary.start >= boundary.stop:
                raise ValueError(f"Boundary {i} must have start < stop, got {boundary}")
            if i == 0 and boundary.start != 0:
                raise ValueError(f"First boundary must start at 0, got {boundary.start}")
            if i > 0 and boundary.start != category_boundaries[i - 1].stop:
                raise ValueError(
                    f"Boundaries must be contiguous: boundary {i} starts at {boundary.start} "
                    f"but boundary {i - 1} ends at {category_boundaries[i - 1].stop}"
                )

    @staticmethod
    def _boundaries_from_pandas(categorical: pd.Categorical | pd.Series) -> list[slice]:
        """Compute category boundaries from a pandas Categorical or Series.

        Parameters
        ----------
        categorical
            A pandas Categorical or Series with categorical dtype.
            Data must be sorted by category.

        Returns
        -------
        list[slice]
            Boundaries for each category as slices.

        Raises
        ------
        ValueError
            If the data is not sorted by category or is empty.
        TypeError
            If the input is not a Categorical or categorical Series.
        """
        if isinstance(categorical, pd.Series):
            if not isinstance(categorical.dtype, pd.CategoricalDtype):
                raise TypeError(f"Expected categorical Series, got {categorical.dtype}")
            categorical = categorical.cat
        elif not isinstance(categorical, pd.Categorical):
            raise TypeError(f"Expected pandas.Categorical or categorical Series, got {type(categorical)}")

        codes = categorical.codes
        n_obs = len(codes)

        if n_obs == 0:
            raise ValueError("Cannot create sampler from empty categorical")

        # Check if sorted by finding where codes decrease
        if np.any(np.diff(codes) < 0):
            raise ValueError(
                "Data must be sorted by category. Use df.sort_values('category_column') before creating the sampler."
            )

        # Compute boundaries by finding where codes change
        # We need to handle the case where some categories might be empty
        change_points = np.where(np.diff(codes) != 0)[0] + 1
        starts = np.concatenate([[0], change_points])
        stops = np.concatenate([change_points, [n_obs]])

        return [slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]

    @classmethod
    def from_pandas(
        cls,
        categorical: pd.Categorical | pd.Series,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> CategoricalSampler:
        """Create a CategoricalSampler from a pandas Categorical or Series.

        The data is assumed to be sorted by category. This method computes the
        boundaries for each category based on where values change.

        Parameters
        ----------
        categorical
            A pandas Categorical or Series with categorical dtype.
            Data must be sorted by category.
        chunk_size
            Size of each chunk.
        preload_nchunks
            Number of chunks to load per iteration.
        batch_size
            Number of observations per batch.
        shuffle
            Whether to shuffle chunk and index order within each category.
        rng
            Random number generator for shuffling.

        Returns
        -------
        CategoricalSampler
            A sampler configured with boundaries derived from the categorical.

        Raises
        ------
        ValueError
            If the data is not sorted by category.
        TypeError
            If the input is not a Categorical or categorical Series.

        Examples
        --------
        >>> import pandas as pd
        >>> # Data must be sorted by category
        >>> obs_cat = pd.Categorical(["A", "A", "A", "B", "B", "C", "C", "C", "C"])
        >>> sampler = CategoricalSampler.from_pandas(
        ...     obs_cat,
        ...     batch_size=2,
        ...     chunk_size=4,
        ...     preload_nchunks=2,
        ... )
        """
        boundaries = cls._boundaries_from_pandas(categorical)

        return cls(
            category_boundaries=boundaries,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            shuffle=shuffle,
            rng=rng,
        )

    @property
    def batch_size(self) -> int:
        return self._category_samplers[0].batch_size

    @property
    def shuffle(self) -> bool:
        return self._category_samplers[0].shuffle

    @property
    def n_categories(self) -> int:
        """The number of categories in this sampler."""
        return len(self._category_samplers)

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
        # Validate each category sampler
        for sampler in self._category_samplers:
            sampler.validate(n_obs)

        # Check for worker usage - CategoricalSampler doesn't support workers
        if is_in_worker():
            raise ValueError("CategoricalSampler does not support multiple workers.")

    @staticmethod
    def _iter_batches(
        sampler: ChunkSampler, n_obs: int, chunks_per_batch: int
    ) -> Iterator[tuple[list[slice], np.ndarray]]:
        """Yield per batch given a sampler."""
        for load_request in sampler._sample(n_obs):
            chunks = load_request["chunks"]
            yield from batched(chunks, chunks_per_batch)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        batches_per_load = int((self._preload_nchunks * self._chunk_size) // self._batch_size)
        chunks_per_batch = int(self._batch_size / self._chunk_size)
        batch_generators = [self._iter_batches(sampler, n_obs, chunks_per_batch) for sampler in self._category_samplers]
        # simulate the category order
        category_order = np.concatenate([np.full(n, i) for i, n in enumerate(self._n_batches_per_category)])
        if self._shuffle:
            self._rng.shuffle(category_order)

        # pre-allocate and reshape to batches_per_load x batch_size
        # so that we can shuffle with numpy all at once
        batch_indices = np.arange(batches_per_load * self._batch_size).reshape(batches_per_load, self._batch_size)

        for cat_idxs in batched(category_order, batches_per_load):
            if self._shuffle:
                batch_indices = self._rng.permuted(batch_indices, axis=1)
            yield {
                "chunks": [chunk for cat_idx in cat_idxs for chunk in next(batch_generators[cat_idx])],
                "splits": list(batch_indices[: len(cat_idxs)]),
            }
