"""Categorical sampler for group-stratified data access."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.abc import Sampler
from annbatch.samplers._chunk_sampler import ChunkSampler
from annbatch.samplers._utils import get_worker_handle, is_in_worker
from annbatch.utils import check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from annbatch.types import LoadRequest
    from annbatch.utils import WorkerHandle


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

        self._rng = rng or np.random.default_rng()

        child_rngs = self._rng.spawn(len(category_boundaries))

        # Create a ChunkSampler for each category, using its boundary as the mask
        # Always use drop_last=True internally
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

    @property
    def category_sizes(self) -> list[int]:
        """The size (number of observations) for each category."""
        return [s._mask.stop - s._mask.start for s in self._category_samplers]

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
            raise ValueError(
                "CategoricalSampler does not support multiple workers. Use num_workers=0 in your DataLoader."
            )

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Sample load requests, ensuring each batch is from a single category.

        The sampling strategy:
        1. Collect all load requests from each category sampler into chunk groups
        2. Flatten into individual batches: (chunk_group_id, batch_index_within_group)
        3. Shuffle the batch order across all categories
        4. Group n_categories batches together per load request and
           combine chunks from the selected batches into a single load request
        """
        batch_size = self._category_samplers[0]._batch_size

        # Collect all chunk groups: list of (chunks, n_batches)
        # chunk_group_id is the index into this list
        all_chunk_groups: list[tuple[list[slice], int]] = []

        for sampler in self._category_samplers:
            for load_request in sampler._sample(n_obs):
                chunks = list(load_request["chunks"])
                # Count only non-empty splits (drop_last may produce empty final split)
                n_batches = sum(1 for s in load_request["splits"] if len(s) > 0)
                if n_batches > 0:
                    all_chunk_groups.append((chunks, n_batches))

        if not all_chunk_groups:
            return

        # Flatten into individual batches: (chunk_group_id, batch_index_within_group)
        all_batches: list[tuple[int, int]] = []

        for group_id, (_, n_batches) in enumerate(all_chunk_groups):
            for batch_idx in range(n_batches):
                all_batches.append((group_id, batch_idx))

        if not all_batches:
            return

        # Shuffle the batch order
        batch_order = np.arange(len(all_batches))
        self._rng.shuffle(batch_order)

        # Group batches that share the same chunk_group_id together
        # Yield one load request per unique set of chunk groups
        batches_per_load = len(self._category_samplers)

        for i in range(0, len(batch_order), batches_per_load):
            selected_batch_indices = batch_order[i : i + batches_per_load]

            # Collect unique chunk groups needed for this load request
            # Map: chunk_group_id -> (chunks, list of batch indices within that group)
            groups_in_load: dict[int, list[int]] = {}
            for batch_idx in selected_batch_indices:
                group_id, batch_num = all_batches[batch_idx]
                if group_id not in groups_in_load:
                    groups_in_load[group_id] = []
                groups_in_load[group_id].append(batch_num)

            # Build combined load request
            combined_chunks: list[slice] = []
            combined_splits: list[np.ndarray] = []

            # Track offset for each chunk group in the combined data
            group_offsets: dict[int, int] = {}
            current_offset = 0

            # First pass: add chunks and compute offsets
            for group_id in groups_in_load:
                chunks, _ = all_chunk_groups[group_id]
                group_offsets[group_id] = current_offset
                combined_chunks.extend(chunks)
                current_offset += sum(c.stop - c.start for c in chunks)

            # Second pass: create splits
            for batch_idx in selected_batch_indices:
                group_id, batch_num = all_batches[batch_idx]
                offset = group_offsets[group_id]

                # Create split indices
                start_idx = batch_num * batch_size
                end_idx = start_idx + batch_size
                split_indices = np.arange(start_idx, end_idx) + offset
                if self._category_samplers[0]._shuffle:
                    self._rng.shuffle(split_indices)
                combined_splits.append(split_indices)

            yield {"chunks": combined_chunks, "splits": combined_splits}


class StratifiedCategoricalSampler(CategoricalSampler):
    """Stratified categorical sampler with configurable weights and multi-worker support.

    Samples categories according to weights (uniform by default), yielding
    exactly ``n_yields`` batches total. Supports multi-worker DataLoaders by splitting
    ``n_yields`` across workers.

    Unlike :class:`CategoricalSampler`, this sampler:
    - Yields a fixed number of batches (``n_yields``) rather than exhausting all data
    - Samples with replacement (categories reset when exhausted)
    - Supports multi-worker DataLoaders
    - Allows configurable sampling weights (uniform by default)

    Parameters
    ----------
    category_boundaries
        A sequence of slices defining the boundaries for each category.
        Each slice represents a contiguous range of observations belonging to one category.
        Data must be sorted by category before using this sampler.
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch.
    n_yields
        Total number of batches to yield (split across workers if num_workers > 1).
    weights
        Sampling weights per category. Default is uniform (equal probability).
        Use ``weights=sampler.category_sizes`` for size-proportional sampling.
    shuffle
        Whether to shuffle chunk and index order within each category.
    rng
        Random number generator for shuffling.

    Examples
    --------
    >>> boundaries = [slice(0, 100), slice(100, 250), slice(250, 400)]
    >>> sampler = StratifiedCategoricalSampler(
    ...     category_boundaries=boundaries,
    ...     batch_size=32,
    ...     chunk_size=64,
    ...     preload_nchunks=4,
    ...     n_yields=1000,
    ... )

    Using custom weights (e.g., upsample rare categories):

    >>> sampler = StratifiedCategoricalSampler(
    ...     category_boundaries=boundaries,
    ...     batch_size=32,
    ...     chunk_size=64,
    ...     preload_nchunks=4,
    ...     n_yields=1000,
    ...     weights=[1.0, 2.0, 3.0],  # Category 2 sampled 3x as often as category 0
    ... )
    """

    _n_yields: int
    _weights: np.ndarray

    def __init__(
        self,
        category_boundaries: Sequence[slice],
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        n_yields: int,
        weights: Sequence[float] | None = None,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            category_boundaries,
            chunk_size,
            preload_nchunks,
            batch_size,
            shuffle=shuffle,
            rng=rng,
        )

        # Validate n_yields
        if n_yields < 1:
            raise ValueError("n_yields must be >= 1")
        self._n_yields = n_yields

        # Handle weights (uniform by default)
        if weights is None:
            self._weights = np.ones(self.n_categories, dtype=float)
        else:
            if len(weights) != self.n_categories:
                raise ValueError(f"weights length ({len(weights)}) must match n_categories ({self.n_categories})")
            weights = np.asarray(weights, dtype=float)
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
            if weights.sum() == 0:
                raise ValueError("weights must not sum to zero")
            self._weights = weights

    @property
    def n_yields(self) -> int:
        """Total number of batches to yield."""
        return self._n_yields

    @property
    def weights(self) -> np.ndarray:
        """Sampling weights for each category (not normalized)."""
        return self._weights.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Normalized sampling probabilities for each category."""
        return self._weights / self._weights.sum()

    @classmethod
    def from_pandas(
        cls,
        categorical: pd.Categorical | pd.Series,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        n_yields: int,
        weights: Sequence[float] | None = None,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> StratifiedCategoricalSampler:
        """Create a StratifiedCategoricalSampler from a pandas Categorical or Series.

        This extends :meth:`CategoricalSampler.from_pandas` with additional
        parameters for stratified sampling.

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
        n_yields
            Total number of batches to yield.
        weights
            Sampling weights per category. Default is uniform (equal probability).
        shuffle
            Whether to shuffle chunk and index order within each category.
        rng
            Random number generator for shuffling.

        Returns
        -------
        StratifiedCategoricalSampler
            A sampler configured with boundaries derived from the categorical.

        Raises
        ------
        ValueError
            If the data is not sorted by category.
        TypeError
            If the input is not a Categorical or categorical Series.
        """
        boundaries = cls._boundaries_from_pandas(categorical)
        return cls(
            category_boundaries=boundaries,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            n_yields=n_yields,
            weights=weights,
            shuffle=shuffle,
            rng=rng,
        )

    def validate(self, n_obs: int) -> None:
        """Validate the sampler configuration against the loader's n_obs.

        Unlike CategoricalSampler, this sampler supports multi-worker DataLoaders.

        Parameters
        ----------
        n_obs
            The total number of observations in the loader.

        Raises
        ------
        ValueError
            If the sampler configuration is invalid for the given n_obs.
        """
        # Validate category samplers (skip parent's worker check)
        for sampler in self._category_samplers:
            sampler.validate(n_obs)
        # NOTE: Multi-worker IS supported for stratified (unlike parent CategoricalSampler)

    def _get_worker_handle(self) -> WorkerHandle | None:
        """Get WorkerHandle for worker-specific RNGs."""
        return get_worker_handle(self._rng)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Sample load requests using stratified sampling with replacement.

        Categories are sampled according to weights (uniform by default).
        When a category is exhausted, its iterator is reset (sampling with replacement).
        """
        worker_handle = self._get_worker_handle()

        if worker_handle is not None:
            worker_id = worker_handle.worker_id
            num_workers = worker_handle.num_workers
            # Split n_yields across workers
            worker_n_yields = self._n_yields // num_workers
            if worker_id < (self._n_yields % num_workers):
                worker_n_yields += 1
            # Use worker-specific RNG from handle
            worker_rng = worker_handle.rng
        else:
            worker_n_yields = self._n_yields
            worker_rng = self._rng

        if worker_n_yields == 0:
            return

        probs = self.probabilities
        category_iters: list[Iterator[LoadRequest] | None] = [None] * self.n_categories
        yields_so_far = 0

        while yields_so_far < worker_n_yields:
            # Sample category using worker RNG
            cat_idx = int(worker_rng.choice(self.n_categories, p=probs))

            # Get/reset iterator for this category
            if category_iters[cat_idx] is None:
                category_iters[cat_idx] = iter(self._category_samplers[cat_idx]._sample(n_obs))

            try:
                load_request = next(category_iters[cat_idx])
            except StopIteration:
                # Reset iterator (sample with replacement)
                category_iters[cat_idx] = iter(self._category_samplers[cat_idx]._sample(n_obs))
                load_request = next(category_iters[cat_idx])

            # Yield individual batches from this load request
            for split in load_request["splits"]:
                if yields_so_far >= worker_n_yields:
                    return
                yield {"chunks": load_request["chunks"], "splits": [split]}
                yields_so_far += 1
