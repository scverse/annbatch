"""Categorical sampler for group-stratified data access."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.abc import Sampler
from annbatch.samplers._chunk_sampler import ChunkSampler
from annbatch.utils import check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from annbatch.types import LoadRequest


class CategoricalSampler(Sampler):
    """Categorical sampler for group-stratified batched data access.

    This sampler ensures each batch contains observations from a single category/group.
    It samples from categories proportionally to their size, yielding batches where
    all observations belong to the same category.

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

        boundaries = [slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]

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
        if find_spec("torch"):
            from torch.utils.data import get_worker_info

            if get_worker_info() is not None:
                raise ValueError(
                    "CategoricalSampler does not support multiple workers. Use num_workers=0 in your DataLoader."
                )

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Sample load requests, ensuring each batch is from a single category.

        The sampling strategy:
        1. Collect all load requests from each category sampler
        2. Flatten into individual batches: (chunk_group_id, batch_index_within_group)
        3. Shuffle the batch order across categories
        4. Group batches by their chunk_group_id and yield combined load requests
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

        for group_id, (chunks, n_batches) in enumerate(all_chunk_groups):
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
