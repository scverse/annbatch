"""Categorical sampler for group-stratified data access."""

from __future__ import annotations

import math
from collections import Counter
from itertools import batched
from typing import TYPE_CHECKING, Any

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._random_sampler import RandomSampler
from annbatch.utils import check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from annbatch.io import GroupedCollection
    from annbatch.types import LoadRequest


class CategoricalSampler(Sampler):
    """Categorical sampler for group-stratified batched data access.

    Each batch contains observations from a single category/group,
    drawn with replacement.

    The sampler assumes data is already sorted by category with
    boundaries provided as slices.  Use :meth:`from_collection` to
    construct directly from a :class:`~annbatch.GroupedCollection`.

    Parameters
    ----------
    category_boundaries
        Sequence of slices defining contiguous observation ranges per
        category.  Must start at 0 and be contiguous (no gaps).
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each on-disk chunk range yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    num_samples
        Total number of observations to draw across all categories.
    weights
        Per-category sampling weights.  If ``None`` (default), uniform
        weights are used.  Weights are normalized to sum to 1.
    rng
        Random number generator for shuffling.

    Examples
    --------
    >>> boundaries = [slice(0, 100), slice(100, 250), slice(250, 400)]
    >>> sampler = CategoricalSampler(
    ...     category_boundaries=boundaries,
    ...     batch_size=32,
    ...     chunk_size=32,
    ...     preload_nchunks=4,
    ...     num_samples=3200,
    ... )
    """

    _category_samplers: list[RandomSampler]
    _rng: np.random.Generator
    _num_samples: int

    def __init__(
        self,
        category_boundaries: Sequence[slice],
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        num_samples: int,
        weights: Sequence[float] | np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        n_categories = len(category_boundaries)
        check_lt_1([n_categories, num_samples], ["Number of categories", "num_samples"])
        if batch_size < chunk_size:
            raise ValueError(
                f"batch_size ({batch_size}) cannot be less than chunk_size ({chunk_size}) "
                "because each batch must be from one category."
            )
        self._validate_boundaries(category_boundaries)
        self._rng = rng or np.random.default_rng()
        self._num_samples = num_samples

        if weights is None:
            self._weights = np.ones(n_categories, dtype=np.float64) / n_categories
        else:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != n_categories:
                raise ValueError(f"weights length ({len(w)}) must match number of categories ({n_categories})")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            total = w.sum()
            if total == 0:
                raise ValueError("weights must not all be zero")
            self._weights = w / total

        child_rngs = self._rng.spawn(n_categories)
        n_batches = math.ceil(num_samples / batch_size)

        self._category_samplers = [
            RandomSampler(
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=batch_size,
                replacement=True,
                num_samples=n_batches * batch_size,
                mask=boundary,
                drop_last=True,
                rng=child_rng,
            )
            for boundary, child_rng in zip(category_boundaries, child_rngs, strict=True)
        ]

        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks

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

    @classmethod
    def from_collection(
        cls,
        collection: GroupedCollection,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        num_samples: int,
        weights: Sequence[float] | np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> CategoricalSampler:
        """Create a CategoricalSampler from a :class:`~annbatch.GroupedCollection`.

        Reads the ``group_index`` metadata written by the collection and
        converts each group's ``start``/``stop`` into category boundaries.

        Parameters
        ----------
        collection
            A grouped collection whose ``group_index`` contains per-group
            ``start`` and ``stop`` columns.
        chunk_size
            Size of each chunk.
        preload_nchunks
            Number of chunks to load per iteration.
        batch_size
            Number of observations per batch.
        num_samples
            Total number of observations to draw.
        weights
            Per-category sampling weights.  See class docstring.
        rng
            Random number generator for shuffling.

        Returns
        -------
        CategoricalSampler
        """
        group_index = collection.group_index
        boundaries = [slice(int(row.start), int(row.stop)) for row in group_index.itertuples()]
        return cls(
            category_boundaries=boundaries,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            num_samples=num_samples,
            weights=weights,
            rng=rng,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return True

    @property
    def n_categories(self) -> int:
        """The number of categories in this sampler."""
        return len(self._category_samplers)

    def n_iters(self, n_obs: int) -> int:
        return math.ceil(self._num_samples / self._batch_size)

    def validate(self, n_obs: int) -> None:
        for sampler in self._category_samplers:
            sampler.validate(n_obs)

    @staticmethod
    def _iter_batches(sampler: RandomSampler, n_obs: int, chunks_per_batch: int) -> Iterator[tuple[slice, ...]]:
        """Yield per-batch chunk tuples from a single category sampler."""
        for load_request in sampler._sample(n_obs):
            chunks = load_request["chunks"]
            yield from batched(chunks, chunks_per_batch)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        batches_per_load = int((self._preload_nchunks * self._chunk_size) // self._batch_size)
        chunks_per_batch = int(self._batch_size / self._chunk_size)

        n_batches = self.n_iters(n_obs)
        category_order = self._rng.choice(len(self._category_samplers), size=n_batches, p=self._weights)

        counts = Counter[Any](category_order.tolist())
        batch_generators = {}
        for cat_idx, cat_n_batches in counts.items():
            sampler = self._category_samplers[cat_idx]
            sampler._num_samples = cat_n_batches * self._batch_size
            batch_generators[cat_idx] = self._iter_batches(sampler, n_obs, chunks_per_batch)

        batch_indices = np.arange(batches_per_load * self._batch_size).reshape(batches_per_load, self._batch_size)

        for cat_idxs in batched(category_order, batches_per_load):
            batch_indices = self._rng.permuted(batch_indices, axis=1)
            yield {
                "chunks": [chunk for cat_idx in cat_idxs for chunk in next(batch_generators[cat_idx])],
                "splits": list(batch_indices[: len(cat_idxs)]),
            }
