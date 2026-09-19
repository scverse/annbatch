"""WeightedClassSampler -- class-weighted (but not class-coherent) batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from annbatch.utils import split_given_size

from ._class_sampler import _RunClassSampler

if TYPE_CHECKING:
    import numpy as np


class WeightedClassSampler(_RunClassSampler):
    """Sample batches whose *class composition* follows ``class_weights``.

    Chunks are read exactly as :class:`~annbatch.samplers.ClassSampler` reads them --
    one class per chunk -- but the rows of a whole preload window are shuffled together
    before being split into batches, so each batch mixes classes in expectation
    proportionally to their weights instead of being drawn from a single class.
    """

    def _chunk_schedule(self, n_chunks: int) -> np.ndarray:
        return self._rng.choice(self._rle_manager.emittable_codes, size=n_chunks, p=self._rle_manager.weights)

    def _window_splits(self, ids: np.ndarray) -> list[np.ndarray]:
        """Cut a window's row ids into batches after shuffling the window as a whole."""
        self._rng.shuffle(ids)
        return split_given_size(ids, self._batch_size)
