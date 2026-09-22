"""WeightedClassSampler -- class-weighted (but not class-coherent) batches."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from annbatch.utils import split_given_size

from ._class_sampler import ClassSampler

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class WeightedClassSampler(ClassSampler):
    """Sample batches whose *class composition* follows ``class_weights``.

    Chunks are read exactly as :class:`~annbatch.samplers.ClassSampler` reads them --
    one class per chunk -- but the rows of a whole preload window are shuffled together
    before being split into batches, so each batch mixes classes in expectation
    proportionally to their weights instead of being drawn from a single class.
    """

    def _iter_requests(self) -> Iterator[LoadRequest]:
        n_slices, remainder = divmod(self._num_samples, self._chunk_size)
        if remainder > 0:
            n_slices += 1
        class_of_slices = self._rng.choice(self._rle_manager.codes, size=n_slices, p=self._rle_manager.weights)
        slices = self._rle_manager.slices_from_classes(class_of_slices, self._rng)
        if remainder > 0:
            last = int(slices[-1].start)
            slices[-1] = slice(last, last + remainder)
        window_size = self._preload_nchunks * self._chunk_size
        ids = np.arange(window_size)
        for window in itertools.batched(slices, self._preload_nchunks):
            n_rows = (len(window) - 1) * self._chunk_size + (window[-1].stop - window[-1].start)
            ids_to_use = ids if n_rows == window_size else np.arange(n_rows)
            self._rng.shuffle(ids_to_use)
            splits = split_given_size(ids_to_use, self._batch_size)
            if self._drop_last and splits[-1].size < self._batch_size:
                splits = splits[:-1]
                if not splits:
                    continue
            yield {"requests": list(window), "splits": splits}
