"""WeightedClassSampler -- class-weighted (but not class-coherent) batches."""

from __future__ import annotations

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
        for load_request in super()._iter_requests():
            # rows the parent kept (drop_last already applied), reshuffled across the window
            row_ids = np.concatenate(load_request["splits"])
            self._rng.shuffle(row_ids)
            yield {"requests": load_request["requests"], "splits": split_given_size(row_ids, self._batch_size)}
