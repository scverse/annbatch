"""SequentialSampler -- ordered chunk-based sampler."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._chunk_sampler import iter_from_chunks, validate_chunk_batch_preload_sizes
from annbatch.samplers._utils import get_torch_worker_info

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class FragmentedRandomSampler(Sampler):
    """Random sampler for multiple non-overlapping data ranges.

    This sampler generates random chunks across multiple disjoint regions (masks) of a dataset,
    enabling efficient random sampling from fragmented data regions.

    Adjacent masks are automatically merged internally.
    For example, if masks=[slice(0, 10), slice(10, 20)], they will be merged into a single mask slice(0, 20).
    After this merging step, the sampler will ensure that each mask from the merged list of masks
    covers at least one full chunk and is within the dataset bounds.

    Multiple workers are not supported with this sampler.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    masks
        List of non-overlapping slices defining the data regions to sample from.
        Each slice must have start >= 0, stop > start, and step is 1 or None.
    preload_nchunks
        Number of chunks to load per iteration.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator for shuffling. Note that :func:`torch.manual_seed`
        has no effect on reproducibility here; pass a seeded
        :class:`numpy.random.Generator` to control randomness.
    num_samples
        Total number of observations to draw.
    """

    _batch_size: int
    _chunk_size: int
    _preload_nchunks: int
    _in_memory_size: int
    _num_samples: int
    _masks: list[slice]

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        masks: list[slice],
        num_samples: int,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        validate_chunk_batch_preload_sizes(chunk_size, preload_nchunks, batch_size)

        # standard mask validation
        if not all(mask.stop > mask.start and mask.start >= 0 for mask in masks):
            raise ValueError("All masks must have mask.stop > mask.start and mask.start >= 0.")
        if not all(mask.stop is not None and mask.start is not None for mask in masks):
            raise ValueError("All masks must have non-None start and stop.")
        if not all(mask.step == 1 or mask.step is None for mask in masks):
            raise ValueError("mask.step must be 1 or None for all masks in FragmentedRandomSampler.")

        # enforce that it's non-overlapping and sorted by start index
        # sorting by start index should be same with sorting by stop index otherwise there is an overlap
        sorted_masks = sorted(masks, key=lambda m: m.start)
        starts = np.array([m.start for m in sorted_masks], dtype=np.int64)
        stops = np.array([m.stop for m in sorted_masks], dtype=np.int64)
        if len(sorted_masks) > 1 and not np.all(stops[:-1] <= starts[1:]):
            raise ValueError("Masks must be non-overlapping.")

        # now we will merge any two masks that are adjacent
        is_adj = starts[1:] == stops[:-1]
        if np.any(is_adj):
            new_starts = np.concatenate(([starts[0]], starts[1:][~is_adj]))
            new_stops = np.concatenate((stops[:-1][~is_adj], [stops[-1]]))
            starts, stops = new_starts, new_stops

        if not np.all(stops - starts >= chunk_size):
            raise ValueError("Each mask must cover at least one chunk (mask.stop - mask.start >= chunk_size).")

        # precompute cumulative sums for efficient chunk sampling
        cumsum_centered = np.concatenate([np.array([0]), np.cumsum(stops - starts - self._chunk_size)])
        chunk_start_offsets = np.concatenate([np.array([0]), cumsum_centered[1:] - cumsum_centered[:-1]])

        self._rng = rng or np.random.default_rng()

        self._cumsum_centered, self._chunk_start_offsets = cumsum_centered, chunk_start_offsets
        self._starts, self._stops = starts, stops
        self._num_samples = num_samples
        self._drop_last = drop_last
        self._batch_size, self._chunk_size, self._preload_nchunks = (
            batch_size,
            chunk_size,
            preload_nchunks,
        )

    @property
    def mask(self) -> slice:
        raise NotImplementedError(
            "mask property is not implemented for FragmentedRandomSampler since it operates on multiple masks."
        )

    @mask.setter
    def mask(self, value: slice) -> None:
        raise NotImplementedError(
            "mask property is not implemented for FragmentedRandomSampler since it operates on multiple masks."
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return True

    def n_iters(self, n_obs: int) -> int:
        del n_obs  # not needed
        return (
            self._num_samples // self.batch_size if self._drop_last else math.ceil(self._num_samples / self.batch_size)
        )

    def validate(self, n_obs: int) -> None:
        """Validate if there are any masks that exceed the loader's n_obs."""
        if np.any(self._stops > n_obs):
            raise ValueError(
                f"Sampler has a mask from masks such that mask.stop exceeds loader n_obs ({n_obs}). "
                "The masks given to the sampler must be within the loader's observations."
            )

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        del n_obs  # not needed since we don't infer anything from n_obs
        worker_info = get_torch_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError("Multiple workers are not supported with FragmentedRandomSampler.")

        chunks = self._compute_chunks()
        return iter_from_chunks(
            chunks=chunks,
            batch_rng=self._rng,
            preload_nchunks=self._preload_nchunks,
            batch_size=self._batch_size,
            drop_last=self._drop_last,
            chunk_size=self._chunk_size,
            shuffle=True,
            worker_info=None,
        )

    def _compute_chunks(self):
        n_chunks, remainder = divmod(self._num_samples, self._chunk_size)
        if remainder > 0 and not self._drop_last:
            n_chunks += 1

        num_possible_chunk_starts = self._cumsum_centered[-1]

        offsets = self._rng.integers(num_possible_chunk_starts, size=n_chunks)
        frag_idx = np.searchsorted(self._cumsum_centered, offsets, side="left")

        # there is two layer of remapping here:
        # we need to readjust the distances between masks: done by chunk_start_offsets[frag_idx - 1]
        # adding the actual starts of the masks: done by self._starts[frag_idx - 1]
        chunk_starts = offsets - self._chunk_start_offsets[frag_idx - 1] + self._starts[frag_idx - 1]
        chunks = [slice(int(s), int(s + self._chunk_size)) for s in chunk_starts]
        if remainder > 0 and not self._drop_last:
            chunks[-1] = slice(int(chunk_starts[-1]), int(chunk_starts[-1] + remainder))
        return chunks
