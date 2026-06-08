from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from annbatch.utils import check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class WorkerInfo(NamedTuple):
    """Minimal worker info for RNG handling."""

    id: int
    num_workers: int


def get_torch_worker_info() -> WorkerInfo | None:
    """Get torch DataLoader worker info if available.

    Returns None if torch is not installed or not in a worker process.
    """
    if importlib.util.find_spec("torch"):
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        if info is not None:
            return WorkerInfo(id=info.id, num_workers=info.num_workers)
    return None


def iter_from_slices(
    slices: list[slice],
    batch_rng: np.random.Generator,
    worker_info: WorkerInfo | None,
    preload_nchunks: int,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    chunk_size: int,
) -> Iterator[LoadRequest]:
    # Worker sharding: each worker gets a disjoint subset of slices
    if worker_info is not None:
        slices = np.array_split(slices, worker_info.num_workers)[worker_info.id]
    # Set up the iterator for slices and the batch indices for splits
    slices_per_request = split_given_size(slices, preload_nchunks)
    in_memory_size = preload_nchunks * chunk_size
    batch_indices = np.arange(in_memory_size)
    split_batch_indices = split_given_size(batch_indices, batch_size)
    for request_slices in slices_per_request[:-1]:
        if shuffle:
            # Avoid copies using in-place shuffling since `self.shuffle` should not change mid-training
            batch_rng.shuffle(batch_indices)
            split_batch_indices = split_given_size(batch_indices, batch_size)
        yield {"requests": request_slices, "splits": split_batch_indices}
    # On the last yield, drop the last uneven batch and create new batch_indices since the in-memory size of this last yield could be divisible by batch_size but smaller than preload_nchunks * chunk_size
    final_slices = slices_per_request[-1]
    total_obs_in_last_batch = int(sum(s.stop - s.start for s in final_slices))
    if total_obs_in_last_batch == 0:  # pragma: no cover
        raise RuntimeError("Last batch was found to have no observations. Please open an issue.")
    if drop_last:
        if total_obs_in_last_batch < batch_size:
            return
        total_obs_in_last_batch -= total_obs_in_last_batch % batch_size
    indices = batch_rng.permutation(total_obs_in_last_batch) if shuffle else np.arange(total_obs_in_last_batch)
    batch_indices = split_given_size(indices, batch_size)
    yield {"requests": final_slices, "splits": batch_indices}


def validate_chunk_batch_preload_sizes(
    chunk_size: int,
    preload_nchunks: int,
    batch_size: int,
) -> None:
    check_lt_1([chunk_size, preload_nchunks], ["Chunk size", "Preloaded chunks"])
    preload_size = chunk_size * preload_nchunks

    if batch_size > preload_size:
        raise ValueError(
            "batch_size cannot exceed chunk_size * preload_nchunks. "
            f"Got batch_size={batch_size}, but max is {preload_size}."
        )
    if preload_size % batch_size != 0:
        raise ValueError(
            "chunk_size * preload_nchunks must be divisible by batch_size. "
            f"Got {preload_size} % {batch_size} = {preload_size % batch_size}."
        )


def validate_mask_and_resolve(mask: slice) -> tuple[int, int]:
    """Validate a sampler mask against sanity checks then resolve the start and stop."""
    if mask.step is not None and mask.step != 1:
        raise ValueError(f"mask.step must be 1, but got {mask.step}")
    start, stop = mask.start or 0, mask.stop
    if start < 0:
        raise ValueError("mask.start must be >= 0")
    if stop is not None and start >= stop:
        raise ValueError("mask.start must be < mask.stop when mask.stop is specified")
    return start, stop


def validate_mask_n_obs_and_resolve(mask: slice, n_obs: int) -> tuple[int, int]:
    """Validate a sampler mask against n_obs then resolve the start and stop."""
    start, stop = validate_mask_and_resolve(mask)
    if stop is None:
        stop = n_obs
    if stop > n_obs:
        raise ValueError(
            f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
            "The sampler range must be within the loader's observations."
        )
    if start >= stop:
        raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")
    return start, stop
