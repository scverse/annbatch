from __future__ import annotations

import importlib.util
import itertools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

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


def resolve_class_weights(class_weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    if class_weights is None:
        weights = np.ones(n_classes, dtype=float)
    else:
        weights = np.array(class_weights, dtype=float)
        if weights.shape != (n_classes,):
            raise ValueError(
                f"class_weights must have one weight per class (expected shape ({n_classes},), got {weights.shape})."
            )
    if not (weights > 0).any():
        raise ValueError("class_weights must have at least one positive weight.")
    return weights


def codes_of_categorical(categorical: pd.Categorical, name: str) -> np.ndarray:
    """Return a :class:`pandas.Categorical`'s integer codes, validating it first.

    Rejects a non-categorical -- the common mistake is passing a ``Series`` or ndarray instead of
    its ``.values`` -- and NA entries (``codes == -1``), both of which would otherwise fail later.
    """
    if not isinstance(categorical, pd.Categorical):
        raise TypeError(f"{name} must be a pandas.Categorical.")
    codes = categorical.codes  # pd.Categorical.codes is already an ndarray
    if (codes == -1).any():
        raise ValueError(f"{name} contains NA values (codes == -1). Remove NAs before passing.")
    return codes


def build_run_table(codes: np.ndarray, start: int) -> pd.DataFrame:
    # Run-length encode ``codes``; returns runs with global start/end (offset by ``start``), len and cat.
    # Boundaries where the code changes, plus the range's start and stop.
    edges = np.concatenate([np.array([0]), np.flatnonzero(np.diff(codes)) + 1, np.array([codes.shape[0]])])
    return pd.DataFrame(
        {
            "start": edges[:-1] + start,
            "end": edges[1:] + start,
            "len": np.diff(edges),
            "cat": codes[edges[:-1]],
        }
    )


def iter_windows(
    slices: list[slice],
    *,
    preload_nchunks: int,
    chunk_size: int,
    batch_size: int,
    drop_last: bool,
    rng: np.random.Generator,
) -> Iterator[LoadRequest]:
    # Group ``slices`` into preload windows and split each window into shuffled batches.
    window_size = preload_nchunks * chunk_size
    full_splits = split_given_size(np.arange(window_size), batch_size)
    for window in itertools.batched(slices, preload_nchunks):
        n_rows = (len(window) - 1) * chunk_size + (window[-1].stop - window[-1].start)
        splits = full_splits if n_rows == window_size else split_given_size(np.arange(n_rows), batch_size)
        if drop_last and splits[-1].size < batch_size:
            splits = splits[:-1]
            if not splits:
                continue
        for batch in splits:
            # can't vectorize this because we need to return a list, not ndarray
            rng.shuffle(batch)
        yield {"requests": list(window), "splits": splits}
