from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from annbatch.utils import check_lt_1


def resolve_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Default an omitted generator, and seed one from a (deprecated) integer seed."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, int | np.integer) and not isinstance(rng, bool):
        warnings.warn(
            f"Passing a seed as rng is deprecated; pass np.random.default_rng({rng!r}) instead.",
            FutureWarning,
            # point at the caller's code, however deep the sampler's __init__ chain is
            skip_file_prefixes=(str(Path(__file__).parents[1]),),
        )
        return np.random.default_rng(rng)
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}.")
    return rng


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


def codes_without_na(categorical: pd.Categorical, name: str) -> np.ndarray:
    """The codes of ``categorical``, rejecting anything but a :class:`pandas.Categorical` and NA values."""
    if not isinstance(categorical, pd.Categorical):
        raise TypeError(f"{name} must be a pandas.Categorical, got {type(categorical).__name__}.")
    if (categorical.codes == -1).any():
        raise ValueError(f"{name} contains NA values (codes == -1). Remove NAs before passing.")
    return categorical.codes


def resolve_class_weights(class_weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """One float weight per class, uniform when ``class_weights`` is None; a non-positive weight excludes a class."""
    if class_weights is None:
        weights = np.ones(n_classes, dtype=float)
    elif isinstance(class_weights, pd.Series):
        raise TypeError(
            "class_weights must be an array, not a pandas Series: a Series would be read positionally, "
            "ignoring its index. Pass class_weights.reindex(classes.categories).to_numpy()."
        )
    else:
        weights = np.array(class_weights, dtype=float)
        if weights.shape != (n_classes,):
            raise ValueError(
                f"class_weights must have one weight per class in classes.categories "
                f"(expected shape ({n_classes},), got {weights.shape})."
            )
    if not (weights > 0).any():
        raise ValueError("class_weights must have at least one positive weight.")

    return weights  # full array (0 for excluded); codes are 0..N-1 so direct indexing works
