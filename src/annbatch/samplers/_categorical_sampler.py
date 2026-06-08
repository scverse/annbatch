"""CategoricalSampler -- categorical chunk sampler."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.abc import Sampler
from annbatch.samplers._utils import (
    check_lt_1,
    get_torch_worker_info,
    iter_from_chunks,
    validate_chunk_batch_preload_sizes,
    validate_mask_n_obs_and_resolve,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class CategoricalSampler(Sampler):
    """Categorical chunk sampler.

    The sampler is given an integer category code per observation (e.g. from a
    categorical column). Every chunk it yields lies entirely within a single
    category, so each on-disk read stays contiguous and the chunk's category is
    known for free. A run-length encoding (RLE) of the codes is built for the
    range selected by :attr:`mask` (the whole dataset by default) and a full
    epoch of chunks is drawn in one vectorized pass, so memory scales with the
    number of runs (``<= n_obs / chunk_size``), not the number of categories.

    The distribution *within* a category is uniform over its valid chunk-start
    positions; the distribution *over* categories is uniform by default and is
    reshaped by ``category_weights``.

    **Category selection.** A category with weight ``0`` is excluded: it is never
    sampled and is exempt from the run-length rule. There is no separate
    selection argument -- set a weight to ``0`` to drop a category.

    **Run-length rule.** Every contiguous run of a *non-excluded* category must be
    at least ``chunk_size`` observations long; otherwise no chunk could fit inside
    it and the sampler raises, naming the offending categories.

    **Mask.** Assigning :attr:`mask` restricts sampling to a contiguous range
    ``[start, stop)``. The RLE is rebuilt over ``codes[start:stop]`` (chunk starts
    stay in global coordinates) and the build is cached on the resolved range, so
    reassigning the same mask is free and a new range rebuilds exactly once. The
    weights of the categories still present in the range are renormalized; if no
    category with positive weight remains in the range, assigning the mask raises.

    Sampling is with replacement (chunks are drawn independently). ``num_samples``
    controls the total number of observations drawn per epoch.

    Multiple workers are not supported with this sampler.

    Parameters
    ----------
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch.
    categorical
        A :class:`pandas.Categorical` with one entry per observation, e.g.
        ``df["cell_type"].astype("category")`` or ``df["cell_type"].values``
        when the column already has a categorical dtype. Length must equal the
        loader's ``n_obs``. The obs axis need not be contiguous per category, but
        every run of a non-excluded category must be at least ``chunk_size`` long
        (see the run-length rule above). NA values (``codes == -1``) are not allowed.
    num_samples
        Total number of observations to draw per epoch.
    category_weights
        Optional weights, one per category in ``categorical.categories``
        (so ``len(category_weights) == len(categorical.categories)``), controlling
        how often each category is drawn. A non-positive weight excludes that category
        entirely. When ``None`` (the default) every category is drawn uniformly. For
        proportional (≈ plain global random) sampling pass each category's observation
        count. The weights are kept and, whenever a mask narrows the range, the weights
        of the categories still present are renormalized.
    mask
        Optional contiguous observation range to restrict sampling to. Defaults to
        the whole dataset.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator. Note that :func:`torch.manual_seed` has no effect
        here; pass a seeded :class:`numpy.random.Generator` to control randomness.
    """

    _batch_size: int
    _chunk_size: int
    _preload_nchunks: int
    _num_samples: int

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        categorical: pd.Categorical,
        num_samples: int,
        category_weights: np.ndarray | None = None,
        mask: slice | None = None,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        check_lt_1([num_samples], ["num_samples"])
        if not isinstance(categorical, pd.Categorical):
            raise TypeError(f"categorical must be a pandas.Categorical, got {type(categorical).__name__}.")
        codes = categorical.codes
        if (codes == -1).any():
            raise ValueError("categorical contains NA values (codes == -1). Remove NAs before passing.")
        n_obs = int(codes.shape[0])

        validate_chunk_batch_preload_sizes(chunk_size, preload_nchunks, batch_size)
        if mask is None:
            mask = slice(0, None)
        start, stop = validate_mask_n_obs_and_resolve(mask, n_obs)

        self._codes = codes
        self._n_obs = n_obs
        self._rng = rng or np.random.default_rng()
        self._num_samples = num_samples
        self._drop_last = drop_last
        self._batch_size, self._chunk_size, self._preload_nchunks = batch_size, chunk_size, preload_nchunks
        self._mask = slice(start, stop)
        self._categories = categorical.categories

        # categories and their weights are mask-independent; kept so any mask can renormalize from them
        self._build_categories(categorical.categories, category_weights)

        # eager build for the (default or constructor) range so run-length errors surface early
        self._built_range: tuple[int, int] | None = None
        self._ensure_runs(self._n_obs)

    def _build_categories(self, categories: pd.Index, category_weights: np.ndarray | None) -> None:
        """Resolve the (non-excluded) categories and their renormalizable weights."""
        n_cats = len(categories)
        if category_weights is None:
            weights = np.ones(n_cats, dtype=float)
        else:
            weights = np.asarray(category_weights, dtype=float)
            if weights.shape != (n_cats,):
                raise ValueError(
                    f"category_weights must have one weight per category in categorical.categories "
                    f"(expected shape ({n_cats},), got {weights.shape})."
                )
        if not (weights > 0).any():
            raise ValueError("category_weights must have at least one positive weight.")

        self._weights = weights  # full array (0 for excluded); codes are 0..N-1 so direct indexing works

    @property
    def mask(self) -> slice:
        return self._mask

    @mask.setter
    def mask(self, value: slice) -> None:
        # resolve + eagerly rebuild so range errors (run-length, no active category) surface on assignment
        start, stop = validate_mask_n_obs_and_resolve(value, self._n_obs)
        self._mask = slice(start, stop)
        self._ensure_runs(self._n_obs)

    def _ensure_runs(self, n_obs: int) -> None:
        """Build (or reuse) the RLE for the current mask range, cached on ``(start, stop)``."""
        start, stop = validate_mask_n_obs_and_resolve(self.mask, n_obs)
        if self._built_range == (start, stop):
            return

        masked = self._codes[start:stop]
        boundaries = np.flatnonzero(np.diff(masked)) + 1
        edges = np.concatenate([np.array([0]), boundaries, np.array([masked.shape[0]])])
        run_start = edges[:-1] + start  # offset back to global observation coordinates
        run_len = np.diff(edges)
        run_cat = masked[edges[:-1]]

        # keep only runs of non-excluded categories; excluded (weight 0) runs are exempt from every check
        keep = self._weights[run_cat] > 0
        run_start, run_len, run_cat = run_start[keep], run_len[keep], run_cat[keep]
        if run_cat.size == 0:
            raise ValueError(
                "No category with positive weight is present in the current mask range "
                f"[{start}, {stop}); its renormalized weights would sum to zero."
            )

        # run-length rule: every kept run must hold at least one full chunk
        too_short = run_len < self._chunk_size
        if np.any(too_short):
            bad = np.unique(run_cat[too_short])
            bad_labels = self._categories[bad].tolist()
            raise ValueError(
                f"Every contiguous run must be at least chunk_size ({self._chunk_size}) observations long, "
                f"but {int(too_short.sum())} run(s) are shorter (categories {bad_labels}). "
                "Re-chunk the data so each category's runs are large enough, lower chunk_size, "
                "or exclude these categories with a zero weight."
            )
        n_pos = run_len - self._chunk_size + 1  # number of valid chunk-start positions in the run

        # group runs by category so each category owns a contiguous span of `run_pos_cumsum`
        order = np.argsort(run_cat, kind="stable")
        run_start, run_cat, n_pos = run_start[order], run_cat[order], n_pos[order]

        run_pos_cumsum = np.concatenate([np.array([0]), np.cumsum(n_pos)])
        cat_ids, first = np.unique(run_cat, return_index=True)
        last = np.append(first[1:], len(run_cat))

        self._run_start = run_start
        self._run_pos_cumsum = run_pos_cumsum
        self._cat_ids = cat_ids
        self._cat_base = self._run_pos_cumsum[first]  # the value in `cum` where each category begins
        self._cat_total = (
            self._run_pos_cumsum[last] - self._run_pos_cumsum[first]
        )  # num of valid chunk positions per category
        w = self._weights[cat_ids]
        self._probs = w / w.sum()
        self._built_range = (start, stop)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return True

    def n_batches(self, n_obs: int) -> int:
        del n_obs  # determined by num_samples, not the loader size
        return (
            self._num_samples // self.batch_size if self._drop_last else math.ceil(self._num_samples / self.batch_size)
        )

    def validate(self, n_obs: int) -> None:
        """Validate that the codes describe exactly the loader's observations."""
        if n_obs != self._n_obs:
            raise ValueError(
                f"codes length ({self._n_obs}) does not match loader n_obs ({n_obs}). "
                "The categorical column must describe exactly the loader's observations."
            )
        self._ensure_runs(n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError("Multiple workers are not supported with CategoricalSampler.")

        self._ensure_runs(n_obs)
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

    def _compute_chunks(self) -> list[slice]:
        n_chunks, remainder = divmod(self._num_samples, self._chunk_size)
        if remainder > 0 and not self._drop_last:
            n_chunks += 1

        # 1) pick a category for each chunk according to the sampling policy
        cat_of_draw = self._rng.choice(len(self._cat_ids), size=n_chunks, p=self._probs)

        # 2) one uniform draw within each chosen category's flat span of valid positions
        local_off = self._rng.random(n_chunks) * self._cat_total[cat_of_draw]
        global_off = self._cat_base[cat_of_draw] + local_off

        # 3) map the flat offset -> run -> absolute chunk start (the searchsorted trick,
        #    generalized across every category at once)
        run_idx = np.searchsorted(self._run_pos_cumsum, global_off, side="right") - 1
        within = global_off - self._run_pos_cumsum[run_idx]
        chunk_starts = self._run_start[run_idx] + within
        # NB: self._cat_ids[cat_of_draw] is the category label of each chunk, available for free.

        chunks = [slice(int(s), int(s + self._chunk_size)) for s in chunk_starts]
        if remainder > 0 and not self._drop_last:
            last = int(chunk_starts[-1])
            chunks[-1] = slice(last, last + remainder)
        return chunks
