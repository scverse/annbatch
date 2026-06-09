"""CategoricalSampler -- categorical chunk sampler."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.abc import Sampler
from annbatch.samplers._utils import (
    check_lt_1,
    get_torch_worker_info,
    validate_chunk_batch_preload_sizes,
    validate_mask_n_obs_and_resolve,
)
from annbatch.utils import split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class CategoricalSampler(Sampler):
    """Sample category-coherent batches with replacement.

    Every batch the :class:`~annbatch.Loader` yields is drawn from a single category:
    a category is drawn ``c ~ Categorical(p)`` (``p`` proportional to
    ``category_weights``, uniform by default), then the batch's observations are drawn
    from ``c``. A load request may span several categories but no batch mixes them,
    which makes over- or under-sampling specific populations straightforward.

    Sampling is **with replacement** -- each pass draws ``num_samples`` observations
    rather than partitioning a fixed epoch -- so there is no notion of an epoch and the
    number of iterations is fixed. The chunk_size should be divided by the batch_size
    so that every batch stays within one category; see the implementation notes below.

    *Category selection.* A category with a non-positive weight is excluded: it is
    never sampled and its runs are exempt from the run-length rule below. Set a
    weight to ``0`` to drop a category; there is no separate exclusion argument.

    *Run-length rule.* Every contiguous run of a *non-excluded* category must span
    at least ``chunk_size`` observations; otherwise no aligned slice fits inside it
    and the sampler raises at construction, naming the offending categories by their
    label (not their integer code).
    *Mask.* Assigning :attr:`mask` restricts sampling to a contiguous observation
    range ``[start, stop)``. The RLE is rebuilt over that window (slice starts stay
    in global coordinates) and cached on the resolved ``(start, stop)`` pair, so
    reassigning the same mask is free. Category weights are renormalized from the
    original values over only the categories present in the new range; if no
    category with a positive weight remains, the assignment raises.

    Multiple workers are not supported with this sampler.

    Implementation
    --------------
    A run-length encoding (RLE) of ``categorical.codes`` is built over the :attr:`mask`
    range. Each draw picks a category ``c ~ Categorical(p)`` then a uniform slice-start
    within ``c``, and a prefix-sum lookup maps it to the absolute slice in
    *O(log n_runs)*. Because ``chunk_size`` must be a whole number of ``batch_size``
    blocks, every batch cut from a slice stays within its one category. Memory scales
    with the number of runs (``<= n_obs // chunk_size``).





    Parameters
    ----------
    chunk_size
        Number of observations in each slice yielded. Also the minimum run length
        required of every non-excluded category (see the run-length rule).
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
        Total number of observations to draw.
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
        if chunk_size % batch_size != 0:
            # each chunk must hold a whole number of batches so that every batch stays
            # within a single slice (hence a single category); see _iter_requests.
            raise ValueError(
                "chunk_size must be divisible by batch_size so that each batch stays within one category. "
                f"Got chunk_size={chunk_size} % batch_size={batch_size} = {chunk_size % batch_size}."
            )
        if mask is None:
            mask = slice(0, None)
        start, stop = validate_mask_n_obs_and_resolve(mask, n_obs)

        self._categorical = categorical
        self._n_obs = n_obs
        self._rng = rng or np.random.default_rng()
        self._num_samples = num_samples
        self._drop_last = drop_last
        self._batch_size, self._chunk_size, self._preload_nchunks = batch_size, chunk_size, preload_nchunks
        self._mask = slice(start, stop)

        # categories and their weights are mask-independent; kept so any mask can renormalize from them
        self._build_categories(category_weights)

        # eager build for the (default or constructor) range so run-length errors surface early
        self._built_range: tuple[int, int] | None = None
        self._ensure_runs(self._n_obs)

    def _build_categories(self, category_weights: np.ndarray | None) -> None:
        """Resolve the (non-excluded) categories and their renormalizable weights."""
        n_cats = len(self._categorical.categories)
        if category_weights is None:
            weights = np.ones(n_cats, dtype=float)
        else:
            weights = np.array(category_weights, dtype=float)
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

        masked = self._categorical.codes[start:stop]
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
            bad_labels = self._categorical.categories[bad].tolist()
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
        self._cat_base = self._run_pos_cumsum[first]  # value in `run_pos_cumsum` where each category's span begins
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
                f"categorical length ({self._n_obs}) does not match loader n_obs ({n_obs}). "
                "The categorical column must describe exactly the loader's observations."
            )
        self._ensure_runs(n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError("Multiple workers are not supported with CategoricalSampler.")

        self._ensure_runs(n_obs)
        return self._iter_requests()

    def _iter_requests(self) -> Iterator[LoadRequest]:
        n_slices, remainder = divmod(self._num_samples, self._chunk_size)
        if remainder > 0 and not self._drop_last:
            n_slices += 1

        # 1) pick a category for each slice according to the sampling policy
        cat_of_draw = self._rng.choice(len(self._cat_ids), size=n_slices, p=self._probs)

        # 2) one uniform draw within each chosen category's flat span of valid positions
        local_off = (self._rng.random(n_slices) * self._cat_total[cat_of_draw]).astype(np.intp)
        global_off = self._cat_base[cat_of_draw] + local_off

        # 3) map the flat offset -> run -> absolute slice start (the searchsorted trick,
        #    generalized across every category at once)
        run_idx = np.searchsorted(self._run_pos_cumsum, global_off, side="right") - 1
        within = global_off - self._run_pos_cumsum[run_idx]
        slice_starts = self._run_start[run_idx] + within
        # NB: self._cat_ids[cat_of_draw] is the category label of each slice, available for free.

        slices = [slice(int(s), int(s + self._chunk_size)) for s in slice_starts]
        if remainder > 0 and not self._drop_last:
            last = int(slice_starts[-1])
            slices[-1] = slice(last, last + remainder)

        # now allocate the splits
        n_windows, window_remainder = divmod(self._num_samples, self._preload_nchunks * self._chunk_size)
        n_splits = self._chunk_size // self._batch_size
        splits = [np.arange(self._batch_size) for _ in range(n_splits)]
        windows = list(itertools.batched(slices, self._preload_nchunks))
        for i in range(n_windows - 1):
            for batch in splits:
                # can't vectorize this because we need to return a list, not ndarray
                self._rng.shuffle(batch)
            yield {
                "requests": list(windows[i]),
                "splits": splits,
            }

        if window_remainder > 0:
            final_window = windows[-1]
            n_rows = sum(s.stop - s.start for s in final_window)
            splits = split_given_size(np.arange(n_rows), self._batch_size)
            for batch in splits:
                self._rng.shuffle(batch)
            yield {"requests": list(final_window), "splits": splits}
