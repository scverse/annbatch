"""CategoricalSampler -- single-pass, vectorized categorical chunk sampler.

This is the "one sampler, not N composed samplers" alternative to wrapping a
:class:`~annbatch.samplers.FragmentedRandomSampler` per category. Instead of
holding a list of slices (or a per-category sampler object) it keeps only:

* the run-length encoding (RLE) of contiguous category runs as numpy int arrays, and
* per-category offsets into a single flat prefix-sum of valid chunk-start positions.

A whole epoch's chunks are then drawn in one vectorized numpy pass -- no python
loop over categories, no per-category :class:`numpy.random.Generator`, and memory
that scales with the number of *runs* (``<= n_obs / chunk_size``) rather than with
``n_categories * n_obs``.
"""

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


class CategoricalSampler(Sampler):
    """Category-coherent random sampler over a fragmented categorical column.

    Every chunk that is yielded lies entirely within a single category, so each
    on-disk read stays contiguous *and* its category label is known for free.
    The distribution *within* a category is uniform over its valid chunk-start
    positions; the distribution *over* categories is uniform by default and can
    be reshaped with ``category_weights`` (e.g. pass per-category observation
    counts for proportional sampling).

    **Run-length rule.** Every contiguous run of a category must be at least
    ``chunk_size`` observations long. Otherwise no chunk-size read could ever
    land inside it, so rather than silently ignoring such a run the sampler
    raises at construction and names the offending categories. Re-chunk the data
    (so each category's fragments are large enough) or lower ``chunk_size``.

    Sampling is with replacement (chunks are drawn independently), mirroring
    :class:`~annbatch.samplers.FragmentedRandomSampler`. ``num_samples`` controls
    the total number of observations drawn per epoch.

    Multiple workers are not supported with this sampler.

    Parameters
    ----------
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch.
    codes
        Integer category code per observation, e.g. ``df["cell_type"].cat.codes``
        from the input dataframe. Length must equal the loader's ``n_obs``.
        Codes do not need to be contiguous on the obs axis -- a category may be
        spread across many runs (fragments) -- but every run must be at least
        ``chunk_size`` long (see the run-length rule above).
    num_samples
        Total number of observations to draw per epoch.
    categories
        Optional subset of category codes to sample from. When ``None`` (the
        default) every category present in ``codes`` is sampled. Requested codes
        must exist in ``codes``, and the run-length rule is only enforced for the
        selected categories (a short run in a category you do not sample is fine).
    category_weights
        Optional unnormalized weights aligned with :attr:`categories` controlling
        how often each category is drawn. When ``None`` (the default) categories
        are drawn uniformly. For proportional (≈ plain global random) sampling,
        pass each category's observation count, e.g. ``np.bincount(codes)[sampler.categories]``.
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
        codes: np.ndarray,
        num_samples: int,
        categories: np.ndarray | None = None,
        category_weights: np.ndarray | None = None,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        validate_chunk_batch_preload_sizes(chunk_size, preload_nchunks, batch_size)

        codes = np.asarray(codes)
        if codes.ndim != 1:
            raise ValueError("codes must be a 1D array of category codes (one per observation).")

        self._n_obs = int(codes.shape[0])
        self._rng = rng or np.random.default_rng()
        self._num_samples = num_samples
        self._drop_last = drop_last
        self._batch_size, self._chunk_size, self._preload_nchunks = batch_size, chunk_size, preload_nchunks

        self._build_runs(codes, chunk_size, categories)
        self._build_category_probs(category_weights)

    def _build_runs(self, codes: np.ndarray, chunk_size: int, categories: np.ndarray | None) -> None:
        """RLE the codes and precompute the per-category prefix-sum of chunk positions."""
        # contiguous runs of identical category code
        boundaries = np.flatnonzero(np.diff(codes)) + 1
        run_start = np.concatenate([np.array([0], dtype=np.int64), boundaries.astype(np.int64)])
        run_len = np.concatenate([boundaries, np.array([codes.shape[0]])]).astype(np.int64) - run_start
        run_cat = codes[run_start]

        # restrict to the requested subset of categories, if any
        if categories is not None:
            categories = np.asarray(categories)
            missing = np.setdiff1d(categories, np.unique(codes))
            if missing.size:
                raise ValueError(f"Requested categories {missing.tolist()} are not present in codes.")
            keep = np.isin(run_cat, categories)
            run_start, run_len, run_cat = run_start[keep], run_len[keep], run_cat[keep]
        if run_cat.size == 0:
            raise ValueError("No categories to sample from.")

        # run-length rule: every (selected) run must hold at least one full chunk.
        too_short = run_len < chunk_size
        if np.any(too_short):
            bad = np.unique(run_cat[too_short])
            raise ValueError(
                f"Every contiguous run must be at least chunk_size ({chunk_size}) observations long, "
                f"but {int(too_short.sum())} run(s) are shorter (categories {bad.tolist()}). "
                "Re-chunk the data so each category's fragments are large enough, or lower chunk_size."
            )
        n_pos = run_len - chunk_size + 1  # number of valid chunk-start positions in the run

        # group runs by category so each category owns a contiguous span of `cum`
        order = np.argsort(run_cat, kind="stable")
        run_start, run_cat, run_len, n_pos = run_start[order], run_cat[order], run_len[order], n_pos[order]

        cum = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(n_pos)])
        cat_ids, first = np.unique(run_cat, return_index=True)
        last = np.append(first[1:], len(run_cat))

        self._run_start = run_start
        self._cum = cum
        self._cat_ids = cat_ids
        self._cat_base = cum[first]  # offset into `cum` where each category begins
        self._cat_total = cum[last] - cum[first]  # # of valid chunk positions per category

    def _build_category_probs(self, category_weights: np.ndarray | None) -> None:
        if category_weights is None:
            weights = np.ones(self._cat_ids.shape, dtype=float)
        else:
            weights = np.asarray(category_weights, dtype=float)
            if weights.shape != self._cat_ids.shape:
                raise ValueError(
                    f"category_weights must align with categories (expected shape {self._cat_ids.shape}, "
                    f"got {weights.shape}). See the `categories` property for the expected order."
                )
        if np.any(weights < 0) or weights.sum() == 0:
            raise ValueError("category_weights must be non-negative and not all zero.")
        self._probs = weights / weights.sum()

    @property
    def categories(self) -> np.ndarray:
        """Category codes this sampler draws from, in the order ``category_weights`` expects."""
        return self._cat_ids

    @property
    def mask(self) -> slice:
        raise NotImplementedError(
            "mask property is not implemented for CategoricalSampler since it operates on a categorical column."
        )

    @mask.setter
    def mask(self, value: slice) -> None:
        raise NotImplementedError(
            "mask property is not implemented for CategoricalSampler since it operates on a categorical column."
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return True

    def n_iters(self, n_obs: int) -> int:
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

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        del n_obs  # nothing inferred from n_obs
        worker_info = get_torch_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError("Multiple workers are not supported with CategoricalSampler.")

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
        local_off = (self._rng.random(n_chunks) * self._cat_total[cat_of_draw]).astype(np.int64)
        global_off = self._cat_base[cat_of_draw] + local_off

        # 3) map the flat offset -> run -> absolute chunk start (the searchsorted trick,
        #    generalized across every category at once)
        run_idx = np.searchsorted(self._cum, global_off, side="right") - 1
        within = global_off - self._cum[run_idx]
        chunk_starts = self._run_start[run_idx] + within
        # NB: self._cat_ids[cat_of_draw] is the category label of each chunk, available for free.

        chunks = [slice(int(s), int(s + self._chunk_size)) for s in chunk_starts]
        if remainder > 0 and not self._drop_last:
            last = int(chunk_starts[-1])
            chunks[-1] = slice(last, last + remainder)
        return chunks
