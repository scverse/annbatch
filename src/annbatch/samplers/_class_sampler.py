"""ClassSampler -- class-based chunk sampler."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.abc import BaseClassSampler
from annbatch.samplers._utils import (
    build_run_table,
    check_lt_1,
    codes_of_categorical,
    get_torch_worker_info,
    iter_windows,
    resolve_class_weights,
    validate_chunk_batch_preload_sizes,
    validate_mask_n_obs_and_resolve,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from annbatch.types import LoadRequest


class _RunClassSampler(BaseClassSampler):
    """Shared run-length/slice machinery for class-coherent samplers.

    Every subclass reduces its problem to one integer ``_codes`` array over ``_category_labels``.
    Because the runs, the emittable set, the per-batch schedule, and :attr:`vocab`
    all live in that one code space, the whole :class:`~annbatch.abc.BaseClassSampler`
    contract is satisfied here -- so any subclass can be
    bound onto, matched by the labels in :attr:`vocab`.
    """

    _batch_size: int
    _chunk_size: int
    _preload_nchunks: int
    _num_samples: int
    _n_obs: int
    _rng: np.random.Generator
    _class_rng: np.random.Generator
    _split_rng: np.random.Generator
    _drop_last: bool
    _mask: slice
    _codes: np.ndarray
    _weights: np.ndarray
    _class_runs: pd.DataFrame
    _per_class_sampling_info: pd.DataFrame

    def __init__(
        self,
        *,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        num_samples: int,
        drop_last: bool,
        mask: slice | None,
        rng: np.random.Generator | None,
        codes: np.ndarray,
        weights: np.ndarray,
        category_labels: Sequence,
    ):
        check_lt_1([num_samples], ["num_samples"])
        validate_chunk_batch_preload_sizes(chunk_size, preload_nchunks, batch_size)

        self._batch_size, self._chunk_size, self._preload_nchunks = batch_size, chunk_size, preload_nchunks
        self._num_samples = num_samples
        self._drop_last = drop_last
        self._rng = rng or np.random.default_rng()
        self._spawn_class_split_rngs()  # derive independent class-choice and slice/split streams
        self._codes = codes  # pd.Categorical.codes / pd.factorize output are already ndarrays
        self._weights = weights  # full array (<=0 for excluded); codes are 0..N-1 so direct indexing works
        self._category_labels = category_labels
        self._n_obs = int(self._codes.shape[0])

        if mask is None:
            mask = slice(0, None)
        start, stop = validate_mask_n_obs_and_resolve(mask, self._n_obs)
        self._mask = slice(start, stop)

        # eager build for the (default or constructor) range so run-length errors surface early
        self._built_range: tuple[int, int] | None = None
        self._ensure_runs(self._n_obs)

    def _spawn_class_split_rngs(self) -> None:
        # independent streams (numpy sequence-of-seeds pattern): class choice vs slice/shuffle,
        # so the class schedule is reproducible regardless of the split randomness and vice versa
        root = self._rng.integers(np.iinfo(np.int64).max)
        self._class_rng = np.random.default_rng([0, root])
        self._split_rng = np.random.default_rng([1, root])

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._rng = value
        self._spawn_class_split_rngs()

    @property
    def mask(self) -> slice:
        return self._mask

    @mask.setter
    def mask(self, value: slice) -> None:
        # resolve + eagerly rebuild so range errors (run-length, no active class) surface on assignment
        start, stop = validate_mask_n_obs_and_resolve(value, self._n_obs)
        self._mask = slice(start, stop)
        self._ensure_runs(self._n_obs)

    def _ensure_runs(self, n_obs: int) -> None:
        start, stop = validate_mask_n_obs_and_resolve(self._mask, n_obs)
        if self._built_range == (start, stop):
            return

        # Per-run table: start/end in global coordinates, length, and class
        runs = build_run_table(self._codes[start:stop], start)

        # keep only runs of non-excluded classes; excluded (weight <=0) runs are exempt from every check
        runs = runs.loc[self._weights[runs["cat"].to_numpy()] > 0].reset_index(drop=True)
        if runs.empty:
            raise ValueError(
                "No class with positive weight is present in the current mask range "
                f"[{start}, {stop}); its renormalized weights would sum to zero."
            )

        # run-length rule: every kept run must hold at least one full chunk
        too_short_mask = runs["len"].to_numpy() < self._chunk_size
        if np.any(too_short_mask):
            bad = np.unique(runs.loc[too_short_mask, "cat"].to_numpy())
            bad_labels = pd.Index(self._category_labels)[bad].tolist()
            raise ValueError(
                f"Every contiguous run must be at least chunk_size ({self._chunk_size}) observations long, "
                f"but {int(too_short_mask.sum())} run(s) are shorter (classes {bad_labels}). "
                "Re-chunk the data so each class's runs are large enough, lower chunk_size, "
                "or exclude these classes with a zero weight."
            )

        # Sort runs by class so each class's runs are contiguous in the table;
        # `first_row_in_runs_of_class` then indexes directly into the sorted run table.
        self._class_runs = runs.sort_values("cat", kind="stable").reset_index(drop=True)

        # Per-class table: probability, number of runs, and offset into the sorted run table
        classes_to_sample, n_runs_per_class = np.unique(self._class_runs["cat"].to_numpy(), return_counts=True)
        w = self._weights[classes_to_sample]
        self._per_class_sampling_info = pd.DataFrame(
            {
                "prob": w / w.sum(),
                "n_runs": n_runs_per_class.astype(np.int64),
                "first_row_in_runs_of_class": np.concatenate(([0], np.cumsum(n_runs_per_class[:-1]))).astype(np.int64),
            },
            index=pd.Index(classes_to_sample, name="cat"),
        )

        self._built_range = (start, stop)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def shuffle(self) -> bool:
        return True

    def n_batches(self, n_obs: int) -> int:
        del n_obs  # determined by num_samples, not the loader size
        if self._drop_last:
            return self._num_samples // self._batch_size
        return math.ceil(self._num_samples / self._batch_size)

    def validate(self, n_obs: int) -> None:
        if n_obs != self._n_obs:
            raise ValueError(
                f"This sampler describes {self._n_obs} observations, which does not match loader n_obs ({n_obs}). "
                "The class labels must describe exactly the loader's observations."
            )
        self._ensure_runs(n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError(f"Multiple workers are not supported with {type(self).__name__}.")

        self._ensure_runs(n_obs)
        return self._iter_requests()

    @property
    def vocab(self) -> pd.Index:
        # return as-is (a MultiIndex stays a MultiIndex -- pd.Index() would flatten it to tuples),
        # so a downstream `on` projects it column-wise without a from_tuples round-trip
        labels = self._category_labels
        return labels if isinstance(labels, pd.Index) else pd.Index(labels)

    def emittable_codes(self) -> np.ndarray:
        return self._per_class_sampling_info.index.to_numpy()

    def batch_codes(self) -> np.ndarray:
        # Per-batch category code for a full pass -- the exact classes sample() would draw, from the class
        # schedule alone (no slice/window building). _draw_class_of_slice is the first (and only class-picking)
        # rng use in _iter_requests, so this matches a real pass's per-batch class exactly.
        n_slices = math.ceil(self._num_samples / self._chunk_size)
        slice_codes = self._per_class_sampling_info.index.to_numpy()[self._draw_class_of_slice(n_slices)]
        # batch j occupies rows [j*batch_size, (j+1)*batch_size); coherent, so its class is that of
        # slice (j*batch_size)//chunk_size
        return slice_codes[(np.arange(self.n_batches(0)) * self._batch_size) // self._chunk_size]

    @abstractmethod
    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:
        # return n_slices row indices into _per_class_sampling_info: the class of each slice
        ...

    def _iter_requests(self) -> Iterator[LoadRequest]:
        n_slices, remainder = divmod(self._num_samples, self._chunk_size)
        if remainder > 0:
            n_slices += 1

        class_of_slice = self._draw_class_of_slice(n_slices)

        # Sample one of the possible run positions within a class i.e.,
        # [a: slice(0, 10), b: slice(10, 20), a: slice(20, 30)]
        # would have two possible run positions for a (one of 0 and 2) and one for b (just 1)
        class_n_runs = self._per_class_sampling_info["n_runs"].to_numpy()
        possible_run_pos_within_a_class = self._split_rng.integers(class_n_runs[class_of_slice])
        # Generate a position into the runs table to get the run to fetch within
        first_row_of_class = self._per_class_sampling_info["first_row_in_runs_of_class"].to_numpy()
        chosen = first_row_of_class[class_of_slice] + possible_run_pos_within_a_class
        run_starts = self._class_runs["start"].to_numpy()[chosen]
        run_ends = self._class_runs["end"].to_numpy()[chosen]
        # Now sample a valid start position within each chunk
        slice_starts = self._split_rng.integers(run_starts, run_ends - self._chunk_size + 1)

        slices = [slice(int(s), int(s + self._chunk_size)) for s in slice_starts]
        if remainder > 0:
            last = int(slice_starts[-1])
            slices[-1] = slice(last, last + remainder)

        yield from iter_windows(
            slices,
            preload_nchunks=self._preload_nchunks,
            chunk_size=self._chunk_size,
            batch_size=self._batch_size,
            drop_last=self._drop_last,
            rng=self._split_rng,
        )


class ClassSampler(_RunClassSampler):
    """Sample class-coherent batches with replacement.

    Every batch the :class:`~annbatch.Loader` yields is drawn from a single class:
    a class is drawn ``c ~ Categorical(p)`` (``p`` proportional to
    ``class_weights``, uniform by default), then the batch's observations are drawn
    from ``c``. A load request may span several classes but no batch mixes them,
    which makes over- or under-sampling specific populations straightforward.

    Sampling is **with replacement** -- each pass draws ``num_samples`` observations
    rather than partitioning a fixed epoch -- so there is no notion of an epoch and the
    number of iterations is fixed. The only size requirement is that
    ``chunk_size * preload_nchunks`` is divisible by ``batch_size`` (already enforced by the
    loader).

    *Class selection.* A class with a non-positive weight is excluded: it is
    never sampled and its runs are exempt from the run-length rule below. Set a
    weight to ``0`` to drop a class; there is no separate exclusion argument.

    *Run-length rule.* Every contiguous run of a *non-excluded* class must span
    at least ``chunk_size`` observations; otherwise no aligned slice fits inside it
    and the sampler raises at construction, naming the offending classes by their
    label.

    *Mask.* Assigning :attr:`mask` restricts sampling to a contiguous observation
    range ``[start, stop)``. The RLE is rebuilt over that window (slice starts stay
    in global coordinates) and cached on the resolved ``(start, stop)`` pair, so
    reassigning the same mask is free. Class weights are renormalized from the
    original values over only the classes present in the new range; if no
    class with a positive weight remains, the assignment raises.

    Multiple workers are not supported with this sampler.

    Implementation
    --------------
    A run-length encoding (RLE) of ``classes.codes`` is built over the :attr:`mask`
    range. A class boundary may only fall where a chunk edge and a batch edge coincide,
    which happens every ``lcm(chunk_size, batch_size)`` rows; so classes are assigned per
    *group* of ``group_chunks = batch_size // gcd(chunk_size, batch_size)`` chunks (one ``lcm``
    block) -- one class ``c ~ Categorical(p)`` is drawn independently for each group and
    shared across its chunks. Each chunk is then a single-class on-disk read and each batch
    falls inside one group, hence one class. Drawing per *minimal* group packs as many
    classes into a window as coherence allows: up to ``preload_nchunks // group_chunks``
    distinct classes (equivalently ``preload_nchunks * chunk_size // lcm(chunk_size,
    batch_size)``). ``preload_nchunks`` is always a multiple of ``group_chunks`` because
    ``chunk_size * preload_nchunks`` is divisible by ``batch_size``, so groups tile each window.
    A uniform chunk-start within ``c`` is drawn per chunk (a prefix-sum lookup maps it to the
    absolute slice in *O(log n_runs)*), and rows within each batch are shuffled, so batches are
    class-coherent but not ordered. Memory scales with the number of runs
    (``<= n_obs // chunk_size``).

    Examples
    --------
    >>> from annbatch import Loader
    >>> from annbatch.samplers import ClassSampler
    >>> # Get categorical column from collection
    >>> classes = collection.obs(columns=["categories"])["categories"].values
    >>> sampler = ClassSampler(
    ...     chunk_size=10,
    ...     preload_nchunks=4,
    ...     batch_size=10,
    ...     classes=classes,
    ...     num_samples=1000,
    ... )
    >>> loader = Loader(batch_sampler=sampler).use_collection(collection)

    Parameters
    ----------
    chunk_size
        Number of observations in each slice yielded. Also the minimum run length
        required of every non-excluded class (see the run-length rule).
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch. ``chunk_size * preload_nchunks`` must be divisible
        by it; it need not divide or be a multiple of ``chunk_size``.
    classes
        A :class:`pandas.Categorical` with one entry per observation, e.g.
        ``df["cell_type"].values`` when the column already has a categorical dtype.
        If loading categories from a :class:`~annbatch.DatasetCollection`, they can be retrieved via
        ``collection.obs(columns=["cell_type"])["cell_type"].values`` (if the column was stored with categorical dtype)
        or converted using ``pd.Categorical(collection.obs(columns=["label"])["label"])`` (if stored as integers or strings).
        Length must equal the loader's ``n_obs``. The obs axis need not be contiguous per class, but
        every run of a non-excluded class must be at least ``chunk_size`` long
        (see the run-length rule above). NA values (``codes == -1``) are not allowed.
    num_samples
        Total number of observations to draw.
    class_weights
        Optional weights, one per class in ``classes.categories``
        (so ``len(class_weights) == len(classes.categories)``), controlling
        how often each class is drawn. A non-positive weight excludes that class
        entirely. When ``None`` (the default) every class is drawn uniformly. For
        proportional (≈ plain global random) sampling pass each class's observation
        count. The weights are kept and, whenever a mask narrows the range, the weights
        of the classes still present are renormalized.
    mask
        Optional contiguous observation range to restrict sampling to. Defaults to
        the whole dataset.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator. Note that :func:`torch.manual_seed` has no effect
        here; pass a seeded :class:`numpy.random.Generator` to control randomness.
    """

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        classes: pd.Categorical,
        num_samples: int,
        class_weights: np.ndarray | None = None,
        mask: slice | None = None,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        codes = codes_of_categorical(classes, "classes")
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            num_samples=num_samples,
            drop_last=drop_last,
            mask=mask,
            rng=rng,
            codes=codes,
            weights=resolve_class_weights(class_weights, len(classes.categories)),
            category_labels=classes.categories,
        )

    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:
        # one class per lcm(chunk_size, batch_size) group, repeated across the group's chunks
        group_chunks = self._batch_size // math.gcd(self._chunk_size, self._batch_size)
        n_groups = math.ceil(n_slices / group_chunks)
        group_classes = self._class_rng.choice(
            len(self._per_class_sampling_info), size=n_groups, p=self._per_class_sampling_info["prob"].to_numpy()
        )
        return np.repeat(group_classes, group_chunks)[:n_slices]
