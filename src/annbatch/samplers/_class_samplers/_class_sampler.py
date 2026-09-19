"""ClassSampler -- class-based chunk sampler."""

from __future__ import annotations

import itertools
import math
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._utils import (
    check_lt_1,
    codes_of_categorical,
    get_torch_worker_info,
    validate_chunk_batch_preload_sizes,
)
from annbatch.utils import split_given_size

from ._rle_manager import RLEManager

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from annbatch.types import LoadRequest


class _RunClassSampler(Sampler):
    """Shared machinery for samplers that read whole same-class chunks out of an RLE.

    Subclasses decide two things: which class each chunk is read from (:meth:`_chunk_schedule`)
    and how a loaded window is cut into batches (:meth:`_window_splits`). The constructor, the
    remainder handling and the window loop are the same for all of them and live here.
    """

    _batch_size: int
    _chunk_size: int
    _preload_nchunks: int
    _num_samples: int
    _n_obs: int
    _rng: np.random.Generator  # never None: the constructor defaults it
    _drop_last: bool
    _classes: pd.Categorical
    _rle_manager: RLEManager

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
        check_lt_1([num_samples], ["num_samples"])
        validate_chunk_batch_preload_sizes(chunk_size, preload_nchunks, batch_size)
        self._n_obs = len(codes_of_categorical(classes, "classes"))

        self._batch_size, self._chunk_size, self._preload_nchunks = batch_size, chunk_size, preload_nchunks
        self._num_samples = num_samples
        self._drop_last = drop_last
        self._rng = rng or np.random.default_rng()
        self._classes = classes
        # classes and their weights are mask-independent; the RLE keeps them so any mask can
        # renormalize from them. It is handed no rng: `self._rng` is read per draw, so that
        # re-assigning it (as DistributedSampler does per rank) reaches the slice starts too.
        self._rle_manager = RLEManager(
            mask=slice(0, None) if mask is None else mask,
            classes=classes,
            weights=class_weights,
            chunk_size=chunk_size,
        )

    @property
    def mask(self) -> slice:
        return self._rle_manager.mask

    @mask.setter
    def mask(self, value: slice) -> None:
        # resolve + eagerly rebuild so range errors (run-length, no active class) surface on assignment
        self._rle_manager.mask = value

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return True

    @property
    def _n_batch_slots(self) -> int:
        """Batches one pass cuts, *before* ``drop_last`` discards a trailing short one."""
        return math.ceil(self._num_samples / self._batch_size)

    def n_batches(self, n_obs: int) -> int:
        del n_obs  # determined by num_samples, not the loader size
        if self._drop_last:
            return self._num_samples // self._batch_size
        return self._n_batch_slots

    def validate(self, n_obs: int) -> None:
        """Validate that the codes describe exactly the loader's observations."""
        if n_obs != self._n_obs:
            raise ValueError(
                f"classes length ({self._n_obs}) does not match loader n_obs ({n_obs}). "
                "The classes column must describe exactly the loader's observations."
            )

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError(f"Multiple workers are not supported with {type(self).__name__}.")

        return self._iter_requests()

    # -- policy ------------------------------------------------------------------------

    @abstractmethod
    def _chunk_schedule(self, n_chunks: int) -> np.ndarray:
        """Class code into ``classes.categories`` for each of ``n_chunks`` chunks of a pass."""

    def _window_splits(self, ids: np.ndarray) -> list[np.ndarray]:
        """Cut a window's row ids into batches. Rows are shuffled within each batch."""
        splits = split_given_size(ids, self._batch_size)
        for batch in splits:
            # can't vectorize this because we need to return a list, not ndarray
            self._rng.shuffle(batch)
        return splits

    # -- one pass ----------------------------------------------------------------------

    def _slices_for_pass(self) -> list[slice]:
        """The ``chunk_size`` slices a whole pass reads, the last one short if needed."""
        n_chunks, remainder = divmod(self._num_samples, self._chunk_size)
        if remainder > 0:
            n_chunks += 1
        slices = self._rle_manager.slices_from_classes(self._chunk_schedule(n_chunks), self._rng)
        if remainder > 0:
            last = int(slices[-1].start)
            slices[-1] = slice(last, last + remainder)
        return slices

    def _iter_requests(self) -> Iterator[LoadRequest]:
        window_size = self._preload_nchunks * self._chunk_size
        full_ids = np.arange(window_size)
        for window in itertools.batched(self._slices_for_pass(), self._preload_nchunks):
            n_rows = (len(window) - 1) * self._chunk_size + (window[-1].stop - window[-1].start)
            splits = self._window_splits(full_ids if n_rows == window_size else np.arange(n_rows))
            if self._drop_last and splits[-1].size < self._batch_size:
                splits = splits[:-1]
                if not splits:
                    continue  # the whole window was one short batch; there is nothing to yield
            yield {"requests": list(window), "splits": splits}


class _ScheduledClassSampler(_RunClassSampler):
    """A class sampler that decides the class of every *batch* up front.

    :meth:`batch_schedule` draws one class code per batch of a pass, and each chunk is read
    from the class of the batch covering its rows. Being class-per-batch is what makes a
    sampler replayable, which is why :class:`~annbatch.samplers.BoundClassSampler` takes one.
    """

    @property
    def vocab(self) -> pd.Index:
        """The labels a schedule's class codes index into."""
        return self._classes.categories

    @property
    def emittable_codes(self) -> np.ndarray:
        """The class codes a schedule may contain, given the current mask and class weights."""
        return self._rle_manager.emittable_codes

    @abstractmethod
    def batch_schedule(self) -> np.ndarray:
        """Draw a class code into ``vocab`` for each batch of one pass.

        Returns ``ceil(num_samples / batch_size)`` of them, including the trailing short batch
        that ``drop_last`` would discard. Each call *draws*, advancing ``rng``.
        """

    def _chunk_schedule(self, n_chunks: int) -> np.ndarray:
        # a chunk is read from the class of the batch holding its first row
        return self.batch_schedule()[(np.arange(n_chunks) * self._chunk_size) // self._batch_size]


class ClassSampler(_ScheduledClassSampler):
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

    def batch_schedule(self) -> np.ndarray:
        # A class may only change every lcm(chunk_size, batch_size) rows -- the only boundary
        # where a chunk edge and a batch edge coincide -- i.e. every `slots_per_group` batches.
        slots_per_group = self._chunk_size // math.gcd(self._chunk_size, self._batch_size)
        n_slots = self._n_batch_slots
        groups = self._rng.choice(
            self._rle_manager.emittable_codes, size=math.ceil(n_slots / slots_per_group), p=self._rle_manager.weights
        )
        return np.repeat(groups, slots_per_group)[:n_slots]
