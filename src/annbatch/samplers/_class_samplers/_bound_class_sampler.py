"""BoundClassSampler -- replay another class sampler's per-batch schedule on a second column."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from annbatch.samplers._utils import codes_of_categorical

from ._class_sampler import ClassSampler, _ScheduledClassSampler


def _as_multiindex(index: pd.Index) -> pd.MultiIndex | None:
    if isinstance(index, pd.MultiIndex):
        return index
    if len(index) > 0 and isinstance(index[0], tuple):
        return pd.MultiIndex.from_tuples(index)
    return None


def to_level_arrays(index: pd.Index) -> list[pd.Index]:
    """Split a (possibly flattened) MultiIndex into one array per level."""
    mi = _as_multiindex(index)
    if mi is None:
        return [index]
    return [mi.get_level_values(i) for i in range(mi.nlevels)]


def project_index(labels: pd.Index, positions: tuple[int, ...] | None) -> pd.Index:
    """Keep only ``positions`` of each (possibly flattened) MultiIndex label."""
    if positions is None:
        return labels
    mi = _as_multiindex(labels)
    if mi is None:
        if tuple(positions) != (0,):
            raise ValueError(
                f"Cannot project single-column labels onto positions {positions}; a single column has only position 0."
            )
        return labels
    if len(positions) == 1:
        return mi.get_level_values(positions[0])
    return pd.MultiIndex.from_arrays([mi.get_level_values(p) for p in positions])


def grouped_weighted_choice(
    group_of_item: np.ndarray,
    weight_of_item: np.ndarray,
    group_of_draw: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one item per entry of ``group_of_draw``, weighted, from within that entry's group.

    Returns positions into ``group_of_item``. Every value of ``group_of_draw`` must appear in
    ``group_of_item``; the caller checks that, since it can say what a missing group means.
    """
    order = np.argsort(group_of_item, kind="stable")
    group, group_start = np.unique(group_of_item[order], return_index=True)
    group_end = np.append(group_start[1:], order.shape[0])
    cum_weight = np.cumsum(weight_of_item[order])
    hi = cum_weight[group_end - 1]
    lo = np.concatenate(([0.0], hi[:-1]))

    g = np.searchsorted(group, group_of_draw)
    target = lo[g] + rng.random(group_of_draw.shape[0]) * (hi[g] - lo[g])
    hit = np.clip(np.searchsorted(cum_weight, target, side="right"), group_start[g], group_end[g] - 1)
    return order[hit]


class BoundClassSampler(_ScheduledClassSampler):
    """Replay another class sampler's per-batch class schedule against a second annotation.

    The ``inner_sampler`` decides the class of each batch; this sampler then draws that
    batch's observations from ``classes_to_bind_on``, so one pass reads two annotations in step
    with each other. A bound sampler can itself serve as another's ``inner_sampler``, so these
    chain.

    ``batch_size`` must be a multiple of ``chunk_size``: the replayed class of a batch is read
    as whole chunks, and a chunk shared by two batches could not serve two different inner
    classes.

    .. important::
        The inner sampler is taken by value, not by reference: it is copied at construction, so
        nothing you do to your own instance afterwards reaches this one.

        ``mask`` is *not* forwarded: the inner sampler supplies a class sequence, not rows, and
        need not even span the same observations. If it schedules a class that has no run left
        inside your range, the pass raises rather than substituting another.

        Build the inner with ``drop_last=True`` unless its ``batch_size`` divides its
        ``num_samples``: a trailing short batch of the inner's is replayed here as a *full*
        batch, so the pass yields the same number of batches but more rows.

    Parameters
    ----------
    inner_sampler
        The sampler whose per-batch classes are replayed: a
        :class:`~annbatch.samplers.ClassSampler` or another
        :class:`~annbatch.samplers.BoundClassSampler`. A
        :class:`~annbatch.samplers.WeightedClassSampler` mixes classes within a batch and so
        has no per-batch class to replay.
    chunk_size
        Number of observations in each slice yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch; must be a multiple of ``chunk_size``.
    classes_to_bind_on
        A :class:`pandas.Categorical` with one entry per observation, whose classes must
        be a subset of the inner sampler's. Multi-column labels can be built with
        ``pd.Categorical(pd.MultiIndex.from_arrays(...).to_flat_index())``.
    on
        Positions *within the inner sampler's own labels* to match on, for an inner built on
        multi-column labels. ``on=(0,)`` matches ``classes_to_bind_on`` against the first
        column of each inner label. Give one position per column of ``classes_to_bind_on``,
        whose labels must then be a subset of the *projected* inner labels. Defaults to
        matching whole labels.
    classes
        Optional second annotation subdividing each bound class, so a batch is coherent on
        the bound class *and* drawn from one of its subclasses.
    class_weights
        Weights over ``classes.categories``, controlling how often each subclass is drawn
        within its bound class. A non-positive weight excludes that subclass.
    mask
        Optional contiguous observation range to restrict sampling to.
    rng
        Random number generator for everything this sampler draws: the schedule it replays,
        and the run and start within each replayed class. Passing it here is the same as
        assigning :attr:`rng` afterwards.
    """

    _inner_sampler: ClassSampler | BoundClassSampler
    _classes_to_bind_on: pd.Categorical

    def __init__(
        self,
        inner_sampler: ClassSampler | BoundClassSampler,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        classes_to_bind_on: pd.Categorical,
        on: tuple[int, ...] | None = None,
        classes: pd.Categorical | None = None,
        class_weights: np.ndarray | None = None,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        if not isinstance(inner_sampler, ClassSampler | BoundClassSampler):
            raise TypeError(
                f"inner_sampler must be a ClassSampler or a BoundClassSampler, got "
                f"{type(inner_sampler).__name__}, which has no per-batch class to replay."
            )
        # Unlike the other class samplers, this one replays one class per *batch* as whole
        # chunks, so a batch that is not a whole number of chunks cannot be replayed: a shared
        # chunk would need two classes at once.
        if batch_size % chunk_size != 0:
            raise ValueError(
                "batch_size must be a multiple of chunk_size so each batch replays one inner class as whole chunks. "
                f"Got chunk_size={chunk_size}, batch_size={batch_size}."
            )
        outer_codes = codes_of_categorical(classes_to_bind_on, "classes_to_bind_on")

        if on is not None:
            try:
                on = tuple(int(position) for position in on)
            except TypeError:
                raise TypeError(f"on must be a tuple of label positions or None, got {type(on).__name__}.") from None
            n_inner = len(to_level_arrays(inner_sampler.vocab))
            if not on or any(not 0 <= position < n_inner for position in on):
                raise ValueError(
                    f"on must be a non-empty tuple of positions in [0, {n_inner}), since the inner sampler's "
                    f"labels have {n_inner} column(s). Got {on}."
                )
            n_bound = len(to_level_arrays(classes_to_bind_on.categories))
            if len(on) != n_bound:
                raise ValueError(
                    f"on selects {len(on)} of the inner sampler's label columns but classes_to_bind_on labels "
                    f"have {n_bound}; they must match to be compared. Got on={on}."
                )

        # a Categorical's categories are unique, so outer_codes already index them directly
        bound_classes = classes_to_bind_on.categories
        inner_proj = project_index(inner_sampler.vocab, on)
        inner_to_bound = bound_classes.get_indexer(inner_proj)

        # scatter rather than np.unique: outer_codes is n_obs long, so this stays O(n_obs)
        present = np.zeros(len(bound_classes), dtype=bool)
        present[outer_codes] = True
        present_codes = np.flatnonzero(present)

        present_classes = bound_classes[present_codes]
        not_in_inner = inner_proj.unique().get_indexer(present_classes) < 0
        if not_in_inner.any():
            raise ValueError(
                f"classes_to_bind_on has classes {list(present_classes[not_in_inner])} not present in the inner "
                "sampler's classes; classes_to_bind_on must be a subset of the inner sampler's classes."
            )
        emittable_inner_bound = inner_to_bound[inner_sampler.emittable_codes]
        drawable = np.isin(emittable_inner_bound, present_codes)
        if not drawable.all():
            missing = inner_proj[inner_sampler.emittable_codes][~drawable].unique()
            raise ValueError(f"The inner sampler can emit classes {list(missing)} absent from classes_to_bind_on.")
        emittable_bound = np.unique(emittable_inner_bound)

        self._inner_sampler = copy.deepcopy(inner_sampler)
        self._classes_to_bind_on = classes_to_bind_on
        self._on = on
        self._inner_to_bound = inner_to_bound

        joint_classes, weights = self._build_joint(emittable_bound, classes, class_weights)
        super().__init__(
            chunk_size,
            preload_nchunks,
            batch_size,
            classes=joint_classes,
            num_samples=inner_sampler.n_batches(0) * batch_size,
            class_weights=weights,
            mask=mask,
            drop_last=False,
            rng=rng,
        )
        # One generator, not two: the base constructor assigns `_rng` directly, so go through
        # the setter to hand the same one to the copy. Otherwise the copy keeps the generator it
        # was deep-copied with and `rng=` would silently steer only the rows, not the schedule.
        self.rng = self._rng

    def _build_joint(
        self,
        emittable_bound: np.ndarray,
        classes: pd.Categorical | None,
        class_weights: np.ndarray | None,
    ) -> tuple[pd.Categorical, np.ndarray]:
        """The categorical this sampler draws from: the bound class, optionally times a subclass."""
        bound_classes = self._classes_to_bind_on.categories
        outer_codes = self._classes_to_bind_on.codes
        if classes is None:
            if class_weights is not None:
                raise ValueError("class_weights was given but classes is None; pass a secondary `classes` too.")
            self._joint_to_bound = np.arange(len(bound_classes))
            self._joint_weight = np.ones(len(bound_classes), dtype=float)
            joint = pd.Categorical.from_codes(outer_codes, categories=bound_classes.to_flat_index())
            return joint, self._drawable_weights(emittable_bound)

        sec_codes = codes_of_categorical(classes, "classes")
        if len(classes) != outer_codes.shape[0]:
            raise ValueError(
                f"classes must be the same length as classes_to_bind_on ({outer_codes.shape[0]}), got {len(classes)}."
            )
        n_sec = len(classes.categories)
        if hasattr(class_weights, "index"):
            raise TypeError(
                "class_weights must be an array, not a pandas Series: a Series would be read in positional "
                "order and its index ignored. Pass class_weights.reindex(classes.categories).to_numpy()."
            )
        sec_weights = np.ones(n_sec) if class_weights is None else np.asarray(class_weights, dtype=float)
        if sec_weights.shape != (n_sec,):
            raise ValueError(
                f"class_weights must have one weight per class in classes.categories "
                f"(expected shape ({n_sec},), got {sec_weights.shape})."
            )
        if not (sec_weights > 0).any():
            raise ValueError("class_weights must have at least one positive weight.")

        joint_codes, joint_raw = pd.factorize(outer_codes.astype(np.int64) * n_sec + sec_codes)
        j_bound, j_sec = joint_raw // n_sec, joint_raw % n_sec
        self._joint_to_bound = j_bound
        self._joint_weight = sec_weights[j_sec]

        labels = pd.MultiIndex.from_arrays(
            to_level_arrays(bound_classes.take(j_bound)) + to_level_arrays(classes.categories.take(j_sec))
        )
        joint = pd.Categorical.from_codes(joint_codes, categories=labels.to_flat_index())
        return joint, self._drawable_weights(emittable_bound)

    def _drawable_weights(self, emittable_bound: np.ndarray) -> np.ndarray:
        """Each joint class's weight, zeroed where its bound class is not one the inner can emit."""
        return np.where(np.isin(self._joint_to_bound, emittable_bound), self._joint_weight, 0.0)

    @property
    def rng(self) -> np.random.Generator | None:
        """The random number generator used by this sampler and by its copy of the inner one."""
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator | None) -> None:
        # One generator for both: the copy draws the schedule, we draw the runs and starts from
        # it afterwards. Sharing it is what makes a re-assignment change the schedule too.
        self._rng = value
        self._inner_sampler.rng = value

    @property
    def classes_to_bind_on(self) -> pd.Categorical:
        """The annotation this sampler draws its batches from."""
        return self._classes_to_bind_on

    def batch_schedule(self) -> np.ndarray:
        drawable_codes = self.emittable_codes
        drawable_bound = self._joint_to_bound[drawable_codes]
        drawable_weight = self._joint_weight[drawable_codes]

        # only the batches the inner sampler actually yields; with drop_last its schedule holds
        # one more slot, for the short batch it discards.
        inner_of_batch = self._inner_sampler.batch_schedule()[: self._inner_sampler.n_batches(0)]
        # Construction proved every emittable inner class maps to a real bound code, and the copy
        # is private, so no -1 from get_indexer can reach here.
        bound_of_batch = self._inner_to_bound[inner_of_batch]
        # grouped_weighted_choice would silently draw from the wrong group for a class that has
        # no drawable run at all, so check before rather than after.
        undrawable = bound_of_batch[~np.isin(bound_of_batch, drawable_bound)]
        if undrawable.size:
            raise ValueError(
                f"Class {self._classes_to_bind_on.categories[undrawable[0]]!r} emitted by the inner sampler has no drawable "
                "run of at least chunk_size in the current range."
            )
        return drawable_codes[grouped_weighted_choice(drawable_bound, drawable_weight, bound_of_batch, self._rng)]
