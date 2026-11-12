"""BoundClassSampler -- replay a class sampler's per-batch classes on a second annotation."""

from __future__ import annotations

import copy
import numbers
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.samplers._utils import (
    codes_without_na,
    resolve_class_weights,
    resolve_rng,
    validate_chunk_batch_preload_sizes,
)

from ._class_sampler import ClassSampler, _BatchClassPureSampler, _ClassRunSampler

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


def _label_columns(index: pd.Index) -> list[pd.Index]:
    """Split a (possibly flattened) MultiIndex into one array per level."""
    if not isinstance(index, pd.MultiIndex):
        if len(index) == 0 or not isinstance(index[0], tuple):
            return [index]
        index = pd.MultiIndex.from_tuples(index)
    return [index.get_level_values(i) for i in range(index.nlevels)]


class BoundClassSampler(_BatchClassPureSampler, _ClassRunSampler):
    """Replay a class sampler's per-batch classes on a second annotation.

    Batch ``i`` reads rows of a class matching inner batch ``i``'s class on the ``on`` columns, picked by
    ``class_weights``. Each pass replays a private copy of ``inner_sampler``, so this sampler
    stays paired, every pass, with any other copy of it iterated alongside, e.g. the one given to
    the primary :class:`~annbatch.Loader`. Bound samplers chain.

    Distributed training is not supported yet: neither this sampler nor ``inner_sampler`` can be a
    :class:`~annbatch.samplers.DistributedSampler`, and a primary wrapped in one does not pair.

    Every pass starts from its own seed, so pairing survives a primary pass that is cut short
    (a ``break``, ``limit_train_batches``) and pickling: restored copies start a fresh, paired pass
    rather than continuing the interrupted one, as long as both loaders use the same torch worker
    setup. Give the primary a generator of its own: changing its ``mask`` or ``rng`` after
    building this sampler, or letting another sampler draw from its generator, unpairs them.

    Parameters
    ----------
    chunk_size
        Number of observations in each slice yielded.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch; a multiple of ``chunk_size``.
    inner_sampler
        A :class:`~annbatch.samplers.ClassSampler` or :class:`BoundClassSampler`. Copied at construction.
    classes
        One class per observation, as for :class:`~annbatch.samplers.ClassSampler`. For multiple
        columns use ``pd.Categorical(multi_index.to_flat_index())``.
    on
        ``{inner position: own position}``: the label columns that must agree, e.g. ``{0: 0}``.
        Defaults to matching whole labels.
    class_weights
        Weights over ``classes.categories``, choosing among the classes that match a batch; a
        non-positive weight excludes that class. Defaults to uniform.
    mask
        Optional contiguous observation range to restrict sampling to.
    rng
        Random number generator for this sampler's own draws, copied so this sampler never advances
        the one you pass. The replayed classes come from the copy of ``inner_sampler`` and its generator.
    """

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        inner_sampler: ClassSampler | BoundClassSampler,
        classes: pd.Categorical,
        on: dict[int, int] | None = None,
        class_weights: np.ndarray | None = None,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        # TODO: support DistributedSampler through a public distributable interface that each sampler implements.
        # Here: distribute the inner like the primary, and optionally shard own rows per class (not if small).
        if not isinstance(inner_sampler, _BatchClassPureSampler):
            raise TypeError(
                f"inner_sampler must be a ClassSampler or BoundClassSampler, got {type(inner_sampler).__name__}."
            )
        self._inner = copy.deepcopy(inner_sampler)
        n_inner_batches = self._inner.n_batches(len(self._inner.classes))
        if n_inner_batches == 0:
            raise ValueError("inner_sampler yields no batches, so there is nothing to replay.")
        validate_chunk_batch_preload_sizes(chunk_size, preload_nchunks, batch_size)
        if batch_size % chunk_size != 0:
            raise ValueError(f"batch_size ({batch_size}) must be a multiple of chunk_size ({chunk_size}).")
        own_codes = codes_without_na(classes, "classes")
        inner_labels = self._inner.classes.categories
        emittable = self._inner._rle_manager.codes

        if on is None:
            inner_key, own_key = inner_labels, classes.categories
        else:
            if not isinstance(on, dict) or not all(
                isinstance(p, numbers.Integral) and not isinstance(p, bool) for p in (*on, *on.values())
            ):
                raise TypeError(f"on must be a dict of int label positions {{inner: own}}, got {on!r}.")
            inner_columns, own_columns = _label_columns(inner_labels), _label_columns(classes.categories)
            if (
                len(on) == 0
                or any(not 0 <= p < len(inner_columns) for p in on)
                or any(not 0 <= p < len(own_columns) for p in on.values())
            ):
                raise ValueError(
                    f"on must map positions in [0, {len(inner_columns)}) of the inner labels to positions in "
                    f"[0, {len(own_columns)}) of classes, got {on}."
                )
            inner_key = pd.MultiIndex.from_arrays([inner_columns[p] for p in on])
            own_key = pd.MultiIndex.from_arrays([own_columns[p] for p in on.values()])
        class_to_group, groups = pd.factorize(own_key)
        inner_to_group = groups.get_indexer(inner_key)
        if (inner_to_group[emittable] < 0).any():
            missing = inner_labels[emittable[inner_to_group[emittable] < 0]]
            raise ValueError(f"classes has no match for classes the inner sampler can emit: {list(missing)}.")

        class_weight = resolve_class_weights(class_weights, len(classes.categories))
        emittable_groups = np.unique(inner_to_group[emittable])
        has_weighted_rows = np.zeros(len(groups), dtype=bool)
        present = np.bincount(own_codes, minlength=len(classes.categories)) > 0
        has_weighted_rows[class_to_group[(class_weight > 0) & present]] = True
        if not has_weighted_rows[emittable_groups].all():
            missing = groups[emittable_groups[~has_weighted_rows[emittable_groups]]]
            raise ValueError(f"classes has no positive-weight rows matching emittable classes {list(missing)}.")

        super().__init__(
            chunk_size,
            preload_nchunks,
            batch_size,
            classes=classes,
            num_samples=n_inner_batches * batch_size,
            # classes matching nothing the inner emits are never read
            class_weights=np.where(np.isin(class_to_group, emittable_groups), class_weight, 0.0),
            mask=mask,
            drop_last=False,
            rng=copy.deepcopy(rng),
        )
        self._inner_to_group = inner_to_group
        self._emittable_groups = emittable_groups
        self._group_labels = groups
        self._class_to_group = class_to_group
        self._class_weight = class_weight
        self._check_mask_keeps_emittable_classes()

    @_ClassRunSampler.mask.setter
    def mask(self, value: slice) -> None:
        previous = self.mask
        _ClassRunSampler.mask.fset(self, value)
        try:
            self._check_mask_keeps_emittable_classes()
        except ValueError:
            _ClassRunSampler.mask.fset(self, previous)
            raise

    @property
    def rng(self) -> np.random.Generator | None:
        """The sampler's own copy of the generator it was given; assigning one also stores a copy."""
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator | None) -> None:
        self._rng = copy.deepcopy(resolve_rng(value))

    def _check_mask_keeps_emittable_classes(self) -> None:
        hidden = np.setdiff1d(self._emittable_groups, self._class_to_group[self._rle_manager.codes])
        if hidden.size > 0:
            raise ValueError(
                f"Classes {list(self._group_labels[hidden])} the inner sampler can emit have no run of at least "
                "chunk_size in the current mask."
            )

    def _replay_inner_pass(self) -> np.ndarray:
        """The class of every batch of one pass of the inner copy."""
        inner = self._inner
        inner_codes = inner.classes.codes
        classes_per_request = []
        for request in inner.sample(len(inner_codes)):
            chunks = request["requests"]
            starts = np.fromiter((s.start for s in chunks), dtype=np.int64, count=len(chunks))
            chunk_ends = np.cumsum(np.fromiter((s.stop - s.start for s in chunks), dtype=np.int64, count=len(chunks)))
            # split j holds rows j * batch_size ..., so that row's chunk gives batch j's class
            first_rows = np.arange(len(request["splits"])) * inner.batch_size
            classes_per_request.append(inner_codes[starts[np.searchsorted(chunk_ends, first_rows, side="right")]])
        return np.concatenate(classes_per_request)

    def _iter_requests(self, rng: np.random.Generator) -> Iterator[LoadRequest]:
        drawable = self._rle_manager.codes
        drawable_group = self._class_to_group[drawable]
        group_of_batch = self._inner_to_group[self._replay_inner_pass()]

        # weighted pick among each batch's matching classes: with weights normalized per group,
        # the cumulative weights of the k-th group span (k, k + 1]
        by_group = np.argsort(drawable_group, kind="stable")
        present_groups, group_start = np.unique(drawable_group[by_group], return_index=True)
        group_end = np.append(group_start[1:], by_group.size)
        segment_of_code = np.repeat(np.arange(present_groups.size), group_end - group_start)
        sorted_weight = self._class_weight[drawable[by_group]]
        cum_weight = np.cumsum(sorted_weight / np.add.reduceat(sorted_weight, group_start)[segment_of_code])
        segment_of_batch = np.searchsorted(present_groups, group_of_batch)
        picked = np.clip(
            np.searchsorted(cum_weight, segment_of_batch + rng.random(segment_of_batch.size), side="right"),
            group_start[segment_of_batch],
            group_end[segment_of_batch] - 1,
        )

        slices = self._rle_manager.slices_from_classes(
            np.repeat(drawable[by_group[picked]], self._batch_size // self._chunk_size), rng
        )
        yield from self._shuffled_batches(slices, rng)
