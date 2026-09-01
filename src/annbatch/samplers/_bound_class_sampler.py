"""BoundClassSampler -- class schedule bound to an inner class sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from annbatch.samplers._class_sampler import _RunClassSampler
from annbatch.samplers._utils import (
    codes_of_categorical,
    grouped_weighted_choice,
    project_index,
    resolve_class_weights,
    to_level_arrays,
)

if TYPE_CHECKING:
    from annbatch.abc import BaseClassSampler


class BoundClassSampler(_RunClassSampler):
    """Bind a class sampler's per-batch class schedule onto another obs table.

    For every batch the inner sampler yields, this sampler yields one batch of its own
    ``batch_size`` observations of the matching class, drawn from ``classes_to_bind_on``
    (this sampler's own obs) and read as contiguous chunks. Classes are matched by *label*.

    When the inner balances over several columns (its categories are tuples) you can bind on
    a subset with ``on`` -- a ``dict`` mapping inner tuple positions to ``classes_to_bind_on``
    tuple positions (``{0: 0, 1: 1}``); ``None`` matches the whole label. ``classes_to_bind_on``
    (after ``on`` projection) must be a subset of the inner sampler's
    :attr:`~annbatch.abc.BaseClassSampler.vocab`.

    With no ``classes``, rows of the matched class are drawn uniformly. Passing a secondary
    ``classes`` (a single column, row-aligned with ``classes_to_bind_on``) plus ``class_weights``
    adds conditional sampling: which rows of the matched class are drawn is weighted by the
    secondary class (a non-positive weight excludes it), with ``class_weights`` one flat array
    in ``classes.categories`` order.

    ``batch_size`` must be a multiple of ``chunk_size``; the total is
    ``inner_sampler.n_batches() * batch_size`` (one full batch per inner batch). Every drawable
    run must be at least ``chunk_size`` long (stricter with a secondary ``classes``, since runs
    are then cut on the joint key). Multiple workers are not supported.

    Parameters
    ----------
    inner_sampler
        Any :class:`~annbatch.abc.BaseClassSampler` (typically a :class:`ClassSampler`) whose
        per-batch class schedule and batch count drive sampling.
    chunk_size, preload_nchunks, batch_size
        Sizing for this sampler; ``batch_size`` must be a multiple of ``chunk_size``.
    classes_to_bind_on
        A :class:`pandas.Categorical`, one entry per observation; matched (via ``on``) to the
        class the inner picks. Its (projected) categories must be a subset of the inner's
        :attr:`~annbatch.abc.BaseClassSampler.vocab`.
    on
        Optional ``dict`` mapping inner tuple positions to ``classes_to_bind_on`` positions.
        ``None`` matches the whole label.
    classes
        Optional secondary single-column :class:`pandas.Categorical`, the same length as
        ``classes_to_bind_on``, for conditional (within-class) sampling.
    class_weights
        Optional weights, one per ``classes.categories``; a non-positive weight excludes that class.
    mask, rng
        Optional observation range and random number generator (independent of the inner's).
    """

    _inner_sampler: BaseClassSampler
    _classes_to_bind_on: pd.Categorical

    def __init__(
        self,
        inner_sampler: BaseClassSampler,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        classes_to_bind_on: pd.Categorical,
        on: dict[int, int] | None = None,
        classes: pd.Categorical | None = None,
        class_weights: np.ndarray | None = None,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        if batch_size % chunk_size != 0:
            raise ValueError(
                "batch_size must be a multiple of chunk_size so each batch replays one inner class as whole chunks. "
                f"Got chunk_size={chunk_size}, batch_size={batch_size}."
            )
        # "outer" == classes_to_bind_on (this sampler's obs); outer_codes is the per-obs category code
        outer_codes = codes_of_categorical(classes_to_bind_on, "classes_to_bind_on")

        if on is None:
            inner_pos = outer_pos = None
        elif isinstance(on, dict):
            inner_pos, outer_pos = tuple(on.keys()), tuple(on.values())
        else:
            raise TypeError("on must be a dict[int, int] or None.")

        # keep only the bound positions of each label, e.g. vocab [(A, x), ...] -> [(A,), ...]
        inner_proj = project_index(inner_sampler.vocab, inner_pos)
        # factorize the projected outer categories into a shared class space both sides map into:
        # shared_classes = distinct projected labels, outer_to_shared = per outer category -> shared code.
        # e.g. categories [(A,x), (A,y), (B,x)] onto pos (0,) -> [A, A, B], so shared_classes [A, B] and
        # outer_to_shared [0, 0, 1].
        outer_to_shared, shared_classes = pd.factorize(project_index(classes_to_bind_on.categories, outer_pos))
        shared_obs_codes = outer_to_shared[outer_codes]  # per-obs shared code, e.g. outer_codes [0, 2, 1] -> [0, 1, 0]
        inner_to_shared = shared_classes.get_indexer(inner_proj)  # per inner category -> shared code

        # Only shared classes that actually occur in the obs matter for the subset/drawable rules:
        # factorizing over `.categories` also surfaces declared-but-unused categories (pandas keeps
        # these after subsetting), which must not trigger the checks.
        # the following is vectorized in case of many classes.
        present = np.zeros(len(shared_classes), dtype=bool)
        present[shared_obs_codes] = True
        present_codes = np.flatnonzero(present)

        # classes to bind on (its present classes) must be a subset of the inner sampler's classes
        present_classes = shared_classes[present_codes]
        not_in_inner = inner_proj.unique().get_indexer(present_classes) < 0
        if not_in_inner.any():
            raise ValueError(
                f"classes_to_bind_on has classes {list(present_classes[not_in_inner])} not present in the inner "
                "sampler's classes; classes_to_bind_on must be a subset of the inner sampler's classes."
            )
        # and every class the inner can emit must be present here, so it is drawable
        emittable_inner = inner_sampler.emittable_codes()
        emittable_inner_shared = inner_to_shared[emittable_inner]  # shared code of each emittable inner category
        drawable = np.isin(emittable_inner_shared, present_codes)  # np.isin treats the -1 "absent" code as not present
        if not drawable.all():
            missing = inner_proj[emittable_inner][~drawable].unique()
            raise ValueError(f"The inner sampler can emit classes {list(missing)} absent from classes_to_bind_on.")
        # the distinct shared classes the inner will ever ask this bound to produce
        emittable_shared = np.unique(emittable_inner_shared)

        self._inner_sampler = inner_sampler
        self._classes_to_bind_on = classes_to_bind_on
        self._on = on
        self._inner_to_shared = inner_to_shared  # per inner category -> shared code
        self._shared_classes = shared_classes  # projected shared class keys (for error messages)

        # build the joint (shared, secondary) codes and weights, and the per-joint-label MultiIndex
        codes, weights, labels = self._build_joint(
            shared_obs_codes, shared_classes, emittable_shared, classes, class_weights
        )
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            num_samples=inner_sampler.n_batches(0) * batch_size,
            drop_last=False,  # num_samples is a multiple of batch_size, so every batch is full
            mask=mask,
            rng=rng,
            codes=codes,
            weights=weights,
            category_labels=labels,
        )

    def _build_joint(
        self,
        shared_obs_codes: np.ndarray,
        shared_classes: pd.Index,
        emittable_shared: np.ndarray,
        classes: pd.Categorical | None,
        class_weights: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        # joint consists of the shared class (from classes_to_bind_on)
        # and the secondary class (from classes)
        # case 1: no secondary sampling
        if classes is None:
            if class_weights is not None:
                raise ValueError("class_weights was given but classes is None; pass a secondary `classes` too.")
            # identity mapping and uniform weighting
            self._joint_to_shared = np.arange(len(shared_classes))
            self._joint_weight = np.ones(len(shared_classes), dtype=float)
            drawable = np.isin(self._joint_to_shared, emittable_shared).astype(float)
            return shared_obs_codes, drawable, shared_classes

        sec_codes = codes_of_categorical(classes, "classes")
        if len(classes) != shared_obs_codes.shape[0]:
            raise ValueError(
                f"classes must be the same length as classes_to_bind_on ({shared_obs_codes.shape[0]}), got {len(classes)}."
            )
        n_sec = len(classes.categories)
        sec_weights = resolve_class_weights(class_weights, n_sec)

        # runs are cut on the joint (shared, secondary) key so each slice is coherent in both.
        joint_codes, joint_raw = pd.factorize(shared_obs_codes.astype(np.int64) * n_sec + sec_codes)
        j_shared = joint_raw // n_sec
        j_sec = joint_raw % n_sec
        self._joint_to_shared = j_shared
        self._joint_weight = sec_weights[j_sec]
        drawable = np.where(np.isin(j_shared, emittable_shared), sec_weights[j_sec], 0.0)

        # Flatten (shared, secondary) into one flat label per joint class -- when the shared key is itself a
        # tuple (a multi-column inner), keeping it nested would hide those columns from a downstream
        # `on`, so a chain could not condition on them. Flat labels compose: (cellline, drug) + batch
        # -> (cellline, drug, batch), which the next level can project any position of. Built column-wise
        # (take + from_arrays) so it stays vectorized.
        labels = pd.MultiIndex.from_arrays(
            to_level_arrays(shared_classes.take(j_shared)) + to_level_arrays(classes.categories.take(j_sec))
        )
        return joint_codes, drawable, labels

    @property
    def classes_to_bind_on(self) -> pd.Categorical:
        return self._classes_to_bind_on

    def _draw_class_of_slice(self, n_slices: int) -> np.ndarray:
        # the joint classes present here, with the shared class and secondary weight of each
        present_codes = self._per_class_sampling_info.index.to_numpy()
        present_shared = self._joint_to_shared[present_codes]
        present_weight = self._joint_weight[present_codes]

        # a mask can leave a class the inner still emits with no drawable run in the current range
        shared_of_batch = self._inner_to_shared[self._inner_sampler.batch_codes()]
        drawable = np.zeros(len(self._shared_classes), dtype=bool)
        drawable[present_shared] = True
        undrawable = shared_of_batch[~drawable[shared_of_batch]]
        if undrawable.size:
            raise ValueError(
                f"Class {self._shared_classes[undrawable[0]]!r} emitted by the inner sampler has no drawable "
                "run of at least chunk_size in the current range."
            )

        # pick a joint of each batch's shared class, weighted by the secondary -- one vectorized grouped draw
        positions = grouped_weighted_choice(present_shared, present_weight, shared_of_batch, self._class_rng)
        group_chunks = self._batch_size // self._chunk_size
        return np.repeat(positions, group_chunks)
