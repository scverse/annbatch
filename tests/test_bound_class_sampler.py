"""Tests for BoundClassSampler.

The sampler replays an inner :class:`~annbatch.samplers.ClassSampler`'s per-batch class
schedule onto its own observations: every batch is class-coherent and full, the class of
each batch matches the inner sampler's corresponding batch (even across different category
orderings and obs lengths), an optional per-class weight controls which rows are drawn
within each matched class, and the whole thing is reproducible and picklable (both RNGs
round-trip).
"""

from __future__ import annotations

import copy
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from annbatch.abc import BaseClassSampler
from annbatch.samplers import BoundClassSampler, ClassSampler
from annbatch.samplers._utils import WorkerInfo, grouped_weighted_choice, project_index


def make_inner(
    labels,
    *,
    chunk_size: int = 10,
    preload_nchunks: int = 4,
    batch_size: int = 10,
    num_samples: int = 1000,
    drop_last: bool = True,
    seed: int = 0,
    **kwargs,
) -> ClassSampler:
    classes = labels if isinstance(labels, pd.Categorical) else pd.Categorical(labels)
    return ClassSampler(
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        classes=classes,
        num_samples=num_samples,
        drop_last=drop_last,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def make_bound(
    inner: ClassSampler,
    classes_to_bind_on,
    *,
    chunk_size: int = 10,
    preload_nchunks: int = 4,
    batch_size: int = 10,
    seed: int = 1,
    **kwargs,
) -> BoundClassSampler:
    bind = classes_to_bind_on if isinstance(classes_to_bind_on, pd.Categorical) else pd.Categorical(classes_to_bind_on)
    return BoundClassSampler(
        inner,
        chunk_size,
        preload_nchunks,
        batch_size,
        classes_to_bind_on=bind,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def window_indices(load_request) -> np.ndarray:
    """Observation indices of a load request's window, in yield order."""
    return np.concatenate([np.arange(s.start, s.stop) for s in load_request["requests"]])


def batch_obs_indices(sampler: BaseClassSampler, n_obs: int):
    """Yield the observation indices of each yielded batch (one array per batch)."""
    for load_request in sampler.sample(n_obs):
        window = window_indices(load_request)
        for split in load_request["splits"]:
            yield window[split]


def batch_infos(sampler: BoundClassSampler, condition_codes: np.ndarray, n_obs: int) -> list[tuple[int, int, int]]:
    """For each yielded batch: ``(class_code, size, n_unique_classes)``."""
    infos = []
    for idx in batch_obs_indices(sampler, n_obs):
        unique = np.unique(condition_codes[idx])
        infos.append((int(unique[0]), int(idx.size), int(unique.size)))
    return infos


def batch_keys(sampler: BaseClassSampler, key_of_obs, n_obs: int) -> list:
    """Per-batch key from ``key_of_obs``, asserting each batch is coherent on it."""
    key = np.asarray(key_of_obs)
    keys = []
    for idx in batch_obs_indices(sampler, n_obs):
        unique = set(key[idx].tolist())
        assert len(unique) == 1, "each batch must be coherent on the bound key"
        keys.append(unique.pop())
    return keys


def inner_batch_labels(inner: ClassSampler) -> list:
    """The class *label* of each batch a full pass of the inner sampler yields."""
    return list(inner.vocab[inner.batch_codes()])


# =============================================================================
# Construction / validation
# =============================================================================


@pytest.mark.parametrize(
    ("kwargs", "classes_to_bind_on", "error_type", "match"),
    [
        pytest.param(
            {"chunk_size": 4, "preload_nchunks": 3, "batch_size": 6},
            np.repeat([0, 1], 100),
            ValueError,
            "batch_size must be a multiple of chunk_size",
            id="batch_not_multiple_of_chunk",
        ),
        pytest.param(
            {},
            ["B"] * 200,  # inner can emit T, but T is absent here
            ValueError,
            "absent from classes_to_bind_on",
            id="inner_class_absent",
        ),
        pytest.param(
            {},
            ["B"] * 100 + ["Z"] * 100,  # Z is not one of the inner sampler's classes
            ValueError,
            "subset of the inner sampler's classes",
            id="not_subset_of_inner",
        ),
        pytest.param(
            {},
            pd.Categorical.from_codes([-1, 0] * 100, categories=["B", "T"]),
            ValueError,
            "NA values",
            id="classes_na",
        ),
        pytest.param(
            {},
            (["B"] * 3 + ["T"] * 97) * 2,
            ValueError,
            "at least chunk_size",
            id="run_too_short",
        ),
        pytest.param(
            {"on": [0, 1]},  # not a dict
            np.repeat(["B", "T"], 100),
            TypeError,
            "on must be a dict",
            id="on_not_dict",
        ),
        pytest.param(
            {"class_weights": np.array([1.0, 1.0])},  # weights but no secondary `classes`
            np.repeat(["B", "T"], 100),
            ValueError,
            "class_weights was given but classes is None",
            id="class_weights_without_classes",
        ),
        pytest.param(
            {"classes": pd.Categorical(["d1", "d2"] * 5)},  # length 10 != classes_to_bind_on length 200
            np.repeat(["B", "T"], 100),
            ValueError,
            "classes must be the same length",
            id="classes_length_mismatch",
        ),
    ],
)
def test_invalid_construction(kwargs, classes_to_bind_on, error_type, match):
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    with pytest.raises(error_type, match=match):
        make_bound(inner, classes_to_bind_on, **kwargs)


def test_unused_bound_category_is_ignored():
    # pandas categoricals keep zero-observation categories after subsetting; an unused category
    # absent from the inner must be ignored, not tripped over by the subset check.
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    bound = pd.Categorical(["B"] * 100 + ["T"] * 100, categories=["B", "T", "Ghost"])
    sampler = make_bound(inner, bound)
    assert sampler.n_batches(0) == inner.n_batches(0)


def test_grouped_weighted_choice():
    # group 0 -> items {0: w3, 1: w1}; group 1 -> item {2: w1}
    group_of_item = np.array([0, 0, 1])
    weight_of_item = np.array([3.0, 1.0, 1.0])
    rng = np.random.default_rng(0)

    picks0 = grouped_weighted_choice(group_of_item, weight_of_item, np.zeros(4000, dtype=int), rng)
    assert set(picks0.tolist()) <= {0, 1}, "group-0 draws pick only group-0 items"
    assert abs((picks0 == 0).mean() - 0.75) < 0.03, "within group 0, items are weighted 3:1"

    picks1 = grouped_weighted_choice(group_of_item, weight_of_item, np.ones(10, dtype=int), rng)
    assert (picks1 == 2).all(), "group-1 draws pick its only item"


def test_project_index_keeps_multiindex_and_cross_matches():
    # positions=None returns the labels unchanged -- a MultiIndex stays a MultiIndex (not flattened to
    # tuples) -- and it must still get_indexer-match an equivalent object-tuple Index. That cross-type
    # match is what lets a chained bound match its MultiIndex vocab against object-tuple categories.
    mi = pd.MultiIndex.from_tuples([("c1", "b1"), ("c1", "b2"), ("c2", "b1")])
    proj = project_index(mi, None)
    assert isinstance(proj, pd.MultiIndex), "a MultiIndex is preserved, not flattened"
    obj = pd.Index([("c2", "b1"), ("c1", "b1")], tupleize_cols=False)  # same tuples, object dtype
    assert proj.get_indexer(obj).tolist() == [2, 0], "MultiIndex <-> object-tuple Index cross-match"


def test_project_index_single_column_rejects_nonzero_position():
    # a single-column Index has only position 0 (the whole label); any other position is a usage error
    idx = pd.Index(["B", "T", "NK"])
    assert project_index(idx, (0,)) is idx, "position 0 is the whole label -> returned unchanged"
    with pytest.raises(ValueError, match="only position 0"):
        project_index(idx, (1,))


def test_bound_can_be_inner_of_another_bound():
    # a BoundClassSampler is itself a BaseClassSampler (it exposes vocab + the code schedule), so it
    # can be the inner of another bound: the outer replays the mid-bound's per-batch class, by label.
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=200, seed=0)
    mid = make_bound(inner, np.repeat(["B", "T"], 100), seed=1)  # dataset B
    outer = make_bound(mid, np.repeat(["B", "T"], 100), seed=2)  # dataset C
    assert isinstance(mid, BaseClassSampler)
    assert outer.n_batches(0) == mid.n_batches(0) == inner.n_batches(0)

    # exact replay: rebuild the mid-bound identically and read the schedule the outer will replay
    mid_ref = make_bound(
        make_inner(np.repeat(["B", "T"], 100), num_samples=200, seed=0), np.repeat(["B", "T"], 100), seed=1
    )
    expected = list(mid_ref.vocab[mid_ref.batch_codes()])

    outer_key = np.repeat(["B", "T"], 100)  # dataset C bind key per obs
    assert batch_keys(outer, outer_key, len(outer_key)) == expected


def test_chained_bound_coarsens_joint_onto_match():
    # a mid-bound with a secondary (donor) exposes the joint (cell_type, donor) as its vocab; an outer
    # bound can coarsen back onto cell_type (joint position 0) via `on`, even though its own data is a
    # single column -- the P(x | y) cascade, matched by label through the stack.
    ct = np.repeat(["B", "T"], 100)
    donor = np.tile(np.repeat(["d1", "d2"], 50), 2)
    inner = make_inner(ct, num_samples=200, seed=0)
    mid = make_bound(inner, ct, classes=pd.Categorical(donor), class_weights=np.array([1.0, 1.0]), seed=1)
    assert set(mid.vocab) == {("B", "d1"), ("B", "d2"), ("T", "d1"), ("T", "d2")}

    outer = make_bound(mid, ct, on={0: 0}, seed=2)  # coarsen joint -> cell_type
    batch_keys(outer, ct, len(ct))  # asserts each batch is coherent on the coarsened cell_type


def test_chained_bound_composes_columns():
    # each level adds a column: inner balances (cellline, drug); mid keeps that match and adds batch
    # as its secondary, so its vocab flattens to (cellline, drug, batch) -- not nested ((cl, dr), ba)
    # -- and a third can then condition on all three columns.
    cl = np.repeat(["c1", "c2"], 100)
    dr = np.tile(np.repeat(["dA", "dB"], 50), 2)
    ba = np.tile(np.repeat(["b1", "b2"], 25), 4)
    cd = pd.Categorical(pd.MultiIndex.from_arrays([cl, dr]).to_flat_index())
    inner = make_inner(cd, num_samples=200, seed=0)
    mid = make_bound(inner, cd, classes=pd.Categorical(ba), class_weights=np.array([1.0, 1.0]), seed=1)
    assert all(len(v) == 3 for v in mid.vocab), "joint vocab is flat (cellline, drug, batch)"

    cdb = pd.Categorical(pd.MultiIndex.from_arrays([cl, dr, ba]).to_flat_index())
    third = make_bound(mid, cdb, on={0: 0, 1: 1, 2: 2}, seed=2)
    batch_keys(third, cdb.codes, len(cdb))  # asserts each batch coherent on (cellline, drug, batch)


def test_chained_bound_matches_whole_multiindex_vocab():
    # bind on=None onto a mid-bound whose vocab is a MultiIndex (its joint (cellline, batch)). The
    # whole-label path returns that MultiIndex as-is and must match the outer's object-tuple
    # categories -- the exact MultiIndex <-> object-tuple case the direct `return labels` relies on.
    cl = np.repeat(["c1", "c2"], 100)
    ba = np.tile(np.repeat(["b1", "b2"], 50), 2)
    inner = make_inner(cl, num_samples=200, seed=0)
    mid = make_bound(inner, cl, on={0: 0}, classes=pd.Categorical(ba), class_weights=np.array([1.0, 1.0]), seed=1)
    assert isinstance(mid.vocab, pd.MultiIndex), "the mid-bound's joint vocab is a MultiIndex"

    joint = pd.Categorical(pd.MultiIndex.from_arrays([cl, ba]).to_flat_index())  # dataset C: object tuples
    outer = make_bound(mid, joint, seed=2)  # on=None -> match the whole (cellline, batch) joint
    assert outer.n_batches(0) == mid.n_batches(0)
    batch_keys(outer, joint.codes, len(joint))  # asserts each batch coherent on the whole (cellline, batch) joint


def test_classes_to_bind_on_must_be_categorical():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    with pytest.raises(TypeError, match="classes_to_bind_on must be a pandas.Categorical"):
        BoundClassSampler(inner, 10, 4, 10, classes_to_bind_on=np.repeat([0, 1], 100))


def test_classes_to_bind_on_property_returns_input():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    bind = pd.Categorical(np.repeat(["B", "T"], 100))
    assert make_bound(inner, bind).classes_to_bind_on is bind


def test_validate_rejects_n_obs_mismatch():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    sampler = make_bound(inner, np.repeat(["B", "T"], 100))
    with pytest.raises(ValueError, match="does not match loader n_obs"):
        sampler.validate(n_obs=999)


def test_multiple_workers_not_supported():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    sampler = make_bound(inner, np.repeat(["B", "T"], 100))
    with (
        patch(
            "annbatch.samplers._class_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(NotImplementedError, match="Multiple workers"),
    ):
        list(sampler.sample(200))


# =============================================================================
# Core behavior
# =============================================================================


def test_n_batches_matches_inner():
    inner = make_inner(np.repeat(["B", "T", "NK", "Mono"], 100), num_samples=1000)
    sampler = make_bound(inner, np.repeat(["B", "T", "NK", "Mono"], 50))
    assert sampler.n_batches(200) == inner.n_batches(0)


def test_shuffle_is_true():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=100)
    assert make_bound(inner, np.repeat(["B", "T"], 100)).shuffle is True


@pytest.mark.parametrize(
    ("chunk_size", "batch_size", "preload_nchunks"),
    [
        pytest.param(10, 10, 4, id="batch_eq_chunk"),
        pytest.param(5, 10, 4, id="batch_two_chunks"),
        pytest.param(5, 20, 4, id="batch_four_chunks"),
    ],
)
def test_every_batch_is_full_and_coherent(chunk_size, batch_size, preload_nchunks):
    inner = make_inner(np.repeat(["B", "T", "NK", "Mono"], 100), num_samples=1000)
    condition = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 50), categories=["Mono", "NK", "T", "B"])
    sampler = make_bound(
        inner, condition, chunk_size=chunk_size, batch_size=batch_size, preload_nchunks=preload_nchunks
    )
    infos = batch_infos(sampler, condition.codes, len(condition))
    assert len(infos) == inner.n_batches(0)
    assert all(n_unique == 1 for _, _, n_unique in infos), "every batch must be class-coherent"
    assert all(size == batch_size for _, size, _ in infos), "every batch must be full"


def test_replays_inner_per_batch_classes():
    # different category *orderings* and different obs *lengths* -> matched purely by label
    a_labels = pd.Categorical(np.repeat(["B", "T", "NK", "Mono"], 100))
    condition = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 50), categories=["Mono", "NK", "T", "B"])
    inner_for_bound = make_inner(a_labels, num_samples=1000, seed=7)
    sampler = make_bound(inner_for_bound, condition)

    infos = batch_infos(sampler, condition.codes, len(condition))
    bound_labels = [condition.categories[code] for code, _, _ in infos]

    expected = inner_batch_labels(make_inner(a_labels, num_samples=1000, seed=7))
    assert bound_labels == expected


def test_reproducible_with_same_seeds():
    def build():
        inner = make_inner(np.repeat(["B", "T", "NK"], 100), num_samples=600, seed=3)
        return make_bound(
            inner,
            np.repeat(
                ["NK", "T", "B"],
                60,
            ),
            seed=5,
        )

    codes = pd.Categorical(np.repeat(["NK", "T", "B"], 60)).codes
    assert batch_infos(build(), codes, 180) == batch_infos(build(), codes, 180)


def test_two_passes_differ():
    # both rngs advance across sample() calls, like ClassSampler
    inner = make_inner(np.repeat(["B", "T", "NK"], 100), num_samples=600)
    sampler = make_bound(inner, np.repeat(["NK", "T", "B"], 60))
    codes = pd.Categorical(np.repeat(["NK", "T", "B"], 60)).codes
    first = batch_infos(sampler, codes, 180)
    second = batch_infos(sampler, codes, 180)
    assert first != second


# =============================================================================
# Subset binding (`on`)
# =============================================================================


def _tuple_cat(rows: list[tuple], reps: int = 40) -> pd.Categorical:
    """Categorical of tuple labels, each distinct row in a contiguous run of ``reps``."""
    return pd.Categorical(pd.MultiIndex.from_tuples([row for row in rows for _ in range(reps)]).to_flat_index())


def test_on_binds_subset_by_position():
    # inner balances over (cell_type, donor, batch); bind another dataset on (cell_type, donor)
    inner_classes = _tuple_cat([("B", "d1", "x"), ("B", "d2", "y"), ("T", "d1", "x"), ("T", "d2", "y")])
    condition = _tuple_cat([("B", "d1"), ("B", "d2"), ("T", "d1"), ("T", "d2")], reps=30)
    sampler = make_bound(make_inner(inner_classes, num_samples=2000, seed=0), condition, on={0: 0, 1: 1}, seed=1)

    # replay matches the inner's per-batch class projected onto (cell_type, donor)
    expected = list(
        project_index(inner_classes.categories, (0, 1))[
            make_inner(inner_classes, num_samples=2000, seed=0).batch_codes()
        ]
    )
    condition_key = project_index(condition.categories, (0, 1))[condition.codes]
    assert batch_keys(sampler, condition_key, len(condition)) == expected


# =============================================================================
# Conditional (within-class) sampling
# =============================================================================


def test_within_class_weights_shares():
    # inner emits cell type B; within B, weight donors d1:d2 = 3:1 via the secondary `classes`
    inner = make_inner(pd.Categorical(["B"] * 200), num_samples=40_000)
    classes_to_bind_on = pd.Categorical(["B"] * 400)
    donor = pd.Categorical((["d1"] * 20 + ["d2"] * 20) * 10)
    sampler = make_bound(inner, classes_to_bind_on, classes=donor, class_weights=np.array([3.0, 1.0]))

    drawn = donor.codes[np.concatenate([window_indices(lr) for lr in sampler.sample(len(donor))])]
    shares = np.bincount(drawn, minlength=2) / drawn.size  # donor.categories sorted: d1=0, d2=1
    assert abs(shares[0] - 0.75) < 0.02 and abs(shares[1] - 0.25) < 0.02


def test_within_class_zero_weight_excludes_and_exempts_run_length():
    inner = make_inner(pd.Categorical(["B"] * 200), num_samples=2000)
    classes_to_bind_on = pd.Categorical(["B"] * 400)
    # d2 lives only in short (3-row) runs; excluding it with weight 0 must exempt those runs
    donor = pd.Categorical((["d1"] * 37 + ["d2"] * 3) * 10)
    sampler = make_bound(inner, classes_to_bind_on, classes=donor, class_weights=np.array([1.0, 0.0]))

    drawn = {int(c) for lr in sampler.sample(len(donor)) for c in donor.codes[window_indices(lr)]}
    assert drawn == {0}, "only the positive-weight secondary class should be drawn"

    # giving the short-run class a positive weight -> run-length rule now applies and fails
    with pytest.raises(ValueError, match="at least chunk_size"):
        make_bound(inner, classes_to_bind_on, classes=donor, class_weights=np.array([1.0, 1.0]))


# =============================================================================
# Coverage and weighting (larger table)
# =============================================================================


def test_covers_all_drawable_obs_and_respects_weights():
    # A larger condition table (dataset B) described as a DataFrame. Adversarial on purpose:
    #   * (B, d1) and (NK, d2) each appear in two separate runs (non-contiguous),
    #   * donor d3 is excluded (weight 0) and even appears in a length-1 run (must stay exempt),
    #   * cell type Mono is never emitted by the inner sampler (must never be drawn),
    #   * donors present differ per cell type, so weights renormalize within each cell type,
    #   * the inner sampler uses a different category ordering.
    blocks = [
        ("B", "d1", 30),
        ("B", "d2", 30),
        ("T", "d1", 30),
        ("T", "d3", 20),  # d3 excluded (weight 0)
        ("NK", "d2", 30),
        ("NK", "d3", 1),  # excluded AND shorter than chunk_size -> must stay exempt from the run-length rule
        ("Mono", "d1", 30),  # Mono is never emitted by the inner sampler -> never drawn, exempt
        ("B", "d1", 30),  # a second, separate run of (B, d1)
        ("NK", "d2", 30),  # a second, separate run of (NK, d2)
    ]
    obs = pd.DataFrame(
        {
            "cell_type": np.repeat([ct for ct, _, _ in blocks], [n for _, _, n in blocks]),
            "donor": np.repeat([dn for _, dn, _ in blocks], [n for _, _, n in blocks]),
        }
    )
    classes_to_bind_on = pd.Categorical(obs["cell_type"])
    donor = pd.Categorical(obs["donor"])
    donor_weights = pd.Series({"d1": 3.0, "d2": 1.0, "d3": 0.0})  # d3 excluded
    class_weights = donor_weights.reindex(donor.categories).to_numpy()

    # the inner sampler (dataset A) drives the cell-type schedule; Mono excluded, different ordering
    inner_cell = pd.Categorical(np.repeat(["Mono", "NK", "T", "B"], 60))
    inner_weights = np.where(inner_cell.categories == "Mono", 0.0, 1.0)
    inner = ClassSampler(
        10,
        4,
        10,
        classes=inner_cell,
        num_samples=60_000,
        class_weights=inner_weights,
        drop_last=True,
        rng=np.random.default_rng(0),
    )
    sampler = BoundClassSampler(
        inner,
        3,
        4,
        3,
        classes_to_bind_on=classes_to_bind_on,
        classes=donor,
        class_weights=class_weights,
        rng=np.random.default_rng(1),
    )

    # every observation the sampler reads off disk
    drawn_idx = np.concatenate([window_indices(lr) for lr in sampler.sample(len(obs))])
    drawn = obs.iloc[drawn_idx]

    # what *should* be drawable: an emittable cell type and a positive-weight donor
    drawable = obs[obs["cell_type"].isin(["B", "T", "NK"]) & obs["donor"].map(donor_weights).gt(0)]

    # coverage: every drawable observation is hit, and nothing outside the drawable set ever is
    assert set(drawn_idx.tolist()) == set(drawable.index)

    # weighting: within each cell type, donor shares track the renormalized positive weights
    observed = drawn.groupby("cell_type")["donor"].value_counts(normalize=True)
    expected = drawable.drop_duplicates(["cell_type", "donor"]).copy()
    expected["w"] = expected["donor"].map(donor_weights)
    expected["share"] = expected["w"] / expected.groupby("cell_type")["w"].transform("sum")
    for row in expected.itertuples():
        assert abs(observed[row.cell_type, row.donor] - row.share) < 0.02, (
            f"{row.cell_type}/{row.donor}: {observed[row.cell_type, row.donor]:.3f} vs {row.share:.3f}"
        )


# =============================================================================
# Mask
# =============================================================================


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_restricts_range(via):
    # inner knows B and T (so condition is a valid subset) but only emits B (T weight 0)
    inner = make_inner(pd.Categorical(np.repeat(["B", "T"], 100)), num_samples=500, class_weights=np.array([1.0, 0.0]))
    condition = pd.Categorical(["B"] * 100 + ["T"] * 100)
    if via == "constructor":
        sampler = make_bound(inner, condition, mask=slice(0, 100))
    else:
        sampler = make_bound(inner, condition)
        sampler.mask = slice(0, 100)
    chunks = [c for lr in sampler.sample(len(condition)) for c in lr["requests"]]
    assert all(0 <= c.start and c.stop <= 100 for c in chunks), "chunks must stay within the mask range"


def test_mask_with_emittable_class_absent_raises():
    # inner emits B and T; masking to the T-only region leaves B with no drawable run,
    # which surfaces when the (B-emitting) inner schedule is replayed during sampling
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=200)
    condition = pd.Categorical(["B"] * 100 + ["T"] * 100)
    sampler = make_bound(inner, condition)
    sampler.mask = slice(100, 200)
    with pytest.raises(ValueError, match="no drawable run"):
        list(sampler.sample(len(condition)))


# =============================================================================
# Serialization / RNG
# =============================================================================


def test_pickle_roundtrip_continues_identically():
    def build():
        inner = make_inner(np.repeat(["B", "T", "NK"], 100), num_samples=600, seed=2)
        return make_bound(inner, np.repeat(["NK", "T", "B"], 60), seed=9)

    codes = pd.Categorical(np.repeat(["NK", "T", "B"], 60)).codes
    original = build()
    restored = pickle.loads(pickle.dumps(original))
    # both the inner and outer RNG states must survive the round-trip
    assert batch_infos(original, codes, 180) == batch_infos(restored, codes, 180)


def test_deepcopy_is_independent():
    inner = make_inner(np.repeat(["B", "T"], 100), num_samples=200)
    sampler = make_bound(inner, np.repeat(["B", "T"], 100))
    clone = copy.deepcopy(sampler)
    codes = pd.Categorical(np.repeat(["B", "T"], 100)).codes
    # a clone taken before consuming reproduces the same stream
    assert batch_infos(clone, codes, 200) == batch_infos(sampler, codes, 200)


# =============================================================================
# Loader integration
# =============================================================================


def test_bound_class_sampler_from_collection(simple_collection):
    from annbatch import Loader

    _, collection = simple_collection

    condition = collection.obs(columns=["src_path"])["src_path"].values
    categories = condition.categories
    # inner over a synthetic dataset A with the SAME category labels but a different length/order
    inner = ClassSampler(
        chunk_size=1,
        preload_nchunks=4,
        batch_size=4,
        classes=pd.Categorical(np.repeat(categories[::-1], 20), categories=categories),
        num_samples=100,
        drop_last=True,
        rng=np.random.default_rng(0),
    )
    sampler = BoundClassSampler(
        inner,
        1,
        4,
        4,
        classes_to_bind_on=condition,
        rng=np.random.default_rng(0),
    )

    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None)
    loader.use_collection(collection)

    batches = list(loader)
    assert len(batches) == inner.n_batches(0)
    for batch in batches:
        assert batch["X"].shape == (4, 100)
        assert len(np.unique(batch["obs"]["src_path"])) == 1, "every batch must be class-coherent"
