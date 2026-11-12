from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from annbatch.samplers import BoundClassSampler, DistributedSampler, WeightedClassSampler
from tests.test_class_sampler import make_sampler

CELL_TYPE = np.repeat(["B", "T"], 100)
FOUR_TYPES = np.repeat(["B", "T", "NK", "Mono"], 100)
DONOR = pd.Categorical((["d1"] * 20 + ["d2"] * 20) * 5)


def make_inner(labels, **kwargs):
    return make_sampler(pd.Categorical(labels), drop_last=True, **kwargs)


def make_bound(inner, classes, *, chunk_size=10, preload_nchunks=4, batch_size=10, seed=1, **kwargs):
    return BoundClassSampler(
        chunk_size,
        preload_nchunks,
        batch_size,
        inner_sampler=inner,
        classes=pd.Categorical(classes),
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def batches(sampler, n_obs: int) -> list[np.ndarray]:
    out = []
    for load_request in sampler.sample(n_obs):
        window = np.concatenate([np.arange(s.start, s.stop) for s in load_request["requests"]])
        out.extend(window[split] for split in load_request["splits"])
    return out


def keys_of(rows: list[np.ndarray], key_of_obs) -> list:
    key = np.asarray(key_of_obs)
    keys = []
    for idx in rows:
        unique = set(key[idx].tolist())
        assert len(unique) == 1, "each batch must be coherent"
        keys.append(unique.pop())
    return keys


def batch_keys(sampler, key_of_obs) -> list:
    return keys_of(batches(sampler, len(key_of_obs)), key_of_obs)


def abandon_after_first_request(sampler, n_obs: int) -> None:
    requests = sampler.sample(n_obs)
    next(requests)
    requests.close()  # as a `break` out of the loader does


def joint(*columns) -> pd.Categorical:
    return pd.Categorical(pd.MultiIndex.from_arrays(columns).to_flat_index())


# (cell type, donor, batch) and (cell type, donor)
THREE_COLUMNS = joint(*(np.repeat(col, 40) for col in (["B", "B", "T", "T"], ["d1", "d2"] * 2, ["x", "y"] * 2)))
TWO_COLUMNS = joint(*(np.repeat(col, 30) for col in (["B", "B", "T", "T"], ["d1", "d2"] * 2)))


@pytest.mark.parametrize(
    ("inner_kwargs", "classes", "kwargs", "exc", "match"),
    [
        pytest.param(
            {},
            CELL_TYPE,
            {"chunk_size": 4, "preload_nchunks": 3, "batch_size": 6},
            ValueError,
            "must be a multiple of chunk_size",
            id="batch_not_multiple_of_chunk",
        ),
        pytest.param({}, CELL_TYPE, {"chunk_size": 0}, ValueError, "Chunk size", id="chunk_zero"),
        pytest.param({}, ["B"] * 200, {}, ValueError, "no match for classes", id="class_absent"),
        pytest.param({}, (["B"] * 3 + ["T"] * 97) * 2, {}, ValueError, "at least chunk_size", id="run_short"),
        pytest.param(
            {},
            joint(CELL_TYPE, np.repeat(["d1", "d2"], 100)),
            {"on": {0: 0}, "class_weights": [0.0, 1.0]},
            ValueError,
            "no positive-weight rows",
            id="group_weighted_out",
        ),
        pytest.param(
            {},
            CELL_TYPE,
            {"class_weights": pd.Series([3.0, 1.0], index=["T", "B"])},
            TypeError,
            "not a pandas Series",
            id="series_class_weights",
        ),
        pytest.param(
            {"cls": WeightedClassSampler}, CELL_TYPE, {}, TypeError, "must be a ClassSampler", id="weighted_inner"
        ),
        pytest.param({"num_samples": 5}, CELL_TYPE, {}, ValueError, "yields no batches", id="inner_without_batches"),
    ],
)
def test_invalid_construction(inner_kwargs, classes, kwargs, exc, match):
    inner = make_sampler(pd.Categorical(CELL_TYPE), **({"num_samples": 100, "drop_last": True} | inner_kwargs))
    with pytest.raises(exc, match=match):
        make_bound(inner, classes, **kwargs)


@pytest.mark.parametrize(
    ("chunk_size", "batch_size"),
    [pytest.param(10, 10, id="batch_eq_chunk"), pytest.param(5, 10, id="two_chunks"), pytest.param(5, 20, id="four")],
)
def test_stays_paired_with_the_primary_every_pass(chunk_size, batch_size):
    oracle = make_inner(FOUR_TYPES, num_samples=400)
    primary = copy.deepcopy(oracle)
    condition = np.repeat(["Mono", "NK", "T", "B"], 50)
    sampler = make_bound(oracle, condition, chunk_size=chunk_size, batch_size=batch_size)
    for _ in range(3):
        rows = batches(sampler, len(condition))
        assert all(r.size == batch_size for r in rows), "every batch must be full"
        assert keys_of(rows, condition) == batch_keys(primary, FOUR_TYPES)


def test_a_short_last_inner_batch_is_paired_with_a_full_one():
    oracle = make_sampler(pd.Categorical(FOUR_TYPES), num_samples=205)
    primary = copy.deepcopy(oracle)
    sampler = make_bound(oracle, FOUR_TYPES)
    for _ in range(2):
        rows = batches(sampler, len(FOUR_TYPES))
        assert [r.size for r in rows] == [10] * 21
        assert keys_of(rows, FOUR_TYPES) == batch_keys(primary, FOUR_TYPES)


def test_stays_paired_after_the_primary_cuts_a_pass_short():
    oracle = make_inner(FOUR_TYPES, num_samples=400)
    primary = copy.deepcopy(oracle)
    sampler = make_bound(oracle, FOUR_TYPES)
    batch_keys(primary, FOUR_TYPES)
    batch_keys(sampler, FOUR_TYPES)
    abandon_after_first_request(primary, len(FOUR_TYPES))
    batch_keys(sampler, FOUR_TYPES)
    for _ in range(2):
        assert batch_keys(sampler, FOUR_TYPES) == batch_keys(primary, FOUR_TYPES)


def test_restored_pickles_start_a_fresh_paired_pass():
    oracle = make_inner(FOUR_TYPES, num_samples=400)
    primary, sampler = copy.deepcopy(oracle), make_bound(oracle, FOUR_TYPES)
    abandon_after_first_request(primary, len(FOUR_TYPES))
    abandon_after_first_request(sampler, len(FOUR_TYPES))
    primary, sampler = pickle.loads(pickle.dumps(primary)), pickle.loads(pickle.dumps(sampler))
    for _ in range(2):
        assert batch_keys(sampler, FOUR_TYPES) == batch_keys(primary, FOUR_TYPES)


def test_chained_bound_stays_paired():
    oracle = make_inner(CELL_TYPE, num_samples=200)
    primary = copy.deepcopy(oracle)
    outer = make_bound(make_bound(copy.deepcopy(oracle), CELL_TYPE), CELL_TYPE, seed=2)
    for _ in range(3):
        assert batch_keys(outer, CELL_TYPE) == batch_keys(primary, CELL_TYPE)


def test_wrapping_in_a_distributed_sampler_is_rejected():
    with pytest.raises(TypeError, match="cannot wrap a BoundClassSampler"):
        DistributedSampler(make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE), dist_info=lambda: (0, 2))


def test_a_distributed_inner_is_rejected():
    dist = DistributedSampler(make_inner(CELL_TYPE, num_samples=200), dist_info=lambda: (0, 2))
    with pytest.raises(TypeError, match="must be a ClassSampler"):
        make_bound(dist, CELL_TYPE)


def test_changing_the_inner_after_binding_does_not_reach_the_copy():
    inner = make_inner(CELL_TYPE, num_samples=200)
    primary = copy.deepcopy(inner)
    sampler = make_bound(inner, CELL_TYPE)
    inner.mask, inner.rng = slice(0, 100), np.random.default_rng(999)
    list(inner.sample(len(CELL_TYPE)))
    assert batch_keys(sampler, CELL_TYPE) == batch_keys(primary, CELL_TYPE)


def test_the_bound_generator_moves_rows_not_the_schedule():
    inner = make_inner(CELL_TYPE, num_samples=200)
    x, y = make_bound(inner, CELL_TYPE, seed=1), make_bound(inner, CELL_TYPE, seed=2)
    rows_x, rows_y = batches(x, len(CELL_TYPE)), batches(y, len(CELL_TYPE))
    assert keys_of(rows_x, CELL_TYPE) == keys_of(rows_y, CELL_TYPE)
    assert not all(np.array_equal(a, b) for a, b in zip(rows_x, rows_y, strict=True))


def test_a_mask_hiding_an_emitted_class_is_refused_by_the_constructor():
    with pytest.raises(ValueError, match=r"\['T'\] the inner sampler can emit"):
        make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE, mask=slice(0, 100))


def test_a_mask_hiding_an_emitted_class_is_refused_on_assignment():
    sampler = make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE)
    with pytest.raises(ValueError, match=r"\['T'\] the inner sampler can emit"):
        sampler.mask = slice(0, 100)  # only B survives here, but the inner still emits T
    assert sampler.mask == slice(0, len(CELL_TYPE)), "the refused mask is not kept"


def test_a_refused_mask_keeps_the_previous_narrowed_mask():
    sampler = make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE)
    sampler.mask = slice(50, 200)  # still holds both B and T
    with pytest.raises(ValueError, match="the inner sampler can emit"):
        sampler.mask = slice(0, 100)
    assert sampler.mask == slice(50, 200)


def test_assigning_no_generator_gives_a_fresh_one():
    sampler = make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE)
    sampler.rng = None
    assert isinstance(sampler.rng, np.random.Generator)
    assert len(batches(sampler, len(CELL_TYPE))) > 0


@pytest.mark.parametrize(
    "on", [{0: 0, 1: 1}, {np.int64(0): np.int64(0), np.int64(1): np.int64(1)}], ids=["int", "np_int64"]
)
def test_on_matches_label_columns(on):
    oracle = make_inner(THREE_COLUMNS, num_samples=2000)
    primary = copy.deepcopy(oracle)
    sampler = make_bound(oracle, TWO_COLUMNS, on=on)
    assert batch_keys(sampler, TWO_COLUMNS) == [label[:2] for label in batch_keys(primary, THREE_COLUMNS)]


def test_on_with_single_column_tuples():
    oracle = make_inner(CELL_TYPE, num_samples=200)
    primary = copy.deepcopy(oracle)
    single = joint(CELL_TYPE)  # one-tuple labels like ("B",)
    sampler = make_bound(oracle, single, on={0: 0})
    assert batch_keys(sampler, single) == [(label,) for label in batch_keys(primary, CELL_TYPE)]


@pytest.mark.parametrize(
    ("on", "exc", "match"),
    [
        pytest.param({5: 0}, ValueError, r"positions in \[0, 3\) of the inner", id="inner_position_out_of_range"),
        pytest.param({0: 5}, ValueError, r"positions in \[0, 2\) of classes", id="own_position_out_of_range"),
        pytest.param({}, ValueError, "on must map", id="empty"),
        pytest.param((0, 1), TypeError, "must be a dict", id="tuple"),
        pytest.param({0.9: 0}, TypeError, "must be a dict of int", id="float_position"),
        pytest.param({True: 0}, TypeError, "must be a dict of int", id="bool_position"),
        pytest.param({0: False}, TypeError, "must be a dict of int", id="bool_own_position"),
        pytest.param({3: 0}, ValueError, "on must map", id="inner_position_at_len"),
        pytest.param({0: 2}, ValueError, "on must map", id="own_position_at_len"),
        pytest.param({0: -2}, ValueError, "on must map", id="own_position_negative"),
        pytest.param({-3: 0}, ValueError, "on must map", id="inner_position_negative"),
    ],
)
def test_invalid_on_is_rejected(on, exc, match):
    with pytest.raises(exc, match=match):
        make_bound(make_inner(THREE_COLUMNS, num_samples=200), TWO_COLUMNS, on=on)


def test_matching_classes_are_coherent_and_weighted():
    sampler = make_bound(
        make_inner(["B"] * 200, num_samples=40_000),
        joint(["B"] * 200, np.asarray(DONOR)),  # (B, d1) and (B, d2) both match the inner's B
        batch_size=20,
        on={0: 0},
        class_weights=[3.0, 1.0],
    )
    donors = batch_keys(sampler, DONOR)  # asserts each batch is one donor
    share_d1 = donors.count("d1") / len(donors)
    assert abs(share_d1 - 0.75) < 0.03


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_restricts_range(via):
    # B lives on both sides of the mask edge, so an ignored mask would read B rows past 150
    bind_on = ["B"] * 50 + ["T"] * 100 + ["B"] * 50
    inner = make_inner(CELL_TYPE, num_samples=500)
    if via == "constructor":
        sampler = make_bound(inner, bind_on, mask=slice(0, 150))
    else:
        sampler = make_bound(inner, bind_on)
        sampler.mask = slice(0, 150)
    chunks = [c for lr in sampler.sample(len(bind_on)) for c in lr["requests"]]
    assert all(c.stop <= 150 for c in chunks)


def test_bound_classes_the_inner_never_emits_are_never_read():
    # Z is also shorter than chunk_size
    bind_on = np.array(["B"] * 100 + ["T"] * 97 + ["Z"] * 3)
    sampler = make_bound(make_inner(CELL_TYPE, num_samples=200), bind_on)
    assert set(batch_keys(sampler, bind_on)) == {"B", "T"}


def test_n_batches_matches_a_pass():
    sampler = make_bound(make_inner(FOUR_TYPES, num_samples=400), FOUR_TYPES, chunk_size=5, batch_size=20)
    assert sampler.n_batches(len(FOUR_TYPES)) == len(batches(sampler, len(FOUR_TYPES)))


def test_bound_categories_in_another_order_still_pair():
    oracle = make_inner(FOUR_TYPES, num_samples=400)
    primary = copy.deepcopy(oracle)
    sampler = make_bound(oracle, pd.Categorical(FOUR_TYPES, categories=["T", "NK", "Mono", "B"]))
    assert batch_keys(sampler, FOUR_TYPES) == batch_keys(primary, FOUR_TYPES)


def test_on_matches_columns_in_the_order_given():
    swapped = joint(*(np.repeat(col, 30) for col in (["d1", "d2"] * 2, ["B", "B", "T", "T"])))  # (donor, cell type)
    oracle = make_inner(THREE_COLUMNS, num_samples=400)
    primary = copy.deepcopy(oracle)
    sampler = make_bound(oracle, swapped, on={1: 0, 0: 1})  # keys deliberately out of order
    assert batch_keys(sampler, swapped) == [(label[1], label[0]) for label in batch_keys(primary, THREE_COLUMNS)]


def test_rows_are_shuffled_within_a_batch():
    sampler = make_bound(make_inner(CELL_TYPE, num_samples=200), CELL_TYPE)
    assert any((np.diff(rows) < 0).any() for rows in batches(sampler, len(CELL_TYPE)))


def test_weights_are_normalized_among_the_matching_classes():
    # T rows are all d1, so B's share of d1 must follow the weights alone
    donor = pd.Categorical(["d1"] * 20 + ["d2"] * 20 + ["d1"] * 20 + ["d2"] * 40 + ["d1"] * 100)
    sampler = make_bound(
        make_inner(CELL_TYPE, num_samples=20_000),
        joint(CELL_TYPE, np.asarray(donor)),  # categories (B, d1), (B, d2), (T, d1)
        batch_size=20,
        on={0: 0},
        class_weights=[3.0, 1.0, 1.0],
    )
    rows = batches(sampler, len(CELL_TYPE))
    donors_of_b = [d for cell, d in zip(keys_of(rows, CELL_TYPE), keys_of(rows, donor), strict=True) if cell == "B"]
    assert abs(donors_of_b.count("d1") / len(donors_of_b) - 0.75) < 0.03


@pytest.mark.parametrize(
    ("classes", "exc", "match"),
    [
        pytest.param(np.asarray(CELL_TYPE), TypeError, "must be a pandas.Categorical", id="not_categorical"),
        pytest.param(pd.Categorical(["B"] * 100 + [None] * 100), ValueError, "NA values", id="na"),
        pytest.param(
            pd.Categorical(["B"] * 200, categories=["B", "T"]),
            ValueError,
            "no positive-weight rows",
            id="unused_category",
        ),
    ],
)
def test_invalid_classes(classes, exc, match):
    with pytest.raises(exc, match=match):
        BoundClassSampler(10, 4, 10, inner_sampler=make_inner(CELL_TYPE, num_samples=200), classes=classes)


@pytest.mark.parametrize("via", ["inner", "chain", "setter"])
def test_the_primarys_own_generator_is_copied_so_pairing_holds(via):
    primary = make_inner(CELL_TYPE, num_samples=200)
    if via == "inner":
        sampler = BoundClassSampler(
            10, 4, 10, inner_sampler=primary, classes=pd.Categorical(CELL_TYPE), rng=primary.rng
        )
    elif via == "chain":
        middle = make_bound(primary, CELL_TYPE)
        sampler = BoundClassSampler(10, 4, 10, inner_sampler=middle, classes=pd.Categorical(CELL_TYPE), rng=primary.rng)
    else:
        sampler = make_bound(primary, CELL_TYPE)
        sampler.rng = primary.rng
    for _ in range(3):
        assert batch_keys(sampler, CELL_TYPE) == batch_keys(primary, CELL_TYPE)


def test_a_class_the_inner_never_emits_needs_no_match():
    # T has weight 0 in the inner, so it is never emitted and need not exist in the bound classes
    inner = make_inner(CELL_TYPE, num_samples=200, class_weights=np.array([1.0, 0.0]))
    bind_on = np.array(["B"] * 200)
    assert set(batch_keys(make_bound(inner, bind_on), bind_on)) == {"B"}


def test_inner_batches_spanning_several_chunks_stay_paired():
    oracle = make_inner(FOUR_TYPES, num_samples=400, chunk_size=5, batch_size=10)
    primary = copy.deepcopy(oracle)
    sampler = make_bound(oracle, FOUR_TYPES)
    assert batch_keys(sampler, FOUR_TYPES) == batch_keys(primary, FOUR_TYPES)


def test_on_pairs_inner_position_with_own_position():
    donor = np.tile(np.repeat(["d1", "d2"], 50), 2)
    oracle = make_inner(joint(CELL_TYPE, donor), num_samples=200)  # cell type is inner column 0
    primary = copy.deepcopy(oracle)
    sampler = make_bound(oracle, joint(donor, CELL_TYPE), on={0: 1})  # ... and own column 1
    assert keys_of(batches(sampler, 200), CELL_TYPE) == [
        cell for cell, _ in batch_keys(primary, joint(CELL_TYPE, donor))
    ]


def test_classes_is_read_only():
    classes = pd.Categorical(CELL_TYPE)
    sampler = BoundClassSampler(10, 4, 10, inner_sampler=make_inner(CELL_TYPE, num_samples=200), classes=classes)
    assert sampler.classes.equals(classes)
    with pytest.raises(ValueError, match="read-only"):
        sampler.classes[0] = "T"
    with pytest.raises(AttributeError):
        sampler.classes = classes


class _TopOfUnitInterval:
    """A generator whose random() always returns the largest float below 1."""

    def __init__(self, rng):
        self._rng = rng

    def random(self, size):
        return np.full(size, np.nextafter(1.0, 0.0))

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_a_draw_at_the_top_of_a_group_stays_in_that_group():
    # ten equally weighted (B, s_i) classes: their normalized cumulative weight rounds to just below 1
    subclasses = np.concatenate([np.repeat([f"s{i}" for i in range(10)], 10), np.repeat(["t"], 100)])
    oracle = make_inner(CELL_TYPE, num_samples=200)
    primary = copy.deepcopy(oracle)
    sampler = make_bound(oracle, joint(CELL_TYPE, subclasses), on={0: 0})  # one chunk per batch
    requests = sampler._iter_requests(_TopOfUnitInterval(np.random.default_rng(0)))
    chunks = [np.arange(s.start, s.stop) for lr in requests for s in lr["requests"]]
    assert keys_of(chunks, CELL_TYPE) == batch_keys(primary, CELL_TYPE)
