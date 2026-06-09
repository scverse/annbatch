"""Tests for CategoricalSampler.

The passing tests check the sampler does what it promises: every chunk is
category-coherent, categories are drawn with the requested weights (a zero weight
excludes a category), masks restrict and renormalize correctly, and the
bookkeeping (``num_samples`` / ``n_batches`` / validation) is correct.

The final test (``test_pure_categorical_batches_unsupported``) is expected to
**fail**. It is deliberately not marked ``xfail``: it documents, with the real
:class:`~annbatch.Loader` ordering contract, why the sampler cannot currently
yield *category-pure batches*.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from annbatch.samplers import CategoricalSampler
from annbatch.samplers._utils import WorkerInfo


def make_sampler(
    categorical: pd.Categorical,
    *,
    num_samples: int = 1000,
    chunk_size: int = 10,
    preload_nchunks: int = 4,
    batch_size: int = 10,
    seed: int = 0,
    **kwargs,
) -> CategoricalSampler:
    """Build a sampler with sane defaults so each test only states what matters."""
    return CategoricalSampler(
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        categorical=categorical,
        num_samples=num_samples,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def _chunk_categories(chunks: list[slice], codes: np.ndarray) -> list[int]:
    """Category of each chunk, or -1 if the chunk straddles more than one category."""
    return [int(u[0]) if (u := np.unique(codes[c])).size == 1 else -1 for c in chunks]


def _collect_chunks(sampler: CategoricalSampler, n_obs: int) -> list[slice]:
    return [c for load_request in sampler.sample(n_obs) for c in load_request["requests"]]


def _draw_shares(sampler: CategoricalSampler, codes: np.ndarray) -> dict[int, float]:
    """Fraction of drawn chunks belonging to each category."""
    cats = np.array(_chunk_categories(_collect_chunks(sampler, len(codes)), codes))
    vals, counts = np.unique(cats, return_counts=True)
    return {int(v): cnt / counts.sum() for v, cnt in zip(vals, counts, strict=True)}


def _assert_shares(sampler: CategoricalSampler, codes: np.ndarray, expected: dict[int, float], atol: float = 0.02):
    shares = _draw_shares(sampler, codes)
    assert set(shares) == set(expected), f"sampled categories {sorted(shares)} != {sorted(expected)}"
    for cat, exp in expected.items():
        assert abs(shares[cat] - exp) <= atol, f"category {cat}: share {shares[cat]:.3f} vs expected {exp}"


# =============================================================================
# Construction / validation
# =============================================================================


@pytest.mark.parametrize(
    ("categorical", "kwargs", "error_type", "match"),
    [
        pytest.param(
            pd.Categorical([0, 0, 1, 1, 2, 2]), {}, ValueError, "at least chunk_size", id="all_runs_too_short"
        ),
        pytest.param(
            pd.Categorical([0] * 30 + [1] * 30 + [0] * 3),
            {},
            ValueError,
            r"at least chunk_size.*\[0\]",
            id="one_run_too_short",
        ),
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"num_samples": 0},
            ValueError,
            "num_samples must be greater than 1",
            id="num_samples",
        ),
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"category_weights": np.ones(3)},
            ValueError,
            "one weight per category",
            id="weights_shape",
        ),
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"category_weights": np.zeros(2)},
            ValueError,
            "at least one positive",
            id="weights_zero",
        ),
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"mask": slice(0, 500)},
            ValueError,
            "exceeds loader n_obs",
            id="mask_out_of_range",
        ),
        # chunk_size(10) * preload_nchunks(4) = 40 < batch_size
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"batch_size": 50},
            ValueError,
            "batch_size cannot exceed",
            id="batch_gt_preload",
        ),
        # 40 % 30 != 0
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"batch_size": 30},
            ValueError,
            "must be divisible",
            id="batch_not_divisible",
        ),
        # preload_size 40 % 4 == 0, but chunk_size 10 % 4 != 0 -> batches would span categories
        pytest.param(
            pd.Categorical(np.repeat([0, 1], 50)),
            {"batch_size": 4},
            ValueError,
            "must be divisible",
            id="batch_not_divides_chunk",
        ),
        pytest.param(np.repeat([0, 1], 50), {}, TypeError, "pandas.Categorical", id="not_categorical"),
        pytest.param(
            pd.Categorical.from_codes([-1, 0, 0, 1, 1] * 20, categories=[0, 1]),
            {},
            ValueError,
            "NA values",
            id="na_values",
        ),
    ],
)
def test_invalid_construction(
    categorical: pd.Categorical | np.ndarray, kwargs: dict, error_type: type[Exception], match: str
):
    with pytest.raises(error_type, match=match):
        make_sampler(categorical, **kwargs)


def test_validate_rejects_n_obs_mismatch():
    sampler = make_sampler(pd.Categorical(np.repeat([0, 1], 50)), num_samples=50)
    with pytest.raises(ValueError, match="does not match loader n_obs"):
        sampler.validate(n_obs=999)


def test_multiple_workers_not_supported():
    sampler = make_sampler(pd.Categorical(np.repeat([0, 1], 50)), num_samples=50)
    with (
        patch(
            "annbatch.samplers._categorical_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(NotImplementedError, match="Multiple workers"),
    ):
        list(sampler.sample(100))


# =============================================================================
# Core behavior
# =============================================================================


@pytest.mark.parametrize(
    "codes",
    [
        pytest.param(np.repeat([0, 1, 2, 3], 100), id="contiguous"),
        # each category split into two runs interleaved with the others
        pytest.param(np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50), id="non_contiguous"),
        pytest.param(np.array([2] * 40 + [0] * 40 + [1] * 40 + [0] * 40 + [2] * 40), id="non_contiguous_3cat"),
    ],
)
def test_chunks_are_category_coherent(codes: np.ndarray):
    chunks = _collect_chunks(make_sampler(pd.Categorical(codes), num_samples=1000), len(codes))
    assert -1 not in _chunk_categories(chunks, codes), "every chunk must lie within a single category"
    # chunks stay in-bounds and are full size (num_samples is a multiple of chunk_size here)
    assert all(0 <= c.start and c.stop <= len(codes) and c.stop - c.start == 10 for c in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "batch_size", "preload_nchunks"),
    [
        pytest.param(10, 10, 4, id="batch_eq_chunk"),
        pytest.param(10, 5, 4, id="batch_lt_chunk"),
        pytest.param(20, 5, 3, id="many_batches_per_chunk"),
    ],
)
def test_batches_are_category_coherent(chunk_size: int, batch_size: int, preload_nchunks: int):
    # the preload window mixes several categories, but each *batch* (split) must not.
    codes = np.repeat([0, 1, 2, 3], 100)
    sampler = make_sampler(
        pd.Categorical(codes),
        num_samples=400,
        chunk_size=chunk_size,
        batch_size=batch_size,
        preload_nchunks=preload_nchunks,
    )
    for load_request in sampler.sample(len(codes)):
        concat = np.concatenate([codes[s.start : s.stop] for s in load_request["requests"]])
        for split in load_request["splits"]:
            assert np.unique(concat[split]).size == 1, "every batch must lie within a single category"


def test_shuffle_is_true():
    assert make_sampler(pd.Categorical(np.repeat([0, 1], 50))).shuffle is True


def test_noncontiguous_category_samples_all_runs():
    # category 0 lives in two separate runs; over many draws both should be hit.
    codes = np.array([0] * 50 + [1] * 50 + [0] * 50, dtype=np.int64)
    starts = [
        c.start
        for c in _collect_chunks(make_sampler(pd.Categorical(codes), num_samples=5000), len(codes))
        if codes[c.start] == 0
    ]
    assert any(s < 50 for s in starts) and any(s >= 100 for s in starts), "both runs of category 0 should be sampled"


@pytest.mark.parametrize(
    ("codes", "weights", "expected"),
    [
        pytest.param(np.repeat([0, 1, 2, 3], 100), None, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}, id="uniform"),
        pytest.param(np.array([0] * 300 + [1] * 100), np.array([300.0, 100.0]), {0: 0.75, 1: 0.25}, id="proportional"),
        pytest.param(np.repeat([0, 1, 2], 100), np.array([6.0, 3.0, 1.0]), {0: 0.6, 1: 0.3, 2: 0.1}, id="explicit"),
        pytest.param(
            np.repeat([0, 1, 2, 3], 100), np.array([1.0, 0.0, 1.0, 0.0]), {0: 0.5, 2: 0.5}, id="zero_excludes"
        ),
        pytest.param(np.repeat([0, 1, 2], 100), np.array([1.0, -1.0, 1.0]), {0: 0.5, 2: 0.5}, id="negative_excludes"),
    ],
)
def test_category_draw_shares(codes: np.ndarray, weights: np.ndarray | None, expected: dict[int, float]):
    _assert_shares(make_sampler(pd.Categorical(codes), num_samples=40_000, category_weights=weights), codes, expected)


def test_zero_weight_category_exempt_from_run_length_rule():
    codes = np.array([0] * 30 + [1] * 3 + [2] * 30, dtype=np.int64)  # category 1 has a 3-row run
    # excluding category 1 with a zero weight -> its short run is exempt, no error
    make_sampler(pd.Categorical(codes), category_weights=np.array([1.0, 0.0, 1.0]))
    # giving it a positive weight -> the short run violates the run-length rule
    with pytest.raises(ValueError, match="at least chunk_size"):
        make_sampler(pd.Categorical(codes), category_weights=np.array([1.0, 1.0, 1.0]))


def test_run_length_error_names_category_labels():
    # codes are alphabetical (B=0, NK=1, T=2), so matching 'B' proves the error prints
    # this is expected to fail because the min chunk size is 10 and B has a 3-row run
    # the label, not the raw code -- the whole point of taking a pd.Categorical.
    cat = pd.Categorical(["T"] * 30 + ["B"] * 3 + ["NK"] * 30)  # "B" has a 3-row run
    with pytest.raises(ValueError, match=r"categories \['B'\]"):
        make_sampler(cat, chunk_size=10)


def test_absent_category_weight_is_ignored():
    # "c" is declared but has no observations; its weight must be silently dropped and
    # the present categories renormalize among themselves (here -> 50/50).
    cat = pd.Categorical(["a"] * 100 + ["b"] * 100, categories=["a", "b", "c"])
    codes = np.asarray(cat.codes)
    sampler = make_sampler(cat, num_samples=40_000, category_weights=np.array([1.0, 1.0, 5.0]))
    _assert_shares(sampler, codes, {0: 0.5, 1: 0.5})


# =============================================================================
# Mask
# =============================================================================


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_restricts_range(via: str):
    codes = np.array([0] * 100 + [1] * 100, dtype=np.int64)
    if via == "constructor":
        sampler = make_sampler(pd.Categorical(codes), num_samples=500, mask=slice(0, 100))
    else:
        sampler = make_sampler(pd.Categorical(codes), num_samples=500)
        sampler.mask = slice(0, 100)
    chunks = _collect_chunks(sampler, len(codes))
    assert all(0 <= c.start and c.stop <= 100 for c in chunks), "chunks must stay within the mask range"
    assert {int(np.unique(codes[c])[0]) for c in chunks} == {0}


def test_mask_renormalizes_from_original_weights():
    codes = np.concatenate([np.full(100, 0), np.full(100, 1), np.full(100, 2)]).astype(np.int64)
    sampler = make_sampler(pd.Categorical(codes), num_samples=40_000, category_weights=np.array([3.0, 1.0, 6.0]))
    _assert_shares(sampler, codes, {0: 0.3, 1: 0.1, 2: 0.6})  # full range
    sampler.mask = slice(0, 200)  # only categories 0 and 1 -> renormalize [3, 1] from the originals
    _assert_shares(sampler, codes, {0: 0.75, 1: 0.25})
    sampler.mask = slice(0, None)  # back to the full range -> original weights restored
    _assert_shares(sampler, codes, {0: 0.3, 1: 0.1, 2: 0.6})


@pytest.mark.parametrize("via", ["constructor", "setter"])
def test_mask_with_no_positive_weight_in_range_raises(via: str):
    codes = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    weights = np.array([1.0, 0.0])  # category 1 excluded -> the [50, 100) range has no sampleable category
    if via == "constructor":
        with pytest.raises(ValueError, match="positive weight is present"):
            make_sampler(pd.Categorical(codes), category_weights=weights, mask=slice(50, 100))
    else:
        sampler = make_sampler(pd.Categorical(codes), category_weights=weights)
        with pytest.raises(ValueError, match="positive weight is present"):
            sampler.mask = slice(50, 100)


# =============================================================================
# Bookkeeping
# =============================================================================


@pytest.mark.parametrize(
    ("num_samples", "batch_size", "drop_last", "expected_iters"),
    [
        pytest.param(100, 10, False, 10, id="exact"),
        pytest.param(105, 10, False, 11, id="partial_kept"),
        pytest.param(105, 10, True, 10, id="partial_dropped"),
    ],
)
def test_n_batches(num_samples: int, batch_size: int, drop_last: bool, expected_iters: int):
    sampler = make_sampler(
        pd.Categorical(np.repeat([0, 1], 100)),
        num_samples=num_samples,
        preload_nchunks=2,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    assert sampler.n_batches(200) == expected_iters


@pytest.mark.parametrize(
    ("chunk_size", "batch_size", "preload_nchunks", "num_samples", "drop_last"),
    [
        pytest.param(10, 10, 4, 400, False, id="exact_windows"),  # num_samples a multiple of the window
        pytest.param(10, 10, 4, 410, False, id="partial_window"),  # trailing window has fewer slices
        pytest.param(10, 10, 4, 405, False, id="remainder_slice"),  # trailing slice shorter than chunk_size
        pytest.param(10, 5, 4, 405, False, id="multi_batch_remainder"),  # >1 batch/chunk + short last batch
        pytest.param(20, 5, 2, 410, False, id="many_batches_per_chunk"),
        pytest.param(10, 10, 1, 355, False, id="preload_one"),  # one slice per window
        pytest.param(10, 5, 3, 305, True, id="drop_last_multi_batch"),  # drop_last omits the sub-chunk remainder
        pytest.param(10, 10, 4, 400, True, id="drop_last_exact"),  # drop_last is a no-op when it divides evenly
    ],
)
def test_sampling_invariants(chunk_size: int, batch_size: int, preload_nchunks: int, num_samples: int, drop_last: bool):
    codes = np.repeat([0, 1, 2, 3], 250)  # 4 categories, every run >> chunk_size
    n = len(codes)
    sampler = make_sampler(
        pd.Categorical(codes),
        num_samples=num_samples,
        chunk_size=chunk_size,
        batch_size=batch_size,
        preload_nchunks=preload_nchunks,
        drop_last=drop_last,
    )

    requests: list[slice] = []
    batches_seen = total_obs = 0
    for lr in sampler.sample(n):
        requests.extend(lr["requests"])
        concat = np.concatenate([codes[s.start : s.stop] for s in lr["requests"]])
        for split in lr["splits"]:
            batches_seen += 1
            total_obs += split.size
            assert np.unique(concat[split]).size == 1, "every batch must lie within a single category"

    # drop_last omits the final sub-chunk remainder; otherwise every requested observation is yielded
    expected_obs = (num_samples // chunk_size) * chunk_size if drop_last else num_samples
    assert total_obs == expected_obs
    assert sampler.n_batches(n) == batches_seen, "n_batches must match the batches actually yielded"
    assert all(0 <= s.start and s.stop <= n for s in requests), "every chunk must stay in-bounds"
    assert -1 not in _chunk_categories(requests, codes), "every chunk must lie within a single category"
