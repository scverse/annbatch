"""Tests for CategoricalSampler."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from annbatch import CategoricalSampler


def _get_category_for_index(index: int, boundaries: list[slice]) -> int:
    for i, boundary in enumerate(boundaries):
        if boundary.start <= index < boundary.stop:
            return i
    raise ValueError(f"Index {index} not in any category boundary")


def collect_batch_categories(sampler, n_obs, boundaries):
    """Return a list with one category id per batch, derived from the chunks each batch loads."""
    categories = []
    for load_request in sampler.sample(n_obs):
        chunks = load_request["chunks"]
        chunk_indices = []
        for chunk in chunks:
            chunk_indices.extend(range(chunk.start, chunk.stop))
        for split in load_request["splits"]:
            first_idx = chunk_indices[split[0]]
            categories.append(_get_category_for_index(first_idx, boundaries))
    return categories


# =============================================================================
# Construction
# =============================================================================


def test_basic_construction():
    boundaries = [slice(0, 100), slice(100, 200), slice(200, 300)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        n_iters=20,
    )
    assert sampler.batch_size == 10
    assert sampler.n_categories == 3
    assert sampler.shuffle is True
    assert sampler.n_iters(300) == 20


def test_custom_weights():
    boundaries = [slice(0, 100), slice(100, 200)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        n_iters=50,
        weights=[3, 1],
    )
    assert sampler.n_categories == 2
    np.testing.assert_allclose(sampler._weights, [0.75, 0.25])


# =============================================================================
# Boundary validation
# =============================================================================


@pytest.mark.parametrize(
    "boundaries,error_match",
    [
        pytest.param([slice(0, 10), slice(10, 5)], "start < stop", id="start_gte_stop"),
        pytest.param([slice(0, 10, 2)], "step=1", id="step_not_one"),
        pytest.param([slice(None, 10)], "explicit start and stop", id="none_start"),
        pytest.param([slice(0, None)], "explicit start and stop", id="none_stop"),
        pytest.param(["not a slice"], "Expected slice", id="not_a_slice"),
        pytest.param([slice(5, 15)], "must start at 0", id="not_starting_at_zero"),
        pytest.param([slice(0, 10), slice(15, 25)], "contiguous", id="gap_between_boundaries"),
    ],
)
def test_invalid_boundary_raises(boundaries, error_match):
    with pytest.raises((ValueError, TypeError), match=error_match):
        CategoricalSampler(
            category_boundaries=boundaries,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=1,
            n_iters=5,
        )


def test_empty_boundaries_raises():
    with pytest.raises(ValueError):
        CategoricalSampler(
            category_boundaries=[],
            batch_size=10,
            chunk_size=10,
            preload_nchunks=1,
            n_iters=5,
        )


def test_wrong_weights_length():
    with pytest.raises(ValueError, match="weights length"):
        CategoricalSampler(
            category_boundaries=[slice(0, 100), slice(100, 200)],
            batch_size=10,
            chunk_size=10,
            preload_nchunks=1,
            n_iters=5,
            weights=[1, 2, 3],
        )


def test_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        CategoricalSampler(
            category_boundaries=[slice(0, 100), slice(100, 200)],
            batch_size=10,
            chunk_size=10,
            preload_nchunks=1,
            n_iters=5,
            weights=[-1, 2],
        )


def test_zero_weights():
    with pytest.raises(ValueError, match="must not all be zero"):
        CategoricalSampler(
            category_boundaries=[slice(0, 100)],
            batch_size=10,
            chunk_size=10,
            preload_nchunks=1,
            n_iters=5,
            weights=[0],
        )


# =============================================================================
# n_iters correctness
# =============================================================================


def test_yields_correct_number_of_batches():
    boundaries = [slice(0, 100), slice(100, 200), slice(200, 300)]
    n_iters = 15
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        n_iters=n_iters,
    )
    total_batches = 0
    for lr in sampler.sample(300):
        total_batches += len(lr["splits"])
    assert total_batches == n_iters


# =============================================================================
# Each batch is single-category
# =============================================================================


@pytest.mark.parametrize(
    "boundaries",
    [
        pytest.param([slice(0, 100), slice(100, 200), slice(200, 300)], id="equal"),
        pytest.param([slice(0, 50), slice(50, 150), slice(150, 300)], id="unequal"),
    ],
)
def test_each_batch_from_single_category(boundaries):
    n_obs = boundaries[-1].stop
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        n_iters=20,
        rng=np.random.default_rng(42),
    )
    for lr in sampler.sample(n_obs):
        chunks = lr["chunks"]
        chunk_indices = []
        for chunk in chunks:
            chunk_indices.extend(range(chunk.start, chunk.stop))
        for split in lr["splits"]:
            cats = {_get_category_for_index(chunk_indices[idx], boundaries) for idx in split}
            assert len(cats) == 1, f"Batch spans multiple categories: {cats}"


# =============================================================================
# Weights affect distribution
# =============================================================================


def test_weights_bias_distribution():
    boundaries = [slice(0, 100), slice(100, 200)]
    n_iters = 1000
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        n_iters=n_iters,
        weights=[9, 1],
        rng=np.random.default_rng(0),
    )
    cats = collect_batch_categories(sampler, 200, boundaries)
    counts = Counter(cats)
    ratio = counts[0] / max(counts[1], 1)
    assert ratio > 3, f"Expected category 0 to dominate, got ratio {ratio}"


def test_uniform_weights_roughly_equal():
    boundaries = [slice(0, 100), slice(100, 200), slice(200, 300)]
    n_iters = 900
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        n_iters=n_iters,
        rng=np.random.default_rng(42),
    )
    cats = collect_batch_categories(sampler, 300, boundaries)
    counts = Counter(cats)
    for cat in range(3):
        assert abs(counts[cat] - n_iters / 3) < n_iters * 0.15, f"Category {cat} count {counts[cat]} too far from expected"


# =============================================================================
# Reproducibility
# =============================================================================


def test_rng_reproducibility():
    boundaries = [slice(0, 100), slice(100, 200)]

    def get_cats(seed):
        sampler = CategoricalSampler(
            category_boundaries=boundaries,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=2,
            n_iters=30,
            rng=np.random.default_rng(seed),
        )
        return collect_batch_categories(sampler, 200, boundaries)

    assert get_cats(42) == get_cats(42)
    assert get_cats(42) != get_cats(99)


# =============================================================================
# Splits have correct batch size
# =============================================================================


def test_splits_have_correct_batch_size():
    boundaries = [slice(0, 100), slice(100, 200)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        n_iters=20,
    )
    for lr in sampler.sample(200):
        for split in lr["splits"]:
            assert len(split) == 10


# =============================================================================
# batch_size < chunk_size rejected
# =============================================================================


def test_batch_size_less_than_chunk_size_raises():
    with pytest.raises(ValueError, match="cannot be less than chunk_size"):
        CategoricalSampler(
            category_boundaries=[slice(0, 100)],
            batch_size=5,
            chunk_size=10,
            preload_nchunks=2,
            n_iters=5,
        )
