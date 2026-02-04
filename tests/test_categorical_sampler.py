"""Tests for CategoricalSampler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from annbatch import CategoricalSampler, StratifiedCategoricalSampler


def collect_all_indices(sampler, n_obs):
    """Helper to collect all indices from sampler, organized by load request."""
    all_indices = []
    for load_request in sampler.sample(n_obs):
        indices_in_request = []
        for chunk in load_request["chunks"]:
            indices_in_request.extend(range(chunk.start, chunk.stop))
        all_indices.append(indices_in_request)
    return all_indices


def collect_flat_indices(sampler, n_obs):
    """Helper to collect all indices flattened from splits (the actual batch indices)."""
    indices = []
    for load_request in sampler.sample(n_obs):
        # Build chunk indices mapping (indices into concatenated chunk data -> original indices)
        chunk_indices = []
        for chunk in load_request["chunks"]:
            chunk_indices.extend(range(chunk.start, chunk.stop))
        # Collect actual batch indices from splits
        for split in load_request["splits"]:
            for idx in split:
                indices.append(chunk_indices[idx])
    return indices


# =============================================================================
# Basic construction tests
# =============================================================================


def test_basic_construction():
    """Test basic CategoricalSampler construction."""
    boundaries = [slice(0, 100), slice(100, 200), slice(200, 300)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
    )
    assert sampler.batch_size == 10
    assert sampler.n_categories == 3
    assert sampler.category_sizes == [100, 100, 100]
    assert sampler.shuffle is False


def test_from_pandas_categorical():
    """Test construction from pandas Categorical."""
    # Create sorted categorical data
    categories = pd.Categorical(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
    sampler = CategoricalSampler.from_pandas(
        categories,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
    )
    assert sampler.n_categories == 3
    assert sampler.category_sizes == [50, 30, 20]


def test_from_pandas_series():
    """Test construction from pandas Series with categorical dtype."""
    series = pd.Series(pd.Categorical(["X"] * 40 + ["Y"] * 60))
    sampler = CategoricalSampler.from_pandas(
        series,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
    )
    assert sampler.n_categories == 2
    assert sampler.category_sizes == [40, 60]


def test_from_pandas_unsorted_raises():
    """Test that unsorted data raises ValueError."""
    categories = pd.Categorical(["A", "B", "A", "B"])  # Not sorted
    with pytest.raises(ValueError, match="Data must be sorted"):
        CategoricalSampler.from_pandas(
            categories,
            batch_size=2,
            chunk_size=4,
            preload_nchunks=1,
        )


def test_from_pandas_non_categorical_raises():
    """Test that non-categorical Series raises TypeError."""
    series = pd.Series(["A", "B", "C"])  # Not categorical
    with pytest.raises(TypeError, match="Expected categorical"):
        CategoricalSampler.from_pandas(
            series,
            batch_size=2,
            chunk_size=4,
            preload_nchunks=1,
        )


def test_from_pandas_empty_raises():
    """Test that empty categorical raises ValueError."""
    categories = pd.Categorical([])
    with pytest.raises(ValueError, match="empty"):
        CategoricalSampler.from_pandas(
            categories,
            batch_size=2,
            chunk_size=4,
            preload_nchunks=1,
        )


# =============================================================================
# Boundary validation tests
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
    """Test that invalid boundaries raise appropriate errors."""
    with pytest.raises((ValueError, TypeError), match=error_match):
        CategoricalSampler(
            category_boundaries=boundaries,
            batch_size=5,
            chunk_size=10,
            preload_nchunks=1,
        )


def test_empty_boundaries_raises():
    """Test that empty boundaries list raises ValueError."""
    with pytest.raises(ValueError):
        CategoricalSampler(
            category_boundaries=[],
            batch_size=5,
            chunk_size=10,
            preload_nchunks=1,
        )


# =============================================================================
# Coverage tests
# =============================================================================


@pytest.mark.parametrize(
    "category_sizes,chunk_size,preload_nchunks,batch_size",
    [
        pytest.param([100, 100, 100], 20, 2, 10, id="equal_categories"),
        pytest.param([50, 150, 100], 25, 2, 10, id="unequal_categories"),
        pytest.param([30, 30, 30], 10, 3, 5, id="small_categories"),
        pytest.param([200], 50, 2, 25, id="single_category"),
        pytest.param([10, 20, 30, 40], 10, 1, 5, id="many_categories"),
    ],
)
def test_coverage_all_indices(category_sizes, chunk_size, preload_nchunks, batch_size):
    """Test that sampler covers all indices exactly once."""
    # Build boundaries from category sizes
    boundaries = []
    start = 0
    for size in category_sizes:
        boundaries.append(slice(start, start + size))
        start += size
    n_obs = sum(category_sizes)

    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        shuffle=False,
    )

    all_indices = collect_flat_indices(sampler, n_obs)
    assert set(all_indices) == set(range(n_obs)), "Should cover all indices"
    assert len(all_indices) == n_obs, "Should cover each index exactly once"


def _get_category_for_index(index: int, boundaries: list[slice]) -> int:
    """Helper to find which category an index belongs to."""
    for i, boundary in enumerate(boundaries):
        if boundary.start <= index < boundary.stop:
            return i
    raise ValueError(f"Index {index} not in any category boundary")


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(
    "boundaries",
    [
        pytest.param([slice(0, 100), slice(100, 200), slice(200, 300)], id="equal_categories"),
        pytest.param([slice(0, 50), slice(50, 150), slice(150, 300)], id="unequal_categories"),
        pytest.param([slice(0, 30), slice(30, 60), slice(60, 90), slice(90, 120)], id="many_categories"),
    ],
)
def test_each_split_from_single_category(boundaries, shuffle):
    """Test that each split (batch) within a load request is from a single category.

    Note: The CategoricalSampler combines batches from multiple categories into
    a single load request for efficiency, but each split within that request
    should only contain indices from a single category.
    """
    n_obs = boundaries[-1].stop

    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        shuffle=shuffle,
        rng=np.random.default_rng(42),
    )

    for load_request in sampler.sample(n_obs):
        chunks = load_request["chunks"]
        if len(chunks) == 0:
            continue

        # Build mapping from concatenated chunk index to original index
        chunk_indices = []
        for chunk in chunks:
            chunk_indices.extend(range(chunk.start, chunk.stop))

        # Verify each split contains indices from only one category
        for split in load_request["splits"]:
            if len(split) == 0:
                continue

            # Get the category of the first index in this split
            first_original_idx = chunk_indices[split[0]]
            expected_category = _get_category_for_index(first_original_idx, boundaries)

            # Verify all indices in this split belong to the same category
            for idx in split:
                original_idx = chunk_indices[idx]
                split_category = _get_category_for_index(original_idx, boundaries)
                assert split_category == expected_category, (
                    f"Split index {idx} (original {original_idx}) belongs to category {split_category}, "
                    f"but expected category {expected_category}"
                )


# =============================================================================
# Shuffle tests
# =============================================================================


def test_shuffle_changes_order():
    """Test that shuffling changes the order of indices within categories."""
    boundaries = [slice(0, 100), slice(100, 200)]
    n_obs = 200

    sampler_no_shuffle = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        shuffle=False,
    )

    sampler_shuffle = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        shuffle=True,
        rng=np.random.default_rng(42),
    )

    indices_no_shuffle = collect_flat_indices(sampler_no_shuffle, n_obs)
    indices_shuffle = collect_flat_indices(sampler_shuffle, n_obs)

    # Both should cover same indices
    assert set(indices_no_shuffle) == set(indices_shuffle)

    # But order should differ
    assert indices_no_shuffle != indices_shuffle


# =============================================================================
# Validation tests
# =============================================================================


def test_validate_boundary_exceeds_n_obs():
    """Test validation fails when boundary exceeds n_obs."""
    boundaries = [slice(0, 100), slice(100, 300)]  # Second boundary goes to 300
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
    )

    with pytest.raises(ValueError, match="exceeds loader n_obs"):
        sampler.validate(n_obs=200)  # n_obs is only 200


def test_validate_passes_for_valid_config():
    """Test validation passes for valid configuration."""
    boundaries = [slice(0, 100), slice(100, 200)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
    )
    # Should not raise
    sampler.validate(n_obs=200)


# =============================================================================
# Batch size validation tests
# =============================================================================


@pytest.mark.parametrize(
    "batch_size,chunk_size,preload_nchunks,error_match",
    [
        pytest.param(100, 10, 2, "batch_size cannot exceed", id="batch_exceeds_preload"),
        pytest.param(7, 10, 2, "must be divisible by batch_size", id="not_divisible"),
    ],
)
def test_invalid_batch_size_raises(batch_size, chunk_size, preload_nchunks, error_match):
    """Test that invalid batch_size configurations raise ValueError."""
    with pytest.raises(ValueError, match=error_match):
        CategoricalSampler(
            category_boundaries=[slice(0, 100)],
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )


# =============================================================================
# Drop last tests
# =============================================================================


def test_drop_last_enforced():
    """Test that incomplete batches are always dropped (drop_last is enforced)."""
    # 45 obs, batch_size 10 -> should get 4 complete batches (40 obs)
    boundaries = [slice(0, 45)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
    )

    total_obs = 0
    for load_request in sampler.sample(45):
        for split in load_request["splits"]:
            total_obs += len(split)

    assert total_obs == 40, "should drop incomplete batch"


# =============================================================================
# Splits structure tests
# =============================================================================


def test_splits_have_correct_batch_size():
    """Test that splits have correct batch sizes (all complete batches)."""
    boundaries = [slice(0, 100)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
    )

    for load_request in sampler.sample(100):
        splits = load_request["splits"]
        # All splits should have exactly batch_size elements (drop_last is enforced)
        for split in splits:
            assert len(split) == 10


# =============================================================================
# Integration with from_pandas
# =============================================================================


def test_from_pandas_integration():
    """Test full integration with from_pandas and sampling."""
    # Simulate sorted obs column
    n_obs = 150
    categories = pd.Categorical(["celltype_A"] * 50 + ["celltype_B"] * 70 + ["celltype_C"] * 30)

    sampler = CategoricalSampler.from_pandas(
        categories,
        batch_size=10,
        chunk_size=25,
        preload_nchunks=2,
        shuffle=True,
        rng=np.random.default_rng(123),
    )

    all_indices = collect_flat_indices(sampler, n_obs)
    assert set(all_indices) == set(range(n_obs))
    assert len(all_indices) == n_obs


# =============================================================================
# Reproducibility tests
# =============================================================================


def test_rng_reproducibility():
    """Test that same RNG seed gives same results."""
    boundaries = [slice(0, 100), slice(100, 200)]

    def get_indices(seed):
        sampler = CategoricalSampler(
            category_boundaries=boundaries,
            batch_size=10,
            chunk_size=20,
            preload_nchunks=2,
            shuffle=True,
            rng=np.random.default_rng(seed),
        )
        return collect_flat_indices(sampler, 200)

    indices1 = get_indices(42)
    indices2 = get_indices(42)
    indices3 = get_indices(99)

    assert indices1 == indices2, "Same seed should give same results"
    assert indices1 != indices3, "Different seeds should give different results"


# =============================================================================
# StratifiedCategoricalSampler tests
# =============================================================================


def test_stratified_basic_construction():
    """Test basic StratifiedCategoricalSampler construction."""
    boundaries = [slice(0, 100), slice(100, 200), slice(200, 300)]
    sampler = StratifiedCategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        n_yields=50,
    )
    assert sampler.batch_size == 10
    assert sampler.n_categories == 3
    assert sampler.n_yields == 50
    assert sampler.shuffle is False
    # Default weights are uniform
    np.testing.assert_array_equal(sampler.weights, [1.0, 1.0, 1.0])


def test_stratified_custom_weights():
    """Test StratifiedCategoricalSampler with custom weights."""
    boundaries = [slice(0, 100), slice(100, 200), slice(200, 300)]
    sampler = StratifiedCategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        n_yields=50,
        weights=[1.0, 2.0, 3.0],
    )
    np.testing.assert_array_equal(sampler.weights, [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(sampler.probabilities, [1 / 6, 2 / 6, 3 / 6])


def test_stratified_n_yields_count():
    """Test that exactly n_yields batches are yielded."""
    boundaries = [slice(0, 100), slice(100, 200)]
    n_yields = 25

    sampler = StratifiedCategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        n_yields=n_yields,
        rng=np.random.default_rng(42),
    )

    batch_count = 0
    for load_request in sampler.sample(200):
        batch_count += len(load_request["splits"])

    assert batch_count == n_yields


def test_stratified_n_yields_invalid():
    """Test that n_yields < 1 raises ValueError."""
    boundaries = [slice(0, 100)]
    with pytest.raises(ValueError, match="n_yields must be >= 1"):
        StratifiedCategoricalSampler(
            category_boundaries=boundaries,
            batch_size=10,
            chunk_size=20,
            preload_nchunks=2,
            n_yields=0,
        )


def test_stratified_weights_validation():
    """Test weight validation errors."""
    boundaries = [slice(0, 100), slice(100, 200)]

    # Wrong length
    with pytest.raises(ValueError, match="weights length"):
        StratifiedCategoricalSampler(
            category_boundaries=boundaries,
            batch_size=10,
            chunk_size=20,
            preload_nchunks=2,
            n_yields=10,
            weights=[1.0],  # Only 1 weight for 2 categories
        )

    # Negative weights
    with pytest.raises(ValueError, match="non-negative"):
        StratifiedCategoricalSampler(
            category_boundaries=boundaries,
            batch_size=10,
            chunk_size=20,
            preload_nchunks=2,
            n_yields=10,
            weights=[1.0, -1.0],
        )

    # Zero sum
    with pytest.raises(ValueError, match="not sum to zero"):
        StratifiedCategoricalSampler(
            category_boundaries=boundaries,
            batch_size=10,
            chunk_size=20,
            preload_nchunks=2,
            n_yields=10,
            weights=[0.0, 0.0],
        )


def test_stratified_replacement():
    """Test that categories are reset when exhausted (sampling with replacement)."""
    # Small category with only 2 complete batches possible
    boundaries = [slice(0, 20)]  # 20 obs, batch_size=10, drop_last=True -> 2 batches
    n_yields = 10  # Request more than available

    sampler = StratifiedCategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=1,
        n_yields=n_yields,
        rng=np.random.default_rng(42),
    )

    batch_count = 0
    for load_request in sampler.sample(20):
        batch_count += len(load_request["splits"])

    # Should still yield n_yields batches due to replacement
    assert batch_count == n_yields


def test_stratified_each_batch_single_category():
    """Test that each batch in stratified sampling is from a single category."""
    boundaries = [slice(0, 100), slice(100, 200), slice(200, 300)]
    n_obs = 300

    sampler = StratifiedCategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        n_yields=50,
        shuffle=True,
        rng=np.random.default_rng(42),
    )

    for load_request in sampler.sample(n_obs):
        chunks = load_request["chunks"]
        if len(chunks) == 0:
            continue

        # Build mapping from concatenated chunk index to original index
        chunk_indices = []
        for chunk in chunks:
            chunk_indices.extend(range(chunk.start, chunk.stop))

        # Verify each split contains indices from only one category
        for split in load_request["splits"]:
            if len(split) == 0:
                continue

            first_original_idx = chunk_indices[split[0]]
            expected_category = _get_category_for_index(first_original_idx, boundaries)

            for idx in split:
                original_idx = chunk_indices[idx]
                split_category = _get_category_for_index(original_idx, boundaries)
                assert split_category == expected_category


def test_stratified_rng_reproducibility():
    """Test that same RNG seed gives same results for stratified sampler."""
    boundaries = [slice(0, 100), slice(100, 200)]

    def get_batches(seed):
        sampler = StratifiedCategoricalSampler(
            category_boundaries=boundaries,
            batch_size=10,
            chunk_size=20,
            preload_nchunks=2,
            n_yields=20,
            shuffle=True,
            rng=np.random.default_rng(seed),
        )
        return collect_flat_indices(sampler, 200)

    indices1 = get_batches(42)
    indices2 = get_batches(42)
    indices3 = get_batches(99)

    assert indices1 == indices2, "Same seed should give same results"
    assert indices1 != indices3, "Different seeds should give different results"


def test_stratified_from_pandas():
    """Test StratifiedCategoricalSampler.from_pandas construction."""
    categories = pd.Categorical(["A"] * 50 + ["B"] * 30 + ["C"] * 20)

    sampler = StratifiedCategoricalSampler.from_pandas(
        categories,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        n_yields=30,
    )

    assert sampler.n_categories == 3
    assert sampler.category_sizes == [50, 30, 20]
    assert sampler.n_yields == 30


def test_stratified_from_pandas_with_weights():
    """Test StratifiedCategoricalSampler.from_pandas with custom weights."""
    categories = pd.Categorical(["A"] * 50 + ["B"] * 30 + ["C"] * 20)

    sampler = StratifiedCategoricalSampler.from_pandas(
        categories,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        n_yields=30,
        weights=[3.0, 2.0, 1.0],
    )

    np.testing.assert_array_equal(sampler.weights, [3.0, 2.0, 1.0])


def test_stratified_uniform_weights_distribution():
    """Test that uniform weights sample categories roughly equally."""
    boundaries = [slice(0, 100), slice(100, 200), slice(200, 300)]
    n_obs = 300
    n_yields = 300  # Large number for statistical significance

    sampler = StratifiedCategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        n_yields=n_yields,
        rng=np.random.default_rng(42),
    )

    # Count batches per category
    category_counts = [0, 0, 0]
    for load_request in sampler.sample(n_obs):
        chunks = load_request["chunks"]
        chunk_indices = []
        for chunk in chunks:
            chunk_indices.extend(range(chunk.start, chunk.stop))

        for split in load_request["splits"]:
            if len(split) > 0:
                first_idx = chunk_indices[split[0]]
                cat = _get_category_for_index(first_idx, boundaries)
                category_counts[cat] += 1

    # With uniform weights, each category should get roughly 1/3 of batches
    # Allow 20% tolerance for randomness
    expected = n_yields / 3
    for count in category_counts:
        assert abs(count - expected) < expected * 0.3, f"Category count {count} too far from expected {expected}"
