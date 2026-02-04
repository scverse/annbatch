"""Tests for CategoricalSampler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from annbatch import CategoricalSampler


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
    """Helper to collect all indices flattened."""
    indices = []
    for load_request in sampler.sample(n_obs):
        for chunk in load_request["chunks"]:
            indices.extend(range(chunk.start, chunk.stop))
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
def test_each_load_request_from_single_category(boundaries, shuffle):
    """Test that each load request contains chunks from a single category."""
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
        # Check all chunks are from the same category
        chunks = load_request["chunks"]
        if len(chunks) == 0:
            continue

        # Find which category this load request belongs to based on first chunk
        first_chunk_start = chunks[0].start
        category_idx = _get_category_for_index(first_chunk_start, boundaries)
        category_boundary = boundaries[category_idx]

        # Verify ALL chunks are within the same category boundary
        for chunk in chunks:
            assert chunk.start >= category_boundary.start, (
                f"Chunk start {chunk.start} outside category {category_idx} boundary {category_boundary}"
            )
            assert chunk.stop <= category_boundary.stop, (
                f"Chunk stop {chunk.stop} outside category {category_idx} boundary {category_boundary}"
            )

        # Also verify all indices in splits map back to the same category
        # The splits are indices into the concatenated chunks, so we need to map them back
        chunk_indices = []
        for chunk in chunks:
            chunk_indices.extend(range(chunk.start, chunk.stop))

        for split in load_request["splits"]:
            for idx in split:
                original_idx = chunk_indices[idx]
                split_category = _get_category_for_index(original_idx, boundaries)
                assert split_category == category_idx, (
                    f"Split index {idx} (original {original_idx}) belongs to category {split_category}, "
                    f"but load request is for category {category_idx}"
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


def test_drop_last_removes_incomplete_batches():
    """Test that drop_last removes incomplete batches."""
    # 45 obs, batch_size 10 -> should get 4 complete batches (40 obs) with drop_last=True
    boundaries = [slice(0, 45)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        drop_last=True,
    )

    total_obs = 0
    for load_request in sampler.sample(45):
        for split in load_request["splits"]:
            total_obs += len(split)

    assert total_obs == 40, "drop_last should remove incomplete batch"


def test_no_drop_last_keeps_incomplete_batches():
    """Test that drop_last=False keeps incomplete batches."""
    boundaries = [slice(0, 45)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        drop_last=False,
    )

    total_obs = 0
    for load_request in sampler.sample(45):
        for split in load_request["splits"]:
            total_obs += len(split)

    assert total_obs == 45, "should keep all observations"


# =============================================================================
# Splits structure tests
# =============================================================================


def test_splits_have_correct_batch_size():
    """Test that splits have correct batch sizes."""
    boundaries = [slice(0, 100)]
    sampler = CategoricalSampler(
        category_boundaries=boundaries,
        batch_size=10,
        chunk_size=20,
        preload_nchunks=2,
        drop_last=False,
    )

    for load_request in sampler.sample(100):
        splits = load_request["splits"]
        # All splits except possibly the last should have batch_size elements
        for split in splits[:-1]:
            assert len(split) == 10
        # Last split should have 1 to batch_size elements
        if splits:
            assert 1 <= len(splits[-1]) <= 10


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
