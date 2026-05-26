"""Tests for FragmentedRandomSampler."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from annbatch.samplers._fragmented_random_sampler import FragmentedRandomSampler
from annbatch.samplers._utils import WorkerInfo
from tests.test_sampler import collect_indices

# =============================================================================
# Mask Validation Tests
# =============================================================================


@pytest.mark.parametrize(
    ("masks", "error_match"),
    [
        pytest.param([slice(0, 20)], None, id="valid_single_mask"),
        pytest.param([slice(0, 20), slice(30, 50)], None, id="valid_two_masks"),
        pytest.param([slice(0, 15), slice(15, 30)], None, id="adjacent_masks_merged"),
        pytest.param([slice(30, 50), slice(0, 20)], None, id="unsorted_masks_sorted"),
        pytest.param(
            [slice(0, 20), slice(10, 30)],
            "non-overlapping",
            id="overlapping_masks",
        ),
        pytest.param(
            [slice(0, 20, 2)],
            "mask.step must be 1 or None",
            id="step_not_one",
        ),
        pytest.param(
            [slice(20, 10)],
            "mask.stop > mask.start",
            id="start_greater_than_stop",
        ),
        pytest.param(
            [slice(-1, 10)],
            "mask.start >= 0",
            id="negative_start",
        ),
        pytest.param(
            [slice(0, 5)],
            "at least one chunk",
            id="mask_smaller_than_chunk",
        ),
        pytest.param(
            [slice(0, 20), slice(30, 35)],
            "at least one chunk",
            id="second_mask_too_small",
        ),
    ],
)
def test_mask_validation(masks: list[slice], error_match: str | None):
    """Test mask validation at construction."""
    kwargs = {
        "chunk_size": 10,
        "preload_nchunks": 2,
        "batch_size": 5,
        "masks": masks,
        "num_samples": 50,
    }
    if error_match:
        with pytest.raises((ValueError, TypeError), match=error_match):
            FragmentedRandomSampler(**kwargs)
    else:
        sampler = FragmentedRandomSampler(**kwargs)
        assert sampler is not None


# =============================================================================
# Mask Coverage Tests
# =============================================================================


@pytest.mark.parametrize(
    ("masks", "num_samples", "drop_last"),
    [
        pytest.param([slice(0, 100)], 50, False, id="single_mask_partial"),
        pytest.param([slice(0, 100)], 100, False, id="single_mask_full"),
        pytest.param([slice(0, 100), slice(200, 300)], 100, False, id="two_masks"),
        pytest.param(
            [slice(0, 100), slice(200, 300)],
            90,
            False,
            id="three_masks",
        ),
    ],
)
def test_mask_coverage(
    masks: list[slice],
    num_samples: int,
    drop_last: bool,
):
    """Test that sampler yields correct number of samples from specified masks."""
    chunk_size, preload_nchunks, batch_size = 10, 2, 5
    sampler = FragmentedRandomSampler(
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        masks=masks,
        num_samples=num_samples,
        drop_last=drop_last,
        rng=np.random.default_rng(42),
    )

    all_indices, all_chunks, splits = collect_indices(sampler, n_obs=max(m.stop for m in masks))

    # Verify all indices are within at least one mask
    mask_ranges = set()
    for mask in masks:
        mask_ranges.update(range(mask.start, mask.stop))

    assert set(all_indices).issubset(mask_ranges), "All indices should be within mask ranges"

    # Verify batch count matches n_iters
    expected_iters = sampler.n_iters(n_obs=500)
    assert len(splits) == expected_iters, f"Expected {expected_iters} batches, got {len(splits)}"


# =============================================================================
# Batch and Iteration Count Tests
# =============================================================================


@pytest.mark.parametrize(
    ("masks", "num_samples", "batch_size", "drop_last", "expected_iters"),
    [
        pytest.param([slice(0, 100)], 50, 5, False, 10, id="exact_division"),
        pytest.param([slice(0, 100)], 52, 5, False, 11, id="ceil_division"),
        pytest.param([slice(0, 100)], 52, 5, True, 10, id="floor_division"),
        pytest.param([slice(0, 100), slice(200, 300)], 75, 10, False, 8, id="multiple_masks"),
        pytest.param([slice(0, 100), slice(200, 300)], 75, 10, True, 7, id="multiple_masks_drop"),
        pytest.param(
            [slice(0, 100), slice(200, 300), slice(400, 500)], 150, 5, False, 30, id="three_masks"
        ),
    ],
)
def test_batch_and_iteration_counts(
    masks: list[slice],
    num_samples: int,
    batch_size: int,
    drop_last: bool,
    expected_iters: int,
):
    """Test n_iters property and actual batch counts for various configurations."""
    sampler = FragmentedRandomSampler(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=batch_size,
        masks=masks,
        num_samples=num_samples,
        drop_last=drop_last,
        rng=np.random.default_rng(42),
    )
    assert sampler.n_iters(n_obs=500) == expected_iters

    assert sampler.shuffle is True, "Shuffle property should always return True"
    assert sampler.batch_size == batch_size, "Batch size property should return the correct value"

    n_obs = max(m.stop for m in masks)
    all_indices, all_chunks, splits = collect_indices(sampler, n_obs)

    # Verify actual batch count matches the property
    assert len(splits) == expected_iters
    # Verify structure is sound
    assert len(all_chunks) > 0
    for chunk in all_chunks:
        assert chunk.stop - chunk.start > 0


# =============================================================================
# Randomness and Reproducibility Tests
# =============================================================================


@pytest.mark.parametrize(
    ("masks", "num_samples"),
    [
        pytest.param([slice(0, 100)], 50, id="single_mask"),
        pytest.param([slice(0, 100), slice(200, 300)], 100, id="two_masks"),
    ],
)
def test_reproducibility_with_same_seed(masks: list[slice], num_samples: int):
    """Test same seed produces same chunk sequence."""
    kwargs = {
        "chunk_size": 10,
        "preload_nchunks": 2,
        "batch_size": 5,
        "masks": masks,
        "num_samples": num_samples,
    }

    indices1, _, _ = collect_indices(
        FragmentedRandomSampler(**kwargs, rng=np.random.default_rng(42)),
        n_obs=max(m.stop for m in masks),
    )
    indices2, _, _ = collect_indices(
        FragmentedRandomSampler(**kwargs, rng=np.random.default_rng(42)),
        n_obs=max(m.stop for m in masks),
    )
    indices3, _, _ = collect_indices(
        FragmentedRandomSampler(**kwargs, rng=np.random.default_rng(99)),
        n_obs=max(m.stop for m in masks),
    )

    assert indices1 == indices2, "Same seed should produce identical sequences"
    assert indices1 != indices3, "Different seeds should produce different sequences"


def test_mask_property_not_implemented():
    """Test that mask property getter and setter raise NotImplementedError."""
    sampler = FragmentedRandomSampler(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=5,
        masks=[slice(0, 100)],
        num_samples=50,
    )
    with pytest.raises(NotImplementedError, match="mask property is not implemented"):
        _ = sampler.mask

    with pytest.raises(NotImplementedError, match="mask property is not implemented"):
        sampler.mask = slice(0, 50)


@pytest.mark.parametrize(
    ("masks", "n_obs", "should_fail"),
    [
        pytest.param([slice(0, 100)], 100, False, id="valid_n_obs"),
        pytest.param([slice(0, 100)], 99, True, id="n_obs_too_small"),
        pytest.param([slice(0, 100), slice(200, 300)], 300, False, id="multiple_masks_valid"),
        pytest.param([slice(0, 100), slice(200, 300)], 299, True, id="multiple_masks_invalid"),
    ],
)
def test_validate(masks: list[slice], n_obs: int, should_fail: bool):
    """Test validate() checks n_obs against mask bounds."""
    sampler = FragmentedRandomSampler(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=5,
        masks=masks,
        num_samples=50,
    )
    if should_fail:
        with pytest.raises(ValueError, match="mask.stop.*exceeds loader n_obs"):
            sampler.validate(n_obs)
    else:
        sampler.validate(n_obs)


# =============================================================================
# Error and Edge Case Tests
# =============================================================================


def test_multiple_workers_not_supported():
    """Test that multiple workers raise NotImplementedError."""
    sampler = FragmentedRandomSampler(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=5,
        masks=[slice(0, 100)],
        num_samples=50,
        rng=np.random.default_rng(42),
    )
    with (
        patch(
            "annbatch.samplers._fragmented_random_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(NotImplementedError, match="Multiple workers are not supported"),
    ):
        list(sampler.sample(n_obs=500))


@pytest.mark.parametrize(
    ("chunk_size", "preload_nchunks", "batch_size", "should_fail"),
    [
        pytest.param(10, 0, 5, True, id="preload_zero"),
        pytest.param(0, 2, 5, True, id="chunk_zero"),
        pytest.param(10, 2, 5, False, id="valid_config"),
        pytest.param(10, 2, 10, False, id="batch_equals_preload_size"),
    ],
)
def test_chunk_batch_preload_size_validation(
    chunk_size: int,
    preload_nchunks: int,
    batch_size: int,
    should_fail: bool,
):
    """Test validation of chunk, batch, and preload sizes."""
    if should_fail:
        with pytest.raises((ValueError, ZeroDivisionError)):
            FragmentedRandomSampler(
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=batch_size,
                masks=[slice(0, 100)],
                num_samples=50,
            )
    else:
        sampler = FragmentedRandomSampler(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            masks=[slice(0, 100)],
            num_samples=50,
        )
        assert sampler is not None


def test_small_masks_with_multiple_chunks():
    """Test sampling from masks that span exactly 2 chunks."""
    chunk_size = 10
    sampler = FragmentedRandomSampler(
        chunk_size=chunk_size,
        preload_nchunks=2,
        batch_size=5,
        masks=[slice(0, 20), slice(40, 60)],
        num_samples=15,
        rng=np.random.default_rng(42),
    )
    all_indices, all_chunks, splits = collect_indices(sampler, n_obs=60)

    assert len(splits) == math.ceil(15 / 5)
    # All indices must be in one of the two mask ranges
    mask_set = set(range(0, 20)) | set(range(40, 60))
    assert set(all_indices).issubset(mask_set)


def test_remainder_handling():
    """Test that remainder samples are handled correctly with drop_last."""
    num_samples = 23
    batch_size = 5
    chunk_size = 10
    sampler = FragmentedRandomSampler(
        chunk_size=chunk_size,
        preload_nchunks=2,
        batch_size=batch_size,
        masks=[slice(0, 100)],
        num_samples=num_samples,
        drop_last=False,
        rng=np.random.default_rng(42),
    )

    _, _, splits = collect_indices(sampler, n_obs=100)

    expected_batches = math.ceil(num_samples / batch_size)
    assert len(splits) == expected_batches

    # Last batch should have the remainder
    last_batch_size = num_samples % batch_size or batch_size
    assert len(splits[-1]) == last_batch_size
