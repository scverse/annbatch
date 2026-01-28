"""Tests for ChunkSampler."""

from __future__ import annotations

import numpy as np
import pytest

from annbatch import ChunkSampler
from annbatch.abc import Sampler

# TODO(selmanozleyen): Check for the validation within the _get_worker_handle method. Mock worker handle wouldn't make sense
# but overall one must  also think about how validation can't be independent of the worker handle.


def collect_indices(sampler, n_obs):
    """Helper to collect all indices from sampler."""
    indices = []
    for load_request in sampler.sample(n_obs):
        assert len(load_request["splits"]) > 0, "splits must be non-empty"
        assert all(len(s) > 0 for s in load_request["splits"]), "splits must be non-empty"
        for s in load_request["chunks"]:
            indices.extend(range(s.start, s.stop))
    return indices


class MockWorkerHandle:
    """Simulates torch worker context for testing without actual DataLoader."""

    def __init__(self, worker_id: int, num_workers: int, seed: int = 42):
        self.worker_id = worker_id
        self._num_workers = num_workers
        self._rng = np.random.default_rng(seed)

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def shuffle(self, obj):
        self._rng.shuffle(obj)

    def get_part_for_worker(self, obj: np.ndarray) -> np.ndarray:
        return np.array_split(obj, self._num_workers)[self.worker_id]


class ChunkSamplerWithMockWorkerHandle(ChunkSampler):
    def set_worker_handle(self, worker_handle: MockWorkerHandle):
        self.worker_handle = worker_handle

    def _get_worker_handle(self) -> MockWorkerHandle | None:
        return self.worker_handle


# =============================================================================
# Mask coverage tests
# =============================================================================


@pytest.mark.parametrize(
    "n_obs,chunk_size,start,stop,batch_size,preload_nchunks,shuffle",
    [
        # Basic full dataset
        pytest.param(100, 10, None, None, 5, 2, False, id="full_dataset"),
        # mask.start only
        pytest.param(100, 10, 30, None, 5, 2, False, id="start_at_chunk_boundary"),
        pytest.param(100, 10, 35, None, 5, 2, False, id="start_not_at_chunk_boundary"),
        pytest.param(120, 12, 90, None, 3, 1, False, id="start_near_end"),
        pytest.param(100, 10, 20, None, 5, 2, False, id="start_mask_stop_none"),
        # mask.stop only
        pytest.param(50, 10, None, 50, 5, 2, False, id="stop_at_chunk_boundary"),
        pytest.param(47, 10, None, 47, 5, 2, False, id="stop_not_at_chunk_boundary"),
        # Both bounds
        pytest.param(60, 10, 20, 60, 5, 2, False, id="both_at_chunk_boundaries"),
        pytest.param(67, 10, 23, 67, 5, 2, False, id="both_not_at_chunk_boundaries"),
        pytest.param(28, 10, 22, 28, 2, 1, False, id="single_chunk_span"),
        pytest.param(100, 10, 15, 85, 5, 2, False, id="both_non_aligned"),
        pytest.param(100, 10, 20, 80, 5, 2, False, id="both_aligned"),
        # Edge cases
        pytest.param(100, 10, 95, 100, 10, 1, False, id="very_small_mask"),
        # With shuffle
        pytest.param(100, 10, 30, None, 5, 2, True, id="shuffle_with_start"),
        pytest.param(75, 10, 25, 75, 5, 2, True, id="shuffle_with_both_bounds"),
    ],
)
def test_mask_coverage(n_obs, chunk_size, start, stop, batch_size, preload_nchunks, shuffle):
    """Test sampler covers exactly the expected range, and ordering is correct when not shuffled."""
    sampler = ChunkSampler(
        mask=slice(start, stop),
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        shuffle=shuffle,
        rng=np.random.default_rng(42) if shuffle else None,
    )

    expected_start = start if start is not None else 0
    expected_stop = stop if stop is not None else n_obs
    expected_indices = list(range(expected_start, expected_stop))

    all_indices = collect_indices(sampler, n_obs)

    # Always check coverage
    if shuffle:
        assert set(all_indices) == set(expected_indices), "Sampler should cover all expected indices"
    else:
        assert all_indices == expected_indices, f"all_indices: {all_indices} != expected_indices: {expected_indices}"

    sampler.validate(n_obs)


def test_batch_sizes_match_expected_pattern():
    """Test that batch sizes match expected pattern."""
    n_obs, chunk_size, preload_nchunks, batch_size = 103, 10, 2, 5
    # last chunk is incomplete and is also the last batch in the load request
    expected_last_chunk_size = 3
    expected_last_batch_size = 3
    expected_last_num_splits = 1
    expected_num_load_requests = 6
    sampler = ChunkSampler(
        mask=slice(0, None),
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
    )

    all_requests = list(sampler.sample(n_obs))
    assert len(all_requests) == expected_num_load_requests
    for req_idx, load_request in enumerate(all_requests[:-1]):
        assert all(chunk.stop - chunk.start == chunk_size for chunk in load_request["chunks"]), (
            f"chunk size mismatch at request {req_idx}:",
            f"chunks: {load_request['chunks']}",
        )
        assert all(len(split) == batch_size for split in load_request["splits"]), (
            f"batch size mismatch at request {req_idx}:splits: {load_request['splits']}"
        )
    last_request = all_requests[-1]
    assert len(last_request["splits"]) == expected_last_num_splits, "last request num splits mismatch"
    assert all(chunk.stop - chunk.start == expected_last_chunk_size for chunk in last_request["chunks"]), (
        "last request chunk size mismatch",
        f"chunks: {last_request['chunks']}",
    )
    assert all(len(split) == expected_last_batch_size for split in last_request["splits"]), (
        "last request batch size mismatch",
        f"splits: {last_request['splits']}",
    )


# =============================================================================
# Worker tests
# =============================================================================


@pytest.mark.parametrize(
    "n_obs,chunk_size,preload_nchunks,batch_size,num_workers,drop_last",
    [
        pytest.param(200, 10, 2, 10, 2, True, id="two_workers"),
        pytest.param(300, 10, 3, 10, 3, True, id="three_workers"),
        # checks how it works with batch_size=1 since it is the default case and might be used in torch later
        pytest.param(600, 10, 4, 1, 4, False, id="batch_size_one_torch_dataloader_case"),
        pytest.param(100, 10, 4, 1, 1, False, id="batch_size_one_single_worker_case"),
        pytest.param(95, 10, 4, 1, 1, False, id="batch_size_one_non_divisible_obs_case"),
        pytest.param(100, 10, 4, 1, 3, False, id="batch_size_one_three_workers_uneven_case"),
    ],
)
def test_workers_cover_full_dataset_without_overlap(
    n_obs, chunk_size, preload_nchunks, batch_size, num_workers, drop_last
):
    """Test workers cover full dataset without overlap. Also checks if there are empty splits in any of the load requests."""
    all_worker_indices = []
    for worker_id in range(num_workers):
        worker_handle = MockWorkerHandle(worker_id, num_workers)
        sampler = ChunkSamplerWithMockWorkerHandle(
            mask=slice(0, None),
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            drop_last=drop_last,
        )
        sampler.set_worker_handle(worker_handle)
        all_worker_indices.append(collect_indices(sampler, n_obs))

    # All workers should have disjoint chunks
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            assert set(all_worker_indices[i]).isdisjoint(all_worker_indices[j])

    # Together they cover the full dataset
    assert set().union(*all_worker_indices) == set(range(n_obs))


# =============================================================================
# Validation tests
# =============================================================================


@pytest.mark.parametrize(
    "mask,n_obs,error_match",
    [
        pytest.param(slice(0, 100), 100, None, id="valid_config"),
        pytest.param(slice(0, 200), 100, "mask.stop.*exceeds loader n_obs", id="stop_exceeds_n_obs"),
    ],
)
def test_validate(mask, n_obs, error_match):
    """Test validate behavior for various configurations."""
    sampler = ChunkSampler(mask=mask, batch_size=5, chunk_size=10, preload_nchunks=2)
    if error_match:
        with pytest.raises(ValueError, match=error_match):
            sampler.validate(n_obs=n_obs)
    else:
        sampler.validate(n_obs=n_obs)


@pytest.mark.parametrize(
    "mask,error_match",
    [
        pytest.param(slice(-1, 100), "mask.start must be >= 0", id="negative_start"),
        pytest.param(slice(50, 50), "mask.start must be < mask.stop", id="start_equals_stop"),
        pytest.param(slice(100, 50), "mask.start must be < mask.stop", id="start_greater_than_stop"),
        pytest.param(slice(0, 100, 2), "mask.step must be 1, but got 2", id="step_not_one"),
    ],
)
def test_invalid_mask_raises(mask, error_match):
    """Test that invalid mask configurations raise ValueError at construction."""
    with pytest.raises(ValueError, match=error_match):
        ChunkSampler(mask=mask, batch_size=5, chunk_size=10, preload_nchunks=2)


# =============================================================================
# n_obs change tests (To verify nothing is cached between calls.)
# =============================================================================


@pytest.mark.parametrize(
    "n_obs_values,expected_ranges",
    [
        pytest.param([50, 100], [range(50), range(100)], id="increase_changes_result"),
        pytest.param([100, 100], [range(100), range(100)], id="same_gives_same_coverage"),
    ],
)
def test_n_obs_coverage(n_obs_values, expected_ranges):
    """Test that n_obs changes affect sampling results appropriately."""
    sampler = ChunkSampler(mask=slice(0, None), batch_size=5, chunk_size=10, preload_nchunks=2, shuffle=False)

    results = [collect_indices(sampler, n) for n in n_obs_values]

    for result, expected in zip(results, expected_ranges, strict=True):
        assert result == list(expected), f"result: {result} != expected: {expected}"


# =============================================================================
# Automatic batching tests (when splits not provided)
# =============================================================================


class SimpleSampler(Sampler):
    """Test sampler that yields LoadRequests without splits."""

    def __init__(self, batch_size: int | None, provide_splits: bool = False, shuffle: bool = True):
        self._batch_size = batch_size
        self._provide_splits = provide_splits
        self._shuffle = shuffle

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    def validate(self, n_obs: int) -> None:
        """No validation needed for test sampler."""
        pass

    def _sample(self, n_obs: int):
        """Yield a simple LoadRequest with chunks but no splits."""
        # Create chunks of size 10
        chunk_size = 10
        if self._provide_splits:
            # Provide splits explicitly - yield separate LoadRequests per chunk
            for start in range(0, n_obs, chunk_size):
                stop = min(start + chunk_size, n_obs)
                indices = np.arange(stop - start)
                yield {"chunks": [slice(start, stop)], "splits": [indices]}
        else:
            # Don't provide splits - yield single LoadRequest with all chunks
            # so base class can batch across all data
            chunks = []
            for start in range(0, n_obs, chunk_size):
                stop = min(start + chunk_size, n_obs)
                chunks.append(slice(start, stop))
            yield {"chunks": chunks}


def test_automatic_batching_without_splits():
    """Test that base Sampler class automatically generates splits when not provided."""
    batch_size = 3
    n_obs = 25
    sampler = SimpleSampler(batch_size=batch_size, provide_splits=False)

    all_indices = []
    all_batch_sizes = []

    for load_request in sampler.sample(n_obs):
        # Verify splits were added by base class
        assert "splits" in load_request
        assert load_request["splits"] is not None
        assert len(load_request["splits"]) > 0

        # Collect batch sizes
        for split in load_request["splits"]:
            all_batch_sizes.append(len(split))
            all_indices.extend(split)

    # Verify all indices are covered (should be 0-24)
    assert len(all_indices) == n_obs
    assert set(all_indices) == set(range(n_obs))

    # Verify batch sizes (all should be batch_size except possibly the last one)
    for batch_sz in all_batch_sizes[:-1]:
        assert batch_sz == batch_size

    # Last batch can be smaller or equal to batch_size
    assert all_batch_sizes[-1] <= batch_size
    assert all_batch_sizes[-1] > 0


def test_automatic_batching_requires_batch_size():
    """Test that automatic batching raises error when batch_size is None."""
    sampler = SimpleSampler(batch_size=None, provide_splits=False)
    n_obs = 20

    with pytest.raises(ValueError, match="batch_size must be set when splits are not provided"):
        list(sampler.sample(n_obs))


def test_explicit_splits_override_automatic_batching():
    """Test that when splits are explicitly provided, automatic batching is not used."""
    batch_size = 3
    n_obs = 20
    sampler = SimpleSampler(batch_size=batch_size, provide_splits=True)

    for load_request in sampler.sample(n_obs):
        # Verify splits exist
        assert "splits" in load_request
        # In our SimpleSampler with provide_splits=True, each chunk becomes one split
        # with sequential indices (not randomly batched)
        for split in load_request["splits"]:
            # Check that indices are sequential (which means auto-batching wasn't used)
            assert np.array_equal(split, np.arange(len(split)))


def test_automatic_batching_respects_shuffle_flag():
    """Test that automatic batching respects the shuffle parameter."""
    batch_size = 3
    n_obs = 25

    # Test with shuffle=False - should maintain order
    sampler_no_shuffle = SimpleSampler(batch_size=batch_size, provide_splits=False, shuffle=False)
    all_indices_no_shuffle = []

    for load_request in sampler_no_shuffle.sample(n_obs):
        for split in load_request["splits"]:
            all_indices_no_shuffle.extend(split)

    # Without shuffling, indices should be in order
    assert all_indices_no_shuffle == list(range(n_obs)), "Without shuffle, indices should be sequential"

    # Test with shuffle=True - should randomize order
    sampler_shuffle = SimpleSampler(batch_size=batch_size, provide_splits=False, shuffle=True)
    all_indices_shuffle = []

    for load_request in sampler_shuffle.sample(n_obs):
        for split in load_request["splits"]:
            all_indices_shuffle.extend(split)

    # With shuffling, indices should be different from sequential (with very high probability)
    # But should still cover all indices
    assert set(all_indices_shuffle) == set(range(n_obs)), "With shuffle, all indices should be covered"
    assert all_indices_shuffle != list(range(n_obs)), "With shuffle, indices should not be sequential"
