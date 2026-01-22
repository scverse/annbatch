"""Tests for ChunkSampler."""

from __future__ import annotations

import numpy as np
import pytest

from annbatch import ChunkSampler

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
