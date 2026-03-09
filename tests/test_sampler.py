"""Tests for ChunkSampler and ChunkSamplerWithReplacement."""

from __future__ import annotations

import math
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from annbatch import ChunkSampler, ChunkSamplerWithReplacement
from annbatch.abc import Sampler
from annbatch.samplers._utils import WorkerInfo


def collect_indices(sampler: Sampler, n_obs: int) -> list[int]:
    """Helper to collect all indices from sampler."""
    indices: list[int] = []
    for load_request in sampler.sample(n_obs):
        assert len(load_request["splits"]) > 0, "splits must be non-empty"
        assert all(len(s) > 0 for s in load_request["splits"]), "splits must be non-empty"
        for s in load_request["chunks"]:
            indices.extend(range(s.start, s.stop))
    return indices


# =============================================================================
# Mask coverage tests
# =============================================================================


@pytest.mark.parametrize(
    ("n_obs", "chunk_size", "start", "stop", "batch_size", "preload_nchunks", "shuffle", "drop_last"),
    [
        # Basic full dataset
        pytest.param(100, 10, None, None, 5, 2, False, False, id="full_dataset"),
        # mask.start only
        pytest.param(100, 10, 30, None, 5, 2, False, False, id="start_at_chunk_boundary"),
        pytest.param(100, 10, 35, None, 5, 2, False, False, id="start_not_at_chunk_boundary"),
        pytest.param(120, 12, 90, None, 3, 1, False, False, id="start_near_end"),
        pytest.param(100, 10, 20, None, 5, 2, False, False, id="start_mask_stop_none"),
        # mask.stop only
        pytest.param(50, 10, None, 50, 5, 2, False, False, id="stop_at_chunk_boundary"),
        pytest.param(47, 10, None, 47, 5, 2, False, False, id="stop_not_at_chunk_boundary"),
        # Both bounds
        pytest.param(60, 10, 20, 60, 5, 2, False, False, id="both_at_chunk_boundaries"),
        pytest.param(67, 10, 23, 67, 5, 2, False, False, id="both_not_at_chunk_boundaries"),
        pytest.param(28, 10, 22, 28, 2, 1, False, False, id="single_chunk_span"),
        pytest.param(100, 10, 15, 85, 5, 2, False, False, id="both_non_aligned"),
        pytest.param(100, 10, 20, 80, 5, 2, False, False, id="both_aligned"),
        # Edge cases
        pytest.param(100, 10, 95, 100, 10, 1, False, False, id="very_small_mask"),
        # With shuffle
        pytest.param(100, 10, 30, None, 5, 2, True, False, id="shuffle_with_start"),
        pytest.param(75, 10, 25, 75, 5, 2, True, False, id="shuffle_with_both_bounds"),
        # drop_last edge cases: remainder less than batch_size
        pytest.param(45, 20, None, None, 10, 2, False, True, id="drop_last_remainder_less_than_batch"),
        pytest.param(5, 20, None, None, 10, 2, False, True, id="drop_last_total_less_than_batch"),
    ],
)
def test_mask_coverage(
    n_obs: int,
    chunk_size: int,
    start: int | None,
    stop: int | None,
    batch_size: int,
    preload_nchunks: int,
    shuffle: bool,
    drop_last: bool,
):
    """Test sampler covers exactly the expected range, and ordering is correct when not shuffled."""
    sampler = ChunkSampler(
        mask=slice(start, stop),
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        shuffle=shuffle,
        drop_last=drop_last,
        rng=np.random.default_rng(42) if shuffle else None,
    )

    expected_start = start if start is not None else 0
    expected_stop = stop if stop is not None else n_obs
    if drop_last:
        # With drop_last, only complete batches are yielded
        total_obs = expected_stop - expected_start
        expected_stop = expected_start + (total_obs // batch_size) * batch_size
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
    ("n_obs", "chunk_size", "preload_nchunks", "batch_size", "num_workers", "drop_last"),
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
    n_obs: int, chunk_size: int, preload_nchunks: int, batch_size: int, num_workers: int, drop_last: bool
):
    """Test workers cover full dataset without overlap. Also checks if there are empty splits in any of the load requests."""
    all_worker_indices = []
    for worker_id in range(num_workers):
        sampler = ChunkSampler(
            mask=slice(0, None),
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            drop_last=drop_last,
        )
        # we patch the function where it is called
        with patch(
            "annbatch.samplers._chunk_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=worker_id, num_workers=num_workers),
        ):
            all_worker_indices.append(collect_indices(sampler, n_obs))

    # All workers should have disjoint chunks
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            assert set(all_worker_indices[i]).isdisjoint(all_worker_indices[j])

    # Together they cover the full dataset
    assert set().union(*all_worker_indices) == set(range(n_obs))


@pytest.mark.parametrize(
    "sampler_class",
    [partial(ChunkSampler, shuffle=True), partial(ChunkSamplerWithReplacement, n_iters=10)],
    ids=["without_replacement", "with_replacement"],
)
def test_batch_shuffle_is_reproducible_with_same_seed_rng(sampler_class):
    """Test that sampling is fully reproducible with the same seed and differs with another."""
    n_obs, chunk_size, preload_nchunks, batch_size = 100, 10, 2, 5

    def make_sampler(seed: int) -> Sampler:
        return sampler_class(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            rng=np.random.default_rng(seed),
        )

    indices1 = collect_indices(make_sampler(42), n_obs)
    indices2 = collect_indices(make_sampler(42), n_obs)
    indices3 = collect_indices(make_sampler(99), n_obs)

    assert indices1 == indices2, "Sampling should be reproducible with same seed"
    assert indices1 != indices3, "Different seeds should produce different results"


# =============================================================================
# Validation tests
# =============================================================================


@pytest.mark.parametrize(
    ("mask", "n_obs", "error_match"),
    [
        pytest.param(slice(0, 100), 100, None, id="valid_config"),
        pytest.param(slice(0, 200), 100, "mask.stop.*exceeds loader n_obs", id="stop_exceeds_n_obs"),
    ],
)
def test_validate(mask: slice, n_obs: int, error_match: str | None):
    """Test validate behavior for various configurations."""
    sampler = ChunkSampler(mask=mask, batch_size=5, chunk_size=10, preload_nchunks=2)
    if error_match:
        with pytest.raises(ValueError, match=error_match):
            sampler.validate(n_obs=n_obs)
    else:
        sampler.validate(n_obs=n_obs)


@pytest.mark.parametrize(
    "sampler_class",
    [ChunkSampler, partial(ChunkSamplerWithReplacement, n_iters=10)],
    ids=["without_replacement", "with_replacement"],
)
@pytest.mark.parametrize(
    ("mask", "error_match"),
    [
        pytest.param(slice(-1, 100), "mask.start must be >= 0", id="negative_start"),
        pytest.param(slice(50, 50), "mask.start must be < mask.stop", id="start_equals_stop"),
        pytest.param(slice(100, 50), "mask.start must be < mask.stop", id="start_greater_than_stop"),
        pytest.param(slice(0, 100, 2), "mask.step must be 1, but got 2", id="step_not_one"),
    ],
)
def test_invalid_mask_raises(sampler_class: type[Sampler], mask: slice, error_match: str):
    """Test that invalid mask configurations raise ValueError at construction."""
    with pytest.raises(ValueError, match=error_match):
        sampler_class(chunk_size=10, preload_nchunks=2, batch_size=5, mask=mask)


@pytest.mark.parametrize(
    ("kwargs", "n_obs", "error_match"),
    [
        pytest.param({"n_iters": 0}, None, "n_iters", id="n_iters_zero"),
        pytest.param({"n_iters": -1}, None, "n_iters", id="n_iters_negative"),
        pytest.param({"n_iters": 3}, 5, "smaller than chunk_size", id="n_obs_smaller_than_chunk"),
        pytest.param(
            {"n_iters": 3, "mask": slice(50, 55)}, 100, "smaller than chunk_size", id="mask_range_smaller_than_chunk"
        ),
    ],
)
def test_invalid_sample_with_replacement(kwargs: dict, n_obs: int | None, error_match: str):
    """Test that invalid configurations raise ValueError for ChunkSamplerWithReplacement."""
    defaults = {"chunk_size": 10, "preload_nchunks": 2, "batch_size": 5}
    with pytest.raises(ValueError, match=error_match):
        sampler = ChunkSamplerWithReplacement(**(defaults | kwargs))
        if n_obs is not None:
            list(sampler.sample(n_obs))


@pytest.mark.parametrize(
    ("n_obs", "chunk_size", "preload_nchunks", "batch_size", "n_iters", "mask"),
    [
        pytest.param(100, 10, 2, 5, 10, slice(0, None), id="basic"),
        pytest.param(100, 10, 2, 5, 50, slice(0, None), id="more_iters_than_obs"),
        pytest.param(100, 10, 2, 5, 1, slice(0, None), id="single_iter"),
        pytest.param(100, 10, 2, 5, 10, slice(20, 80), id="with_mask"),
        pytest.param(103, 10, 2, 5, 20, slice(0, None), id="non_divisible_obs"),
        pytest.param(100, 10, 2, 5, 7, slice(0, None), id="tail_with_batch_lt_chunk"),
    ],
)
def test_replacement_invariants(
    n_obs: int, chunk_size: int, preload_nchunks: int, batch_size: int, n_iters: int, mask: slice
):
    """Test ChunkSamplerWithReplacement yields correct n_iters, chunks within bounds, uniform chunk sizes."""
    start = mask.start or 0
    stop = mask.stop or n_obs
    sampler = ChunkSamplerWithReplacement(
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        n_iters=n_iters,
        mask=mask,
        rng=np.random.default_rng(42),
    )
    count = 0
    for load_request in sampler.sample(n_obs):
        assert len(load_request["chunks"]) > 0, "Load request must have at least one chunk"
        for chunk in load_request["chunks"]:
            assert chunk.stop - chunk.start == chunk_size, f"Non-uniform chunk: {chunk}"
            assert chunk.start >= start, f"Chunk start {chunk.start} < mask start {start}"
            assert chunk.stop <= stop, f"Chunk stop {chunk.stop} > mask stop {stop}"
        count += len(load_request["splits"])
    assert count == n_iters, f"Expected {n_iters} batches, got {count}"


def test_replacement_with_multiple_workers_raises():
    """Test that ChunkSamplerWithReplacement raises when used with multiple workers."""
    sampler = ChunkSamplerWithReplacement(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=5,
        n_iters=20,
        rng=np.random.default_rng(42),
    )
    with (
        patch(
            "annbatch.samplers._chunk_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(ValueError, match="Multiple workers are not supported with this sampler"),
    ):
        list(sampler.sample(100))


@pytest.mark.parametrize(
    ("sampler", "n_obs", "expected"),
    [
        pytest.param(
            ChunkSamplerWithReplacement(
                chunk_size=10, preload_nchunks=2, batch_size=5, n_iters=50, rng=np.random.default_rng(42)
            ),
            100,
            50,
            id="replacement_returns_n_iters",
        ),
        pytest.param(
            ChunkSampler(chunk_size=10, preload_nchunks=2, batch_size=5),
            100,
            20,
            id="without_replacement_full_epoch",
        ),
        pytest.param(
            ChunkSampler(chunk_size=10, preload_nchunks=2, batch_size=5, drop_last=True),
            100,
            20,
            id="without_replacement_drop_last_exact",
        ),
        pytest.param(
            ChunkSampler(chunk_size=10, preload_nchunks=2, batch_size=5),
            103,
            21,
            id="without_replacement_ceil",
        ),
        pytest.param(
            ChunkSampler(chunk_size=10, preload_nchunks=2, batch_size=5, drop_last=True),
            103,
            20,
            id="without_replacement_drop_last_floor",
        ),
    ],
)
def test_n_iters_property(sampler: Sampler, n_obs: int, expected: int):
    """Test that n_iters() returns the correct value for different configurations."""
    assert sampler.n_iters(n_obs) == expected


# =============================================================================
# n_obs change tests (To verify nothing is cached between calls.)
# =============================================================================


@pytest.mark.parametrize(
    ("n_obs_values", "expected_ranges"),
    [
        pytest.param([50, 100], [range(50), range(100)], id="increase_changes_result"),
        pytest.param([100, 100], [range(100), range(100)], id="same_gives_same_coverage"),
    ],
)
def test_n_obs_coverage(n_obs_values: list[int], expected_ranges: list[range]):
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

    def __init__(self, batch_size: int | None, provide_splits: bool = False, shuffle: bool | None = True):
        self._batch_size = batch_size
        self._provide_splits = provide_splits
        self._shuffle = shuffle

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def shuffle(self) -> bool | None:
        return self._shuffle

    def n_iters(self, n_obs: int) -> int:
        if self._batch_size is None or self._batch_size == 0:
            return 1
        return math.ceil(n_obs / self._batch_size)

    def validate(self, n_obs: int) -> None:
        """No validation needed for test sampler."""
        pass

    def _sample(self, n_obs: int):
        """Yield LoadRequests with or without splits."""
        chunk_size = 10
        chunks = []
        for start in range(0, n_obs, chunk_size):
            stop = min(start + chunk_size, n_obs)
            if self._provide_splits:
                # Yield one LoadRequest per chunk with splits
                yield {"chunks": [slice(start, stop)], "splits": [np.arange(stop - start)]}
            else:
                # Accumulate chunks
                chunks.append(slice(start, stop))

        # Yield accumulated chunks without splits
        if not self._provide_splits:
            yield {"chunks": chunks}


@pytest.mark.parametrize(
    ("batch_size", "shuffle"),
    [
        pytest.param(None, True, id="missing_batch_size"),
        pytest.param(3, None, id="missing_shuffle"),
    ],
)
def test_automatic_batching_requires_batch_size_and_shuffle(batch_size: int | None, shuffle: bool | None):
    """Test that automatic batching raises error when batch_size or shuffle is None."""
    sampler = SimpleSampler(batch_size=batch_size, provide_splits=False, shuffle=shuffle)
    n_obs = 20

    with pytest.raises(ValueError):
        list(sampler.sample(n_obs))


def test_explicit_splits_override_automatic_batching():
    """Test that explicit splits are not overridden by automatic batching."""
    sampler = SimpleSampler(batch_size=3, provide_splits=True)

    for load_request in sampler.sample(n_obs=20):
        # Verify splits are sequential (not randomly batched)
        for split in load_request["splits"]:
            assert np.array_equal(split, np.arange(len(split)))


@pytest.mark.parametrize("shuffle", [False, True])
def test_automatic_batching_respects_shuffle_flag(shuffle: bool):
    """Test automatic batching generates splits and respects shuffle parameter."""
    batch_size, n_obs = 3, 25
    sampler = SimpleSampler(batch_size=batch_size, provide_splits=False, shuffle=shuffle)

    all_indices = []
    for load_request in sampler.sample(n_obs):
        assert "splits" in load_request and load_request["splits"]
        for split in load_request["splits"]:
            assert 0 < len(split) <= batch_size
            all_indices.extend(split)

    # Verify coverage
    assert set(all_indices) == set(range(n_obs))

    # Verify shuffle behavior
    if shuffle:
        assert all_indices != list(range(n_obs)), "Indices should be shuffled"
    else:
        assert all_indices == list(range(n_obs)), "Indices should be sequential"
