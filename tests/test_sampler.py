"""Tests for SliceSampler."""

from __future__ import annotations

import numpy as np
import pytest

from annbatch.sampler import SliceSampler


class TestSliceSamplerBasic:
    """Tests for basic SliceSampler functionality."""

    def test_full_dataset(self):
        """Test sampler covers full dataset"""
        n_obs = 100
        slice_size = 10
        preload_nslices = 2
        batch_size = 5

        sampler = SliceSampler(
            mask=slice(0, n_obs),
            batch_size=batch_size,
            slice_size=slice_size,
            preload_nslices=preload_nslices,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        assert all_indices == set(range(n_obs))

    def test_batch_sizes(self):
        """Test that batch sizes match expected carry-over pattern."""
        n_obs = 100
        slice_size = 10
        preload_nslices = 2
        batch_size = 7

        # Example with these params (slice_size=10, preload_nslices=2, batch_size=7):
        # Each iter loads slice_size * preload_nslices = 20 obs
        # Iter 1: 20 obs → [7, 7, 6], leftover=6
        # Iter 2: 20 + 6 = 26 → [7, 7, 7, 5], leftover=5
        # Iter 3: 20 + 5 = 25 → [7, 7, 7, 4], leftover=4
        # Iter 4: 20 + 4 = 24 → [7, 7, 7, 3], leftover=3
        # Iter 5: 20 + 3 = 23 → [7, 7, 7, 2], final partial yielded
        import math

        obs_per_iter = slice_size * preload_nslices
        n_iters = math.ceil(n_obs / obs_per_iter)

        expected_sizes_per_iter = []
        leftover = 0
        for _ in range(n_iters):
            total_obs = obs_per_iter + leftover
            n_full_batches = total_obs // batch_size
            remainder = total_obs % batch_size
            sizes = [batch_size] * n_full_batches
            if remainder > 0:
                sizes.append(remainder)
            expected_sizes_per_iter.append(sizes)
            leftover = remainder

        sampler = SliceSampler(
            mask=slice(0, n_obs),
            batch_size=batch_size,
            slice_size=slice_size,
            preload_nslices=preload_nslices,
        )

        for i, load_request in enumerate(sampler):
            actual_sizes = [len(split) for split in load_request.splits]
            assert actual_sizes == expected_sizes_per_iter[i], (
                f"Iter {i}: expected {expected_sizes_per_iter[i]}, got {actual_sizes}"
            )


class TestSliceSamplerMaskStart:
    """Tests for SliceSampler with non-zero mask.start."""

    def test_mask_start_at_slice_boundary(self):
        """Test mask.start aligned with slice boundary."""
        n_obs = 100
        slice_size = 10
        start = 30  # Aligned with slice 3

        sampler = SliceSampler(
            mask=slice(start, n_obs),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start
        assert max(all_indices) == n_obs - 1

    def test_mask_start_not_at_slice_boundary(self):
        """Test mask.start not aligned with slice boundary."""
        n_obs = 100
        slice_size = 10
        start = 35  # Not aligned - middle of slice 3

        sampler = SliceSampler(
            mask=slice(start, n_obs),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start

    def test_mask_start_near_end(self):
        """Test mask.start near the end of dataset."""
        n_obs = 100
        slice_size = 10
        start = 90

        sampler = SliceSampler(
            mask=slice(start, n_obs),
            batch_size=3,
            slice_size=slice_size,
            preload_nslices=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, n_obs))
        assert all_indices == expected
        assert len(all_indices) == 10


class TestSliceSamplerMaskStop:
    """Tests for SliceSampler with custom mask.stop."""

    def test_mask_stop_at_slice_boundary(self):
        """Test mask.stop aligned with slice boundary."""
        slice_size = 10
        stop = 50  # Aligned with end of slice 4

        sampler = SliceSampler(
            mask=slice(0, stop),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(0, stop))
        assert all_indices == expected
        assert max(all_indices) == stop - 1

    def test_mask_stop_not_at_slice_boundary(self):
        """Test mask.stop not aligned with slice boundary."""
        slice_size = 10
        stop = 47  # Middle of slice 4

        sampler = SliceSampler(
            mask=slice(0, stop),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(0, stop))
        assert all_indices == expected
        assert max(all_indices) == stop - 1


class TestSliceSamplerBothMaskBounds:
    """Tests for SliceSampler with both mask.start and mask.stop."""

    def test_both_at_slice_boundaries(self):
        """Test both start and stop aligned with slice boundaries."""
        slice_size = 10
        start = 20
        stop = 60

        sampler = SliceSampler(
            mask=slice(start, stop),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, stop))
        assert all_indices == expected
        assert len(all_indices) == stop - start

    def test_both_not_at_slice_boundaries(self):
        """Test both start and stop not aligned with slice boundaries."""
        slice_size = 10
        start = 23
        stop = 67

        sampler = SliceSampler(
            mask=slice(start, stop),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, stop))
        assert all_indices == expected
        assert min(all_indices) == start
        assert max(all_indices) == stop - 1

    def test_single_slice_span(self):
        """Test start and stop within a single slice."""
        slice_size = 10
        start = 22
        stop = 28  # Same slice as start

        sampler = SliceSampler(
            mask=slice(start, stop),
            batch_size=2,
            slice_size=slice_size,
            preload_nslices=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, stop))
        assert all_indices == expected
        assert len(all_indices) == 6

    def test_worker_shard_simulation(self):
        """Test simulating DataLoader worker sharding (600 obs, 4 workers)."""
        n_obs = 600
        slice_size = 10
        num_workers = 4
        per_worker = n_obs // num_workers  # 150

        all_worker_indices = set()
        for worker_id in range(num_workers):
            start = worker_id * per_worker
            if worker_id == num_workers - 1:
                stop = n_obs
            else:
                stop = start + per_worker

            sampler = SliceSampler(
                mask=slice(start, stop),
                batch_size=10,
                slice_size=slice_size,
                preload_nslices=4,
            )

            worker_indices = set()
            for load_request in sampler:
                for s in load_request.slices:
                    worker_indices.update(range(s.start, s.stop))

            # Check this worker got the right range
            expected = set(range(start, stop))
            assert worker_indices == expected

            # Add to global set
            all_worker_indices.update(worker_indices)

        # All workers together should cover the full dataset exactly once
        assert all_worker_indices == set(range(n_obs))


class TestSliceSamplerWithShuffle:
    """Tests for SliceSampler with shuffling enabled."""

    def test_shuffle_with_mask_start(self):
        """Test shuffle works correctly with non-zero mask.start."""
        n_obs = 100
        slice_size = 10
        start = 30

        sampler = SliceSampler(
            mask=slice(start, n_obs),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, n_obs))
        assert all_indices == expected

    def test_shuffle_with_both_bounds(self):
        """Test shuffle works correctly with both mask.start and mask.stop."""
        slice_size = 10
        start = 25
        stop = 75

        sampler = SliceSampler(
            mask=slice(start, stop),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, stop))
        assert all_indices == expected


class TestSliceSamplerEdgeCases:
    """Tests for edge cases."""

    def test_very_small_mask(self):
        """Test with a very small mask (smaller than batch_size and slice_size)."""
        slice_size = 10
        start = 95
        stop = 100  # Only 5 observations

        sampler = SliceSampler(
            mask=slice(start, stop),
            batch_size=10,  # Larger than mask size
            slice_size=slice_size,
            preload_nslices=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, stop))
        assert all_indices == expected

    def test_mask_start_equals_slice_size(self):
        """Test mask.start exactly equals slice_size."""
        n_obs = 100
        slice_size = 10
        start = 10  # Exactly one slice in

        sampler = SliceSampler(
            mask=slice(start, n_obs),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start

    @pytest.mark.parametrize(
        "start,stop",
        [
            (0, 100),  # Full range
            (15, 85),  # Both non-aligned
            (20, 80),  # Both aligned
            (0, 50),  # Only stop set
            (50, 100),  # Only start set (effectively)
        ],
    )
    def test_parametrized_ranges(self, start, stop):
        """Test various start/stop combinations cover correct range."""
        slice_size = 10

        sampler = SliceSampler(
            mask=slice(start, stop),
            batch_size=5,
            slice_size=slice_size,
            preload_nslices=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start, stop))
        assert all_indices == expected


class MockWorkerHandle:
    """Simulates torch worker context for testing without actual DataLoader."""

    def __init__(self, worker_id: int, num_workers: int, seed: int = 42):
        self.worker_id = worker_id
        self._num_workers = num_workers
        self._rng = np.random.default_rng(seed)  # Same seed = consistent shuffle across workers

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def shuffle(self, obj):
        self._rng.shuffle(obj)

    def get_part_for_worker(self, obj: np.ndarray) -> np.ndarray:
        chunks_split = np.array_split(obj, self._num_workers)
        return chunks_split[self.worker_id]


class TestSliceSamplerWithWorkers:
    """Tests for SliceSampler with simulated DataLoader workers."""

    def test_two_workers_divisible_config(self):
        """Test 2 workers with divisible config cover full dataset without overlap."""
        n_obs = 200
        slice_size = 10
        preload_nslices = 2
        batch_size = 10  # 10 * 2 = 20, divisible by 10
        num_workers = 2

        all_worker_indices = []
        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            sampler = SliceSampler(
                mask=slice(0, n_obs),
                batch_size=batch_size,
                slice_size=slice_size,
                preload_nslices=preload_nslices,
            )
            sampler.set_worker_handle(worker_handle)

            worker_indices = set()
            for load_request in sampler:
                for s in load_request.slices:
                    worker_indices.update(range(s.start, s.stop))
            all_worker_indices.append(worker_indices)

        # Workers should have disjoint chunks
        assert all_worker_indices[0].isdisjoint(all_worker_indices[1])
        # Together they cover the full dataset
        assert all_worker_indices[0] | all_worker_indices[1] == set(range(n_obs))

    def test_three_workers_divisible_config(self):
        """Test 3 workers with divisible config (odd worker count)."""
        n_obs = 300
        slice_size = 10
        preload_nslices = 3
        batch_size = 10  # 10 * 3 = 30, divisible by 10
        num_workers = 3

        all_worker_indices = []
        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            sampler = SliceSampler(
                mask=slice(0, n_obs),
                batch_size=batch_size,
                slice_size=slice_size,
                preload_nslices=preload_nslices,
            )
            sampler.set_worker_handle(worker_handle)

            worker_indices = set()
            for load_request in sampler:
                for s in load_request.slices:
                    worker_indices.update(range(s.start, s.stop))
            all_worker_indices.append(worker_indices)

        # All workers should have disjoint chunks
        for i in range(num_workers):
            for j in range(i + 1, num_workers):
                assert all_worker_indices[i].isdisjoint(all_worker_indices[j])
        # Together they cover the full dataset
        combined = set()
        for indices in all_worker_indices:
            combined |= indices
        assert combined == set(range(n_obs))

    def test_workers_drop_last_warns(self):
        """Test that drop_last=True with workers emits warning."""
        worker_handle = MockWorkerHandle(0, 2)

        with pytest.warns(UserWarning, match="multiple workers"):
            sampler = SliceSampler(
                mask=slice(0, 100),
                batch_size=7,  # Non-divisible
                slice_size=10,
                preload_nslices=2,
                drop_last=True,
            )
            sampler.set_worker_handle(worker_handle)

    def test_workers_non_divisible_without_drop_last_raises(self):
        """Test that non-divisible config without drop_last raises ValueError."""
        worker_handle = MockWorkerHandle(0, 2)

        with pytest.raises(ValueError, match="divisible by batch_size"):
            sampler = SliceSampler(
                mask=slice(0, 100),
                batch_size=7,  # 10 * 2 = 20, not divisible by 7
                slice_size=10,
                preload_nslices=2,
                drop_last=False,
            )
            sampler.set_worker_handle(worker_handle)

    def test_single_worker_no_divisibility_check(self):
        """Test that non-divisible config with num_workers=1 does NOT raise."""
        worker_handle = MockWorkerHandle(0, num_workers=1)

        # This would raise with num_workers > 1, but should be fine with 1
        sampler = SliceSampler(
            mask=slice(0, 100),
            batch_size=7,  # 10 * 2 = 20, not divisible by 7
            slice_size=10,
            preload_nslices=2,
            drop_last=False,
        )
        # Should NOT raise - single worker doesn't need divisibility
        sampler.set_worker_handle(worker_handle)

    def test_single_worker_drop_last_no_warning(self):
        """Test that drop_last=True with num_workers=1 does NOT warn."""
        import warnings

        worker_handle = MockWorkerHandle(0, num_workers=1)

        sampler = SliceSampler(
            mask=slice(0, 100),
            batch_size=7,
            slice_size=10,
            preload_nslices=2,
            drop_last=True,
        )

        # Should NOT warn - single worker doesn't have the multi-worker drop issue
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            sampler.set_worker_handle(worker_handle)  # Should not raise

    def test_two_workers_drop_last_drops_per_worker(self):
        """Test drop_last=True drops only the final partial batch (intermediate partials are for carry-over)."""
        n_obs = 200
        slice_size = 10
        preload_nslices = 2
        batch_size = 7  # Non-divisible: 20 / 7 = 2 full batches + 6 leftover per iter
        num_workers = 2

        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            with pytest.warns(UserWarning):
                sampler = SliceSampler(
                    mask=slice(0, n_obs),
                    batch_size=batch_size,
                    slice_size=slice_size,
                    preload_nslices=preload_nslices,
                    drop_last=True,
                )
                sampler.set_worker_handle(worker_handle)

            all_requests = list(sampler)
            # On the final iteration, all splits should be batch_size
            # (the final partial is dropped when drop_last=True)
            if all_requests:
                final_request = all_requests[-1]
                for split in final_request.splits:
                    assert len(split) == batch_size, (
                        f"Worker {worker_id}: final request should have no partial, "
                        f"expected {batch_size}, got {len(split)}"
                    )


class TestSliceSamplerValidation:
    """Tests for SliceSampler validation."""

    def test_validate_passes_for_valid_config(self):
        """Test validate passes for a valid configuration."""
        sampler = SliceSampler(
            mask=slice(0, 100),
            batch_size=5,
            slice_size=10,
            preload_nslices=2,
        )
        # Should not raise
        sampler.validate(n_obs=100)

    def test_validate_passes_when_stop_equals_n_obs(self):
        """Test validate passes when mask.stop equals n_obs."""
        sampler = SliceSampler(
            mask=slice(0, 100),
            batch_size=5,
            slice_size=10,
            preload_nslices=2,
        )
        # Should not raise - mask.stop == n_obs is valid
        sampler.validate(n_obs=100)

    def test_validate_mask_stop_exceeds_n_obs(self):
        """Test validate raises when mask.stop > n_obs."""
        sampler = SliceSampler(
            mask=slice(0, 200),
            batch_size=5,
            slice_size=10,
            preload_nslices=2,
        )
        with pytest.raises(ValueError, match="mask.stop.*exceeds loader n_obs"):
            sampler.validate(n_obs=100)

    def test_invalid_mask_start(self):
        """Test that negative mask.start raises."""
        with pytest.raises(ValueError, match="mask.start must be >= 0"):
            SliceSampler(
                mask=slice(-1, 100),
                batch_size=5,
                slice_size=10,
                preload_nslices=2,
            )

    def test_mask_start_equals_stop_raises(self):
        """Test that mask.start == mask.stop raises."""
        with pytest.raises(ValueError, match="mask.start must be >= 0 and < mask.stop"):
            SliceSampler(
                mask=slice(50, 50),
                batch_size=5,
                slice_size=10,
                preload_nslices=2,
            )

    def test_mask_start_greater_than_stop_raises(self):
        """Test that mask.start > mask.stop raises."""
        with pytest.raises(ValueError, match="mask.start must be >= 0 and < mask.stop"):
            SliceSampler(
                mask=slice(100, 50),
                batch_size=5,
                slice_size=10,
                preload_nslices=2,
            )

    def test_mask_stop_none_raises(self):
        """Test that mask.stop=None raises."""
        with pytest.raises(ValueError, match="mask.stop must be specified"):
            SliceSampler(
                mask=slice(0, None),
                batch_size=5,
                slice_size=10,
                preload_nslices=2,
            )
