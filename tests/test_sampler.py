"""Tests for SliceSampler."""

from __future__ import annotations

import numpy as np
import pytest

from annbatch.sampler import SliceSampler


class TestSliceSamplerBasic:
    """Tests for basic SliceSampler functionality."""

    def test_full_dataset(self):
        """Test sampler covers full dataset when no start/end specified."""
        n_obs = 100
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 5

        sampler = SliceSampler(
            start_index=0,
            end_index=n_obs,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        assert all_indices == set(range(n_obs))

    def test_batch_sizes(self):
        """Test that batch sizes match expected carry-over pattern."""
        n_obs = 100
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 7

        # Example with these params (chunk_size=10, preload_nchunks=2, batch_size=7):
        # Each iter loads chunk_size * preload_nchunks = 20 obs
        # Iter 1: 20 obs → [7, 7, 6], leftover=6
        # Iter 2: 20 + 6 = 26 → [7, 7, 7, 5], leftover=5
        # Iter 3: 20 + 5 = 25 → [7, 7, 7, 4], leftover=4
        # Iter 4: 20 + 4 = 24 → [7, 7, 7, 3], leftover=3
        # Iter 5: 20 + 3 = 23 → [7, 7, 7, 2], final partial yielded
        import math

        obs_per_iter = chunk_size * preload_nchunks
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
            start_index=0,
            end_index=n_obs,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )

        for i, load_request in enumerate(sampler):
            actual_sizes = [len(split) for split in load_request.splits]
            assert actual_sizes == expected_sizes_per_iter[i], (
                f"Iter {i}: expected {expected_sizes_per_iter[i]}, got {actual_sizes}"
            )


class TestSliceSamplerStartIndex:
    """Tests for SliceSampler with non-zero start_index."""

    def test_start_index_at_chunk_boundary(self):
        """Test start_index aligned with chunk boundary."""
        n_obs = 100
        chunk_size = 10
        start_index = 30  # Aligned with chunk 3

        sampler = SliceSampler(
            start_index=start_index,
            end_index=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start_index
        assert max(all_indices) == n_obs - 1

    def test_start_index_not_at_chunk_boundary(self):
        """Test start_index not aligned with chunk boundary."""
        n_obs = 100
        chunk_size = 10
        start_index = 35  # Not aligned - middle of chunk 3

        sampler = SliceSampler(
            start_index=start_index,
            end_index=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start_index

    def test_start_index_near_end(self):
        """Test start_index near the end of dataset."""
        n_obs = 100
        chunk_size = 10
        start_index = 90

        sampler = SliceSampler(
            start_index=start_index,
            end_index=n_obs,
            batch_size=3,
            chunk_size=chunk_size,
            preload_nchunks=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert len(all_indices) == 10


class TestSliceSamplerEndIndex:
    """Tests for SliceSampler with custom end_index."""

    def test_end_index_at_chunk_boundary(self):
        """Test end_index aligned with chunk boundary."""
        chunk_size = 10
        end_index = 50  # Aligned with end of chunk 4

        sampler = SliceSampler(
            start_index=0,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(0, end_index))
        assert all_indices == expected
        assert max(all_indices) == end_index - 1

    def test_end_index_not_at_chunk_boundary(self):
        """Test end_index not aligned with chunk boundary."""
        chunk_size = 10
        end_index = 47  # Middle of chunk 4

        sampler = SliceSampler(
            start_index=0,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(0, end_index))
        assert all_indices == expected
        assert max(all_indices) == end_index - 1


class TestSliceSamplerBothIndices:
    """Tests for SliceSampler with both start_index and end_index."""

    def test_both_at_chunk_boundaries(self):
        """Test both start and end aligned with chunk boundaries."""
        chunk_size = 10
        start_index = 20
        end_index = 60

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected
        assert len(all_indices) == end_index - start_index

    def test_both_not_at_chunk_boundaries(self):
        """Test both start and end not aligned with chunk boundaries."""
        chunk_size = 10
        start_index = 23
        end_index = 67

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected
        assert min(all_indices) == start_index
        assert max(all_indices) == end_index - 1

    def test_single_chunk_span(self):
        """Test start and end within a single chunk."""
        chunk_size = 10
        start_index = 22
        end_index = 28  # Same chunk as start

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=2,
            chunk_size=chunk_size,
            preload_nchunks=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected
        assert len(all_indices) == 6

    def test_worker_shard_simulation(self):
        """Test simulating DataLoader worker sharding (600 obs, 4 workers)."""
        n_obs = 600
        chunk_size = 10
        num_workers = 4
        per_worker = n_obs // num_workers  # 150

        all_worker_indices = set()
        for worker_id in range(num_workers):
            start_index = worker_id * per_worker
            if worker_id == num_workers - 1:
                end_index = n_obs
            else:
                end_index = start_index + per_worker

            sampler = SliceSampler(
                start_index=start_index,
                end_index=end_index,
                batch_size=10,
                chunk_size=chunk_size,
                preload_nchunks=4,
            )

            worker_indices = set()
            for load_request in sampler:
                for s in load_request.slices:
                    worker_indices.update(range(s.start, s.stop))

            # Check this worker got the right range
            expected = set(range(start_index, end_index))
            assert worker_indices == expected

            # Add to global set
            all_worker_indices.update(worker_indices)

        # All workers together should cover the full dataset exactly once
        assert all_worker_indices == set(range(n_obs))


class TestSliceSamplerWithShuffle:
    """Tests for SliceSampler with shuffling enabled."""

    def test_shuffle_with_start_index(self):
        """Test shuffle works correctly with non-zero start_index."""
        n_obs = 100
        chunk_size = 10
        start_index = 30

        sampler = SliceSampler(
            start_index=start_index,
            end_index=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected

    def test_shuffle_with_both_indices(self):
        """Test shuffle works correctly with both start and end index."""
        chunk_size = 10
        start_index = 25
        end_index = 75

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected


class TestSliceSamplerEdgeCases:
    """Tests for edge cases."""

    def test_very_small_shard(self):
        """Test with a very small shard (smaller than batch_size)."""
        chunk_size = 10
        start_index = 95
        end_index = 100  # Only 5 observations

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=10,  # Larger than shard size
            chunk_size=chunk_size,
            preload_nchunks=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected

    def test_start_equals_chunk_size(self):
        """Test start_index exactly equals chunk_size."""
        n_obs = 100
        chunk_size = 10
        start_index = 10  # Exactly one chunk in

        sampler = SliceSampler(
            start_index=start_index,
            end_index=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start_index

    @pytest.mark.parametrize(
        "start_index,end_index",
        [
            (0, 100),  # Full range
            (15, 85),  # Both non-aligned
            (20, 80),  # Both aligned
            (0, 50),  # Only end set
            (50, 100),  # Only start set (effectively)
        ],
    )
    def test_parametrized_ranges(self, start_index, end_index):
        """Test various start/end combinations cover correct range."""
        chunk_size = 10

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected


class MockWorkerHandle:
    """Simulates torch worker context for testing without actual DataLoader."""

    def __init__(self, worker_id: int, num_workers: int, seed: int = 42):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._rng = np.random.default_rng(seed)  # Same seed = consistent shuffle across workers

    def shuffle(self, obj):
        self._rng.shuffle(obj)

    def get_part_for_worker(self, obj: np.ndarray) -> np.ndarray:
        chunks_split = np.array_split(obj, self.num_workers)
        return chunks_split[self.worker_id]


class TestSliceSamplerWithWorkers:
    """Tests for SliceSampler with simulated DataLoader workers."""

    def test_two_workers_divisible_config(self):
        """Test 2 workers with divisible config cover full dataset without overlap."""
        n_obs = 200
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 10  # 10 * 2 = 20, divisible by 10
        num_workers = 2

        all_worker_indices = []
        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            sampler = SliceSampler(
                start_index=0,
                end_index=n_obs,
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
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
        chunk_size = 10
        preload_nchunks = 3
        batch_size = 10  # 10 * 3 = 30, divisible by 10
        num_workers = 3

        all_worker_indices = []
        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            sampler = SliceSampler(
                start_index=0,
                end_index=n_obs,
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
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
                start_index=0,
                end_index=100,
                batch_size=7,  # Non-divisible
                chunk_size=10,
                preload_nchunks=2,
                drop_last=True,
            )
            sampler.set_worker_handle(worker_handle)

    def test_workers_non_divisible_without_drop_last_raises(self):
        """Test that non-divisible config without drop_last raises ValueError."""
        worker_handle = MockWorkerHandle(0, 2)

        with pytest.raises(ValueError, match="divisible by batch_size"):
            sampler = SliceSampler(
                start_index=0,
                end_index=100,
                batch_size=7,  # 10 * 2 = 20, not divisible by 7
                chunk_size=10,
                preload_nchunks=2,
                drop_last=False,
            )
            sampler.set_worker_handle(worker_handle)

    def test_two_workers_drop_last_drops_per_worker(self):
        """Test drop_last=True drops only the final partial batch (intermediate partials are for carry-over)."""
        n_obs = 200
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 7  # Non-divisible: 20 / 7 = 2 full batches + 6 leftover per iter
        num_workers = 2

        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            with pytest.warns(UserWarning):
                sampler = SliceSampler(
                    start_index=0,
                    end_index=n_obs,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
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
            start_index=0,
            end_index=100,
            batch_size=5,
            chunk_size=10,
            preload_nchunks=2,
        )
        # Should not raise
        sampler.validate(n_obs=100)

    def test_validate_passes_when_end_equals_n_obs(self):
        """Test validate passes when end_index equals n_obs."""
        sampler = SliceSampler(
            start_index=0,
            end_index=100,
            batch_size=5,
            chunk_size=10,
            preload_nchunks=2,
        )
        # Should not raise - end_index == n_obs is valid
        sampler.validate(n_obs=100)

    def test_validate_end_index_exceeds_n_obs(self):
        """Test validate raises when end_index > n_obs."""
        sampler = SliceSampler(
            start_index=0,
            end_index=200,
            batch_size=5,
            chunk_size=10,
            preload_nchunks=2,
        )
        with pytest.raises(ValueError, match="end_index.*exceeds loader n_obs"):
            sampler.validate(n_obs=100)

    def test_invalid_start_index(self):
        """Test that negative start_index raises."""
        with pytest.raises(ValueError, match="start_index must be >= 0"):
            SliceSampler(
                start_index=-1,
                end_index=100,
                batch_size=5,
                chunk_size=10,
                preload_nchunks=2,
            )

    def test_start_equals_end_raises(self):
        """Test that start_index == end_index raises."""
        with pytest.raises(ValueError, match="start_index must be >= 0 and < end_index"):
            SliceSampler(
                start_index=50,
                end_index=50,
                batch_size=5,
                chunk_size=10,
                preload_nchunks=2,
            )

    def test_start_greater_than_end_raises(self):
        """Test that start_index > end_index raises."""
        with pytest.raises(ValueError, match="start_index must be >= 0 and < end_index"):
            SliceSampler(
                start_index=100,
                end_index=50,
                batch_size=5,
                chunk_size=10,
                preload_nchunks=2,
            )
