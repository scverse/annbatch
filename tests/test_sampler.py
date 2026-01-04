"""Tests for SliceSampler."""

from __future__ import annotations

import numpy as np
import pytest

from annbatch.sampler import SliceSampler


class TestSliceSamplerBasic:
    """Tests for basic SliceSampler functionality."""

    def test_full_dataset(self):
        """Test sampler covers full dataset."""
        n_obs = 100
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 5

        sampler = SliceSampler(
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )

        all_indices = set()
        for load_request in sampler.sample(n_obs):
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
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )

        for i, load_request in enumerate(sampler.sample(n_obs)):
            actual_sizes = [len(split) for split in load_request.splits]
            assert actual_sizes == expected_sizes_per_iter[i], (
                f"Iter {i}: expected {expected_sizes_per_iter[i]}, got {actual_sizes}"
            )


class TestSliceSamplerWithShuffle:
    """Tests for SliceSampler with shuffling enabled."""

    def test_shuffle_covers_all_obs(self):
        """Test shuffle works correctly and covers all observations."""
        n_obs = 100
        chunk_size = 10

        sampler = SliceSampler(
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for load_request in sampler.sample(n_obs):
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(n_obs))
        assert all_indices == expected


class TestSliceSamplerEdgeCases:
    """Tests for edge cases."""

    def test_small_dataset(self):
        """Test with a very small dataset (smaller than batch_size)."""
        n_obs = 5
        chunk_size = 10
        batch_size = 10  # Larger than n_obs

        sampler = SliceSampler(
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=1,
        )

        all_indices = set()
        for load_request in sampler.sample(n_obs):
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(n_obs))
        assert all_indices == expected

    @pytest.mark.parametrize(
        "n_obs",
        [100, 150, 200, 50],
    )
    def test_parametrized_n_obs(self, n_obs):
        """Test various n_obs values cover correct range."""
        chunk_size = 10

        sampler = SliceSampler(
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler.sample(n_obs):
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(n_obs))
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
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
            )
            sampler.set_worker_handle(worker_handle)

            worker_indices = set()
            for load_request in sampler.sample(n_obs):
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
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
            )
            sampler.set_worker_handle(worker_handle)

            worker_indices = set()
            for load_request in sampler.sample(n_obs):
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
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                    drop_last=True,
                )
                sampler.set_worker_handle(worker_handle)

            all_requests = list(sampler.sample(n_obs))
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

    def test_validate_positive_n_obs(self):
        """Test validate accepts positive n_obs."""
        sampler = SliceSampler(
            batch_size=5,
            chunk_size=10,
            preload_nchunks=2,
        )
        # Should not raise
        sampler.validate(100)

    def test_validate_zero_n_obs(self):
        """Test validate rejects zero n_obs."""
        sampler = SliceSampler(
            batch_size=5,
            chunk_size=10,
            preload_nchunks=2,
        )
        with pytest.raises(ValueError, match="n_obs must be at least 1"):
            sampler.validate(0)

    def test_validate_negative_n_obs(self):
        """Test validate rejects negative n_obs."""
        sampler = SliceSampler(
            batch_size=5,
            chunk_size=10,
            preload_nchunks=2,
        )
        with pytest.raises(ValueError, match="n_obs must be at least 1"):
            sampler.validate(-1)
