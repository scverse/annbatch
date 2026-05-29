"""Tests for CategoricalSampler.

The passing tests check the sampler does what it promises: every chunk is
category-coherent, categories are drawn uniformly, and the bookkeeping
(``num_samples`` / ``n_iters`` / validation) is correct.

The final test (``test_pure_categorical_batches_unsupported``) is expected to
**fail**. It is deliberately not marked ``xfail``: it documents, with the real
:class:`~annbatch.Loader` ordering contract, why the sampler cannot currently
yield *category-pure batches*. See the PR thread.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from annbatch.samplers import CategoricalSampler
from annbatch.samplers._utils import WorkerInfo


def _chunk_categories(chunks: list[slice], codes: np.ndarray) -> list[int]:
    """Category of each chunk, or -1 if the chunk straddles more than one category."""
    out = []
    for c in chunks:
        u = np.unique(codes[c])
        out.append(int(u[0]) if u.size == 1 else -1)
    return out


def _collect_chunks(sampler: CategoricalSampler, n_obs: int) -> list[slice]:
    chunks: list[slice] = []
    for load_request in sampler.sample(n_obs):
        chunks.extend(load_request["chunks"])
    return chunks


# =============================================================================
# Construction / validation
# =============================================================================


def test_codes_must_be_1d():
    with pytest.raises(ValueError, match="1D array"):
        CategoricalSampler(
            chunk_size=10,
            preload_nchunks=2,
            batch_size=5,
            codes=np.zeros((10, 2), dtype=int),
            num_samples=50,
        )


def test_no_category_large_enough_raises():
    # every run is shorter than chunk_size
    codes = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="at least chunk_size"):
        CategoricalSampler(
            chunk_size=10,
            preload_nchunks=2,
            batch_size=5,
            codes=codes,
            num_samples=10,
        )


def test_validate_rejects_n_obs_mismatch():
    codes = np.repeat([0, 1], 50)
    sampler = CategoricalSampler(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=10,
        codes=codes,
        num_samples=50,
    )
    with pytest.raises(ValueError, match="does not match loader n_obs"):
        sampler.validate(n_obs=999)


def test_mask_property_not_supported():
    codes = np.repeat([0, 1], 50)
    sampler = CategoricalSampler(chunk_size=10, preload_nchunks=2, batch_size=10, codes=codes, num_samples=50)
    with pytest.raises(NotImplementedError, match="mask property"):
        _ = sampler.mask
    with pytest.raises(NotImplementedError, match="mask property"):
        sampler.mask = slice(0, 10)


def test_multiple_workers_not_supported():
    codes = np.repeat([0, 1], 50)
    sampler = CategoricalSampler(chunk_size=10, preload_nchunks=2, batch_size=10, codes=codes, num_samples=50)
    with (
        patch(
            "annbatch.samplers._categorical_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(NotImplementedError, match="Multiple workers"),
    ):
        list(sampler.sample(len(codes)))


def test_runs_shorter_than_chunk_size_are_dropped():
    # cat 0 has one good run (>= chunk_size) and one tiny run; cat 1 only a good run.
    codes = np.array([0] * 30 + [1] * 30 + [0] * 3, dtype=np.int64)
    sampler = CategoricalSampler(
        chunk_size=10, preload_nchunks=2, batch_size=10, codes=codes, num_samples=200, rng=np.random.default_rng(0)
    )
    # both categories are still sampleable; the 3-row tail of cat 0 is never a chunk start
    assert set(sampler.categories.tolist()) == {0, 1}
    chunks = _collect_chunks(sampler, len(codes))
    assert all(c.stop <= 60 for c in chunks), "no chunk should reach into the dropped 3-row run"


# =============================================================================
# Core behavior
# =============================================================================


@pytest.mark.parametrize(
    "codes",
    [
        pytest.param(np.repeat([0, 1, 2, 3], 100), id="contiguous"),
        # each category fragmented into two runs interleaved with the others
        pytest.param(np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50), id="fragmented"),
        pytest.param(np.array([2] * 40 + [0] * 40 + [1] * 40 + [0] * 40 + [2] * 40), id="fragmented_3cat"),
    ],
)
def test_chunks_are_category_coherent(codes: np.ndarray):
    sampler = CategoricalSampler(
        chunk_size=10, preload_nchunks=4, batch_size=10, codes=codes, num_samples=1000, rng=np.random.default_rng(0)
    )
    chunks = _collect_chunks(sampler, len(codes))
    cats = _chunk_categories(chunks, codes)
    assert -1 not in cats, "every chunk must lie entirely within a single category"
    # chunks stay in-bounds and are full size (num_samples is a multiple of chunk_size here)
    assert all(0 <= c.start and c.stop <= len(codes) and c.stop - c.start == 10 for c in chunks)


def test_fragmented_category_samples_all_runs():
    # cat 0 lives in two separate runs; over many draws both should be hit.
    codes = np.array([0] * 50 + [1] * 50 + [0] * 50, dtype=np.int64)
    sampler = CategoricalSampler(
        chunk_size=10, preload_nchunks=4, batch_size=10, codes=codes, num_samples=5000, rng=np.random.default_rng(0)
    )
    chunks = _collect_chunks(sampler, len(codes))
    starts_cat0 = [c.start for c in chunks if codes[c.start] == 0]
    hit_first_run = any(s < 50 for s in starts_cat0)
    hit_second_run = any(s >= 100 for s in starts_cat0)
    assert hit_first_run and hit_second_run, "both fragments of category 0 should be sampled"


def test_categories_drawn_uniformly():
    # 4 categories of different sizes -- uniform balancing means equal *chunk counts*,
    # independent of category size.
    codes = np.array([0] * 200 + [1] * 100 + [2] * 400 + [3] * 100, dtype=np.int64)
    sampler = CategoricalSampler(
        chunk_size=10, preload_nchunks=4, batch_size=10, codes=codes, num_samples=40_000, rng=np.random.default_rng(0)
    )
    chunks = _collect_chunks(sampler, len(codes))
    cats = np.array(_chunk_categories(chunks, codes))
    _, counts = np.unique(cats, return_counts=True)
    shares = counts / counts.sum()
    assert np.allclose(shares, 0.25, atol=0.02), f"expected ~uniform 0.25 per category, got {shares}"


@pytest.mark.parametrize(
    ("num_samples", "batch_size", "drop_last", "expected_iters"),
    [
        pytest.param(100, 10, False, 10, id="exact"),
        pytest.param(105, 10, False, 11, id="partial_kept"),
        pytest.param(105, 10, True, 10, id="partial_dropped"),
    ],
)
def test_n_iters(num_samples: int, batch_size: int, drop_last: bool, expected_iters: int):
    codes = np.repeat([0, 1], 100)
    sampler = CategoricalSampler(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=batch_size,
        codes=codes,
        num_samples=num_samples,
        drop_last=drop_last,
        rng=np.random.default_rng(0),
    )
    assert sampler.n_iters(len(codes)) == expected_iters


def test_num_samples_respected():
    codes = np.repeat([0, 1, 2], 100)
    sampler = CategoricalSampler(
        chunk_size=10, preload_nchunks=3, batch_size=10, codes=codes, num_samples=300, rng=np.random.default_rng(0)
    )
    total = sum(len(s) for lr in sampler.sample(len(codes)) for s in lr["splits"])
    assert total == 300


# =============================================================================
# The inevitable failure: category-pure *batches* are not achievable today.
# =============================================================================


def _loader_in_memory_global_index(chunks: list[slice], dataset_shapes: list[int]) -> np.ndarray:
    """Replay ``Loader._slices_to_dataset_rows`` ordering (see src/annbatch/loader.py:481).

    The loader groups requested chunks by dataset index and concatenates the
    in-memory buffer **in dataset order**, not in the chunk order the sampler
    emitted. ``splits`` then index into this reordered buffer (loader.py:797).
    This returns, for each buffer position, the global obs index living there.
    """
    global_index = np.concatenate([np.arange(s.start, s.stop) for s in chunks])
    ordered: list[np.ndarray] = []
    b_start = 0
    for shape in dataset_shapes:
        b_end = b_start + shape
        mask = (global_index >= b_start) & (global_index < b_end)
        if mask.any():
            ordered.append(global_index[mask])
        b_start = b_end
    return np.concatenate(ordered)


def test_pure_categorical_batches_unsupported():
    """EXPECTED TO FAIL (not xfail) -- documents the pure-batch gap.

    Each category is fragmented across two datasets, and every individual chunk
    is category-coherent. Yet the batches the loader yields are not category-pure:
    the loader reorders the in-memory buffer by dataset index (which the sampler
    is blind to) and the sampler shuffles across categories within a preload
    window. To fix this the Sampler API needs dataset-boundary information and
    per-category splits.
    """
    # ds0 = rows [0, 40), ds1 = rows [40, 80); each category straddles both datasets
    codes = np.array([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20, dtype=np.int64)
    dataset_shapes = [40, 40]

    sampler = CategoricalSampler(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=10,
        codes=codes,
        num_samples=80,
        rng=np.random.default_rng(0),
    )

    impure = 0
    for load_request in sampler.sample(len(codes)):
        buffer_global = _loader_in_memory_global_index(load_request["chunks"], dataset_shapes)
        for split in load_request["splits"]:
            batch_cats = codes[buffer_global[split]]
            if np.unique(batch_cats).size != 1:
                impure += 1

    assert impure == 0, (
        f"{impure} batch(es) mixed categories despite every chunk being category-coherent. "
        "The loader concatenates the in-memory buffer by dataset index (loader.py:481) and the "
        "sampler shuffles across categories within a preload window, so splits cannot carve out "
        "category-pure batches without dataset-boundary awareness in the Sampler API."
    )
