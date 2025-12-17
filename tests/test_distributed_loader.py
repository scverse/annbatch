from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from annbatch import Loader


def _dense_datasets_from_store(store_path: Path) -> list[zarr.Array]:
    return [zarr.open(p)["X"] for p in sorted(store_path.glob("*.zarr"))]


@pytest.mark.parametrize("world_size", [2, 3])
def test_distributed_no_overlap_drop_last_indices(
    monkeypatch,
    adata_with_zarr_path_same_var_space: tuple[object, Path],
    world_size: int,
):
    # chunk_size chosen so that we drop some tail data at chunk granularity
    chunk_size = 7
    preload_nchunks = 1
    batch_size = 1

    store_path = adata_with_zarr_path_same_var_space[1]
    datasets = _dense_datasets_from_store(store_path)

    per_rank_indices: list[np.ndarray] = []
    for rank in range(world_size):
        monkeypatch.setattr("annbatch.loader.get_rank_and_world_size", lambda r=rank: (r, world_size))

        ds = Loader(
            shuffle=False,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            return_index=True,
            preload_to_gpu=False,
            to_torch=False,
            distributed=True,
            drop_last_indices=True,
            pad_indices=False,
        )
        ds.add_datasets(datasets)

        idxs = np.concatenate([idx for _, _, idx in ds]).ravel()
        per_rank_indices.append(idxs)

    # No overlap between ranks
    for i in range(world_size):
        for j in range(i + 1, world_size):
            assert set(per_rank_indices[i]).isdisjoint(set(per_rank_indices[j]))

    # Dropped tail at chunk granularity -> total yielded is a multiple of (chunk_size * world_size)
    total = sum(len(v) for v in per_rank_indices)
    assert total % (chunk_size * world_size) == 0


def test_distributed_padding_repeats_when_enabled(monkeypatch, adata_with_zarr_path_same_var_space: tuple[object, Path]):
    world_size = 2
    chunk_size = 7
    preload_nchunks = 1
    batch_size = 1

    store_path = adata_with_zarr_path_same_var_space[1]
    datasets = _dense_datasets_from_store(store_path)

    all_indices = []
    for rank in range(world_size):
        monkeypatch.setattr("annbatch.loader.get_rank_and_world_size", lambda r=rank: (r, world_size))

        ds = Loader(
            shuffle=False,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            return_index=True,
            preload_to_gpu=False,
            to_torch=False,
            distributed=True,
            drop_last_indices=False,
            pad_indices=True,
        )
        ds.add_datasets(datasets)
        all_indices.append(np.concatenate([idx for _, _, idx in ds]).ravel())

    concatenated = np.concatenate(all_indices)
    assert len(concatenated) % (chunk_size * world_size) == 0
    # Padding implies repeats (since underlying dataset is finite)
    assert len(np.unique(concatenated)) <= len(concatenated)
    assert len(np.unique(concatenated)) == sum(d.shape[0] for d in datasets)


def test_distributed_deterministic_for_fixed_rank(monkeypatch, adata_with_zarr_path_same_var_space: tuple[object, Path]):
    world_size = 2
    rank = 0
    monkeypatch.setattr("annbatch.loader.get_rank_and_world_size", lambda: (rank, world_size))

    store_path = adata_with_zarr_path_same_var_space[1]
    datasets = _dense_datasets_from_store(store_path)

    ds = Loader(
        shuffle=True,
        shuffle_seed=123,
        chunk_size=7,
        preload_nchunks=2,
        batch_size=3,
        return_index=True,
        preload_to_gpu=False,
        to_torch=False,
        distributed=True,
        drop_last_indices=True,
        pad_indices=False,
    )
    ds.add_datasets(datasets)
    ds.set_epoch(0)

    idxs1 = np.concatenate([idx for _, _, idx in ds]).ravel()
    idxs2 = np.concatenate([idx for _, _, idx in ds]).ravel()
    assert np.array_equal(idxs1, idxs2)


