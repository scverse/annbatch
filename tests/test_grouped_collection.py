"""Tests for GroupedCollection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
import zarr

from annbatch import CategoricalSampler, GroupedCollection, Loader
from annbatch.io import V1_GROUPED_ENCODING

if TYPE_CHECKING:
    from pathlib import Path


def test_grouped_collection_basic(adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby="label",
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        shuffle=False,
    )
    assert not collection.is_empty
    store = zarr.open(output)
    assert V1_GROUPED_ENCODING.items() <= store.attrs.items()
    assert list(store.attrs["groupby_keys"]) == ["label"]


def test_grouped_collection_compound_groupby(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_compound.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        shuffle=True,
        random_seed=0,
    )
    assert not collection.is_empty
    store = zarr.open(output)
    assert list(store.attrs["groupby_keys"]) == ["label", "store_id"]


def test_group_index_metadata(adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_meta.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        random_seed=0,
    )
    group_index = collection.group_index
    assert {"label", "store_id", "start", "stop", "count"}.issubset(group_index.columns)
    n_obs_total = sum(ad.read_h5ad(p).n_obs for p in paths)
    assert int(group_index["count"].sum()) == n_obs_total
    assert int(group_index["start"].iloc[0]) == 0
    assert np.all(group_index["start"].to_numpy() < group_index["stop"].to_numpy())
    assert np.array_equal(
        group_index["count"].to_numpy(),
        group_index["stop"].to_numpy() - group_index["start"].to_numpy(),
    )


def test_grouped_data_is_contiguous_by_group(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_contiguous.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        shuffle=False,
    )
    adata_grouped = ad.concat([ad.io.read_elem(g) for g in collection], join="outer")
    grouped_obs = adata_grouped.obs[["label", "store_id"]].astype("string").fillna("<NA>")

    group_index = collection.group_index
    for row in group_index.itertuples():
        sliced = grouped_obs.iloc[int(row.start) : int(row.stop)]
        assert (sliced["label"] == str(row.label)).all()
        assert (sliced["store_id"] == str(row.store_id)).all()


def test_grouped_collection_non_empty_raises(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_nonempty.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths, groupby="label", n_obs_per_chunk=2, zarr_shard_size=10, n_obs_per_dataset=200
    )
    with pytest.raises(RuntimeError, match="non-empty"):
        collection.add_adatas(paths, groupby="label", n_obs_per_chunk=2, zarr_shard_size=10, n_obs_per_dataset=200)


def test_grouped_collection_missing_groupby_raises(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_bad_key.zarr"
    with pytest.raises(ValueError, match="Could not find groupby"):
        GroupedCollection(output).add_adatas(
            paths, groupby="nonexistent_col", n_obs_per_chunk=2, zarr_shard_size=10, n_obs_per_dataset=200
        )


# =============================================================================
# Loader integration
# =============================================================================


def test_loader_with_grouped_collection(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    grouped = GroupedCollection(tmp_path / "grouped_loader.zarr").add_adatas(
        paths,
        groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        random_seed=0,
    )
    loader = Loader(
        chunk_size=10,
        preload_nchunks=4,
        batch_size=20,
        shuffle=False,
        return_index=True,
        preload_to_gpu=False,
        to_torch=False,
    ).use_collection(grouped)

    all_indices = []
    for batch in loader:
        assert batch["X"].shape[0] > 0
        all_indices.append(batch["index"])
    stacked = np.concatenate(all_indices)
    assert len(stacked) == loader.n_obs
    assert np.array_equal(np.sort(stacked), np.arange(loader.n_obs))


# =============================================================================
# CategoricalSampler.from_collection
# =============================================================================


def test_categorical_sampler_from_collection(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    grouped = GroupedCollection(tmp_path / "grouped_catsampler.zarr").add_adatas(
        paths,
        groupby="label",
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        shuffle=False,
    )
    group_index = grouped.group_index
    min_cat_size = int(group_index["count"].min())
    chunk_size = min(min_cat_size, 2)
    batch_size = chunk_size
    sampler = CategoricalSampler.from_collection(
        grouped,
        chunk_size=chunk_size,
        preload_nchunks=2,
        batch_size=batch_size,
        num_samples=20 * batch_size,
        rng=np.random.default_rng(0),
    )
    assert sampler.n_categories == len(group_index)
    assert sampler.n_iters(0) == 20

    total_batches = 0
    for lr in sampler.sample(int(group_index["stop"].iloc[-1])):
        total_batches += len(lr["splits"])
    assert total_batches == 20
