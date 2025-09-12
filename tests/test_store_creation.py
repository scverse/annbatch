from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest

from arrayloaders import add_anndata_to_sharded_chunks_directory, create_anndata_chunks_directory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("densify", [True, False])
def test_store_creation(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
    shuffle: bool,
    densify: bool,
):
    var_subset = [f"gene_{i}" for i in range(100)]

    (adata_with_h5_path_different_var_space[1] / "zarr_store").mkdir(parents=True, exist_ok=True)
    create_anndata_chunks_directory(
        [
            adata_with_h5_path_different_var_space[1] / f
            for f in sorted(adata_with_h5_path_different_var_space[1].iterdir())
            if str(f).endswith(".h5ad")
        ],
        adata_with_h5_path_different_var_space[1] / "zarr_store",
        var_subset=var_subset,
        chunk_size=10,
        shard_size=20,
        n_obs_per_output_anndata=60,
        shuffle=shuffle,
        should_denseify=densify,
    )

    adata_orig = adata_with_h5_path_different_var_space[0]
    print("zarr", list((adata_with_h5_path_different_var_space[1] / "zarr_store").iterdir()))
    adata = ad.concat(
        [
            ad.read_zarr(zarr_path)
            for zarr_path in sorted((adata_with_h5_path_different_var_space[1] / "zarr_store").iterdir())
        ],
        join="outer",
    )
    assert adata.X.shape[0] == adata_orig.X.shape[0]
    assert adata.X.shape[1] == adata_orig.X.shape[1]
    assert np.array_equal(
        sorted(adata.var.index),
        sorted(adata_orig.var.index),
    )
    if not shuffle:
        np.testing.assert_array_equal(
            adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray(),
            adata_orig.X if isinstance(adata_orig.X, np.ndarray) else adata_orig.X.toarray(),
        )


@pytest.mark.parametrize("densify", [True, False])
@pytest.mark.parametrize("read_full_anndatas", [True, False])
def test_store_extension(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
    densify: bool,
    read_full_anndatas: bool,
    shuffle: bool,
):
    store_path = adata_with_h5_path_different_var_space[1] / "zarr_store"
    all_h5_paths = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    original = all_h5_paths
    additional = all_h5_paths[4:]  # don't add everything to get a "different" var space
    # create new store
    create_anndata_chunks_directory(
        [adata_with_h5_path_different_var_space[1] / f for f in original if str(f).endswith(".h5ad")],
        store_path,
        chunk_size=10,
        shard_size=20,
        n_obs_per_output_anndata=60,
        shuffle=True,
        should_denseify=densify,
    )
    # add h5ads to existing store
    add_anndata_to_sharded_chunks_directory(
        additional,
        store_path,
        read_full_anndatas=read_full_anndatas,
        chunk_size=10,
        shard_size=20,
    )

    adata = ad.concat(
        [ad.read_zarr(zarr_path) for zarr_path in (adata_with_h5_path_different_var_space[1] / "zarr_store").iterdir()]
    )
    adata_orig = adata_with_h5_path_different_var_space[0]
    expected_adata = ad.concat([adata_orig, adata_orig[adata_orig.obs["store_id"] >= 4]], join="outer")
    assert adata.X.shape[1] == expected_adata.X.shape[1]
    assert adata.X.shape[0] == expected_adata.X.shape[0]
