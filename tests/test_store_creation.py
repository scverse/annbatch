from __future__ import annotations

import glob
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from arrayloaders import add_to_collection, create_anndata_collection

if TYPE_CHECKING:
    from pathlib import Path


def test_store_creation_default(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
):
    var_subset = [f"gene_{i}" for i in range(100)]
    h5_files = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    output_path = adata_with_h5_path_different_var_space[1].parent / "zarr_store_creation_test_default"
    output_path.mkdir(parents=True, exist_ok=True)
    create_anndata_collection(
        [adata_with_h5_path_different_var_space[1] / f for f in h5_files if str(f).endswith(".h5ad")],
        output_path,
        var_subset=var_subset,
        zarr_chunk_size=10,
        zarr_shard_size=20,
        n_obs_per_dataset=60,
    )
    assert isinstance(ad.read_zarr(next((output_path).iterdir())).X, sp.csr_matrix)
    assert sorted(glob.glob(str(output_path / "dataset_*.zarr"))) == sorted(str(p) for p in (output_path).iterdir())


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("densify", [True, False])
def test_store_creation(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
    shuffle: bool,
    densify: bool,
):
    var_subset = [f"gene_{i}" for i in range(100)]
    h5_files = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    output_path = adata_with_h5_path_different_var_space[1].parent / f"zarr_store_creation_test_{shuffle}_{densify}"
    output_path.mkdir(parents=True, exist_ok=True)
    create_anndata_collection(
        [adata_with_h5_path_different_var_space[1] / f for f in h5_files if str(f).endswith(".h5ad")],
        output_path,
        var_subset=var_subset,
        zarr_chunk_size=10,
        zarr_shard_size=20,
        n_obs_per_dataset=60,
        shuffle=shuffle,
        should_denseify=densify,
    )

    adata_orig = adata_with_h5_path_different_var_space[0]
    adata = ad.concat(
        [ad.read_zarr(zarr_path) for zarr_path in sorted((output_path).iterdir())],
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
):
    all_h5_paths = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    store_path = (
        adata_with_h5_path_different_var_space[1].parent / f"zarr_store_extension_test_{densify}_{read_full_anndatas}"
    )
    original = all_h5_paths
    additional = all_h5_paths[4:]  # don't add everything to get a "different" var space
    # create new store
    create_anndata_collection(
        original,
        store_path,
        zarr_chunk_size=10,
        zarr_shard_size=20,
        n_obs_per_dataset=60,
        shuffle=True,
        should_denseify=densify,
    )
    # add h5ads to existing store
    add_to_collection(
        additional,
        store_path,
        read_full_anndatas=read_full_anndatas,
        zarr_chunk_size=10,
        zarr_shard_size=20,
    )

    adata = ad.concat([ad.read_zarr(zarr_path) for zarr_path in sorted(store_path.iterdir())])
    adata_orig = adata_with_h5_path_different_var_space[0]
    expected_adata = ad.concat([adata_orig, adata_orig[adata_orig.obs["store_id"] >= 4]], join="outer")
    assert adata.X.shape[1] == expected_adata.X.shape[1]
    assert adata.X.shape[0] == expected_adata.X.shape[0]
