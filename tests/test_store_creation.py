from __future__ import annotations

import glob
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import add_to_collection, create_anndata_collection

if TYPE_CHECKING:
    from pathlib import Path


def _read_lazy_x_and_obs_only(path) -> ad.AnnData:
    adata_ = ad.experimental.read_lazy(path)
    if adata_.raw is not None:
        x = adata_.raw.X
        var = adata_.raw.var
    else:
        x = adata_.X
        var = adata_.var

    return ad.AnnData(
        X=x,
        obs=adata_.obs.to_memory(),
        var=var.to_memory(),
    )


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
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=60,
    )
    assert isinstance(ad.read_zarr(next((output_path).iterdir())).X, sp.csr_matrix)
    assert sorted(glob.glob(str(output_path / "dataset_*.zarr"))) == sorted(str(p) for p in (output_path).iterdir())


def test_store_creation_drop_elem(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
):
    var_subset = [f"gene_{i}" for i in range(100)]
    h5_files = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    output_path = adata_with_h5_path_different_var_space[1].parent / "zarr_store_creation_drop_elems"
    output_path.mkdir(parents=True, exist_ok=True)

    create_anndata_collection(
        [adata_with_h5_path_different_var_space[1] / f for f in h5_files if str(f).endswith(".h5ad")],
        output_path,
        var_subset=var_subset,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=60,
        load_function=_read_lazy_x_and_obs_only,
    )
    adata_output = ad.read_zarr(next(output_path.iterdir()))
    assert "arr" not in adata_output.obsm
    assert adata_output.raw is None


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
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=60,
        shuffle=shuffle,
        should_denseify=densify,
    )

    adata_orig = adata_with_h5_path_different_var_space[0]
    # subset to var_subset
    adata_orig = adata_orig[:, adata_orig.var.index.isin(var_subset)]
    adata_orig.obs_names_make_unique()
    adata = ad.concat(
        [ad.read_zarr(zarr_path) for zarr_path in sorted(output_path.iterdir())],
        join="outer",
    )
    assert adata.X.shape[0] == adata_orig.X.shape[0]
    assert adata.X.shape[1] == adata_orig.X.shape[1]
    assert np.array_equal(
        sorted(adata.var.index),
        sorted(adata_orig.var.index),
    )
    assert "arr" in adata.obsm
    if not shuffle:
        np.testing.assert_array_equal(
            adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray(),
            adata_orig.X if isinstance(adata_orig.X, np.ndarray) else adata_orig.X.toarray(),
        )
        np.testing.assert_array_equal(
            adata.raw.X if isinstance(adata.raw.X, np.ndarray) else adata.raw.X.toarray(),
            adata_orig.raw.X if isinstance(adata_orig.raw.X, np.ndarray) else adata_orig.raw.X.toarray(),
        )
        np.testing.assert_array_equal(adata.obsm["arr"], adata_orig.obsm["arr"])
        adata.obs.index = adata_orig.obs.index  # correct for concat
        pd.testing.assert_frame_equal(adata.obs, adata_orig.obs)
    z = zarr.open(output_path / "dataset_0.zarr")
    assert z["obsm"]["arr"].chunks[0] == 10, z["obsm"]["arr"]
    if not densify:
        assert z["X"]["indices"].chunks[0] == 10, z["X"]["indices"]
    else:
        assert z["X"].chunks[0] == 10, z["X"]["indices"]


def test_heterogeneous_structure_store_creation(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
):
    h5_files = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    output_path = adata_with_h5_path_different_var_space[1].parent / "zarr_store_creation_test_heterogeneous"
    output_path.mkdir(parents=True, exist_ok=True)
    h5_paths = [adata_with_h5_path_different_var_space[1] / f for f in h5_files if str(f).endswith(".h5ad")]
    create_anndata_collection(
        h5_paths,
        output_path,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=60,
        load_function=_read_lazy_x_and_obs_only,
        shuffle=False,  # don't shuffle -> want to check if the right attributes get taken
    )

    adatas_orig = []
    for file in h5_paths:
        dataset = ad.read_h5ad(file)
        adatas_orig.append(
            ad.AnnData(
                X=dataset.X if dataset.raw is None else dataset.raw.X,
                obs=dataset.obs,
                var=dataset.var if dataset.raw is None else dataset.raw.var,
            )
        )

    adata_orig = ad.concat(adatas_orig, join="outer")
    adata_orig.obs_names_make_unique()
    adata = ad.concat([ad.read_zarr(zarr_path) for zarr_path in sorted(output_path.iterdir())])

    pd.testing.assert_frame_equal(adata_orig.var, adata.var)
    pd.testing.assert_frame_equal(adata_orig.obs, adata.obs)
    np.testing.assert_array_equal(adata_orig.X.toarray(), adata.X.toarray())


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
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=60,
        shuffle=True,
        should_denseify=densify,
    )
    # add h5ads to existing store
    add_to_collection(
        additional,
        store_path,
        read_full_anndatas=read_full_anndatas,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=5,
        zarr_dense_shard_size=10,
    )

    adata = ad.concat([ad.read_zarr(zarr_path) for zarr_path in sorted(store_path.iterdir())])
    adata_orig = adata_with_h5_path_different_var_space[0]
    expected_adata = ad.concat([adata_orig, adata_orig[adata_orig.obs["store_id"] >= 4]], join="outer")
    assert adata.X.shape[1] == expected_adata.X.shape[1]
    assert adata.X.shape[0] == expected_adata.X.shape[0]
    assert "arr" in adata.obsm
    z = zarr.open(store_path / "dataset_0.zarr")
    assert z["obsm"]["arr"].chunks[0] == 5, z["obsm"]["arr"]
    if not densify:
        assert z["X"]["indices"].chunks[0] == 10, z["X"]["indices"]
    else:
        assert z["X"].chunks[0] == 5, z["X"]["indices"]
