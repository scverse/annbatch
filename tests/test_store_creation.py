from __future__ import annotations

import glob
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import add_to_collection, create_anndata_collection, write_sharded

if TYPE_CHECKING:
    from pathlib import Path


def test_write_sharded_bad_chunk_size(tmp_path):
    adata = ad.AnnData(np.random.randn(10, 20))
    z = zarr.open(tmp_path / "foo.zarr")
    with pytest.raises(ValueError, match=r"Choose a dense"):
        write_sharded(z, adata, dense_chunk_obs=20)


def test_write_sharded_shard_size_too_big(tmp_path):
    adata = ad.AnnData(np.random.randn(10, 20))
    z = zarr.open(tmp_path / "foo.zarr")
    write_sharded(z, adata, dense_chunk_obs=5, dense_shard_obs=20)


@pytest.mark.parametrize("elem_name", ["obsm", "layers", "raw"])
def test_store_creation_with_different_keys(elem_name, tmp_path):
    adata_1 = ad.AnnData(X=np.random.randn(10, 20))
    extra_args = (
        {elem_name: {"arr": np.random.randn(10, 20)}} if elem_name != "raw" else {"raw": {"X": np.random.randn(10, 20)}}
    )
    adata_2 = ad.AnnData(X=np.random.randn(10, 20), **extra_args)
    path_1 = tmp_path / "just_x.h5ad"
    path_2 = tmp_path / "with_extra_key.h5ad"
    adata_1.write_h5ad(path_1)
    adata_2.write_h5ad(path_2)
    with pytest.warns(UserWarning, match=rf"Some anndatas have {elem_name}"):
        create_anndata_collection(
            [path_1, path_2],
            tmp_path / "collection",
            zarr_sparse_chunk_size=10,
            zarr_sparse_shard_size=20,
            zarr_dense_chunk_obs=5,
            zarr_dense_shard_obs=10,
            n_obs_per_dataset=10,
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
        zarr_dense_chunk_obs=10,
        zarr_dense_shard_obs=20,
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

    def transform(a: ad.AnnData) -> ad.AnnData:
        del a.obsm
        del a.raw
        return a

    create_anndata_collection(
        [adata_with_h5_path_different_var_space[1] / f for f in h5_files if str(f).endswith(".h5ad")],
        output_path,
        var_subset=var_subset,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_obs=10,
        zarr_dense_shard_obs=20,
        n_obs_per_dataset=60,
        transform_input_adata=transform,
    )
    adata_output = ad.read_zarr(next((output_path).iterdir()))
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
        zarr_dense_chunk_obs=5,
        zarr_dense_shard_obs=10,
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
    assert z["obsm"]["arr"].chunks[0] == 5, z["obsm"]["arr"]
    if not densify:
        assert z["X"]["indices"].chunks[0] == 10, z["X"]["indices"]
    else:
        assert z["X"].chunks[0] == 5, z["X"]["indices"]


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
        zarr_dense_chunk_obs=10,
        zarr_dense_shard_obs=20,
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
        zarr_dense_chunk_obs=5,
        zarr_dense_shard_obs=10,
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
