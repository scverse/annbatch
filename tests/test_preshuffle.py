from __future__ import annotations

import glob
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import DatasetCollection, write_sharded
from annbatch.io import V1_ENCODING

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from pathlib import Path


def test_write_sharded_bad_chunk_size(tmp_path: Path):
    adata = ad.AnnData(np.random.randn(10, 20))
    z = zarr.open(tmp_path / "foo.zarr")
    with pytest.raises(ValueError, match=r"Choose a dense"):
        write_sharded(z, adata, dense_chunk_size=20)


@pytest.mark.parametrize(
    ["chunk_size", "expected_shard_size"],
    [pytest.param(3, 9, id="n_obs_not_divisible_by_chunk"), pytest.param(5, 10, id="n_obs_divisible_by_chunk")],
)
def test_write_sharded_shard_size_too_big(tmp_path: Path, chunk_size: int, expected_shard_size: int):
    adata = ad.AnnData(np.random.randn(10, 20))
    z = zarr.open(tmp_path / "foo.zarr")
    write_sharded(z, adata, dense_chunk_size=chunk_size, dense_shard_size=20)
    assert z["X"].shards == (expected_shard_size, 20)  # i.e., the closest multiple to `dense_chunk_size`


@pytest.mark.parametrize("elem_name", ["obsm", "layers", "raw", "obs"])
def test_store_creation_warnings_with_different_keys(elem_name: Literal["obsm", "layers", "raw"], tmp_path: Path):
    adata_1 = ad.AnnData(X=np.random.randn(10, 20))
    extra_args = {
        elem_name: {"arr" if elem_name != "raw" else "X": np.random.randn(10, 20) if elem_name != "obs" else ["a"] * 10}
    }
    adata_2 = ad.AnnData(X=np.random.randn(10, 20), **extra_args)
    path_1 = tmp_path / "just_x.h5ad"
    path_2 = tmp_path / "with_extra_key.h5ad"
    adata_1.write_h5ad(path_1)
    adata_2.write_h5ad(path_2)
    with pytest.warns(UserWarning, match=rf"Found {elem_name} keys.* not present in all anndatas"):
        DatasetCollection(tmp_path / "collection.zarr").add_adatas(
            [path_1, path_2],
            zarr_sparse_chunk_size=10,
            zarr_sparse_shard_size=20,
            zarr_dense_chunk_size=5,
            zarr_dense_shard_size=10,
            n_obs_per_dataset=10,
            shuffle_chunk_size=10,
        )


def test_store_creation_no_warnings_with_custom_load(tmp_path: Path):
    adata_1 = ad.AnnData(X=np.random.randn(10, 20))
    adata_2 = ad.AnnData(X=np.random.randn(10, 20), layers={"arr": np.random.randn(10, 20)})
    path_1 = tmp_path / "just_x.h5ad"
    path_2 = tmp_path / "with_extra_key.h5ad"
    adata_1.write_h5ad(path_1)
    adata_2.write_h5ad(path_2)
    collection = DatasetCollection(tmp_path / "collection.zarr").add_adatas(
        [path_1, path_2],
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=5,
        zarr_dense_shard_size=10,
        n_obs_per_dataset=10,
        shuffle_chunk_size=5,
        load_adata=lambda x: ad.AnnData(X=ad.io.read_elem(h5py.File(x)["X"])),
    )
    assert len(ad.read_zarr(next(iter(collection))).layers.keys()) == 0


def test_store_creation_path_added_to_obs(tmp_path: Path):
    adata_1 = ad.AnnData(X=np.random.randn(10, 20))
    adata_2 = adata_1.copy()
    path_1 = tmp_path / "adata_1.h5ad"
    path_2 = tmp_path / "adata_2.h5ad"
    adata_1.write_h5ad(path_1)
    adata_2.write_h5ad(path_2)
    paths = [path_1, path_2]
    output_dir = tmp_path / "path_src_collection.zarr"
    collection = DatasetCollection(output_dir).add_adatas(
        paths,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=5,
        zarr_dense_shard_size=10,
        n_obs_per_dataset=10,
        shuffle_chunk_size=5,
        shuffle=False,
    )
    adata_result = ad.concat([ad.io.read_elem(g) for g in collection], join="outer")
    pd.testing.assert_extension_array_equal(
        adata_result.obs["src_path"].array,
        pd.Categorical(([str(path_1)] * 10) + ([str(path_2)] * 10), categories=[str(p) for p in paths]),
    )


@pytest.mark.parametrize("elem_name", ["obsm", "layers", "raw", "obs"])
@pytest.mark.parametrize("load_adata", [ad.read_h5ad, ad.experimental.read_lazy])
def test_store_addition_different_keys(
    elem_name: Literal["obsm", "layers", "raw"],
    tmp_path: Path,
    load_adata: Callable[[PathLike[str] | str], ad.AnnData],
):
    adata_orig = ad.AnnData(X=np.random.randn(100, 20))
    orig_path = tmp_path / "orig.h5ad"
    adata_orig.write_h5ad(orig_path)
    output_path = tmp_path / "zarr_store_addition_different_keys.zarr"
    collection = DatasetCollection(output_path)
    collection.add_adatas(
        [orig_path],
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=50,
        shuffle_chunk_size=10,
    )
    extra_args = {
        elem_name: {"arr" if elem_name != "raw" else "X": np.random.randn(10, 20) if elem_name != "obs" else ["a"] * 10}
    }
    adata = ad.AnnData(X=np.random.randn(10, 20), **extra_args)
    additional_path = tmp_path / "with_extra_key.h5ad"
    adata.write_h5ad(additional_path)
    with pytest.warns(UserWarning, match=rf"Found {elem_name} keys.* not present in all anndatas"):
        collection.add_adatas(
            [additional_path],
            load_adata=load_adata,
            zarr_sparse_chunk_size=10,
            zarr_sparse_shard_size=20,
            zarr_dense_chunk_size=5,
            zarr_dense_shard_size=10,
            shuffle_chunk_size=2,
        )


def test_h5ad_and_zarr_simultaneously(tmp_path: Path):
    with pytest.raises(ValueError, match=r"Do not set `is_collection_h5ad` to True when also passing in a zarr Group."):
        DatasetCollection(zarr.open_group(tmp_path / "foo.zarr"), is_collection_h5ad=True)


@pytest.mark.parametrize("is_collection_h5ad", [True, False], ids=["h5ad", "zarr"])
def test_store_creation_default(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
    is_collection_h5ad: bool,
):
    h5_files = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    output_path = (
        adata_with_h5_path_different_var_space[1].parent
        / f"{'h5ad' if is_collection_h5ad else 'zarr'}_store_creation_test_default"
    )
    with pytest.warns(
        UserWarning,
        match=r"collections have the `.zarr` suffix"
        if (is_zarr := not is_collection_h5ad)
        else r"Loading h5ad is currently not supported",
    ):
        kwargs = {} if is_zarr else {"is_collection_h5ad": True}
        collection = DatasetCollection(output_path, **kwargs).add_adatas(
            [adata_with_h5_path_different_var_space[1] / f for f in h5_files if str(f).endswith(".h5ad")]
        )
    assert isinstance(
        ad.io.read_elem(next(iter(collection)) if is_zarr else h5py.File(next(output_path.iterdir()))).X, sp.csr_matrix
    )
    assert len(list(iter(collection) if is_zarr else output_path.iterdir())) == 1
    # Test directory structure to make sure nothing extraneous was written
    assert sorted(glob.glob(str(output_path / f"dataset_*{'.h5ad' if is_collection_h5ad else ''}"))) == sorted(
        str(p) for p in (output_path).iterdir() if ((p.is_dir() and is_zarr) or not is_zarr)
    )
    store = zarr.open(output_path)
    with nullcontext() if is_zarr else pytest.raises(ValueError, match=r"Cannot iterate through"):
        assert list(iter(collection)) == [store[k] for k in sorted(store.keys())]
        assert V1_ENCODING.items() <= store.attrs.items()


@pytest.mark.parametrize("shuffle", [pytest.param(True, id="shuffle"), pytest.param(False, id="no_shuffle")])
@pytest.mark.parametrize(
    "load_adata", [pytest.param(None, id="default_read"), pytest.param(ad.experimental.read_lazy, id="fully_lazy")]
)
def test_store_creation(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
    shuffle: bool,
    load_adata: Callable[[str], ad.AnnData],
):
    var_subset = [f"gene_{i}" for i in range(100)]
    h5_files = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    output_path = (
        adata_with_h5_path_different_var_space[1].parent
        / f"zarr_store_creation_test_{shuffle}_{'default_read' if load_adata is None else 'custom_read'}.zarr"
    )
    collection = DatasetCollection(output_path).add_adatas(
        [adata_with_h5_path_different_var_space[1] / f for f in h5_files if str(f).endswith(".h5ad")],
        var_subset=var_subset,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=5,
        zarr_dense_shard_size=10,
        n_obs_per_dataset=50,
        shuffle_chunk_size=10,
        shuffle=shuffle,
        **({"load_adata": load_adata} if load_adata is not None else {}),
    )
    assert not DatasetCollection(output_path).is_empty
    assert V1_ENCODING.items() <= zarr.open(output_path).attrs.items()

    adata_orig = adata_with_h5_path_different_var_space[0]
    # make sure all category dtypes match
    adatas_shuffled = [ad.io.read_elem(g) for g in collection]
    for adata in adatas_shuffled:
        assert adata.obs["label"].dtype == adata_orig.obs["label"].dtype
    # subset to var_subset
    adata_orig = adata_orig[:, adata_orig.var.index.isin(var_subset)]
    adata_orig.obs_names_make_unique()
    adata = ad.concat(
        adatas_shuffled,
        join="outer",
    )
    del adata.obs["src_path"]
    assert adata.X.shape[0] == adata_orig.X.shape[0]
    assert adata.X.shape[1] == adata_orig.X.shape[1]
    assert np.array_equal(
        sorted(adata.var.index),
        sorted(adata_orig.var.index),
    )
    assert "arr" in adata.obsm
    if shuffle:
        # If it's shuffled I'd expect more than 90% of elements to be out of order
        assert sum(adata_orig.obs_names != adata.obs_names) > (0.9 * adata.shape[0])
        assert adata_orig.obs_names.isin(adata.obs_names).all()
        adata = adata[adata_orig.obs_names].copy()
    else:
        assert (adata_orig.obs_names == adata.obs_names).all()
    np.testing.assert_array_equal(
        adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray(),
        adata_orig.X if isinstance(adata_orig.X, np.ndarray) else adata_orig.X.toarray(),
    )
    np.testing.assert_array_equal(
        adata.raw.X if isinstance(adata.raw.X, np.ndarray) else adata.raw.X.toarray(),
        adata_orig.raw.X if isinstance(adata_orig.raw.X, np.ndarray) else adata_orig.raw.X.toarray(),
    )
    np.testing.assert_array_equal(adata.obsm["arr"], adata_orig.obsm["arr"])

    # correct for concat misordering the categories
    adata.obs["label"] = adata.obs["label"].cat.reorder_categories(adata_orig.obs["label"].dtype.categories)

    pd.testing.assert_frame_equal(adata.obs, adata_orig.obs)
    z = zarr.open(output_path / "dataset_0")
    assert z["obsm"]["arr"].chunks[0] == 5, z["obsm"]["arr"]
    assert z["X"]["indices"].chunks[0] == 10


def _read_lazy_x_and_obs_only_from_raw(path) -> ad.AnnData:
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


@pytest.mark.parametrize(
    "adata_with_h5_path_different_var_space",
    [{"all_adatas_have_raw": False}],
    indirect=True,
)
def test_mismatched_raw_concat(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
):
    h5_files = sorted(adata_with_h5_path_different_var_space[1].iterdir())
    output_path = adata_with_h5_path_different_var_space[1].parent / "zarr_store_creation_test_heterogeneous.zarr"
    h5_paths = [adata_with_h5_path_different_var_space[1] / f for f in h5_files if str(f).endswith(".h5ad")]
    collection = DatasetCollection(output_path).add_adatas(
        h5_paths,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=30,
        shuffle_chunk_size=10,
        shuffle=False,  # don't shuffle -> want to check if the right attributes get taken
        load_adata=_read_lazy_x_and_obs_only_from_raw,
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
    adata = ad.concat([ad.io.read_elem(g) for g in collection])
    del adata.obs["src_path"]
    pd.testing.assert_frame_equal(adata_orig.var, adata.var)
    pd.testing.assert_frame_equal(adata_orig.obs, adata.obs)
    np.testing.assert_array_equal(adata_orig.X.toarray(), adata.X.toarray())


@pytest.mark.parametrize("load_adata", [ad.read_h5ad, ad.experimental.read_lazy])
def test_store_extension(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
    load_adata: Callable[[PathLike[str] | str], ad.AnnData],
):
    all_h5_paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")
    store_path = (
        adata_with_h5_path_different_var_space[1].parent / f"zarr_store_extension_test_{load_adata.__name__}.zarr"
    )
    original = all_h5_paths
    additional = all_h5_paths[4:]  # don't add everything to get a "different" var space
    # create new store
    collection = DatasetCollection(store_path)
    collection.add_adatas(
        original,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=60,
        shuffle_chunk_size=10,
        shuffle=True,
    )
    # add h5ads to existing store
    collection.add_adatas(
        additional,
        load_adata=load_adata,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=5,
        zarr_dense_shard_size=10,
        n_obs_per_dataset=50,
        shuffle_chunk_size=10,
    )
    adatas_on_disk = [ad.io.read_elem(g) for g in collection]
    adata = ad.concat(adatas_on_disk)
    adata_orig = adata_with_h5_path_different_var_space[0]
    expected_adata = ad.concat([adata_orig, adata_orig[adata_orig.obs["store_id"] >= 4]], join="outer")
    assert adata.X.shape[1] == expected_adata.X.shape[1]
    assert adata.X.shape[0] == expected_adata.X.shape[0]
    # check categoricals to make sure the dtypes match
    for a in [*adatas_on_disk, adata]:
        assert a.obs["label"].dtype == expected_adata.obs["label"].dtype
    assert "arr" in adata.obsm
    z = zarr.open(store_path / "dataset_0")
    assert z["obsm"]["arr"].chunks == (5, z["obsm"]["arr"].shape[1])
    assert z["X"]["indices"].chunks[0] == 10


def test_empty(tmp_path: Path):
    g = zarr.open(tmp_path / "empty.zarr")
    collection = DatasetCollection(g)
    assert collection.is_empty
    # Doesn't matter what errors as long as this function runs, but not to completion
    with pytest.raises(TypeError):
        collection.add_adatas()
    assert not (V1_ENCODING.items() <= g.attrs.items())
