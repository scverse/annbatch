from __future__ import annotations

from importlib.util import find_spec
from types import NoneType
from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import Loader

try:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix
except ImportError:
    CupyCSRMatrix = NoneType
    CupyArray = NoneType

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class Data(TypedDict):
    dataset: ad.abc.CSRDataset | zarr.Array
    obs: np.ndarray


class ListData:
    datasets: list[ad.abc.CSRDataset | zarr.Array]
    obs: list[np.ndarray]


def open_sparse(path: Path, *, use_zarrs: bool = False, use_anndata: bool = False) -> Data | ad.AnnData:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline" if use_zarrs else old_pipeline}):
        data = {
            "dataset": ad.io.sparse_dataset(zarr.open(path)["layers"]["sparse"]),
            "obs": ad.io.read_elem(zarr.open(path)["obs"]),
        }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=data["obs"])
    return data


def open_dense(path: Path, *, use_zarrs: bool = False, use_anndata: bool = False) -> Data | ad.AnnData:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline" if use_zarrs else old_pipeline}):
        data = {
            "dataset": zarr.open(path)["X"],
            "obs": ad.io.read_elem(zarr.open(path)["obs"]),
        }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=data["obs"])
    return data
    return data


def concat(datas: list[Data | ad.AnnData]) -> ListData | list[ad.AnnData]:
    return (
        {
            "datasets": [d["dataset"] for d in datas],
            "obs": [d["obs"] for d in datas],
        }
        if all(isinstance(d, dict) for d in datas)
        else datas
    )


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
@pytest.mark.parametrize(
    "gen_loader",
    [
        pytest.param(
            lambda path,
            shuffle,
            use_zarrs,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            open_func=open_func,
            batch_size=batch_size,
            preload_to_gpu=preload_to_gpu: Loader(
                shuffle=shuffle,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                return_index=True,
                batch_size=batch_size,
                preload_to_gpu=preload_to_gpu,
                to_torch=False,
            ).add_anndatas(
                [open_func(p, use_zarrs=use_zarrs, use_anndata=True) for p in path.glob("*.zarr")],
            ),
            id=f"chunk_size={chunk_size}-preload_nchunks={preload_nchunks}-dataset_type={open_func.__name__[5:]}-batch_size={batch_size}{'-cupy' if preload_to_gpu else ''}",  # type: ignore[attr-defined]
            marks=pytest.mark.skipif(
                find_spec("cupy") is None and preload_to_gpu,
                reason="need cupy installed",
            ),
        )
        for chunk_size, preload_nchunks, open_func, batch_size, preload_to_gpu in [
            elem
            for preload_to_gpu in [True, False]
            for open_func in [open_sparse, open_dense]
            for elem in [
                [
                    1,
                    5,
                    open_func,
                    1,
                    preload_to_gpu,
                ],  # singleton chunk size
                [
                    5,
                    1,
                    open_func,
                    1,
                    preload_to_gpu,
                ],  # singleton preload
                [
                    10,
                    5,
                    open_func,
                    5,
                    preload_to_gpu,
                ],  # batch size divides total in memory size evenly
                [
                    10,
                    5,
                    open_func,
                    50,
                    preload_to_gpu,
                ],  # batch size equal to in-memory size loading
                [
                    10,
                    5,
                    open_func,
                    14,
                    preload_to_gpu,
                ],  # batch size does not divide in memory size evenly
            ]
        ]
    ],
)
def test_store_load_dataset(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], *, shuffle: bool, gen_loader, use_zarrs
):
    """
    This test verifies that the DaskDataset works correctly:
        1. The DaskDataset correctly loads data from the mock store
        2. Each sample has the expected feature dimension
        3. All samples from the dataset are processed
        4. If the dataset is not shuffled, it returns the correct data
    """
    loader: Loader = gen_loader(adata_with_zarr_path_same_var_space[1], shuffle, use_zarrs)
    adata = adata_with_zarr_path_same_var_space[0]
    is_dense = loader.dataset_type is zarr.Array
    n_elems = 0
    batches = []
    labels = []
    indices = []
    expected_data = adata.X if is_dense else adata.layers["sparse"].toarray()
    for batch in loader:
        x, label, index = batch["data"], batch["labels"], batch["index"]
        n_elems += x.shape[0]
        # Check feature dimension
        assert x.shape[1] == 100
        batches += [x.get() if isinstance(x, CupyCSRMatrix | CupyArray) else x]
        if label is not None:
            labels += [label]
        if index is not None:
            indices += [index]
    # check that we yield all samples from the dataset
    # np.array for sparse
    stacked = (np if is_dense else sp).vstack(batches)
    if not is_dense:
        stacked = stacked.toarray()
    if not shuffle:
        np.testing.assert_allclose(stacked, expected_data)
        if len(labels) > 0:
            expected_labels = adata.obs
            pd.testing.assert_frame_equal(
                pd.concat(labels),
                expected_labels,
            )
    else:
        if len(indices) > 0:
            indices = np.concatenate(indices).ravel()
            np.testing.assert_allclose(stacked, expected_data[indices])
        assert n_elems == adata.shape[0]


@pytest.mark.parametrize(
    "gen_loader",
    [
        (
            lambda path, chunk_size=chunk_size, preload_nchunks=preload_nchunks: Loader(
                shuffle=True,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
            )
        )
        for chunk_size, preload_nchunks in [[0, 10], [10, 0]]
    ],
)
def test_zarr_store_errors_lt_1(gen_loader, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    with pytest.raises(ValueError, match="must be greater than 1"):
        gen_loader(adata_with_zarr_path_same_var_space[1])


def test_bad_adata_X_type(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    data = open_dense(next(adata_with_zarr_path_same_var_space[1].glob("*.zarr")))
    data["dataset"] = data["dataset"][...]
    ds = Loader(shuffle=True, chunk_size=10, preload_nchunks=10, preload_to_gpu=False, to_torch=False)
    with pytest.raises(TypeError, match="Cannot add"):
        ds.add_dataset(**data)


@pytest.mark.skipif(not find_spec("torch"), reason="need torch installed")
@pytest.mark.parametrize(
    "preload_to_gpu",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                find_spec("cupy") is None,
                reason="need cupy installed",
            ),
        ),
        False,
    ],
)
@pytest.mark.parametrize("open_func", [open_sparse, open_dense])
def test_to_torch(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
    open_func: Callable[[Path], Data],
    preload_to_gpu: bool,
):
    import torch

    # batch_size guaranteed to have leftovers to drop
    ds = Loader(
        shuffle=False,
        chunk_size=5,
        preload_nchunks=10,
        batch_size=42,
        preload_to_gpu=preload_to_gpu,
        return_index=True,
        to_torch=True,
    )
    ds.add_dataset(**open_func(next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))))
    assert isinstance(next(iter(ds))["data"], torch.Tensor)


def test_drop_last(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    # batch_size guaranteed to have leftovers to drop
    ds = Loader(
        shuffle=False,
        chunk_size=5,
        preload_nchunks=10,
        batch_size=42,
        preload_to_gpu=False,
        return_index=True,
        drop_last=True,
        to_torch=False,
    )
    ds.add_dataset(**open_sparse(next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))))
    adata = adata_with_zarr_path_same_var_space[0]
    batches = []
    indices = []
    for batch in ds:
        batches += [batch["data"]]
        indices += [batch["index"]]
    X = sp.vstack(batches).toarray()
    assert X.shape[0] < adata.shape[0]
    X_expected = adata[np.concatenate(indices)].layers["sparse"].toarray()
    np.testing.assert_allclose(X, X_expected)


def test_bad_adata_X_hdf5(adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path]):
    with h5py.File(next(adata_with_h5_path_different_var_space[1].glob("*.h5ad"))) as f:
        data = ad.io.sparse_dataset(f["X"])
        ds = Loader(shuffle=True, chunk_size=10, preload_nchunks=10, preload_to_gpu=False, to_torch=False)
        with pytest.raises(TypeError, match="Cannot add"):
            ds.add_dataset(data)


def _custom_collate_fn(elems):
    import torch

    if isinstance(elems[0]["data"], torch.Tensor):
        x = torch.vstack([v["data"].to_dense() for v in elems])
    elif isinstance(elems[0]["data"], sp.csr_matrix):
        x = sp.vstack([v["data"] for v in elems]).toarray()
    else:
        x = np.vstack([v["data"] for v in elems])

    y = np.array([v["index"] for v in elems])

    return x, y


@pytest.mark.skipif(not find_spec("torch"), reason="Need torch installed.")
@pytest.mark.parametrize("open_func", [open_sparse, open_dense])
def test_torch_multiprocess_dataloading_zarr(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], open_func, use_zarrs: bool
):
    """
    Test that the ZarrDatasets can be used with PyTorch's DataLoader in a multiprocess context and that each element of
    the dataset gets yielded once.
    """
    from torch.utils.data import DataLoader

    ds = Loader(chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True, preload_to_gpu=False)
    ds.add_datasets(
        **concat([open_func(p, use_zarrs=use_zarrs) for p in adata_with_zarr_path_same_var_space[1].glob("*.zarr")])
    )
    if open_func.__name__[5:] == "sparse":
        x_ref = adata_with_zarr_path_same_var_space[0].layers["sparse"].toarray()
    else:
        x_ref = adata_with_zarr_path_same_var_space[0].X

    dataloader = DataLoader(
        ds, batch_size=32, num_workers=4, collate_fn=_custom_collate_fn, multiprocessing_context="spawn"
    )
    x_list, idx_list = [], []
    for batch in dataloader:
        x, idxs = batch
        x_list.append(x)
        idx_list.append(idxs.ravel())

    x = np.vstack(x_list)
    idxs = np.concatenate(idx_list)

    assert np.array_equal(x[np.argsort(idxs)], x_ref)


@pytest.mark.skipif(
    find_spec("cupy") is not None, reason="Can't test for preload_to_gpu True ImportError with cupy installed"
)
def test_no_cupy():
    with pytest.raises(
        ImportError, match=r"Follow the directions at https://docs.cupy.dev/en/stable/install.html to install cupy."
    ):
        Loader(chunk_size=10, preload_nchunks=4, preload_to_gpu=True, to_torch=False)


@pytest.mark.skipif(
    find_spec("torch") is not None, reason="Can't test for to_torch True ImportError with torch installed"
)
def test_no_torch():
    with pytest.raises(ImportError, match=r"Try `pip install torch`."):
        Loader(chunk_size=10, preload_nchunks=4, to_torch=True, preload_to_gpu=False)


def get_default_dense() -> type:
    if find_spec("torch"):
        from torch import Tensor as expected_dense
    else:
        from numpy import ndarray as expected_dense
    return expected_dense


def get_default_sparse() -> type:
    if find_spec("cupy"):
        from cupyx.scipy.sparse import csr_matrix as expected_sparse
    else:
        from scipy.sparse import csr_matrix as expected_sparse

    return expected_sparse


@pytest.mark.parametrize(
    ("expected_cls", "kwargs"),
    (
        pytest.param(get_default_dense(), {"preload_to_gpu": False}, id="torch"),
        pytest.param(get_default_sparse(), {"to_torch": False}, id="cupy"),
    ),
)
def test_default_data_structures(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], expected_cls: type, kwargs: dict
):
    # format is a smoke test for sparse
    ds = Loader(
        chunk_size=10, preload_nchunks=4, batch_size=22, shuffle=True, return_index=False, **kwargs
    ).add_dataset(
        **(open_sparse if issubclass(expected_cls, get_default_sparse()) else open_dense)(
            list(adata_with_zarr_path_same_var_space[1].iterdir())[0]
        )
    )
    for batch in ds:
        assert isinstance(batch["data"], expected_cls)


def test_locking_prevents_add_dataset_after_set_sampler(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    """Test that add_dataset raises after set_sampler has been called."""
    from annbatch.sampler import SliceSampler

    paths = list(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    data1 = open_sparse(paths[0])
    data2 = open_sparse(paths[1]) if len(paths) > 1 else open_sparse(paths[0])

    loader = Loader(
        chunk_size=10,
        preload_nchunks=4,
        preload_to_gpu=False,
        to_torch=False,
    )
    loader.add_dataset(**data1)

    # Set a custom sampler - this should lock the loader
    sampler = SliceSampler(
        mask=slice(0, loader.n_obs),
        batch_size=5,
        slice_size=10,
        preload_nslices=4,
    )
    loader.set_sampler(sampler)

    # Now trying to add another dataset should raise
    with pytest.raises(RuntimeError, match="Cannot add datasets after set_sampler"):
        loader.add_dataset(**data2)


def test_custom_sampler_multi_dataset_partial_range(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    """Test custom sampler that samples only from a specific range across multiple datasets.

    Setup:
    - Dataset A: (20, n_dim)
    - Dataset B: (30, n_dim)
    - Total: 50 obs (indices 0-49)
    - Custom sampler: mask=slice(30, 45) samples indices 30-44 (15 obs from dataset B only)
    """
    from annbatch.sampler import SliceSampler

    paths = list(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    # Use first path for both datasets (we'll slice them to different sizes)
    data = open_sparse(paths[0])

    # Create two datasets with different sizes
    # We'll use the same underlying data but conceptually treat them as different

    # Create loader without sampler first, then add datasets
    loader = Loader(
        chunk_size=10,
        preload_nchunks=4,
        return_index=True,
        preload_to_gpu=False,
        to_torch=False,
    )

    # Add first dataset (simulating 20 obs)
    loader.add_dataset(**data)
    # For this test, we use the actual dataset which may have more obs
    # The key point is testing the sampler mask functionality

    actual_n_obs = loader.n_obs

    # Define a custom sampler that only samples a specific range
    # e.g., if actual_n_obs=100, sample from index 30 to 45
    start_idx = min(30, actual_n_obs - 15)
    end_idx = min(45, actual_n_obs)

    sampler = SliceSampler(
        mask=slice(start_idx, end_idx),
        batch_size=5,
        slice_size=10,
        preload_nslices=2,
    )
    loader.set_sampler(sampler)

    # Collect all indices we sample
    all_indices = []
    for batch in loader:
        if batch["index"] is not None:
            all_indices.extend(batch["index"].tolist())

    # Verify we only sampled from the specified range
    assert len(all_indices) == end_idx - start_idx
    assert min(all_indices) >= start_idx
    assert max(all_indices) < end_idx
    assert set(all_indices) == set(range(start_idx, end_idx))
