from __future__ import annotations

import platform
from importlib.util import find_spec
from types import NoneType
from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
import zarr
import zarrs  # noqa: F401
from torch.utils.data import DataLoader

from arrayloaders import ZarrDenseDataset, ZarrSparseDataset

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
            "obs": ad.io.read_elem(zarr.open(path)["obs"])["label"].to_numpy(),
        }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=pd.DataFrame({"label": data["obs"]}))
    return data


def open_dense(path: Path, *, use_zarrs: bool = False, use_anndata: bool = False) -> Data | ad.AnnData:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline" if use_zarrs else old_pipeline}):
        data = {
            "dataset": zarr.open(path)["X"],
            "obs": ad.io.read_elem(zarr.open(path)["obs"])["label"].to_numpy(),
        }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=pd.DataFrame({"label": data["obs"]}))
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
            dataset_class=dataset_class,
            batch_size=batch_size,
            preload_to_gpu=preload_to_gpu,
            obs_keys=obs_keys: dataset_class(
                shuffle=shuffle,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                return_index=True,
                batch_size=batch_size,
                preload_to_gpu=preload_to_gpu,
                to_torch=False,
            ).add_anndatas(
                [
                    (open_sparse if issubclass(dataset_class, ZarrSparseDataset) else open_dense)(
                        p, use_zarrs=use_zarrs, use_anndata=True
                    )
                    for p in path.glob("*.zarr")
                ],
                obs_keys=obs_keys,
            ),
            id=f"chunk_size={chunk_size}-preload_nchunks={preload_nchunks}-obs_keys={obs_keys}-dataset_class={dataset_class.__name__}-layer_keys={layer_keys}-batch_size={batch_size}{'-cupy' if preload_to_gpu else ''}",  # type: ignore[attr-defined]
            marks=pytest.mark.skipif(
                find_spec("cupy") is None and preload_to_gpu,
                reason="need cupy installed",
            ),
        )
        for chunk_size, preload_nchunks, obs_keys, dataset_class, layer_keys, batch_size, preload_to_gpu in [
            elem
            for preload_to_gpu in [True, False]
            for obs_keys in [None, "label"]
            for dataset_class in [ZarrDenseDataset, ZarrSparseDataset]  # type: ignore[list-item]
            for elem in [
                [
                    1,
                    5,
                    obs_keys,
                    dataset_class,
                    None,
                    1,
                    preload_to_gpu,
                ],  # singleton chunk size
                [
                    5,
                    1,
                    obs_keys,
                    dataset_class,
                    None,
                    1,
                    preload_to_gpu,
                ],  # singleton preload
                [
                    10,
                    5,
                    obs_keys,
                    dataset_class,
                    None,
                    5,
                    preload_to_gpu,
                ],  # batch size divides total in memory size evenly
                [
                    10,
                    5,
                    obs_keys,
                    dataset_class,
                    None,
                    50,
                    preload_to_gpu,
                ],  # batch size equal to in-memory size loading
                [
                    10,
                    5,
                    obs_keys,
                    dataset_class,
                    None,
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
    loader = gen_loader(adata_with_zarr_path_same_var_space[1], shuffle, use_zarrs)
    adata = adata_with_zarr_path_same_var_space[0]
    is_dense = isinstance(loader, ZarrDenseDataset)
    n_elems = 0
    batches = []
    labels = []
    indices = []
    expected_data = adata.X if is_dense else adata.layers["sparse"].toarray()
    for batch in loader:
        x, label, index = batch
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
            expected_labels = adata.obs["label"]
            np.testing.assert_allclose(
                np.concatenate(labels).ravel(),
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
            lambda path,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            dataset_class=dataset_class: dataset_class(
                shuffle=True,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
            )
        )
        for chunk_size, preload_nchunks in [[0, 10], [10, 0]]
        for dataset_class in [ZarrSparseDataset, ZarrDenseDataset]
    ],
)
def test_zarr_store_errors_lt_1(gen_loader, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    with pytest.raises(ValueError, match="must be greater than 1"):
        gen_loader(adata_with_zarr_path_same_var_space[1])


def test_bad_adata_X_type(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    data = open_dense(next(adata_with_zarr_path_same_var_space[1].glob("*.zarr")))
    data["dataset"] = data["dataset"][...]
    ds = ZarrDenseDataset(shuffle=True, chunk_size=10, preload_nchunks=10, preload_to_gpu=False)
    with pytest.raises(TypeError, match="Cannot create"):
        ds.add_dataset(**data)


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
@pytest.mark.parametrize(
    ["dataset_class", "open_func"], [[ZarrSparseDataset, open_sparse], [ZarrDenseDataset, open_dense]]
)
def test_to_torch(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
    dataset_class: type[ZarrDenseDataset] | type[ZarrSparseDataset],
    open_func: Callable[[Path], Data],
    preload_to_gpu: bool,
):
    # batch_size guaranteed to have leftovers to drop
    ds = dataset_class(
        shuffle=False,
        chunk_size=5,
        preload_nchunks=10,
        batch_size=42,
        preload_to_gpu=preload_to_gpu,
        return_index=True,
        to_torch=True,
    )
    ds.add_dataset(**open_func(next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))))
    assert isinstance(next(iter(ds))[0], torch.Tensor)


def test_drop_last(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    # batch_size guaranteed to have leftovers to drop
    ds = ZarrSparseDataset(
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
    for x, _, idx in ds:
        batches += [x]
        indices += [idx]
    X = sp.vstack(batches).toarray()
    assert X.shape[0] < adata.shape[0]
    X_expected = adata[np.concatenate(indices)].layers["sparse"].toarray()
    np.testing.assert_allclose(X, X_expected)


def test_bad_adata_X_hdf5(adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path]):
    with h5py.File(next(adata_with_h5_path_different_var_space[1].glob("*.h5ad"))) as f:
        data = ad.io.sparse_dataset(f["X"])
        ds = ZarrDenseDataset(shuffle=True, chunk_size=10, preload_nchunks=10, preload_to_gpu=False)
        with pytest.raises(TypeError, match="Cannot create"):
            ds.add_dataset(data)


def _custom_collate_fn(elems):
    if isinstance(elems[0][0], torch.Tensor):
        x = torch.vstack([v[0].to_dense() for v in elems])
    elif isinstance(elems[0][0], sp.csr_matrix):
        x = sp.vstack([v[0] for v in elems]).toarray()
    else:
        x = np.vstack([v[0] for v in elems])

    if len(elems[0]) == 2:
        y = np.array([v[1] for v in elems])
    else:
        y = np.array([v[2] for v in elems])

    return x, y


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="See: https://github.com/scverse/anndata/issues/2021 potentially",
)
def test_dataloader_fails_linux_with_anndata(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    ds = ZarrSparseDataset(chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True, preload_to_gpu=False)
    ds.add_anndatas(
        [
            open_sparse(p, use_zarrs=True, use_anndata=True)
            for p in adata_with_zarr_path_same_var_space[1].glob("*.zarr")
        ]
    )
    dataloader = DataLoader(
        ds,
        batch_size=32,
        num_workers=4,
        collate_fn=_custom_collate_fn,
    )
    with pytest.raises(NotImplementedError, match=r"why we can't load anndata from torch"):
        next(iter(dataloader))
    ds = ZarrSparseDataset(chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True)
    ds.add_datasets(**concat([open_sparse(p) for p in adata_with_path[1].glob("*.zarr")]))
    dataloader = DataLoader(
        ds,
        batch_size=32,
        num_workers=4,
        collate_fn=_custom_collate_fn,
    )
    next(iter(dataloader))


@pytest.mark.parametrize("loader", [ZarrDenseDataset, ZarrSparseDataset])
@pytest.mark.skipif(
    platform.system() == "Linux",
    reason="See: https://github.com/scverse/anndata/issues/2021 potentially",
)
def test_torch_multiprocess_dataloading_zarr(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], loader, use_zarrs
):
    """
    Test that the ZarrDatasets can be used with PyTorch's DataLoader in a multiprocess context and that each element of
    the dataset gets yielded once.
    """

    if issubclass(loader, ZarrSparseDataset):
        ds = ZarrSparseDataset(chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True, preload_to_gpu=False)
        ds.add_datasets(
            **concat(
                [open_sparse(p, use_zarrs=use_zarrs) for p in adata_with_zarr_path_same_var_space[1].glob("*.zarr")]
            )
        )
        x_ref = adata_with_zarr_path_same_var_space[0].layers["sparse"].toarray()
    elif issubclass(loader, ZarrDenseDataset):
        ds = ZarrDenseDataset(chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True, preload_to_gpu=False)
        ds.add_datasets(
            **concat(
                [open_dense(p, use_zarrs=use_zarrs) for p in adata_with_zarr_path_same_var_space[1].glob("*.zarr")]
            )
        )
        x_ref = adata_with_zarr_path_same_var_space[0].X
    else:
        raise ValueError("Unknown loader type")

    dataloader = DataLoader(
        ds,
        batch_size=32,
        num_workers=4,
        collate_fn=_custom_collate_fn,
    )
    x_list, idx_list = [], []
    for batch in dataloader:
        x, idxs = batch
        x_list.append(x)
        idx_list.append(idxs.ravel())

    x = np.vstack(x_list)
    idxs = np.concatenate(idx_list)

    assert np.array_equal(x[np.argsort(idxs)], x_ref)


@pytest.mark.skipif(find_spec("cupy") is not None, reason="Can't test for no cupy if cupy is there")
def test_no_cupy(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    ds = ZarrDenseDataset(
        chunk_size=10,
        preload_nchunks=4,
        shuffle=True,
        return_index=True,
        preload_to_gpu=True,
    ).add_dataset(**open_dense(list(adata_with_zarr_path_same_var_space[1].iterdir())[0]))
    with pytest.raises(ImportError, match=r"even though `preload_to_gpu` argument"):
        next(iter(ds))
