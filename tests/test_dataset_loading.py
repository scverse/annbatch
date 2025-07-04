from __future__ import annotations

import platform
from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import zarr
import zarrs  # noqa: F401

from arrayloaders.io import (
    DaskDataset,
    ZarrDenseDataset,
    ZarrSparseDataset,
    read_lazy_store,
)

if TYPE_CHECKING:
    from pathlib import Path


class Data(TypedDict):
    dataset: ad.abc.CSRDataset | zarr.Array
    obs: np.ndarray


class ListData(TypedDict):
    datasets: list[ad.abc.CSRDataset | zarr.Array]
    obs: list[np.ndarray]


def open_sparse(path: Path, *, use_zarrs: bool = False) -> Data:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline"
            if use_zarrs
            else old_pipeline
        }
    ):
        return {
            "dataset": ad.io.sparse_dataset(zarr.open(path)["layers"]["sparse"]),
            "obs": ad.io.read_elem(zarr.open(path)["obs"])["label"].to_numpy(),
        }


def open_dense(path: Path, *, use_zarrs: bool = False) -> Data:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline"
            if use_zarrs
            else old_pipeline
        }
    ):
        return {
            "dataset": zarr.open(path)["X"],
            "obs": ad.io.read_elem(zarr.open(path)["obs"])["label"].to_numpy(),
        }


def concat(dicts: list[Data]) -> ListData:
    return {
        "datasets": [d["dataset"] for d in dicts],
        "obs": [d["obs"] for d in dicts],
    }


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
@pytest.mark.parametrize(
    "gen_loader",
    [
        pytest.param(
            lambda path, shuffle, use_zarrs: DaskDataset(
                read_lazy_store(path, obs_columns=["label"]),
                label_column="label",
                n_chunks=4,
                shuffle=shuffle,
            ),
            id="dask",
        ),
        *(
            pytest.param(
                lambda path,
                shuffle,
                use_zarrs,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                dataset_class=dataset_class,
                batch_size=batch_size: dataset_class(
                    shuffle=shuffle,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                    return_index=True,
                    batch_size=batch_size,
                ).add_datasets(
                    **concat(
                        [
                            (
                                open_sparse
                                if issubclass(dataset_class, ZarrSparseDataset)
                                else open_dense
                            )(p, use_zarrs=use_zarrs)
                            for p in path.glob("*.zarr")
                        ]
                    )
                ),
                id=f"chunk_size={chunk_size}-preload_nchunks={preload_nchunks}-obs_keys={obs_keys}-dataset_class={dataset_class.__name__}-layer_keys={layer_keys}-batch_size={batch_size}",  # type: ignore[attr-defined]
            )
            for chunk_size, preload_nchunks, obs_keys, dataset_class, layer_keys, batch_size in [
                elem
                for dataset_class in [ZarrDenseDataset, ZarrSparseDataset]  # type: ignore[list-item]
                for elem in [
                    [1, 5, None, dataset_class, None, 1],  # singleton chunk size
                    [5, 1, None, dataset_class, None, 1],  # singleton preload
                    [
                        10,
                        5,
                        None,
                        dataset_class,
                        None,
                        5,
                    ],  # batch size divides total in memory size evenly
                    [
                        10,
                        5,
                        None,
                        dataset_class,
                        None,
                        50,
                    ],  # batch size equal to in-memory size loading
                    [
                        10,
                        5,
                        None,
                        dataset_class,
                        None,
                        15,
                    ],  # batch size does not divide in memory size evenly
                ]
            ]
        ),
    ],
)
def test_store_load_dataset(mock_store: Path, *, shuffle: bool, gen_loader, use_zarrs):
    """
    This test verifies that the DaskDataset works correctly:
        1. The DaskDataset correctly loads data from the mock store
        2. Each sample has the expected feature dimension
        3. All samples from the dataset are processed
        4. If the dataset is not shuffled, it returns the correct data
    """
    adata = read_lazy_store(mock_store, obs_columns=["label"])

    loader = gen_loader(mock_store, shuffle, use_zarrs)
    is_dense = isinstance(loader, ZarrDenseDataset | DaskDataset)
    n_elems = 0
    batches = []
    labels = []
    indices = []
    expected_data = (
        adata.X.compute() if is_dense else adata.layers["sparse"].compute().toarray()
    )
    for batch in loader:
        if isinstance(loader, DaskDataset):
            x, label = batch
            index = None
        else:
            x, label, index = batch
        n_elems += 1 if (is_dask := isinstance(loader, DaskDataset)) else x.shape[0]
        # Check feature dimension
        assert x.shape[0 if is_dask else 1] == 100
        batches += [x]
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
            np.testing.assert_allclose(np.array(labels).ravel(), expected_labels)
    else:
        if len(indices) > 0:
            indices = np.concatenate(indices).ravel()
            np.testing.assert_allclose(stacked, expected_data[indices])
        assert n_elems == adata.shape[0]


@pytest.mark.parametrize(
    "gen_loader",
    [
        lambda path: DaskDataset(
            read_lazy_store(path, obs_columns=["label"])[:0],
            label_column="label",
            n_chunks=4,
            shuffle=True,
        ),
        lambda path: DaskDataset(
            read_lazy_store(path, obs_columns=["label"]),
            label_column="label",
            n_chunks=0,
            shuffle=True,
        ),
        *(
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
        ),
    ],
)
def test_zarr_store_errors_lt_1(gen_loader, mock_store):
    with pytest.raises(ValueError, match="must be greater than 1"):
        gen_loader(mock_store)


def test_bad_adata_X_type(mock_store):
    data = open_dense(next(mock_store.glob("*.zarr")))
    data["dataset"] = data["dataset"][...]
    ds = ZarrDenseDataset(
        shuffle=True,
        chunk_size=10,
        preload_nchunks=10,
    )
    with pytest.raises(TypeError, match="Cannot add a dataset"):
        ds.add_dataset(**data)


def _custom_collate_fn(elems):
    if isinstance(elems[0][0], sp.csr_matrix):
        x = sp.vstack([v[0] for v in elems]).toarray()
    else:
        x = np.vstack([v[0] for v in elems])

    if len(elems[0]) == 2:
        y = np.array([v[1] for v in elems])
    else:
        y = np.array([v[2] for v in elems])

    return x, y


@pytest.mark.parametrize("loader", [DaskDataset, ZarrDenseDataset, ZarrSparseDataset])
@pytest.mark.skipif(
    platform.system() == "Linux",
    reason="See: https://github.com/scverse/anndata/issues/2021 potentially",
)
def test_torch_multiprocess_dataloading_zarr(mock_store, loader, use_zarrs):
    """
    Test that the ZarrDatasets can be used with PyTorch's DataLoader in a multiprocess context and that each element of
    the dataset gets yielded once.
    """
    from torch.utils.data import DataLoader

    if issubclass(loader, ZarrSparseDataset):
        ds = ZarrSparseDataset(
            chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True
        )
        ds.add_datasets(
            **concat(
                [open_sparse(p, use_zarrs=use_zarrs) for p in mock_store.glob("*.zarr")]
            )
        )
        x_ref = (
            read_lazy_store(mock_store, obs_columns=["label"])
            .layers["sparse"]
            .compute()
            .toarray()
        )
    elif issubclass(loader, ZarrDenseDataset):
        ds = ZarrDenseDataset(
            chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True
        )
        ds.add_datasets(
            **concat(
                [open_dense(p, use_zarrs=use_zarrs) for p in mock_store.glob("*.zarr")]
            )
        )
        x_ref = read_lazy_store(mock_store, obs_columns=["label"]).X.compute(
            scheduler="single-threaded"
        )
    elif issubclass(loader, DaskDataset):
        adata = read_lazy_store(mock_store, obs_columns=["label"])
        adata.obs["order"] = np.arange(adata.shape[0])
        ds = DaskDataset(
            adata,
            label_column="order",
            n_chunks=4,
            shuffle=True,
        )
        x_ref = adata.X.compute(scheduler="single-threaded")
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
