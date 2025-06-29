from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from arrayloaders.io import (
    DaskDataset,
    ZarrDenseDataset,
    ZarrSparseDataset,
    read_lazy_store,
)

if TYPE_CHECKING:
    from pathlib import Path


def open_sparse(path: Path):
    return ad.AnnData(
        X=ad.io.sparse_dataset(zarr.open(path)["layers"]["sparse"]),
        obs=ad.io.read_elem(zarr.open(path)["obs"]),
        layers={"data": ad.io.sparse_dataset(zarr.open(path)["layers"]["sparse"])},
    )


def open_dense(path: Path):
    return ad.AnnData(
        X=zarr.open(path)["X"],
        obs=ad.io.read_elem(zarr.open(path)["obs"]),
        layers={"data": zarr.open(path)["X"]},
    )


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
@pytest.mark.parametrize(
    "gen_loader",
    [
        lambda path, shuffle: DaskDataset(
            read_lazy_store(path, obs_columns=["label"]),
            label_column="label",
            n_chunks=4,
            shuffle=shuffle,
        ),
        *(
            pytest.param(
                lambda path,
                shuffle,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                dataset_class=dataset_class,
                obs_keys=obs_keys,
                layer_keys=layer_keys: dataset_class(
                    shuffle=shuffle,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                ).add_anndatas(
                    [
                        (
                            open_sparse
                            if issubclass(dataset_class, ZarrSparseDataset)
                            else open_dense
                        )(p)
                        for p in path.glob("*.zarr")
                    ],
                    layer_keys,
                    obs_keys,
                ),
                id=f"chunk_size={chunk_size}-preload_nchunks={preload_nchunks}-obs_keys={obs_keys}-dataset_class={dataset_class}-layer_keys={layer_keys}",
            )
            for chunk_size, preload_nchunks, obs_keys, dataset_class, layer_keys in [
                elem
                for dataset_class in [ZarrDenseDataset, ZarrSparseDataset]  # type: ignore[list-item]
                for elem in [
                    [1, 5, None, dataset_class, None],  # singleton chunk size
                    [5, 1, None, dataset_class, None],  # singleton preload
                    [10, 5, "label", dataset_class, None],  # singleton label key
                    [
                        10,
                        5,
                        ["label", "label", "label"],
                        dataset_class,
                        None,
                    ],  # list label key
                    [10, 5, None, dataset_class, "data"],  # singleton data key
                    [
                        10,
                        5,
                        None,
                        dataset_class,
                        ["data", "data", "data"],
                    ],  # list data key
                ]
            ]
        ),
    ],
)
def test_store_load_data(mock_store: Path, *, shuffle: bool, gen_loader):
    """
    This test verifies that the DaskDataset works correctly:
        1. The DaskDataset correctly loads data from the mock store
        2. Each sample has the expected feature dimension
        3. All samples from the dataset are processed
        4. If the dataset is not shuffled, it returns the correct data
    """
    adata = read_lazy_store(mock_store, obs_columns=["label"])

    loader = gen_loader(mock_store, shuffle)
    n_elems = 0
    batches = []
    for batch in loader:
        x, label = batch
        n_elems += 1
        # Check feature dimension
        assert x.shape[0 if (is_dense := isinstance(x, np.ndarray)) else 1] == 100
        if not shuffle:
            batches += [x]
        if label is not None:
            assert isinstance(label, np.int64)

    # check that we yield all samples from the dataset
    if not shuffle:
        # np.array for sparse
        stacked = (np if is_dense else sp).vstack(batches)
        if not is_dense:
            stacked = stacked.toarray()
            expected = adata.layers["sparse"].compute().toarray()
        else:
            expected = adata.X.compute()
        np.testing.assert_allclose(stacked, expected)
    else:
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
        lambda path: ZarrDenseDataset(
            shuffle=True,
            preload_nchunks=0,
        ),
        lambda path: ZarrDenseDataset(
            shuffle=True,
            preload_nchunks=4,
        ).add_anndatas(
            [open_dense(p) for p in path.glob("*.zarr")],
            None,
            obs_keys=[],
        ),
        *(
            (
                lambda path,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks: ZarrSparseDataset(
                    shuffle=True,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                )
            )
            for chunk_size, preload_nchunks in [[0, 10], [10, 0]]
        ),
    ],
)
def test_zarr_store_errors_lt_1(gen_loader, mock_store):
    with pytest.raises(ValueError, match="must be greater than 1"):
        gen_loader(mock_store)


@pytest.mark.parametrize(
    "layer_keys_less", [True, False], ids=["anndatas_shorter", "layer_keys_shorter"]
)
def test_layers_keys_anndata_mismatch(mock_store, layer_keys_less):
    with pytest.raises(ValueError, match="must match number of layer keys"):
        ZarrSparseDataset(
            shuffle=True,
            chunk_size=10,
            preload_nchunks=10,
        ).add_anndatas(
            [open_sparse(p) for p in mock_store.glob("*.zarr")]
            if layer_keys_less
            else [open_sparse(next(mock_store.glob("*.zarr")))],
            (["sparse"] * len(list(mock_store.glob("*.zarr"))))
            if not layer_keys_less
            else ["sparse"],
        )
