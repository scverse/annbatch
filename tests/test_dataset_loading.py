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
        lambda path, shuffle: ZarrDenseDataset(
            [zarr.open(p)["X"] for p in path.glob("*.zarr")],
            obs_list=[
                ad.io.read_elem(zarr.open(p)["obs"]) for p in path.glob("*.zarr")
            ],
            shuffle=shuffle,
            obs_column="label",
        ),
        *(
            (
                lambda path,
                shuffle,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks: ZarrSparseDataset(
                    [
                        ad.io.sparse_dataset(zarr.open(p)["layers"]["sparse"])
                        for p in path.glob("*.zarr")
                    ],
                    shuffle=shuffle,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                )
            )
            for chunk_size, preload_nchunks in [[1, 10], [10, 1], [10, 5], [5, 10]]
        ),
    ],
    ids=[
        "dask",
        "dense",
        "sparse_chunksize_1",
        "sparse_preload_1",
        "sparse_chunksize_gt_preload",
        "sparse_preload_gt_chunksize",
    ],
)
def test_zarr_store(mock_store: Path, *, shuffle: bool, gen_loader):
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
        x, _ = batch
        n_elems += 1
        # Check feature dimension
        assert x.shape[0 if (is_dense := isinstance(x, np.ndarray)) else 1] == 100
        if not shuffle:
            batches += [x]

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
            [zarr.open(p)["X"] for p in path.glob("*.zarr")],
            obs_list=[
                ad.io.read_elem(zarr.open(p)["obs"]) for p in path.glob("*.zarr")
            ],
            shuffle=True,
            obs_column="label",
            preload_nchunks=0,
        ),
        lambda path: ZarrDenseDataset(
            [zarr.open(p)["X"] for p in path.glob("*.zarr")],
            obs_list=[],
            shuffle=True,
            obs_column="label",
            preload_nchunks=4,
        ),
        *(
            (
                lambda path,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks: ZarrSparseDataset(
                    [
                        ad.io.sparse_dataset(zarr.open(p)["layers"]["sparse"])
                        for p in path.glob("*.zarr")
                    ],
                    shuffle=True,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                )
            )
            for chunk_size, preload_nchunks in [[0, 10], [10, 0]]
        ),
        lambda path: ZarrSparseDataset(
            [],
            shuffle=True,
            chunk_size=10,
            preload_nchunks=10,
        ),
    ],
)
def test_zarr_store_errors_lt_1(gen_loader, mock_store):
    with pytest.raises(ValueError, match="must be greater than 1"):
        gen_loader(mock_store)
