import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from arrayloaders.io import ClassificationDataModule
from arrayloaders.io.dask_loader import DaskDataset, _sample_rows, read_lazy_store
from arrayloaders.io.store_creation import _write_sharded


@pytest.fixture(autouse=True)
def anndata_settings():
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr


@pytest.fixture
def mock_store(tmp_path, n_shards: int = 3):
    """Create a mock Zarr store for testing."""
    feature_dim = 100
    n_cells_per_shard = 200

    for shard in range(n_shards):
        adata = ad.AnnData(
            X=da.random.random(
                (n_cells_per_shard, feature_dim), chunks=(10, -1)
            ).astype("f4"),
            obs=pd.DataFrame(
                {
                    "label": np.random.default_rng().integers(
                        0, 5, size=n_cells_per_shard
                    )
                },
                index=np.arange(n_cells_per_shard).astype(str),
            ),
        )

        f = zarr.open(tmp_path / f"chunk_{shard}.zarr", mode="w", zarr_format=3)
        _write_sharded(
            f,
            adata,
            chunk_size=10,
            shard_size=20,
        )

    return tmp_path


def test_zarr_store(mock_store):
    """
    This test verifies that the DaskDataset works correctly:
        1. The DaskDataset correctly loads data from the mock store
        2. Each sample has the expected feature dimension
        3. All samples from the dataset are processed
    """
    adata = read_lazy_store(mock_store, obs_columns=["label"])

    loader = DaskDataset(adata, label_column="label", n_chunks=4)
    n_elems = 0
    for batch in loader:
        x, y = batch
        n_elems += 1
        assert x.shape[0] == 100  # Check feature dimension

    # check that we yield all samples from the dataset
    assert n_elems == adata.shape[0]


def test_datamodule(mock_store):
    """
    This test verifies that the dataloader for training works correctly:
        1. The training dataloader correctly initializes with training data
        2. The train_dataloader produces batches with expected dimensions
        3. The batch size matches the configured value
    """
    adata = read_lazy_store(mock_store, obs_columns=["label"])
    dm = ClassificationDataModule(
        adata_train=adata,
        adata_val=None,
        label_column="label",
        train_dataloader_kwargs={
            "batch_size": 15,
            "drop_last": True,
        },
    )

    for batch in dm.train_dataloader():
        x, y = batch
        assert x.shape[1] == 100
        assert x.shape[0] == 15  # Check batch size
        assert y.shape[0] == 15


def test_datamodule_inference(mock_store):
    """
    This test verifies that the dataloader for inference (validation) works correctly:
        1. The validation dataloader correctly loads data from the mock store
        2. The batches have the expected feature dimension
        3. All data points from the original dataset are correctly included
        4. The order of the samples are correctly preserved during loading
    """
    adata = read_lazy_store(mock_store, obs_columns=["label"])
    dm = ClassificationDataModule(
        adata_train=None,
        adata_val=adata,
        label_column="label",
        train_dataloader_kwargs={
            "batch_size": 15,
        },
    )

    x_list, y_list = [], []
    for batch in dm.val_dataloader():
        x, y = batch
        x_list.append(x.detach().numpy())
        y_list.append(y.detach().numpy())
        assert x.shape[1] == 100

    assert np.array_equal(np.vstack(x_list), adata.X.compute())
    assert np.array_equal(np.hstack(y_list), adata.obs["label"].to_numpy())


def test_sample_rows_basic():
    """
    This test checks the _sample_rows function without shuffling.

    Verifies that the function yields the expected (x, y) pairs
    when given lists of arrays and labels, and shuffle is set to False.
    """
    x_list = [np.arange(6).reshape(3, 2), np.arange(8, 16).reshape(4, 2)]
    y_list = [np.array([0, 1, 2]), np.array([3, 4, 5, 6])]
    # Test without shuffling
    result = list(_sample_rows(x_list, y_list, shuffle=False))
    expected = [
        (np.array([0, 1]), 0),
        (np.array([2, 3]), 1),
        (np.array([4, 5]), 2),
        (np.array([8, 9]), 3),
        (np.array([10, 11]), 4),
        (np.array([12, 13]), 5),
        (np.array([14, 15]), 6),
    ]
    for (x, y), (ex, ey) in zip(result, expected):
        np.testing.assert_array_equal(x, ex)
        assert y == ey


def test_sample_rows_shuffle():
    """
    This test checks the _sample_rows function with shuffling enabled.

    Ensures that all unique (x, y) pairs are present in the result,
    regardless of order, when shuffle is set to True.
    """
    x_list = [np.arange(6).reshape(3, 2), np.arange(8, 16).reshape(4, 2)]
    y_list = [np.array([0, 1, 2]), np.array([3, 4, 5, 6])]
    result = list(_sample_rows(x_list, y_list, shuffle=True))
    # Should have all unique pairs, order may differ
    assert sorted([tuple(x) + (y,) for x, y in result]) == [
        (0, 1, 0),
        (2, 3, 1),
        (4, 5, 2),
        (8, 9, 3),
        (10, 11, 4),
        (12, 13, 5),
        (14, 15, 6),
    ]
