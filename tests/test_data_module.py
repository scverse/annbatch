import numpy as np
from arrayloaders.io import ClassificationDataModule
from arrayloaders.io.dask_loader import read_lazy_store


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
