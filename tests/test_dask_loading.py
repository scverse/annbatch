from pathlib import Path

import numpy as np
import pytest

from arrayloaders.io.dask_loader import DaskDataset, read_lazy_store


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
def test_zarr_store(mock_store: Path, *, shuffle: bool):
    """
    This test verifies that the DaskDataset works correctly:
        1. The DaskDataset correctly loads data from the mock store
        2. Each sample has the expected feature dimension
        3. All samples from the dataset are processed
        4. If the dataset is not shuffled, it returns the correct data
    """
    adata = read_lazy_store(mock_store, obs_columns=["label"])

    loader = DaskDataset(adata, label_column="label", n_chunks=4, shuffle=shuffle)
    n_elems = 0
    batches = []
    for batch in loader:
        x, _ = batch
        n_elems += 1
        assert x.shape[0] == 100  # Check feature dimension
        if not shuffle:
            batches += [x]

    # check that we yield all samples from the dataset
    if not shuffle:
        assert (np.vstack(batches) == adata.X.compute()).all()
    else:
        assert n_elems == adata.shape[0]
