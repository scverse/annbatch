import os
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import random as sparse_random

from arrayloaders.io.dask_loader import read_lazy_store
from arrayloaders.io.store_creation import create_store_from_h5ads


@pytest.fixture(autouse=True)
def anndata_settings():
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr


@pytest.fixture
def mock_anndatas(tmp_path: Path, n_adatas: int = 4):
    """Create mock anndata objects for testing."""
    (tmp_path / "adatas").mkdir(parents=True, exist_ok=True)
    n_features = [random.randint(50, 100) for _ in range(n_adatas)]
    n_cells = [random.randint(50, 100) for _ in range(n_adatas)]

    for i, (m, n) in enumerate(zip(n_cells, n_features)):
        adata = ad.AnnData(
            X=sparse_random(m, n, density=0.1, format="csr", dtype="f4"),
            obs=pd.DataFrame(
                {"label": np.random.default_rng().integers(0, 5, size=m)},
                index=np.arange(m).astype(str),
            ),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(n)]),
        )

        adata.write_h5ad(tmp_path / f"adatas/adata_{i}.h5ad", compression="gzip")

    return tmp_path


def test_store_creation(tmp_path: Path):
    var_subset = [f"gene_{i}" for i in range(100)]

    (tmp_path / "zarr_store").mkdir(parents=True, exist_ok=True)
    create_store_from_h5ads(
        [
            tmp_path / f"adatas/{f}"
            for f in os.listdir(tmp_path / "adatas")
            if f.endswith(".h5ad")
        ],
        tmp_path / "zarr_store",
        var_subset,
        chunk_size=10,
        shard_size=20,
        shuffle_buffer_size=60,
    )

    adatas = [
        ad.read_h5ad(tmp_path / f"adatas/{f}")
        for f in os.listdir(tmp_path / "adatas")
        if f.endswith(".h5ad")
    ]
    adata = read_lazy_store(tmp_path / "zarr_store")
    assert adata.X.shape[0] == sum([adata.shape[0] for adata in adatas])
    assert adata.X.shape[1] == len(
        [gene for gene in var_subset if gene in adata.var.index.tolist()]
    )
    assert np.array_equal(
        sorted(adata.var.index.tolist()),
        sorted([gene for gene in var_subset if gene in adata.var.index.tolist()]),
    )
