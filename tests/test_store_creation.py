from __future__ import annotations

import random
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from arrayloaders import add_h5ads_to_store, create_store_from_h5ads
from scipy.sparse import random as sparse_random

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def anndata_settings():
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr


@pytest.fixture
def mock_anndatas_path(tmp_path: Path, n_adatas: int = 4) -> tuple[ad.AnnData, Path]:
    """Create mock anndata objects for testing."""
    tmp_path = tmp_path / "adatas"
    tmp_path.mkdir(parents=True, exist_ok=True)
    n_features = [random.randint(50, 100) for _ in range(n_adatas)]
    n_cells = [random.randint(50, 100) for _ in range(n_adatas)]
    adatas = []
    for i, (m, n) in enumerate(zip(n_cells, n_features, strict=True)):
        adata = ad.AnnData(
            X=sparse_random(m, n, density=0.1, format="csr", dtype="f4"),
            obs=pd.DataFrame(
                {"label": np.random.default_rng().integers(0, 5, size=m)},
                index=np.arange(m).astype(str),
            ),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(n)]),
        )

        adata.write_h5ad(tmp_path / f"adata_{i}.h5ad", compression="gzip")
        adatas += [adata]
    return ad.concat(adatas, join="outer"), tmp_path


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("densify", [True, False])
def test_store_creation(
    mock_anndatas_path: tuple[ad.AnnData, Path],
    shuffle: bool,
    densify: bool,
):
    var_subset = [f"gene_{i}" for i in range(100)]

    (mock_anndatas_path[1] / "zarr_store").mkdir(parents=True, exist_ok=True)
    create_store_from_h5ads(
        [mock_anndatas_path[1] / f for f in mock_anndatas_path[1].iterdir() if str(f).endswith(".h5ad")],
        mock_anndatas_path[1] / "zarr_store",
        var_subset,
        chunk_size=10,
        shard_size=20,
        buffer_size=60,
        shuffle=shuffle,
        should_denseify=densify,
    )

    adata_orig = mock_anndatas_path[0]
    adata = ad.concat(
        [ad.read_zarr(zarr_path) for zarr_path in (mock_anndatas_path[1] / "zarr_store").iterdir()], join="outer"
    )
    assert adata.X.shape[0] == adata_orig.X.shape[0]
    assert adata.X.shape[1] == adata_orig.X.shape[1]
    assert np.array_equal(
        sorted(adata.var.index),
        sorted(adata_orig.var.index),
    )


@pytest.mark.parametrize("densify", [True, False])
@pytest.mark.parametrize("cache_h5ads", [True, False])
def test_store_extension(mock_anndatas_path: tuple[ad.AnnData, Path], densify: bool, cache_h5ads: bool):
    store_path = mock_anndatas_path[1] / "zarr_store"
    # create new store
    create_store_from_h5ads(
        [mock_anndatas_path[1] / f for f in mock_anndatas_path[1].iterdir() if str(f).endswith(".h5ad")],
        store_path,
        chunk_size=10,
        shard_size=20,
        buffer_size=60,
        shuffle=True,
        should_denseify=densify,
    )
    # add h5ads to existing store
    add_h5ads_to_store(
        [mock_anndatas_path[1] / f for f in mock_anndatas_path[1].iterdir() if str(f).endswith(".h5ad")],
        store_path,
        cache_h5ads=cache_h5ads,
        chunk_size=10,
        shard_size=20,
    )

    adata = ad.concat([ad.read_zarr(zarr_path) for zarr_path in (mock_anndatas_path[1] / "zarr_store").iterdir()])
    adata_orig = ad.concat([mock_anndatas_path[0], mock_anndatas_path[0]], join="outer")
    assert adata.X.shape[0] == adata_orig.X.shape[0]
