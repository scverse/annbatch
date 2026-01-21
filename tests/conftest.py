from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from scipy.sparse import random as sparse_random

from annbatch import write_sharded
from annbatch.io import DatasetCollection

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(params=[False, True], ids=["zarr-python", "zarrs"])
def use_zarrs(request):
    return request.param


@pytest.fixture(scope="session")
def adata_with_zarr_path_same_var_space(tmpdir_factory, n_shards: int = 3) -> Generator[tuple[ad.AnnData, Path]]:
    """Create a mock Zarr store for testing."""
    feature_dim = 100
    n_cells_per_shard = 200
    tmp_path = Path(tmpdir_factory.mktemp("stores"))
    adata_lst = []
    for shard in range(n_shards):
        adata = ad.AnnData(
            X=np.random.random((n_cells_per_shard, feature_dim)).astype("f4"),
            obs=pd.DataFrame(
                {"label": np.random.default_rng().integers(0, 5, size=n_cells_per_shard)},
                index=np.arange(n_cells_per_shard).astype(str),
            ),
            layers={
                "sparse": sp.random(
                    n_cells_per_shard, feature_dim, format="csr", rng=np.random.default_rng(), dtype="int32"
                )
            },
        )
        adata_lst += [adata]
        f = zarr.open_group(tmp_path / f"chunk_{shard}.zarr", mode="w", zarr_format=3)
        write_sharded(
            f,
            adata,
            sparse_chunk_size=10,
            sparse_shard_size=20,
            dense_chunk_size=10,
            dense_shard_size=20,
        )
    yield (
        # need to match directory iteration order for correctness so can't just concatenate
        ad.concat([ad.read_zarr(tmp_path / shard) for shard in tmp_path.iterdir() if str(shard).endswith(".zarr")]),
        tmp_path,
    )


@pytest.fixture(scope="session")
def adata_with_h5_path_different_var_space(
    tmpdir_factory,
    request,
) -> tuple[ad.AnnData, Path]:
    """Create mock anndata objects for testing."""
    params = getattr(request, "param", {})
    n_adatas = params.get("n_adatas", 6)
    all_adatas_have_raw = params.get("all_adatas_have_raw", True)

    tmp_path = Path(tmpdir_factory.mktemp("raw_adatas"))
    tmp_path = tmp_path / "h5_files"
    tmp_path.mkdir()
    n_features = [random.randint(50, 100) for _ in range(n_adatas)]
    n_cells = [random.randint(50, 100) for _ in range(n_adatas)]
    adatas = []
    for i, (m, n) in enumerate(zip(n_cells, n_features, strict=True)):
        var_idx = [f"gene_{gene}" for gene in range(n // 2)] + [f"gene_{gene}_{i}" for gene in range(n // 2, n)]
        obs_idx = [f"cell_{j}" for j in np.arange(m).astype(str) + f"-{i}"]
        adata = ad.AnnData(
            X=sparse_random(m, n, density=0.1, format="csr", dtype="f4"),
            obs=pd.DataFrame(
                {
                    "label": pd.Categorical([str(m), str(m), *(["a"] * (m - 2))]),
                    "store_id": [i] * m,
                    "numeric": np.arange(m),
                },
                index=obs_idx,
            ),
            var=pd.DataFrame(index=var_idx),
            obsm={"arr": np.random.randn(m, 10), "df": pd.DataFrame({"numeric": np.arange(m)}, index=obs_idx)},
            varm={"arr": np.random.randn(n, 10), "df": pd.DataFrame({"numeric": np.arange(n)}, index=var_idx)},
        )
        if all_adatas_have_raw or (i % 2 == 0):
            adata_raw = adata[:, adata.var.index[: (n // 2)]].copy()
            adata_raw.obsm = None
            adata.raw = adata_raw
        adata.write_h5ad(tmp_path / f"adata_{i}.h5ad", compression="gzip")
        adatas += [adata]
    return ad.concat(
        [ad.read_h5ad(tmp_path / shard) for shard in sorted(tmp_path.iterdir()) if str(shard).endswith(".h5ad")],
        join="outer",
    ), tmp_path


@pytest.fixture(scope="session")
def simple_collection(
    tmpdir_factory, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]
) -> tuple[DatasetCollection, ad.AnnData]:
    zarr_stores = sorted(f for f in adata_with_zarr_path_same_var_space[1].iterdir() if f.is_dir())
    output_path = Path(tmpdir_factory.mktemp("zarr_folder")) / "simple_fixture.zarr"
    collection = DatasetCollection(output_path).add_adatas(
        zarr_stores,
        zarr_sparse_chunk_size=10,
        zarr_sparse_shard_size=20,
        zarr_dense_chunk_size=10,
        zarr_dense_shard_size=20,
        n_obs_per_dataset=60,
        shuffle_chunk_size=10,
    )
    return ad.concat([ad.io.read_elem(ds) for ds in collection], join="outer"), collection
