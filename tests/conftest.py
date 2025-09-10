from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import write_sharded

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def anndata_settings():
    ad.settings.zarr_write_format = 3  # Needed to support sharding in Zarr


@pytest.fixture(params=[False, True], ids=["zarr-python", "zarrs"])
def use_zarrs(request):
    return request.param


@pytest.fixture(scope="session")
def adata_with_path(tmpdir_factory, n_shards: int = 3) -> Generator[tuple[ad.AnnData, Path]]:
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
                    n_cells_per_shard,
                    feature_dim,
                    format="csr",
                    rng=np.random.default_rng(),
                )
            },
        )
        adata_lst += [adata]
        f = zarr.open_group(tmp_path / f"chunk_{shard}.zarr", mode="w", zarr_format=3)
        write_sharded(
            f,
            adata,
            chunk_size=10,
            shard_size=20,
        )
    yield (
        # need to match directory iteration order for correctness so can't just concatenate
        ad.concat([ad.read_zarr(tmp_path / shard) for shard in tmp_path.iterdir() if str(shard).endswith(".zarr")]),
        tmp_path,
    )
