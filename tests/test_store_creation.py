from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest

from arrayloaders import add_h5ads_to_store, create_store_from_h5ads

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("densify", [True, False])
def test_store_creation(
    raw_adatas_with_h5: tuple[ad.AnnData, Path],
    shuffle: bool,
    densify: bool,
):
    var_subset = [f"gene_{i}" for i in range(100)]

    (raw_adatas_with_h5[1] / "zarr_store").mkdir(parents=True, exist_ok=True)
    create_store_from_h5ads(
        [raw_adatas_with_h5[1] / f for f in raw_adatas_with_h5[1].iterdir() if str(f).endswith(".h5ad")],
        raw_adatas_with_h5[1] / "zarr_store",
        var_subset,
        chunk_size=10,
        shard_size=20,
        buffer_size=60,
        shuffle=shuffle,
        should_denseify=densify,
    )

    adata_orig = raw_adatas_with_h5[0]
    adata = ad.concat(
        [ad.read_zarr(zarr_path) for zarr_path in (raw_adatas_with_h5[1] / "zarr_store").iterdir()], join="outer"
    )
    assert adata.X.shape[0] == adata_orig.X.shape[0]
    assert adata.X.shape[1] == adata_orig.X.shape[1]
    assert np.array_equal(
        sorted(adata.var.index),
        sorted(adata_orig.var.index),
    )


@pytest.mark.parametrize("densify", [True, False])
@pytest.mark.parametrize("cache_h5ads", [True, False])
def test_store_extension(raw_adatas_with_h5: tuple[ad.AnnData, Path], densify: bool, cache_h5ads: bool):
    store_path = raw_adatas_with_h5[1] / "zarr_store"
    # create new store
    create_store_from_h5ads(
        [raw_adatas_with_h5[1] / f for f in raw_adatas_with_h5[1].iterdir() if str(f).endswith(".h5ad")],
        store_path,
        chunk_size=10,
        shard_size=20,
        buffer_size=60,
        shuffle=True,
        should_denseify=densify,
    )
    # add h5ads to existing store
    add_h5ads_to_store(
        [raw_adatas_with_h5[1] / f for f in raw_adatas_with_h5[1].iterdir() if str(f).endswith(".h5ad")],
        store_path,
        cache_h5ads=cache_h5ads,
        chunk_size=10,
        shard_size=20,
    )

    adata = ad.concat([ad.read_zarr(zarr_path) for zarr_path in (raw_adatas_with_h5[1] / "zarr_store").iterdir()])
    adata_orig = ad.concat([raw_adatas_with_h5[0], raw_adatas_with_h5[0]], join="outer")
    assert adata.X.shape[0] == adata_orig.X.shape[0]
