from __future__ import annotations

import random
import subprocess
from importlib.util import find_spec
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

if find_spec("jax"):
    import jax

    jax.config.update("jax_enable_x64", True)


def load_x_obs_var(g: zarr.Group) -> ad.AnnData:
    """Load only ``X``/``obs``/``var`` from a group, without the transitional obsm/obsp/layers ``FutureWarning``.

    Tests that don't exercise ``obsm``/``obsp``/``layers`` pass this as ``load_adata`` to opt out of the
    (currently warning) default loader :func:`annbatch.utils.load_x_and_obs_and_var`. TODO(obsm): once the
    loader yields those elements, revisit the call sites of this helper to also cover them.
    """
    var = g["var"]
    return ad.AnnData(
        X=g["X"] if isinstance(g["X"], zarr.Array) else ad.io.sparse_dataset(g["X"]),
        obs=ad.io.read_elem(g["obs"]),
        var=pd.DataFrame(index=pd.Index(ad.io.read_elem(var[var.attrs.get("_index")]))),
    )


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
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(feature_dim)]),
            obs=pd.DataFrame(
                {"label": np.random.default_rng().integers(0, 5, size=n_cells_per_shard)},
                index=np.arange(n_cells_per_shard).astype(str),
            ),
            layers={
                "sparse": sp.random(
                    n_cells_per_shard, feature_dim, format="csr", rng=np.random.default_rng(), dtype="int32"
                )
            },
            obsm={"3d": np.random.default_rng().random((n_cells_per_shard, 2, 32, 32))},
        )
        adata_lst += [adata]
        f = zarr.open_group(tmp_path / f"chunk_{shard}.zarr", mode="w")
        write_sharded(
            f,
            adata,
            n_obs_per_chunk=10,
            shard_size=20,
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
    merge = params.get("merge", None)

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
            var=pd.DataFrame(
                index=var_idx,
                data={
                    f"only_{i}": pd.array(range(n), dtype="int64"),
                    f"partial_share_{i % 3}": pd.array(range(n), dtype="int64"),
                    "same": pd.array(range(n), dtype="int64"),
                },
            ),
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
        merge=merge,
    ), tmp_path


@pytest.fixture(scope="session")
def simple_collection(
    tmpdir_factory, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]
) -> tuple[DatasetCollection, ad.AnnData]:
    zarr_stores = sorted(f for f in adata_with_zarr_path_same_var_space[1].iterdir() if f.is_dir())
    output_path = Path(tmpdir_factory.mktemp("zarr_folder")) / "simple_fixture.zarr"
    collection = DatasetCollection(output_path).add_adatas(
        zarr_stores,
        n_obs_per_chunk=10,
        shard_size=20,
        dataset_size=60,
        shuffle_chunk_size=10,
    )
    return ad.concat([ad.io.read_elem(ds) for ds in collection], join="outer"), collection


@pytest.fixture(scope="session", params=[False, True], ids=["same-dtype", "mixed-dtype"])
def maybe_mixed_dtype_collection(
    request, tmpdir_factory, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]
) -> tuple[ad.AnnData, DatasetCollection, bool]:
    """Like ``simple_collection``, but optionally rewrites the first dataset's
    X (and sparse layer) with a different dtype to exercise the dtype-promotion
    code path in ``Loader._concatenate_outs``. Returns ``(adata, collection, is_mixed)``."""
    zarr_stores = sorted(f for f in adata_with_zarr_path_same_var_space[1].iterdir() if f.is_dir())
    output_path = Path(tmpdir_factory.mktemp("zarr_folder")) / "mixed_dtype_fixture.zarr"
    collection = DatasetCollection(output_path).add_adatas(
        zarr_stores,
        n_obs_per_chunk=10,
        shard_size=20,
        dataset_size=60,
        shuffle_chunk_size=10,
    )
    is_mixed = bool(request.param)
    if is_mixed:
        with ad.settings.override(auto_shard_zarr_v3=True, zarr_write_format=3):
            first = next(iter(collection))
            new_X = first["X"][...].astype("f8")
            del first["X"]
            ad.io.write_elem(first, "X", new_X)
            sparse_layer = ad.io.read_elem(first["layers"]["sparse"]).astype("int64")
            del first["layers"]["sparse"]
            ad.io.write_elem(first["layers"], "sparse", sparse_layer)

        datasets = list(collection)
        first_X_dtype = datasets[0]["X"].dtype
        first_sparse_dtype = datasets[0]["layers"]["sparse"]["data"].dtype
        assert any(ds["X"].dtype != first_X_dtype for ds in datasets[1:]), (
            "mixed-dtype fixture failed to produce differing X dtypes"
        )
        assert any(ds["layers"]["sparse"]["data"].dtype != first_sparse_dtype for ds in datasets[1:]), (
            "mixed-dtype fixture failed to produce differing sparse layer dtypes"
        )
    return ad.concat([ad.io.read_elem(ds) for ds in collection], join="outer"), collection, is_mixed


def pytest_itemcollected(item: pytest.Item) -> None:
    """Define behavior of pytest.mark.{gpu,array_api}."""
    is_marked = len(list(item.iter_markers(name="gpu"))) > 0
    if is_marked:
        try:
            has_gpu = (
                subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            )
        except FileNotFoundError:
            has_gpu = False
        if not has_gpu:
            item.add_marker(pytest.mark.skip())
