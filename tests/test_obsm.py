from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import Loader

if TYPE_CHECKING:
    from pathlib import Path

skip_if_no_numba = pytest.mark.skipif(
    find_spec("numba") is None, reason="Can't test for in-memory sparse without numba"
)

N_VAR = 12
N_EMB = 5
SIZES = (50, 30, 40)


def _build_adatas(
    rng: np.random.Generator, *, sparse: bool
) -> tuple[list[ad.AnnData], np.ndarray, np.ndarray, np.ndarray]:
    """Build imaginary adatas with X, obsm['X_emb'], obs (batch+label) and var.

    Returns the adatas plus the globally-concatenated X, X_emb and label arrays so
    that batches can be checked against the original rows via ``return_index``.
    """
    adatas, x_all, emb_all, label_all = [], [], [], []
    for k, n in enumerate(SIZES):
        x = rng.random((n, N_VAR)).astype("f4")
        emb = rng.random((n, N_EMB)).astype("f4")
        labels = rng.integers(0, 4, size=n)
        adatas.append(
            ad.AnnData(
                X=sp.csr_matrix(x) if sparse else x,
                obs=pd.DataFrame(
                    {"batch": pd.Categorical([f"donor{k}"] * n), "label": labels},
                    index=[f"donor{k}_cell_{i}" for i in range(n)],
                ),
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VAR)]),
                obsm={"X_emb": emb},
            )
        )
        x_all.append(x)
        emb_all.append(emb)
        label_all.append(labels)
    return adatas, np.vstack(x_all), np.vstack(emb_all), np.concatenate(label_all)


def _to_dense_np(x) -> np.ndarray:
    return x.toarray() if sp.issparse(x) else np.asarray(x)


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
@pytest.mark.parametrize(
    "sparse",
    [pytest.param(False, id="dense"), pytest.param(True, id="sparse", marks=skip_if_no_numba)],
)
def test_obsm_in_memory_alignment(*, sparse: bool, shuffle: bool):
    """obsm['X_emb'] is yielded and stays row-aligned with X, obs and index."""
    rng = np.random.default_rng(0)
    adatas, x_all, emb_all, label_all = _build_adatas(rng, sparse=sparse)

    loader = Loader(
        batch_size=16,
        chunk_size=4,
        preload_nchunks=8,
        shuffle=shuffle,
        return_index=True,
        to=None,
        preload_to_gpu=False,
        rng=np.random.default_rng(1),
    ).add_adatas(adatas, obsm_keys=["X_emb"])

    seen_idx = []
    for batch in loader:
        assert "obsm" in batch
        emb = np.asarray(batch["obsm"]["X_emb"])
        idx = batch["index"]
        x = _to_dense_np(batch["X"])
        # shapes line up across X, obsm and obs
        assert emb.shape == (len(idx), N_EMB)
        assert x.shape[0] == emb.shape[0] == batch["obs"].shape[0]
        # every row matches the original global rows at these indices
        np.testing.assert_allclose(emb, emb_all[idx])
        np.testing.assert_allclose(x, x_all[idx], atol=1e-6)
        np.testing.assert_array_equal(batch["obs"]["label"].to_numpy(), label_all[idx])
        seen_idx.append(idx)

    all_idx = np.concatenate(seen_idx)
    assert sorted(all_idx.tolist()) == list(range(sum(SIZES)))


def test_obsm_multiple_keys():
    """Several obsm keys can be requested at once and are all yielded."""
    rng = np.random.default_rng(3)
    adatas = []
    for n in SIZES:
        adatas.append(
            ad.AnnData(
                X=rng.random((n, N_VAR)).astype("f4"),
                obs=pd.DataFrame({"label": rng.integers(0, 3, n)}),
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VAR)]),
                obsm={"X_emb": rng.random((n, N_EMB)).astype("f4"), "X_pca": rng.random((n, 3)).astype("f4")},
            )
        )
    loader = Loader(
        batch_size=16, chunk_size=4, preload_nchunks=8, shuffle=True, to=None, preload_to_gpu=False
    ).add_adatas(adatas, obsm_keys=["X_emb", "X_pca"])
    batch = next(iter(loader))
    assert set(batch["obsm"]) == {"X_emb", "X_pca"}
    assert batch["obsm"]["X_emb"].shape[1] == N_EMB
    assert batch["obsm"]["X_pca"].shape[1] == 3


def test_obsm_none_by_default():
    """Without ``obsm_keys`` the batch carries an explicit ``obsm=None``."""
    rng = np.random.default_rng(4)
    adata = ad.AnnData(
        X=rng.random((20, N_VAR)).astype("f4"),
        obs=pd.DataFrame({"label": rng.integers(0, 3, 20)}),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VAR)]),
        obsm={"X_emb": rng.random((20, N_EMB)).astype("f4")},
    )
    loader = Loader(batch_size=8, chunk_size=2, preload_nchunks=4, to=None, preload_to_gpu=False).add_adata(adata)
    batch = next(iter(loader))
    assert batch["obsm"] is None


def test_obsm_on_disk_alignment(tmp_path: Path):
    """obsm backed by an on-disk zarr.Array is fetched and stays aligned."""
    rng = np.random.default_rng(5)
    adatas_mem, x_all, emb_all, _ = _build_adatas(rng, sparse=False)
    for k, a in enumerate(adatas_mem):
        a.write_zarr(tmp_path / f"a{k}.zarr")

    def load(k: int) -> ad.AnnData:
        g = zarr.open_group(tmp_path / f"a{k}.zarr", mode="r")
        return ad.AnnData(
            X=g["X"],
            obs=ad.io.read_elem(g["obs"]),
            var=pd.DataFrame(index=pd.Index(ad.io.read_elem(g["var"][g["var"].attrs["_index"]]))),
            obsm={"X_emb": g["obsm"]["X_emb"]},
        )

    adatas = [load(k) for k in range(len(SIZES))]
    assert isinstance(adatas[0].obsm["X_emb"], zarr.Array)

    loader = Loader(
        batch_size=6,
        chunk_size=3,
        preload_nchunks=4,
        shuffle=True,
        return_index=True,
        to=None,
        preload_to_gpu=False,
        rng=np.random.default_rng(6),
    ).add_adatas(adatas, obsm_keys=["X_emb"])

    seen = 0
    for batch in loader:
        idx = batch["index"]
        np.testing.assert_allclose(np.asarray(batch["obsm"]["X_emb"]), emb_all[idx])
        np.testing.assert_allclose(_to_dense_np(batch["X"]), x_all[idx], atol=1e-6)
        seen += len(idx)
    assert seen == sum(SIZES)


def test_obsm_missing_key_raises():
    rng = np.random.default_rng(7)
    adata = ad.AnnData(
        X=rng.random((10, N_VAR)).astype("f4"),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VAR)]),
        obsm={"X_emb": rng.random((10, N_EMB)).astype("f4")},
    )
    with pytest.raises(KeyError, match="not found in adata.obsm"):
        Loader(batch_size=2, chunk_size=2, preload_nchunks=2, to=None, preload_to_gpu=False).add_adata(
            adata, obsm_keys=["does_not_exist"]
        )


def test_obsm_inconsistent_presence_raises():
    rng = np.random.default_rng(8)

    def mk(*, with_obsm: bool) -> ad.AnnData:
        return ad.AnnData(
            X=rng.random((10, N_VAR)).astype("f4"),
            obs=pd.DataFrame({"label": np.zeros(10)}),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VAR)]),
            obsm={"X_emb": rng.random((10, N_EMB)).astype("f4")} if with_obsm else {},
        )

    loader = Loader(batch_size=2, chunk_size=2, preload_nchunks=2, to=None, preload_to_gpu=False)
    loader.add_adata(mk(with_obsm=False))
    with pytest.raises(ValueError, match="without obsm"):
        loader.add_adata(mk(with_obsm=True), obsm_keys=["X_emb"])


def test_obsm_mismatched_feature_shape_raises():
    rng = np.random.default_rng(9)
    a1 = ad.AnnData(
        X=rng.random((10, N_VAR)).astype("f4"),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VAR)]),
        obsm={"X_emb": rng.random((10, N_EMB)).astype("f4")},
    )
    a2 = ad.AnnData(
        X=rng.random((10, N_VAR)).astype("f4"),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VAR)]),
        obsm={"X_emb": rng.random((10, N_EMB + 3)).astype("f4")},
    )
    with pytest.raises(ValueError, match="feature shape"):
        Loader(batch_size=2, chunk_size=2, preload_nchunks=2, to=None, preload_to_gpu=False).add_adatas(
            [a1, a2], obsm_keys=["X_emb"]
        )


def test_obsm_wrong_nobs_raises():
    rng = np.random.default_rng(10)
    adata = ad.AnnData(
        X=rng.random((10, N_VAR)).astype("f4"),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VAR)]),
    )
    with pytest.raises(ValueError, match="rows but the dataset has"):
        Loader(batch_size=2, chunk_size=2, preload_nchunks=2, to=None, preload_to_gpu=False).add_dataset(
            adata.X, obsm={"X_emb": rng.random((7, N_EMB)).astype("f4")}
        )
