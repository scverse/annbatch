from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from annbatch import AnnDataField, Loader


def test_anndatafield_works_via_loader(tmp_path):
    n_obs, n_vars = 4, 3
    X = np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    counts = X + 1
    X_pca = np.arange(n_obs * 2, dtype=np.float32).reshape(n_obs, 2)

    store = zarr.open_group(tmp_path / "dummy.zarr", mode="w", zarr_format=3)
    store.create_array("X", data=X, chunks=(2, n_vars))
    layers_g = store.create_group("layers")
    layers_g.create_array("counts", data=counts, chunks=(2, n_vars))

    obs = pd.DataFrame(
        {
            "label_int": np.array([0, 1, 2, 3], dtype=np.int64),
            "label_str": ["a", "b", "c", "d"],
        },
        index=[str(i) for i in range(n_obs)],
    )
    adata = ad.AnnData(
        X=store["X"],
        obs=obs,
        layers={"counts": layers_g["counts"]},
        obsm={"X_pca": X_pca},
    )

    mapping = {"a": 1, "b": 2, "c": 3, "d": 4}
    ds = Loader(
        shuffle=False,
        chunk_size=2,
        preload_nchunks=1,
        batch_size=2,
        return_index=True,
        preload_to_gpu=False,
        to_torch=False,
    ).add_anndatas(
        [adata],
        adata_fields={
            "label_int": AnnDataField(attr="obs", key="label_int"),
            "label_int_str": AnnDataField(attr="obs", key="label_int", convert_fn=lambda s: s.astype(str)),
            "label_str_int": AnnDataField(attr="obs", key="label_str", convert_fn=lambda s: s.map(mapping).to_numpy()),
            "counts": AnnDataField(attr="layers", key="counts"),
            "X_pca": AnnDataField(attr="obsm", key="X_pca"),
        },
    )

    xs, idxs = [], []
    labels = {k: [] for k in ["label_int", "label_int_str", "label_str_int", "counts", "X_pca"]}
    for x, y, idx in ds:
        xs.append(x)
        idxs.append(idx)
        for k in labels:
            labels[k].append(y[k])

    idxs = np.concatenate(idxs).ravel()
    np.testing.assert_array_equal(np.vstack(xs), X)
    np.testing.assert_array_equal(np.concatenate(labels["label_int"]).ravel(), obs["label_int"].to_numpy())
    np.testing.assert_array_equal(
        np.concatenate(labels["label_int_str"]).ravel(), obs["label_int"].astype(str).to_numpy()
    )
    np.testing.assert_array_equal(
        np.concatenate(labels["label_str_int"]).ravel(), obs["label_str"].map(mapping).to_numpy()
    )
    np.testing.assert_array_equal(np.vstack(labels["counts"]), counts)
    np.testing.assert_array_equal(np.vstack(labels["X_pca"]), X_pca)
    np.testing.assert_array_equal(idxs, np.arange(n_obs))
