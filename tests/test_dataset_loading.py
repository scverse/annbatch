from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from arrayloaders.io import (
    DaskDataset,
    ZarrDenseDataset,
    ZarrSparseDataset,
    read_lazy_store,
)

if TYPE_CHECKING:
    from pathlib import Path


def open_sparse(path: Path):
    return ad.AnnData(
        X=ad.io.sparse_dataset(zarr.open(path)["layers"]["sparse"]),
        obs=ad.io.read_elem(zarr.open(path)["obs"]),
        layers={"data": ad.io.sparse_dataset(zarr.open(path)["layers"]["sparse"])},
    )


def open_dense(path: Path):
    return ad.AnnData(
        X=zarr.open(path)["X"],
        obs=ad.io.read_elem(zarr.open(path)["obs"]),
        layers={"data": zarr.open(path)["X"]},
    )


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
@pytest.mark.parametrize(
    "gen_loader",
    [
        pytest.param(
            lambda path, shuffle: DaskDataset(
                read_lazy_store(path, obs_columns=["label"]),
                label_column="label",
                n_chunks=4,
                shuffle=shuffle,
            ),
            id="dask",
        ),
        *(
            pytest.param(
                lambda path,
                shuffle,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                dataset_class=dataset_class,
                obs_keys=obs_keys,
                layer_keys=layer_keys: dataset_class(
                    shuffle=shuffle,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                    return_index=True,
                ).add_anndatas(
                    [
                        (
                            open_sparse
                            if issubclass(dataset_class, ZarrSparseDataset)
                            else open_dense
                        )(p)
                        for p in path.glob("*.zarr")
                    ],
                    layer_keys,
                    obs_keys,
                ),
                id=f"chunk_size={chunk_size}-preload_nchunks={preload_nchunks}-obs_keys={obs_keys}-dataset_class={dataset_class.__name__}-layer_keys={layer_keys}",
                # type: ignore[attr-defined]
            )
            for chunk_size, preload_nchunks, obs_keys, dataset_class, layer_keys in [
                elem
                for dataset_class in [ZarrDenseDataset, ZarrSparseDataset]  # type: ignore[list-item]
                for elem in [
                    [1, 5, None, dataset_class, None],  # singleton chunk size
                    [5, 1, None, dataset_class, None],  # singleton preload
                    [10, 5, "label", dataset_class, None],  # singleton label key
                    [
                        10,
                        5,
                        ["label", "label", "label"],
                        dataset_class,
                        None,
                    ],  # list label key
                    [10, 5, None, dataset_class, "data"],  # singleton data key
                    [
                        10,
                        5,
                        None,
                        dataset_class,
                        ["data", "data", "data"],
                    ],  # list data key
                ]
            ]
        ),
    ],
)
def test_store_load_dataset(mock_store: Path, *, shuffle: bool, gen_loader):
    """
    This test verifies that the DaskDataset works correctly:
        1. The DaskDataset correctly loads data from the mock store
        2. Each sample has the expected feature dimension
        3. All samples from the dataset are processed
        4. If the dataset is not shuffled, it returns the correct data
    """
    adata = read_lazy_store(mock_store, obs_columns=["label"])

    loader = gen_loader(mock_store, shuffle)
    is_dense = isinstance(loader, ZarrDenseDataset | DaskDataset)
    n_elems = 0
    batches = []
    labels = []
    expected_data = (
        adata.X.compute() if is_dense else adata.layers["sparse"].compute().toarray()
    )
    for batch in loader:
        if isinstance(loader, DaskDataset):
            x, label = batch
            indices = None
        else:
            x, label, indices = batch
        n_elems += 1
        # Check feature dimension
        assert x.shape[0 if is_dense else 1] == 100
        if not shuffle:
            batches += [x]
            if label is not None:
                labels += [label]
        if indices is not None:
            assert (
                (x if is_dense else x.toarray()) == expected_data[indices, ...]
            ).all()
    # check that we yield all samples from the dataset
    if not shuffle:
        # np.array for sparse
        stacked = (np if is_dense else sp).vstack(batches)
        if not is_dense:
            stacked = stacked.toarray()
        np.testing.assert_allclose(stacked, expected_data)
        if len(labels) > 0:
            expected_labels = adata.obs["label"]
            np.testing.assert_allclose(np.array(labels), expected_labels)
    else:
        assert n_elems == adata.shape[0]


@pytest.mark.parametrize(
    "gen_loader",
    [
        lambda path: DaskDataset(
            read_lazy_store(path, obs_columns=["label"])[:0],
            label_column="label",
            n_chunks=4,
            shuffle=True,
        ),
        lambda path: DaskDataset(
            read_lazy_store(path, obs_columns=["label"]),
            label_column="label",
            n_chunks=0,
            shuffle=True,
        ),
        lambda path: ZarrDenseDataset(
            shuffle=True,
            preload_nchunks=0,
        ),
        lambda path: ZarrDenseDataset(
            shuffle=True,
            preload_nchunks=4,
        ).add_anndatas(
            [open_dense(p) for p in path.glob("*.zarr")],
            None,
            obs_keys=[],
        ),
        *(
            (
                lambda path,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks: ZarrSparseDataset(
                    shuffle=True,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                )
            )
            for chunk_size, preload_nchunks in [[0, 10], [10, 0]]
        ),
    ],
)
def test_zarr_store_errors_lt_1(gen_loader, mock_store):
    with pytest.raises(ValueError, match="must be greater than 1"):
        gen_loader(mock_store)


@pytest.mark.parametrize(
    "layer_keys_less", [True, False], ids=["anndatas_shorter", "layer_keys_shorter"]
)
def test_layers_keys_anndata_mismatch(mock_store, layer_keys_less):
    with pytest.raises(ValueError, match="must match number of layer keys"):
        ZarrSparseDataset(
            shuffle=True,
            chunk_size=10,
            preload_nchunks=10,
        ).add_anndatas(
            [open_sparse(p) for p in mock_store.glob("*.zarr")]
            if layer_keys_less
            else [open_sparse(next(mock_store.glob("*.zarr")))],
            (["sparse"] * len(list(mock_store.glob("*.zarr"))))
            if not layer_keys_less
            else ["sparse"],
        )


@pytest.mark.parametrize(
    ["dataset_class", "init_anndata_args", "add_anndata_args", "error"],
    [
        pytest.param(
            ZarrSparseDataset,
            lambda path: {
                "adatas": [open_sparse(p) for p in path.glob("*.zarr")],
                "obs_keys": "label",
            },
            lambda path: {
                "adata": open_sparse(next(path.glob("*.zarr"))),
                "obs_key": None,
            },
            ValueError,
            id="add_unlabeled_to_labeled",
        ),
        pytest.param(
            ZarrSparseDataset,
            lambda path: {
                "adatas": [open_sparse(p) for p in path.glob("*.zarr")],
                "obs_keys": None,
            },
            lambda path: {
                "adata": open_sparse(next(path.glob("*.zarr"))),
                "obs_key": "label",
            },
            ValueError,
            id="add_labeled_to_unlabeled",
        ),
        pytest.param(
            ZarrSparseDataset,
            lambda path: {
                "adatas": [open_sparse(p) for p in path.glob("*.zarr")],
                "obs_keys": "label",
            },
            lambda path: {
                "adata": open_dense(next(path.glob("*.zarr"))),
                "obs_key": "label",
            },
            TypeError,
            id="add_dense_to_sparse",
        ),
        pytest.param(
            ZarrDenseDataset,
            lambda path: {
                "adatas": [open_dense(p) for p in path.glob("*.zarr")],
                "obs_keys": "label",
            },
            lambda path: {
                "adata": open_sparse(next(path.glob("*.zarr"))),
                "obs_key": "label",
            },
            TypeError,
            id="add_sparse_to_dense",
        ),
    ],
)
def test_add_data_bad_obs(
    mock_store, dataset_class, init_anndata_args, add_anndata_args, error
):
    ds = dataset_class(
        shuffle=True,
        chunk_size=10,
        preload_nchunks=10,
    ).add_anndatas(
        **init_anndata_args(mock_store),
    )
    with pytest.raises(error, match="Cannot add a dataset"):
        ds.add_anndata(**add_anndata_args(mock_store))


def test_bad_adata_X_type(mock_store):
    adata = open_dense(next(mock_store.glob("*.zarr")))
    adata.X = adata.X[...]
    ds = ZarrDenseDataset(
        shuffle=True,
        chunk_size=10,
        preload_nchunks=10,
    )
    with pytest.raises(TypeError, match="Cannot add a dataset"):
        ds.add_anndata(adata)


def _custom_collate_fn(elems):
    if isinstance(elems[0][0], sp.csr_matrix):
        x = sp.vstack([v[0] for v in elems]).toarray()
    else:
        x = np.vstack([v[0] for v in elems])

    if len(elems[0]) == 2:
        y = np.array([v[1] for v in elems])
    else:
        y = np.array([v[2] for v in elems])

    return x, y


@pytest.mark.parametrize("loader", [DaskDataset, ZarrDenseDataset, ZarrSparseDataset])
def test_torch_multiprocess_dataloading_zarr(mock_store, loader):
    """
    Test that the ZarrDatasets can be used with PyTorch's DataLoader in a multiprocess context and that each element of
    the dataset gets yielded once.
    """
    from torch.utils.data import DataLoader

    if issubclass(loader, ZarrSparseDataset):
        ds = ZarrSparseDataset(
            chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True
        )
        ds.add_anndatas([open_sparse(p) for p in mock_store.glob("*.zarr")])
        x_ref = (
            read_lazy_store(mock_store, obs_columns=["label"])
            .layers["sparse"]
            .compute()
            .toarray()
        )
    elif issubclass(loader, ZarrDenseDataset):
        ds = ZarrDenseDataset(
            chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True
        )
        ds.add_anndatas([open_dense(p) for p in mock_store.glob("*.zarr")])
        x_ref = read_lazy_store(mock_store, obs_columns=["label"]).X.compute()
    elif issubclass(loader, DaskDataset):
        adata = read_lazy_store(mock_store, obs_columns=["label"])
        adata.obs["order"] = np.arange(adata.shape[0])
        ds = DaskDataset(
            adata,
            label_column="order",
            n_chunks=4,
            shuffle=True,
        )
        x_ref = adata.X.compute()
    else:
        raise ValueError("Unknown loader type")

    dataloader = DataLoader(
        ds,
        batch_size=32,
        num_workers=4,
        collate_fn=_custom_collate_fn,
    )
    x_list, idx_list = [], []
    for batch in dataloader:
        x, idxs = batch
        x_list.append(x)
        idx_list.append(idxs)

    x = np.vstack(x_list)
    idxs = np.concatenate(idx_list)

    assert np.array_equal(x[np.argsort(idxs)], x_ref)
