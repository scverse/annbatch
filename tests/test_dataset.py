from __future__ import annotations

from importlib.util import find_spec
from types import NoneType
from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import ChunkSampler, Loader, write_sharded
from annbatch.abc import Sampler

try:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix
except ImportError:
    CupyCSRMatrix = NoneType
    CupyArray = NoneType

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from annbatch.io import DatasetCollection


class Data(TypedDict):
    dataset: ad.abc.CSRDataset | zarr.Array
    obs: np.ndarray


class ListData:
    datasets: list[ad.abc.CSRDataset | zarr.Array]
    obs: list[np.ndarray]


def open_sparse(path: Path | zarr.Group, *, use_zarrs: bool = False, use_anndata: bool = False) -> Data | ad.AnnData:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline" if use_zarrs else old_pipeline}):
        if not isinstance(path, zarr.Group):
            path = zarr.open(path)
        data = {
            "dataset": ad.io.sparse_dataset(path["layers"]["sparse"]),
            "obs": ad.io.read_elem(path["obs"]),
        }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=data["obs"])
    return data


def open_dense(path: Path | zarr.Group, *, use_zarrs: bool = False, use_anndata: bool = False) -> Data | ad.AnnData:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline" if use_zarrs else old_pipeline}):
        if not isinstance(path, zarr.Group):
            path = zarr.open(path)
        data = {
            "dataset": path["X"],
            "obs": ad.io.read_elem(path["obs"]),
        }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=data["obs"])
    return data


def concat(datas: list[Data | ad.AnnData]) -> ListData | list[ad.AnnData]:
    return (
        {
            "datasets": [d["dataset"] for d in datas],
            "obs": [d["obs"] for d in datas],
        }
        if all(isinstance(d, dict) for d in datas)
        else datas
    )


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
@pytest.mark.parametrize(
    "gen_loader",
    [
        pytest.param(
            lambda collection,
            shuffle,
            use_zarrs,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            open_func=open_func,
            batch_size=batch_size,
            preload_to_gpu=preload_to_gpu: Loader(
                shuffle=shuffle,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                return_index=True,
                batch_size=batch_size,
                preload_to_gpu=preload_to_gpu,
                to_torch=False,
            ).use_collection(
                collection,
                **(
                    {"load_adata": lambda group: open_func(group, use_zarrs=use_zarrs, use_anndata=True)}
                    if open_func is not None
                    else {}
                ),
            ),
            id=f"chunk_size={chunk_size}-preload_nchunks={preload_nchunks}-open_func={open_func.__name__[5:] if open_func is not None else 'None'}-batch_size={batch_size}{'-cupy' if preload_to_gpu else ''}",  # type: ignore[attr-defined]
            marks=pytest.mark.skipif(
                find_spec("cupy") is None and preload_to_gpu,
                reason="need cupy installed",
            ),
        )
        for chunk_size, preload_nchunks, open_func, batch_size, preload_to_gpu in [
            elem
            for preload_to_gpu in [True, False]
            for open_func in [open_sparse, open_dense, None]
            for elem in [
                [
                    1,
                    5,
                    open_func,
                    1,
                    preload_to_gpu,
                ],  # singleton chunk size
                [
                    5,
                    1,
                    open_func,
                    1,
                    preload_to_gpu,
                ],  # singleton preload
                [
                    10,
                    5,
                    open_func,
                    5,
                    preload_to_gpu,
                ],  # batch size divides total in memory size evenly
                [
                    10,
                    5,
                    open_func,
                    50,
                    preload_to_gpu,
                ],  # batch size equal to in-memory size loading
            ]
        ]
    ],
)
def test_store_load_dataset(
    simple_collection: tuple[ad.AnnData, DatasetCollection], *, shuffle: bool, gen_loader, use_zarrs
):
    """
    This test verifies that the DaskDataset works correctly:
        1. The DaskDataset correctly loads data from the mock store
        2. Each sample has the expected feature dimension
        3. All samples from the dataset are processed
        4. If the dataset is not shuffled, it returns the correct data
    """
    loader: Loader = gen_loader(simple_collection[1], shuffle, use_zarrs)
    adata = simple_collection[0]
    is_dense = loader.dataset_type is zarr.Array
    n_elems = 0
    batches = []
    obs = []
    indices = []
    expected_data = adata.X if is_dense else adata.layers["sparse"].toarray()
    for batch in loader:
        x, label, index = batch["X"], batch["obs"], batch["index"]
        n_elems += x.shape[0]
        # Check feature dimension
        assert x.shape[1] == 100
        batches += [x.get() if isinstance(x, CupyCSRMatrix | CupyArray) else x]
        if label is not None:
            obs += [label]
        if index is not None:
            indices += [index]
    # check that we yield all samples from the dataset
    # np.array for sparse
    stacked = (np if is_dense else sp).vstack(batches)
    if not is_dense:
        stacked = stacked.toarray()
    if not shuffle:
        np.testing.assert_allclose(stacked, expected_data)
        if len(obs) > 0:
            expected_labels = adata.obs
            pd.testing.assert_frame_equal(
                pd.concat(obs),
                expected_labels,
            )
    else:
        if len(indices) > 0:
            indices = np.concatenate(indices).ravel()
            np.testing.assert_allclose(stacked, expected_data[indices])
        assert n_elems == adata.shape[0]


@pytest.mark.parametrize(
    "gen_loader",
    [
        (
            lambda path, chunk_size=chunk_size, preload_nchunks=preload_nchunks: Loader(
                shuffle=True,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
            )
        )
        for chunk_size, preload_nchunks in [[0, 10], [10, 0]]
    ],
)
def test_zarr_store_errors_lt_1(gen_loader, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    with pytest.raises(ValueError, match="must be greater than 1"):
        gen_loader(adata_with_zarr_path_same_var_space[1])


def test_bad_adata_X_type(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    data = open_dense(next(adata_with_zarr_path_same_var_space[1].glob("*.zarr")))
    data["dataset"] = data["dataset"][...]
    ds = Loader(shuffle=True, chunk_size=10, preload_nchunks=10, preload_to_gpu=False, to_torch=False)
    with pytest.raises(TypeError, match="Cannot add"):
        ds.add_dataset(**data)


def test_use_collection_twice(simple_collection: tuple[ad.AnnData, DatasetCollection]):
    ds = Loader()
    ds = ds.use_collection(simple_collection[1])
    with pytest.raises(RuntimeError, match="You should not add multiple collections"):
        ds.use_collection(simple_collection[1])


@pytest.mark.gpu
@pytest.mark.skipif(not find_spec("torch"), reason="need torch installed")
@pytest.mark.parametrize(
    "preload_to_gpu",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                find_spec("cupy") is None,
                reason="need cupy installed",
            ),
        ),
        False,
    ],
)
@pytest.mark.parametrize("open_func", [open_sparse, open_dense])
def test_to_torch(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
    open_func: Callable[[Path], Data],
    preload_to_gpu: bool,
):
    import torch

    # batch_size guaranteed to have leftovers to drop
    ds = Loader(
        shuffle=False,
        chunk_size=5,
        preload_nchunks=10,
        batch_size=25,
        preload_to_gpu=preload_to_gpu,
        return_index=True,
        to_torch=True,
    )
    ds.add_dataset(**open_func(next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))))
    assert isinstance(next(iter(ds))["X"], torch.Tensor)


@pytest.mark.parametrize("drop_last", [True, False], ids=["drop", "kept"])
def test_drop_last(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], drop_last: bool):
    # batch_size guaranteed to have last batch to drop
    chunk_size = 14
    preload_nchunks = 3
    batch_size = 21
    zarr_path = next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    adata = ad.read_zarr(zarr_path)
    ds = Loader(
        shuffle=False,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        preload_to_gpu=False,
        return_index=True,
        drop_last=drop_last,
        to_torch=False,
    )
    ds.add_dataset(**open_sparse(zarr_path))
    batches = []
    indices = []
    for batch in ds:
        batches += [batch["X"]]
        indices += [batch["index"]]
    total_obs = adata.shape[0]
    remainder = total_obs % batch_size
    assert remainder != 0, f"batch_size {batch_size} must not divide evenly into {total_obs} observations"
    for batch in batches[:-1]:
        assert batch.shape[0] == batch_size
    assert batches[-1].shape[0] == (batch_size if drop_last else remainder)
    X = sp.vstack(batches).toarray()
    assert X.shape[0] == (total_obs - remainder if drop_last else total_obs)
    X_expected = adata[np.concatenate(indices)].layers["sparse"].toarray()
    np.testing.assert_allclose(X, X_expected)


def test_bad_adata_X_hdf5(adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path]):
    with h5py.File(next(adata_with_h5_path_different_var_space[1].glob("*.h5ad"))) as f:
        data = ad.io.sparse_dataset(f["X"])
        ds = Loader(shuffle=True, chunk_size=10, preload_nchunks=10, preload_to_gpu=False, to_torch=False)
        with pytest.raises(TypeError, match="Cannot add"):
            ds.add_dataset(data)


def _custom_collate_fn(elems):
    import torch

    if isinstance(elems[0]["X"], torch.Tensor):
        x = torch.vstack([v["X"].to_dense() for v in elems])
    elif isinstance(elems[0]["X"], sp.csr_matrix):
        x = sp.vstack([v["X"] for v in elems]).toarray()
    else:
        x = np.vstack([v["X"] for v in elems])

    y = np.array([v["index"] for v in elems])

    return x, y


@pytest.mark.gpu
@pytest.mark.skipif(not find_spec("torch"), reason="Need torch installed.")
@pytest.mark.parametrize("open_func", [open_sparse, open_dense])
def test_torch_multiprocess_dataloading_zarr(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], open_func, use_zarrs: bool
):
    """
    Test that the ZarrDatasets can be used with PyTorch's DataLoader in a multiprocess context and that each element of
    the dataset gets yielded once.
    """
    from torch.utils.data import DataLoader

    ds = Loader(chunk_size=10, preload_nchunks=4, shuffle=True, return_index=True, preload_to_gpu=False)
    ds.add_datasets(
        **concat([open_func(p, use_zarrs=use_zarrs) for p in adata_with_zarr_path_same_var_space[1].glob("*.zarr")])
    )
    if open_func.__name__[5:] == "sparse":
        x_ref = adata_with_zarr_path_same_var_space[0].layers["sparse"].toarray()
    else:
        x_ref = adata_with_zarr_path_same_var_space[0].X

    dataloader = DataLoader(
        ds, batch_size=32, num_workers=4, collate_fn=_custom_collate_fn, multiprocessing_context="spawn"
    )
    x_list, idx_list = [], []
    for batch in dataloader:
        x, idxs = batch
        x_list.append(x)
        idx_list.append(idxs.ravel())

    x = np.vstack(x_list)
    idxs = np.concatenate(idx_list)

    assert np.array_equal(x[np.argsort(idxs)], x_ref)


@pytest.mark.skipif(
    find_spec("cupy") is not None, reason="Can't test for preload_to_gpu True ImportError with cupy installed"
)
def test_no_cupy():
    with pytest.raises(
        ImportError, match=r"Follow the directions at https://docs.cupy.dev/en/stable/install.html to install cupy."
    ):
        Loader(chunk_size=10, preload_nchunks=4, preload_to_gpu=True, to_torch=False)


@pytest.mark.skipif(
    find_spec("torch") is not None, reason="Can't test for to_torch True ImportError with torch installed"
)
def test_no_torch():
    with pytest.raises(ImportError, match=r"Try `pip install torch`."):
        Loader(chunk_size=10, preload_nchunks=4, to_torch=True, preload_to_gpu=False)


def get_default_dense() -> type:
    if find_spec("torch"):
        from torch import Tensor as expected_dense
    else:
        from numpy import ndarray as expected_dense
    return expected_dense


def get_default_sparse() -> type:
    if find_spec("cupy"):
        from cupyx.scipy.sparse import csr_matrix as expected_sparse
    else:
        from scipy.sparse import csr_matrix as expected_sparse

    return expected_sparse


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("expected_cls", "kwargs"),
    (
        pytest.param(get_default_dense(), {"preload_to_gpu": False}, id="torch"),
        pytest.param(get_default_sparse(), {"to_torch": False}, id="cupy"),
    ),
)
def test_default_data_structures(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], expected_cls: type, kwargs: dict
):
    # format is a smoke test for sparse
    ds = Loader(
        chunk_size=10, preload_nchunks=4, batch_size=20, shuffle=True, return_index=False, **kwargs
    ).add_dataset(
        **(open_sparse if issubclass(expected_cls, get_default_sparse()) else open_dense)(
            list(adata_with_zarr_path_same_var_space[1].iterdir())[0]
        )
    )
    assert isinstance(next(iter(ds))["X"], expected_cls)


def test_no_obs(simple_collection: tuple[ad.AnnData, DatasetCollection]):
    # No obs loaded is actually None
    ds = Loader(
        chunk_size=10,
        preload_nchunks=4,
        batch_size=20,
    ).use_collection(
        simple_collection[1],
        load_adata=lambda g: ad.AnnData(X=ad.io.sparse_dataset(g["layers"]["sparse"])),
    )
    assert next(iter(ds))["obs"] is None


@pytest.mark.gpu
@pytest.mark.skipif(not find_spec("cupy"), reason="need cupy installed")
@pytest.mark.parametrize(
    ("dtype_in", "expected"),
    [(np.int16, np.float32), (np.int32, np.float64), (np.float32, np.float32), (np.float64, np.float64)],
)
def test_preload_dtype(tmp_path: Path, dtype_in: np.dtype, expected: np.dtype):
    z = zarr.open(tmp_path / "foo.zarr")
    write_sharded(z, ad.AnnData(X=sp.random(100, 10, dtype=dtype_in, format="csr", rng=np.random.default_rng())))
    adata = ad.AnnData(X=ad.io.sparse_dataset(z["X"]))
    loader = Loader(preload_to_gpu=True, batch_size=10, chunk_size=10, preload_nchunks=2, to_torch=False).add_anndata(
        adata
    )
    assert next(iter(loader))["X"].dtype == expected


def test_add_dataset_validation_failure_preserves_state(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    """Test that failed validation in add_dataset doesn't modify internal state."""

    class FailOnSecondValidateSampler(Sampler):
        """A sampler that fails validation after the first call."""

        def __init__(self):
            self._validate_count = 0

        def validate(self, n_obs: int) -> None:
            self._validate_count += 1
            if self._validate_count > 1:
                raise ValueError("Validation failed on second add")

        @property
        def batch_size(self) -> int:
            return 10

        @property
        def worker_handle(self):
            return None

        def _sample(self, n_obs: int, worker_handle=None):
            yield from []

    paths = list(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    data1 = open_dense(paths[0])
    data2 = open_dense(paths[1])

    sampler = FailOnSecondValidateSampler()
    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to_torch=False)

    # First add succeeds
    loader.add_dataset(**data1)

    # Capture state before failed add
    n_datasets_before = len(loader._train_datasets)
    shapes_before = loader._shapes.copy()

    # Second add should fail validation BEFORE modifying state
    with pytest.raises(ValueError, match="Validation failed on second add"):
        loader.add_dataset(**data2)

    # State should be unchanged
    assert len(loader._train_datasets) == n_datasets_before
    assert loader._shapes == shapes_before


def test_given_batch_sampler_samples_subset_of_combined_datasets(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    """Test given batch sampler that samples only a specific range from combined datasets.

    Uses multiple zarr files from fixture, combines them, and samples a subset.
    """
    paths = list(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    datas = [open_dense(p) for p in paths]

    # Calculate expected n_obs before creating loader
    expected_n_obs = sum(d["dataset"].shape[0] for d in datas)
    start_idx, end_idx = expected_n_obs // 4, expected_n_obs // 2

    sampler = ChunkSampler(
        mask=slice(start_idx, end_idx),
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
    )

    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to_torch=False, return_index=True)
    loader.add_datasets(**concat(datas))

    # Collect all yielded indices
    all_indices = []
    for batch in loader:
        all_indices.append(batch["index"])

    stacked_indices = np.concatenate(all_indices)

    # Verify we got exactly the expected range
    assert set(stacked_indices) == set(range(start_idx, end_idx))
    assert len(stacked_indices) == end_idx - start_idx


@pytest.mark.parametrize("kwarg", [{"chunk_size": 10}, {"batch_size": 10}])
def test_cannot_provide_batch_sampler_with_sampler_args(kwarg):
    """Test that providing batch_sampler with sampler args raises in constructor."""
    chunk_sampler = ChunkSampler(mask=slice(0, 50), batch_size=5, chunk_size=10, preload_nchunks=2)
    with pytest.raises(ValueError, match="Cannot specify.*when providing a custom sampler"):
        Loader(batch_sampler=chunk_sampler, preload_to_gpu=False, to_torch=False, **kwarg)
