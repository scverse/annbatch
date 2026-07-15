from __future__ import annotations

import contextlib
import math
from importlib.util import find_spec
from types import NoneType
from typing import TYPE_CHECKING, Literal, TypedDict

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import Loader, write_sharded
from annbatch.abc import Sampler
from annbatch.samplers import SequentialSampler
from tests.conftest import load_x_obs_var

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

skip_if_no_cupy = pytest.mark.skipif(find_spec("cupy") is None, reason="Can't test for preload_to_gpu without cupy")
skip_if_no_torch = pytest.mark.skipif(find_spec("torch") is None, reason="Need torch installed.")
skip_if_no_numba = pytest.mark.skipif(find_spec("numba") is None, reason="Can't test for in-memory without numba")
skip_if_no_jax = pytest.mark.skipif(find_spec("jax") is None, reason="Need jax installed.")


class Data(TypedDict):
    dataset: ad.abc.CSRDataset | zarr.Array
    obs: np.ndarray


class ListData(TypedDict):
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
            "var": ad.io.read_elem(path["var"]),
        }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=data["obs"], var=data["var"])
    return data


def open_in_memory_sparse(
    path: Path | zarr.Group, *, use_zarrs: bool = False, use_anndata: bool = False
) -> Data | ad.AnnData:
    if not isinstance(path, zarr.Group):
        path = zarr.open(path)
    data = {
        "dataset": ad.io.read_elem(path["layers"]["sparse"]),
        "obs": ad.io.read_elem(path["obs"]),
        "var": ad.io.read_elem(path["var"]),
    }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=data["obs"], var=data["var"])
    return data


def open_in_memory_dense(
    path: Path | zarr.Group, *, use_zarrs: bool = False, use_anndata: bool = False
) -> Data | ad.AnnData:
    if not isinstance(path, zarr.Group):
        path = zarr.open(path)
    data = {
        "dataset": ad.io.read_elem(path["X"]),
        "obs": ad.io.read_elem(path["obs"]),
        "var": ad.io.read_elem(path["var"]),
    }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=data["obs"], var=data["var"])
    return data


def open_dense(path: Path | zarr.Group, *, use_zarrs: bool = False, use_anndata: bool = False) -> Data | ad.AnnData:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline" if use_zarrs else old_pipeline}):
        if not isinstance(path, zarr.Group):
            path = zarr.open(path)
        data = {
            "dataset": path["X"],
            "obs": ad.io.read_elem(path["obs"]),
            "var": ad.io.read_elem(path["var"]),
        }
    if use_anndata:
        return ad.AnnData(X=data["dataset"], obs=data["obs"], var=data["var"])
    return data


def open_3d(path: Path | zarr.Group, *, use_zarrs: bool = False) -> Data:
    old_pipeline = zarr.config.get("codec_pipeline.path")

    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline" if use_zarrs else old_pipeline}):
        if not isinstance(path, zarr.Group):
            path = zarr.open(path)
        data = {
            "dataset": path["obsm"]["3d"],
            "obs": ad.io.read_elem(path["obs"]),
            "var": ad.io.read_elem(path["var"]),
        }
    return data


def concat(datas: list[Data | ad.AnnData]) -> ListData | list[ad.AnnData]:
    return (
        {
            "datasets": [d["dataset"] for d in datas],
            "obs": [d["obs"] for d in datas],
            "var": [d["var"] for d in datas],
        }
        if all(isinstance(d, dict) for d in datas)
        else datas
    )


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffled", "unshuffled"])
@pytest.mark.parametrize(
    "gen_loader",
    [
        pytest.param(
            lambda collection, shuffle, use_zarrs, chunk_size=chunk_size, preload_nchunks=preload_nchunks, open_func=open_func, batch_size=batch_size, preload_to_gpu=preload_to_gpu: (
                Loader(
                    shuffle=shuffle,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                    return_index=True,
                    batch_size=batch_size,
                    preload_to_gpu=preload_to_gpu,
                    to=None,
                ).use_collection(
                    collection,
                    # X/obs/var only; obsm/layers loading is a future concern (see `load_x_obs_var`)
                    load_adata=(
                        (lambda group: open_func(group, use_zarrs=use_zarrs, use_anndata=True))
                        if open_func is not None
                        else load_x_obs_var
                    ),
                )
            ),
            id=f"chunk_size={chunk_size}-preload_nchunks={preload_nchunks}-open_func={open_func.__name__[5:] if open_func is not None else 'None'}-batch_size={batch_size}{'-cupy' if preload_to_gpu else ''}",  # type: ignore[attr-defined]
            marks=([skip_if_no_cupy, pytest.mark.gpu] if preload_to_gpu else [])
            + ([skip_if_no_numba] if open_func is open_in_memory_sparse else []),
        )
        for chunk_size, preload_nchunks, open_func, batch_size, preload_to_gpu in [
            elem
            for preload_to_gpu in [True, False]
            for open_func in [
                open_sparse,
                open_dense,
                open_in_memory_dense,
                open_in_memory_sparse,
                None,
            ]
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
    maybe_mixed_dtype_collection: tuple[ad.AnnData, DatasetCollection, bool],
    *,
    shuffle: bool,
    gen_loader,
    use_zarrs,
):
    """
    This test verifies that the DaskDataset works correctly:
        1. The DaskDataset correctly loads data from the mock store
        2. Each sample has the expected feature dimension
        3. All samples from the dataset are processed
        4. If the dataset is not shuffled, it returns the correct data
    """
    adata, collection, is_mixed = maybe_mixed_dtype_collection
    if is_mixed:
        with pytest.warns(UserWarning, match="Adding dataset with dtype"):
            loader: Loader = gen_loader(collection, shuffle, use_zarrs)
    else:
        loader: Loader = gen_loader(collection, shuffle, use_zarrs)
    if use_zarrs and loader.dataset_type in {np.ndarray, sp.csr_matrix, sp.csr_array}:
        pytest.skip("No need to run zarrs with in-memory")
    is_dense = loader.dataset_type in {zarr.Array, np.ndarray}
    n_elems = 0
    batches = []
    obs = []
    indices = []
    var_dfs = []
    expected_data = adata.X if is_dense else adata.layers["sparse"].toarray()
    for batch in loader:
        x, label, var, index = batch["X"], batch["obs"], batch["var"], batch["index"]
        n_elems += x.shape[0]
        # Check feature dimension
        assert x.shape[1] == 100
        batches += [x.get() if isinstance(x, CupyCSRMatrix | CupyArray) else x]
        if label is not None:
            obs += [label]
        if var is not None:
            var_dfs += [var]
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
    # Check var is consistently yielded and matches expected
    if len(var_dfs) > 0:
        expected_var = adata.var
        # var should be the same for every batch (feature dimension, not obs)
        for var_df in var_dfs:
            pd.testing.assert_frame_equal(var_df, expected_var)


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


def test_use_collection_twice(simple_collection: tuple[ad.AnnData, DatasetCollection]):
    ds = Loader(to=None)
    ds = ds.use_collection(simple_collection[1], load_adata=load_x_obs_var)
    with pytest.raises(RuntimeError, match="You should not add multiple collections"):
        ds.use_collection(simple_collection[1], load_adata=load_x_obs_var)


@contextlib.contextmanager
def expect_transitional_warning(*, present: bool):
    transitional_msg = "Only `X`, `obs`, and `var` are kept"
    with pytest.warns(FutureWarning, match=transitional_msg) if present else contextlib.nullcontext():
        yield


@pytest.mark.parametrize("custom_loader", [False, True], ids=["default", "custom-loader"])
def test_use_collection_transitional_warning(
    simple_collection: tuple[ad.AnnData, DatasetCollection], *, custom_loader: bool
):
    """`use_collection` defaults to `load_x_and_obs_and_var` (warns on obsm/layers); a custom loader opts out."""
    _, collection = simple_collection
    loader = Loader(chunk_size=10, preload_nchunks=4, to=None, preload_to_gpu=False)
    # the collection has obsm/layers on disk, so the default loader warns while the X/obs/var-only loader does not
    load_adata = {"load_adata": lambda g: open_dense(g, use_anndata=True)} if custom_loader else {}
    with expect_transitional_warning(present=not custom_loader):
        loader.use_collection(collection, **load_adata)


@pytest.mark.parametrize("has_extras", [True, False], ids=["with-extras", "without-extras"])
@pytest.mark.parametrize("method", ["add_adata", "add_adatas"])
def test_add_adata_warns_with_extras(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], *, method: str, has_extras: bool
):
    """`add_adata`/`add_adatas` warn iff the in-memory AnnData carries obsm/layers that get dropped for now."""
    adata = adata_with_zarr_path_same_var_space[0]  # has obsm/3d and layers/sparse
    if not has_extras:
        adata = ad.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
    loader = Loader(chunk_size=10, preload_nchunks=4, to=None, preload_to_gpu=False)

    with expect_transitional_warning(present=has_extras):
        getattr(loader, method)(adata if if method == "add_adata" else [adata])

    assert next(iter(loader))["X"].shape[1] == 100


def test_add_adatas_warns_exactly_once_about_each_extra():
    """`add_adatas` warns once *per unique* dropped element: distinct extras each warn, duplicates are deduped."""
    n_obs, n_var = 40, 100
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_var)])
    x = np.random.default_rng().random((n_obs, n_var)).astype("f4")
    adata_obsm = ad.AnnData(X=x.copy(), var=var, obsm={"pca": np.zeros((n_obs, 5), dtype="f4")})
    adata_obsp = ad.AnnData(X=x.copy(), var=var, obsp={"conn": sp.eye(n_obs, format="csr", dtype="f4")})
    adata_obsm_again = ad.AnnData(X=x.copy(), var=var, obsm={"pca": np.zeros((n_obs, 5), dtype="f4")})

    loader = Loader(chunk_size=10, preload_nchunks=4, to=None, preload_to_gpu=False)
    with pytest.warns(FutureWarning) as record:
        loader.add_adatas([adata_obsm, adata_obsp, adata_obsm_again])

    msgs = [str(w.message) for w in record if issubclass(w.category, FutureWarning)]
    # `obsm/pca` is carried by two adatas but warned about once; `obsp/conn` warns once -> two warnings total
    assert len(msgs) == 2
    assert sum("obsm/pca" in m for m in msgs) == 1
    assert sum("obsp/conn" in m for m in msgs) == 1


@pytest.mark.gpu
@pytest.mark.parametrize(
    "to", [pytest.param("torch", marks=skip_if_no_torch), pytest.param("jax", marks=skip_if_no_jax), None]
)
@pytest.mark.parametrize(
    "preload_to_gpu",
    [
        pytest.param(
            True,
            marks=skip_if_no_cupy,
        ),
        False,
    ],
    ids=["preload_to_gpu", "dont_preload_to_gpu"],
)
@pytest.mark.parametrize("open_func", [open_sparse, open_dense])
def test_to(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
    open_func: Callable[[Path], Data],
    preload_to_gpu: bool,
    to: Literal["torch", "jax"],
):
    # batch_size guaranteed to have leftovers to drop
    ds = Loader(
        shuffle=False,
        chunk_size=5,
        preload_nchunks=10,
        batch_size=25,
        preload_to_gpu=preload_to_gpu,
        return_index=True,
        to=to,
    )
    ds.add_dataset(**open_func(next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))))
    if to == "torch":
        import torch

        assert isinstance(next(iter(ds))["X"], torch.Tensor)
    elif to == "jax":
        import jax

        assert isinstance(next(iter(ds))["X"], jax.Array if open_func is open_dense else jax.experimental.sparse.CSR)
    elif preload_to_gpu:
        import cupy
        import cupyx

        assert isinstance(
            next(iter(ds))["X"], cupy.ndarray if open_func is open_dense else cupyx.scipy.sparse.csr_matrix
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({}, "implicit use of torch", marks=skip_if_no_torch, id="implicit"),
        pytest.param({"to_torch": True}, "will be replaced by the explicit", marks=skip_if_no_torch, id="true"),
        pytest.param({"to_torch": False}, "To explicitly disable torch conversion", id="false"),
    ],
)
def test_to_default_warns(kwargs: dict, match: str):
    with pytest.warns(DeprecationWarning, match=match):
        Loader(chunk_size=10, preload_nchunks=4, preload_to_gpu=False, **kwargs)


@skip_if_no_torch
def test_legacy_implicit(adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
    import torch

    with pytest.warns(DeprecationWarning, match="`to_torch`'s implicit"):
        ds = Loader(chunk_size=10, preload_nchunks=4)

    ds.add_datasets(**concat([open_sparse(p) for p in adata_with_zarr_path_same_var_space[1].glob("*.zarr")]))

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
        to=None,
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


@pytest.mark.parametrize("drop_last", [True, False], ids=["drop", "kept"])
def test_len(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
    drop_last: bool,
):
    zarr_path = next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    data = open_sparse(zarr_path)
    n_obs = data["dataset"].shape[0]
    batch_size = 32

    loader = Loader(
        shuffle=False,
        batch_size=batch_size,
        preload_to_gpu=False,
        to=None,
        drop_last=drop_last,
    )
    loader.add_dataset(**data)

    expected_len = n_obs // batch_size if drop_last else math.ceil(n_obs / batch_size)
    assert len(loader) == expected_len
    # Also verify len matches the actual number of yielded batches
    actual_batches = sum(1 for _ in loader)
    assert len(loader) == actual_batches


def test_bad_adata_X_hdf5(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path],
):
    with h5py.File(next(adata_with_h5_path_different_var_space[1].glob("*.h5ad"))) as f:
        data = ad.io.sparse_dataset(f["X"])
        ds = Loader(
            shuffle=True,
            chunk_size=10,
            preload_nchunks=10,
            preload_to_gpu=False,
            to=None,
        )
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
@skip_if_no_torch
@pytest.mark.parametrize("open_func", [open_sparse, open_dense])
def test_torch_multiprocess_dataloading_zarr(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
    open_func,
    use_zarrs: bool,
):
    """
    Test that Loader can be used with PyTorch's DataLoader in a multiprocess context and that each element of
    the dataset gets yielded once.
    """
    from torch.utils.data import DataLoader

    ds = Loader(
        chunk_size=10,
        preload_nchunks=4,
        shuffle=True,
        return_index=True,
        preload_to_gpu=False,
        to="torch",
    )
    ds.add_datasets(
        **concat([open_func(p, use_zarrs=use_zarrs) for p in adata_with_zarr_path_same_var_space[1].glob("*.zarr")])
    )
    if open_func.__name__[5:] == "sparse":
        x_ref = adata_with_zarr_path_same_var_space[0].layers["sparse"].toarray()
    else:
        x_ref = adata_with_zarr_path_same_var_space[0].X

    dataloader = DataLoader(
        ds,
        batch_size=32,
        num_workers=4,
        collate_fn=_custom_collate_fn,
        multiprocessing_context="spawn",
    )
    x_list, idx_list = [], []
    for batch in dataloader:
        x, idxs = batch
        x_list.append(x)
        idx_list.append(idxs.ravel())

    x = np.vstack(x_list)
    idxs = np.concatenate(idx_list)

    assert np.array_equal(x[np.argsort(idxs)], x_ref)


@pytest.mark.parametrize(
    "preload_to_gpu",
    [False, pytest.param(True, marks=[pytest.mark.gpu, skip_if_no_cupy])],
    ids=["no_cupy", "cupy"],
)
@pytest.mark.parametrize(
    "to",
    [None, pytest.param("torch", marks=[skip_if_no_torch]), pytest.param("jax", marks=[skip_if_no_jax])],
    ids=["no_to", "torch", "jax"],
)
def test_3d(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
    use_zarrs: bool,
    preload_to_gpu: bool,
    to: Literal["jax", "torch"] | None,
):
    ds = Loader(
        chunk_size=10,
        preload_nchunks=4,
        shuffle=True,
        return_index=True,
        preload_to_gpu=preload_to_gpu,
        to=to,
    )
    ds.add_datasets(
        **concat([open_3d(p, use_zarrs=use_zarrs) for p in adata_with_zarr_path_same_var_space[1].glob("*.zarr")])
    )
    x_ref = adata_with_zarr_path_same_var_space[0].obsm["3d"]

    x_list, idx_list = [], []
    for batch in ds:
        x, idxs = batch["X"], batch["index"]
        if preload_to_gpu and to is None:
            import cupy as cp

            assert isinstance(x, cp.ndarray)
            x = x.get()
        elif to == "torch":
            import torch

            assert isinstance(x, torch.Tensor)
            x = x.cpu().numpy()
        elif to == "jax":
            import jax

            assert isinstance(x, jax.Array)
            x = np.array(x)
        x_list.append(x)
        idx_list.append(idxs.ravel())
    x = np.vstack(x_list)
    idxs = np.concatenate(idx_list)

    assert np.array_equal(x[np.argsort(idxs)], x_ref)


@pytest.mark.parametrize(
    "kwargs",
    [
        *(
            pytest.param(
                {"to": lib, "preload_to_gpu": False},
                marks=pytest.mark.skipif(
                    find_spec(lib) is not None,
                    reason=f"Can't test for to='{lib}' True ImportError with {lib} installed",
                ),
                id=lib,
            )
            for lib in ["jax", "torch"]
        ),
        pytest.param(
            {"preload_to_gpu": True, "to": None},
            marks=pytest.mark.skipif(
                find_spec("cupy") is not None,
                reason="Can't test for preload_to_gpu True ImportError with cupy installed",
                id="cupy",
            ),
        ),
    ],
)
def test_missing_gpu_lib(kwargs: dict):
    with pytest.raises(ImportError, match=rf"Could not find {kwargs['to'] if kwargs['to'] is not None else 'cupy'}"):
        Loader(chunk_size=10, preload_nchunks=4, **kwargs)


@pytest.mark.skipif(
    find_spec("numba") is not None,
    reason="Can't test for sparse in-memory ImportError with numba installed",
)
def test_no_numba_in_memory_sparse(monkeypatch: pytest.MonkeyPatch):
    loader = Loader(chunk_size=10, preload_nchunks=4, to=None, preload_to_gpu=False)
    sparse_data = sp.csr_matrix(np.eye(10, dtype=np.float32))
    with pytest.raises(
        ImportError,
        match=r"numba must be installed for in-memory sparse data",
    ):
        loader.add_dataset(sparse_data)


def test_no_obs_no_var(simple_collection: tuple[ad.AnnData, DatasetCollection]):
    # No obs loaded is actually None
    ds = Loader(chunk_size=10, preload_nchunks=4, batch_size=20, to=None).use_collection(
        simple_collection[1],
        load_adata=lambda g: ad.AnnData(X=ad.io.sparse_dataset(g["layers"]["sparse"])),
    )
    assert next(iter(ds))["obs"] is None


def test_mismatched_var_raises_error(tmp_path: Path, subtests):
    """Test that adding anndatas/datasets with different var dataframes raises an error."""
    n_obs, n_vars = 100, 50

    # Create first anndata with var index gene_0, gene_1, ...
    z1 = zarr.open(tmp_path / "adata1.zarr")
    adata1 = ad.AnnData(
        X=sp.random(n_obs, n_vars, format="csr", rng=np.random.default_rng()),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
    )
    write_sharded(z1, adata1)
    adata1_on_disk = ad.AnnData(
        X=ad.io.sparse_dataset(z1["X"]),
        var=adata1.var,
    )

    # Create second anndata with different var index: different_gene_0, different_gene_1, ...
    z2 = zarr.open(tmp_path / "adata2.zarr")
    adata2 = ad.AnnData(
        X=sp.random(n_obs, n_vars, format="csr", rng=np.random.default_rng()),
        var=pd.DataFrame(index=[f"different_gene_{i}" for i in range(n_vars)]),
    )
    write_sharded(z2, adata2)
    adata2_on_disk = ad.AnnData(
        X=ad.io.sparse_dataset(z2["X"]),
        var=adata2.var,
    )

    with subtests.test(msg="add_adata"):
        loader = Loader(chunk_size=10, preload_nchunks=4, batch_size=20, to=None)
        loader.add_adata(adata1_on_disk)
        with pytest.raises(ValueError, match="All datasets must have identical var DataFrames"):
            loader.add_adata(adata2_on_disk)

    with subtests.test(msg="add_adatas"):
        loader = Loader(chunk_size=10, preload_nchunks=4, batch_size=20, to=None)
        with pytest.raises(ValueError, match="All datasets must have identical var DataFrames"):
            loader.add_adatas([adata1_on_disk, adata2_on_disk])

    with subtests.test(msg="add_dataset"):
        loader = Loader(chunk_size=10, preload_nchunks=4, batch_size=20, to=None)
        loader.add_dataset(adata1_on_disk.X, var=adata1_on_disk.var)
        with pytest.raises(ValueError, match="All datasets must have identical var DataFrames"):
            loader.add_dataset(adata2_on_disk.X, var=adata2_on_disk.var)

    with subtests.test(msg="add_datasets"):
        loader = Loader(chunk_size=10, preload_nchunks=4, batch_size=20, to=None)
        with pytest.raises(ValueError, match="All datasets must have identical var DataFrames"):
            loader.add_datasets(
                [adata1_on_disk.X, adata2_on_disk.X],
                var=[adata1_on_disk.var, adata2_on_disk.var],
            )


@pytest.mark.gpu
@skip_if_no_cupy
@pytest.mark.parametrize(
    ("dtype_in", "expected"),
    [
        (np.int16, np.float32),
        (np.int32, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
def test_preload_dtype(tmp_path: Path, dtype_in: np.dtype, expected: np.dtype):
    z = zarr.open(tmp_path / "foo.zarr")
    write_sharded(
        z,
        ad.AnnData(X=sp.random(100, 10, dtype=dtype_in, format="csr", rng=np.random.default_rng())),
    )
    adata = ad.AnnData(X=ad.io.sparse_dataset(z["X"]))
    loader = Loader(
        preload_to_gpu=True,
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        to=None,
    ).add_adata(adata)
    assert next(iter(loader))["X"].dtype == expected


def test_add_dataset_validation_failure_preserves_state(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    """Test that failed validation in add_dataset doesn't modify internal state."""

    class FailOnSecondValidateSampler(Sampler):
        """A sampler that fails validation after the first call."""

        def __init__(self):
            self._validate_count = 0

        def n_batches(self, n_obs: int) -> int:
            return math.ceil(n_obs / self.batch_size)

        def validate(self, n_obs: int) -> None:
            self._validate_count += 1
            if self._validate_count > 1:
                raise ValueError("Validation failed on second add")

        @property
        def batch_size(self) -> int:
            return 10

        @property
        def shuffle(self) -> bool:
            return False

        @property
        def worker_handle(self):
            return None

        def _sample(self, n_obs: int, worker_handle=None):
            yield from []

    paths = list(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    data1 = open_dense(paths[0])
    data2 = open_dense(paths[1])

    sampler = FailOnSecondValidateSampler()
    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None)

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

    sampler = SequentialSampler(
        mask=slice(start_idx, end_idx),
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
    )

    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None, return_index=True)
    loader.add_datasets(**concat(datas))

    # Collect all yielded indices
    all_indices = []
    for batch in loader:
        all_indices.append(batch["index"])

    stacked_indices = np.concatenate(all_indices)

    # Verify we got exactly the expected range
    assert set(stacked_indices) == set(range(start_idx, end_idx))
    assert len(stacked_indices) == end_idx - start_idx


@pytest.mark.parametrize("kwarg", [{"chunk_size": 10}, {"batch_size": 10}, {"rng": np.random.default_rng(0)}])
def test_cannot_provide_batch_sampler_with_sampler_args(kwarg):
    """Test that providing batch_sampler with sampler args raises in constructor."""
    chunk_sampler = SequentialSampler(mask=slice(0, 50), batch_size=5, chunk_size=10, preload_nchunks=2)
    with pytest.raises(ValueError, match="Cannot specify.*when providing a custom sampler"):
        Loader(batch_sampler=chunk_sampler, preload_to_gpu=False, to=None, **kwarg)


def test_rng(simple_collection: tuple[ad.AnnData, DatasetCollection]):
    ds1 = Loader(
        chunk_size=10,
        preload_nchunks=4,
        batch_size=20,
        shuffle=True,
        rng=np.random.default_rng(0),
        to=None,
    )
    ds2 = Loader(
        chunk_size=10,
        preload_nchunks=4,
        batch_size=20,
        shuffle=True,
        rng=np.random.default_rng(0),
        to=None,
    )
    ds1.use_collection(simple_collection[1], load_adata=load_x_obs_var)
    ds2.use_collection(simple_collection[1], load_adata=load_x_obs_var)
    for batch1, batch2 in zip(ds1, ds2, strict=True):
        np.testing.assert_equal(batch1["X"], batch2["X"])


class _FixedRequestSampler(SequentialSampler):
    """Emits one preselected LoadRequest with chunk-order splits, reusing SequentialSampler's plumbing."""

    def __init__(self, requests: list[slice], splits: list[np.ndarray], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._requests, self._splits = requests, splits

    def _sample(self, n_obs: int):
        yield {"requests": self._requests, "splits": self._splits}


def test_splits_are_chunk_order_across_datasets(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    """Splits index in chunk order; the loader undoes its dataset-grouped buffer layout.

    The request lists a chunk from dataset 1 *before* a chunk from dataset 0, so the loader's
    in-memory buffer (grouped by dataset) is the reverse of chunk order. Each split selects one
    chunk's positions, so each batch must be exactly that chunk's rows. Before the chunk-order
    remap, the dataset-grouped buffer leaked the wrong rows into each batch -- this test would
    have failed.
    """
    paths = sorted(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    data0, data1 = open_dense(paths[0]), open_dense(paths[1])
    n0 = data0["dataset"].shape[0]  # dataset 0 -> global [0, n0) ; dataset 1 -> global [n0, ...)

    slice_from_data1 = slice(n0, n0 + 10)
    slice_from_data0 = slice(0, 10)
    split_from_data1 = np.arange(0, 10)
    split_from_data0 = np.arange(10, 20)
    sampler = _FixedRequestSampler(
        batch_size=10,
        preload_nchunks=2,
        chunk_size=10,
        # split 0 from data1, split 1 from data0
        requests=[slice_from_data1, slice_from_data0],
        splits=[split_from_data1, split_from_data0],
    )
    loader = Loader(batch_sampler=sampler, return_index=True, preload_to_gpu=False, to=None)
    loader.add_dataset(**data0)
    loader.add_dataset(**data1)

    batches = list(loader)
    # split 0 -> chunk A -> dataset 1's first 10 rows ; split 1 -> chunk B -> dataset 0's first 10 rows
    assert np.array_equal(batches[0]["index"], np.arange(slice_from_data1.start, slice_from_data1.stop))
    assert np.array_equal(batches[1]["index"], np.arange(slice_from_data0.start, slice_from_data0.stop))
    # data follows the index: each batch is exactly the requested chunk's rows read off disk
    np.testing.assert_array_equal(np.asarray(batches[0]["X"]), np.asarray(data1["dataset"][0:10]))
    np.testing.assert_array_equal(np.asarray(batches[1]["X"]), np.asarray(data0["dataset"][0:10]))


def test_chunks_deprecation_warning(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    paths = sorted(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
    data0 = open_dense(paths[0])

    class ChunksSampler(SequentialSampler):
        def _sample(self, n_obs: int):
            yield {"chunks": [slice(0, 10)], "splits": [np.arange(10)]}

    loader = Loader(
        batch_sampler=ChunksSampler(batch_size=10, preload_nchunks=2, chunk_size=10),
        return_index=True,
        preload_to_gpu=False,
        to=None,
    )
    loader.add_dataset(**data0)

    with pytest.warns(DeprecationWarning, match=r"The `chunks` key"):
        batches = list(loader)

    assert len(batches) == 1
