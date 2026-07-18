from __future__ import annotations

import inspect
import itertools
import re
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Concatenate, Literal, Protocol, overload

import anndata as ad
import numpy as np
import pandas as pd
import scipy as sp
import zarr

from .compat import CupyArray, CupyCSRMatrix, JaxArray, JAXCSRMatrix, Tensor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from annbatch.loader import Loader
    from annbatch.types import OutputInMemoryArray_T


def validate_sampler[**Param, RetType](
    method: Callable[Concatenate[Loader, Param], RetType],
) -> Callable[Concatenate[Loader, Param], RetType]:
    """Decorator that validates n_obs before modifying state.

    Expects the first positional argument to be either:
    - A single object with a `.shape` attribute
    - A list of objects with `.shape` attributes

    The total n_obs is computed as sum of shape[0] values for a list of objects or the shape[0] value for a single object.
    """
    sig = inspect.signature(method)
    if len(sig.parameters) < 2:
        raise ValueError("validate_sampler decorator expects at least two positional arguments after 'self'")
    first_param_name = list(sig.parameters.keys())[1]

    @wraps(method)
    def wrapper(self: Loader, *args: Param.args, **kwargs: Param.kwargs) -> RetType:
        if len(args) > 0:
            first_arg = args[0]
        else:
            first_arg = kwargs[first_param_name]

        n_obs = sum(item.shape[0] for item in first_arg) if isinstance(first_arg, list) else first_arg.shape[0]
        self.batch_sampler.validate(n_obs)
        return method(self, *args, **kwargs)

    return wrapper


def split_given_size(a: np.ndarray, size: int) -> list[np.ndarray]:
    """Wrapper around `np.split` to split up an array into `size` chunks"""
    return np.split(a, np.arange(size, len(a), size))


def interval_indexer_from_slices(slices: Iterable[slice]) -> pd.IntervalIndex:
    """Generate an IntervalIndex from a list of slices representing start-stop bounds."""
    len_bounds = list(itertools.accumulate((sum(s.stop - s.start for s in v) for v in slices), initial=0))
    starts = len_bounds[:-1]
    ends = len_bounds[1:]
    return pd.IntervalIndex.from_tuples(
        list(zip(starts, ends, strict=True)),
        closed="left",
    )


@dataclass
class CSRContainer:
    """A low-cost container for moving around the buffers of a CSR object"""

    elems: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int]
    dtype: np.dtype


# TODO: make this part of the public zarr or zarrs-python API.
# We can do chunk coalescing in zarrs based on integer arrays, so I think
# there would make sense with ezclump or similar.
# Another "solution" would be for zarrs to support integer indexing properly, if that pipeline works,
# or make this an "experimental setting" and to use integer indexing for the zarr-python pipeline.
# See: https://github.com/zarr-developers/zarr-python/issues/3175 for why this is better than simpler alternatives.
class MultiBasicIndexer(zarr.core.indexing.Indexer):
    """Custom indexer to enable joint fetching of disparate slices"""

    def __init__(self, indexers: list[zarr.core.indexing.Indexer]):
        self.shape = (sum(i.shape[0] for i in indexers), *indexers[0].shape[1:])
        self.drop_axes = indexers[0].drop_axes  # maybe?
        self.indexers = indexers

    def __iter__(self):
        total = 0
        for i in self.indexers:
            for c in i:
                out_selection = c[2]
                gap = out_selection[0].stop - out_selection[0].start
                yield type(c)(c[0], c[1], (slice(total, total + gap), *out_selection[1:]), c[3])
                total += gap


def _spawn_worker_rng(rng: np.random.Generator, worker_id: int) -> np.random.Generator:
    """Create a worker-specific RNG using the sequence-of-integers seeding pattern.

    Uses NumPy's recommended approach for multi-process RNG. See:
    https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds
    """
    root_seed = rng.integers(np.iinfo(np.int64).max)
    return np.random.default_rng([worker_id, root_seed])


def check_lt_1(vals: list[int], obs: list[str]) -> None:
    """Raise a ValueError if any of the values are less than one.

    The format of the error is "{obs[i]} must be greater than 1, got {values[i]}"
    and is raised based on the first found less than one value.

    Parameters
    ----------
        vals
            The values to check < 1
        obs
            The label for the value in the error if the value is less than one.

    Raises
    ------
        ValueError: _description_
    """
    if any(is_lt_1 := [v < 1 for v in vals]):
        label, value = next(
            (label, value)
            for label, value, check in zip(
                obs,
                vals,
                is_lt_1,
                strict=True,
            )
            if check
        )
        raise ValueError(f"{label} must be greater than 1, got {value}")


class SupportsShape(Protocol):  # noqa: D101
    @property
    def shape(self) -> tuple[int, int] | list[int]: ...  # noqa: D102


def check_var_shapes(objs: list[SupportsShape]) -> None:
    """Small utility function to check that all objects have the same shape along the second axis"""
    if not all(objs[0].shape[1] == d.shape[1] for d in objs):
        raise ValueError("TODO: All datasets must have same shape along the var axis.")


@overload
def convert(input: OutputInMemoryArray_T, preload_to_gpu: bool, to: Literal["torch"]) -> Tensor: ...
@overload
def convert(input: OutputInMemoryArray_T, preload_to_gpu: bool, to: Literal["jax"]) -> JaxArray | JAXCSRMatrix: ...
def convert(
    input: OutputInMemoryArray_T, preload_to_gpu: bool, to: Literal["torch", "jax"]
) -> Tensor | JaxArray | JAXCSRMatrix:
    """Convert the input array to an output array based on the user's to argument"""
    if to == "torch":
        return _to_torch(input, preload_to_gpu)
    return _to_jax(input)


def _to_jax(input: OutputInMemoryArray_T) -> JaxArray | JAXCSRMatrix:
    """Convert to jax"""
    import jax.numpy as jnp
    from jax.experimental.sparse import CSR

    if isinstance(input, CupyArray | np.ndarray):
        return jnp.from_dlpack(input)
    if isinstance(input, CupyCSRMatrix | sp.sparse.csr_matrix):
        return CSR(
            (
                jnp.from_dlpack(input.data),
                jnp.from_dlpack(input.indices),
                jnp.from_dlpack(input.indptr),
            ),
            shape=input.shape,
        )
    raise TypeError(f"Cannot convert {type(input)} to jax.Array")


def _to_torch(input: OutputInMemoryArray_T, preload_to_gpu: bool) -> Tensor:
    """Convert to torch"""
    import torch

    preload_to_gpu_warning_msg = (
        "preload_to_gpu will only apply to cupy arrays for in-memory handling in the next minor release."
        "You will be responsible for cpu-gpu transfers if cupy is not used. We recommend `.cuda(non_blocking=True, **kwargs)"
    )

    if isinstance(input, sp.sparse.csr_matrix):
        # TODO: better way to toggle this off for "production" but on for tests?
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state", UserWarning)
                # https://github.com/pytorch/pytorch/issues/178309
                # Without this, under `enable=True` above, there would be the above error.
                if input.nnz == 0:
                    indptr = torch.from_numpy(input.indptr)
                    tensor = torch.sparse_csr_tensor(
                        indptr,
                        torch.tensor([], dtype=indptr.dtype),
                        torch.tensor([]),
                        size=input.shape,
                        dtype=torch.from_numpy(input.data).dtype,  # TODO: better way to do this?
                    )
                else:
                    tensor = torch.sparse_csr_tensor(
                        torch.from_numpy(input.indptr),
                        torch.from_numpy(input.indices),
                        torch.from_numpy(input.data),
                        size=input.shape,
                    )
            if preload_to_gpu:
                warnings.warn(preload_to_gpu_warning_msg, FutureWarning, stacklevel=2)
                return tensor.cuda(non_blocking=True)
            return tensor
    if isinstance(input, np.ndarray):
        tensor = torch.from_numpy(input)
        if preload_to_gpu:
            warnings.warn(preload_to_gpu_warning_msg, FutureWarning, stacklevel=2)
            return tensor.cuda(non_blocking=True)
        return tensor
    if isinstance(input, CupyArray):
        return torch.from_dlpack(input)
    if isinstance(input, CupyCSRMatrix):
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state", UserWarning)
                return torch.sparse_csr_tensor(
                    torch.from_dlpack(input.indptr),
                    torch.from_dlpack(input.indices),
                    torch.from_dlpack(input.data),
                    size=input.shape,
                )
    raise TypeError(f"Cannot convert {type(input)} to torch.Tensor")


def warn_ignored_obs_aligned(ignored: list[str], *, stacklevel: int) -> None:
    """Warn that observation-aligned ``obsm``/``obsp``/``layers`` elements are dropped for now.

    ``ignored`` is a list of ``"<elem>/<key>"`` names that are being discarded. The warning is
    emitted only once per unique message (mirroring anndata's ``warn_once``) so repeated calls -
    e.g. over a whole collection via :meth:`Dataset.add_adatas` - do not spam identical warnings.
    """
    if not ignored:
        return
    msg = (
        "Only `X`, `obs`, and `var` are kept for now; the following observation-aligned elements are "
        f"ignored: {sorted(ignored)}. A future release will additionally load and yield them if they are "
        'uniformly present across `AnnData` objects i.e., every object has `obsm["pca"]` if even one has it. '
        "To silence this warning, drop these elements beforehand (e.g. via a custom `load_adata`)."
    )
    warnings.warn(msg, FutureWarning, stacklevel=stacklevel + 1)
    # Show this exact message only once (see anndata.utils.warn_once); `"once"` is unreliable in REPLs/notebooks.
    warnings.filterwarnings("ignore", message=re.escape(msg), category=FutureWarning)


def load_x_and_obs_and_var(g: zarr.Group) -> ad.AnnData:
    """Load X as a sparse array or dense zarr array and obs from a group.

    .. note::
        For now only ``X``, ``obs``, and ``var`` are loaded; any observation-aligned ``obsm``, ``obsp``,
        and ``layers`` elements found on disk are ignored and a :class:`FutureWarning` is emitted. A future
        release will additionally load and yield them.
    """
    warn_ignored_obs_aligned(
        [f"{elem}/{key}" for elem in ("obsm", "obsp", "layers") if elem in g for key in g[elem]],
        stacklevel=2,
    )
    var = g["var"]
    return ad.AnnData(
        X=g["X"] if isinstance(g["X"], zarr.Array) else ad.io.sparse_dataset(g["X"]),
        obs=ad.io.read_elem(g["obs"]),
        var=pd.DataFrame(index=pd.Index(ad.io.read_elem(var[var.attrs.get("_index")]))),
    )
