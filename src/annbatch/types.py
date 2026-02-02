from __future__ import annotations

from typing import NotRequired, TypedDict

import anndata as ad
import numpy as np
import pandas as pd  # noqa: TC002
from scipy import sparse as sp
from zarr import Array as ZarrArray

from .compat import CupyArray, CupyCSRMatrix, Tensor
from .utils import CSRContainer

type BackingArray_T = ad.abc.CSRDataset | ZarrArray
type InputInMemoryArray_T = CSRContainer | np.ndarray
type OutputInMemoryArray_T = sp.csr_matrix | np.ndarray | CupyCSRMatrix | CupyArray | Tensor


class LoadRequest(TypedDict):
    """Load request from sampler.

    This is the request format Loader will expect from the sampler.
    Not satisfying the constrains documented here may result in unexpected behavior.

    Attributes
    ----------
    chunks
        Chunks to load - a list of slices with a range of chunk_size except the last one which may be smaller but not empty.
    splits
        How the in-memory data should be split into batches after it is read off disk and concatenated in-memory.
        A list of splits, last one may be partial but not empty i.e. 1 <= len(last_split) <= batch_size.
        If not provided, the sampler's batch_size property will be used to automatically generate splits.

    Notes
    -----
    **In-memory data ordering**: When chunks span multiple datasets, the loader groups and fetches
    chunks by dataset index for efficiency. The resulting in-memory data is ordered by dataset index,
    not by the original order of chunks in the `chunks` list. Within each dataset, chunks maintain
    their relative order from the original list.

    For example, given two datasets (dataset 0: rows 0-99, dataset 1: rows 100-199) and chunks
    ``[slice(100, 110), slice(0, 10), slice(110, 120)]``, the in-memory data will be ordered as:
    ``[rows 0-10 from ds0, rows 100-110 from ds1, rows 110-120 from ds1]`` i.e., sorted by dataset index.

    The `splits` indices must account for this ordering. For a single dataset, the in-memory order
    naturally matches the chunk order since there's only one dataset to fetch from.

    """

    chunks: list[slice]
    splits: NotRequired[list[np.ndarray]]


class LoaderOutput[OutputInMemoryArray: OutputInMemoryArray_T](TypedDict):
    """The output of the loader, the "data matrix" with its obs, optional, and index, also optional."""

    X: OutputInMemoryArray_T.__value__  # TODO: remove after sphinx 9 - myst compat
    obs: pd.DataFrame | None
    index: np.ndarray | None
