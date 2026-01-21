from __future__ import annotations

from typing import TypedDict

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
    """

    chunks: list[slice]
    splits: list[np.ndarray]


class LoaderOutput[OutputInMemoryArray: OutputInMemoryArray_T](TypedDict):
    """The output of the loader, the "data matrix" with its obs, optional, and index, also optional."""

    X: OutputInMemoryArray_T.__value__  # TODO: remove after sphinx 9 - myst compat
    obs: pd.DataFrame | None
    index: np.ndarray | None
