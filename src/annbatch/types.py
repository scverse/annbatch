from __future__ import annotations

from types import NoneType
from typing import TypedDict

import anndata as ad
import numpy as np
from scipy import sparse as sp

from annbatch.utils import CSRContainer

try:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix  # pragma: no cover
except ImportError:
    CupyCSRMatrix = NoneType
    CupyArray = NoneType
import pandas as pd  # noqa: TC002
from zarr import Array as ZarrArray

try:
    from torch import Tensor
except ImportError:
    Tensor = NoneType

type BackingArray_T = ad.abc.CSRDataset | ZarrArray
type InputInMemoryArray_T = CSRContainer | np.ndarray
type OutputInMemoryArray_T = sp.csr_matrix | np.ndarray | CupyCSRMatrix | CupyArray | Tensor


class LoaderOutput(TypedDict):
    """The output of the loader, the "data matrix" with its labels, optional, and index, also optional."""

    data: OutputInMemoryArray_T.__value__
    labels: pd.DataFrame | None
    index: np.ndarray | None
