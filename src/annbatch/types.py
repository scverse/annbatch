from __future__ import annotations

from types import NoneType
from typing import TYPE_CHECKING, TypedDict

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
from zarr import Array as ZarrArray

if TYPE_CHECKING:
    import pandas as pd

type BackingArray_T = ad.abc.CSRDataset | ZarrArray
type InputInMemoryArray_T = CSRContainer | np.ndarray
type OutputInMemoryArray_T = sp.csr_matrix | np.ndarray | CupyCSRMatrix | CupyArray


class LoaderOuput(TypedDict):
    """The output of the loader, the "data matrix" with its labels, optional, and index, also optional"""

    data: OutputInMemoryArray_T
    labels: pd.DataFrame | None
    index: np.ndarray | None
