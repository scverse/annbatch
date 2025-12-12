from __future__ import annotations

from types import NoneType

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

type BackingArray_T = ad.abc.CSRDataset | ZarrArray
type InputInMemoryArray_T = CSRContainer | np.ndarray
type OutputInMemoryArray_T = sp.csr_matrix | np.ndarray | CupyCSRMatrix | CupyArray
