from __future__ import annotations

from types import NoneType
from typing import TypeVar

import anndata as ad
import numpy as np
import zarr
from scipy import sparse as sp

from arrayloaders.utils import CSRContainer

try:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix  # pragma: no cover
except ImportError:
    CupyCSRMatrix = NoneType
    CupyArray = NoneType

OutputInMemoryArray = sp.csr_matrix | np.ndarray | CupyCSRMatrix | CupyArray

OnDiskArray = TypeVar("OnDiskArray", ad.abc.CSRDataset, zarr.Array)


InputInMemoryArray = TypeVar("InputInMemoryArray", CSRContainer, np.ndarray)
