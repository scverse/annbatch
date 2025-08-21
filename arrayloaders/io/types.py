from __future__ import annotations

from typing import TypeVar

import anndata as ad
import numpy as np
import zarr
from scipy import sparse as sp

OnDiskArray = TypeVar("OnDiskArray", ad.abc.CSRDataset, zarr.Array)
InMemoryArray = TypeVar("InMemoryArray", sp.csr_matrix, np.ndarray)
BackingArray = TypeVar(
    "BackingArray", ad.abc.CSRDataset, zarr.Array, sp.csr_matrix, np.ndarray
)
