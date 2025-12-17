from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, TypedDict

import anndata as ad
import numpy as np
from scipy import sparse as sp

from annbatch.utils import CSRContainer

if find_spec("cupy") or TYPE_CHECKING:
    from cupy import ndarray as CupyArray
else:

    class CupyArray:  # noqa: D101
        pass


if find_spec("cupyx") or TYPE_CHECKING:
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix  # pragma: no cover
else:

    class CupyCSRMatrix:  # noqa: D101
        pass


import pandas as pd  # noqa: TC002
from zarr import Array as ZarrArray

if find_spec("torch") or TYPE_CHECKING:
    from torch import Tensor
else:

    class Tensor:  # noqa: D101
        pass


type BackingArray_T = ad.abc.CSRDataset | ZarrArray
type InputInMemoryArray_T = CSRContainer | np.ndarray
type OutputInMemoryArray_T = sp.csr_matrix | np.ndarray | CupyCSRMatrix | CupyArray | Tensor


class LoaderOutput(TypedDict):
    """The output of the loader, the "data matrix" with its labels, optional, and index, also optional."""

    data: OutputInMemoryArray_T.__value__
    labels: pd.DataFrame | None
    index: np.ndarray | None
