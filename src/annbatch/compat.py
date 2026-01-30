from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING or find_spec("torch"):
    from torch import Tensor
    from torch.utils.data import IterableDataset
else:
    IterableDataset = type("IterableDataset", (), {"__module__": "torch.utils.data"})
    Tensor = type("Tensor", (), {"__module__": "torch"})

if TYPE_CHECKING or find_spec("cupy"):
    from cupy import ndarray as CupyArray
else:
    CupyArray = type("ndarray", (), {"__module__": "cupy"})

if TYPE_CHECKING or find_spec("cupyx"):
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix
else:
    CupyCSRMatrix = type("csr_matrix", (), {"__module__": "cupyx.scipy.sparse"})
