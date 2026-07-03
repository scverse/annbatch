from __future__ import annotations

from typing import NotRequired, TypedDict

import anndata as ad
import numpy as np
import pandas as pd  # noqa: TC002
from scipy import sparse as sp
from zarr import Array as ZarrArray

from .compat import CupyArray, CupyCSRMatrix, JaxArray, JAXCSRMatrix, Tensor
from .utils import CSRContainer

type BackingArray_T = ad.abc.CSRDataset | ZarrArray | sp.csr_array | sp.csr_matrix | np.ndarray
type InputInMemoryArray_T = CSRContainer | np.ndarray
type OutputInMemoryArray_T = sp.csr_matrix | np.ndarray | CupyCSRMatrix | CupyArray | Tensor | JaxArray | JAXCSRMatrix


class LoadRequest(TypedDict):
    """Load request from sampler.

    This is the request format Loader will expect from the sampler.
    Not satisfying the constrains documented here may result in unexpected behavior.

    Attributes
    ----------
    requests
        Either an array of indices to load that must be the same size as the sum of the split sizes,
        or a list of slices with a range of chunk_size except the last one which may be smaller but not empty.

        .. versionchanged:: 0.2.0
            Renamed from ``chunks`` to ``requests``.
    splits
        How the in-memory data should be split into batches after it is read off disk and after all the chunks are loaded and concatenated in the order requested by `chunks`.
        A list of splits, last one may be partial but not empty i.e. 1 <= len(last_split) <= batch_size.
        If not provided, the sampler's batch_size property will be used to automatically generate splits.

    Notes
    -----

    .. warning::
        This is a **behaviour change in 0.2.0**. Before 0.2.0, ``splits`` had to index into the
        loader's dataset-grouped physical layout (i.e. observations ordered by dataset index, not by
        chunk order), so custom samplers had to account for that reordering themselves. They now index
        in chunk order and the loader remaps them internally. Custom samplers written for earlier
        versions that compensated for the dataset reordering must drop that compensation.

    """

    requests: list[slice] | np.ndarray
    splits: NotRequired[list[np.ndarray]]


class LoaderOutput[OutputInMemoryArray: OutputInMemoryArray_T](TypedDict):
    """The output of the loader, the "data matrix" with its obs, optional, var, optional, and index, also optional."""

    X: OutputInMemoryArray_T.__value__  # TODO: remove after sphinx 9 - myst compat
    obs: pd.DataFrame | None
    var: pd.DataFrame | None
    index: np.ndarray | None
