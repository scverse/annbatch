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
        How the in-memory data should be split into batches after it is read off disk.
        A list of splits, last one may be partial but not empty i.e. 1 <= len(last_split) <= batch_size.
        If not provided, the sampler's batch_size property will be used to automatically generate splits.

    Notes
    -----
    **Split index space**: ``splits`` index into the data in **chunk order** -- i.e. position ``j`` is
    the ``j``-th observation when the chunks are concatenated in the order they appear in ``chunks``.
    The sampler does not need to know how the loader lays out memory. Internally the loader fetches
    chunks grouped by dataset index for efficiency (so the physical buffer is ordered by dataset, not
    by chunk order), but it remaps each split back to chunk order before yielding, so this regrouping
    is invisible to the sampler.

    For example, given two datasets (dataset 0: rows 0-99, dataset 1: rows 100-199) and chunks
    ``[slice(100, 110), slice(0, 10), slice(110, 120)]``, chunk-order position 0 is row 100, position
    10 is row 0, position 20 is row 110 -- regardless of the dataset-grouped physical layout.

    .. warning::
        This is a **behaviour change in 0.2.0**. Before 0.2.0, ``splits`` had to index into the
        loader's dataset-grouped physical layout (i.e. observations ordered by dataset index, not by
        chunk order), so custom samplers had to account for that reordering themselves. They now index
        in chunk order and the loader remaps them internally. Custom samplers written for earlier
        versions that compensated for the dataset reordering must drop that compensation.

    """

    chunks: list[slice]
    splits: NotRequired[list[np.ndarray]]


class LoaderOutput[OutputInMemoryArray: OutputInMemoryArray_T](TypedDict):
    """The output of the loader, the "data matrix" with its obs, optional, var, optional, and index, also optional."""

    X: OutputInMemoryArray_T.__value__  # TODO: remove after sphinx 9 - myst compat
    obs: pd.DataFrame | None
    var: pd.DataFrame | None
    index: np.ndarray | None
