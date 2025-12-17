from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from anndata import AnnData


@dataclass(frozen=True)
class AnnDataField:
    """
    Minimal, extensible field accessor for AnnData-like objects.

    This is intentionally small: for now only `attr="obs"` is supported.
    The design mirrors Cellarium's `AnnDataField` and can be extended to `X`,
    `layers`, `obsm`, etc.
    """

    attr: Literal["obs"]
    key: str
    convert_fn: Callable[[Any], Any] | None = None

    def __call__(self, adata: AnnData) -> np.ndarray:
        if self.attr != "obs":
            raise NotImplementedError(f"AnnDataField(attr={self.attr!r}) is not supported yet.")

        value = adata.obs[self.key]
        if self.convert_fn is not None:
            value = self.convert_fn(value)
        return np.asarray(value)
