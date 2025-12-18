from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from anndata import AnnData


@dataclass(frozen=True)
class AnnDataField:
    """
    Minimal, extensible field accessor for AnnData-like objects.

    Mirrors Cellarium's `AnnDataField`: select an AnnData attribute via `attr`,
    optionally index into it via `key`, and optionally apply `convert_fn`.
    """

    attr: str
    key: list[str] | str | None = None
    convert_fn: Callable[[Any], Any] | None = None

    def __call__(self, adata: AnnData) -> np.ndarray:
        """Extract this field from an AnnData-like object.

        `attr` is looked up on `adata` (e.g. ``"X"``, ``"obs"``, ``"layers"``, ``"obsm"``).
        If `key` is provided, the attribute is indexed with `key` (e.g. a column in
        ``obs`` or an entry in ``layers``). If ``convert_fn`` is provided, it is applied
        to the selected value; otherwise the selected value is converted via
        :func:`numpy.asarray`.

        Parameters
        ----------
        adata
            AnnData-like object to read from.

        Returns
        -------
        numpy.ndarray
            Array representation of the selected field.
        """
        value = attrgetter(self.attr)(adata)
        if self.key is not None:
            value = value[self.key]

        if self.convert_fn is not None:
            value = self.convert_fn(value)
        return np.asarray(value)
