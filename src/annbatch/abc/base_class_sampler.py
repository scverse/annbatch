"""Abstract interface for class-coherent samplers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from annbatch.abc.sampler import Sampler

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class BaseClassSampler(Sampler):
    """Abstract interface for class-coherent samplers.

    A class sampler draws class-coherent batches and can be *bound onto*: it emits an integer
    class *code* per batch (:meth:`batch_codes`) together with the label :attr:`vocab`
    those codes index into, and a :class:`~annbatch.samplers.BoundClassSampler` replays that
    schedule onto another obs table -- matching **by label**, since the two tables are factorized
    independently. Any sampler implementing this interface can serve as such an inner.
    """

    @property
    @abstractmethod
    def vocab(self) -> pd.Index:
        """The label vocabulary the emitted codes index into (the code -> label decoder)."""

    @abstractmethod
    def emittable_codes(self) -> np.ndarray:
        """The codes (into :attr:`vocab`) this sampler can draw."""

    @abstractmethod
    def batch_codes(self) -> np.ndarray:
        """The class code (into :attr:`vocab`) of each batch a full pass yields (length :meth:`n_batches`).

        Consumes the sampler's rng, so a :class:`~annbatch.samplers.BoundClassSampler` observes
        the same class schedule a standalone pass would.
        """
