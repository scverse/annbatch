"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class Sampler(ABC):
    """Base sampler class.

    Samplers control how data is batched and loaded from the underlying datasets.
    """

    def sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Sample load requests given the total number of observations.

        Parameters
        ----------
        n_obs
            The total number of observations available.

        Yields
        ------
        LoadRequest
            Load requests for batching data.
        """
        self.validate(n_obs)
        yield from self._sample(n_obs)

    @abstractmethod
    def validate(self, n_obs: int) -> None:
        """Validate the sampler configuration against the given n_obs.

        This method is called at the start of each `sample()` call.
        Override this method to add custom validation for sampler parameters.

        Parameters
        ----------
        n_obs
            The total number of observations in the loader.

        Raises
        ------
        ValueError
            If the sampler configuration is invalid for the given n_obs.
        """

    @abstractmethod
    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Implementation of the sample method.

        This method is called by the sample method to perform the actual sampling after
        validation has passed.

        Parameters
        ----------
        n_obs
            The total number of observations available.

        Yields
        ------
        LoadRequest
            Load requests for batching data.
        """
