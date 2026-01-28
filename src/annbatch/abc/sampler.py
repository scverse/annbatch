"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class Sampler(ABC):
    """Base sampler class.

    Samplers control how data is batched and loaded from the underlying datasets.
    """

    @property
    @abstractmethod
    def batch_size(self) -> int | None:
        """The batch size for data loading.

        Note
        ----
        This property is only used when the `splits` argument is not supplied
        in the LoadRequest. When `splits` are explicitly provided, they determine
        the batch boundaries instead.

        Returns
        -------
        int
            The number of observations per batch.
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
        for load_request in self._sample(n_obs):
            # If splits are not provided, generate them based on batch_size
            if "splits" not in load_request or load_request["splits"] is None:
                batch_size = self.batch_size
                if batch_size is None:
                    raise ValueError("batch_size must be set when splits are not provided in LoadRequest")

                # Calculate total observations from chunks
                total_obs = sum(chunk.stop - chunk.start for chunk in load_request["chunks"])

                # Generate random permutation and split into batches
                indices = np.random.permutation(total_obs)
                splits = []
                for i in range(0, total_obs, batch_size):
                    batch_indices = indices[i : i + batch_size]
                    if len(batch_indices) > 0:  # Ensure non-empty batches
                        splits.append(batch_indices)

                load_request["splits"] = splits

            yield load_request

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
