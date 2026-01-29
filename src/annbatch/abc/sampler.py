"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from annbatch.utils import split_given_size

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
        This property is only used when the `splits` argument is not supplied in the {class}`annbatch.types.LoadRequest`.
        When `splits` are explicitly provided, they determine the batch boundaries instead.

        Returns
        -------
        int
            The number of observations per batch.
        """

    @property
    @abstractmethod
    def shuffle(self) -> bool | None:
        """Whether to shuffle in memory data during sampling.

        Shuffling of on-disk data is up to the user (controlled by `chunks` parameter in {class}`annbatch.types.LoadRequest`).

        Returns
        -------
        bool
            True if data should be shuffled, False otherwise.
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
            if "splits" not in load_request:
                batch_size = self.batch_size
                if batch_size is None:
                    raise ValueError("batch_size must be set when splits are not provided in LoadRequest")
                shuffle = self.shuffle
                if shuffle is None:
                    raise ValueError("shuffle must be set when splits are not provided in LoadRequest")

                # Calculate total observations from chunks
                total_obs = sum(chunk.stop - chunk.start for chunk in load_request["chunks"])

                # Generate indices with optional shuffling and split into batches
                indices = np.random.permutation(total_obs) if shuffle else np.arange(total_obs)
                load_request["splits"] = split_given_size(indices, batch_size)

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
