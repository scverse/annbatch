"""Sampler classes for efficient data access."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NoReturn, Self

import numpy as np

from annbatch.utils import split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


def _attr_equal(a: object, b: object) -> bool:
    """Structural equality for a single value stored in a sampler's ``__dict__``.

    Handles the container types samplers keep as state -- most importantly the
    :class:`numpy.random.Generator`, whose equality must compare the live *bit
    generator state* (not object identity) so a round-tripped sampler counts as
    equal to its source. numpy arrays and pandas objects (which return
    element-wise ``==``) are compared structurally, and nested samplers dispatch
    back to :meth:`Sampler.__eq__`.
    """
    if isinstance(a, np.random.Generator) or isinstance(b, np.random.Generator):
        return (
            isinstance(a, np.random.Generator)
            and isinstance(b, np.random.Generator)
            and a.bit_generator.state == b.bit_generator.state
        )
    if isinstance(a, Sampler) or isinstance(b, Sampler):
        return a == b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and bool(np.array_equal(a, b))
    # pandas DataFrame/Series/Index/Categorical all expose a structural `.equals`
    if hasattr(a, "equals") and hasattr(b, "equals") and type(a) is type(b):
        return bool(a.equals(b))
    return bool(a == b)


class Sampler(ABC):
    """Base sampler class.

    Samplers control how data is batched and loaded from the underlying datasets.
    """

    _mask: slice = slice(0, None)
    _rng: np.random.Generator | None = None

    def __eq__(self, other: object) -> bool:
        """Two samplers are equal iff they have the same type and the same state.

        State includes the random number generator's *bit generator state*, so a
        sampler equals a pickle/deepcopy round-trip of itself but not a fresh (or
        differently advanced) sampler built from the same seed.
        """
        if type(self) is not type(other):
            return NotImplemented
        if self.__dict__.keys() != other.__dict__.keys():
            return False
        return all(_attr_equal(self.__dict__[key], other.__dict__[key]) for key in self.__dict__)

    def __copy__(self) -> NoReturn:
        """Refuse shallow copies -- they would share the rng with the original.

        A shallow copy keeps the same :class:`numpy.random.Generator` object, so
        advancing one sampler would advance the "copy" too, silently breaking the
        independence a copy is meant to provide. Use :func:`copy.deepcopy` instead.
        """
        raise TypeError(
            f"{type(self).__name__} does not support shallow copying: a shallow copy would share the "
            "random number generator with the original, so the two would not sample independently. "
            "Use copy.deepcopy() instead."
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Deep-copy every attribute into an independent sampler (rng included)."""
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            setattr(new, key, copy.deepcopy(value, memo))
        return new

    @property
    def mask(self) -> slice:
        """The observation range this sampler operates on."""
        return self._mask

    @mask.setter
    def mask(self, value: slice) -> None:
        self._mask = value

    @property
    def rng(self) -> np.random.Generator | None:
        """The random number generator used by this sampler."""
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator | None) -> None:
        self._rng = value

    @property
    @abstractmethod
    def batch_size(self) -> int | None:
        """The batch size for data loading.

        Note
        ----
        This property is only used when the `splits` argument is not supplied in the :class:`annbatch.types.LoadRequest`.
        When `splits` are explicitly provided, they determine the batch boundaries instead.

        Returns
        -------
        int
            The number of observations per batch.
        """

    @property
    @abstractmethod
    def shuffle(self) -> bool:
        """Whether data is shuffled.

        If `batch_size` is provided and :attr:`annbatch.types.LoadRequest.splits` is not, in-memory loaded data will be shuffled or not based on this param.

        Shuffling of on-disk data is up to the user (controlled by `requests` parameter in :class:`annbatch.types.LoadRequest`).

        Returns
        -------
        bool
            True if data should be shuffled, False otherwise.
        """

    @abstractmethod
    def n_batches(self, n_obs: int) -> int:
        """Return the number of batches.

        Parameters
        ----------
        n_obs
            The total number of observations available.

        Returns
        -------
        int
            The total number of batches this sampler will produce.
        """

    def n_iters(self, n_obs: int) -> int:
        """Return the number of batches.

        .. deprecated:: 0.2.0
            Use :meth:`n_batches` instead.
        """
        import warnings

        warnings.warn(
            "n_iters is deprecated, use n_batches instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_batches(n_obs)

    def sample(self, n_obs: int) -> Iterator[LoadRequest]:
        """Sample load requests given the total number of observations.

        Base implementation simply calls :meth:`~annbatch.abc.Sampler.validate` and then yields via :meth:`~annbatch.abc.Sampler._sample`.

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

                # Calculate total observations from requests
                total_obs = sum(chunk.stop - chunk.start for chunk in load_request["requests"])

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
