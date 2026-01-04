"""Sampler classes for efficient slice-based data access from Zarr stores.

This module provides samplers optimized for slice-based data access patterns:

- :class:`~annbatch.sampler.Sampler`: Abstract base class for all samplers.
- :class:`~annbatch.sampler.SliceSampler`: Slice-based access for full or
  partial dataset iteration.

"""

from annbatch.sampler._sampler import Sampler, SliceSampler

# Update __module__ so Sphinx can find the re-exported classes
Sampler.__module__ = "annbatch.sampler"
SliceSampler.__module__ = "annbatch.sampler"

__all__ = [
    "Sampler",
    "SliceSampler",
]
