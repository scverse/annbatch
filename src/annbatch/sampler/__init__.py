"""Sampler classes for efficient slice-based data access.

This module provides samplers optimized for slice-based data access patterns:

- :class:`~annbatch.sampler.Sampler`: Abstract base class for all samplers.
- :class:`~annbatch.sampler.SliceSampler`: Slice-based access for full or
  partial dataset iteration.
- :class:`~annbatch.sampler.LoadRequest`: Request object yielded by samplers.

"""

from annbatch.sampler._sampler import LoadRequest, Sampler, SliceSampler

# Update __module__ so Sphinx can find the re-exported classes
LoadRequest.__module__ = "annbatch.sampler"
Sampler.__module__ = "annbatch.sampler"
SliceSampler.__module__ = "annbatch.sampler"

__all__ = [
    "LoadRequest",
    "Sampler",
    "SliceSampler",
]

