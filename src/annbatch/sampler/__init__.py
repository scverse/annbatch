"""Sampler classes for efficient chunk-aligned data access from Zarr stores.

This module provides samplers optimized for chunk-based data access patterns:

- :class:`~annbatch.sampler.Sampler`: Abstract base class for all samplers.
- :class:`~annbatch.sampler.SliceSampler`: Chunk-aligned access for full or
  partial dataset iteration. Supports sharding via ``start_index`` and
  ``end_index`` for distributed training.
- :class:`~annbatch.sampler.CategoricalSampler`: Category-stratified sampling
  that first samples a category, then samples within that category.

"""

from annbatch.sampler._sampler import Sampler, SliceSampler

# Update __module__ so Sphinx can find the re-exported classes
Sampler.__module__ = "annbatch.sampler"
SliceSampler.__module__ = "annbatch.sampler"

__all__ = [
    "Sampler",
    "SliceSampler",
]
