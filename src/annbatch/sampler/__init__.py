"""Sampler classes for efficient chunk-based data access.

This module provides samplers optimized for chunk-based data access patterns.
"""

from annbatch.sampler import abc
from annbatch.sampler._chunk_sampler import ChunkSampler

__all__ = [
    "ChunkSampler",
    "abc",
]
