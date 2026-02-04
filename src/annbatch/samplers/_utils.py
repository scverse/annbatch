"""Utility functions for samplers."""

from __future__ import annotations


def validate_batch_size(batch_size: int, chunk_size: int, preload_nchunks: int) -> None:
    """Validate batch_size against chunk_size and preload_nchunks constraints.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk.
    preload_nchunks
        Number of chunks to preload.

    Raises
    ------
    ValueError
        If batch_size exceeds preload_size or preload_size is not divisible by batch_size.
    """
    preload_size = chunk_size * preload_nchunks

    if batch_size > preload_size:
        raise ValueError(
            "batch_size cannot exceed chunk_size * preload_nchunks. "
            f"Got batch_size={batch_size}, but max is {preload_size}."
        )
    if preload_size % batch_size != 0:
        raise ValueError(
            "chunk_size * preload_nchunks must be divisible by batch_size. "
            f"Got {preload_size} % {batch_size} = {preload_size % batch_size}."
        )
