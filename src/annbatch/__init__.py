from __future__ import annotations

from importlib.metadata import version

from . import abc, types
from .io import DatasetCollection, write_sharded
from .loader import Loader
from .samplers._chunk_sampler import ChunkSampler

__version__ = version("annbatch")

__all__ = [
    "Loader",
    "DatasetCollection",
    "types",
    "write_sharded",
    "ChunkSampler",
    "abc",
]
