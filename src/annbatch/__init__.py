from __future__ import annotations

from importlib.metadata import version
from importlib.util import find_spec

from packaging.version import Version

if find_spec("cupy-cuda12x") and Version(version("torch")) >= Version("2.11"):
    raise RuntimeError(
        "cupy-cuda12x requires torch < 2.11 because >=2.11 ships with cuda 13. Either upgrade cupy or downgrade torch."
    )

from . import abc, types
from .io import DatasetCollection, write_sharded
from .loader import Loader
from .samplers._chunk_sampler import ChunkSampler, ChunkSamplerDistributed

__version__ = version("annbatch")

__all__ = [
    "Loader",
    "DatasetCollection",
    "types",
    "write_sharded",
    "ChunkSampler",
    "ChunkSamplerDistributed",
    "abc",
]
