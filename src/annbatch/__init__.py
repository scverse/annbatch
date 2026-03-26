from __future__ import annotations

from importlib.metadata import version
from importlib.util import find_spec

from packaging.version import Version

if find_spec("cupy"):
    import cupy as cp

    cuda_version = cp.cuda.runtime.runtimeGetVersion()
    # Is this safe?
    if int(str(cuda_version)[:2]) != version("cuda-toolkit")[0]:
        msg = (
            "Found mismatched `cupy` compiled version and `cuda-toolkit` version."
            "For example, cupy-cuda12x requires torch < 2.11, which ships with cuda 12 by default, or >=2.11 with cuda 12 because >=2.11 ships with cuda 13 by default."
            "See the torch release notes: https://github.com/pytorch/pytorch/releases/tag/v2.11.0."
            "Either ensure torch gets cuda 12 wheels via `--extra-index-url https://download.pytorch.org/whl/cu128` or upgrade `cupy`."
        )
        raise RuntimeError(msg)

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
