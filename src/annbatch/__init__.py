from __future__ import annotations

from importlib.metadata import version
from importlib.util import find_spec

from packaging.version import Version

if find_spec("cupy") and find_spec("cuda-toolkit"):
    import cupy as cp

    cupy_expected_runtime_cuda_version = cp.cuda.runtime.runtimeGetVersion()
    # Is this safe?
    if int(str(cupy_expected_runtime_cuda_version)[:2]) != version("cuda-toolkit")[0]:
        msg = (
            "Found mismatched `cupy` compiled version and `cuda-toolkit` version."
            "For example, cupy-cuda12x requires torch < 2.11, which ships with cuda 12 by default, or >=2.11 with cuda 12 because >=2.11 ships with cuda 13 by default."
            "See the torch release notes: https://github.com/pytorch/pytorch/releases/tag/v2.11.0."
            "Either ensure torch gets cuda 12 wheels via `--extra-index-url https://download.pytorch.org/whl/cu128` or upgrade `cupy`."
        )
        raise RuntimeError(msg)

from . import abc, samplers, types
from .io import DatasetCollection, write_sharded
from .loader import Loader
from .samplers import ChunkSampler

__version__ = version("annbatch")

__all__ = [
    "Loader",
    "ChunkSampler",
    "DatasetCollection",
    "samplers",
    "types",
    "write_sharded",
    "abc",
]
