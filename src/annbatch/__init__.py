from __future__ import annotations

from importlib.metadata import version

from . import abc, samplers, types
from .io import DatasetCollection, write_sharded
from .loader import Loader

__version__ = version("annbatch")

__all__ = [
    "Loader",
    "DatasetCollection",
    "samplers",
    "types",
    "write_sharded",
    "abc",
]
