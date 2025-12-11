from importlib.metadata import version

from . import types
from .io import add_to_collection, create_anndata_collection, write_sharded
from .loader import Loader

__version__ = version("annbatch")

__all__ = ["Loader", "write_sharded", "add_to_collection", "create_anndata_collection", "types"]
