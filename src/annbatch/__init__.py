from importlib.metadata import version

from . import types
from .fields import AnnDataField
from .io import add_to_collection, create_anndata_collection, write_sharded
from .loader import Loader

__version__ = version("annbatch")

__all__ = ["AnnDataField", "Loader", "write_sharded", "add_to_collection", "create_anndata_collection", "types"]
