from importlib.metadata import version

from .batcher import Batcher
from .io import add_to_collection, create_anndata_collection, write_sharded

__version__ = version("annbatch")

__all__ = [
    "Batcher",
    "write_sharded",
    "add_to_collection",
    "create_anndata_collection",
]
