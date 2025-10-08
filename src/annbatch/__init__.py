from importlib.metadata import version

from .dense import ZarrDenseDataset
from .io import add_to_collection, create_anndata_collection, write_sharded
from .sparse import ZarrSparseDataset

__version__ = version("annbatch")

__all__ = [
    "ZarrSparseDataset",
    "ZarrDenseDataset",
    "write_sharded",
    "add_to_collection",
    "create_anndata_collection",
]
