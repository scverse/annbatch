from importlib.metadata import version

from .dense import ZarrDenseDataset
from .io import add_anndata_to_chunks_directory, create_anndata_chunks_directory, write_sharded
from .sparse import ZarrSparseDataset

__version__ = version("arrayloaders")

__all__ = [
    "ZarrSparseDataset",
    "ZarrDenseDataset",
    "write_sharded",
    "add_anndata_to_chunks_directory",
    "create_anndata_chunks_directory",
]
