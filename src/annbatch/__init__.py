from importlib.metadata import version

from .dense import ZarrDenseDataset
from .io import add_h5ads_to_store, create_store_from_h5ads, write_sharded
from .sparse import ZarrSparseDataset

__version__ = version("annbatch")

__all__ = ["ZarrSparseDataset", "ZarrDenseDataset", "write_sharded", "add_h5ads_to_store", "create_store_from_h5ads"]
