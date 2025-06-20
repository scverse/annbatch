"""IO.

Lightning data modules.

.. autosummary::
   :toctree: .

   ClassificationDataModule

Torch data loaders.

.. autosummary::
   :toctree: .

   DaskDataset
   read_lazy

Array store creation.

.. autosummary::
   :toctree: .

   create_store_from_h5ads

"""

from .dask_loader import DaskDataset, read_lazy
from .datamodules import ClassificationDataModule
from .store_creation import create_store_from_h5ads
from .zarr_loader import ZarrDenseDataset
