from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
from torch.utils.data import DataLoader

from .dask_loader import DaskDataset

if TYPE_CHECKING:
    from typing import Literal

    import anndata as ad


class ClassificationDataModule(L.LightningDataModule):
    """A LightningDataModule for classification tasks using arrayloaders.io.DaskDataset.

    Args:
        adata_train: anndata.AnnData object containing the training data.
        adata_val: anndata.AnnData object containing the validation data.
        label_column: Name of the column in `obs` that contains the target values.
        train_dataloader_kwargs: Additional keyword arguments passed to the torch DataLoader for the training dataset.
        val_dataloader_kwargs: Additional keyword arguments passed to the torch DataLoader for the validation dataset.
        n_chunks: Number of chunks of the underlying dask.array to load at a time. Loading more chunks at a time can improve performance and randomness, but increases memory usage.
        dask_scheduler: The Dask scheduler to use for parallel computation. Use "synchronous" for single-threaded execution or "threads" for multithreaded execution.

    Examples:
        >>> from arrayloaders.io.datamodules import ClassificationDataModule
        >>> from arrayloaders.io.dask_loader import read_lazy_store
        >>> adata_train = read_lazy_store("path/to/train/store", obs_columns=["label"])
        >>> adata_train.obs["y"] = adata_train.obs["label"].cat.codes.to_numpy().astype("i8")
        >>> datamodule = ClassificationDataModule(
        ...     adata_train=adata_train,
        ...     adata_val=None,
        ...     label_column="label",
        ...     train_dataloader_kwargs={
        ...         "batch_size": 2048,
        ...         "drop_last": True,
        ...         "num_workers": 4
        ...     },
        ... )
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        adata_train: ad.AnnData | None,
        adata_val: ad.AnnData | None,
        label_column: str,
        train_dataloader_kwargs=None,
        val_dataloader_kwargs=None,
        n_chunks: int = 8,
        dask_scheduler: Literal["synchronous", "threads"] = "threads",
    ):
        super().__init__()
        if train_dataloader_kwargs is None:
            train_dataloader_kwargs = {}
        if val_dataloader_kwargs is None:
            val_dataloader_kwargs = {}

        self.adata_train = adata_train
        self.adata_val = adata_val
        self.label_col = label_column
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs
        self.n_chunks = n_chunks
        self.dask_scheduler = dask_scheduler

    def train_dataloader(self):
        train_dataset = DaskDataset(
            self.adata_train,
            label_column=self.label_col,
            n_chunks=self.n_chunks,
            dask_scheduler=self.dask_scheduler,
        )

        return DataLoader(train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self):
        val_dataset = DaskDataset(
            self.adata_val,
            label_column=self.label_col,
            shuffle=False,
            n_chunks=self.n_chunks,
            dask_scheduler=self.dask_scheduler,
        )

        return DataLoader(val_dataset, **self.val_dataloader_kwargs)
