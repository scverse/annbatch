from typing import Literal

import anndata as ad
import lightning as L
from torch.utils.data import DataLoader

from .dask_loader import DaskDataset


class ClassificationDataModule(L.LightningDataModule):
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
