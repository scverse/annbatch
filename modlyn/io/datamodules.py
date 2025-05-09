import anndata as ad
import lightning as L
from torch.utils.data import DataLoader

from modlyn.io.loading import ZarrDataset


class ClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        adata_train: ad.AnnData,
        adata_val: ad.AnnData,
        label_column: str,
        train_dataloader_kwargs=None,
        val_dataloader_kwargs=None,
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

    def train_dataloader(self):
        train_dataset = ZarrDataset(
            self.adata_train,
            label_column=self.label_col,
        )

        return DataLoader(train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self):
        val_dataset = ZarrDataset(
            self.adata_val,
            label_column=self.label_col,
            shuffle=False,
        )

        return DataLoader(val_dataset, **self.val_dataloader_kwargs)
