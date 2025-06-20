from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, MetricCollection


class Linear(L.LightningModule):
    def __init__(self, n_genes: int, n_covariates: int, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.linear = torch.nn.Linear(n_genes, n_covariates)

        metrics = MetricCollection(
            [
                F1Score(num_classes=n_covariates, average="macro", task="multiclass"),
                Accuracy(num_classes=n_covariates, task="multiclass"),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, inputs):
        return self.linear(inputs)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, targets)
        self.log("train_loss", loss)
        metrics = self.train_metrics(targets, preds)
        self.log_dict(metrics)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, targets)
        self.log("val_loss", loss)
        metrics = self.val_metrics(targets, preds)
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
