"""LightningModule around SO3brep: cross-entropy, accuracy / IoU logging, AdamW with warm-up and cosine decay."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor, nn


def targets(batch: dict, segmentation: bool) -> Tensor:
    """Face labels for segmentation, part labels for classification."""
    return batch["graph"].y.long() if segmentation else batch["label"]


class Task(pl.LightningModule):
    def __init__(self, net: nn.Module, *, num_classes: int, segmentation: bool, lr: float,
                 weight_decay: float = 0.05, warmup_epochs: int = 5, config: dict | None = None):
        super().__init__()
        self.model = net
        self.segmentation, self.lr = segmentation, lr
        self.weight_decay, self.warmup_epochs = weight_decay, warmup_epochs
        self.save_hyperparameters({"config": config or {}})
        stages = ("train", "val")
        self.metrics = nn.ModuleDict({f"{s}_acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                                      for s in stages})
        if segmentation:
            for s in stages:
                self.metrics[f"{s}_iou"] = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes)

    def forward(self, batch: dict) -> Tensor:
        return self.model(batch["graph"])

    def _step(self, batch: dict, stage: str) -> Tensor:
        logits = self(batch)
        target = targets(batch, self.segmentation)
        loss = F.cross_entropy(logits, target, ignore_index=-1)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=len(batch["filename"]))
        valid = target >= 0
        if valid.any():
            pred = logits[valid].argmax(-1)
            self.metrics[f"{stage}_acc"](pred, target[valid])
            if self.segmentation:
                self.metrics[f"{stage}_iou"](pred, target[valid])
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def _log_epoch(self, stage: str):
        for key, metric in self.metrics.items():
            if key.startswith(stage):
                self.log(key, metric.compute(), prog_bar=key == "val_acc")
                metric.reset()

    def on_train_epoch_end(self):
        self._log_epoch("train")

    def on_validation_epoch_end(self):
        self._log_epoch("val")

    def configure_optimizers(self):
        """AdamW; linear warm-up over `warmup_epochs`, then cosine decay to lr / 50 at the last epoch."""
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        epochs = self.trainer.max_epochs if self.trainer is not None and self.trainer.max_epochs else 300
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=self.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - self.warmup_epochs,
                                                            eta_min=self.lr / 50)
        schedule = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[self.warmup_epochs])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": schedule, "interval": "epoch"}}
