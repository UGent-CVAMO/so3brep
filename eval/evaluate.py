"""Test-set scores of a trained run on the original and the rotated test parts.

    python -m eval.evaluate runs/benchmark/fusion360/seed42

The rotated test parts are written by preprocessing: every test part rotated about the three
coordinate axes by angles drawn uniformly from [0, 2π). Scores: accuracy and, for segmentation,
per-face accuracy and the mean IoU over classes. Faces without a label (-1) are ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from dataloaders.datasets import DATASETS
from dataloaders.loaders import make_loader
from models.so3brep import SO3brep
from train.config import Config, load
from train.task import Task, targets


def scores(preds: np.ndarray, labels: np.ndarray, segmentation: bool) -> dict:
    keep = labels >= 0
    preds, labels = preds[keep], labels[keep]
    out = {"acc": float(accuracy_score(labels, preds))}
    if segmentation:
        classes = np.union1d(labels, preds)
        cm = confusion_matrix(labels, preds, labels=classes)
        tp = np.diag(cm)
        out["iou"] = float(np.mean(tp / (cm.sum(0) + cm.sum(1) - tp)))
    return out


@torch.no_grad()
def predict(task: Task, loader) -> tuple[np.ndarray, np.ndarray]:
    task.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: (v.to(task.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        preds.append(task(batch).argmax(-1).cpu().numpy())
        labels.append(targets(batch, task.segmentation).cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


def load_task(checkpoint: str | Path, config: Config, device: torch.device) -> Task:
    info = DATASETS[config.data.dataset]
    net = SO3brep(info.num_classes, info.segmentation, **config.model)
    task = Task(net, num_classes=info.num_classes, segmentation=info.segmentation, lr=config.train.lr)
    task.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    return task.to(device)


def evaluate(task: Task, config: Config) -> dict:
    segmentation = DATASETS[config.data.dataset].segmentation
    out = {}
    for name, rotated in (("original", False), ("rotated", True)):
        out[name] = scores(*predict(task, make_loader(config.data, "test", rotated=rotated)), segmentation)
        print(name, {k: round(v, 4) for k, v in out[name].items()})
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", type=Path, help="run folder written by train.train (config.yaml, best.ckpt)")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.root=/data/fusion360")
    args = p.parse_args(argv)
    config = load(args.run / "config.yaml", args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate(load_task(args.run / "best.ckpt", config, device), config)
    path = args.run / "metrics.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(previous | result, indent=2))


if __name__ == "__main__":
    main()
