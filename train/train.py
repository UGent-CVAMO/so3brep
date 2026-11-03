"""Train SO3brep for one or more seeds, then score each run on the original and rotated test parts.

    python -m train.train configs/benchmark/fusion360.yaml --seed 42 [--set train.patience=10]

Writes <out>/<name>/seed<seed>/: config.yaml, best.ckpt (lowest validation loss), last.ckpt,
metrics.csv (per epoch) and metrics.json (test scores).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from dataloaders.datasets import DATASETS
from dataloaders.loaders import make_loader
from eval.evaluate import evaluate, load_task
from models.so3brep import SO3brep
from train.config import Config, load
from train.task import Task


def run(config: Config, seed: int, wandb_project: str | None = None) -> dict:
    pl.seed_everything(seed, workers=True)
    info = DATASETS[config.data.dataset]
    out = Path(config.out) / config.name / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    train = make_loader(config.data, "train", seed=seed)
    val = make_loader(config.data, "val")
    (out / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    net = SO3brep(info.num_classes, info.segmentation, **config.model)
    task = Task(net, num_classes=info.num_classes, segmentation=info.segmentation, lr=config.train.lr,
                weight_decay=config.train.weight_decay, warmup_epochs=config.train.warmup_epochs,
                config=config.to_dict())
    n_params = sum(p.numel() for p in net.parameters())
    print(f"{config.name} seed {seed}: {len(train.dataset)} train / {len(val.dataset)} val parts, "
          f"{n_params:,} parameters")

    loggers = [CSVLogger(str(out), name="", version="")]
    if wandb_project:
        from lightning.pytorch.loggers import WandbLogger
        loggers.append(WandbLogger(project=wandb_project, name=f"{config.name}-{seed}"))
    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        callbacks=[
            ModelCheckpoint(dirpath=str(out), filename="best", monitor="val_loss", mode="min", save_last=True),
            EarlyStopping(monitor="val_loss", mode="min", patience=config.train.patience,
                          min_delta=config.train.min_delta),
        ],
        logger=loggers, accelerator="auto", devices=1, log_every_n_steps=10, num_sanity_val_steps=0,
        enable_progress_bar=sys.stdout.isatty(),
    )
    trainer.fit(task, train, val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = load_task(out / "best.ckpt", config, device)
    result = {"seed": seed, "epochs": trainer.current_epoch, "params": n_params} | evaluate(best, config)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config")
    p.add_argument("--seed", type=int, nargs="*", help="default: the seeds listed in the config")
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. data.batch_size=16")
    p.add_argument("--wandb", metavar="PROJECT", help="also log to Weights & Biases")
    args = p.parse_args(argv)
    for seed in args.seed or load(args.config, args.set).seeds:
        run(load(args.config, args.set), seed, args.wandb)


if __name__ == "__main__":
    main()
