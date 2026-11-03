"""DataLoader of one split."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from dataloaders.datasets import BRepGraphs, collate_pyg, read_split
from train.config import DataConfig


def make_loader(cfg: DataConfig, split: str, *, rotated: bool = False, seed: int = 0,
                shuffle: bool | None = None) -> DataLoader:
    """Aligned (or, with `rotated`, preprocessing-rotated) graphs of `split`; the training split is
    shuffled and, if `cfg.train_subset` is set, reduced to that many parts (seeded by `seed`)."""
    root = Path(cfg.root)
    graphs = root / ("graphs_rotated" if rotated else "graphs") / cfg.graphs
    ds = BRepGraphs(cfg.dataset, graphs, stems=read_split(root, split), labels_dir=root / "labels")
    if split == "train" and cfg.train_subset:
        ds.subset(cfg.train_subset, seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=(split == "train") if shuffle is None else shuffle,
                      collate_fn=collate_pyg, num_workers=cfg.num_workers, pin_memory=cfg.num_workers > 0)
