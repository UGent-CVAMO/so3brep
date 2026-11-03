"""Parameters, inference and training-step time and peak memory of SO3brep on one batch (paper Tab. 4).

    python -m eval.cost --dataset fusion360 --root data/fusion360

Times are mean ± std over `--iters` timed passes after `--warmup` untimed ones on the first training
batch. A training step is forward, backward and an AdamW update.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from dataloaders.datasets import DATASETS
from dataloaders.loaders import make_loader
from models.so3brep import SO3brep
from train.config import DataConfig
from train.task import targets


def timed(fn, device, warmup: int, iters: int) -> tuple[float, float, float]:
    """(mean ms, std ms, peak MB)."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(1000 * (time.perf_counter() - t0))
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else float("nan")
    return statistics.mean(times), statistics.pstdev(times), peak


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="fusion360")
    p.add_argument("--root", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=1000)
    a = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = DATASETS[a.dataset]
    loader = make_loader(DataConfig(dataset=a.dataset, root=a.root, batch_size=a.batch_size, num_workers=0),
                         "train", shuffle=False)
    batch = next(iter(loader))
    graph, target = batch["graph"].to(device), targets(batch, info.segmentation).to(device)
    net = SO3brep(info.num_classes, info.segmentation).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)

    net.eval()
    with torch.no_grad():
        fwd = timed(lambda: net(graph), device, a.warmup, a.iters)

    def step():
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(net(graph), target, ignore_index=-1).backward()
        opt.step()

    net.train()
    train = timed(step, device, a.warmup, a.iters)
    print(f"SO3brep  {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters  "
          f"inference {fwd[0]:.1f} ± {fwd[1]:.1f} ms  train step {train[0]:.1f} ± {train[1]:.1f} ms  "
          f"memory {train[2]:.0f} MB")


if __name__ == "__main__":
    main()
