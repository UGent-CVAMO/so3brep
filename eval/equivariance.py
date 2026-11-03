"""Relative equivariance error of SO3brep's last-layer node irreps (paper Sec. 4.5).

    Δ_EQ = || Φ(R·X) - ρ(R) Φ(X) || / || Φ(R·X) ||,   ρ = ⊕_l D^(l) over the chosen degrees

averaged over test parts and uniformly random rotations; the paper reports the l = 1 channels
(the default, `--degrees 1`).

    python -m eval.equivariance runs/benchmark/solidletters/seed42 --parts 1000 --rotations 100
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
from e3nn.o3 import Irreps

from dataloaders.loaders import make_loader
from dataloaders.rotate import haar, rotate_pyg
from eval.evaluate import load_task
from train.config import load


def irreps_of_degree(irreps: Irreps, degrees: set[int]) -> tuple[list[int], Irreps]:
    """Column indices and irreps of the blocks with the given degrees."""
    cols, kept, start = [], [], 0
    for mul, ir in irreps:
        width = mul * ir.dim
        if ir.l in degrees:
            cols += list(range(start, start + width))
            kept.append((mul, ir))
        start += width
    return cols, Irreps(kept)


@torch.no_grad()
def equivariance_error(net, data, rotations: int, degrees: set[int], gen: torch.Generator) -> list[float]:
    cols, irreps = irreps_of_degree(net.gnn.irreps_node, degrees)
    h = net.encode(data)[2][:, cols]
    errors = []
    for R in haar(rotations, gen).to(h.device):
        h_rot = net.encode(rotate_pyg(data, R))[2][:, cols]
        expected = h @ irreps.D_from_matrix(R).T
        errors.append(float((h_rot - expected).norm() / h_rot.norm().clamp_min(1e-12)))
    return errors


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", type=Path, help="run folder of an SO3brep model")
    p.add_argument("--parts", type=int, default=1000)
    p.add_argument("--rotations", type=int, default=100)
    p.add_argument("--degrees", type=int, nargs="+", default=[1])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    config = load(args.run / "config.yaml", ["data.batch_size=1", "data.num_workers=2"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_task(args.run / "best.ckpt", config, device).model.eval()
    gen = torch.Generator().manual_seed(args.seed)
    errors = []
    for i, batch in enumerate(make_loader(config.data, "test", shuffle=False)):
        if i == args.parts:
            break
        errors += equivariance_error(net, batch["graph"].to(device), args.rotations, set(args.degrees), gen)
    print(f"Δ_EQ = {statistics.mean(errors):.3e} ± {statistics.pstdev(errors):.3e} "
          f"({min(args.parts, i + 1)} parts × {args.rotations} rotations, degrees {args.degrees})")


if __name__ == "__main__":
    main()
