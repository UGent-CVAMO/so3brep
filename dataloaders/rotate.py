"""Uniformly random rotations of a part on the fly (used to measure the equivariance error).

Preprocessing scales each part to unit radius and casts its rays in the faces' local frames, so a
rotated part's graph is the aligned graph with its world-frame fields rotated: face centres, normals
and tangents (`center_nuv`, `pos`), edge midpoints and the edge-curve samples (position, tangent and
the two face normals). Local-frame fields (`x_grid_local`, `vision_grids`, `face_feat`) do not change.
"""

from __future__ import annotations

import torch
from torch import Tensor


def haar(n: int, generator: torch.Generator | None = None) -> Tensor:
    """n rotations drawn uniformly from SO(3): QR of a Gaussian matrix with the sign fixed."""
    z = torch.randn(n, 3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q[torch.det(q) < 0, :, 0] *= -1
    return q.float()


def _rot(t: Tensor, R: Tensor, *starts: int) -> Tensor:
    """Rotate the 3-vectors starting at `starts` along the last dimension of `t`."""
    t = t.float().clone()
    for s in starts:
        t[..., s:s + 3] = t[..., s:s + 3] @ R.T
    return t


def rotate_pyg(data, R: Tensor):
    """The PyG graph (or batch) of the part rotated by R."""
    g = data.clone()
    R = R.to(g.center_nuv.device)
    g.center_nuv = _rot(g.center_nuv, R, 0, 3, 6, 9)
    g.pos = _rot(g.pos, R, 0)
    if getattr(g, "edge_mid", None) is not None:
        g.edge_mid = _rot(g.edge_mid, R, 0)
    if getattr(g, "edge_curve_pts", None) is not None:
        g.edge_curve_pts = _rot(g.edge_curve_pts, R, 0, 3, 6, 9)
    return g
