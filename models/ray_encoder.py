"""Spherical-harmonic encoding of the rays cast from each face (paper Sec. 3.2, "Ray field").

From the centre of every face, M rays are cast over the outward hemisphere (+n) and M over the
inward one (-n), on a Fibonacci grid fixed in the face's local frame. Each ray gives three signals
per hemisphere; projecting them on real spherical harmonics gives, per face,

    f_l = (1/M) sum_k f(d_k) Y_l(d_k),   l = 0, 1, 2,

computed in the local frame (so they do not depend on the pose of the part). For l > 0 the signals
are first centred on their hemisphere mean, so those coefficients carry only directional variation.
"""

from __future__ import annotations

import math

import torch
from e3nn.o3 import spherical_harmonics
from torch import Tensor, nn


def fibonacci_hemisphere(m: int) -> Tensor:
    """m unit directions on the hemisphere z > 0, in the order the ray caster used them."""
    idx = torch.arange(m, dtype=torch.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - (idx + 0.5) / m
    r = torch.sqrt(1.0 - z * z)
    t = golden * idx
    return torch.stack([r * torch.cos(t), r * torch.sin(t), z], dim=-1).float()


class RayEncoder(nn.Module):
    """Ray grids (N, 6, n_el, n_az) -> scalars (N, scalar_dim), l=1 (N, n_vector, 3), l=2 (N, n_tensor, 5).

    Channels of the stored grid: 0 hit, 1 distance, 2 normal dot (outward); 3 hit, 4 distance,
    5 normal dot (inward). Inward channel 0 is replaced by the proximity hit / (1 + distance),
    which is large behind thin walls. Invariant inputs (l=0 means and the norms of the l=1 and
    l=2 coefficients) go through an MLP; the l>0 coefficients are mixed across signal channels by
    one bias-free linear map per hemisphere, which keeps them irreps of the local frame.
    """

    def __init__(
        self,
        n_el: int = 6,
        n_az: int = 12,
        scalar_dim: int = 64,
        n_vector: int = 16,
        n_tensor: int = 8,
        hidden: int = 64,
    ):
        super().__init__()
        if n_vector % 2 or n_tensor % 2:
            raise ValueError("n_vector and n_tensor are split over two hemispheres and must be even")
        self.M = n_el * n_az
        self.n_vector, self.n_tensor = n_vector, n_tensor

        fwd = fibonacci_hemisphere(self.M)
        bwd = fwd.clone()
        bwd[:, 2] = -bwd[:, 2]
        for name, dirs in (("fwd", fwd), ("bwd", bwd)):
            for l in (0, 1, 2):
                Y = spherical_harmonics([l], dirs, normalize=True, normalization="component")
                self.register_buffer(f"_Y{l}_{name}", Y)  # (M, 2l+1)

        self.scalar_mlp = nn.Sequential(
            nn.LayerNorm(18),  # 6 channels x (l=0 mean, |l=1|, |l=2|)
            nn.Linear(18, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, scalar_dim),
        )
        if n_vector:
            self.vec_proj_ov = nn.Linear(3, n_vector // 2, bias=False)
            self.vec_proj_iv = nn.Linear(3, n_vector // 2, bias=False)
        if n_tensor:
            self.t2_proj_ov = nn.Linear(3, n_tensor // 2, bias=False)
            self.t2_proj_iv = nn.Linear(3, n_tensor // 2, bias=False)

    def _project(self, signals: Tensor, Y: Tensor) -> Tensor:
        """Discrete SH transform: (N, 3, M) x (M, D) -> (N, 3, D)."""
        return torch.bmm(signals, Y.unsqueeze(0).expand(signals.shape[0], -1, -1)) / self.M

    def forward(self, grids: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        n = grids.shape[0]
        g = grids.reshape(n, grids.shape[1], self.M)
        outward = g[:, 0:3]
        inward = torch.stack([g[:, 3] / (g[:, 4] + 1.0), g[:, 4], g[:, 5]], dim=1)
        signals = torch.cat([outward, inward], dim=1)  # (N, 6, M)

        means = signals.mean(dim=-1)
        centred = signals - means.unsqueeze(-1)
        ov, iv = centred[:, :3], centred[:, 3:]
        l1_ov, l2_ov = self._project(ov, self._Y1_fwd), self._project(ov, self._Y2_fwd)
        l1_iv, l2_iv = self._project(iv, self._Y1_bwd), self._project(iv, self._Y2_bwd)

        norms1 = torch.cat([l1_ov.norm(dim=-1), l1_iv.norm(dim=-1)], dim=1)
        norms2 = torch.cat([l2_ov.norm(dim=-1), l2_iv.norm(dim=-1)], dim=1)
        scalars = self.scalar_mlp(torch.cat([means, norms1, norms2], dim=-1))

        def mix(name, x_ov, x_iv):
            return torch.cat([getattr(self, f"{name}_ov")(x_ov.transpose(1, 2)).transpose(1, 2),
                              getattr(self, f"{name}_iv")(x_iv.transpose(1, 2)).transpose(1, 2)], dim=1)

        vectors = mix("vec_proj", l1_ov, l1_iv) if self.n_vector else None
        tensors = mix("t2_proj", l2_ov, l2_iv) if self.n_tensor else None
        return scalars, vectors, tensors
