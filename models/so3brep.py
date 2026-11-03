"""SO3brep: SO(3)-equivariant network on the B-rep face-adjacency graph (paper Sec. 3).

Per face:
  * a local reference frame R_i = [u | v | n] from the kernel's tangent and outward normal;
  * scalars: surface attributes (7), a UV-grid CNN embedding, invariant ray statistics;
  * irreps: ray harmonics of degree 1 and 2, computed in the local frame and lifted to the world
    frame with Wigner D-matrices, plus the frame axes as l=1 channels.
Per edge: four world-frame directions (their harmonics) and nine invariant scalars.
A stack of CG message-passing layers (`gnn.py`) and an l=0 readout give per-face and per-part
embeddings; a head turns them into class logits.

The `message` option selects the message function of Tab. 3: "cg" (SO3brep), or the matched
"attention", "concat" and "invariant" variants (`matched.py`).
"""

from __future__ import annotations

import math

import torch
from e3nn.o3 import matrix_to_angles
from e3nn.o3._wigner import so3_generators
from torch import Tensor, nn

from models.gnn import CGMessagePassing, unit_direction
from models.matched import MatchedMessagePassing
from models.ray_encoder import RayEncoder
from models.uvgrid import SurfaceEncoder

MESSAGES = ("cg", "attention", "concat", "invariant")

# Stored edge features (preprocessing order): Concave, Convex, Smooth, EdgeLength, Circular, Closed,
# Elliptical, NonRationalBSpline, RationalBSpline, Straight. Smooth is dropped (smooth edges have both
# convexity flags at zero) and Elliptical is too rare to keep, which leaves the eight curve attributes
# of the paper.
EDGE_FEATURE_COLUMNS = [0, 1, 3, 4, 5, 7, 8, 9]
EDGE_LENGTH_COLUMN = 2  # position of EdgeLength after the selection above
EDGE_SCALARS = 1 + len(EDGE_FEATURE_COLUMNS)  # normal cosine and the curve attributes
EDGE_DIRECTIONS = 4  # centroid to centroid, through the curve midpoint (two), curve chord


def local_frames(center_nuv: Tensor) -> Tensor:
    """(N, 12) [centre, normal, u-tangent, ...] -> (N, 3, 3) frames with rows [u, v, n].

    u is the kernel's u-tangent made orthogonal to the normal (a fixed axis if the tangent is
    degenerate), and v = n x u.
    """
    n = nn.functional.normalize(center_nuv[:, 3:6], dim=-1)
    u = center_nuv[:, 6:9]
    u = u - (u * n).sum(-1, keepdim=True) * n
    u_norm = u.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(n).scatter_(1, n.abs().argmin(dim=-1, keepdim=True), 1.0)
    fallback = nn.functional.normalize(fallback - (fallback * n).sum(-1, keepdim=True) * n, dim=-1)
    u = torch.where((u_norm < 1e-4).expand_as(u), fallback, u / u_norm.clamp(min=1e-12))
    return torch.stack([u, torch.cross(n, u, dim=-1), n], dim=1)


_generators: dict = {}


def wigner_d(l: int, R: Tensor) -> Tensor:
    """Wigner D^(l) of a batch of rotation matrices (N, 3, 3), computed on R's device (e3nn's ZYZ convention)."""
    alpha, beta, gamma = (a[..., None, None] % (2 * math.pi) for a in matrix_to_angles(R))
    key = (l, R.device, R.dtype)
    if key not in _generators:
        _generators[key] = so3_generators(l).to(device=R.device, dtype=R.dtype)
    X = _generators[key]
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])


class Head(nn.Module):
    """Linear -> BatchNorm -> SiLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.BatchNorm1d(in_dim // 2), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(in_dim // 2, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SO3brep(nn.Module):
    """Defaults are the paper model (Sec. 4, "Implementation").

    Args:
        num_classes, segmentation: task (per-face logits if segmentation, else per part).
        message: "cg" | "attention" | "concat" | "invariant" (Tab. 3).
        l_max: highest irrep degree, 2, 1 or 0 (Fig. 4).
        surface_attrs, uv_grid, rays: the three node encoders (Fig. 4); the surface attributes include
            the u and v frame axes, so without them only the normal is an l=1 frame channel.
        num_rbf: size of the Gaussian radial basis of the centroid distance.
        use_node_features, use_edge_features, attention: architecture ablations (Fig. 4).
    """

    def __init__(
        self,
        num_classes: int,
        segmentation: bool,
        *,
        message: str = "cg",
        scalar_channels: int = 64,
        vector_channels: int = 16,
        tensor_channels: int = 8,
        l_max: int = 2,
        layers: int = 3,
        heads: int = 8,
        attention: bool = True,
        readout_dim: int = 128,
        uv_dim: int = 64,
        ray_dim: int = 64,
        ray_hidden: int = 64,
        ray_grid: tuple[int, int] = (6, 12),
        radial_hidden: int = 64,
        num_rbf: int = 16,
        surface_attrs: bool = True,
        uv_grid: bool = True,
        rays: bool = True,
        use_node_features: bool = True,
        use_edge_features: bool = True,
        dropout: float = 0.15,
        channel_dropout: float = 0.1,
    ):
        super().__init__()
        if message not in MESSAGES:
            raise ValueError(f"message must be one of {MESSAGES}, got {message!r}")
        if l_max < 2:
            tensor_channels = 0
        if l_max < 1:
            vector_channels = 0
        self.segmentation = segmentation
        self.message = message
        self.invariant = message == "invariant"   # keep node features in the local frames
        self.surface_attrs, self.use_edge_features = surface_attrs, use_edge_features
        self.vector_channels, self.tensor_channels = vector_channels, tensor_channels

        fused = 0
        self.ray_encoder = None
        if rays:
            self.ray_encoder = RayEncoder(*ray_grid, scalar_dim=ray_dim, n_vector=vector_channels,
                                          n_tensor=tensor_channels, hidden=ray_hidden)
            fused += ray_dim
        self.face_attr_proj = None
        if surface_attrs:
            self.face_attr_proj = nn.Linear(7, ray_dim)
            fused += ray_dim
        self.surf_encoder = None
        if uv_grid:
            self.surf_encoder = SurfaceEncoder(in_channels=7, output_dims=uv_dim)
            fused += uv_dim
        self.scalar_proj = nn.Sequential(nn.BatchNorm1d(fused), nn.SiLU(), nn.Dropout(dropout),
                                         nn.Linear(fused, scalar_channels))

        n_vector_in = (vector_channels if rays else 0) + (3 if surface_attrs else 1) if vector_channels else 0
        n_tensor_in = tensor_channels if (rays and tensor_channels) else 0
        common = dict(layers=layers, heads=heads, out_dim=readout_dim, radial_hidden=radial_hidden,
                      num_rbf=num_rbf, channel_dropout=channel_dropout,
                      use_edge_features=use_edge_features, use_node_features=use_node_features)
        if message == "cg":
            self.gnn = CGMessagePassing(scalar_channels, vector_channels, tensor_channels, n_vector_in, n_tensor_in,
                                        attention=attention, edge_scalars=EDGE_SCALARS, **common)
        else:
            self.gnn = MatchedMessagePassing(
                scalar_channels, vector_channels, tensor_channels, n_vector_in, n_tensor_in,
                attention=message == "attention", invariant=self.invariant,
                edge_scalars=EDGE_SCALARS + (3 * EDGE_DIRECTIONS if self.invariant else 0), **common)

        in_dim = 3 * readout_dim if segmentation else 2 * readout_dim
        self.seg = Head(in_dim, num_classes, dropout) if segmentation else None
        self.clf = None if segmentation else Head(in_dim, num_classes, dropout)

    # ------------------------------------------------------------------ node features

    def node_features(self, data, frames: Tensor) -> Tensor:
        """(N, irreps) node input: fused scalars, then l=1 channels, then l=2 channels (world frame)."""
        n = frames.shape[0]
        scalars, l1, l2 = [], None, None
        if self.ray_encoder is not None:
            ray_scalars, l1, l2 = self.ray_encoder(data.vision_grids)
            scalars.append(ray_scalars)
        if self.face_attr_proj is not None:
            attrs = data.face_feat.clone()
            attrs[:, 5] = torch.log1p(attrs[:, 5])  # face area
            scalars.append(self.face_attr_proj(attrs))
        if self.surf_encoder is not None:
            scalars.append(self.surf_encoder(data.x_grid_local))
        parts = [self.scalar_proj(torch.cat(scalars, dim=1))]

        if not self.invariant:  # lift from the local frame to the world frame: x_world = R^T x_local
            if l1 is not None:
                l1 = torch.bmm(l1, frames)
            if l2 is not None:
                l2 = torch.bmm(l2, wigner_d(2, frames.transpose(1, 2)).transpose(1, 2))
        if self.vector_channels:
            axes = torch.eye(3, device=frames.device).expand(n, 3, 3) if self.invariant else frames
            axes = axes if self.surface_attrs else axes[:, 2:3]
            parts.append((torch.cat([l1, axes], dim=1) if l1 is not None else axes).reshape(n, -1))
        if self.tensor_channels and l2 is not None:
            parts.append(l2.reshape(n, -1))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ edge features

    def edge_scalars(self, data, frames: Tensor) -> Tensor:
        """(E, 9): normal cosine of the two faces, then the eight curve attributes (length log-compressed)."""
        src, dst = data.edge_index
        cos = (frames[src, 2] * frames[dst, 2]).sum(-1, keepdim=True)
        curve = data.edge_attr[:, EDGE_FEATURE_COLUMNS].clone()
        curve[:, EDGE_LENGTH_COLUMN] = torch.log1p(curve[:, EDGE_LENGTH_COLUMN].clamp(min=0.0))
        return torch.cat([cos, curve], dim=-1)

    def _sender_frame_directions(self, data, frames: Tensor) -> Tensor:
        """The four edge directions expressed in the sender's local frame (invariant variant only)."""
        pos, mid = data.pos, data.edge_mid
        src, dst = data.edge_index
        first, last = data.edge_curve_pts[:, 0, :3], data.edge_curve_pts[:, -1, :3]
        dirs = [unit_direction(pos[src] - pos[dst], pos[src], pos[dst]),
                unit_direction(mid - pos[src], mid, pos[src]),
                unit_direction(pos[dst] - mid, pos[dst], mid),
                unit_direction(last - first, last, first)]
        R = frames[src]
        return torch.cat([torch.bmm(R, d.unsqueeze(-1)).squeeze(-1) for d in dirs], dim=-1)

    # ------------------------------------------------------------------ forward

    def encode(self, data) -> tuple[Tensor, Tensor, Tensor]:
        """Per-face embeddings, graph embedding and the last-layer node irreps."""
        frames = local_frames(data.center_nuv)
        x = self.node_features(data, frames)
        e = self.edge_scalars(data, frames)
        if not self.use_edge_features:
            e = torch.zeros_like(e)
        if self.invariant:
            e = torch.cat([e, self._sender_frame_directions(data, frames)], dim=-1)
        return self.gnn(x, data.pos, data.edge_index, data.batch, e, data.edge_curve_pts, data.edge_mid)

    def forward(self, data) -> Tensor:
        node_emb, graph_emb, _ = self.encode(data)
        if self.segmentation:
            return self.seg(torch.cat([node_emb, graph_emb[data.batch]], dim=1))
        return self.clf(graph_emb)
