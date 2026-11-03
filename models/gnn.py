"""SO(3)-equivariant message passing on the face-adjacency graph (paper Sec. 3.4).

Node states are irreps `C0x0e + C1x1o + C2x2e`. Each layer:
    h  <- h + V( sum_j  alpha_ji * m_{j->i} )            message block, Eq. (7)-(8)
    h  <- h + FFN(RMSNorm(h))                             gated feed-forward block
with m_{j->i} = W( CG(h_j, Y(e_ji)) * w(a_ji, rbf(r_ji)) ): a Clebsch-Gordan tensor product of the
sender's irreps with the spherical harmonics of four edge directions, gated per coupling path by a
radial network on invariant edge scalars. Attention weights come from l=0 channels only, so they are
invariant. The readout keeps only l=0 channels.

`MessagePassing` holds the parts shared with the non-equivariant message functions in `matched.py`: edge
directions and their harmonics, the input projection, the ablation switches and the pooling.
"""

from __future__ import annotations

import math

import torch
from e3nn.nn import Gate
from e3nn.o3 import ElementwiseTensorProduct, Irreps, Linear, TensorProduct, spherical_harmonics
from torch import Tensor, nn
from torch_geometric.nn import global_max_pool, global_mean_pool

# ----------------------------------------------------------------------------- small pieces

def unit_direction(diff: Tensor, a: Tensor, b: Tensor, rel_tol: float = 1e-4) -> Tensor:
    """`diff = a - b` as a unit vector, or zero where a and b coincide up to float rounding.

    A zero vector has zero harmonics for every l > 0, so an edge whose two points coincide (the
    ends of a closed curve) carries no direction instead of a random one.
    """
    norm = diff.norm(dim=-1, keepdim=True)
    scale = a.norm(dim=-1, keepdim=True) + b.norm(dim=-1, keepdim=True)
    return torch.where(norm > rel_tol * scale, diff / norm.clamp(min=1e-8), torch.zeros_like(diff))


def node_irreps(n_scalar: int, n_vector: int, n_tensor: int) -> Irreps:
    parts = [f"{n_scalar}x0e"]
    if n_vector:
        parts.append(f"{n_vector}x1o")
    if n_tensor:
        parts.append(f"{n_tensor}x2e")
    return Irreps(" + ".join(parts))


def edge_irreps(l_max: int, n_dirs: int) -> Irreps:
    """Harmonics of `n_dirs` edge directions up to l_max (l=0 is constant, so it is kept once)."""
    return Irreps(" + ".join(
        f"{1 if l == 0 else n_dirs}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(l_max + 1)
    ))


def scatter_softmax(logits: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Softmax of (E, H) logits over the incoming edges of each node."""
    maxes = logits.new_full((num_nodes, logits.shape[-1]), float("-inf"))
    maxes.scatter_reduce_(0, index.unsqueeze(-1).expand_as(logits), logits, reduce="amax")
    exp = (logits - maxes[index]).exp()
    denom = logits.new_zeros(num_nodes, logits.shape[-1])
    denom.scatter_add_(0, index.unsqueeze(-1).expand_as(exp), exp)
    return exp / denom[index].clamp(min=1e-12)


class GaussianSmearing(nn.Module):
    """Gaussian radial basis of a distance, centres evenly spaced on [0, cutoff] (SchNet)."""

    def __init__(self, num_rbf: int = 16, cutoff: float = 8.0):
        super().__init__()
        offset = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("means", offset)
        self.register_buffer("betas", torch.full((num_rbf,), -0.5 / (offset[1] - offset[0]).item() ** 2))

    def forward(self, dist: Tensor) -> Tensor:
        return torch.exp(self.betas * (dist - self.means) ** 2)


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Two hidden layers with LayerNorm + SiLU; used for radial weights and concat messages."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim, bias=False),
    )


class EquivariantRMSNorm(nn.Module):
    """RMS normalisation per irrep block with a learned scale per channel (Equiformer)."""

    def __init__(self, irreps: Irreps, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(mul)) for mul, _ in irreps])
        self._blocks, start = [], 0
        for mul, ir in irreps:
            self._blocks.append((start, start + mul * ir.dim, mul, ir.dim))
            start += mul * ir.dim

    def forward(self, x: Tensor) -> Tensor:
        out = []
        for scale, (s, e, mul, d) in zip(self.scales, self._blocks):
            chunk = x[:, s:e]
            chunk = chunk / (chunk.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
            out.append((chunk.reshape(-1, mul, d) * scale[None, :, None]).reshape(-1, mul * d))
        return torch.cat(out, dim=-1)


class GatedFFN(nn.Module):
    """Equivariant feed-forward: SiLU on scalars, sigmoid scalar gates on l > 0 (Weiler et al.)."""

    def __init__(self, irreps: Irreps, expansion: int = 2):
        super().__init__()
        n_scalar = sum(mul for mul, ir in irreps if ir.l == 0)
        higher = Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0])
        scalars = Irreps(f"{n_scalar * expansion}x0e")
        gates = Irreps(f"{higher.num_irreps}x0e") if higher.num_irreps else Irreps("")
        self.gate = Gate(scalars, [nn.SiLU()], gates, [torch.sigmoid] * len(gates), higher)
        self.linear_1 = Linear(irreps, scalars + gates + higher if higher.num_irreps else scalars)
        self.linear_2 = Linear(self.gate.irreps_out, irreps)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.gate(self.linear_1(x)))


class EquivariantDropout(nn.Module):
    """Drops whole irrep channels: one Bernoulli scalar per channel, broadcast over its 2l+1 components."""

    def __init__(self, irreps: Irreps, p: float):
        super().__init__()
        self.num_irreps = irreps.num_irreps
        self.drop = nn.Dropout(p)
        self.mul = ElementwiseTensorProduct(irreps, Irreps(f"{self.num_irreps}x0e"))

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        return self.mul(x, self.drop(x.new_ones(x.shape[0], self.num_irreps)))


# ----------------------------------------------------------------------------- CG message

class TensorProductMessage(nn.Module):
    """m = W( CG(h_src, Y(e)) * w(edge scalars) ), Eq. (7).

    Every path couples each input channel u with each edge direction v ("uvu": the directions are
    summed into channel u). The radial network gives one gate per input channel and path, and these
    gates are tiled over the (u, v) pairs: pair (u, v) uses gate (4u + v) mod C, with C the number of
    input channels. All four directions keep their own weight, so messages tell them apart.
    """

    def __init__(self, irreps_node: Irreps, irreps_edge: Irreps, edge_scalars: int, hidden: int, num_rbf: int):
        super().__init__()
        self.rbf = GaussianSmearing(num_rbf)
        targets = {ir for _, ir in irreps_node}

        def paths(edge):
            instr, mid = [], []
            for i_in, (mul, ir_in) in enumerate(irreps_node):
                for i_e, (_, ir_e) in enumerate(edge):
                    for ir_out in ir_in * ir_e:
                        if ir_out in targets:
                            instr.append((i_in, i_e, len(mid), "uvu", True))
                            mid.append((mul, ir_out))
            return instr, Irreps(mid)

        instructions, irreps_mid = paths(irreps_edge)
        self.tp = TensorProduct(irreps_node, irreps_edge, irreps_mid, instructions=instructions,
                                shared_weights=False, internal_weights=False)
        self.proj = Linear(irreps_mid, irreps_node)

        # 1/sqrt(number of edge directions feeding each output path)
        fan_in: dict[int, int] = {}
        for ins in self.tp.instructions:
            fan_in[ins.i_out] = fan_in.get(ins.i_out, 0) + irreps_edge[ins.i_in2].mul
        rescale = [fan_in.get(i, 1) ** -0.5 for i, (mul, ir) in enumerate(irreps_mid) for _ in range(mul * ir.dim)]
        self.register_buffer("_tp_rescale_vec", torch.tensor(rescale, dtype=torch.float32))

        # tile the per-channel gates over the (channel, direction) pairs of each path, see the docstring
        repeat, base = [], 0
        for ins in self.tp.instructions:
            mul_in = irreps_node[ins.i_in1].mul
            repeat += list(range(base, base + mul_in)) * irreps_edge[ins.i_in2].mul
            base += mul_in
        self.register_buffer("_weight_repeat_idx", torch.tensor(repeat, dtype=torch.long))

        self.weight_net = mlp(num_rbf + edge_scalars, hidden, base)
        bound = 1.0 / math.sqrt(hidden)
        self.weight_offset = nn.Parameter(torch.empty(base).uniform_(-bound, bound))

    def forward(self, h_src: Tensor, edge_sh: Tensor, edge_dist: Tensor, edge_attr: Tensor) -> Tensor:
        w_in = torch.cat([self.rbf(edge_dist), edge_attr], dim=-1)
        weights = (self.weight_net(w_in) + self.weight_offset)[:, self._weight_repeat_idx]
        return self.proj(self.tp(h_src, edge_sh, weights) * self._tp_rescale_vec)


class CGLayer(nn.Module):
    """One SO3brep layer: CG messages, invariant multi-head attention, gated FFN, channel dropout."""

    def __init__(self, irreps: Irreps, irreps_edge: Irreps, *, heads: int, attention: bool,
                 edge_scalars: int, radial_hidden: int, num_rbf: int, channel_dropout: float):
        super().__init__()
        self.heads, self.attention = heads, attention
        self.msg_block = TensorProductMessage(irreps, irreps_edge, edge_scalars, radial_hidden, num_rbf)
        self._n_scalar = irreps[0].mul
        self._blocks, start = [], 0
        for mul, ir in irreps:
            if mul % heads:
                raise ValueError(f"{mul}x{ir} is not divisible into {heads} heads")
            self._blocks.append((start, start + mul * ir.dim, mul, ir.dim))
            start += mul * ir.dim
        if attention:
            self.scalar_q_proj = nn.Linear(self._n_scalar, heads, bias=False)
            self.scalar_k_proj = nn.Linear(self._n_scalar, heads, bias=False)
            self._scale = float(self._n_scalar) ** -0.5
        self.norm = EquivariantRMSNorm(irreps)
        self.value_proj = Linear(irreps, irreps)
        self.norm2 = EquivariantRMSNorm(irreps)
        self.ffn = GatedFFN(irreps)
        self.equiv_dropout = EquivariantDropout(irreps, channel_dropout) if channel_dropout > 0 else None

    def _weight_heads(self, msg: Tensor, attn: Tensor) -> Tensor:
        """Each head scales an equal slice of the channels of every irrep block."""
        out = torch.empty_like(msg)
        for s, e, mul, d in self._blocks:
            chunk = msg[:, s:e].reshape(-1, self.heads, mul // self.heads, d) * attn[:, :, None, None]
            out[:, s:e] = chunk.reshape(-1, mul * d)
        return out

    def forward(self, x: Tensor, edge_index: Tensor, edge_sh: Tensor, edge_dist: Tensor,
                edge_attr: Tensor) -> Tensor:
        src, dst = edge_index
        h = self.norm(x)
        msg = self.msg_block(h[src], edge_sh, edge_dist, edge_attr)
        if self.attention:
            hs = h[:, :self._n_scalar]
            logits = self.scalar_q_proj(hs)[dst] * self.scalar_k_proj(hs)[src] * self._scale
            msg = self._weight_heads(msg, scatter_softmax(logits, dst, x.shape[0]))
        agg = x.new_zeros(x.shape).scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        x = x + self.value_proj(agg)
        x = x + self.ffn(self.norm2(x))
        if self.equiv_dropout is not None:
            x = self.equiv_dropout(x)
        return x


class ScalarReadout(nn.Module):
    """Keeps the l=0 channels only, so the output is invariant."""

    def __init__(self, n_scalar: int, out_dim: int):
        super().__init__()
        self.n_scalar = n_scalar
        self.readout_proj = nn.Sequential(
            nn.BatchNorm1d(n_scalar), nn.Linear(n_scalar, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.readout_proj(h[:, :self.n_scalar])


# ----------------------------------------------------------------------------- the stack

class MessagePassing(nn.Module):
    """Edge geometry, input projection, the layers, readout and graph pooling.

    Edge directions (Eq. 5), all in the world frame: centroid to centroid, sender centroid to the
    curve midpoint, midpoint to the receiver centroid, and the chord of the curve. Their harmonics
    are the edge irreps of Eq. (6).

    Ablation switches:
        use_edge_features=False  keeps only the constant l=0 harmonic and a constant distance;
        use_node_features=False  replaces the face features by learned constants;
        zero_edge_harmonics=True zeros all edge harmonics (the invariant variant of Tab. 3, whose
                                 directions enter as invariant scalars instead).
    """

    def __init__(self, n_scalar: int, n_vector: int, n_tensor: int, n_vector_in: int, n_tensor_in: int,
                 *, layers: int, out_dim: int, use_edge_features: bool = True,
                 use_node_features: bool = True, zero_edge_harmonics: bool = False):
        super().__init__()
        self.n_scalar = n_scalar
        self.irreps_node = node_irreps(n_scalar, n_vector, n_tensor)
        self.l_max = 2 if n_tensor else (1 if n_vector else 0)
        self.irreps_edge = edge_irreps(self.l_max, 4)
        self.input_proj = Linear(node_irreps(n_scalar, n_vector_in, n_tensor_in), self.irreps_node)
        self.use_edge_features = use_edge_features
        self.use_node_features = use_node_features
        self.zero_edge_harmonics = zero_edge_harmonics
        self.n_layers = layers
        if not use_node_features:
            self.node_const = nn.ParameterList([nn.Parameter(torch.zeros(n_scalar)) for _ in range(layers)])

    def edge_harmonics(self, pos: Tensor, edge_index: Tensor, edge_curve_pts: Tensor,
                       edge_mid: Tensor) -> tuple[Tensor, Tensor]:
        """(E, irreps_edge.dim) harmonics and (E, 1) centroid distances."""
        src, dst = edge_index
        rel = pos[src] - pos[dst]
        dist = rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ls = list(range(self.l_max + 1))
        base = spherical_harmonics(ls, unit_direction(rel, pos[src], pos[dst]), normalize=True,
                                   normalization="component")
        if not self.use_edge_features:  # topology only: the constant l=0 harmonic, unit distance
            edge_sh = base.new_zeros(base.shape[0], self.irreps_edge.dim)
            edge_sh[:, :1] = base[:, :1]
            return edge_sh, torch.ones_like(dist)

        blocks, start = {}, 0
        for l in ls:
            blocks[l] = [base[:, start:start + 2 * l + 1]]
            start += 2 * l + 1
        mid, first, last = edge_mid, edge_curve_pts[:, 0, :3], edge_curve_pts[:, -1, :3]
        dirs = torch.stack([
            unit_direction(mid - pos[src], mid, pos[src]),
            unit_direction(pos[dst] - mid, pos[dst], mid),
            unit_direction(last - first, last, first),
        ])  # (3, E, 3)
        for l in ls[1:]:
            sh = spherical_harmonics([l], dirs.reshape(-1, 3), normalize=True, normalization="component")
            blocks[l] += list(sh.reshape(3, -1, 2 * l + 1))
        edge_sh = torch.cat([b for l in ls for b in blocks[l]], dim=-1)
        if self.zero_edge_harmonics:
            edge_sh = torch.zeros_like(edge_sh)
        return edge_sh, dist

    def forward(self, node_feats: Tensor, pos: Tensor, edge_index: Tensor, batch: Tensor, edge_attr: Tensor,
                edge_curve_pts: Tensor, edge_mid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns per-face embeddings, the graph embedding [max-pool, mean-pool] and the last node irreps."""
        edge_sh, dist = self.edge_harmonics(pos, edge_index, edge_curve_pts, edge_mid)
        h = self.input_proj(node_feats)
        for i, layer in enumerate(self.layers):
            if not self.use_node_features:
                h = h.new_zeros(h.shape)
                h[:, :self.n_scalar] = self.node_const[i]
            h = layer(h, edge_index, edge_sh, dist, edge_attr)
        node_emb = self.readout(h)
        graph_emb = torch.cat([global_max_pool(node_emb, batch), global_mean_pool(node_emb, batch)], dim=-1)
        return node_emb, graph_emb, h


class CGMessagePassing(MessagePassing):
    """The SO3brep network body: a stack of `CGLayer`s with a scalar readout."""

    def __init__(self, n_scalar: int, n_vector: int, n_tensor: int, n_vector_in: int, n_tensor_in: int, *,
                 layers: int, heads: int, attention: bool, out_dim: int, edge_scalars: int,
                 radial_hidden: int, num_rbf: int, channel_dropout: float,
                 use_edge_features: bool = True, use_node_features: bool = True):
        super().__init__(n_scalar, n_vector, n_tensor, n_vector_in, n_tensor_in, layers=layers, out_dim=out_dim,
                         use_edge_features=use_edge_features,
                         use_node_features=use_node_features)
        self.layers = nn.ModuleList([
            CGLayer(self.irreps_node, self.irreps_edge, heads=heads, attention=attention,
                    edge_scalars=edge_scalars, radial_hidden=radial_hidden, num_rbf=num_rbf,
                    channel_dropout=channel_dropout)
            for _ in range(layers)
        ])
        self.readout = ScalarReadout(n_scalar, out_dim)
