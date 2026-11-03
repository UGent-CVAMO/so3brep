"""Non-equivariant message functions on the same inputs as SO3brep (paper Sec. 4.3, Tab. 3).

The node and edge features are those of SO3brep, flattened. The CG tensor product is replaced by
an MLP on the concatenation [h_src, edge harmonics, rbf(distance), edge scalars], optionally with
multi-head attention over all channels, and the readout uses all channels:

    attention   concat-MLP messages + dot-product attention, world-frame features;
    concat      concat-MLP messages, summed, world-frame features;
    invariant   concat-MLP messages, summed; node features stay in their local frames and the four
                edge directions enter as scalars expressed in the sender's frame, so the model is
                rotation-invariant (see `models.so3brep.SO3brep._sender_frame_directions`).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.gnn import GaussianSmearing, MessagePassing, mlp, scatter_softmax


class ConcatMessage(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, edge_scalars: int, hidden: int, num_rbf: int):
        super().__init__()
        self.rbf = GaussianSmearing(num_rbf)
        self.mlp = mlp(node_dim + edge_dim + num_rbf + edge_scalars, hidden, node_dim)

    def forward(self, h_src: Tensor, edge_sh: Tensor, edge_dist: Tensor, edge_attr: Tensor) -> Tensor:
        return self.mlp(torch.cat([h_src, edge_sh, self.rbf(edge_dist), edge_attr], dim=-1))


class MatchedLayer(nn.Module):
    """Same block structure as `CGLayer`, with plain LayerNorm, MLP messages and an MLP feed-forward."""

    def __init__(self, dim: int, edge_dim: int, *, heads: int, attention: bool, edge_scalars: int,
                 radial_hidden: int, num_rbf: int, dropout: float):
        super().__init__()
        self.dim, self.heads, self.attention = dim, heads, attention
        self.msg_block = ConcatMessage(dim, edge_dim, edge_scalars, radial_hidden, num_rbf)
        self.norm = nn.LayerNorm(dim)
        self.value_proj = nn.Linear(dim, dim)
        if attention:
            self.head_dim = dim // heads
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            hidden = radial_hidden // 2
            self.edge_bias_net = nn.Sequential(nn.Linear(num_rbf + edge_scalars, hidden), nn.SiLU(),
                                               nn.Linear(hidden, heads))
            nn.init.zeros_(self.edge_bias_net[-1].weight)
            nn.init.zeros_(self.edge_bias_net[-1].bias)
            self.rbf = GaussianSmearing(num_rbf)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor, edge_index: Tensor, edge_sh: Tensor, edge_dist: Tensor,
                edge_attr: Tensor) -> Tensor:
        src, dst = edge_index
        h = self.norm(x)
        msg = self.msg_block(h[src], edge_sh, edge_dist, edge_attr)
        if self.attention:
            edge_inv = torch.cat([self.rbf(edge_dist), edge_attr], dim=-1)
            q = self.q_proj(h)[dst].view(-1, self.heads, self.head_dim)
            k = self.k_proj(h)[src].view(-1, self.heads, self.head_dim)
            logits = (q * k).sum(-1) * self.head_dim ** -0.5 + self.edge_bias_net(edge_inv)
            attn = scatter_softmax(logits, dst, x.shape[0])
            msg = (msg.view(-1, self.heads, self.head_dim) * attn.unsqueeze(-1)).reshape(-1, self.dim)
        agg = x.new_zeros(x.shape).scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        x = x + self.value_proj(agg)
        x = x + self.ffn(self.norm2(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class FlatReadout(nn.Module):
    """Readout over all channels (the matched models make no invariance promise)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(in_dim), nn.Linear(in_dim, out_dim), nn.SiLU(),
                                 nn.Linear(out_dim, out_dim))

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class MatchedMessagePassing(MessagePassing):
    def __init__(self, n_scalar: int, n_vector: int, n_tensor: int, n_vector_in: int, n_tensor_in: int, *,
                 layers: int, heads: int, attention: bool, out_dim: int, edge_scalars: int,
                 radial_hidden: int, num_rbf: int, channel_dropout: float,
                 invariant: bool, use_edge_features: bool = True, use_node_features: bool = True):
        super().__init__(n_scalar, n_vector, n_tensor, n_vector_in, n_tensor_in, layers=layers, out_dim=out_dim,
                         use_edge_features=use_edge_features,
                         use_node_features=use_node_features, zero_edge_harmonics=invariant)
        dim = self.irreps_node.dim
        self.layers = nn.ModuleList([
            MatchedLayer(dim, self.irreps_edge.dim, heads=heads, attention=attention, edge_scalars=edge_scalars,
                         radial_hidden=radial_hidden, num_rbf=num_rbf, dropout=channel_dropout)
            for _ in range(layers)
        ])
        self.readout = FlatReadout(dim, out_dim)

