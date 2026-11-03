"""CNN encoder for the UV grid of a face.

A face is sampled on a regular H x W grid of its (u, v) parameter domain. Each
grid point carries 7 channels: position (3), surface normal (3) and a trim mask
(1). The encoder is three 3x3 convolutions followed by global average pooling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """3x3 convolution, batch norm, LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


class SurfaceEncoder(nn.Module):
    """UV-grid CNN: (N, C, H, W) -> (N, output_dims).

    The xyz channels are divided by the diagonal of the face's bounding box (in its local frame) and
    the log of that diagonal is appended to the pooled features, so the CNN sees shape and size
    separately.
    """

    def __init__(self, in_channels: int = 7, output_dims: int = 64):
        super().__init__()
        self.conv1 = conv_block(in_channels, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(128 + 1, output_dims, bias=False), nn.BatchNorm1d(output_dims),
                                nn.LeakyReLU(), nn.Dropout(0.1))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        n = x.size(0)
        xyz = x[:, :3].reshape(n, 3, -1)
        diag = torch.norm(xyz.max(dim=-1)[0] - xyz.min(dim=-1)[0], dim=1, keepdim=True).clamp(min=1e-6)
        x = torch.cat([x[:, :3] / diag.view(n, 1, 1, 1), x[:, 3:]], dim=1)
        feats = self.final_pool(self.conv3(self.conv2(self.conv1(x)))).view(n, -1)
        return self.fc(torch.cat([feats, torch.log(diag)], dim=-1))
