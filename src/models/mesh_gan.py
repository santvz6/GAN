"""
Mesh GAN: generates point clouds of `num_points` 3D vertices directly.

Generator: MLP z -> (num_points * 3) -> reshape (num_points, 3)
Discriminator: PointNet-style per-point MLP + max-pool global feature + classifier head

PointNet uses 1D convolutions over points (permutation-equivariant) followed
by a max pool that yields a permutation-invariant global feature. The head
maps that feature to a scalar WGAN score (no sigmoid).
"""

import torch
import torch.nn as nn

from src.config.hparams import MeshHParams


class MeshGenerator(nn.Module):
    def __init__(self, hparams: MeshHParams = None):
        super().__init__()
        self.hp = hparams or MeshHParams()
        dims = self.hp.g_hidden_dims     # e.g. [1024, 4096]

        layers = []
        in_d = self.hp.noise_dim
        for out_d in dims:
            layers += [
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_d = out_d
        layers.append(nn.Linear(in_d, self.hp.num_points * 3))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return x.view(-1, self.hp.num_points, 3)


class PointNetDiscriminator(nn.Module):
    def __init__(self, hparams: MeshHParams = None):
        super().__init__()
        self.hp = hparams or MeshHParams()
        pn_dims   = self.hp.d_pn_dims    # e.g. [64, 128, 256]
        head_dims = self.hp.d_head_dims  # e.g. [256, 128]

        # Per-point MLP via Conv1d over (B, 3, N)
        pn_layers = []
        in_c = 3
        for out_c in pn_dims:
            pn_layers += [
                nn.Conv1d(in_c, out_c, kernel_size=1),
                nn.LayerNorm([out_c, self.hp.num_points]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = out_c
        self.pointnet = nn.Sequential(*pn_layers)
        self.global_feat_dim = in_c

        # Classifier head on the max-pooled global feature
        head = []
        in_d = self.global_feat_dim
        for out_d in head_dims:
            head += [nn.Linear(in_d, out_d), nn.LeakyReLU(0.2, inplace=True)]
            in_d = out_d
        head.append(nn.Linear(in_d, 1))  # no sigmoid — WGAN
        self.head = nn.Sequential(*head)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: (B, N, 3) -> (B, 3, N) for Conv1d
        x = pc.transpose(1, 2).contiguous()
        x = self.pointnet(x)              # (B, C, N)
        x = x.max(dim=2).values           # (B, C) global feature
        return self.head(x)               # (B, 1)


if __name__ == "__main__":
    hp = MeshHParams()
    g = MeshGenerator(hp)
    d = PointNetDiscriminator(hp)
    z = torch.randn(2, hp.noise_dim)
    pc = g(z)
    print(f"G(z): {tuple(pc.shape)}  | D(pc): {tuple(d(pc).shape)}")
    print(f"G params: {sum(p.numel() for p in g.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in d.parameters()):,}")
