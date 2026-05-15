"""DCGAN-style generator that maps z -> (C, H, W) image."""
import math
import torch
import torch.nn as nn

from src.config.hparams import ImgHParams


class ImgGenerator(nn.Module):
    """
    Architecture (default 64x64, 1 channel):
        z (B, noise_dim, 1, 1)
        -> ConvT 4x4, fm*8 -> 4x4
        -> ConvT 4x4 s2, fm*4 -> 8x8
        -> ConvT 4x4 s2, fm*2 -> 16x16
        -> ConvT 4x4 s2, fm   -> 32x32
        -> ConvT 4x4 s2, C    -> 64x64, tanh
    """

    def __init__(self, hparams: ImgHParams | None = None):
        super().__init__()
        self.hp = hparams or ImgHParams()
        nz = self.hp.noise_dim
        fm = self.hp.g_feature_maps
        C = self.hp.channels
        size = self.hp.image_size

        # Total doublings to go from 4x4 to image_size
        n_doublings = int(math.log2(size)) - 2
        if 2 ** (n_doublings + 2) != size:
            raise ValueError(f"image_size must be a power of 2 >= 8, got {size}")

        layers: list[nn.Module] = []
        peak = fm * (2 ** (n_doublings - 1))  # e.g. fm*8 when size=64

        # Block 0: 1x1 -> 4x4 (stride 1, no padding)
        layers += [
            nn.ConvTranspose2d(nz, peak, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(peak),
            nn.ReLU(True),
        ]
        in_ch = peak

        # Middle blocks: each doubles resolution and halves channels
        for i in range(n_doublings - 2, -1, -1):
            out_ch = fm * (2 ** i)
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
            in_ch = out_ch

        # Final: in_ch -> C at the target resolution, tanh to [-1,1]
        layers += [
            nn.ConvTranspose2d(in_ch, C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)  # (B, nz) -> (B, nz, 1, 1)
        return self.net(z)


if __name__ == "__main__":
    g = ImgGenerator()
    z = torch.randn(4, g.hp.noise_dim)
    out = g(z)
    total = sum(p.numel() for p in g.parameters())
    print(f"ImgGenerator output: {out.shape}, params: {total:,}")
