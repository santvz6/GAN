"""WGAN-GP-friendly image discriminator (a.k.a. critic).

No BatchNorm — uses LayerNorm via GroupNorm(num_groups=1) which is safe under
gradient penalty, since BN couples samples in a batch and breaks GP's per-sample
gradient assumption.
"""
import math
import torch
import torch.nn as nn

from src.config.hparams import ImgHParams


def _layer_norm(num_channels: int) -> nn.Module:
    # GroupNorm with 1 group == LayerNorm over (C, H, W)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=True)


class ImgDiscriminator(nn.Module):
    """
    Architecture (default 64x64, 1 channel):
        x (B, C, 64, 64)
        -> Conv 4x4 s2, fm    -> 32x32
        -> Conv 4x4 s2, fm*2  -> 16x16
        -> Conv 4x4 s2, fm*4  ->  8x8
        -> Conv 4x4 s2, fm*8  ->  4x4
        -> Conv 4x4 s1, 1     ->  1x1
        flatten -> scalar (no sigmoid)
    """

    def __init__(self, hparams: ImgHParams | None = None):
        super().__init__()
        self.hp = hparams or ImgHParams()
        C = self.hp.channels
        fm = self.hp.d_feature_maps
        size = self.hp.image_size

        n_down = int(math.log2(size)) - 2
        if 2 ** (n_down + 2) != size:
            raise ValueError(f"image_size must be a power of 2 >= 8, got {size}")

        layers: list[nn.Module] = []
        in_ch = C

        # First conv: no norm (standard DCGAN/WGAN convention)
        out_ch = fm
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = out_ch

        # Downsample blocks
        for i in range(1, n_down):
            out_ch = fm * (2 ** i)
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                _layer_norm(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch

        # Final 4x4 conv -> 1x1
        layers += [nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=0)]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), 1)


if __name__ == "__main__":
    d = ImgDiscriminator()
    x = torch.randn(4, d.hp.channels, d.hp.image_size, d.hp.image_size)
    out = d(x)
    total = sum(p.numel() for p in d.parameters())
    print(f"ImgDiscriminator output: {out.shape}, params: {total:,}")
