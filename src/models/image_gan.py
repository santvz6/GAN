"""
DCGAN for 128x128 silhouettes — WGAN-GP-compatible.

Generator:    z(noise_dim) -> FC -> reshape(8*base, 4, 4)
                            -> 5x ConvTranspose2d(stride=2) with LayerNorm + ReLU
                            -> final ConvT to image_size, Tanh

Discriminator: img(C, H, W) -> 4x Conv2d(stride=2) with LayerNorm + LeakyReLU
                            -> Flatten -> Linear -> 1 (no sigmoid; WGAN)

LayerNorm (not BatchNorm) is used in D because WGAN-GP's gradient penalty
breaks BatchNorm's per-sample independence assumption.
"""

import torch
import torch.nn as nn

from src.config.hparams import ImageHParams


def _layer_norm_2d(num_channels: int, h: int, w: int) -> nn.LayerNorm:
    return nn.LayerNorm([num_channels, h, w])


class ImageGenerator(nn.Module):
    def __init__(self, hparams: ImageHParams = None):
        super().__init__()
        self.hp = hparams or ImageHParams()
        base = self.hp.g_base_ch
        C = self.hp.channels

        self.fc = nn.Linear(self.hp.noise_dim, base * 8 * 4 * 4)

        def up(in_c, out_c, h_out, w_out):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                _layer_norm_2d(out_c, h_out, w_out),
                nn.ReLU(inplace=True),
            )

        self.up1 = up(base * 8, base * 4, 8,   8)    # 4 -> 8
        self.up2 = up(base * 4, base * 2, 16,  16)   # 8 -> 16
        self.up3 = up(base * 2, base,     32,  32)   # 16 -> 32
        self.up4 = up(base,     base // 2, 64, 64)   # 32 -> 64

        self.out = nn.Sequential(
            nn.ConvTranspose2d(base // 2, C, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.hp.g_base_ch * 8, 4, 4)
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        return self.out(x)


class ImageDiscriminator(nn.Module):
    def __init__(self, hparams: ImageHParams = None):
        super().__init__()
        self.hp = hparams or ImageHParams()
        base = self.hp.d_base_ch
        C = self.hp.channels
        S = self.hp.image_size

        def down(in_c, out_c, h_out, w_out, use_norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_norm)]
            if use_norm:
                layers.append(_layer_norm_2d(out_c, h_out, w_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # 128 -> 64 -> 32 -> 16 -> 8
        h = S
        self.d1 = down(C,       base,     h//2,  h//2,  use_norm=False); h //= 2
        self.d2 = down(base,    base * 2, h//2,  h//2);                  h //= 2
        self.d3 = down(base*2,  base * 4, h//2,  h//2);                  h //= 2
        self.d4 = down(base*4,  base * 8, h//2,  h//2);                  h //= 2

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 8 * h * h, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.d1(x); x = self.d2(x); x = self.d3(x); x = self.d4(x)
        return self.head(x)


if __name__ == "__main__":
    hp = ImageHParams()
    g = ImageGenerator(hp)
    d = ImageDiscriminator(hp)
    z = torch.randn(4, hp.noise_dim)
    fake = g(z)
    print(f"G(z): {tuple(fake.shape)}  | D(fake): {tuple(d(fake).shape)}")
    print(f"G params: {sum(p.numel() for p in g.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in d.parameters()):,}")
