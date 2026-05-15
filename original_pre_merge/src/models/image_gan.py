"""
DCGAN para generar imágenes 128×128 de esqueletos humanos.

Usa convoluciones transpuestas (upsampling) en el generador
y convoluciones con stride 2 en el discriminador.

Input del Generador: z ∈ R^128
Output del Generador: imagen (3, 128, 128), valores en [-1, 1]
"""

import torch
import torch.nn as nn


LATENT_DIM = 128
IMAGE_CHANNELS = 3
IMAGE_SIZE = 128


def _make_gen_block(in_ch, out_ch, kernel=4, stride=2, padding=1, use_bn=True):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _make_disc_block(in_ch, out_ch, kernel=4, stride=2, padding=1, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """
    DCGAN Generator: z(128) → imagen(3, 128, 128)

    Upsampling progresivo desde 4×4 hasta 128×128:
    4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        ngf = 64  # feature maps base

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # Ajuste final: salida exacta 128×128
        self.conv_blocks = nn.Sequential(
            _make_gen_block(ngf * 8, ngf * 8),    # 4→8
            _make_gen_block(ngf * 8, ngf * 4),    # 8→16
            _make_gen_block(ngf * 4, ngf * 2),    # 16→32
            _make_gen_block(ngf * 2, ngf),         # 32→64
            nn.ConvTranspose2d(ngf, IMAGE_CHANNELS, 4, 2, 1),  # 64→128
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, 128)
        Returns:
            img: (batch, 3, 128, 128)
        """
        ngf = 64
        x = self.fc(z)
        x = x.view(x.size(0), ngf * 8, 4, 4)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator: imagen(3, 128, 128) → escalar

    Downsampling progresivo desde 128×128 hasta 4×4.
    """

    def __init__(self):
        super().__init__()
        ndf = 64

        self.net = nn.Sequential(
            # No BN en la primera capa
            nn.Conv2d(IMAGE_CHANNELS, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),          # 128→64

            _make_disc_block(ndf, ndf * 2),            # 64→32
            _make_disc_block(ndf * 2, ndf * 4),        # 32→16
            _make_disc_block(ndf * 4, ndf * 8),        # 16→8
            _make_disc_block(ndf * 8, ndf * 8),        # 8→4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),  # 4→1
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (batch, 3, 128, 128)
        Returns:
            score: (batch, 1)
        """
        return self.net(img).view(-1, 1)


def compute_gradient_penalty(D: nn.Module, real: torch.Tensor,
                              fake: torch.Tensor, device: torch.device,
                              lambda_gp: float = 10.0) -> torch.Tensor:
    """Gradient penalty WGAN-GP para imágenes."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real)

    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
