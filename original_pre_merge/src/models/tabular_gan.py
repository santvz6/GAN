"""
Tabular GAN con WGAN-GP para generar coordenadas 3D de joints.

Dimensiones:
  - Input del Generador: z ∈ R^128 (ruido latente)
  - Output del Generador: joints ∈ R^72 (24 joints × 3 coords)
  - Input del Discriminador: joints ∈ R^72
  - Output del Discriminador: escalar (sin activación — WGAN)

Loss WGAN-GP:
  L_D = E[D(fake)] - E[D(real)] + lambda * gradient_penalty
  L_G = -E[D(fake)]
"""

import torch
import torch.nn as nn


LATENT_DIM = 128
JOINTS_DIM = 72


class Generator(nn.Module):
    """
    Genera vectores de 72 dimensiones (joints) desde ruido z.

    Arquitectura: MLP con Batch Normalization y LeakyReLU.
    La salida usa Tanh → valores en [-1, 1].
    """

    def __init__(self, latent_dim: int = LATENT_DIM, output_dim: int = JOINTS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            joints: (batch, 72)
        """
        return self.net(z)


class Discriminator(nn.Module):
    """
    Clasifica si un vector de joints es real o falso.

    Arquitectura: MLP con LeakyReLU y Dropout.
    Sin activación final (WGAN — salida no acotada).
    """

    def __init__(self, input_dim: int = JOINTS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 72)
        Returns:
            score: (batch, 1) — sin activación (WGAN)
        """
        return self.net(x)


def compute_gradient_penalty(D: nn.Module, real: torch.Tensor,
                              fake: torch.Tensor, device: torch.device,
                              lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Calcula el gradient penalty de WGAN-GP.

    Fuerza al discriminador a ser 1-Lipschitz interpolando
    entre muestras reales y falsas.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)

    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interp = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return penalty
