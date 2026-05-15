"""
GAN para generar nubes de puntos 3D (6890 puntos, topología SMPL).

Generador: MLP z→(6890×3)
Discriminador: PointNet-style (max-pool global sobre features por punto)
"""

import torch
import torch.nn as nn


LATENT_DIM = 128
N_POINTS = 6890


class Generator(nn.Module):
    """
    Genera nubes de puntos 3D desde vector latente.

    z(128) → MLP → reshape → (6890, 3)
    """

    def __init__(self, latent_dim: int = LATENT_DIM, n_points: int = N_POINTS):
        super().__init__()
        self.n_points = n_points

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),

            nn.Linear(8192, n_points * 3),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, 128)
        Returns:
            points: (batch, n_points, 3)
        """
        x = self.net(z)
        return x.view(x.size(0), self.n_points, 3)


class PointNetDiscriminator(nn.Module):
    """
    Discriminador estilo PointNet.

    Procesa cada punto independientemente, luego agrega con max-pool global.
    Entrada: (batch, 6890, 3)
    Salida: (batch, 1) — sin activación (WGAN)
    """

    def __init__(self, n_points: int = N_POINTS):
        super().__init__()

        # Capas por punto (shared MLP)
        self.point_net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # MLP global (después del max-pool)
        self.global_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (batch, n_points, 3)
        Returns:
            score: (batch, 1)
        """
        x = points.transpose(1, 2)       # (batch, 3, n_points)
        x = self.point_net(x)             # (batch, 512, n_points)
        x = x.max(dim=2)[0]              # (batch, 512) — max global pooling
        return self.global_net(x)


def compute_gradient_penalty(D: nn.Module, real: torch.Tensor,
                              fake: torch.Tensor, device: torch.device,
                              lambda_gp: float = 10.0) -> torch.Tensor:
    """Gradient penalty WGAN-GP para nubes de puntos."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device).expand_as(real)

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
