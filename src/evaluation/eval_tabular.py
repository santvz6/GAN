"""
Evaluación del Tabular GAN.

Métricas:
  1. MMD (Maximum Mean Discrepancy) — mide similitud distribucional
     entre joints reales y generados. Usa kernel RBF de scipy.
     MMD ≈ 0 → distribuciones idénticas.

  2. Bone Length Error — error en longitud de huesos SMPL.
     Compara si los huesos generados tienen longitudes plausibles.
"""

import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cdist

from src.config.paths import Paths
from src.models.tabular_gan import Generator


LATENT_DIM = 128
N_SAMPLES = 1000

# Conexiones de huesos para calcular longitudes
SMPL_BONES = [
    (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (9, 14), (13, 16), (14, 17),
    (16, 18), (17, 19),
]


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray,
                    sigma: float = 1.0) -> float:
    """
    Calcula MMD con kernel RBF entre dos conjuntos de muestras.

    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Args:
        X: (n, d) muestras reales
        Y: (m, d) muestras generadas
        sigma: ancho de banda del kernel RBF

    Returns:
        mmd: float — valor de la distancia (≥ 0)
    """
    def rbf(A, B):
        dists = cdist(A, B, metric='sqeuclidean')
        return np.exp(-dists / (2 * sigma ** 2))

    k_xx = rbf(X, X).mean()
    k_yy = rbf(Y, Y).mean()
    k_xy = rbf(X, Y).mean()

    return float(k_xx - 2 * k_xy + k_yy)


def compute_bone_lengths(joints_flat: np.ndarray) -> np.ndarray:
    """
    Calcula longitudes de huesos para N poses.

    Args:
        joints_flat: (N, 72) joints aplanados

    Returns:
        lengths: (N, n_bones)
    """
    joints = joints_flat.reshape(-1, 24, 3)
    lengths = []
    for (i, j) in SMPL_BONES:
        diff = joints[:, i, :] - joints[:, j, :]
        length = np.linalg.norm(diff, axis=-1)
        lengths.append(length)
    return np.stack(lengths, axis=1)  # (N, n_bones)


def evaluate_tabular(checkpoint_path: str = None, n_samples: int = N_SAMPLES):
    """
    Evalúa el Tabular GAN comparando distribución real vs generada.

    Returns:
        dict con métricas 'mmd' y 'bone_length_mae'
    """
    if checkpoint_path is None:
        checkpoint_path = str(Paths.TABULAR_CHECKPOINT)

    # Cargar joints reales
    joints_path = Paths.JOINTS_NPZ
    if not joints_path.exists():
        raise FileNotFoundError(f"Joints reales no encontrados: {joints_path}")

    real_data = np.load(str(joints_path))
    real_joints = real_data['joints']

    # Subsample si hay muchos
    idx = np.random.choice(len(real_joints), min(n_samples, len(real_joints)), replace=False)
    real_joints = real_joints[idx]

    # Generar muestras
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(latent_dim=LATENT_DIM).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    G.eval()

    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=device)
        fake_joints = G(z).cpu().numpy()

    # MMD
    mmd = compute_mmd_rbf(real_joints[:500], fake_joints[:500])

    # Bone length error
    real_lengths = compute_bone_lengths(real_joints)
    fake_lengths = compute_bone_lengths(fake_joints)
    bone_mae = float(np.abs(real_lengths.mean(0) - fake_lengths.mean(0)).mean())

    print(f"\n[EVAL TABULAR]")
    print(f"  MMD (RBF):          {mmd:.6f}  (menor es mejor)")
    print(f"  Bone Length MAE:    {bone_mae:.6f}  (menor es mejor)")

    return {'mmd': mmd, 'bone_length_mae': bone_mae}


if __name__ == "__main__":
    evaluate_tabular()
