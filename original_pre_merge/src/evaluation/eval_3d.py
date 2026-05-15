"""
Evaluación del Mesh GAN en 3D.

Métricas:
  1. Chamfer Distance — distancia bidireccional entre nubes de puntos.
     Para cada punto en A, encuentra el más cercano en B y viceversa.
     CD ≈ 0 → mallas muy similares.

  2. F-Score@τ — fracción de puntos que tienen un vecino más cercano
     en la otra nube a distancia ≤ τ.
     F-score@0.01 → precisión y recall a 1cm de tolerancia.
"""

import numpy as np
import torch
from pathlib import Path

from src.config.paths import Paths
from src.models.mesh_gan import Generator, PointNetDiscriminator
from src.data.nomo3d_loader import NOMO3DDataset


LATENT_DIM = 128
N_POINTS = 6890
N_SAMPLES = 100
TAU = 0.01  # tolerancia F-score en metros


def chamfer_distance_numpy(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calcula Chamfer Distance entre dos nubes de puntos.

    CD(A,B) = mean_{a∈A} min_{b∈B} ||a-b||² + mean_{b∈B} min_{a∈A} ||a-b||²

    Args:
        A: (n, 3)
        B: (m, 3)

    Returns:
        cd: float
    """
    # Distancias A→B
    diff_ab = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (n, m, 3)
    dists_ab = np.sum(diff_ab ** 2, axis=-1)              # (n, m)
    cd_ab = dists_ab.min(axis=1).mean()

    # Distancias B→A
    cd_ba = dists_ab.min(axis=0).mean()

    return float(cd_ab + cd_ba)


def fscore_numpy(A: np.ndarray, B: np.ndarray, tau: float = TAU) -> float:
    """
    Calcula F-Score@τ entre dos nubes de puntos.

    Precision: fracción de puntos en A con vecino en B a distancia ≤ τ
    Recall: fracción de puntos en B con vecino en A a distancia ≤ τ
    F = 2 * P * R / (P + R)
    """
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (n, m, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))         # (n, m)

    precision = (dists.min(axis=1) <= tau).mean()
    recall = (dists.min(axis=0) <= tau).mean()

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def evaluate_3d(checkpoint_path: str = None, n_samples: int = N_SAMPLES):
    """
    Evalúa el Mesh GAN comparando mallas reales (NOMO3D) y generadas.

    Returns:
        dict con métricas 'chamfer_distance' y 'fscore'
    """
    if checkpoint_path is None:
        checkpoint_path = str(Paths.MESH_CHECKPOINT)

    # Cargar mallas reales de NOMO3D
    nomo = NOMO3DDataset()
    n_real = min(n_samples, len(nomo))
    real_meshes = np.stack([nomo[i].numpy() for i in range(n_real)])  # (n, 6890, 3)

    # Generar mallas
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(latent_dim=LATENT_DIM, n_points=N_POINTS).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    G.eval()

    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=device)
        fake_meshes = G(z).cpu().numpy()  # (n, 6890, 3)

    # Calcular métricas por par (real_i, fake_i)
    cds, fscores = [], []
    for i in range(min(n_real, n_samples)):
        cd = chamfer_distance_numpy(real_meshes[i], fake_meshes[i])
        fs = fscore_numpy(real_meshes[i], fake_meshes[i], tau=TAU)
        cds.append(cd)
        fscores.append(fs)

    cd_mean = float(np.mean(cds))
    fs_mean = float(np.mean(fscores))

    print(f"\n[EVAL 3D]")
    print(f"  Chamfer Distance: {cd_mean:.6f}  (menor es mejor)")
    print(f"  F-Score@{TAU}:    {fs_mean:.4f}  (mayor es mejor, máx=1.0)")

    return {'chamfer_distance': cd_mean, 'fscore': fs_mean}


if __name__ == "__main__":
    evaluate_3d()
