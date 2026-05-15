"""
Filtra cuerpos generados por el Tabular GAN usando el discriminador.

Criterio: score_D > 0.8 (sigmoid del output del discriminador WGAN-GP)

Pipeline:
  1. Carga checkpoint del Generador y Discriminador
  2. Genera lotes z ~ N(0,1)
  3. G(z) → joints (batch, 72)
  4. D(joints).sigmoid() → scores
  5. Conserva los que superen el umbral
  6. Guarda en internal/data/generated_joints.npz
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from src.config.paths import Paths
from src.models.tabular_gan import Generator, Discriminator


LATENT_DIM = 128
THRESHOLD = 0.8
BATCH_SIZE = 512


def filter_generated_bodies(n_target: int = 1000,
                             checkpoint_path: str = None,
                             out_path: str = None,
                             threshold: float = THRESHOLD):
    """
    Genera n_target cuerpos que superen el umbral del discriminador.

    Args:
        n_target: número de cuerpos a generar
        checkpoint_path: ruta al checkpoint del Tabular GAN
        out_path: ruta de salida para el npz
        threshold: umbral mínimo de score_D (default 0.8)
    """
    if checkpoint_path is None:
        checkpoint_path = str(Paths.TABULAR_CHECKPOINT)

    if out_path is None:
        out_path = str(Paths.GENERATED_JOINTS_NPZ)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[FILTER] Usando dispositivo: {device}")

    # Cargar modelos
    G = Generator(latent_dim=LATENT_DIM).to(device)
    D = Discriminator().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    G.eval()
    D.eval()

    accepted = []
    total_generated = 0

    print(f"[FILTER] Generando hasta tener {n_target} cuerpos con score_D > {threshold}...")
    with torch.no_grad():
        while len(accepted) < n_target:
            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
            fake = G(z)                                 # (batch, 72)
            score = torch.sigmoid(D(fake))              # (batch, 1)
            mask = (score.squeeze() > threshold)
            accepted_batch = fake[mask].cpu().numpy()
            accepted.extend(accepted_batch)
            total_generated += BATCH_SIZE

            if total_generated % 10000 == 0:
                rate = len(accepted) / total_generated * 100
                print(f"  Generados: {total_generated} | Aceptados: {len(accepted)} "
                      f"({rate:.1f}%) | Objetivo: {n_target}")

    joints = np.array(accepted[:n_target], dtype=np.float32)
    print(f"[FILTER] {len(joints)} cuerpos aceptados de {total_generated} generados.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, joints=joints)
    print(f"[FILTER] Guardado en {out_path}")

    return joints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filtra cuerpos generados por score_D')
    parser.add_argument('--n', type=int, default=1000,
                        help='Número de cuerpos a generar (default: 1000)')
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help='Umbral mínimo score_D (default: 0.8)')
    args = parser.parse_args()

    filter_generated_bodies(n_target=args.n, threshold=args.threshold)
