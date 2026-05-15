"""
Entrenamiento del Mesh GAN (3D) con WGAN-GP.

Datos:
  - NOMO3D: ~356 scans OBJ subsampled a 6890 puntos
  - Mallas SMPL sintéticas generadas por el Tabular GAN (si existen)

Hiperparámetros:
  - Batch size: 16 (menor por uso de memoria con 6890 puntos)
  - Learning rate: 1e-4 (Adam, beta1=0.0, beta2=0.9)
  - Epochs: 200
  - Ratio D/G: 5:1
  - Lambda GP: 10
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from pathlib import Path
import json

from src.config.paths import Paths
from src.models.mesh_gan import Generator, PointNetDiscriminator, compute_gradient_penalty
from src.data.nomo3d_loader import NOMO3DDataset


# Hiperparámetros
LATENT_DIM = 128
BATCH_SIZE = 16
LR = 1e-4
BETAS = (0.0, 0.9)
N_EPOCHS = 200
N_CRITIC = 5
LAMBDA_GP = 10.0
CHECKPOINT_EVERY = 50
N_POINTS = 6890
SEED = 42


class MeshNPZDataset(torch.utils.data.Dataset):
    """Carga mallas guardadas como .npz (6890, 3)."""

    def __init__(self, meshes_dir: str):
        self.files = sorted(Path(meshes_dir).glob('*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        verts = data['vertices'].astype(np.float32)  # (6890, 3)
        return torch.tensor(verts)


def get_mesh_loader(batch_size: int = BATCH_SIZE) -> DataLoader:
    """Combina NOMO3D + mallas sintéticas SMPL."""
    datasets = []

    # NOMO3D
    nomo = NOMO3DDataset()
    if len(nomo) > 0:
        datasets.append(nomo)

    # Mallas SMPL sintéticas (si existen)
    meshes_dir = Paths.MESHES_DIR
    if meshes_dir.exists() and len(list(meshes_dir.glob('*.npz'))) > 0:
        smpl_ds = MeshNPZDataset(str(meshes_dir))
        datasets.append(smpl_ds)
        print(f"[MESH] Añadidas {len(smpl_ds)} mallas SMPL sintéticas.")

    if not datasets:
        raise ValueError("No se encontraron mallas para entrenar.")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[MESH] Dataset total: {len(combined)} mallas")

    return DataLoader(combined, batch_size=batch_size, shuffle=True, drop_last=True)


def train_mesh():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MESH] Dispositivo: {device}")

    loader = get_mesh_loader()

    G = Generator(latent_dim=LATENT_DIM, n_points=N_POINTS).to(device)
    D = PointNetDiscriminator(n_points=N_POINTS).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    ckpt_dir = Paths.MESH_EXP_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = {'d_loss': [], 'g_loss': [], 'gp': []}

    print(f"[MESH] Iniciando entrenamiento — {N_EPOCHS} épocas...")
    for epoch in range(1, N_EPOCHS + 1):
        d_losses, g_losses, gps = [], [], []

        for real in loader:
            if isinstance(real, (list, tuple)):
                real = real[0]
            real = real.to(device)   # (bs, 6890, 3)
            bs = real.size(0)

            # ---- Paso D ----
            for _ in range(N_CRITIC):
                z = torch.randn(bs, LATENT_DIM, device=device)
                fake = G(z).detach()

                d_real = D(real).mean()
                d_fake = D(fake).mean()
                gp = compute_gradient_penalty(D, real, fake, device, LAMBDA_GP)
                d_loss = d_fake - d_real + gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            d_losses.append(d_loss.item())
            gps.append(gp.item())

            # ---- Paso G ----
            z = torch.randn(bs, LATENT_DIM, device=device)
            g_loss = -D(G(z)).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_losses.append(g_loss.item())

        d_mean = np.mean(d_losses)
        g_mean = np.mean(g_losses)
        gp_mean = np.mean(gps)
        log['d_loss'].append(d_mean)
        log['g_loss'].append(g_mean)
        log['gp'].append(gp_mean)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Época [{epoch:3d}/{N_EPOCHS}] "
                  f"D: {d_mean:.4f} | G: {g_mean:.4f} | GP: {gp_mean:.4f}")

        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS:
            ckpt_path = ckpt_dir / f'checkpoint_ep{epoch:03d}.pt'
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
            }, str(ckpt_path))
            print(f"  Checkpoint guardado: {ckpt_path}")

    with open(str(ckpt_dir / 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f"[MESH] Entrenamiento completado. Checkpoints en {ckpt_dir}")
    return G, D


if __name__ == "__main__":
    train_mesh()
