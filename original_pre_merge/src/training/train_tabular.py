"""
Entrenamiento del Tabular GAN con WGAN-GP.

Hiperparámetros:
  - Batch size: 64
  - Learning rate: 1e-4 (Adam, beta1=0.0, beta2=0.9)
  - Epochs: 200
  - Ratio D/G: 5 pasos del discriminador por 1 del generador
  - Lambda GP: 10

Datos: joints.npz (N, 72) — joints normalizados de AMASS
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

from src.config.paths import Paths
from src.models.tabular_gan import Generator, Discriminator, compute_gradient_penalty


# Hiperparámetros
LATENT_DIM = 128
BATCH_SIZE = 64
LR = 1e-4
BETAS = (0.0, 0.9)
N_EPOCHS = 200
N_CRITIC = 5         # pasos D por paso G
LAMBDA_GP = 10.0
CHECKPOINT_EVERY = 50
SEED = 42


def train_tabular():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TABULAR] Dispositivo: {device}")

    # Cargar datos
    joints_path = Paths.JOINTS_NPZ
    if not joints_path.exists():
        raise FileNotFoundError(
            f"No encontrado: {joints_path}\n"
            "Ejecuta primero: python src/data/preprocess_joints.py"
        )

    data = np.load(str(joints_path))
    joints = torch.tensor(data['joints'], dtype=torch.float32)
    print(f"[TABULAR] Datos cargados: {joints.shape}")

    dataset = TensorDataset(joints)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Modelos
    G = Generator(latent_dim=LATENT_DIM).to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    # Directorio de checkpoints
    ckpt_dir = Paths.TABULAR_EXP_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = {'d_loss': [], 'g_loss': [], 'gp': []}

    print(f"[TABULAR] Iniciando entrenamiento — {N_EPOCHS} épocas...")
    for epoch in range(1, N_EPOCHS + 1):
        d_losses, g_losses, gps = [], [], []

        for batch_idx, (real,) in enumerate(loader):
            real = real.to(device)
            bs = real.size(0)

            # ---- Paso D (N_CRITIC veces) ----
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
            fake = G(z)
            g_loss = -D(fake).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_losses.append(g_loss.item())

        # Log
        d_mean = np.mean(d_losses)
        g_mean = np.mean(g_losses)
        gp_mean = np.mean(gps)
        log['d_loss'].append(d_mean)
        log['g_loss'].append(g_mean)
        log['gp'].append(gp_mean)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Época [{epoch:3d}/{N_EPOCHS}] "
                  f"D: {d_mean:.4f} | G: {g_mean:.4f} | GP: {gp_mean:.4f}")

        # Checkpoint
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

    # Guardar log
    with open(str(ckpt_dir / 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f"[TABULAR] Entrenamiento completado. Checkpoints en {ckpt_dir}")
    return G, D


if __name__ == "__main__":
    train_tabular()
