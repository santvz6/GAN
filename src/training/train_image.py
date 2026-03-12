"""
Entrenamiento del Image GAN con WGAN-GP.

Datos:
  - TNT15: ~25k PNGs segmentados de personas (128×128)
  - Renders de esqueletos generados (si existen)

Hiperparámetros:
  - Batch size: 64
  - Learning rate: 1e-4 (Adam, beta1=0.0, beta2=0.9)
  - Epochs: 200
  - Ratio D/G: 5:1
  - Lambda GP: 10
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

from src.config.paths import Paths
from src.models.image_gan import Generator, Discriminator, compute_gradient_penalty
from src.data.tn15_loader import TNT15Dataset


# Hiperparámetros
LATENT_DIM = 128
BATCH_SIZE = 64
LR = 1e-4
BETAS = (0.0, 0.9)
N_EPOCHS = 200
N_CRITIC = 5
LAMBDA_GP = 10.0
CHECKPOINT_EVERY = 50
SEED = 42
IMAGE_SIZE = 128


class SkeletonImageDataset(torch.utils.data.Dataset):
    """Carga imágenes PNG de esqueletos generados."""

    def __init__(self, images_dir: str, image_size: int = IMAGE_SIZE):
        self.paths = sorted(Path(images_dir).glob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


def get_combined_loader(batch_size: int = BATCH_SIZE) -> DataLoader:
    """Combina TNT15 + renders de esqueletos en un solo DataLoader."""
    datasets = []

    # TNT15
    tnt15 = TNT15Dataset()
    if len(tnt15) > 0:
        datasets.append(tnt15)

    # Renders de esqueletos (si existen)
    skel_dir = Paths.SKELETON_IMAGES_DIR
    if skel_dir.exists() and len(list(skel_dir.glob('*.png'))) > 0:
        skel_ds = SkeletonImageDataset(str(skel_dir))
        datasets.append(skel_ds)
        print(f"[IMAGE] Añadidos {len(skel_ds)} renders de esqueletos.")

    if not datasets:
        raise ValueError("No se encontraron imágenes para entrenar.")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[IMAGE] Dataset total: {len(combined)} imágenes")

    return DataLoader(combined, batch_size=batch_size, shuffle=True, drop_last=True)


def train_image():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[IMAGE] Dispositivo: {device}")

    loader = get_combined_loader()

    G = Generator(latent_dim=LATENT_DIM).to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    ckpt_dir = Paths.IMAGE_EXP_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log = {'d_loss': [], 'g_loss': [], 'gp': []}

    print(f"[IMAGE] Iniciando entrenamiento — {N_EPOCHS} épocas...")
    for epoch in range(1, N_EPOCHS + 1):
        d_losses, g_losses, gps = [], [], []

        for real in loader:
            if isinstance(real, (list, tuple)):
                real = real[0]
            real = real.to(device)
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

    print(f"[IMAGE] Entrenamiento completado. Checkpoints en {ckpt_dir}")
    return G, D


if __name__ == "__main__":
    train_image()
