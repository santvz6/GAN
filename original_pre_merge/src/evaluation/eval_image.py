"""
Evaluación del Image GAN usando FID (Fréchet Inception Distance).

FID mide la similitud entre distribuciones de imágenes reales y generadas
comparando estadísticas de features extraídas de InceptionV3.
FID ≈ 0 → imágenes generadas indistinguibles de las reales.

Requiere: pip install pytorch-fid
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from src.config.paths import Paths
from src.models.image_gan import Generator


LATENT_DIM = 128
N_FID_SAMPLES = 2048  # mínimo recomendado para FID estable


def save_images_for_fid(images: torch.Tensor, out_dir: str):
    """Guarda tensores de imágenes como PNG para pytorch-fid."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Desnormalizar de [-1,1] a [0,255]
    images = ((images + 1) / 2 * 255).clamp(0, 255).byte().cpu()

    for i, img_t in enumerate(images):
        img_np = img_t.permute(1, 2, 0).numpy()  # (H, W, 3)
        img = Image.fromarray(img_np)
        img.save(str(out_dir / f'gen_{i:05d}.png'))


def evaluate_image(checkpoint_path: str = None, n_samples: int = N_FID_SAMPLES):
    """
    Calcula FID entre imágenes reales (TNT15) y generadas.

    Returns:
        fid_score: float
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("[EVAL IMAGE] pytorch-fid no instalado. Instalar con: pip install pytorch-fid")
        return None

    if checkpoint_path is None:
        checkpoint_path = str(Paths.IMAGE_CHECKPOINT)

    # Directorio de imágenes reales
    real_dir = str(Paths.TNT15_IMAGES_DIR)

    # Generar imágenes fake
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(latent_dim=LATENT_DIM).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    G.eval()

    tmp_dir = tempfile.mkdtemp(prefix='gan_eval_')
    try:
        print(f"[EVAL IMAGE] Generando {n_samples} imágenes...")
        all_imgs = []
        batch = 64
        with torch.no_grad():
            for start in range(0, n_samples, batch):
                bs = min(batch, n_samples - start)
                z = torch.randn(bs, LATENT_DIM, device=device)
                imgs = G(z)
                all_imgs.append(imgs)

        all_imgs = torch.cat(all_imgs, dim=0)
        save_images_for_fid(all_imgs, tmp_dir)

        print(f"[EVAL IMAGE] Calculando FID...")
        score = fid_score.calculate_fid_given_paths(
            [real_dir, tmp_dir],
            batch_size=64,
            device=str(device),
            dims=2048,
        )
        print(f"\n[EVAL IMAGE] FID: {score:.2f}  (menor es mejor)")
        return score

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    evaluate_image()
