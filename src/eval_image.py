"""
FID (Fréchet Inception Distance) for the Image GAN.

FID measures the distance between distributions of real and generated
images using statistics from a pretrained InceptionV3 network. Lower
is better.

Implementation uses the `pytorch-fid` package, which takes two folders
of PNGs and returns the score. We:
  1. dump N generator samples to a temporary folder
  2. compare against a folder of TNT15 reals (also re-dumped at the
     correct resolution to avoid biasing FID on rescaling alone)
  3. call pytorch-fid's CLI/programmatic API
"""

import os
import shutil
import subprocess
import sys
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import ImageHParams
from src.data.tnt15_loader import TNT15Dataset
from src.models.image_gan import ImageGenerator


def _denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = (x.clamp(-1, 1) + 1) * 127.5
    return x.detach().cpu().numpy().astype(np.uint8)


def _dump_reals(out_dir, n: int, image_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = TNT15Dataset(image_size=image_size, include_skeleton_renders=False, max_samples=n)
    for i in range(len(ds)):
        arr = _denorm_to_uint8(ds[i].squeeze())
        Image.fromarray(arr, mode="L").convert("RGB").save(out_dir / f"real_{i:05d}.png")


def _dump_fakes(G, out_dir, n: int, hp: ImageHParams, device, batch_size: int = 32):
    out_dir.mkdir(parents=True, exist_ok=True)
    G.eval()
    idx = 0
    with torch.no_grad():
        for _ in tqdm(range((n + batch_size - 1) // batch_size), desc="dumping fakes"):
            bs = min(batch_size, n - idx)
            z = torch.randn(bs, hp.noise_dim, device=device)
            imgs = G(z).cpu()
            for j in range(bs):
                arr = _denorm_to_uint8(imgs[j].squeeze())
                Image.fromarray(arr, mode="L").convert("RGB").save(out_dir / f"fake_{idx:05d}.png")
                idx += 1


def evaluate_fid(n_samples: int = 1000) -> float:
    Paths.init_project()
    hp = ImageHParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = ImageGenerator(hp).to(device)
    ckpts = sorted((Paths.EXPERIMENTS_DIR / "image").glob("image_wgangp_*.pt"), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError("No image GAN checkpoints found in internal/experiments/image/")
    G.load_state_dict(torch.load(ckpts[-1], map_location=device)['G_state_dict'])

    real_dir = Paths.TEMP_DIR / "fid_real"
    fake_dir = Paths.TEMP_DIR / "fid_fake"
    if real_dir.exists(): shutil.rmtree(real_dir)
    if fake_dir.exists(): shutil.rmtree(fake_dir)

    _dump_reals(real_dir, n_samples, hp.image_size)
    _dump_fakes(G, fake_dir, n_samples, hp, device)

    cmd = [sys.executable, "-m", "pytorch_fid", str(real_dir), str(fake_dir)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = out.stdout.strip().splitlines()[-1]
    score = float(line.split()[-1])
    print(f"FID on {n_samples} samples: {score:.3f}")
    return score


if __name__ == "__main__":
    evaluate_fid()
