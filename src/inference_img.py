"""Generate sample images from a trained image GAN checkpoint."""
import argparse
import os

import torch
from torchvision.utils import save_image, make_grid

from src.config.paths import Paths
from src.config.hparams import ImgHParams
from src.models import ImgGenerator


def _latest_checkpoint():
    files = list(Paths.EXPERIMENTS_DIR.glob(f"{Paths.IMG_CKPT_PREFIX}*.pt"))
    if not files:
        return None
    return sorted(files, key=os.path.getmtime)[-1]


def infer(args):
    Paths.init_project()
    hp = ImgHParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _latest_checkpoint()
    if ckpt_path is None:
        print("No image GAN checkpoints found. Run src/train_img.py first.")
        return

    G = ImgGenerator(hp).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    print(f"Loaded checkpoint: {ckpt_path.name}")

    with torch.no_grad():
        z = torch.randn(args.n, hp.noise_dim, device=device)
        fake = G(z).cpu()
    fake = (fake + 1.0) / 2.0  # de-normalise [-1,1] -> [0,1]

    out_dir = Paths.TEMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid:
        nrow = max(1, int(args.n ** 0.5))
        grid = make_grid(fake, nrow=nrow)
        out_path = out_dir / f"generated_grid_{args.n}.png"
        save_image(grid, out_path)
        print(f"Saved grid -> {out_path}")
    else:
        for i in range(args.n):
            out_path = out_dir / f"generated_{i:04d}.png"
            save_image(fake[i], out_path)
        print(f"Saved {args.n} images to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from the trained image GAN.")
    parser.add_argument("-n", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--grid", action="store_true",
                        help="Save as a single grid PNG instead of N separate files.")
    args = parser.parse_args()
    infer(args)
