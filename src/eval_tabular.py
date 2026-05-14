"""
Tabular evaluation metrics for the conditional WGAN-GP (medidas -> betas).

Reports:
  - MAE per measurement (re-uses src/eval.py's pipeline if available)
  - MMD with an RBF (Gaussian) kernel between real and generated betas

MMD (Maximum Mean Discrepancy) measures the distance between two
distributions. Lower = the generated betas distribution matches the
real one better. We use the unbiased U-statistic estimator.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import HParams
from src.data.dataset import NOMODataset
from src.models import Generator


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    xx = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
    return torch.exp(-xx / (2 * sigma ** 2))


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)) -> float:
    """Unbiased MMD^2 with sum of RBF kernels over multiple bandwidths."""
    val = 0.0
    for s in sigmas:
        Kxx = _gaussian_kernel(x, x, s)
        Kyy = _gaussian_kernel(y, y, s)
        Kxy = _gaussian_kernel(x, y, s)
        m, n = x.size(0), y.size(0)
        Kxx = Kxx - torch.diag(Kxx.diag())
        Kyy = Kyy - torch.diag(Kyy.diag())
        val += Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.mean()
    return float(val.item())


def evaluate_mmd(n_samples: int = 500, sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)) -> float:
    """
    Generate n_samples fake betas conditioned on real measurements, and
    compute MMD against the cached real (pseudo-GT) betas.
    """
    Paths.init_project()
    hp = HParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(hp).to(device)
    ckpts = sorted(Paths.EXPERIMENTS_DIR.glob("wgangp_ckpt_*.pt"), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError("No tabular checkpoints found in internal/experiments/")
    G.load_state_dict(torch.load(ckpts[-1], map_location=device)['G_state_dict'])
    G.eval()

    ds = NOMODataset(split='test')
    n = min(n_samples, len(ds))
    real_betas, fake_betas = [], []
    with torch.no_grad():
        for i in tqdm(range(n), desc="MMD samples"):
            meas, beta, _ = ds[i]
            real_betas.append(beta)
            z = torch.randn(1, hp.noise_dim, device=device)
            cond = meas.unsqueeze(0).to(device)
            fake_betas.append(G(z, cond).squeeze(0).cpu())

    R = torch.stack(real_betas).to(device)
    F = torch.stack(fake_betas).to(device)
    score = mmd_rbf(R, F, sigmas)
    print(f"MMD(real, fake) on {n} samples: {score:.6f}")
    return score


if __name__ == "__main__":
    evaluate_mmd()
