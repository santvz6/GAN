"""
3D evaluation: Chamfer Distance + F-score for the Mesh GAN.

Chamfer Distance: average bidirectional nearest-neighbor distance between
two point clouds. Lower = better.

F-score @ tau: harmonic mean of precision and recall at distance threshold
tau. For each point in set A, it counts as a match if its nearest neighbor
in set B is within tau. F-score reports how much of each set's geometry
is covered by the other.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import MeshHParams
from src.data.nomo3d_obj_loader import NOMO3DPointCloudDataset
from src.models.mesh_gan import MeshGenerator


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric Chamfer between (Na,3) and (Nb,3) on the same device."""
    diff = a.unsqueeze(1) - b.unsqueeze(0)         # (Na, Nb, 3)
    d2 = diff.pow(2).sum(-1)                       # (Na, Nb)
    return float(d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean())


def fscore(a: torch.Tensor, b: torch.Tensor, tau: float = 0.01) -> float:
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    d = diff.pow(2).sum(-1).sqrt()
    precision = (d.min(dim=1).values < tau).float().mean()
    recall    = (d.min(dim=0).values < tau).float().mean()
    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_3d(n_samples: int = 50, tau: float = 0.01) -> dict:
    Paths.init_project()
    hp = MeshHParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = MeshGenerator(hp).to(device)
    ckpts = sorted(Paths.EXPERIMENTS_MESH.glob("mesh_wgangp_*.pt"), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"No mesh GAN checkpoints found in {Paths.EXPERIMENTS_MESH}")
    G.load_state_dict(torch.load(ckpts[-1], map_location=device)['G_state_dict'])
    G.eval()

    real_ds = NOMO3DPointCloudDataset(num_points=hp.num_points)
    n = min(n_samples, len(real_ds))

    cds, fss = [], []
    with torch.no_grad():
        for i in tqdm(range(n), desc="3D eval"):
            real = real_ds[i].to(device)
            z = torch.randn(1, hp.noise_dim, device=device)
            fake = G(z).squeeze(0)
            cds.append(chamfer_distance(real, fake))
            fss.append(fscore(real, fake, tau=tau))

    result = {
        "chamfer_mean":   float(np.mean(cds)),
        "chamfer_median": float(np.median(cds)),
        "fscore_mean":    float(np.mean(fss)),
        "tau":            tau,
        "n_samples":      n,
    }
    print(f"3D Metrics over {n} samples:")
    print(f"  Chamfer mean   : {result['chamfer_mean']:.5f}")
    print(f"  Chamfer median : {result['chamfer_median']:.5f}")
    print(f"  F-score (tau={tau}) : {result['fscore_mean']:.4f}")
    return result


if __name__ == "__main__":
    evaluate_3d()
