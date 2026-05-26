"""Tabular GAN evaluation with FID and Inception Score."""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans

from src.config.paths import Paths
from src.config.hparams import HParams, TabHParams
from src.data.dataset import NOMODataset
from src.models import Generator, TabTransformerGenerator


# ── FID (tabular) ──────────────────────────────────────────────────────────

def _compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Fréchet distance between two multivariate Gaussians."""
    mu_r = real_features.mean(axis=0)
    mu_g = fake_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_g
    # Product of covariance matrices
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    # Numerical stability: remove imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean)
    return float(fid)


# ── Inception Score (tabular) ──────────────────────────────────────────────

class _BetaClassifier(nn.Module):
    """Small MLP classifier for IS computation on beta clusters."""
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _train_classifier(real_betas: np.ndarray, n_classes: int = 10,
                       epochs: int = 100, lr: float = 1e-3):
    """Cluster real betas with K-Means, train classifier, return model."""
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(real_betas)

    X = torch.tensor(real_betas, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    clf = _BetaClassifier(real_betas.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    clf.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(clf(xb), yb).backward()
            opt.step()

    clf.eval()
    return clf


def _compute_inception_score(clf: nn.Module, fake_betas: np.ndarray) -> float:
    """IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )."""
    X = torch.tensor(fake_betas, dtype=torch.float32)
    with torch.no_grad():
        logits = clf(X)
        pyx = torch.softmax(logits, dim=1).numpy()  # p(y|x)

    # Marginal p(y) = E_x[p(y|x)]
    py = pyx.mean(axis=0, keepdims=True)

    # KL divergence per sample
    kl = (pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))).sum(axis=1)
    is_score = float(np.exp(kl.mean()))
    return is_score


# ── Main evaluation entry point ───────────────────────────────────────────

def evaluate_tab():
    Paths.init_project()
    hp_mlp = HParams()
    hp_tab = TabHParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load real betas from test set ──────────────────────────────────
    test_ds = NOMODataset(split='test')
    real_betas_list = []
    meas_list = []
    for idx in range(len(test_ds)):
        meas, betas, _ = test_ds[idx]
        real_betas_list.append(betas.numpy())
        meas_list.append(meas.numpy())

    real_betas = np.stack(real_betas_list)  # (N_test, 10)
    meas_all   = np.stack(meas_list)        # (N_test, 10)
    n_test = real_betas.shape[0]
    print(f"Test samples: {n_test}")

    # ── Load scaler ────────────────────────────────────────────────────
    if not Paths.SCALER_PATH.exists():
        print("No scaler found. Please train a model first.")
        return
    scaler = np.load(str(Paths.SCALER_PATH))
    meas_mean = torch.tensor(scaler["mean"], dtype=torch.float32).to(device)
    meas_std  = torch.tensor(scaler["std"],  dtype=torch.float32).to(device)

    meas_tensor = torch.tensor(meas_all, dtype=torch.float32).to(device)
    meas_norm   = (meas_tensor - meas_mean) / meas_std

    # ── Train IS classifier on real betas ──────────────────────────────
    print("Training IS classifier on real beta clusters...")
    n_classes = min(10, max(2, n_test // 5))  # adaptive number of clusters
    clf = _train_classifier(real_betas, n_classes=n_classes)
    print(f"  Classifier trained ({n_classes} clusters)")

    results = {}

    # ── Evaluate MLP Generator (if checkpoint exists) ──────────────────
    mlp_ckpts = list(Paths.EXPERIMENTS_DIR.glob("wgangp_ckpt_*.pt"))
    if mlp_ckpts:
        latest = sorted(mlp_ckpts, key=os.path.getmtime)[-1]
        G_mlp = Generator(hp_mlp).to(device)
        ckpt = torch.load(latest, map_location=device)
        G_mlp.load_state_dict(ckpt["G_state_dict"])
        G_mlp.eval()
        print(f"\n--- MLP Generator ({latest.name}) ---")

        with torch.no_grad():
            z = torch.randn(n_test, hp_mlp.noise_dim, device=device)
            fake_mlp = G_mlp(z, meas_norm).cpu().numpy()

        fid_mlp = _compute_fid(real_betas, fake_mlp)
        is_mlp  = _compute_inception_score(clf, fake_mlp)
        results["MLP"] = {"FID": fid_mlp, "IS": is_mlp}
        print(f"  FID            : {fid_mlp:.4f}")
        print(f"  Inception Score: {is_mlp:.4f}")

    # ── Evaluate Tab Transformer Generator (if checkpoint exists) ──────
    tab_ckpts = list(Paths.EXPERIMENTS_DIR.glob("wgangp_tab_ckpt_*.pt"))
    if tab_ckpts:
        latest = sorted(tab_ckpts, key=os.path.getmtime)[-1]
        G_tab = TabTransformerGenerator(hp_tab).to(device)
        ckpt = torch.load(latest, map_location=device)
        G_tab.load_state_dict(ckpt["G_state_dict"])
        G_tab.eval()
        print(f"\n--- Tab Transformer Generator ({latest.name}) ---")

        with torch.no_grad():
            z = torch.randn(n_test, hp_tab.noise_dim, device=device)
            fake_tab = G_tab(z, meas_norm).cpu().numpy()

        fid_tab = _compute_fid(real_betas, fake_tab)
        is_tab  = _compute_inception_score(clf, fake_tab)
        results["TabTransformer"] = {"FID": fid_tab, "IS": is_tab}
        print(f"  FID            : {fid_tab:.4f}")
        print(f"  Inception Score: {is_tab:.4f}")

    # ── Summary table ──────────────────────────────────────────────────
    if not results:
        print("\nNo checkpoints found. Train at least one model first.")
        return

    print("\n" + "=" * 50)
    print(f"{'Model':<20} {'FID':>10} {'IS':>10}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['FID']:>10.4f} {metrics['IS']:>10.4f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_tab()
