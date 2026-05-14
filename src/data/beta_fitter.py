"""
Fit pseudo-ground-truth SMPL betas for NOMO3D samples.

For each NOMO3D sample (10 body measurements in cm) we optimize the
10 SMPL shape coefficients (betas) so that the measurements extracted
from the resulting SMPL mesh match the NOMO3D target as closely as
possible.

The SMPL measurement pipeline (mesh-plane intersection + convex hull
for circumferences) is not differentiable, so we use scipy's
Nelder-Mead optimizer (gradient-free, robust).

Output: one .npy file per NOMO3D sample in `internal/data/betas_cache/`.
These pseudo-GT betas are loaded by NOMODataset and used as training
targets for the conditional WGAN-GP.
"""

import numpy as np
import torch
import smplx
from pathlib import Path
from scipy.optimize import minimize
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import HParams
from src.data.dataset import NOMODataset
from src.smpl_module_project.measure import MeasureSMPL


NOMO_TO_SMPL = {
    "Head_Top_Height":      "height",
    "BUST_Circ":             "chest circumference",
    "NaturalWAIST_Circ":     "waist circumference",
    "HIP_Circ":              "hip circumference",
    "NeckBase_Circ":         "neck circumference",
    "Shoulder_to_Shoulder":  "shoulder breadth",
    "Inseam":                "inside leg height",
    "Thigh_Circ":            "thigh left circumference",
    "Bicep_Circ":            "bicep right circumference",
}


class CachedMeasureSMPL(MeasureSMPL):
    """MeasureSMPL that caches the smplx model per gender to avoid reloading on every forward pass."""

    def __init__(self, model_root):
        super().__init__(model_root=model_root)
        self._models = {}

    def _get_model(self, gender):
        if gender not in self._models:
            self._models[gender] = smplx.create(
                model_path=self.body_model_root,
                model_type="smpl",
                gender=gender,
                num_betas=10,
                ext='pkl',
            )
        return self._models[gender]

    def from_body_model(self, gender, shape):
        model = self._get_model(gender)
        out = model(betas=shape.to(torch.float32), return_verts=True)
        self.verts = out.vertices.detach().cpu().numpy().squeeze()
        self.joints = out.joints.squeeze().detach().cpu().numpy()
        self.faces = model.faces
        self.gender = gender


def _fit_one(measurer, hparams, target_meas_tensor, gender, init_betas, max_iter):
    """Optimize one sample's betas via Nelder-Mead. Returns (betas_f32, final_loss)."""
    target_dict = {
        col: target_meas_tensor[i].item()
        for i, col in enumerate(hparams.meas_cols)
    }

    pairs = [
        (smpl_name, target_dict[nomo_name])
        for nomo_name, smpl_name in NOMO_TO_SMPL.items()
        if nomo_name in target_dict and target_dict[nomo_name] > 1e-3
    ]
    if not pairs:
        return init_betas.astype(np.float32), float("inf")

    smpl_names = [p[0] for p in pairs]
    targets    = np.array([p[1] for p in pairs], dtype=np.float64)

    weights = np.ones_like(targets)
    if "height" in smpl_names:
        weights[smpl_names.index("height")] = 3.0

    def loss(betas_np):
        b = torch.tensor(betas_np.reshape(1, -1), dtype=torch.float32)
        measurer.measurements = {}
        try:
            measurer.from_body_model(gender=gender, shape=b)
            measurer.measure(smpl_names)
        except Exception:
            return 1e6
        preds = np.array(
            [measurer.measurements.get(k, 0.0) for k in smpl_names],
            dtype=np.float64,
        )
        return float(np.mean(weights * (preds - targets) ** 2))

    n = init_betas.shape[0]
    step = 0.5
    initial_simplex = np.vstack([
        init_betas.astype(np.float64),
        *[init_betas.astype(np.float64) + step * np.eye(n)[i] for i in range(n)],
    ])

    result = minimize(
        loss,
        init_betas.astype(np.float64),
        method='Nelder-Mead',
        options={
            'maxiter':         max_iter,
            'xatol':           1e-2,
            'fatol':           1e-2,
            'adaptive':        True,
            'initial_simplex': initial_simplex,
        },
    )
    return result.x.astype(np.float32), float(result.fun)


def fit_betas(refit: bool = False, max_iter: int = 60):
    """
    Fit pseudo-GT SMPL betas for all NOMO3D samples and cache them.

    :param refit: re-fit even if cache exists.
    :param max_iter: Nelder-Mead iterations per sample.
    """
    Paths.init_project()   # also imports raw datasets via junctions/copies

    hparams = HParams()

    dataset_train = NOMODataset(split='train')
    dataset_test  = NOMODataset(split='test')
    all_samples   = dataset_train.samples + dataset_test.samples

    if not all_samples:
        print("No NOMO3D samples found. Check Paths.NOMO3D_DIR.")
        return

    measurer = CachedMeasureSMPL(model_root=str(Paths.DATA_DIR))

    cache_dir = dataset_train.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    skipped = 0

    print(f"Fitting betas for {len(all_samples)} NOMO3D samples (max_iter={max_iter})...")
    pbar = tqdm(all_samples, desc="fit_betas")

    for sample in pbar:
        cache_file = cache_dir / f"{sample['id']}.npy"
        if cache_file.exists() and not refit:
            skipped += 1
            continue

        init_betas = np.zeros(hparams.num_betas, dtype=np.float32)
        betas, final_loss = _fit_one(
            measurer, hparams,
            sample['measurements'],
            sample['gender'],
            init_betas,
            max_iter=max_iter,
        )

        np.save(cache_file, betas)
        losses.append(final_loss)
        pbar.set_postfix(loss=f"{final_loss:.2f}", skipped=skipped)

    print(f"\nDone. Fitted: {len(losses)}  Skipped (cached): {skipped}")
    if losses:
        arr = np.array(losses)
        print(
            f"Loss stats (weighted MSE cm²):  "
            f"mean={arr.mean():.3f}  median={np.median(arr):.3f}  max={arr.max():.3f}"
        )


if __name__ == "__main__":
    fit_betas()
