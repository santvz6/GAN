import numpy as np
import torch
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import HParams
from src.data.dataset import NOMODataset
from src.smpl_module_project.measure import MeasureBody

# Map dataset column names -> SMPL measurement names
DATASET_TO_SMPL = {
    "Head_Top_Height":      "height",
    "BUST_Circ":            "chest circumference",
    "NaturalWAIST_Circ":    "waist circumference",
    "HIP_Circ":             "hip circumference",
    "NeckBase_Circ":        "neck circumference",
    "Shoulder_to_Shoulder": "shoulder breadth",
    "Inseam":               "inside leg height",
    "Outseam":              "shoulder to crotch height",
    "Thigh_Circ":           "thigh left circumference",
    "Bicep_Circ":           "bicep right circumference",
}


def _measure_at(measurer, gender: str, betas: np.ndarray, smpl_cols: list) -> np.ndarray:
    """Run SMPL forward + extract measurements. Always resets the cache."""
    measurer.measurements = {}   # MUST reset: measure() skips already-computed keys
    betas_t = torch.tensor(betas[None], dtype=torch.float32)
    measurer.from_body_model(gender=gender, shape=betas_t)
    measurer.measure(smpl_cols)
    return np.array([measurer.measurements.get(col, 0.0) for col in smpl_cols],
                    dtype=np.float32)


def _compute_jacobian(measurer, gender: str, smpl_cols: list,
                      num_betas: int = 10, eps: float = 0.15):
    """
    Linearise measurements around betas=0.
    Returns (meas_zero, J) where J has shape (len(smpl_cols), num_betas).
    Cost: 2*num_betas + 1 SMPL evaluations per gender.
    """
    zero = np.zeros(num_betas, dtype=np.float32)
    meas_zero = _measure_at(measurer, gender, zero, smpl_cols)

    J = np.zeros((len(smpl_cols), num_betas), dtype=np.float32)
    for k in range(num_betas):
        bp = zero.copy(); bp[k] =  eps
        bm = zero.copy(); bm[k] = -eps
        mp = _measure_at(measurer, gender, bp, smpl_cols)
        mm = _measure_at(measurer, gender, bm, smpl_cols)
        J[:, k] = (mp - mm) / (2.0 * eps)

    return meas_zero, J


def _fit_linear(target: np.ndarray, meas_zero: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Solve J @ beta = target - meas_zero via least-squares, clip to [-3,3]."""
    delta = target - meas_zero
    betas, _, _, _ = np.linalg.lstsq(J, delta, rcond=None)
    return np.clip(betas, -3.0, 3.0).astype(np.float32)


def fit_betas():
    Paths.init_project()
    hp = HParams()

    train_ds = NOMODataset(split="train")
    test_ds  = NOMODataset(split="test")
    all_samples = train_ds.samples + test_ds.samples

    print(f"Fitting betas for {len(all_samples)} samples...")

    # Build SMPL-name list in the same order as hparams.meas_cols
    smpl_cols = [DATASET_TO_SMPL[c] for c in hp.meas_cols]

    # Pre-compute one Jacobian per gender (cheap: 21 SMPL evals each)
    jacobians = {}
    for gender in ("MALE", "FEMALE"):
        print(f"  Computing Jacobian for {gender}...")
        m = MeasureBody("smpl", model_root=str(Paths.DATA_DIR))
        meas_zero, J = _compute_jacobian(m, gender, smpl_cols, hp.num_betas)
        jacobians[gender] = (meas_zero, J)
        print(f"    meas_zero: {np.round(meas_zero, 1)}")

    cache_dir = train_ds.cache_dir
    for sample in tqdm(all_samples):
        cache_file = cache_dir / f"{sample['id']}.npy"
        if cache_file.exists():
            continue

        gender = sample["gender"]
        target = sample["measurements"].numpy().astype(np.float32)
        meas_zero, J = jacobians[gender]

        betas = _fit_linear(target, meas_zero, J)
        np.save(cache_file, betas)

    print("Done fitting betas.")


if __name__ == "__main__":
    fit_betas()
