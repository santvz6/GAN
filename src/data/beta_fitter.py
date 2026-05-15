"""Pseudo-ground-truth β fitter for NOMO3D.

Each sample is fit individually with iterative Gauss-Newton on the
measurement residual `target − measure(SMPL(β, gender))`, using
central-difference Jacobians and a backtracking line search.

The fit starts from a linear warm-start (single linearisation around β=0),
which is the previous fitter's solution. Then re-linearises at each
current β and refines. This is a non-linear fit per body, not a global
linear approximation.

The measurement chain is non-differentiable (`MeasureSMPL` detaches verts
to numpy and `measure_circumference` uses `trimesh.intersections.mesh_plane`),
so autograd is not an option — finite differences are the only path.
"""
import csv

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


def _compute_jacobian_at(measurer, gender: str, beta0: np.ndarray, smpl_cols: list,
                         num_betas: int = 10, eps: float = 0.15):
    """
    Linearise measurements around an arbitrary `beta0`.
    Returns (meas_center, J) where J has shape (len(smpl_cols), num_betas).
    Cost: 2*num_betas + 1 SMPL evaluations.
    """
    beta0 = np.asarray(beta0, dtype=np.float32)
    meas_center = _measure_at(measurer, gender, beta0, smpl_cols)

    J = np.zeros((len(smpl_cols), num_betas), dtype=np.float32)
    for k in range(num_betas):
        bp = beta0.copy(); bp[k] += eps
        bm = beta0.copy(); bm[k] -= eps
        mp = _measure_at(measurer, gender, bp, smpl_cols)
        mm = _measure_at(measurer, gender, bm, smpl_cols)
        J[:, k] = (mp - mm) / (2.0 * eps)

    return meas_center, J


def _compute_jacobian(measurer, gender: str, smpl_cols: list,
                      num_betas: int = 10, eps: float = 0.15):
    """Backward-compat: linearise around β=0 (used as warm-start)."""
    return _compute_jacobian_at(
        measurer, gender, np.zeros(num_betas, dtype=np.float32),
        smpl_cols, num_betas, eps,
    )


def _fit_linear(target: np.ndarray, meas_zero: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Solve J @ beta = target - meas_zero via least-squares, clip to [-3,3]."""
    delta = target - meas_zero
    betas, _, _, _ = np.linalg.lstsq(J, delta, rcond=None)
    return np.clip(betas, -3.0, 3.0).astype(np.float32)


def _fit_gauss_newton(measurer, gender: str, target: np.ndarray, beta0: np.ndarray,
                      smpl_cols: list, num_betas: int = 10,
                      max_iters: int = 8, tol: float = 1e-3, rel_tol: float = 1e-4,
                      max_backtracks: int = 5, eps: float = 0.15,
                      verbose: bool = False):
    """
    Iterative Gauss-Newton with backtracking line search.

    Starts from `beta0` (typically the linear warm-start). At each iteration:
      1. Re-linearise: Jₖ at current βₖ via central differences.
      2. Step: solve Jₖ · Δβ = target − measure(βₖ).
      3. Line search: halve `step` until residual decreases or budget runs out.
      4. Stop when ||Δβ|| < tol, or relative residual change < rel_tol,
         or no descent direction is found.

    Returns (β_final, n_accepted_iters, residual_history, converged_flag).
    """
    beta = np.array(beta0, dtype=np.float32).copy()
    meas_cur = _measure_at(measurer, gender, beta, smpl_cols)
    r_norm_cur = float(np.linalg.norm(target - meas_cur))
    history = [r_norm_cur]
    converged = False

    if verbose:
        print(f"  iter 0:                          ||r||={r_norm_cur:.4f}")

    for it in range(1, max_iters + 1):
        meas_center, J = _compute_jacobian_at(
            measurer, gender, beta, smpl_cols, num_betas=num_betas, eps=eps,
        )
        r = target - meas_center
        delta_beta, *_ = np.linalg.lstsq(J, r, rcond=None)

        # Backtracking line search
        step = 1.0
        accepted = False
        for _ in range(max_backtracks + 1):
            beta_trial = np.clip(beta + step * delta_beta, -3.0, 3.0).astype(np.float32)
            meas_trial = _measure_at(measurer, gender, beta_trial, smpl_cols)
            r_norm_trial = float(np.linalg.norm(target - meas_trial))
            if r_norm_trial <= r_norm_cur:
                accepted = True
                break
            step *= 0.5

        if not accepted:
            if verbose:
                print(f"  iter {it}: no descent direction, halting")
            break

        r_norm_prev = r_norm_cur
        beta = beta_trial
        r_norm_cur = r_norm_trial
        history.append(r_norm_cur)

        if verbose:
            print(f"  iter {it}: step={step:.4f}  ||Δβ||={np.linalg.norm(step*delta_beta):.4f}  ||r||={r_norm_cur:.4f}")

        if np.linalg.norm(step * delta_beta) < tol:
            converged = True
            break
        if r_norm_prev > 0 and (r_norm_prev - r_norm_cur) / r_norm_prev < rel_tol:
            converged = True
            break

    return beta, len(history) - 1, history, converged


def fit_betas():
    Paths.init_project()
    hp = HParams()

    train_ds = NOMODataset(split="train")
    test_ds = NOMODataset(split="test")
    all_samples = train_ds.samples + test_ds.samples

    print(f"Fitting betas for {len(all_samples)} samples...")

    smpl_cols = [DATASET_TO_SMPL[c] for c in hp.meas_cols]

    # One MeasureBody + one warm-start Jacobian per gender (~21 SMPL evals each)
    jacobians = {}
    measurers = {}
    for gender in ("MALE", "FEMALE"):
        print(f"  Computing initial Jacobian for {gender}...")
        m = MeasureBody("smpl", model_root=str(Paths.DATA_DIR))
        measurers[gender] = m
        meas_zero, J = _compute_jacobian(m, gender, smpl_cols, hp.num_betas, hp.fit_eps)
        jacobians[gender] = (meas_zero, J)
        print(f"    meas_zero: {np.round(meas_zero, 1)}")

    cache_dir = train_ds.cache_dir
    stats_path = Paths.LOGS_DIR / "beta_fit_stats.csv"
    csv_is_new = not stats_path.exists()

    n_fitted = 0
    with open(stats_path, "a", newline="") as f:
        writer = csv.writer(f)
        if csv_is_new:
            writer.writerow(["id", "gender", "n_iters",
                             "init_residual_cm", "final_residual_cm",
                             "reduction_pct", "converged"])

        for sample in tqdm(all_samples):
            cache_file = cache_dir / f"{sample['id']}.npy"
            if cache_file.exists():
                continue

            gender = sample["gender"]
            target = sample["measurements"].numpy().astype(np.float32)

            # Linear warm-start (single linearisation around β=0)
            meas_zero, J0 = jacobians[gender]
            beta0 = _fit_linear(target, meas_zero, J0)
            r0 = float(np.linalg.norm(
                target - _measure_at(measurers[gender], gender, beta0, smpl_cols)
            ))

            # Iterative non-linear refinement
            beta, n_iters, history, converged = _fit_gauss_newton(
                measurers[gender], gender, target, beta0, smpl_cols,
                num_betas=hp.num_betas,
                max_iters=hp.fit_max_iters,
                tol=hp.fit_tol,
                rel_tol=hp.fit_rel_tol,
                max_backtracks=hp.fit_max_backtracks,
                eps=hp.fit_eps,
            )
            r_final = history[-1]
            reduction_pct = 100.0 * (1.0 - r_final / r0) if r0 > 0 else 0.0

            np.save(cache_file, beta)
            writer.writerow([
                sample["id"], gender, n_iters,
                f"{r0:.4f}", f"{r_final:.4f}",
                f"{reduction_pct:.2f}", converged,
            ])
            n_fitted += 1

    print(f"Done. Fit {n_fitted} samples; stats appended to {stats_path}")


if __name__ == "__main__":
    # Smoke test: fit the first NOMO3D sample with verbose output and compare
    # the iterative residual against the linear warm-start baseline.
    Paths.init_project()
    hp = HParams()
    train_ds = NOMODataset(split="train")
    sample = train_ds.samples[0]
    gender = sample["gender"]
    target = sample["measurements"].numpy().astype(np.float32)
    smpl_cols = [DATASET_TO_SMPL[c] for c in hp.meas_cols]

    print(f"Smoke test on sample id={sample['id']!r} gender={gender}")
    print(f"Target measurements (cm): {np.round(target, 1)}")

    measurer = MeasureBody("smpl", model_root=str(Paths.DATA_DIR))
    meas_zero, J0 = _compute_jacobian(measurer, gender, smpl_cols, hp.num_betas, hp.fit_eps)
    beta0 = _fit_linear(target, meas_zero, J0)
    r0 = float(np.linalg.norm(
        target - _measure_at(measurer, gender, beta0, smpl_cols)
    ))
    print(f"\nLinear warm-start β: {np.round(beta0, 3)}")
    print(f"Warm-start residual: {r0:.4f} cm")

    print("\nIterative Gauss-Newton:")
    beta, n_iters, history, converged = _fit_gauss_newton(
        measurer, gender, target, beta0, smpl_cols,
        num_betas=hp.num_betas,
        max_iters=hp.fit_max_iters,
        tol=hp.fit_tol,
        rel_tol=hp.fit_rel_tol,
        max_backtracks=hp.fit_max_backtracks,
        eps=hp.fit_eps,
        verbose=True,
    )

    print(f"\nFinal β: {np.round(beta, 3)}")
    print(f"n_iters = {n_iters}, converged = {converged}")
    print(f"Residual: {history[0]:.3f} → {history[-1]:.3f} cm "
          f"(reduction: {100*(1 - history[-1]/history[0]):.1f}%)")
