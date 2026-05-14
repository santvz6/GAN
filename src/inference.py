import os
import argparse
import torch
import numpy as np
import trimesh

from src.config.paths import Paths
from src.config.hparams import HParams
from src.models import Generator
from src.smpl_module_project.measure import MeasureBody


def _load_scaler(device):
    """Load mean/std saved by the trainer. Raises if not found."""
    if not Paths.SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {Paths.SCALER_PATH}. "
            "Run training first so the scaler gets saved."
        )
    data = np.load(str(Paths.SCALER_PATH))
    mean = torch.tensor(data["mean"], dtype=torch.float32).to(device)
    std  = torch.tensor(data["std"],  dtype=torch.float32).to(device)
    return mean, std


def infer(args):
    Paths.init_project()
    hp = HParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────
    G = Generator(hp).to(device)
    ckpt_files = list(Paths.EXPERIMENTS_DIR.glob("wgangp_ckpt_*.pt"))
    if not ckpt_files:
        print("No checkpoints found. Please train first.")
        return

    latest_ckpt = sorted(ckpt_files, key=os.path.getmtime)[-1]
    ckpt = torch.load(latest_ckpt, map_location=device)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    print(f"Loaded checkpoint: {latest_ckpt.name}")

    # ── Build raw measurement vector ────────────────────────────────────
    meas_dict = {
        "Head_Top_Height":    args.height,
        "BUST_Circ":          args.bust,
        "NaturalWAIST_Circ":  args.waist,
        "HIP_Circ":           args.hip,
        "NeckBase_Circ":      args.neck,
        "Shoulder_to_Shoulder": args.shoulder,
        "Inseam":             args.inseam,
        "Outseam":            args.outseam,
        "Thigh_Circ":         args.thigh,
        "Bicep_Circ":         args.bicep,
    }
    input_meas = [meas_dict[col] for col in hp.meas_cols]
    cond_raw = torch.tensor([input_meas], dtype=torch.float32).to(device)  # (1, 10)

    # ── Normalise with training scaler ──────────────────────────────────
    meas_mean, meas_std = _load_scaler(device)
    cond = (cond_raw - meas_mean) / meas_std  # same transform as training
    print(f"Normalised cond: {cond.cpu().numpy()}")

    # ── Generate: average N z samples to suppress noise variance ────────
    n_samples = args.n_samples  # default 32
    with torch.no_grad():
        z_batch  = torch.randn(n_samples, hp.noise_dim, device=device)
        cond_rep = cond.expand(n_samples, -1)           # (N, 10)
        betas_batch = G(z_batch, cond_rep)               # (N, 10)
        fake_betas  = betas_batch.mean(dim=0, keepdim=True).cpu()  # (1, 10)

    print(f"Generated betas (avg of {n_samples}): {fake_betas.numpy()}")

    # ── Render with SMPL ────────────────────────────────────────────────
    measurer = MeasureBody("smpl", model_root=str(Paths.DATA_DIR))
    measurer.from_body_model(gender=args.gender.upper(), shape=fake_betas)

    verts = measurer.verts
    faces = measurer.faces
    mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    output_path = Paths.TEMP_DIR / f"generated_{args.gender}_{args.height}.obj"
    mesh.export(output_path)
    print(f"Mesh saved to {output_path}")

    if args.show:
        mesh.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SMPL body from 10 tabular measurements."
    )
    parser.add_argument("--gender",    type=str,   default="FEMALE",
                        choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--height",    type=float, default=170.0)
    parser.add_argument("--bust",      type=float, default=90.0)
    parser.add_argument("--waist",     type=float, default=70.0)
    parser.add_argument("--hip",       type=float, default=95.0)
    parser.add_argument("--neck",      type=float, default=34.0)
    parser.add_argument("--shoulder",  type=float, default=40.0)
    parser.add_argument("--inseam",    type=float, default=80.0)
    parser.add_argument("--outseam",   type=float, default=100.0)
    parser.add_argument("--thigh",     type=float, default=55.0)
    parser.add_argument("--bicep",     type=float, default=28.0)
    parser.add_argument("--n_samples", type=int,   default=32,
                        help="Number of z samples to average (reduces noise variance)")
    parser.add_argument("--show",      action="store_true", help="Show 3D viewer")

    args = parser.parse_args()
    infer(args)
