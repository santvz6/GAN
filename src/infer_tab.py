"""Tab Transformer inference using point cloud input.

Generates a SMPL template point cloud (betas=0), passes it through
the Tab Transformer to predict betas, then renders the body.
"""
import os
import argparse
import torch
import numpy as np
import trimesh

from src.config.paths import Paths
from src.config.hparams import TabHParams
from src.models import TabTransformerGenerator
from src.render_2d import render_mesh_to_png
from src.data.pc_dataset import load_smpl_data, smpl_vertices, _load_smpl_pkl


def _load_pc_scaler(device):
    """Load point cloud normalisation stats."""
    path = Paths.EXPERIMENTS_DIR / "pc_scaler.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"PC scaler not found at {path}. Run train_tab first."
        )
    data = np.load(str(path))
    mean = torch.tensor(data["mean"], dtype=torch.float32).to(device)
    std  = torch.tensor(data["std"],  dtype=torch.float32).to(device)
    return mean, std


def infer(args):
    Paths.init_project()
    hp = TabHParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────
    G = TabTransformerGenerator(hp).to(device)
    ckpt_files = list(Paths.EXPERIMENTS_DIR.glob("wgangp_tab_ckpt_*.pt"))
    if not ckpt_files:
        print("No Tab Transformer checkpoints found. Please train first.")
        return

    latest_ckpt = sorted(ckpt_files, key=os.path.getmtime)[-1]
    ckpt = torch.load(latest_ckpt, map_location=device)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    print(f"Loaded checkpoint: {latest_ckpt.name}")

    # ── Generate point cloud from SMPL template (betas=0) ───────────────
    gender = args.gender.upper()
    smpl_pkl = Paths.SMPL_DIR / f"SMPL_{gender}.pkl"
    if not smpl_pkl.exists():
        print(f"SMPL model not found: {smpl_pkl}")
        return

    v_template, shapedirs = load_smpl_data(str(smpl_pkl))
    template_betas = np.zeros(10, dtype=np.float32)
    verts = smpl_vertices(v_template, shapedirs, template_betas)  # (6890, 3)

    # Subsample (same indices as training)
    np.random.seed(0)
    subsample_idx = np.random.choice(6890, size=hp.n_pc_points, replace=False)
    subsample_idx.sort()
    pc = verts[subsample_idx]  # (n_pc_points, 3)

    print(f"Generated point cloud: {pc.shape} from SMPL template (betas=0)")

    # ── Normalise point cloud ───────────────────────────────────────────
    pc_mean, pc_std = _load_pc_scaler(device)
    pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(device)
    pc_norm = (pc_tensor - pc_mean) / pc_std

    # ── Generate betas: average N z samples ─────────────────────────────
    n_samples = args.n_samples
    with torch.no_grad():
        z_batch = torch.randn(n_samples, hp.noise_dim, device=device)
        pc_rep = pc_norm.expand(n_samples, -1, -1)       # (N, n_pc_points, 3)
        betas_batch = G(z_batch, pc_rep)                   # (N, 10)
        fake_betas = betas_batch.mean(dim=0, keepdim=True).cpu()  # (1, 10)

    print(f"Generated betas (avg of {n_samples}): {fake_betas.numpy()}")

    # ── Render body from generated betas ────────────────────────────────
    gen_betas = fake_betas.numpy().flatten()
    gen_verts = smpl_vertices(v_template, shapedirs, gen_betas)

    # Load faces from SMPL
    smpl_data = _load_smpl_pkl(str(smpl_pkl))
    faces = np.array(smpl_data['f'], dtype=np.int64)

    output_mode = getattr(args, "output", "image")
    output_stem = f"tab_generated_{args.gender}_{args.height:g}"
    mesh = None

    if output_mode in ("mesh", "both"):
        mesh = trimesh.Trimesh(vertices=gen_verts, faces=faces, process=False)
        output_path = Paths.TEMP_DIR / f"{output_stem}.obj"
        mesh.export(output_path)
        print(f"Mesh saved to {output_path}")

    if output_mode in ("image", "both"):
        image_size = getattr(args, "image_size", 512)
        view = getattr(args, "view", "front")
        image_path = Paths.TEMP_DIR / f"{output_stem}_{view}.png"
        render_mesh_to_png(
            verts=gen_verts,
            faces=faces,
            output_path=image_path,
            image_size=image_size,
            view=view,
            draw_edges=getattr(args, "wireframe", False),
        )
        print(f"2D image saved to {image_path}")

    if args.show:
        if mesh is None:
            mesh = trimesh.Trimesh(vertices=gen_verts, faces=faces, process=False)
        mesh.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SMPL body from Tab Transformer (point cloud) model."
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
    parser.add_argument("--output",    type=str,   default="image",
                        choices=["image", "mesh", "both"],
                        help="Output format: 2D PNG image, 3D OBJ mesh, or both.")
    parser.add_argument("--view",      type=str,   default="front",
                        choices=["front", "side", "back"],
                        help="Camera view used for the 2D PNG render.")
    parser.add_argument("--image_size", type=int,  default=512,
                        help="Output PNG size in pixels.")
    parser.add_argument("--wireframe", action="store_true",
                        help="Draw triangle edges in the 2D PNG render.")
    parser.add_argument("--show",      action="store_true", help="Show 3D viewer")

    args = parser.parse_args()
    infer(args)
