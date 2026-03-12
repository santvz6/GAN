"""
Genera mallas SMPL (6890 vértices) desde joints/betas filtrados.

Usa smplx para el forward pass. Los vértices tienen la topología
exacta del modelo SMPL: 6890 vértices con conectividad fija.
"""

import sys
import numpy as np
from pathlib import Path

from src.config.paths import Paths

sys.path.insert(0, str(Paths.SMPL_MODULE_DIR))

SMPL_MODEL_PATH = str(Paths.SMPL_NEUTRAL)


def generate_mesh_from_pose(poses: np.ndarray, betas: np.ndarray,
                             batch_size: int = 64) -> np.ndarray:
    """
    Genera mallas SMPL desde poses y betas.

    Args:
        poses: (N, 72) — body pose axis-angle (primeros 3 = global orient)
        betas: (N, 10) — coeficientes de forma

    Returns:
        vertices: (N, 6890, 3)
    """
    import torch
    import smplx

    N = poses.shape[0]
    all_verts = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bs = end - start

            model = smplx.create(
                SMPL_MODEL_PATH,
                model_type='smpl',
                batch_size=bs,
            )
            model.eval()

            global_orient = torch.tensor(poses[start:end, :3], dtype=torch.float32)
            body_pose = torch.tensor(poses[start:end, 3:], dtype=torch.float32)
            betas_t = torch.tensor(betas[start:end], dtype=torch.float32)

            output = model(
                betas=betas_t,
                global_orient=global_orient,
                body_pose=body_pose,
            )
            verts = output.vertices.cpu().numpy()  # (bs, 6890, 3)
            all_verts.append(verts)

    return np.concatenate(all_verts, axis=0)


def generate_and_save_meshes(n_meshes: int = 500, out_dir: str = None):
    """
    Genera n_meshes mallas aleatorias y las guarda como .npz.
    Usa el checkpoint del Tabular GAN si está disponible.
    """
    import torch

    if out_dir is None:
        out_dir = str(Paths.MESHES_DIR)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intentar cargar cuerpos filtrados
    gen_joints_path = Paths.GENERATED_JOINTS_NPZ
    if gen_joints_path.exists():
        print(f"[SMPL] Cargando cuerpos filtrados desde {gen_joints_path}")
        data = np.load(str(gen_joints_path))
        joints = data['joints'][:n_meshes]
        poses = joints.reshape(-1, 72)
        betas = np.zeros((len(poses), 10), dtype=np.float32)
    else:
        print("[SMPL] No hay cuerpos filtrados. Usando poses aleatorias.")
        np.random.seed(42)
        poses = np.random.randn(n_meshes, 72).astype(np.float32) * 0.1
        betas = np.random.randn(n_meshes, 10).astype(np.float32) * 0.5

    print(f"[SMPL] Generando {len(poses)} mallas...")
    vertices = generate_mesh_from_pose(poses, betas)

    for i, verts in enumerate(vertices):
        path = out_dir / f'mesh_{i:06d}.npz'
        np.savez_compressed(str(path), vertices=verts)

    print(f"[SMPL] {len(vertices)} mallas guardadas en {out_dir}")
    return vertices


if __name__ == "__main__":
    generate_and_save_meshes()
