"""
Extrae joints 3D desde poses SMPL y los normaliza.

Normalización:
  1. Pelvis (joint 0) al origen: joints -= joints[0]
  2. Escala uniforme: dividir por distancia pelvis→cabeza (joint 15)
  3. Aplanar a vector de 72 dimensiones

Guarda en internal/data/joints.npz
"""

import sys
import numpy as np
from pathlib import Path

from src.config.paths import Paths
from src.data.amass_loader import load_all_amass

# Añadir módulo del profesor al path
sys.path.insert(0, str(Paths.SMPL_MODULE_DIR))

SMPL_MODEL_PATH = str(Paths.SMPL_NEUTRAL)


def smpl_forward_joints(poses: np.ndarray, betas: np.ndarray,
                        batch_size: int = 256) -> np.ndarray:
    """
    Aplica SMPL forward pass por lotes y devuelve los 24 joints 3D.

    Args:
        poses: (N, 72) — body pose en axis-angle
        betas: (N, 10) — coeficientes de forma
        batch_size: tamaño del lote para no saturar memoria

    Returns:
        joints: (N, 24, 3)
    """
    import torch
    import smplx

    model = smplx.create(
        SMPL_MODEL_PATH,
        model_type='smpl',
        batch_size=batch_size,
    )
    model.eval()

    N = poses.shape[0]
    all_joints = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bs = end - start

            # Recrear modelo con batch_size correcto si es necesario
            if bs != batch_size:
                model = smplx.create(SMPL_MODEL_PATH, model_type='smpl', batch_size=bs)
                model.eval()

            body_pose_t = torch.tensor(poses[start:end, 3:], dtype=torch.float32)  # (bs, 69)
            global_orient_t = torch.tensor(poses[start:end, :3], dtype=torch.float32)  # (bs, 3)
            betas_t = torch.tensor(betas[start:end], dtype=torch.float32)  # (bs, 10)

            output = model(
                betas=betas_t,
                global_orient=global_orient_t,
                body_pose=body_pose_t,
            )
            joints = output.joints[:, :24, :].cpu().numpy()  # (bs, 24, 3)
            all_joints.append(joints)

    return np.concatenate(all_joints, axis=0)  # (N, 24, 3)


def normalize_joints(joints: np.ndarray) -> np.ndarray:
    """
    Normaliza joints:
      1. Pelvis al origen
      2. Escala por altura (pelvis→cabeza, joint 15)

    Args:
        joints: (N, 24, 3)

    Returns:
        joints_norm: (N, 72) — aplanado y normalizado en [-1, 1] aprox.
    """
    # 1. Centrar en pelvis
    joints = joints - joints[:, 0:1, :]  # (N, 24, 3)

    # 2. Escalar por altura
    height = np.linalg.norm(joints[:, 15, :] - joints[:, 0, :], axis=-1, keepdims=True)  # (N, 1)
    height = np.maximum(height, 1e-6)
    joints = joints / height[:, :, np.newaxis]  # (N, 24, 3)

    # 3. Aplanar
    return joints.reshape(-1, 72).astype(np.float32)


def preprocess_and_save(out_path: str = None):
    """Pipeline completo: carga AMASS → joints → normaliza → guarda."""
    if out_path is None:
        out_path = str(Paths.JOINTS_NPZ)

    print("[1/3] Cargando datos AMASS...")
    data = load_all_amass()
    poses, betas = data['poses'], data['betas']
    print(f"      {poses.shape[0]} frames cargados.")

    print("[2/3] Calculando joints 3D con SMPL forward pass...")
    joints_3d = smpl_forward_joints(poses, betas)
    print(f"      joints_3d shape: {joints_3d.shape}")

    print("[3/3] Normalizando y guardando...")
    joints_norm = normalize_joints(joints_3d)
    print(f"      joints_norm shape: {joints_norm.shape}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, joints=joints_norm)
    print(f"      Guardado en {out_path}")

    return joints_norm


if __name__ == "__main__":
    preprocess_and_save()
