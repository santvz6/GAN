"""
NOMO3D point cloud loader.

Loads the OBJ scans under
  src/dataset/NOMO3D/nomo-scans(repetitions-removed)/{female,male}/
and subsamples each mesh's surface to a fixed number of points
(default 6890 — same as SMPL topology) via trimesh.sample.sample_surface.

Subsampled point clouds are cached as .npy in
internal/data/nomo3d_pointclouds/ so we only pay the trimesh cost once.

Each point cloud is mean-centered and scaled to unit max-norm so
the GAN does not have to learn global scale/position.
"""

import numpy as np
import torch
import trimesh
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from src.config.paths import Paths


def _normalize_pc(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pc, axis=1)) + 1e-8
    return (pc / scale).astype(np.float32)


def subsample_obj(obj_path: Path, num_points: int) -> np.ndarray:
    mesh = trimesh.load(str(obj_path), force='mesh', process=False)
    if mesh.faces is None or len(mesh.faces) == 0:
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        idx = np.random.choice(len(pts), num_points, replace=len(pts) < num_points)
        pts = pts[idx]
    else:
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        pts = np.asarray(pts, dtype=np.float32)
    return _normalize_pc(pts)


class NOMO3DPointCloudDataset(Dataset):
    def __init__(self, num_points: int = 6890, refresh_cache: bool = False):
        self.num_points = num_points
        base = Paths.ROOT / "src" / "dataset" / "NOMO3D" / "nomo-scans(repetitions-removed)"
        if not base.exists():
            raise FileNotFoundError(f"NOMO3D not found at {base}")

        self.obj_files = sorted(list((base / "female").glob("*.obj")) +
                                list((base / "male").glob("*.obj")))
        if not self.obj_files:
            raise RuntimeError(f"No OBJ scans found under {base}")

        self.cache_dir = Paths.DATA_DIR / "nomo3d_pointclouds"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for f in self.obj_files:
            cache = self.cache_dir / f"{f.stem}_{num_points}.npy"
            if refresh_cache or not cache.exists():
                pts = subsample_obj(f, num_points)
                np.save(cache, pts)

    def __len__(self) -> int:
        return len(self.obj_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        f = self.obj_files[idx]
        cache = self.cache_dir / f"{f.stem}_{self.num_points}.npy"
        pts = np.load(cache)
        return torch.from_numpy(pts).float()


if __name__ == "__main__":
    ds = NOMO3DPointCloudDataset(num_points=6890)
    print(f"NOMO3DPointCloudDataset: {len(ds)} scans")
    x = ds[0]
    print(f"First point cloud: shape={tuple(x.shape)} mean={x.mean():.3f} max_norm={x.norm(dim=1).max():.3f}")
