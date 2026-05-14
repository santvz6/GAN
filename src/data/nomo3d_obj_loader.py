"""
NOMO3D point cloud loader.

Loads the OBJ scans from Paths.NOMO3D_FEMALE_OBJ / Paths.NOMO3D_MALE_OBJ
(populated by Paths.init_project() — directory junction or copy from
src/dataset/NOMO3D/nomo-scans(repetitions-removed)/{female,male}/)
and subsamples each mesh's surface to a fixed number of points
(default 6890 — same as SMPL topology) via trimesh.sample.sample_surface.

Subsampled point clouds are cached as .npy in Paths.POINTCLOUDS_CACHE_DIR
so we only pay the trimesh cost once.

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
        Paths.init_project()
        self.num_points = num_points

        if not Paths.NOMO3D_FEMALE_OBJ.exists() or not Paths.NOMO3D_MALE_OBJ.exists():
            raise FileNotFoundError(
                f"NOMO3D OBJ dirs not found at {Paths.NOMO3D_FEMALE_OBJ} / "
                f"{Paths.NOMO3D_MALE_OBJ}. Run Paths.init_project()."
            )

        self.obj_files = sorted(
            list(Paths.NOMO3D_FEMALE_OBJ.glob("*.obj")) +
            list(Paths.NOMO3D_MALE_OBJ.glob("*.obj"))
        )
        if not self.obj_files:
            raise RuntimeError("No OBJ scans found in NOMO3D folders")

        self.cache_dir = Paths.POINTCLOUDS_CACHE_DIR
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
