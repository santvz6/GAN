"""
Carga los scans OBJ de NOMO3D y los submuestrea a 6890 puntos.

Dataset: NOMO3D — nomo-scans(repetitions-removed)/
  - female/ : 177 OBJ (female_0000.obj … female_0176.obj)
  - male/   : 179 OBJ (male_0000.obj  … male_0178.obj)
  - female_meas_txt/ : 177 TXT (medidas antropométricas)
  - male_meas_txt/   : 179 TXT (medidas antropométricas)

Cada OBJ tiene ~57k vértices. Se submuestrea la superficie a 6890 puntos
con trimesh para compatibilidad con la topología SMPL.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset

from src.config.paths import Paths

try:
    import trimesh
except ImportError:
    raise ImportError("trimesh no está instalado. Instala con: pip install trimesh")


N_POINTS = 6890  # puntos por nube (igual que topología SMPL)


def load_obj_as_pointcloud(obj_path: str, n_points: int = N_POINTS) -> np.ndarray:
    """
    Carga un OBJ y muestrea n_points puntos de su superficie.

    Returns:
        points: (n_points, 3) — nube de puntos 3D
    """
    mesh = trimesh.load(obj_path, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float32)


def normalize_pointcloud(points: np.ndarray) -> np.ndarray:
    """
    Centra la nube en su centroide y escala a rango unitario.

    Args:
        points: (N, 3)
    Returns:
        points_norm: (N, 3) centrada y escalada
    """
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.abs(points))
    points = points / (scale + 1e-8)
    return points.astype(np.float32)


class NOMO3DDataset(Dataset):
    """
    Dataset PyTorch con nubes de puntos de NOMO3D.

    Cada elemento es un tensor (6890, 3) normalizado.
    """

    def __init__(self, root_dir: str = None, gender: str = 'all',
                 n_points: int = N_POINTS):
        """
        Args:
            root_dir: ruta a nomo-scans(repetitions-removed)/
            gender: 'female', 'male', o 'all'
            n_points: puntos a muestrear por scan
        """
        if root_dir is None:
            root_dir = str(Paths.NOMO3D_DIR)

        self.root = Path(root_dir)
        self.n_points = n_points
        self.obj_files = self._collect_objs(gender)
        print(f"[NOMO3D] {len(self.obj_files)} scans encontrados (gender={gender})")

    def _collect_objs(self, gender: str):
        files = []
        if gender in ('female', 'all'):
            files += sorted((self.root / 'female').glob('*.obj'))
        if gender in ('male', 'all'):
            files += sorted((self.root / 'male').glob('*.obj'))
        return files

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        points = load_obj_as_pointcloud(str(self.obj_files[idx]), self.n_points)
        points = normalize_pointcloud(points)
        return torch.tensor(points, dtype=torch.float32)  # (6890, 3)


def get_nomo3d_loader(batch_size: int = 16, num_workers: int = 0,
                      shuffle: bool = True, gender: str = 'all'):
    """Devuelve un DataLoader listo para entrenar."""
    dataset = NOMO3DDataset(gender=gender)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
