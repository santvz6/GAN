"""
Carga ficheros .npz de AMASS y extrae poses + betas del modelo SMPL.

Formato de los .npz de AMASS:
  - poses: (N, 156) — pose de todo el cuerpo (primeros 72 = 24 joints × 3 axis-angle)
  - betas: (16,) o (10,) — coeficientes de forma del cuerpo
  - gender: str — 'male', 'female', 'neutral'
  - mocap_framerate: float — fotogramas por segundo
"""

import numpy as np
from pathlib import Path
from src.config.paths import Paths


def load_amass_file(npz_path: str) -> dict:
    """
    Carga un fichero .npz de AMASS.

    Returns dict con:
      - poses: np.ndarray (N, 72) — primeros 72 coeficientes (24 joints body pose)
      - betas: np.ndarray (10,) — primeros 10 coeficientes de forma
      - gender: str
      - framerate: float
    """
    data = np.load(npz_path, allow_pickle=True)

    poses = data['poses'][:, :72].astype(np.float32)  # (N, 72)

    betas_raw = data['betas']
    betas = betas_raw[:10].astype(np.float32)  # primeros 10

    gender = str(data.get('gender', 'neutral'))
    framerate = float(data.get('mocap_framerate', 30.0))

    return {
        'poses': poses,
        'betas': betas,
        'gender': gender,
        'framerate': framerate,
    }


def load_all_amass(amass_dir: str = None) -> dict:
    """
    Carga todos los .npz de AMASS en amass_dir y los concatena.

    Returns dict con:
      - poses: np.ndarray (N_total, 72)
      - betas: np.ndarray (N_total, 10) — mismos betas repetidos por secuencia
    """
    if amass_dir is None:
        amass_dir = Paths.AMASS_DIR

    amass_dir = Path(amass_dir)
    npz_files = sorted(amass_dir.glob('**/*.npz'))

    if not npz_files:
        raise FileNotFoundError(
            f"No se encontraron ficheros .npz en {amass_dir}.\n"
            "Coloca los ficheros de AMASS en src/dataset/AMASS/"
        )

    all_poses, all_betas = [], []
    for f in npz_files:
        try:
            d = load_amass_file(str(f))
            n = d['poses'].shape[0]
            all_poses.append(d['poses'])
            all_betas.append(np.tile(d['betas'], (n, 1)))
        except Exception as e:
            print(f"[WARN] Saltando {f.name}: {e}")
            continue

    if not all_poses:
        raise ValueError("No se pudo cargar ningún fichero AMASS.")

    return {
        'poses': np.concatenate(all_poses, axis=0),
        'betas': np.concatenate(all_betas, axis=0),
    }
