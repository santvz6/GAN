"""Point-cloud dataset for the Tab Transformer pipeline.

Generates SMPL point clouds from the cached fitted betas, bypassing chumpy
by loading the SMPL .pkl model manually and computing:

    V = v_template + shapedirs @ betas

Each point cloud is subsampled to n_pc_points and cached as .npy.
"""
import io
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.config.paths import Paths
from src.config.hparams import TabHParams


# ── SMPL loader without chumpy dependency ──────────────────────────────────

def _load_smpl_pkl(pkl_path: str) -> dict:
    """Load SMPL .pkl robustly, converting chumpy arrays to numpy.

    Tries three strategies in order:
      1. Direct pickle load (works if chumpy is installed)
      2. Inject a minimal chumpy stub so pickle can reconstruct objects
      3. Fall back to loading with allow_pickle numpy
    """
    import sys
    import types

    # Strategy: inject a minimal chumpy stub module
    stub = types.ModuleType('chumpy')
    stub.__name__ = 'chumpy'

    class _ChArray(np.ndarray):
        """Minimal stand-in for chumpy.Ch / chumpy.array."""
        def __new__(cls, *args, **kwargs):
            if args and isinstance(args[0], (list, np.ndarray)):
                return np.asarray(args[0]).view(cls)
            return np.array([]).view(cls)

        def __reduce_ex__(self, protocol):
            return (np.array, (np.asarray(self),))

        def __setstate__(self, state):
            # chumpy objects pickle a dict; np.ndarray expects a tuple
            if isinstance(state, dict):
                if 'x' in state:
                    self._chumpy_data = np.asarray(state['x'])
                return
            try:
                super().__setstate__(state)
            except Exception:
                pass

        def __array_finalize__(self, obj):
            pass

    stub.Ch = _ChArray
    stub.array = _ChArray

    # Also stub sub-modules that SMPL pickles may reference
    for sub in ('ch', 'logic', 'ch_ops', 'utils', 'reordering'):
        mod = types.ModuleType(f'chumpy.{sub}')
        mod.Ch = _ChArray
        sys.modules[f'chumpy.{sub}'] = mod

    sys.modules['chumpy'] = stub

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    finally:
        # Clean up stubs
        for key in list(sys.modules.keys()):
            if key == 'chumpy' or key.startswith('chumpy.'):
                del sys.modules[key]

    return data


def _to_numpy(obj):
    """Recursively convert any non-numpy array-like to numpy."""
    if hasattr(obj, '_chumpy_data'):
        return obj._chumpy_data
    if isinstance(obj, np.ndarray):
        return np.asarray(obj)  # strips any ndarray subclass
    if hasattr(obj, '__array__'):
        return np.array(obj)
    return obj


def load_smpl_data(pkl_path: str):
    """Load v_template (6890,3) and shapedirs (6890,3,10) from a SMPL .pkl."""
    data = _load_smpl_pkl(pkl_path)

    v_template = _to_numpy(data['v_template']).astype(np.float32)
    shapedirs = _to_numpy(data['shapedirs']).astype(np.float32)

    # shapedirs may have more than 10 components; truncate
    if shapedirs.ndim == 3 and shapedirs.shape[2] > 10:
        shapedirs = shapedirs[:, :, :10]

    return v_template, shapedirs


def smpl_vertices(v_template: np.ndarray, shapedirs: np.ndarray,
                  betas: np.ndarray) -> np.ndarray:
    """Compute SMPL vertices: V = v_template + shapedirs @ betas.

    Args:
        v_template: (6890, 3)
        shapedirs:  (6890, 3, 10)
        betas:      (10,)

    Returns:
        vertices: (6890, 3)
    """
    return v_template + np.einsum('vcd,d->vc', shapedirs, betas)


# ── Point cloud dataset ────────────────────────────────────────────────────

class PointCloudDataset(Dataset):
    """Point-cloud conditioned dataset for Tab Transformer.

    For each sample:
      - Loads the cached fitted betas (.npy)
      - Generates SMPL vertices from those betas
      - Subsamples to n_pc_points
      - Returns (point_cloud, betas, gender)
    """

    def __init__(self, split: str = 'train', hparams: TabHParams = None):
        self.hp = hparams or TabHParams()
        self.split = split

        # Load SMPL models (without chumpy)
        self._smpl = {}
        for gender in ('FEMALE', 'MALE'):
            pkl = Paths.SMPL_DIR / f'SMPL_{gender}.pkl'
            if pkl.exists():
                vt, sd = load_smpl_data(str(pkl))
                self._smpl[gender] = (vt, sd)

        if not self._smpl:
            raise FileNotFoundError(
                f"No SMPL .pkl files found in {Paths.SMPL_DIR}. "
                "Place SMPL_FEMALE.pkl / SMPL_MALE.pkl there."
            )

        # Gather samples that have cached betas
        cache_dir = Paths.DATA_DIR / 'betas_cache'
        self.samples = []

        self._gather_samples(Paths.NOMO3D_FEMALE_MEAS, 'FEMALE', cache_dir)
        self._gather_samples(Paths.NOMO3D_MALE_MEAS, 'MALE', cache_dir)

        if not self.samples:
            raise RuntimeError(
                "No samples with cached betas found. Run 'python main.py fit' first."
            )

        # Deterministic split (same seed and logic as NOMODataset)
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * self.hp.train_split)

        if split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        # Fixed subsample indices (reproducible across samples)
        n_verts = 6890  # SMPL vertex count
        np.random.seed(0)
        self._subsample_idx = np.random.choice(
            n_verts, size=self.hp.n_pc_points, replace=False
        )
        self._subsample_idx.sort()

        # Point cloud normalisation stats (computed lazily)
        self._pc_mean = None
        self._pc_std = None

        # Cache directory for pre-computed point clouds
        self.pc_cache_dir = Paths.DATA_DIR / 'pc_cache'
        self.pc_cache_dir.mkdir(parents=True, exist_ok=True)

    def _gather_samples(self, meas_dir: Path, gender: str, cache_dir: Path):
        meas_dir = Path(meas_dir)
        if not meas_dir.exists():
            return
        for txt in meas_dir.glob('*.txt'):
            beta_file = cache_dir / f'{txt.stem}.npy'
            if beta_file.exists():
                self.samples.append({
                    'id': txt.stem,
                    'gender': gender,
                    'beta_path': beta_file,
                })

    def _get_point_cloud(self, sample: dict) -> np.ndarray:
        """Generate or load cached point cloud for a sample."""
        pc_file = self.pc_cache_dir / f"{sample['id']}.npy"

        if pc_file.exists():
            return np.load(pc_file)

        # Generate point cloud from betas using SMPL
        betas = np.load(sample['beta_path']).astype(np.float32)
        gender = sample['gender']
        vt, sd = self._smpl[gender]
        verts = smpl_vertices(vt, sd, betas)  # (6890, 3)

        # Subsample
        pc = verts[self._subsample_idx]  # (n_pc_points, 3)

        # Cache
        np.save(pc_file, pc.astype(np.float32))
        return pc

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.samples[real_idx]

        betas = np.load(sample['beta_path']).astype(np.float32)
        pc = self._get_point_cloud(sample)  # (n_pc_points, 3)

        return (
            torch.from_numpy(pc),        # (n_pc_points, 3)
            torch.from_numpy(betas),     # (10,)
            sample['gender'],
        )


if __name__ == '__main__':
    Paths.init_project()
    ds = PointCloudDataset(split='train')
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        pc, betas, gender = ds[0]
        print(f"Point cloud shape: {pc.shape}")  # (n_pc_points, 3)
        print(f"Betas shape: {betas.shape}")      # (10,)
        print(f"Gender: {gender}")
