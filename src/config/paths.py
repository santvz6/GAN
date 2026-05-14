"""
Centralized path constants for the project (Santi's pattern).

Conventions
-----------
- `Paths.<NAME>` is the canonical location used by the code. Loaders and
  trainers reference these constants only — never hardcoded strings.
- Data lives under `internal/data/...`; experiments under
  `internal/experiments/...`; temp/sample dumps under `internal/temp/...`.
- For large raw datasets (TNT15 images, NOMO3D OBJ scans) the canonical
  internal location is populated by a **directory junction** to
  `src/dataset/...` so we don't duplicate gigabytes. If a junction cannot
  be created (e.g. on Google Drive, which doesn't support reparse
  points), the corresponding `Paths.<NAME>` constant is rewritten at
  init time to point directly at the source location. Either way, code
  that references `Paths.<NAME>` keeps working unchanged.
- Small files (SMPL .pkl, NOMO3D measurement .txt files) are copied
  outright since duplication is negligible.
"""

import os
import shutil
import subprocess
import sys

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ../src/config/paths.py (3 levels)


class Paths:
    ROOT     = BASE_DIR
    SRC      = ROOT / "src"
    INTERNAL = ROOT / "internal"

    DATA_DIR        = INTERNAL / "data"
    EXPERIMENTS_DIR = INTERNAL / "experiments"
    LOGS_DIR        = INTERNAL / "logs"
    TEMP_DIR        = INTERNAL / "temp"

    NOMO3D_DIR = DATA_DIR / "nomo3d"
    SMPL_DIR   = DATA_DIR / "smpl"
    TNT15_DIR  = DATA_DIR / "tnt15"

    NOMO3D_FEMALE_MEAS = NOMO3D_DIR / "female_meas_txt"
    NOMO3D_MALE_MEAS   = NOMO3D_DIR / "male_meas_txt"
    NOMO3D_FEMALE_OBJ  = NOMO3D_DIR / "female"
    NOMO3D_MALE_OBJ    = NOMO3D_DIR / "male"
    TNT15_MODELS       = TNT15_DIR  / "InputFiles" / "Models_31par"

    # Added for Image / Mesh GAN and caches
    TNT15_IMAGES_DIR        = TNT15_DIR  / "Images" / "mr"
    BETAS_CACHE_DIR         = DATA_DIR   / "betas_cache"
    POINTCLOUDS_CACHE_DIR   = DATA_DIR   / "nomo3d_pointclouds"
    SKELETON_RENDERS_DIR    = DATA_DIR   / "skeleton_renders"

    EXPERIMENTS_TABULAR = EXPERIMENTS_DIR / "tabular"
    EXPERIMENTS_IMAGE   = EXPERIMENTS_DIR / "image"
    EXPERIMENTS_MESH    = EXPERIMENTS_DIR / "mesh"

    IMAGE_SAMPLES_DIR = TEMP_DIR / "image_samples"
    MESH_SAMPLES_DIR  = TEMP_DIR / "mesh_samples"
    FID_REAL_DIR      = TEMP_DIR / "fid_real"
    FID_FAKE_DIR      = TEMP_DIR / "fid_fake"

    # Raw dataset sources (only consulted by init_project to link/copy data into canonical locations)
    _RAW_SMPL_SRC          = SRC / "dataset" / "AMASS" / "smpl_module_project" / "data" / "smpl"
    _RAW_NOMO3D_BASE       = SRC / "dataset" / "NOMO3D" / "nomo-scans(repetitions-removed)"
    _RAW_NOMO3D_FEMALE_OBJ = _RAW_NOMO3D_BASE / "female"
    _RAW_NOMO3D_MALE_OBJ   = _RAW_NOMO3D_BASE / "male"
    _RAW_NOMO3D_FEMALE_MEAS = _RAW_NOMO3D_BASE / "female_meas_txt"
    _RAW_NOMO3D_MALE_MEAS   = _RAW_NOMO3D_BASE / "male_meas_txt"
    _RAW_TNT15_IMAGES      = SRC / "dataset" / "TNT15_V1_0" / "Images" / "mr"

    # ------------------------------------------------------------------ #
    @classmethod
    def init_project(cls):
        for d in (cls.INTERNAL, cls.DATA_DIR, cls.EXPERIMENTS_DIR,
                  cls.LOGS_DIR, cls.TEMP_DIR,
                  cls.EXPERIMENTS_TABULAR, cls.EXPERIMENTS_IMAGE, cls.EXPERIMENTS_MESH):
            d.mkdir(parents=True, exist_ok=True)

        cls._rotate_logs()
        cls._import_raw_datasets()

    # ------------------------------------------------------------------ #
    # Raw-dataset import. Small things are copied. Big things are linked
    # via junction, and if that fails the Paths constant is rewritten to
    # point at the source location (transparent to callers).
    # ------------------------------------------------------------------ #
    @classmethod
    def _import_raw_datasets(cls):
        # SMPL .pkl + .json segmentation files (~115 MB total, copy is fine and required by smplx)
        cls._copy_dir_contents(cls._RAW_SMPL_SRC, cls.SMPL_DIR)

        # NOMO3D measurement .txt files — link or fall back (Drive may have synced empty dirs)
        cls.NOMO3D_FEMALE_MEAS = cls._link_or_fallback(cls._RAW_NOMO3D_FEMALE_MEAS, cls.NOMO3D_FEMALE_MEAS)
        cls.NOMO3D_MALE_MEAS   = cls._link_or_fallback(cls._RAW_NOMO3D_MALE_MEAS,   cls.NOMO3D_MALE_MEAS)

        # NOMO3D OBJ scans (~hundreds of MB) — link or fall back
        cls.NOMO3D_FEMALE_OBJ = cls._link_or_fallback(cls._RAW_NOMO3D_FEMALE_OBJ, cls.NOMO3D_FEMALE_OBJ)
        cls.NOMO3D_MALE_OBJ   = cls._link_or_fallback(cls._RAW_NOMO3D_MALE_OBJ,   cls.NOMO3D_MALE_OBJ)

        # TNT15 silhouettes (~1 GB) — link or fall back
        cls.TNT15_IMAGES_DIR = cls._link_or_fallback(cls._RAW_TNT15_IMAGES, cls.TNT15_IMAGES_DIR)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_windows() -> bool:
        return sys.platform.startswith("win")

    @staticmethod
    def _is_junction(p: Path) -> bool:
        """True if p is a Windows directory junction (reparse point)."""
        try:
            import stat
            st = os.lstat(str(p))
            return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except Exception:
            return False

    @classmethod
    def _make_dir_link(cls, src: Path, dst: Path) -> bool:
        """Create a directory junction (Win) / symlink (Unix). Return success."""
        if dst.exists():
            # Only consider an existing link as success; real dirs are not "linked".
            return dst.is_symlink() or (cls._is_windows() and cls._is_junction(dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if cls._is_windows():
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                    check=True, capture_output=True,
                )
            else:
                os.symlink(str(src), str(dst), target_is_directory=True)
            return True
        except Exception:
            return False

    @classmethod
    def _link_or_fallback(cls, src: Path, canonical: Path) -> Path:
        """
        Resolve a "big" dataset location.

        Strategy:
          1. If `canonical` is a symlink/junction (or exists as a real dir
             that holds AT LEAST as many entries as `src`), use canonical.
          2. Otherwise try to create a junction src -> canonical.
          3. If junction creation fails, return `src` directly.
        """
        if not src.exists():
            return canonical  # caller will see a clear "not found" error

        # Case 1: canonical is already a complete copy or a working junction
        if canonical.exists():
            try:
                if canonical.is_symlink() or (
                    canonical.is_dir()
                    and sum(1 for _ in canonical.iterdir()) >= sum(1 for _ in src.iterdir())
                ):
                    return canonical
            except Exception:
                pass

        # Case 2: try to junction
        if cls._make_dir_link(src, canonical):
            return canonical

        # Case 3: fall back to source
        return src

    @classmethod
    def _copy_dir_contents(cls, src: Path, dst: Path):
        """Copy each file inside `src` to `dst` (no recursion). Idempotent."""
        if not src.exists():
            return
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if not f.is_file():
                continue
            target = dst / f.name
            if not target.exists():
                shutil.copy(f, target)

    @classmethod
    def _copy_dir_recursive(cls, src: Path, dst: Path):
        """Copy entire directory tree from src to dst if dst doesn't exist."""
        if not src.exists() or dst.exists():
            return
        shutil.copytree(src, dst)

    @classmethod
    def _rotate_logs(cls, max_logs=10):
        existing_logs = sorted(
            [f for f in cls.LOGS_DIR.glob("*.log")],
            key=os.path.getmtime,
        )
        if len(existing_logs) >= max_logs:
            for log_file in existing_logs[:(len(existing_logs) - max_logs + 1)]:
                shutil.move(str(log_file), str(cls.TEMP_DIR / log_file.name))
