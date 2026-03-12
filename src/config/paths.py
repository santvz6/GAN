import os
import shutil

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # GAN/src/config/paths.py


class Paths:
    # Directorios raíz
    ROOT = BASE_DIR
    SRC = ROOT / "src"
    INTERNAL = ROOT / "internal"

    # Internos (generados)
    DATA_DIR = INTERNAL / "data"
    EXPERIMENTS_DIR = INTERNAL / "experiments"
    LOGS_DIR = INTERNAL / "logs"
    TEMP_DIR = INTERNAL / "temp"

    # Datos intermedios generados
    JOINTS_NPZ = DATA_DIR / "joints.npz"
    GENERATED_JOINTS_NPZ = DATA_DIR / "generated_joints.npz"
    SKELETON_IMAGES_DIR = DATA_DIR / "skeleton_images"
    MESHES_DIR = DATA_DIR / "meshes"

    # Checkpoints de experimentos
    TABULAR_EXP_DIR = EXPERIMENTS_DIR / "tabular"
    IMAGE_EXP_DIR = EXPERIMENTS_DIR / "image"
    MESH_EXP_DIR = EXPERIMENTS_DIR / "mesh"

    # Paths de checkpoints finales (último checkpoint de 200 épocas)
    TABULAR_CHECKPOINT = TABULAR_EXP_DIR / "checkpoint_ep200.pt"
    IMAGE_CHECKPOINT = IMAGE_EXP_DIR / "checkpoint_ep200.pt"
    MESH_CHECKPOINT = MESH_EXP_DIR / "checkpoint_ep200.pt"

    # Datasets fuente
    DATASET_DIR = SRC / "dataset"

    # AMASS
    AMASS_DIR = DATASET_DIR / "AMASS"
    SMPL_MODULE_DIR = AMASS_DIR / "smpl_module_project"
    SMPL_DATA_DIR = SMPL_MODULE_DIR / "data" / "smpl"
    SMPL_NEUTRAL = SMPL_DATA_DIR / "SMPL_NEUTRAL.pkl"
    SMPL_MALE = SMPL_DATA_DIR / "SMPL_MALE.pkl"
    SMPL_FEMALE = SMPL_DATA_DIR / "SMPL_FEMALE.pkl"

    # TNT15
    TNT15_DIR = DATASET_DIR / "TNT15_V1_0"
    TNT15_IMAGES_DIR = TNT15_DIR / "Images" / "mr"

    # NOMO3D
    NOMO3D_DIR = DATASET_DIR / "NOMO3D" / "nomo-scans(repetitions-removed)"
    NOMO3D_FEMALE_DIR = NOMO3D_DIR / "female"
    NOMO3D_MALE_DIR = NOMO3D_DIR / "male"
    NOMO3D_FEMALE_MEAS_DIR = NOMO3D_DIR / "female_meas_txt"
    NOMO3D_MALE_MEAS_DIR = NOMO3D_DIR / "male_meas_txt"

    @classmethod
    def init_project(cls):
        """Crea todos los directorios necesarios para el proyecto."""
        dirs = [
            cls.INTERNAL,
            cls.DATA_DIR,
            cls.EXPERIMENTS_DIR,
            cls.LOGS_DIR,
            cls.TEMP_DIR,
            cls.SKELETON_IMAGES_DIR,
            cls.MESHES_DIR,
            cls.TABULAR_EXP_DIR,
            cls.IMAGE_EXP_DIR,
            cls.MESH_EXP_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        cls._rotate_logs()

    @classmethod
    def _rotate_logs(cls, max_logs: int = 10):
        existing_logs = sorted(
            [f for f in cls.LOGS_DIR.glob("*.log")],
            key=os.path.getmtime
        )
        if len(existing_logs) >= max_logs:
            for log_file in existing_logs[:(len(existing_logs) - max_logs + 1)]:
                shutil.move(str(log_file), str(cls.TEMP_DIR / log_file.name))
