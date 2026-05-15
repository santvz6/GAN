import os
import shutil

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent # ../src/config/paths.py (3 levels)


class Paths:
    ROOT = BASE_DIR
    SRC = ROOT / "src"
    INTERNAL = ROOT / "internal"
   
    DATA_DIR = INTERNAL / "data"   
    EXPERIMENTS_DIR = INTERNAL / "experiments"
    LOGS_DIR = INTERNAL / "logs"
    TEMP_DIR = INTERNAL / "temp"
    SCALER_PATH = EXPERIMENTS_DIR / "meas_scaler.npz"

    NOMO3D_DIR = DATA_DIR / "nomo3d" 
    SMPL_DIR = DATA_DIR / "smpl"
    TNT15_DIR = DATA_DIR / "tnt15"
    
    NOMO3D_FEMALE_MEAS = NOMO3D_DIR / "female_meas_txt"
    NOMO3D_MALE_MEAS = NOMO3D_DIR / "male_meas_txt"
    NOMO3D_FEMALE_OBJ = NOMO3D_DIR / "female"
    NOMO3D_MALE_OBJ = NOMO3D_DIR / "male"
    TNT15_MODELS = TNT15_DIR / "InputFiles" / "Models_31par"

    # Image GAN
    # Resolution order for TNT15:
    #   1. $TNT15_ROOT env var (absolute path to a folder containing 'Images/')
    #   2. internal/data/tnt15  (canonical project-local location)
    # No user-specific paths are hard-coded.
    _TNT15_ENV = os.environ.get("TNT15_ROOT", "").strip()
    TNT15_ROOT = Path(_TNT15_ENV) if _TNT15_ENV else TNT15_DIR
    TNT15_IMAGES_DIR = TNT15_ROOT / "Images"
    IMG_CKPT_PREFIX = "wgangp_img_ckpt_"
    IMG_SAMPLES_DIR = LOGS_DIR / "img_samples"


    @classmethod
    def init_project(cls):
        dirs = [
            cls.INTERNAL,
            cls.DATA_DIR,  cls.EXPERIMENTS_DIR, cls.LOGS_DIR, cls.TEMP_DIR,
            cls.IMG_SAMPLES_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        cls._rotate_logs()

    @classmethod
    def _rotate_logs(cls, max_logs=10):
        existing_logs = sorted(
            [f for f in cls.LOGS_DIR.glob("*.log")], 
            key=os.path.getmtime
        )
        if len(existing_logs) >= max_logs:
            for log_file in existing_logs[:(len(existing_logs) - max_logs + 1)]:
                shutil.move(str(log_file), str(cls.TEMP_DIR / log_file.name))

