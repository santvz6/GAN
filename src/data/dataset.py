import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from src.config.paths import Paths
from src.config.hparams import HParams

class NOMODataset(Dataset):
    def __init__(self, split='train'):
        self.hparams = HParams()
        self.split = split
        
        self.samples = []
        self._load_data(Paths.NOMO3D_FEMALE_MEAS, "FEMALE")
        self._load_data(Paths.NOMO3D_MALE_MEAS, "MALE")
        
        # Simple split
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * self.hparams.train_split)
        
        if split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
        # load or compute betas cache
        self.cache_dir = Paths.DATA_DIR / "betas_cache"
        self.cache_dir.mkdir(exist_ok=True)
            
    def _load_data(self, dir_path, gender):
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return
            
        for file_path in dir_path.glob("*.txt"):
            meas_dict = {}
            with open(file_path, "r") as f:
                for line in f:
                    if line.startswith("MEASURE"):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            # e.g. "MEASURE Seat_Back_Angle=18.7"
                            key_val = parts[1].split('=')
                            if len(key_val) == 2 and key_val[1]:
                                meas_dict[key_val[0]] = float(key_val[1])
            
            # Extract 10 inputs
            input_meas = []
            for col in self.hparams.meas_cols:
                input_meas.append(meas_dict.get(col, 0.0))
            
            self.samples.append({
                "id": file_path.stem,
                "gender": gender,
                "measurements": torch.tensor(input_meas, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.samples[real_idx]
        
        # Load cached betas or return zeros
        beta_file = self.cache_dir / f"{sample['id']}.npy"
        if beta_file.exists():
            betas = torch.from_numpy(np.load(beta_file)).float()
        else:
            betas = torch.zeros(self.hparams.num_betas, dtype=torch.float32)
            
        return sample['measurements'], betas, sample['gender']

if __name__ == "__main__":
    ds = NOMODataset()
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        m, b, g = ds[0]
        print(f"Measurements shape: {m.shape}")
        print(f"Betas shape: {b.shape}")
