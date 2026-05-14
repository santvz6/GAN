"""
TNT15 silhouette dataset loader.

The TNT15 dataset contains pre-segmented PNGs of human silhouettes
under src/dataset/TNT15_V1_0/Images/mr/ in five subdirectories 00..04
(~25k images total).

This loader:
  - finds all PNGs under those subdirs
  - resizes to image_size x image_size
  - converts to grayscale (silhouettes are 1-channel)
  - normalizes to [-1, 1] for tanh-output GAN training
  - optionally mixes in pre-rendered SMPL skeletons from
    internal/data/skeleton_renders/ (set by ImageHParams.use_rendered_skeletons)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config.paths import Paths
from src.config.hparams import ImageHParams


class TNT15Dataset(Dataset):
    def __init__(
        self,
        image_size: int = 128,
        subdirs: Optional[List[str]] = None,
        include_skeleton_renders: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.image_size = image_size
        self.hp = ImageHParams()
        subdirs = subdirs or self.hp.tnt15_subdirs

        tnt_base = Paths.ROOT / "src" / "dataset" / "TNT15_V1_0" / "Images" / "mr"
        if not tnt_base.exists():
            raise FileNotFoundError(f"TNT15 not found at {tnt_base}")

        self.files: List[Path] = []
        for sub in subdirs:
            self.files.extend(sorted((tnt_base / sub).glob("*.png")))

        if include_skeleton_renders:
            sk_dir = Paths.DATA_DIR / "skeleton_renders"
            if sk_dir.exists():
                self.files.extend(sorted(sk_dir.glob("*.png")))

        if max_samples is not None:
            self.files = self.files[:max_samples]

        if not self.files:
            raise RuntimeError("TNT15Dataset is empty; check paths and subdirs.")

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0     # -> [-1, 1]
        return torch.from_numpy(arr).unsqueeze(0)                 # (1, H, W)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._load(self.files[idx])


if __name__ == "__main__":
    ds = TNT15Dataset(image_size=128, include_skeleton_renders=False)
    print(f"TNT15Dataset: {len(ds)} images")
    if len(ds) > 0:
        x = ds[0]
        print(f"First image: shape={tuple(x.shape)} range=[{x.min():.2f}, {x.max():.2f}]")
