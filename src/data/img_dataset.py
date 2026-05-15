"""TNT15 segmented-image dataset for the image GAN."""
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.config.paths import Paths
from src.config.hparams import ImgHParams


class TNT15ImageDataset(Dataset):
    def __init__(self, split: str = "train", hparams: ImgHParams | None = None):
        self.hp = hparams or ImgHParams()
        self.split = split

        root = Paths.TNT15_IMAGES_DIR
        if not Path(root).exists():
            raise FileNotFoundError(
                f"TNT15 images dir not found: {root}. "
                "Update Paths.TNT15_EXTERNAL or place the dataset at internal/data/tnt15/Images."
            )

        self.files: List[Path] = self._gather(root)
        if not self.files:
            raise RuntimeError(f"No PNG frames discovered under {root}.")

        # deterministic split
        n_train = int(len(self.files) * self.hp.train_split)
        self.files = self.files[:n_train] if split == "train" else self.files[n_train:]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=self.hp.channels),
            transforms.Resize(self.hp.image_size),
            transforms.CenterCrop(self.hp.image_size),
            transforms.ToTensor(),                        # [0,1]
            transforms.Normalize([0.5] * self.hp.channels,
                                 [0.5] * self.hp.channels),  # -> [-1,1]
        ])

    def _gather(self, root: Path) -> List[Path]:
        out: List[Path] = []
        for subj in self.hp.subjects:
            for cam in self.hp.cameras:
                cam_dir = root / subj / cam
                if not cam_dir.exists():
                    continue
                frames = sorted(cam_dir.glob("*_segmented.png"))
                # subsample every Nth frame
                out.extend(frames[:: self.hp.frame_stride])
        # stable deterministic order across runs
        out.sort()
        return out

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx])
        return self.transform(img)


if __name__ == "__main__":
    ds = TNT15ImageDataset(split="train")
    print(f"Dataset size (train): {len(ds)}")
    x = ds[0]
    print(f"Sample tensor: {x.shape}, dtype={x.dtype}, range=[{x.min():.3f}, {x.max():.3f}]")
