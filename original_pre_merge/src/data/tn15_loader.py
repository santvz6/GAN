"""
Carga las imágenes segmentadas de TNT15 (siluetas de personas).

Dataset: TNT15_V1_0
Ubicación: src/dataset/TNT15_V1_0/Images/mr/ (subcarpetas 00-04)
Formato: PNG segmentados (fondo negro, persona blanca/gris)
Resolución objetivo: 128×128 RGB

Se usa Pillow (PIL) — opencv NO está instalado.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.config.paths import Paths


class TNT15Dataset(Dataset):
    """
    Dataset PyTorch que carga los PNGs segmentados de TNT15.

    Los PNGs son imágenes en escala de grises (silueta de persona).
    Se convierten a RGB (3 canales) y se normalizan a [-1, 1].
    """

    def __init__(self, root_dir: str = None, image_size: int = 128):
        if root_dir is None:
            root_dir = str(Paths.TNT15_IMAGES_DIR)

        self.root = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # PNG→RGB
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1,1]
        ])

        self.image_paths = self._collect_images()
        print(f"[TNT15] {len(self.image_paths)} imágenes encontradas en {self.root}")

    def _collect_images(self):
        paths = []
        for subdir in sorted(self.root.iterdir()):
            if subdir.is_dir():
                paths.extend(sorted(subdir.glob('*_segmented.png')))
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


def get_tn15_loader(batch_size: int = 64, num_workers: int = 0,
                    shuffle: bool = True) -> torch.utils.data.DataLoader:
    """Devuelve un DataLoader listo para entrenar."""
    dataset = TNT15Dataset()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
