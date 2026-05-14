"""
Render SMPL skeleton as a 2D image (128x128 grayscale).

Joints are projected to the XY plane (orthographic), normalized to fit
the image, then drawn with Pillow as line segments (bones) + circles
(joints). No OpenCV dependency.

Used to complement the TNT15 silhouettes with synthetic-skeleton images
when training the Image GAN.
"""

import numpy as np
import torch
import smplx
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Optional

from src.config.paths import Paths


SMPL_BONES = [
    (0, 1),  (0, 2),  (1, 4),  (2, 5),  (4, 7),  (5, 8),
    (0, 3),  (3, 6),  (6, 9),  (9, 12), (12, 15),
    (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19),
    (18, 20), (19, 21),
]


class SkeletonRenderer:
    """
    Render joints (24,3) from SMPL forward pass to a 128x128 grayscale image.

    Usage:
        renderer = SkeletonRenderer(model_root=str(Paths.DATA_DIR))
        img = renderer.render_from_betas(betas=torch.zeros(1, 10), gender="NEUTRAL")
        img.save("skeleton.png")
    """

    def __init__(
        self,
        model_root: str,
        image_size: int = 128,
        bone_width: int = 2,
        joint_radius: int = 3,
    ):
        self.image_size = image_size
        self.bone_width = bone_width
        self.joint_radius = joint_radius
        self.model_root = model_root
        self._models = {}

    def _get_model(self, gender: str):
        if gender not in self._models:
            self._models[gender] = smplx.create(
                model_path=self.model_root,
                model_type="smpl",
                gender=gender,
                num_betas=10,
                ext="pkl",
            )
        return self._models[gender]

    def joints_from_betas(self, betas: torch.Tensor, gender: str) -> np.ndarray:
        """SMPL forward pass with zero pose; returns joints as (24, 3) numpy."""
        model = self._get_model(gender)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        with torch.no_grad():
            out = model(betas=betas.to(torch.float32), return_verts=True)
        return out.joints.squeeze().cpu().numpy()[:24]

    @staticmethod
    def project_xy(joints: np.ndarray) -> np.ndarray:
        """Orthographic projection: drop Z, then normalize to [0,1]^2."""
        xy = joints[:, :2].copy()
        xy[:, 1] *= -1                         # SMPL Y up -> image Y down
        mn, mx = xy.min(0), xy.max(0)
        span = (mx - mn).max() + 1e-6
        center = (mx + mn) / 2
        xy = (xy - center) / span + 0.5
        return np.clip(xy, 0.05, 0.95)

    def render(self, joints: np.ndarray) -> Image.Image:
        """Draw bones (lines) + joints (circles) on a black background."""
        xy = self.project_xy(joints)
        px = (xy * self.image_size).astype(int)

        img = Image.new("L", (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(img)

        for a, b in SMPL_BONES:
            if a < len(px) and b < len(px):
                draw.line([tuple(px[a]), tuple(px[b])], fill=255, width=self.bone_width)

        r = self.joint_radius
        for x, y in px:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

        return img

    def render_from_betas(self, betas: torch.Tensor, gender: str) -> Image.Image:
        joints = self.joints_from_betas(betas, gender)
        return self.render(joints)


def render_batch_from_betas_cache(
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    image_size: int = 128,
) -> int:
    """
    Render skeleton images for every cached betas .npy in betas_cache/.
    Writes PNGs to internal/data/skeleton_renders/<id>.png.
    Returns number of images written.
    """
    Paths.init_project()
    if output_dir is None:
        output_dir = Paths.DATA_DIR / "skeleton_renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Paths.DATA_DIR / "betas_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Run fit_betas first; {cache_dir} not found")

    renderer = SkeletonRenderer(model_root=str(Paths.DATA_DIR), image_size=image_size)

    files = sorted(cache_dir.glob("*.npy"))
    if limit:
        files = files[:limit]

    n = 0
    for f in files:
        gender = "FEMALE" if f.stem.startswith("female") else "MALE"
        betas = torch.from_numpy(np.load(f)).float()
        img = renderer.render_from_betas(betas, gender)
        img.save(output_dir / f"{f.stem}.png")
        n += 1
    return n


if __name__ == "__main__":
    n = render_batch_from_betas_cache()
    print(f"Rendered {n} skeleton images to internal/data/skeleton_renders/")
