"""
Convierte joints 3D (24, 3) en imágenes 128×128 PNG de esqueleto.

Usa Pillow (PIL) — opencv NO está instalado.
El skeleton se dibuja con:
  - Líneas blancas para los huesos
  - Círculos blancos para los joints
  - Fondo negro
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from src.config.paths import Paths

# Conexiones de huesos SMPL (índices de joints)
SMPL_BONES = [
    (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8),       # piernas
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),             # columna + cabeza
    (9, 13), (9, 14), (13, 16), (14, 17),                   # hombros
    (16, 18), (17, 19), (18, 20), (19, 21),                 # brazos
]

IMAGE_SIZE = 128
JOINT_RADIUS = 3
BONE_WIDTH = 2


def joints3d_to_image(joints: np.ndarray, size: int = IMAGE_SIZE) -> Image.Image:
    """
    Proyecta joints 3D al plano XY y dibuja el esqueleto.

    Args:
        joints: (24, 3) — joints normalizados (pelvis en origen)
        size: lado de la imagen cuadrada

    Returns:
        img: PIL Image RGB 128×128
    """
    # Proyectar al plano XY (descartar Z)
    xy = joints[:, :2].copy()

    # Normalizar a espacio de imagen [margin, size-margin]
    margin = 10
    xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)
    rng = xy_max - xy_min
    rng[rng < 1e-6] = 1.0  # evitar división por cero

    xy = (xy - xy_min) / rng  # [0, 1]
    xy = xy * (size - 2 * margin) + margin  # [margin, size-margin]

    # Voltear eje Y (PIL tiene Y creciente hacia abajo)
    xy[:, 1] = size - xy[:, 1]

    img = Image.new('RGB', (size, size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dibujar huesos
    for (i, j) in SMPL_BONES:
        if i < len(joints) and j < len(joints):
            x0, y0 = int(xy[i, 0]), int(xy[i, 1])
            x1, y1 = int(xy[j, 0]), int(xy[j, 1])
            draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=BONE_WIDTH)

    # Dibujar joints
    for k in range(len(joints)):
        x, y = int(xy[k, 0]), int(xy[k, 1])
        r = JOINT_RADIUS
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(200, 200, 200))

    return img


def render_joints_batch(joints_flat: np.ndarray, out_dir: str = None,
                        prefix: str = 'skeleton') -> list:
    """
    Renderiza un batch de joints aplanados (N, 72) y guarda las imágenes.

    Args:
        joints_flat: (N, 72)
        out_dir: directorio de salida
        prefix: prefijo para los nombres de archivo

    Returns:
        list de rutas guardadas
    """
    if out_dir is None:
        out_dir = str(Paths.SKELETON_IMAGES_DIR)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, j_flat in enumerate(joints_flat):
        joints = j_flat.reshape(24, 3)
        img = joints3d_to_image(joints)
        path = out_dir / f'{prefix}_{i:06d}.png'
        img.save(str(path))
        paths.append(str(path))

    return paths


if __name__ == "__main__":
    # Prueba rápida: genera 5 esqueletos aleatorios
    np.random.seed(42)
    dummy = np.random.randn(5, 72).astype(np.float32)
    saved = render_joints_batch(dummy, prefix='test')
    print(f"Guardados {len(saved)} esqueletos en {Paths.SKELETON_IMAGES_DIR}")
