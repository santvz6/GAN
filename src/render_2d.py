from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _project_vertices(verts: np.ndarray, view: str):
    if view == "front":
        return verts[:, 0], verts[:, 1], verts[:, 2]
    if view == "back":
        return -verts[:, 0], verts[:, 1], -verts[:, 2]
    if view == "side":
        return verts[:, 2], verts[:, 1], -verts[:, 0]
    raise ValueError(f"Unsupported view '{view}'. Use front, side or back.")


def render_mesh_to_png(
    verts: np.ndarray,
    faces: np.ndarray,
    output_path: str | Path,
    image_size: int = 512,
    view: str = "front",
    background=(255, 255, 255),
    draw_edges: bool = False,
) -> Path:
    """Render an SMPL mesh to a simple orthographic 2D PNG.

    This is intentionally lightweight: it avoids an OpenGL context and works in
    headless environments by projecting the mesh and painting depth-sorted faces.
    """
    if image_size < 64:
        raise ValueError("image_size must be at least 64 pixels.")

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected verts with shape (N, 3), got {verts.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces with shape (F, 3), got {faces.shape}.")

    x, y, depth = _project_vertices(verts, view)

    supersample = 2
    canvas = image_size * supersample
    margin = canvas * 0.08

    x_span = max(float(np.ptp(x)), 1e-6)
    y_span = max(float(np.ptp(y)), 1e-6)
    scale = (canvas - 2 * margin) / max(x_span, y_span)

    px = (x - (x.min() + x.max()) * 0.5) * scale + canvas * 0.5
    py = canvas * 0.5 - (y - (y.min() + y.max()) * 0.5) * scale
    points = np.stack([px, py], axis=1)

    face_depth = depth[faces].mean(axis=1)
    depth_min = float(face_depth.min())
    depth_span = max(float(face_depth.max() - depth_min), 1e-6)
    depth_norm = (face_depth - depth_min) / depth_span

    img = Image.new("RGB", (canvas, canvas), background)
    draw = ImageDraw.Draw(img)

    # Paint far faces first so closer faces overwrite them.
    for face_idx in np.argsort(face_depth):
        polygon = [tuple(p) for p in points[faces[face_idx]]]
        tone = int(105 + 95 * depth_norm[face_idx])
        fill = (tone, tone, tone)
        outline = (82, 82, 82) if draw_edges else fill
        draw.polygon(polygon, fill=fill, outline=outline)

    if supersample > 1:
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path
