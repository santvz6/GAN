# CLAUDE.md — Plan de Acción: GAN para Síntesis de Cuerpo Humano

## Regla de Workflow — Git

**Cada cambio realizado debe subirse a GitHub en la rama `dev_taron`.**

```bash
# Flujo obligatorio tras cada cambio
git add <archivos modificados>
git commit -m "descripción del cambio"
git push origin dev_taron
```

- Nunca hacer push a `main` directamente.
- Si la rama no existe localmente: `git checkout -b dev_taron`
- Si ya existe: `git checkout dev_taron` antes de empezar a trabajar.

---

## Regla de Notebook — Documentación del Proyecto

**El proyecto debe tener un notebook principal `notebooks/proyecto_gan.ipynb` que documente TODO lo realizado.**

### Requisitos del notebook:
- Escrito para ser entendido por alguien **sin conocimientos de programación**.
- Cada sección debe tener una explicación en texto antes del código.
- El código debe ir acompañado de comentarios que expliquen qué hace cada parte.
- Incluir visualizaciones (gráficas, imágenes generadas, métricas) con su interpretación.
- Lenguaje claro, sin jerga técnica innecesaria. Cuando se use un término técnico, explicarlo.

### Estructura del notebook:
1. **Introducción** — qué es una GAN y por qué se usa para cuerpos humanos
2. **Dataset** — qué es AMASS, cómo se descargó, qué contiene
3. **Preprocesado** — cómo se extraen los joints y por qué se normalizan
4. **GAN Tabular** — arquitectura, entrenamiento y resultados con visualizaciones
5. **GAN Imagen** — cómo se generan las imágenes de esqueleto, entrenamiento y resultados
6. **GAN 3D** — qué es una nube de puntos, entrenamiento y resultados
7. **Métricas** — qué mide cada métrica (MMD, FID, Chamfer) explicado en lenguaje simple
8. **Conclusiones** — qué funcionó, qué no, y próximos pasos

### Regla:
Cada vez que se implemente o modifique algo en el proyecto, el notebook debe actualizarse para reflejar el cambio con su explicación correspondiente.

---

## Descripción del Proyecto

Entrenamiento de GANs para generar representaciones plausibles del cuerpo humano en tres modalidades:
1. **Tabular** — coordenadas 3D de joints (24 joints × 3 = 72 dimensiones)
2. **Imagen** — renders de esqueleto 128×128 RGB
3. **3D** — malla SMPL completa (6890 vértices)

Módulo SMPL del profesor disponible en `smpl_module_project/`.

---

## Datasets

| Dataset | Rol | Formato | Descarga |
|---------|-----|---------|----------|
| **AMASS** | Tabular (joints) — fuente principal | `.npz` SMPL params | Ya descargado → `internal/data/amass/` |
| **TN15** | Imagen (RGB de actores en movimiento) | Vídeo RGB + meshes 3D | Registrarse en tnt.uni-hannover.de → TNT15_V1_0.zip → `internal/data/tn15/` |
| **NOMO3D** | 3D (scans de 400 personas) | OBJ ~57k verts | zenodo.org/records/3735905 → `internal/data/nomo3d/` |

**Regla:** Solo AMASS se usa para datos tabulares (tiene SMPL params directamente). TN15 y NOMO3D no requieren SMPL fitting — se usan como datos de imagen y 3D respectivamente.

---

## Pipeline General

```
AMASS ──► preprocess_joints.py ──► joints.npz (N×72)
                                        │
                                   Tabular GAN
                                        │
                              generar N cuerpos z~N(0,1)
                                        │
                          discriminator_filter.py (score_D > 0.8)
                                        │
                             "cuerpos perfectos" aceptados
                                  ┌─────┴─────┐
                                  ▼           ▼
                          smpl_mesh_generator  render_skeleton.py
                          6890 vértices        imagen 128×128
                                  │                  │
                      ┌───────────┘                  └───────────┐
                      ▼                                          ▼
           NOMO3D OBJ scans                             TN15 RGB images
           + meshes generadas                           + renders generados
                      │                                          │
                 3D Mesh GAN                               Image GAN
              (6890 verts)                                (128×128)
```

**Criterio de aceptación de cuerpos generados:** `score_D > 0.8` (discriminador entrenado puntúa entre 0 y 1 — solo se aceptan los que se parecen más a un cuerpo real).

**Si faltan datos para imagen o 3D:** usar cuerpos generados y filtrados del Tabular GAN para crear los datasets sintéticos de imagen/3D.

---

## Estructura del Proyecto

```
GAN/
├── CLAUDE.md                         (este archivo)
├── README.md                         (especificación del proyecto)
├── TODO.md                           (checklist de tareas)
├── main.py                           (punto de entrada)
├── windows_environment.yaml          (entorno conda Windows)
├── ubuntu_environment.yaml           (entorno conda Ubuntu)
├── smpl_module_project/              (MODULO DEL PROFESOR - no modificar)
│   ├── measure.py
│   ├── measurement_definitions.py
│   ├── landmark_definitions.py
│   ├── joint_definitions.py
│   ├── utils.py
│   ├── visualize.py
│   ├── evaluate.py
│   ├── demo.py
│   └── data/smpl/
│       ├── SMPL_FEMALE.pkl
│       ├── SMPL_MALE.pkl
│       ├── SMPL_NEUTRAL.pkl
│       ├── smpl_body_parts_2_faces.json
│       └── smpl_vert_segmentation.json
├── src/
│   ├── config/
│   │   ├── paths.py                  (EXISTENTE — paths del proyecto)
│   │   └── utils.py                  (EXISTENTE — utils SMPL)
│   ├── data/
│   │   ├── download_datasets.py      (CREAR — descarga TN15, NOMO3D)
│   │   ├── amass_loader.py           (CREAR — carga .npz de AMASS)
│   │   ├── preprocess_joints.py      (CREAR — extrae joints, normaliza)
│   │   ├── tn15_loader.py            (CREAR — frames RGB de TN15, resize 128×128)
│   │   ├── nomo3d_loader.py          (CREAR — carga OBJ de NOMO3D con trimesh)
│   │   ├── render_skeleton.py        (CREAR — joints → imagen 128×128 con Pillow)
│   │   ├── smpl_mesh_generator.py    (CREAR — joints/betas → malla 6890 verts via smplx)
│   │   └── discriminator_filter.py   (CREAR — genera N cuerpos, filtra score_D > 0.8)
│   ├── models/
│   │   ├── tabular_gan.py            (CREAR — MLP Generator + Discriminator, WGAN-GP)
│   │   ├── image_gan.py              (CREAR — DCGAN)
│   │   └── mesh_gan.py               (CREAR — MLP Gen + PointNet Disc, 6890 verts)
│   ├── training/
│   │   ├── train_tabular.py          (CREAR — bucle WGAN-GP tabular)
│   │   ├── train_image.py            (CREAR — bucle imagen, TN15 + renders)
│   │   └── train_mesh.py             (CREAR — bucle 3D, NOMO3D + meshes generadas)
│   └── evaluation/
│       ├── eval_tabular.py           (CREAR — MMD + bone length error)
│       ├── eval_image.py             (CREAR — FID con pytorch-fid)
│       └── eval_3d.py                (CREAR — Chamfer Distance + F-score)
└── internal/                         (generado por Paths.init_project())
    ├── data/
    │   ├── amass/                    (ficheros .npz de AMASS)
    │   ├── tn15/                     (imágenes RGB de TN15)
    │   ├── nomo3d/                   (scans OBJ de NOMO3D)
    │   ├── joints.npz                (joints normalizados de AMASS)
    │   ├── skeleton_images/          (renders 128×128 generados)
    │   └── meshes/                   (mallas 6890 verts generadas)
    ├── experiments/
    ├── logs/
    └── temp/
```

---

## Reutilización del Módulo del Profesor

**NO modificar** nada dentro de `smpl_module_project/`. Solo importar desde él.

| Módulo | Qué usar | Para qué |
|--------|----------|----------|
| `joint_definitions.py` | `SMPL_IND2JOINT`, `SMPL_JOINT2IND`, `get_joint_regressor()` | Extraer 24 joints de AMASS |
| `landmark_definitions.py` | `SMPL_LANDMARK_INDICES` | Landmarks de normalización (pelvis, head top) |
| `visualize.py` | `Visualizer.create_joint_plot()` | Visualización 3D de resultados |
| `utils.py` | `convex_hull_from_3D_points()` | Circumferencias si se necesitan |
| `evaluate.py` | `evaluate_mae()` | MAE entre medidas reales y generadas |
| `data/smpl/*.pkl` | `SMPL_NEUTRAL.pkl` | Modelo SMPL para preprocessing |

```python
# Ejemplo de importación
import sys
sys.path.insert(0, 'smpl_module_project')
from joint_definitions import SMPL_IND2JOINT, get_joint_regressor
from landmark_definitions import SMPL_LANDMARK_INDICES
from visualize import Visualizer
```

---

## Fases de Implementación

### Fase 1 — Descarga y Preprocesado (Persona 1)

**Archivos:** `src/data/download_datasets.py`, `src/data/amass_loader.py`, `src/data/preprocess_joints.py`

**Descarga de datasets:**
- **NOMO3D** (libre): `wget https://zenodo.org/records/3735905/files/NOMO3D.zip` → `internal/data/nomo3d/`
- **TN15** (requiere registro): tnt.uni-hannover.de → descargar TNT15_V1_0.zip → `internal/data/tn15/`
- **AMASS**: ya descargado → colocar en `internal/data/amass/`

**Preprocesado AMASS → joints.npz:**
1. Los ficheros AMASS contienen: `poses` (N, 156), `betas` (10,), `gender`, `mocap_framerate`
2. Extraer `poses[:, :72]` (body pose, 24 joints × 3 axis-angle) + `betas`
3. Aplicar SMPL forward pass via `smplx` para obtener joints 3D (24, 3)
4. **Normalización:**
   - Pelvis (joint 0) al origen: `joints -= joints[0]`
   - Escala uniforme: dividir por altura (pelvis→head)
   - Aplanar a vector de 72 dimensiones
5. Guardar en `internal/data/joints.npz`: array `(N, 72)`

---

### Fase 2 — Tabular GAN + Filtrado (Persona 1)

**Archivos:** `src/models/tabular_gan.py`, `src/training/train_tabular.py`, `src/data/discriminator_filter.py`

**Arquitectura:**
```
Generator:  z(128) → Linear(128,256) → BN → LeakyReLU
                   → Linear(256,512) → BN → LeakyReLU
                   → Linear(512,72)  → Tanh

Discriminator: x(72) → Linear(72,512)  → LeakyReLU → Dropout(0.3)
                      → Linear(512,256) → LeakyReLU → Dropout(0.3)
                      → Linear(256,1)   (sin activación — WGAN)
```

**Loss: WGAN-GP** — `L_D = E[D(fake)] - E[D(real)] + 10 * GP`, `L_G = -E[D(fake)]`

**Filtrado (discriminator_filter.py):**
```python
# Generar hasta tener N cuerpos aceptados
accepted = []
while len(accepted) < N_target:
    z = torch.randn(batch, 128)
    fake = G(z)              # (batch, 72)
    score = D(fake).sigmoid()  # normalizar a [0,1]
    accepted.extend(fake[score > 0.8])
# Guardar en internal/data/generated_joints.npz
```

---

### Fase 3 — Datos Imagen: TN15 + Renders (Persona 2)

**Archivos:** `src/data/tn15_loader.py`, `src/data/render_skeleton.py`, `src/models/image_gan.py`, `src/training/train_image.py`

**TN15 loader:** Extraer frames RGB de los vídeos, resize a 128×128, guardar como tensores.

**Render skeleton:** joints filtrados → imagen 128×128 con Pillow (opencv NO instalado)
- Proyectar joints 3D → 2D (plano XY)
- Dibujar huesos (líneas) y joints (círculos) sobre fondo negro

**Conexiones de huesos SMPL:**
```python
SMPL_BONES = [
    (0,1),(0,2),(1,4),(2,5),(4,7),(5,8),   # piernas
    (0,3),(3,6),(6,9),(9,12),(12,15),       # columna + cabeza
    (9,13),(9,14),(13,16),(14,17),(16,18),(17,19)  # brazos
]
```

**DCGAN:**
```
Generator: z(128) → FC → reshape(512,4,4)
  → 5× ConvTranspose2d(stride=2, BN, ReLU) → [128×128] → Tanh

Discriminator: img(3,128,128)
  → 4× Conv2d(stride=2, BN, LeakyReLU) → Flatten → Linear → 1
```

**Dataset imagen = TN15 frames + renders de joints filtrados** (si TN15 insuficiente)

---

### Fase 4 — Datos 3D: NOMO3D + Meshes SMPL (Persona 3)

**Archivos:** `src/data/nomo3d_loader.py`, `src/data/smpl_mesh_generator.py`, `src/models/mesh_gan.py`, `src/training/train_mesh.py`

**NOMO3D loader:** Cargar OBJ con `trimesh`, remuestrear superficie a 6890 puntos para aproximar topología SMPL.

**SMPL mesh generator:** Cuerpos filtrados (joints+betas) → SMPL forward pass via `smplx` → malla 6890 verts.

**Mesh GAN (PointNet-style):**
```
Generator: z(128) → MLP → (6890×3)
  Linear(128,1024) → ReLU → Linear(1024,4096) → ReLU
  → Linear(4096, 6890*3) → reshape(6890,3)

Discriminator (PointNet):
  Per-point: (6890,3) → (6890,64) → (6890,128) → (6890,256)
  → MaxPool global → (256,) → Linear(256,128) → Linear(128,1)
```

**Dataset 3D = NOMO3D scans + meshes SMPL de joints filtrados** (si NOMO3D insuficiente)

---

### Fase 5 — Evaluación (Personas 1, 2, 3)

- **`eval_tabular.py`**: MMD (kernel RBF, scipy) + bone length error
- **`eval_image.py`**: FID con `pytorch-fid` (instalar: `pip install pytorch-fid`)
- **`eval_3d.py`**: Chamfer Distance (`chamfer-distance` ya instalado) + F-score (τ=0.01)

---

### Fase 6 — Integración y Reporte (Persona 4)

- `main.py`: `Paths.init_project()` + orquestar todo el pipeline
- `laTex/main.tex`: Abstract, Intro, Metodología, Resultados (tablas métricas, grids de muestras), Conclusiones
- Notebook `notebooks/proyecto_gan.ipynb`: documentación completa (ver sección Regla de Notebook)

---

## Especificaciones Técnicas

| Parámetro | Valor |
|-----------|-------|
| Joints (tabular) | 24 × 3 = **72 dim** |
| 3D mesh | **6890 vértices** (topología SMPL) |
| Imagen | **128×128 RGB** |
| Latent space `z` | dim=128, N(0,1) |
| Loss | **WGAN-GP**, λ=10 |
| Ratio D/G | 5:1 (5 pasos D por 1 G) |
| Batch size | 64 |
| Learning rate | 1e-4 (Adam, β1=0.0, β2=0.9) |
| Epochs | 200+ |
| Filtro discriminador | score_D > **0.8** |
| Semilla | `torch.manual_seed(42)`, `np.random.seed(42)` |
| Checkpoints | cada 50 épocas en `internal/experiments/` |

---

## Dependencias Disponibles (entorno conda `smpl-env`)

Ya instaladas:
- `torch==1.13.1` + CUDA 11.8
- `smplx==0.1.28` — SMPL forward pass
- `trimesh==3.9.35` — cargar OBJ (NOMO3D), operaciones 3D
- `open3d==0.17.0` — visualización point clouds
- `chamfer-distance==0.1` — métrica 3D
- `geomloss==0.2.6` — EMD/Sinkhorn
- `plotly==5.18.0` — visualización 3D interactiva
- `pillow==9.5.0` — render skeleton (opencv NO instalado)
- `scipy`, `numpy`, `matplotlib`, `tqdm`, `pandas`

**Instalar antes de implementar:**
```bash
pip install pytorch-fid   # para FID en eval_image.py
```

**Nota:** `opencv-python` NO está en el entorno. Usar `Pillow` (PIL) para dibujar imágenes de skeleton.

---

## Comandos de Verificación

```bash
# 1. Preprocessing AMASS → joints
python src/data/preprocess_joints.py
# Esperado: internal/data/joints.npz shape=(N, 72)

# 2. Entrenar Tabular GAN
python src/training/train_tabular.py
# Esperado: internal/experiments/tabular/checkpoint_ep200.pt

# 3. Filtrar cuerpos generados (score_D > 0.8)
python src/data/discriminator_filter.py --n 1000
# Esperado: internal/data/generated_joints.npz

# 4. Generar meshes SMPL desde filtrados
python src/data/smpl_mesh_generator.py
# Esperado: internal/data/meshes/*.npz (6890 verts cada una)

# 5. Render imágenes skeleton
python src/data/render_skeleton.py
# Esperado: internal/data/skeleton_images/*.png

# 6. Training Image GAN
python src/training/train_image.py
# Esperado: internal/experiments/image/checkpoint_ep200.pt

# 7. Training Mesh GAN
python src/training/train_mesh.py
# Esperado: internal/experiments/mesh/checkpoint_ep200.pt

# 8. Evaluación
python src/evaluation/eval_tabular.py  # → MMD, bone error
python src/evaluation/eval_image.py    # → FID
python src/evaluation/eval_3d.py       # → Chamfer Distance, F-score
```

---

## División de Tareas

| Persona | Responsabilidad | Entregables |
|---------|----------------|-------------|
| **1** | Data Lead + Tabular GAN | `download_datasets.py`, `amass_loader.py`, `preprocess_joints.py`, `tabular_gan.py`, `train_tabular.py`, `discriminator_filter.py`, `eval_tabular.py` |
| **2** | Image GAN | `tn15_loader.py`, `render_skeleton.py`, `image_gan.py`, `train_image.py`, `eval_image.py` |
| **3** | 3D Mesh GAN | `nomo3d_loader.py`, `smpl_mesh_generator.py`, `mesh_gan.py`, `train_mesh.py`, `eval_3d.py` |
| **4** | Integración + Reporte | `main.py` completo, `laTex/main.tex`, `notebooks/proyecto_gan.ipynb`, consolidación de resultados |

---

## Notas Importantes

- Usar `Paths` de `src/config/paths.py` para todas las rutas (no hardcodear paths)
- Los modelos SMPL están en `smpl_module_project/data/smpl/` — no mover
- Si AMASS no está accesible, generar datos sintéticos con distribución gaussiana como fallback
- Priorizar estabilidad del entrenamiento antes que complejidad arquitectónica
- Documentar versiones de librerías y resultados en `internal/experiments/`
